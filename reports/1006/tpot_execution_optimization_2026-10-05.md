# 2026-10-05：參考 RetroInfer／FreeKV 的 decode execution 優化

後續 cache-control 優化與容量取捨，請見 [續篇](tpot_cache_control_optimization_2026-10-05.md)。以下保留第一輪 v5 的條件與歷史結果。

## 結論與本次研究問題

問題是：在保持目前 Quest → Direct INT4 QK → Twilight Top-p → previous-token resident KV 路徑下，selection 之外的 execution 成本能否下降？本次實際查核兩份 source，完成三項 execution adapter，並做獨立消融、matched formal TPOT、payload correctness 與 memory inventory。

**最終同批 TPOT：48.673853 → 44.786394 ms/token，減少 3.887459 ms（7.9867%）；六組配對全部較快。** 這是約 8% 的已驗證改善，尚未達到 25–30 ms，也未證明已縮小與兩篇 baseline 的公平品質匹配差距。不能從歷史 46.511 ms 直接扣掉這次收益。

## 條件與量測定義

- RTX 5060 Ti 16GB，Llama-3.2-3B-Instruct，BF16，Batch 1；Torch 2.7 / CUDA 12.8，既有 `headinfer_repro` 環境。未更改 driver、套件、CPU governor 或模型。
- 基底為先前 experimental merged-QKV、merged gate/up 與 pointwise 路徑；不是更舊的 grouped-projection runtime。該基底相對舊 runtime 已有數值差異，本次 exact 檢查的對象是此基底。
- Quest B0=8192、Twilight p=.90、Direct INT4 tile16、原 exact Top-p 子圖、GQA union、原 mapped pinned CPU miss／GPU resident hit 路徑保留。
- Frozen cohort：`results/context_p_ruler_39_v1/cohort/32768/timing_requests/` 的 `001_niah_multikey_3_i011`、`002_vt_i002`、`003_qa_1_i011`。實際 prompt 分別為 31938／32639／32363 tokens。
- Formal 每題 32 個 fixed-token-1 forward，排除 D1，使用 D2–D32 synchronized CPU wall；沒有 profiler。三題 × 兩輪 × 兩 arms = 12 個 fresh processes，每 arm 186 個 token samples，表格取六個 run means 平均。配對順序交錯。
- Memory 與 payload gate 是另外的 diagnostic processes，不將其 TPOT 加入 formal；本次不是 130 題 greedy RULER 評分。

## 參考 source 與實際採納方式

### RetroInfer

本機 checkout `external/retroinfer-20261005`，commit `03f912c6e917c380d9d90c5ec85bb0f161ba53ef`，本次未更改。讀取 `library/retroinfer/retroinfer_kernels/src/gather_copy.cu`、`copy_kernel.cuh`、`cache_hub/retroinfer_cache.py` 與 `model_hub/llama.py`。

參考其 fused gather/copy 與 GPU length-aware attention 介面：本方法既有 miss/hit assembly 保留，將後續每 layer 的 16 次 new-token K/V copy 與兩次 resident snapshot copy 合成單一 post-assembly CUDA kernel。這些是 source-level copy 呼叫數，不是推測的 device time。

### FreeKV

本機 checkout `external/freekv-20261005`，官方 commit `2c8a7d25c9f3c7c15ce15b2f84cd03f477bd7469`。既有移植有七個檔案、23 insertions／4 deletions，本次未改。讀取 `source/freekv/kv_cache.py` 的 PagePool／mapping 與 `source/freekv/adapter/modeling.py` 的模型 execution。

參考預配置 backing 與降低算子提交成本的設計：Quest prompt-static min/max 原先每 token stack，現在每 layer 只建立一次 backing，各 entry 改用 views，prefill rebuild 時失效。Score kernel 僅加入 group stride，FP32 reduction 保持原式。

另用 CUDA Graph replay 執行原有 Norm／MLP operations，保留相同算術；沒有替換成另一個 RMS reduction。單一 warmup stream 加 shared graph pool 降低額外 live allocation。這是本方法的 process-local adapter，不能稱為完整移植 RetroInfer 或 FreeKV。

## 最終正式 TPOT（v5）

| 版本 | 六個 run 平均 ms/token | run 平均範圍 ms/token |
| --- | ---: | ---: |
| 同批基底 | 48.673853 | 47.743395–50.709051 |
| finish fusion＋metadata reuse＋Norm/MLP replay | 44.786394 | 42.177213–46.127015 |

| 配對 | 基底 | 最終候選 | 減少 ms/token |
| --- | ---: | ---: | ---: |
| rep1 / 001_niah_multikey_3_i011 | 47.743395 | 46.111716 | 1.631678 |
| rep1 / 002_vt_i002 | 50.709051 | 46.127015 | 4.582036 |
| rep1 / 003_qa_1_i011 | 49.809033 | 45.912602 | 3.896431 |
| rep2 / 001_niah_multikey_3_i011 | 47.747282 | 44.753700 | 2.993582 |
| rep2 / 002_vt_i002 | 48.012411 | 42.177213 | 5.835199 |
| rep2 / 003_qa_1_i011 | 48.021945 | 43.636119 | 4.385826 |

原版本與候選都存在 run 間波動；全部配對保留，未刪掉較小收益的一組。不同批次 baseline 有差異，以下消融只能各自使用同批分母，不能把改善量跨批加總。

### 先前消融與未採納實驗

| 批次／實驗 | 同批結果 | 判斷 |
| --- | --- | --- |
| v1 finish fusion | 47.971381 → 46.797077 ms，2.4479% | 六配對較快，採納 |
| v2 metadata only | 45.561802 → 44.293797 ms，2.7830% | 六配對較快，採納 |
| v2 metadata＋finish | 45.561802 → 43.264357 ms，5.0425% | 六配對較快 |
| v4 再加 Graph replay | 47.700098 → 44.144405 ms，7.4543% | 六配對較快，但額外 memory 過高 |
| vectorized mapped assembly | 90% hit：0.13472 → 0.13414 ms；all-miss：1.302 → 1.306 ms | 無實質收益，最終 flag=0 |
| Direct INT4 tile8／32／64／128 | output 非 exact，max error 約 6.1e-5–1.22e-4 | 未採納，保留 tile16 |
| shared graph pool 單獨消融 | 第一題 independent/shared 都比基底多約 240.008 MiB live | pool 本身未解決容量問題 |
| 單一 warmup stream | 第一題額外 live 240.008 → 20.633 MiB | 採納；支持 stream 配置影響容量，未追蹤每筆 cuBLAS allocation 的根因 |

v1 harness returncode 序列化失敗、v4 memory utility import 路徑失敗均保留原失敗 artifacts；僅修 harness／入口路徑，不改推論算術。v3 Graph pilot 與其 frozen source 也保留，不當正式成果。

## Correctness gate 與範圍

- v5 formal 六配對：prompt、P4／D1／D2／D32／final logits hashes，以及完整 resident count/payload trace 相同。
- v4 三題 payload gate：兩題 32 steps、一題 QA 128 steps。6720 selection records、2240 GQA unions、280 Attention K/V records、43008 CPU new-KV records、5376 resident records 與基底一致；42336 new-KV records 在下一步 consumer 前驗證。
- 最終 v5 單一 warmup stream 版本另跑 QA 128-step payload gate：2688 selection、896 unions、112 Attention K/V、28672 CPU new-KV、3584 resident records 一致；28448 new-KV records 在下一步讀取前核對，最後 step 224 entries 沒有下一步 consumer。D1／D2／D32／D128 與 final logits hashes 相同。
- Selection／Attention payload 只在所列 checkpoints 捕捉；CPU new-KV 每 step 檢查。這證明所測案例相對基底的內容一致，不能擴張為任意 request、所有 steps 的 Attention payload、130 題品質或新硬體皆通過。

## 最終 memory（v5，另六個 fresh processes）

單位 MiB。`cache_unique` 依 backing storage 去重，包含 Quest metadata、INT4、resident 與 cache-owned workspace；Graph runtime 配置不算在 cache_unique，會算入 live 與 nonweight_live。live 是 D32 `torch.cuda.memory_allocated`，不是 reserved、NVML 或 prefill peak。

| 三題平均 | 同批基底 | 最終候選 | 差值 |
| --- | ---: | ---: | ---: |
| cache_unique | 1324.790390 | 1324.790390 | 0 |
| 全部 live allocated | 7472.230957 | 7493.943848 | 21.712891 |
| 扣模型權重後 live | 1344.389893 | 1366.102783 | 21.712891 |

模型去重權重皆 6127.841064 MiB。逐題額外 live 為 20.632813／20.400391／24.105469 MiB；因此最終結論是 KV backing 不增加，但 runtime 仍平均多約 21.7 MiB，不能稱零 VRAM 代價。

## 為何 selection 以外仍可能比論文慢

已驗證能改善的來源是多次 copy dispatch、prompt-static metadata 重建，以及小型 Norm／MLP operations 的 host submission。單獨記錄的 GPU activity 不包含所有 host gaps，且先前 diagnostic 與正式 46.511 ms 並非同 run，不能把其差額直接認定為可移除的 overhead。

另有未解決的 source 介面差異：本方法每 layer 將 `lengths_gpu.cpu().tolist()` 帶回 host，後續 ragged buffer／max_seqlen 依賴 CPU counts；RetroInfer 的 gather 產生 GPU valid_lengths，傳給自己的 weighted flash decoding。RetroInfer 也有 cluster-id 的 CPU handoff 與 WaveBuffer CPU control，不能說它完全沒有同步。

此差異是下一個待驗證假設：若改為 GPU length-aware attention／buffer layout，是否能減少 host control 等待？本次沒有實作，也沒有其獨立 formal ms 歸因或保證收益。需先界定 GPU-valid-length kernel 介面、capacity／VRAM 與內容正確性，再做 matched ablation。不能據此承諾目前方法會降到 25–30 ms；不同 selection／budget／品質仍需公平曲線。

## 可執行入口與限制

在 prepared HeadInfer reproduction root 執行：

```bash
/home/paul/miniconda3/envs/headinfer_repro/bin/python \
  scripts/tpot_execution_opt/run_optimized_case.py \
  --request-dir results/context_p_ruler_39_v1/cohort/32768/timing_requests/001_niah_multikey_3_i011 \
  --output results/my_optimized_case/result.json \
  --preset optimized \
  --decode-steps 32
```

`baseline`／`finish`／`metadata`／`combined`／`optimized` 可做同條件消融；`optimized` 使用最終單一 warmup stream。`--vram-inventory-output` 另輸出 diagnostic memory；`--greedy-probe` 是另一種 decode protocol，不能混入此 fixed-token timing。

Adapter 在 `scripts/tpot_execution_opt/`，使用 process-local hooks 與 deterministic source transformations，依賴所記錄 source、既有 merged-QKV runner、本機模型與編譯環境。Graph 僅驗證單 GPU、sequential Batch 1、相同 capture／replay 順序；不能 concurrent replay 或宣稱其他 batching 通用。沒有修改 production source 或自動切換預設版本。

## Artifacts、重現與公開鏡像

相對路徑皆以 HeadInfer reproduction root 為準：

- 最終 formal：`results/tpot_execution_opt_20261005_v5/formal_analysis.json`、`formal_manifest.json`、`formal/`、`formal_source_snapshot/`。
- 最終 memory：同目錄 `memory_analysis.json`、`memory/`；環境唯讀快照 `environment_readonly.json`。
- 最終 QA gate：`results/tpot_execution_opt_20261005_v5_gate/analysis.json`。
- 前期消融：`results/tpot_execution_opt_20261005_v1/`、`_v2/`、`_v3/`、`_v4/`；三題 gate `results/tpot_execution_opt_20261005_gate/analysis.json`。
- `analyze_trials.py` 從 per-token CSV 重算六個 means，核對 source hashes、protocol 與 payload/count；`analyze_memory.py` 核對 backing 與 live；`analyze_reuse_warm_gate.py` 比對最終 QA payload。
- 本次已準備 public mirror `reports/1007/tpot_execution_optimization_2026-10-05.md` 與 `scripts/tpot_execution_opt/` 的 Python／CUDA source／README。鏡像不是獨立可安裝套件，需 prepared source tree；report 可獨立閱讀。
- 不納入 raw JSON／CSV／log／PT／trace、模型、環境、編譯 binaries 或 cache。不包含 production `source/` 變更。本次未 commit／push。

## 下一個研究決策

最終 preset 可作後續相同 protocol 的 execution 對照；暫不做 130 題。若要繼續縮短 selection 以外的成本，優先驗證 GPU valid-length／ragged attention 介面能否保留原選取語義與受限 VRAM，不能把零碎診斷 scope 或跨批時間差當作可省收益。
