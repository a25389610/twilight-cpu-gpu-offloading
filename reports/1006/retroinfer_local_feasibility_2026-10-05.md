# RetroInfer 本機移植與可行性驗證 — 2026-10-05

## 問題、假設與結論

問題：Microsoft RetroInfer 官方 source 能否在 RTX 5060 Ti 16GB 上使用 Llama-3.2-3B-Instruct，執行 CPU KV offloading、群組 selection 與 GPU KV reuse。

原先假設：Llama 共用實作可讀取 3B 架構設定，但 sm_120 的套件、CUDA kernels 與模型輸出尚需實測。

**已確認：本機 BF16 / Batch 1 的 32,768-token input + 64-token generation 可以執行官方 RetroInfer sparse/offload pipeline。** 這是單一合成文本的 feasibility gate，沒有正式品質 benchmark 或 matched TPOT。不得據此宣稱比現有 Quest/Twilight 系統快、品質相當或支援更長 context。

## Source 與執行環境

- 官方 repo：<https://github.com/microsoft/RetrievalAttention>，SHA `03f912c6e917c380d9d90c5ec85bb0f161ba53ef`。
- 官方 README 使用的 weighted attention：<https://github.com/Starmys/flash-attention/tree/weighted>，SHA `56d96228ada74d6df806b0083bf018d0d57f57e9`。
- RetroInfer CUTLASS：<https://github.com/NVIDIA/cutlass>，SHA `0b55a2f691d69981583568fd9eb69687b1f0de8a`；weighted attention 自己的 CUTLASS submodule 使用該 repo pin。
- GPU NVIDIA GeForce RTX 5060 Ti，16311 MiB，compute capability 12.0；driver 580.173.02。
- CUDA toolkit / PyTorch CUDA 12.8；Python 3.10.20；torch 2.7.0+cu128。
- 獨立 `.venv --system-site-packages` 繼承既有 `headinfer_repro`，overlay transformers 4.49.0、Triton 3.4.0、FlashInfer 0.2.4、pybind11 2.12.0、ninja 1.13.2。
- Full attention 使用既有 flash-attn 2.8.3.post1；摘要與合併 attention 使用官方 weighted_flash_decoding 0.1。
- 沒有安裝官方 torch 2.5.1 / vLLM 0.6.5 整套 benchmark dependencies；只安裝此推論路徑必要套件。

## 修改與失敗證據

1. 建立獨立環境；用 CUDA 12.8 將 RetroInfer `WaveBuffer`、`Copy`、`gemm_softmax` 與 weighted attention 編譯為 sm_120。**沒有修改這些官方 C++/CUDA kernels，也沒有修改 selection 或 attention 演算法。**
2. 新增 `config/Llama-3.2-3B-Instruct.json`，內容沿用官方 3.1-8B config。使用 Python API 載入 3B，跳過官方 CLI 的模型 choices 限制；28 layers、24 query heads、8 KV heads、head_dim 128 由模型讀取。
3. Runner 使用 BF16、Batch 1、Full prefill、cache_ratio=.05、retrieval_budget=.018、estimation_budget=.232、CUDA graphs 關閉，CPU thread pool 設 4 threads。
4. 初次 2K 分群在 Triton 3.3 出現 `computeCapability not supported` / `PassManager::run failed`。獨立環境改成 Triton 3.4 後，同一模型與分群路徑通過；錯誤 log 保留。
5. 初次獨立 weighted probe 提供錯誤 weight shape，依官方 API 修正為 `(8,1,1,128)`；reference 使用 `exp(score) / sum(size * exp(score))` 乘彙總 Value，沒有誤把 cluster size 再乘進 numerator。
6. 最終 `pip check` 仍回報 torch pin 要求 Triton 3.3.0，實際 overlay 是 3.4.0。這是明確的相依版本例外，不能宣稱套件檢查全數通過；目前列出的 runtime gates 通過。

## Correctness gates

所有 GPU gates 均執行 `torch.cuda.synchronize()`，成功條件由 assertions 驗證。

| Gate | 條件 | 觀察 |
|---|---|---|
| Weighted summary attention | FP16 / BF16，8 groups，每組 3 query heads，head_dim 128 | reference allclose；max abs error 0.000017502 / 0.000149533 |
| Summary + exact attention 合併 | 同上，以 logsumexp normalization reference 比較 | allclose |
| 官方 centroid score kernel | FP16 / BF16，8 groups × 3 queries × 2048 centroids | max softmax abs error 0.000012169 / 0.000091407 |
| 混合來源 KV assembly | GPU steady + 非連續 pinned CPU misses + 非連續 GPU hits | 兩種 dtype 逐元素 exact，valid lengths exact |
| Full vs HF reference | 256 input + 32 greedy tokens | 32/32 IDs 一致；最大 logit abs difference 2.0，**未宣稱 logits allclose** |
| Short RetroInfer vs Full | 256 input + 32 output；短輸入退回 dense | 32/32 IDs 一致；不是 sparse/offload gate |
| 未裁剪 RetroInfer vs Full | 2048 input + 32 output，retrieval=1、estimation=0 | 32/32 IDs 一致；max logits abs difference 0.1875，非 bit-exact |

Reference prompt 為重複的英文 library 句子。生成匹配僅證明此 input 的功能 gate，沒有跨任務或多文本品質結論。

## Sparse/offload feasibility

| Input / output | 分群 | 結果 | PyTorch peak allocated |
|---|---|---|---|
| 2048 / 32 | 128 centroids、nprobe 2、group_size 3 | 生成成功，所有 logits finite | 7,432,254,976 bytes |
| 32768 / 64 | 2048 centroids、nprobe 37、204 cache pages、296 execution-buffer pages | 生成成功，所有 logits finite | 10,241,479,680 bytes，約 9.54 GiB |

32K placement audit：

- `list_keys` / `list_values` 實際位於 CPU pinned memory；每 layer shape `(1,8,32700,128)`，兩者合計 3,750,297,600 bytes，約 3.49 GiB。
- `cache_keys` / `cache_values` 實際位於 GPU；每 layer shape `(1,8,204,8,128)`。cache 對應 1632 vectors/head，約 32K 的 4.98%；這不等於全部 GPU memory 比例。
- GPU 同時保存 steady KV、centroids、summed Values、execution buffers 等。
- 最後一次 diagnostic assembly audit：1764 calls（28 layers × 63 decode steps），CPU miss 1,995,793 KV-head vectors，GPU hit 5,999,452 KV-head vectors。
- 以上計數為實際 assembly entries 的 vector 數，排除 steady vectors；約 75.04% vector hit ratio。這是重複文本的單次診斷，不能推廣為一般 hit ratio 或相鄰 token overlap。
- 不同重跑的 k-means atomic reduction / cache 計數有小幅差異；保存各次 artifacts，不混合計數。前兩次 32K output IDs 一致。

## 時間口徑與限制

Runner 每步檢查 finite logits 並搬 logits 至 CPU，audit 讀取 CPU metadata；首次可能包含 JIT。`diagnostic_generate_wall_seconds` 覆蓋 cache initialization、prefill、decode 與最終 synchronize。官方打印的 decode time 結束處未顯式 synchronize，因此不能直接用打印值作正式 TPOT。

本次沒有 matched VRAM / quality / formal TPOT 比較、CUDA graph gate、長生成 index-update gate、128K gate 或多 GPU gate。只有本機 3B / Batch 1 / BF16 / greedy 路徑的列出條件得到驗證。

## 重跑方式與 artifacts

Canonical workspace root 為本機研究目錄。從 root 執行：

```bash
headinfer/headinfer_reproduction/scripts/run_retroinfer_local.sh \
  --context 32768 --generate 64 --audit-copy \
  --output /tmp/retroinfer-32k.json
```

- 外部 checkout：`external/retroinfer-20261005/`、`external/retroinfer-weighted-20261005/`。
- Raw artifacts（相對 `headinfer/headinfer_reproduction/`）：`results/retroinfer_port_20261005/`，包括 builds、failed Triton log、kernel gates、HF gate、各次 JSON / PT / log、environment manifest、pip check。
- Scripts：`scripts/setup_retroinfer_local.sh`、`run_retroinfer_local.sh`、`run_retroinfer_smoke.py`、`probe_retroinfer_kernels.py`、`probe_retroinfer_hf_gate.py`。
- 本報告與 canonical weekly report：`reports/weekly/2026-09-30.md`。
- Public mirror：`reports/1007/retroinfer_local_feasibility_2026-10-05.md`；配套 scripts/config/provenance 準備於 `scripts/retroinfer_local/`、`source/retroinfer_local/`，沒有自動 commit / push。
- 排除：raw JSON / PT / logs、虛擬環境、compiled .so、CUTLASS checkout、模型權重與 HF cache。

## 下一個研究決策

可執行性已確認；若將 RetroInfer 納入正式 baseline，下一步先定義相同 prompt / context / output、品質門檻與 GPU memory 比較方式，再補 RULER quality 與無 instrumentation 的 matched TPOT。RetroInfer 的群組摘要近似與 Quest/Twilight 不同，不可只比較速度。
