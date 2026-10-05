# SpeCache 公開 beta：RTX 5060 Ti／Llama-3.2-3B 本機可行性

日期：2026-10-05。研究層級：5–6，外部 baseline 的相容性與最小驗證。

## 問題與結論

使用者要求把找到的 SpeCache source 跑在現有環境，確認是否需要大幅修改。
假設為 Llama adapter 可沿用，但舊 vLLM／Transformers／CUDA 依賴需要適配。

**已確認**：這份公開 beta 經相容性修改，可在 RTX 5060 Ti 16GB、
Llama-3.2-3B-Instruct、BF16、Batch 1 上執行 CPU-backed SpeCache。
1-bit 與 2-bit 模式各完成一個合成 32768-token prompt＋64 個固定 greedy
output tokens；沒有 OOM，所有檢查的 logits finite。

**重要限制**：這不是 SpeCache 論文效率重現。Repository 自己將
`SpeCacheKVCache` 標示為 accuracy／understanding only、not for efficiency。
其模擬量化與同步搬移會影響實測 TPOT／VRAM，不能直接把這份 beta
的數字當成論文方法的系統競爭力。本輪沒有 RULER score、formal TPOT
或完整 130 題測試。

## Source 與環境

- 公開 source：[intelli-jsong/SpeCache](https://github.com/intelli-jsong/SpeCache)。
  尚未確認它是論文作者官方實作。
- Upstream commit：`22503219ca1f3432189aff65af9c582ca9174bd3`。
- 本機 checkout：workspace 的 `external/specache-20261005/`。
- 模型：`meta-llama/Llama-3.2-3B-Instruct`，使用本機既有 HF checkpoint。
- RTX 5060 Ti，compute capability 12.0；CUDA toolkit 12.8；driver 580.173.02。
- 獨立 `.venv --system-site-packages`，繼承 PyTorch `2.7.0+cu128`、
  FlashAttention `2.8.3.post1`；overlay Transformers `4.49.0`、
  FlashInfer `0.2.4`、Triton `3.4.0`、tokenizers `0.21.4`。
- 未改動 HeadInfer production source 或其套件環境。Triton 3.4 overlay
  與 Torch 的 3.3 dependency pin 不一致，為本機 sm_120 相容配置，
  不宣稱整份 upstream requirements 已滿足。

## 修改範圍

Upstream tracked diff 僅 4 個檔案，33 insertions／18 deletions：

1. `specache/__init__.py`：模型 family import 改為按需載入。
2. `specache/llama.py`：以 FlashInfer 替代 vLLM RoPE／SiLU 呼叫；Q/K 先
   contiguous，保留 inplace 的結果。RoPE cache 使用 HF 模型本身的
   rotary module，包含 Llama-3.2 scaling；FlashInfer 要求 FP32 cache。
3. `specache/tensor_op.py`：延後未使用的 MInference／ShadowKV import。
4. `specache/kv_cache.py`：延後 ShadowKV 專用 import；SpeCache class
   本體沒有更改。

`specache/base.py` 的双 token／speculative selection 路徑，以及
`utils_specache_quant.py` 的量化／residual 更新邏輯與 upstream byte-identical。
沒有新增 previous-token cache、換 selector、改 Top-k 或重做傳輸。
未編譯不會在此次 SpeCache 路徑執行的 ShadowKV extension；實際使用的
FlashInfer norm、RoPE、SiLU JIT modules 已在 sm_120 編譯及執行。

新增 1-bit／2-bit CPU config 僅將 upstream `cpu_mode:false` 改為 `true`。
其他設定保持：K per-channel、V per-token、group=64、residual=64、
prefetch Top-k=64；1-bit `do_specache_quant=true`，2-bit 為 false。

## 驗證與結果

所有 prompt 為相同重複英文合成文本，固定生成、不依 EOS 提前停止。
這是 feasibility probe，不是 benchmark 品質測試。

| Gate | 結果 | 限制 |
|---|---|---|
| 256 context＋32 output，移植 dense vs 原生 HF FA2 | IDs 32/32 一致；最大 logits abs 差 1.375 | 非逐元素 exact／allclose；僅模型 adapter 短 context 對照 |
| 256＋96，2-bit CPU vs GPU backing | output IDs、speculative IDs、所有 captured logits 完全相同 | 跨過 residual=64 更新；僅已測 prompt |
| 32768＋64，2-bit CPU | 完成、finite、無 OOM | 單次合成 feasibility |
| 32768＋64，1-bit CPU | 完成、finite、無 OOM | 單次合成 feasibility |

CPU/GPU 256＋96 對照 prefetch 各 2688 calls（28 layers × 96），CPU
模式實際 CPU gather 5376 calls；32K 各 prefetch 1792、CPU gather 3584。
每次取回 64 KV pairs／KV head，32K 的 selected KV payload 累計
469,762,048 bytes（448 MiB），不含 index、完整 Prefill D2H、新 token
D2H、allocation 或其他拷貝；不能稱為所有 PCIe traffic。

### 32K memory 與診斷 wall（各單次）

| 模式 | Cache GPU unique storage，排除模型權重 | Full CPU KV capacity | 全流程 peak allocated，包含權重／temporary | Instrumented decode wall |
|---|---:|---:|---:|---:|
| 2-bit beta CPU | 1918.026 MiB | 3.507 GiB | 11.429 GiB | 291.070 ms／iteration |
| 1-bit beta CPU | 1918.026 MiB | 3.507 GiB | 11.429 GiB | 289.434 ms／iteration |

Cache storage 以 cache object 內 GPU backing storage 去重，包含量化
數值、scale／zero point、residual、retrieve buffer 與 metadata；
不包含模型權重、模型 RoPE cache 或短期 Attention temporary。
CPU KV 是 max_length=32836 的配置容量，非僅 prompt 的 logical KV。
KV offset 結束為 32831（32768＋63）；最後輸出的 token 尚未寫入 KV。

**上表不是 formal TPOT**：使用 synchronized CPU wall，63 個双 token
decode iterations，不含模型載入、Prefill、pre-decode，但包含逐步 finite
檢查、logits D2H、audit index min/max `.item()` 與 CPU gather 診斷。
每次 iteration 只輸出一個正式 token；另一個是 speculative token。
不能拿 289–291 ms 和現有方法／RetroInfer 的 uninstrumented TPOT 排名。

## 為什麼這份 beta 不適合直接當效能 baseline

以下限制來自 upstream，這輪沒有為了速度重寫它：

- 1-bit／2-bit code 用 `simulate=True`；quantized values 實際 dtype 是
  INT8，一個值占一 byte，沒有 bit packing。因此兩個模式配置相同大小。
- 每 layer decode 先 dequant 完整 KV，再把 fetched high-precision KV
  scatter 回完整張量；Attention 仍涵蓋整段 mixed-precision history。
- `copy_stream` 有建立，但 `base.py` 的 stream prefetch 區段被註解；
  實際直接呼叫 `prefetch_kv_beta`。
- CPU full KV 未 pinned，CPU index／gather／`.to(device)` 為 beta 同步路徑。
- 預設 config 是 `cpu_mode=false`，若不覆寫會把完整 KV 留在 GPU。

因此現在可稱為 **「SpeCache 公開 beta 的 Llama-3.2-3B 相容移植／
accuracy simulation」**。不能稱為「SpeCache 官方實作重現」或聲稱
已重現論文的壓縮率與非同步效率。

## 失敗與處理紀錄

- 原樣 import：因非必要 model-family eagerly import vLLM 而失敗。
  透過按需 import 與 Llama active ops 的 FlashInfer 相容適配修正。
- 初次依賴安裝 solver backtracking 嘗試下載其他 Torch，已停止；
  使用 `--no-deps` overlay 本機已驗證版本，未安裝／修改 base Torch。
- 初次 dense probe：FlashInfer 要求 `cos_sin_cache` FP32；轉型後重跑通過。
- 首次 CPU SpeCache probe：runner 的 audit 用 global offset 檢查，
  忽略最後 layer 才增加 offset；改成逐 layer 寫入長度後通過。
  這是 diagnostic assertion 修正，不是 upstream selection bug。
- 乾淨 upstream 檔案重放 compatibility patch，4/4 與本機修改 byte-identical。

## 重跑方式

從 workspace root 執行：

```bash
bash headinfer/headinfer_reproduction/scripts/setup_specache_local.sh
bash headinfer/headinfer_reproduction/scripts/run_specache_local.sh \
  --mode specache --context 32768 --generate 64 \
  --config specache_2bit_cpu_local.yaml \
  --output /tmp/specache_32k_recheck.json
```

setup 不安裝 upstream 的完整舊 requirements；只適用本機既有
`headinfer_repro` 與已存在的 FlashAttention／模型 checkpoint。

## Artifacts 與下一決策

Canonical artifacts：`results/specache_port_20261005/`，包含六個完成 run
的 JSON/PT/log、`full_hf_gate.json`、`cpu_gpu_gate.json`、
`validation_summary.json`、`source_manifest.json`、`source_snapshot/`、
`compatibility.patch` 與失敗 log。

Scripts：`scripts/{setup_specache_local,run_specache_local}.sh`、
`scripts/{port_specache_local_20261005,run_specache_smoke_20261005,analyze_specache_port_20261005}.py`。

下一決策：這份 beta 可供小型品質／方法行為診斷。若要做正式 TPOT／
VRAM 比較，需要另確認可用的效率實作，或明確揭露一份經另外驗證的
系統重現版本；加入 bit packing、fused quantized attention、pinned
async prefetch 是進一步工程工作，不屬於這次已驗證的相容移植。
本輪沒有執行 130 題、沒有修改使用者研究方法、沒有 commit／push。
