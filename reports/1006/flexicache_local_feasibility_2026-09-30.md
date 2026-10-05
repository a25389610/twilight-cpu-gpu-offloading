# FlexiCache 在 RTX 5060 Ti 的官方 source 可行性試跑

日期：2026-09-30；狀態：source build、256/2048-token inference 通過；32K sparse/offloading 生成已通過（synthetic pilot）。

## 問題與假設

確認 FlexiCache 的官方 vLLM fork 是否能在消費級 RTX 5060 Ti 16GB 使用，
並以既有 Llama-3.2-3B-Instruct 權重試跑。模型切換需新 stability profile；
不得把 8B profile 或任意固定 head 名單冒充 3B 的分類結果。

這輪為 feasibility/correctness probe，沒有 formal TPOT，沒有與目前 Twilight
做效能比較，也不是論文 H100 serving benchmark 的重現。

## Source 與環境

- 官方 source：https://github.com/NazmulTakbir/FlexiCache
- Upstream commit：`ab0f495b10fc6509c1637af35779b29121fb7d3f`。
- 本地 checkout：workspace 下 `external/flexicache-20260930/`。
- GPU：NVIDIA GeForce RTX 5060 Ti，16,311 MiB，compute capability 12.0。
- Driver：580.173.02；nvcc：12.8.93；主機實體 RAM 約 30 GiB。
- Python 3.10.20；PyTorch 2.7.0+cu128；Transformers 4.50.0。
- 使用獨立 `.venv` overlay 既有 PyTorch 環境；沒有修改原 HeadInfer environment。
- 修改 checkout 的配置：CPU KV pool 180 → 1 GiB（2K pilot），32K 試跑改為 4 GiB；avg tokens 20000 → 4096（pilot）→ 32768（長 context）。
- 3B 先登錄空配置收集 `num_unstable_heads=0, rerank=1` 的 diagnostic Top-K；
  已由官方 analysis 產生 56-head synthetic pilot，正式跨任務 profile 尚未驗證。
- 新增 opt-in Top-K logger；僅在 `FLEXICACHE_TOPK_LOG_DIR` 設定且 rerank=1
  時，記錄官方 page scores 選中的 Top-K indices。預設不啟用。

## 安裝失敗與處理

1. 初次 pip editable build 找不到 CMake：補上 `.venv/bin` 到 PATH。
2. PyTorch CMake 找不到 NVTX3，退回不存在的 CUDA::nvToolsExt target：
   將本機已有 NVIDIA NVTX3 headers 連結到該建置預期的目錄；未修改 torch。
3. 官方 pin 的 xgrammar 0.1.16 無法取得：試跑 runtime 改用 0.1.17。
4. JIT extension 以 CUDA target 子目錄作 CUDA_HOME 時，nvcc 找不到 cicc：
   改用 toolkit root，並將 target include 路徑放入 CPATH。

官方提供的 `use_existing_torch.py` 用於沿用 torch 2.7；其移除多份 requirements
中的 torch pin 是建置 helper 的結果。上述為可追蹤的本地移植偏差，不能聲稱
原樣重現官方 Python 3.12 / PyTorch 2.6 環境。初期 Runtime `pip check` 通過；為修正 sm_120 compiler failure，獨立 overlay
改用 Triton 3.4.0。最終 `pip check` 會指出 torch 2.7 pin Triton 3.3.0 的
一項 dependency mismatch；這是明確記錄的移植偏差，不能稱完整依賴相容。

5. 完整 source build 跳過 Hopper-only FA3/FlashMLA targets，保留並建置 `_C`、
   MoE、FlashAttention-2、allocator；editable install exit 0。
6. Dense prefill 首次在 Triton 3.3.0 報 `computeCapability not supported`；
   改用 3.4.0 後，原 3-stage kernel 需 163840 bytes shared memory，硬體上限
   101376 bytes。僅在 sm_120 將 prefill pipeline stages 改為 1，後續生成通過。

## 已通過的小型 kernel gates

| Gate | 條件 | 結果 |
|---|---|---|
| 官方 H2D/D2H CUDA kernels | FP16、BF16；每方向 3 個非連續 blocks，共 24,576 bytes | 兩種 dtype 均逐 element exact |
| dirty block-table 與 Top-K swap CUDA modules | sm_120 編譯、載入 | 通過；此項未獨立驗證 mapping correctness |
| 官方 odd-GQA page scorer | 24 query heads / 8 KV heads、head_dim 128、128 pages，synthetic metadata | 首步與第 16 步 reference allclose 通過，實測最大 BF16 score 差 0 |
| selector gating | head 0 unstable，其餘 stable；rerank=16 | 第 2 步只有 unstable head 評分，stable heads skipped |

Page scorer 的 reference 為 `max_query(sum_dimension(max(q*key_min,q*key_max)))`；
最新 partial page 使用官方 sentinel 規則。Allclose gate 為 rtol=.01 / atol=.25，
此樣本最大差 0 不推廣為所有輸入的 bit-exact。

## 模型整合 gates

| Gate | 設定 | 結果 |
|---|---|---|
| Dense 256 | BF16, batch=1, 256 input + 32 output | 完成生成 |
| FlexiCache 256 | 0 unstable, rerank=1, Top-K 64 pages，未裁剪全部有效 pages | 32 output IDs 與 dense 全相同；沒有 logits gate |
| FlexiCache 2048 profiling | 0 unstable, rerank=1, Top-K 64，64 output | 完成稀疏生成與 Top-K 收集 |
| 官方 stability analysis | 16-step windows, M=56/224 KV heads | 63 decode records，3 windows；產生 single-prompt pilot |
| FlexiCache 2048 hybrid | 56 unstable, rerank=16, Top-K 64，64 output | 完成生成，64 IDs 與同 prompt 的 rerank=1 全相同；pilot 尚未有跨任務品質驗證 |
| FlexiCache 32768 hybrid | 56 unstable, rerank=16, Top-K 64，64 output；prefill chunks 4096，maxlen 33024 | 完成生成並 synchronize；H2D 3 calls / 16053 blocks，D2H 11 calls / 344568 blocks |

## 目前結論與限制

官方 source 的本地移植已在 RTX 5060 Ti / Llama-3.2-3B 完成 dense 與 sparse
生成。這不是原樣安裝：調整建置目標、Triton、prefill pipeline、CPU pool、
allocator 長度參數，並為新模型補 pilot profile。沒有更改 selection scoring。

32K 初試因 avg tokens 沿用 4096，反覆報各層 GPU blocks 不足而 preempt/recompute；
已中止並改成 32768 重試，32K + 64 output 完成。Transfer audit 包裝官方
CUDA transfer entrypoints，計算實際呼叫數與 src block indices 的長度；每 block
為一個 KV head 的 16 tokens。這些 counts 不是 bandwidth 或 latency。
沒有 source code scoring 改動，但 kernel launch 與 dependency 有偏差，不能
宣稱完整計算在所有輸入皆等價。尚無 formal TPOT、RULER 品質或 matched Twilight 比較。

## 本機重跑

```bash
bash scripts/run_flexicache_local.sh \
  --flexicache --unstable-heads 56 --rerank-frequency 16 \
  --prompt-tokens 32768 --max-model-len 33024 --new-tokens 64 \
  --audit-transfers --output /tmp/flexicache-32k-smoke.json
```

在 canonical `headinfer_reproduction/` 目錄執行。環境與 checkout 已保留；
正式量測必須關閉 `--audit-transfers` 及 Top-K logger 並另定計時範圍。
Public mirror 的 drivers 可設定 `FLEXICACHE_SOURCE` 指向本地 checkout。

## Artifacts 與下一步

- Raw：`results/flexicache_port_20260930/`，含 build/install/failure logs、
  kernel gate JSON、generation JSON、Top-K records 與 stability_pilot。
- Driver：`scripts/probe_flexicache_transfer.py`、`scripts/probe_flexicache_selector.py`、
  `scripts/run_flexicache_smoke.py`、`scripts/run_flexicache_local.sh`。
- Source 偏差：`local_port.patch`，SHA256 `356500deafa6c140e0a020ce01f07d0b3544639ea1d8cb1c990f8b6d18e1bce2`；
  helper 改動均含在 patch。NVTX3 header symlink 不在 git diff：link
  `third_party/NVTX/c/include` 至本機 torch environment 的 NVIDIA NVTX include。
- 核對：`verification_summary.json`；最終依賴：`pip_freeze_final.txt` / `pip_check_final.txt`。
- 下一步：正式比較前需多樣文本的 profiling、獨立 quality gate、相同
  VRAM/prompt/生成長度及計時範圍。此輪沒有產生可報告的 formal TPOT。
