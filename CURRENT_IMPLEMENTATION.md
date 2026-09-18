# CURRENT_IMPLEMENTATION

這份文件只描述目前正式採用、用來取得約 140 ms/token 結果的 Twilight
baseline。它不是歷史實驗清單，也不是研究進度報告；判定依據是實際
runner command、case `result.json`、`manifest.json`、source snapshot 與
正式 report。未列入目前 command 的選項，不應視為目前 execution path 的
一部分。

## 1. Baseline identity

| 項目 | 目前正式設定 |
|---|---|
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| dtype / attention | `bfloat16` / `flash_attention_2` |
| Context | 32K request cohort (`ctx32768`；第一題 prompt 為 31,938 tokens，實際各題以 request artifact 為準) |
| Batch size | 1 |
| p | `top_p = 0.90` |
| Decode | fixed-token case（`fixed_token_id = 1`），`decode_steps = 32`，`D1` warm-up，正式 TPOT 取 `D2–D32` 共 31 tokens |
| Head configuration | `short_head_count = 0`，`full_head_count = 224` |
| Cache policy | sink `64`、recent `256`、Quest block size `16` |
| 最新正式 TPOT | 三題 coarse mean **140.691 ms/token**；median **139.521 ms/token**；range **139.390–143.162 ms/token** |
| 正式 report | [`reports/twilight_three_request_complete_2026-09-15.md`](reports/twilight_three_request_complete_2026-09-15.md) |
| 主要 artifact | `results/twilight_three_requests_20260915/manifest.json`、各 request 的 `latest/coarse/result.json` |

TPOT 是 synchronized wall time per output token，排除 prefill/model load；
不要把同一 run 的 `profile-breakdown` diagnostic wall 當成正式 TPOT。三題
最新 coarse 結果為：

| request | D2–D32 mean |
|---|---:|
| `001_niah_multikey_3_i011` | 143.162 ms/token |
| `002_vt_i002` | 139.390 ms/token |
| `003_qa_1_i011` | 139.521 ms/token |

## 2. Source snapshot and authoritative runner

目前 public repository 的 source snapshot 是 commit `9d6316d`（後續的
report/doc commits 不代表 production source 改變）。該 snapshot 的
implementation 與本次 measured artifact 的 source hashes 對應；關鍵
SHA-256 如下：

| source | SHA-256（manifest） |
|---|---|
| `source/headinfer/headinfer/twilight_offload_cache.py` | `ec57dd9117522d1023aa3840ae7145d8a8ad3036cff0ebe9a4e0d3f56fa5c908` |
| `source/headinfer/headinfer/mp.py` | `17cc266001bfa9fa0440ec8371d11cbe75db33cf11cee631d58e6593e5547806` |
| `source/headinfer/headinfer/twilight_fused_qk.py` | `ed75777a4a7c7733341024bba3fc760c846d58f9ade96d0959a26eff840988e7` |
| `source/headinfer/headinfer/twilight_fused_quest.py` | `6f44b412e6502ecc7c620511ce2f8f1652ace292539a0d679c2c0d07ba2c92e9` |
| `source/headinfer/headinfer/cpu_token_union.py` | `bdd9afd7b158cccc28884c4125f28a7089a3759b8f27a57f18c12492c45f2edb` |
| `scripts/run_ruler_partial_h2d_tpot_case_v1.py` | `4367fbee8789d6c28e9a258a6d06391ac18df07dba2cd9e27e1da373e0c26eba` |

正式三題 orchestration runner 是 workspace 中的
`scripts/run_twilight_three_requests_20260915.py`；它先從
`results/twilight_gather_improvements_v1/manifest.json` 取得最新 command，
再對三個 request 以 `latest` version 執行。`manifest.json` 的 command
與 return code 是本 baseline 的 authoritative record。runner 另有
`legacy_reconstructed` branch，但那一支會強制 `qk_backend=pytorch`、
`gather_h2d_chunks=0` 並移除下列 flags，**不是本文件描述的 baseline**。

目前 coarse case 的核心 command（path 以 artifact 中的 request/output
替換）是：

```text
python scripts/run_ruler_partial_h2d_tpot_case_v1.py \
  --request-dir <timing_requests>/<request> \
  --short-head-count 0 --decode-steps 32 \
  --profile-breakdown --profile-breakdown-steps 5 \
  --twilight-top-p 0.90 --twilight-budget-mode dynamic \
  --quest-layer-batched-selection --twilight-gqa-group \
  --twilight-qk-backend triton_prepare \
  --full-flat-h2d-reference-json <matched-full-flat-reference> \
  --local-files-only \
  --twilight-cpu-flat-gather --twilight-cpu-bitmap-union \
  --twilight-direct-attention-layout --twilight-layer-rope \
  --twilight-early-gpu-metadata --twilight-gather-h2d-chunks 4 \
  --twilight-reuse-quant-metadata --twilight-fused-quest-score \
  --twilight-skip-unused-host-views
```

`--profile-breakdown` 只在 TPOT 後額外量測 5 個 diagnostic tokens；它不
改變正式 D2–D32 TPOT 的定義。`full-flat-h2d-reference-json` 是 matched
denominator/reference，不是把 Full attention 放進 Twilight execution
path。

## 3. Flags and execution paths actually enabled

以下值以 runner command 及 `result.json` 為準；`false` 表示該 path 在
目前 baseline 明確未開啟，而不是「尚未查到」。

### Selection / scoring

| option | effective value | 說明 |
|---|---|---|
| `qk_backend` | `triton_prepare` | 只融合 candidate packed-K/scale/min gather、INT4 unpack/dequant 與 FP32 K preparation；後續仍走原本 FP32 QK/Top-p 計算。 |
| `quest_layer_batched_selection` / `layer_batched_selection` | `true` | 以 layer 及 GQA group 批次準備 query/Quest metadata。 |
| `gqa_groupwise_execution` | `true` (`--twilight-gqa-group`) | 將 Q heads 按 KV group 做 union 後進入 varlen attention。 |
| `twilight_top_p` | `0.90` | 第二輪 approximate-QK 的 dynamic Top-p。 |
| `twilight_budget_mode` | `dynamic` | 每個 Q/group 的 selected count 由 cumulative probability 決定，不是固定每 head K。 |
| `candidate_token_budget` | `8192` | 第二輪候選 token 上限；block size `16`，候選 page 上限由此換算。 |
| `matched_budget_fraction` | `0.05` | runner 傳入的 matched-budget 參數；本 baseline 的 effective mode 仍是 `dynamic`。 |
| `fused_quest_score` | `true` | 使用 fused Quest page-score implementation。 |
| `reuse_quant_metadata` | `true` | decode 間重用 packed INT4 quant metadata。 |
| `layer_rope` | `true` (`--twilight-layer-rope`) | layer-level RoPE batching；這不是 full-layer QKV projection batching。 |

### CPU union, gather, transfer and attention layout

| option | effective value | 實際路徑 |
|---|---|---|
| `cpu_bitmap_union` | `true` | CPU NumPy bitmap membership + `flatnonzero` 做每個 GQA group 的 union。 |
| `cpu_flat_gather` | `true` | 建立 flat row indices；目前由 CPU `torch.index_select` 取 K/V。 |
| `cpu_native_gather` | `false` | 不走 C++/OpenMP native gather。 |
| `cpu_run_gather` | `false` | 不走 `CpuKVRunGather` run-copy path。 |
| `gather_h2d_chunks` | `4` | flat host rows 分成四個 chunk；每個 chunk gather 後立即 nonblocking H2D，與下一 chunk 的 CPU gather pipeline。 |
| `direct_attention_layout` | `true` | 直接寫入 GPU attention layout，保留 current-token row，避免每 layer 做完整 history+new-token GPU `cat`。 |
| `early_gpu_metadata` | `true` | 在 CPU gather/H2D 前建立 `cu_seqlens` 等 GPU metadata。 |
| `skip_unused_host_views` | `true` | flat gather path 不建立未使用的 host views。 |
| `fused_final_indices` | `false` | final selected positions/counts 仍走目前既有 bundle 與一次 GPU→CPU transfer。 |
| `writeback_mode` | `delta` | 新 KV 以 delta-only D2H schedule 寫回；不是每 token 重新搬完整 cache。 |
| `activation_wait_mode` | `event` | 以 event-based synchronization 等待 activation/cache readiness。 |
| `cpu_threads` | `6` | result artifact 的 effective PyTorch CPU thread count。 |

### Explicitly off in the same result

`full_layer_flat=false`、`twilight_layer_flat_ragged=false`、
`twilight_layer_projection=false`、`twilight_per_head_varlen_reference=false`、
`quest_layer_flat_ragged=false`、`quest_per_head_varlen_reference=false`、
`sparse_gather_stabilized=false`、`host_memory_trace=false`、
`twilight_detailed_selection_profile=false`。`short_head_count=0` 也表示這
個 32K baseline 沒有另外保留 short-head Full path。

## 4. Actual data flow

目前正式 execution path 可簡化成下列順序；這是依 source control flow
整理，不是以概念圖替代 command：

1. 每一 layer 做 Q/K/V projection；`layer_rope=true` 時，RoPE 以 layer-level batch 方式處理 query/KV。
2. `prepare_layer_selection` 先以 Quest page metadata 做第一輪 page score/Top-k，形成最多 8192 tokens 的 candidate pool。
3. 第二輪以 packed INT4 K 的 quant metadata 做 candidate preparation；`triton_prepare` 融合 gather/unpack/dequant materialization，接著使用 FP32 QK、scale、argsort、Softmax、cumsum/searchsorted 取得 dynamic Top-p counts 與 selected positions。
4. counts、desired counts 與 padded final positions 經一次 GPU→CPU transfer；這個 `.cpu()` 所含的是 queued GPU work 等待加 transfer，不宣稱是純 DMA 時間。
5. CPU 對每個 Q 使用 `sink + selected + recent` positions，並以 `cpu_bitmap_union` 對同一 GQA KV group 做 union。
6. `cpu_flat_gather` 建立 flat row indices；四個 pipeline chunks 各自以 CPU `torch.index_select` gather K/V，立即 enqueue nonblocking H2D。`cpu_native_gather` 與 `cpu_run_gather` 在這條 path 都是關閉的。
7. `direct_attention_layout` 將 history rows 與 current-token slot 直接放進 GPU attention layout；`early_gpu_metadata` 提前準備 varlen metadata。
8. `mp.py` 以 GQA group queries、`cu_seqlens` 與 FlashAttention varlen 執行 attention；layer output 再回到原本 decoder path。新 KV 的 delta-only D2H schedule 仍同時存在。

## 5. Implemented but not part of the baseline

### 已實作、目前 command 沒有採用

- `cpu_native_gather=true`：native C++/OpenMP gather implementation 存在，但目前 baseline 用 CPU `torch.index_select`。
- `cpu_run_gather=true`：`CpuKVRunGather` run-copy/hybrid path 存在，但 runner 沒有開啟。
- `fused_final_indices=true`：fused final-index bundle 存在，但最新 command 沒有 `--twilight-fused-final-indices`。
- `twilight_layer_projection=true`：full-layer QKV projection batching 存在，但 baseline 只有 `layer_rope=true`。
- `qk_backend=triton`：direct INT4→QK backend 存在，但 baseline 使用 `triton_prepare`。
- `full_layer_flat`、`quest_layer_flat_ragged`、`twilight_layer_flat_ragged`、per-head varlen reference、`sparse_gather_stabilized`：均不在最新 coarse command。

### 已測試但沒有納入目前最佳 TPOT path

- Native gather：可執行的 pilot 沒有形成穩定 end-to-end TPOT 優勢，因此不取代目前 CPU `index_select`。
- Run-gather / hybrid copy：已測試但沒有可重現的 TPOT 收益，因此維持 `cpu_run_gather=false`。
- Fused final indices：device bundle 約有小幅 diagnostic 改善，但三題完整 TPOT 沒有改善，故維持 `fused_final_indices=false`。
- `skip_unused_host_views`、early metadata、metadata reuse、4-chunk gather/H2D pipeline 與直接 attention layout 都是在各自 evidence 後組合進最新 full stack；不能把早期單一 pilot 的微小差異當成目前總體 TPOT 的單獨因果量。

### Correctness gate 未通過或不符合目前 exact path

- `qk_backend=triton` 的 direct INT4→QK path：diagnostic hashes / checkpoint exact gate 未通過（最大差異曾達 `0.15625`），因此不採用；目前改用 exact-preserving 的 `triton_prepare` 加原本 FP32 QK。
- `twilight_layer_projection=true` 的 full-layer QKV batching：BF16 exact comparison 未通過，因此只保留 `layer_rope=true`，不在 baseline 開 full-layer projection。

## 6. Evidence boundary

這份文件的 140.691 ms/token 是三個 request 各一輪 coarse formal TPOT，
不是多 trial steady-state confidence interval，也不是把所有 CPU/GPU
diagnostic phase 相加後得到的數字。Selection timer、indices blocking、
CPU union/gather、H2D 與 attention timer 彼此有 scope/clock overlap；它們
只用來定位 control path 與瓶頸，不能直接相加成 TPOT。

若之後要改 selection algorithm，應以本文件的 enabled path 作為固定
baseline，另行保存 command、result、source hash 與 correctness/quality
artifact；不要把本節列出的 experimental/failed path 默默混回 baseline。
