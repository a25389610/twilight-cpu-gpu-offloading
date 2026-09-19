# Twilight GPU-side GQA union + compact index handoff（2026-09-20）

## 結論

本輪在**不改 Twilight Selection membership、不改 CPU KV gather／KV H2D／Attention**的前提下，將原本的「24 個 per-Q padded int64 rows D2H，再由 CPU 做 GQA union」改成：

```text
Quest B0 → INT4 approximate QK → Softmax/Top-p
→ GPU 依每個 Query head 的 allocated prefix 寫入 8 組 membership bitmap
→ GPU 加入 Sink/Recent protected range
→ D2H 只搬 8 組 boolean bitmap
→ CPU flatnonzero 還原 exact sorted group_positions
→ 原 CPU flat gather → 原 4-chunk H2D → 原 direct-layout varlen Attention
```

三題 matched fresh control 的 D2–D32 row-weighted mean TPOT 為 **147.356 ms/token**；GPU union boolean-bitmap 版為 **127.861 ms/token**，TPOT 下降 **13.23%**，等價 throughput speedup **15.25%**。三題皆改善，沒有只靠單一 request。

這是 execution-path 改善，不是新的 Selection 演算法貢獻。`--twilight-gpu-compact-gqa-union` 預設為 false；本輪沒有刪除舊 CPU union，也沒有改預設 production behavior。

## 問題與假設

Current baseline 在每層完成動態 Top-p 後，會把 24 個 Query heads 的 counts 與最多 8192 個 positions/head 組成 padded int64 transfer，先做 per-Q positional sort，再搬到 CPU。CPU 隨後用 bitmap 對同一 KV head 的三個 Query heads 做 exact union。

假設是：最終 CPU gather 只需要 8 個 KV groups 的 exact sorted union，因此可在 GPU 直接形成 group membership，移除 per-Q positional sort、padded rows D2H 與 CPU bitmap union；只要 group positions 完全相同，後續 KV payload 與 Attention 可保持 bit-exact。

## 實作

新增兩個 opt-in flags：

- `--twilight-gpu-compact-gqa-union`：正式候選路徑。
- `--twilight-gpu-union-validate-cpu`：correctness-only；同時計算舊 per-Q/CPU union 並逐 group assertion，正式 TPOT 不開。

正式候選的資料結構是可重用的 CUDA `torch.bool[8, max_cache_len + 1]` membership buffer。每層仍保留原 `order` 與 `allocated`，用 `gather` 取得 rank prefix，但**不做 per-Q positional sort**；inactive ranks 寫到 sentinel，active positions 用 `scatter_(..., True)` 合併至對應 KV group。Sink 與依當層 `old_length` 計算的 Recent range 直接標入 membership。D2H 後，CPU `numpy.flatnonzero` 產生既有 gather 所需的 sorted unique int64 positions。

正式路徑沒有重新計算 CPU union。只有 validation flag 或 selection trace 開啟時，才額外保留 per-Q rows；那些 run 不作 TPOT 主張。

## Correctness gate

條件：`003_qa_1_i011`、32K、p=.90、32 fixed decode steps。

- GPU-vs-CPU union assertion：32 steps × 28 layers × 8 KV heads，共 **7,168** 個 exact elementwise comparisons 全部通過。
- Per-Q selected positions SHA-256：兩路皆為 `c9c96efe01bc1caa7308c90bf07fdb2dd4e411cf4c31a7fda0e5cffe36bc7849`。
- Final group positions SHA-256：兩路皆為 `5f77a191afb25741661559c2bed56d3f809ae8c7b27412845ea6fa0750ea5a1c`。
- D1／D2／D32 的 28-layer K/V + valid-length trace 共 84 rows，整段 SHA-256 皆為 `a6cffcdc4cfcfcc6116f5e75b16b428824ed158990c260feabb6bbbf4c14ddd6`。
- P4／D1／D2／D32 logits hashes 全部相同；D32/final 為 `1ba23407b4d13fc3baa570c53661a0407b7a36ef79ea0c445ee89a5294ffe045`。
- 三題 formal matched pairs 的 final logits hash、group positions hash、selected history rows、group-union rows、KV H2D bytes 均逐題相同。

Correctness/trace run 的 latency 因雙路重算與 hash copies 被刻意污染，不列入 performance 結論。

## 正式 TPOT

共同條件：Llama-3.2-3B-Instruct BF16、FlashAttention-2、Batch 1、32K、p=.90、dynamic Top-p、8192-token B0、32 fixed decode steps；D1 warm-up，正式統計 D2–D32；`triton_prepare`、fused Quest score、metadata reuse、CPU flat gather、4-chunk H2D、direct Attention layout、early GPU metadata 與 layer RoPE 均保持一致。previous-token resident cache 關閉。

| Request | Control mean / median / range (ms) | GPU union mean / median / range (ms) | TPOT decrease | Throughput speedup |
|---|---:|---:|---:|---:|
| `001_niah_multikey_3_i011` | 147.525 / 146.964 / 140.334–157.647 | 127.748 / 127.718 / 123.716–132.107 | 13.41% | 15.48% |
| `002_vt_i002` | 144.361 / 143.893 / 140.609–151.366 | 126.656 / 127.454 / 121.353–132.103 | 12.26% | 13.98% |
| `003_qa_1_i011` | 150.182 / 149.436 / 143.679–164.664 | 129.180 / 128.653 / 126.052–136.576 | 13.98% | 16.26% |
| **93-token row-weighted overall** | **147.356 / 146.533 / 140.334–164.664** | **127.861 / 127.823 / 121.353–136.576** | **13.23%** | **15.25%** |

003 的第二組反向次序 matched repetition 為 control **148.576 ms/token**、GPU union **129.768 ms/token**，TPOT 下降 **12.66%**（throughput speedup **14.49%**），方向一致。

Historical current baseline 的三題平均是約 140.691 ms/token；本輪 fresh control 較慢，因此效能結論只使用同 snapshot、同輪 matched control，不以 historical TPOT 當分母。

## Index、CPU union 與新增成本

以下是每個 process 在 D2–D32 正式 TPOT 後的單一 instrumented diagnostic token；CUDA Events 與 CPU wall scopes 可能重疊，不能相加成 TPOT。

| Request | Index D2H control → GPU union | Reduction | CPU union control → optimized | CPU bitmap decode | KV H2D control / optimized |
|---|---:|---:|---:|---:|---:|
| 001 | 42.010 → 6.840 MiB | 83.72% | 13.925 → 0.099 ms | 5.930 ms | 497.327 / 497.327 MiB |
| 002 | 42.010 → 6.990 MiB | 83.36% | 13.846 → 0.094 ms | 5.676 ms | 480.721 / 480.721 MiB |
| 003 | 42.010 → 6.931 MiB | 83.50% | 16.221 → 0.088 ms | 6.420 ms | 508.512 / 508.512 MiB |

`twilight_index_d2h_bytes` 在 diagnostic mode 還包含每層少量 `allocated/desired` counts；正式 metrics-off path 只搬 membership bitmap。003 的 detailed profile 顯示：

- 舊 per-Q gather/mask/sort (`twilight_final_b1_indices`)：**5.705 ms CUDA/token**。
- 新 GPU membership union：**1.152 ms CUDA/token**。
- index sync/transfer wall：**19.658 → 13.205 ms/token**。
- 新 CPU bitmap decode：**6.074 ms/token**。
- 舊 CPU group union：**13.638 ms/token**，正式 optimized path 不再執行。

因此收益不是靠 overlap 隱藏 Selection latency，而是實際移除了 per-Q sort/padded transfer/CPU union；代價是 GPU scatter/bitmap 與 CPU compact decode。

## VRAM

- Boolean membership persistent allocation：三題分別 **255,808 / 261,416 / 259,208 bytes**（約 0.244–0.249 MiB），取決於 request capacity。
- 額外 rank/scatter temporaries 為 layer-sequential；沒有乘上 28 層常駐。
- 每題 control 與 optimized 的 process-level peak allocated/reserved bytes 相同；既有 `triton_prepare` transient 主導 peak，因此本輪小型 bitmap 沒有提高量到的 peak。

## 已測試但不採用：bit-packed bitmap

另測 GPU 每 8 bits 壓成 1 byte，再由 CPU `unpackbits + flatnonzero`。Index D2H 降到約 **0.864–0.883 MiB/token**，但 CPU decode 增至 **13.36–13.69 ms/token**。三題平均 control/bit-packed 為 **147.183/135.398 ms/token**，throughput speedup **8.70%**，明顯差於 boolean bitmap 的 **15.25%**，因此不採用。失敗結果保留，不能只以最小 bytes 判斷 end-to-end 最佳方案。

## 限制與下一步

- 目前 handoff 是 boolean bitmap，不是 counts + packed int32 positions；它以約 7 MiB/token 換取較低 CPU decode。若要再壓 bytes，需用更快的 GPU compaction或 CPU bitset decoder，而不是目前的 `numpy.unpackbits`。
- Formal TPOT 每題一組 matched pair；003 額外一組 repetition。三題方向一致，但未做更多跨 process repetitions。
- 新路徑仍會 materialize `ranked_prefix`；本輪只移除 positional sort與 CPU-side union/index unpack，沒有改 Quest、INT4 QK、Softmax、Top-p 或 candidate ranking。
- 本輪不包含 previous-token KV reuse、新 KV D2H batching、grouped QKV projection、direct INT4→QK 或任何 Selection membership 改動。

下一個可驗證問題是：能否直接產生 packed sorted positions/counts，並把 decode 降到低於目前約 6 ms，而不重新引入 dynamic-allocation synchronization。

## Reproducibility

- Source Git base：`dee81b223e64cfc8b6bd58e36d9f8079bae58691`（工作樹另有本輪 default-off 修改）。
- Chosen source SHA-256：
  - `source/headinfer/headinfer/twilight_offload_cache.py`: `bf05ff5a43a9c43a9546a1658f35bc9cb6419ba7386282675a4104fb0ae19481`
  - `scripts/run_ruler_partial_h2d_tpot_case_v1.py`: `54f965918fa3dbe308631e251c2e3cc4774ec6cb13347a1e2b4f9dc6fb92623a`
  - `scripts/run_twilight_gpu_compact_union_v1.py`: `d52089364d009e00a37344c9f839b4704105b72e4dcdff4632cc6f6a2c4ce576`
- Formal command：

```bash
/home/paul/miniconda3/envs/headinfer_repro/bin/python scripts/run_twilight_gpu_compact_union_v1.py --repetitions 2 --requests 003_qa_1_i011 --output-dir results/twilight_gpu_compact_union_v1/formal_v2
/home/paul/miniconda3/envs/headinfer_repro/bin/python scripts/run_twilight_gpu_compact_union_v1.py --repetitions 1 --output-dir results/twilight_gpu_compact_union_v1/formal_v2
```

- Machine/runtime：NVIDIA GeForce RTX 5060 Ti、PyTorch 2.7.0+cu128、CUDA 12.8。
- Local artifacts：
  - `results/twilight_gpu_compact_union_v1/correctness_v2/`
  - `results/twilight_gpu_compact_union_v1/formal_v2/`
  - `results/twilight_gpu_compact_union_v1/formal_v3/`（bit-packed failed ablation）
  - `results/twilight_gpu_compact_union_v1/profile/`
  - `results/twilight_gpu_compact_union_v1/analysis/summary.json`

公開同步僅包含本 Markdown 報告；不含 raw JSON/CSV/logits/trace/log。
