# Twilight selected-KV memory locality diagnostic（003，32K，p=.90）

## 目的與範圍

本輪只做 offline diagnostic，沒有修改 Selection algorithm、CPU KV layout、gather implementation 或正式 execution path，也沒有把 diagnostic trace 的時間當成 TPOT。目標是量測 current candidate 的 final sorted `group_positions` 在 CPU historical slab 中是否具有足夠 spatial locality。

實驗條件：`003_qa_1_i011`、Llama-3.2-3B-Instruct、32K request、batch size 1、`p=.90`、32 decode steps；D1 為 warm-up，統計 D2–D32（31 個 decode tokens）。

Trace 來自既有 correctness run，而不是重新插樁 production path：

```text
results/twilight_post_selection_pipeline_v1/correctness_v2/003_qa_1_i011/rep1/control/result.json
```

該 run 的 manifest 明確為 `twilight_post_selection_pipeline=false`、`qk_backend=triton_prepare`、GPU compact GQA union、token-level new-KV D2H batching；source snapshot SHA256 見同一目錄的 `manifest.json`。trace 中共有 7,392 rows；本分析篩選 D2–D32 後為 6,944 rows（31 × 28 layers × 8 KV groups），每一 row 都通過 sorted-unique 驗證。

主要 provenance hash：`twilight_offload_cache.py=d7d0efe889ca379e6890280cf614195b4913da4c7f13a598b31dce1bd319e26d`、`quest_offload_cache.py=2ac05f1b6bcab00b0ba09f0df266c628b61262320af03b8212277c9394863517`、runner=`ffaa2584b93b0a06da51a857a662d267254ac23ef573eeaea0899d12ef97b829`。

分析程式與結果：

```text
scripts/analyze_twilight_selected_kv_locality_v1.py
results/twilight_selected_kv_locality_v1/003_qa_1_i011/analysis/
```

## 計算定義

- Full selected set：trace 中完整 final union，包含 Sink、Recent 與 Twilight remote selection。
- Remote-only：依每個 decode step 的 `old_length = prompt_token_count + decode_step - 1` 排除 `[0, min(64, old_length))` 與 `[max(sink_end, old_length-(256-1)), old_length)`，沒有硬寫固定 recent index。
- Contiguous run：相鄰 sorted positions 差為 1；adjacent gap 報 `next_position - current_position`，所以 contiguous gap 為 1。
- Runtime geometry：trace result 的 `dtype=bfloat16`、`head_dim=128`、K/V 兩份，因此每個 selected history row 為 `128 × 2 × 2 = 512 bytes`；Quest page 為 16 tokens。
- 所有 overall 數字都是 pooled row-weighted 結果，不是先對各 layer/head 百分比取簡單平均。

## Overall locality

| 指標（D2–D32 pooled） | Full selected | Remote-only |
|---|---:|---:|
| selected rows（total / per decode token） | 33,222,382 / 1,071,690 | 31,007,246 / 1,000,234 |
| contiguous runs（total / per decode token） | 11,198,101 / 361,229 | 11,184,403 / 360,787 |
| rows / run | **2.967** | **2.772** |
| run-count reduction (`1 - runs/rows`) | **66.29%** | **63.93%** |
| run length median / max | 1 / 504 | 1 / 406 |
| adjacent gap median / p90 / p95 / max | 1 / 6 / 24 / 4,114 | 1 / 7 / 26 / 4,114 |
| selected tokens in runs ≥2 / ≥4 / ≥8 / ≥16 | 83.10% / 59.90% / 36.15% / 19.84% | 81.89% / 57.03% / 31.59% / 14.11% |

Median run length is only one token in both scopes even though longer runs account for much of the row-weighted token mass. This means the distribution contains many isolated rows plus a non-trivial tail of short contiguous runs, rather than mostly long ranges.

## 16-token page locality與搬移代價

| 指標 | Full selected | Remote-only |
|---|---:|---:|
| touched pages（total / per decode token） | 4,768,595 / 153,826 | 4,623,667 / 149,151 |
| tokens / touched page（平均） | **6.967 / 16 = 43.54%** | **6.706 / 16 = 41.91%** |
| page occupancy median | 6 / 16 = 37.5% | 6 / 16 = 37.5% |
| exact selected KV bytes（total / per token） | 16,221.87 / **523.29 MiB** | 15,140.26 / **488.40 MiB** |
| whole-page fetch bytes（total / per token） | 37,254.65 / **1,201.76 MiB** | 36,122.40 / **1,165.24 MiB** |
| page-fetch / exact-selected inflation | **2.297×（+129.66%）** | **2.386×（+138.59%）** |
| extra bytes per decode token | **678.48 MiB** | **676.84 MiB** |

Pooled page occupancy histogram（page count；完整 1–16 histogram 另存於 `page_occupancy_histogram.csv`）：

| occupancy | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| Full pages | 445,985 | 449,278 | 416,025 | 296,434 | 195,975 | 254,370 |
| Remote-only pages | 445,089 | 448,382 | 415,353 | 295,538 | 195,079 | 122,434 |

Remote-only 排除 protected tokens 後平均 page occupancy 只由 43.54% 降到 41.91%，但 whole-page inflation 由 2.297× 變成 2.386×；因此 Sink/Recent 確實讓 locality 看起來稍好，卻沒有改變整頁搬移昂貴的主要結論。

## Step與layer分布

選幾個 decode step 的 pooled slice：

| step | scope | rows/run | tokens/page | page inflation |
|---:|---|---:|---:|---:|
| D2 | Full | 2.998 | 7.052 | 2.269× |
| D2 | Remote-only | 2.807 | 6.800 | 2.353× |
| D16 | Full | 2.971 | 6.982 | 2.292× |
| D16 | Remote-only | 2.776 | 6.723 | 2.380× |
| D32 | Full | 2.926 | 6.836 | 2.340× |
| D32 | Remote-only | 2.728 | 6.569 | 2.435× |

28-layer pooled summaries are in `layer_summary.csv`. Across layers, `rows/run` ranges 2.384–3.520 (Full) and 2.249–3.323 (Remote-only); page occupancy ranges 5.736–8.153 and 5.434–7.946 tokens/page respectively. Remote p95 gap ranges 17–46 tokens (Full 16–40), so the observation is not confined to one exceptional layer. The least local remote layer by page inflation is layer 22 (2.945×), while layers 18/1 are the most local (about 2.01–2.03×).

## 回答與下一步判斷

1. **selected KV 有中度但不強的 locality。** 約 82–83% selected tokens 落在至少 2-token runs，但 median run 仍為 1，平均只有 2.77–2.97 rows/run；remote-only 比混合 protected set 稍差。
2. **Sink/Recent 的影響存在但有限。** 加入 protected tokens 使 rows/run 增加約 7%，`run≥16` token fraction 由 14.11% 提高到 19.84%，但 page occupancy 仍只有約 44%。
3. **整頁 gather 不適合直接採用。** 16-token page fetch 會將 exact payload 放大到 2.30×（Full）或 2.39×（Remote-only），每 decode token 額外搬約 677–678 MiB；這不是目前 evidence 支持的低風險優化。
4. **run-based gather 有可測試的空間，但不是「大範圍連續 copy」。** Row-level access 理論上可由 33.22M/31.01M rows 合併成 11.20M/11.18M runs，亦即 66.29%/63.93% 的 run-count reduction；然而多數 run 很短，需保留 exact per-row fallback。
5. **下一個最值得 benchmark 的方向：sorted run-copy / hybrid gather。** 建議先做 offline microbenchmark 或 diagnostic-only prototype：對長度 ≥2（再比較 ≥4/≥8）使用 contiguous copy，其餘維持目前 flat gather。page-oriented gather/layout 暫不優先；page-aware Selection 也不能只靠本輪 locality 數據宣稱有效。

這些是 locality characterization，不是 CPU gather TPOT savings。尚未量測 run-aware implementation 的實際 cache/NUMA/IPC/TPOT 效果，也只涵蓋 request `003_qa_1_i011`；因此本輪不更新 `CURRENT_IMPLEMENTATION.md`，不更改 production behavior，且依要求不主動上傳 GitHub。
