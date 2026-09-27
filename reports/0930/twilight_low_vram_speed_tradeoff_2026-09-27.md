# Twilight 低 VRAM 加速路徑：移除 QKV Graph 的 matched 比較

日期：2026-09-27（Asia/Taipei）。本報告回答：能否避免最新最快版本額外約 240.550 MiB 的 GPU allocation，仍取得接近的 Decode TPOT 改善？**可以**。保留完整 Top-p Graph 與 GPU mapping handoff，關閉逐層 QKV Graph，三題各三輪 matched formal TPOT 為 **71.735 → 65.529 ms/token（-8.65%）**；D32 live CUDA allocated 只增加 **1.817 MiB**。完整三項 Graph／handoff 版本同輪為 **63.426 ms/token（-11.58%）**。低 VRAM 版保留約 **74.7%** 的完整版本加速，差 **2.103 ms/token**，但比它少用 **238.734 MiB** D32 live VRAM。

## 條件與邊界

- Llama-3.2-3B-Instruct、BF16、RTX 5060 Ti 16GB、Batch 1、32K；Quest `B0=8192`，Twilight dynamic Top-p `p=.90`，原三題 `001_niah_multikey_3_i011`、`002_vt_i002`、`003_qa_1_i011`。
- 每次 32 個 fixed Decode steps；D1 warm-up，正式 TPOT 是未插入同步 profiling 的 D2–D32 共 31 token 同步 CPU wall。Prefill、model load、D1 Graph capture 不計入 TPOT。
- Frozen baseline 是 GPU bitmap resident cache + GPU mapped CPU KV read，並啟用 09/26 已驗證的三項 VRAM flags：history staging alias、lazy GPU pack、on-demand Attention capacity。完整 historical KV 仍留在 CPU pinned slab。
- **低 VRAM**只增加 `--twilight-exact-top-p-graph --twilight-resident-precompute-gpu-mapping`；**完整 Graph**再增加 `--twilight-projection-graph`。三組共用同一 source、request、execution condition；三輪交錯執行，逐 request／輪次 matched 比較。所有 flags 仍為 default-off opt-in。
- 本輪絕對 TPOT 隨時間有漂移，所以只以本輪的 matched 三組比較計算收益。09/27 前一批的 74.303 → 66.609 ms/token 與本表不能跨批相減。

## Formal TPOT

| Request（三輪平均） | Frozen baseline | 低 VRAM | 完整 Graph | 低 VRAM 改善 | 低 VRAM 距完整 Graph |
|---|---:|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 72.324 | 66.306 | 64.141 | 6.018 | 2.165 |
| 002_vt_i002 | 72.282 | 64.985 | 63.250 | 7.297 | 1.735 |
| 003_qa_1_i011 | 70.599 | 65.298 | 62.888 | 5.301 | 2.410 |
| **三題 × 三輪平均** | **71.735** | **65.529** | **63.426** | **6.205（8.65%）** | **2.103** |

單位皆為 ms/token。低 VRAM 與完整 Graph 各自對 frozen baseline 的九組逐 pair 都變快；低 VRAM 的 paired 改善範圍是 **4.791–9.664 ms/token**，median **5.626 ms/token**。完整 Graph 的平均改善是 **8.309 ms/token（11.58%）**。首輪 002 baseline 偏慢，拉大該 pair 改善；這是保留三輪、同輪配對和報告逐 pair 範圍的理由。

## GPU VRAM

下表是三題 memory inventory 平均，單位 MiB。Memory inventory 是獨立 diagnostic run；它的 TPOT 不參與上表 formal TPOT。`allocated` 是 CUDA allocator live bytes；`reserved` 是 allocator 保留的 pool，不能當成 live KV bytes。

| 指標 | Frozen baseline | 低 VRAM | 完整 Graph | 低 VRAM − baseline |
|---|---:|---:|---:|---:|
| D32 live allocated | 7,468.114 | 7,469.931 | 7,708.664 | **+1.817** |
| Decode peak allocated | 7,586.138 | 7,586.595 | 7,825.207 | **+0.457** |
| D32 reserved | 9,916.667 | 7,904.667 | 8,515.333 | -2,012.000 |
| Through-D1 process peak allocated | 9,119.525 | 9,119.525 | 9,119.525 | 0.000 |
| Unique KV cache GPU storage | 1,324.786 | 1,324.786 | 1,324.786 | **0.000** |

低 VRAM 版與完整 Graph 版的 D32 live allocated 相差 **238.734 MiB**，Decode peak allocated 相差 **238.612 MiB**。完整 Graph 的額外 live memory 主要來自每層 QKV Graph 所保留的 static input／output 與 Graph allocations；不是完整 32K KV 的 GPU mirror。低 VRAM 路徑仍從 CPU pinned historical KV 做 GPU mapped read，previous-token resident K/V、Quest metadata、Twilight INT4 metadata 和 selected Attention-layout K/V 的 unique storage 沒有改變。Reserved 顯著下降可能受 Graph pool／allocator 行為影響；不據此聲稱實體 VRAM 多省 2 GiB。

## Correctness

- 九組 formal 三臂比較中，同 request 的 P4／D1／D2／D32 checkpoint 與 final logits SHA-256 全部一致。
- 003 獨立 selection diagnostic：Quest selection **21,504** rows、Twilight budget **21,504** rows、GQA union **7,168** rows，低 VRAM 與原版逐項 exact；checkpoint／final logits exact。
- 003 獨立 Attention diagnostic：Attention K/V **84** rows、new-KV writeback **7,168** rows，低 VRAM 與原版逐項 exact；checkpoint／final logits exact。
- Diagnostic trace 的 timing 未用作正式 TPOT。這些 gates 只涵蓋目前三題 32K、`p=.90`、Batch 1；尚未聲稱跨 Context、`p`、Batch、GPU 或品質 cohort 一致。

## 判斷與使用方式

若優先控制 Decode VRAM，可採用**低 VRAM opt-in 組合**：在 frozen resident + mapped CPU read + 三項 09/26 VRAM flags 上，只增加完整 Top-p Graph 與 precomputed GPU mapping。這是目前已實測、幾乎不增加 live VRAM 而接近完整 Graph 加速的方案。若多用約 239 MiB live VRAM 可接受，完整 Graph 再快約 2.1 ms/token。先前嘗試共享 QKV Graph memory pool 只降低 reserved，沒有降低 live allocated，因此不能把它算成已找到的更省 VRAM QKV Graph 實作。

本輪**沒有修改 Twilight Selection 演算法、INT4／Quest 精度、selected membership、Attention K/V 或 logits**；差異來自原運算的 Graph submission 與 mapping handoff 排程。指令範例見 raw `results/twilight_low_vram_speed_20260927_v1/matched/rep1/003_qa_1_i011/low_vram/command.json`。

## Artifact 與 source

- Raw commands、formal result／per-token CSV／logits、memory inventory、correctness traces、分析 JSON／CSV：`results/twilight_low_vram_speed_20260927_v1/`。正式逐輪數據：`matched_rows.csv`；摘要：`matched_summary.json`；gate／memory：`gates_summary.json` 與 `gates_memory.csv`。Raw 檔僅保存在本地，不放公開鏡像。
- 分析器：`scripts/analyze_twilight_low_vram_matched_20260927.py`、`scripts/analyze_twilight_low_vram_gates_20260927.py`。
- Runner/source 在三臂 matched run 間未修改。SHA-256：runner `73d1d8137b02447b3ddc327c217f08b9611652963eb32745a7afefc3026b874b`；`mp.py` `805b40565c797ffb4c388a9c0937396ff996be1ebd1263735040d32ad4092e9f`；`twilight_offload_cache.py` `8491eb046506ffef127bcc02b1ea877be11ad0310bf84312567235fb1753e16e`；Top-p Graph helper `7af49f6e1e3f69a3743e9d9551f414217442d31861347b5248c7963b80800b56`。
