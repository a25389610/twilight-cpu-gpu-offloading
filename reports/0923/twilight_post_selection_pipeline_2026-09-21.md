# Twilight post-Selection system pipeline（2026-09-21）

## 結論

本輪只重排 Top-p membership decision 之後的 system work，沒有修改 Quest B0、
INT4 approximate QK、ranking、Softmax、Top-p、selected membership、GQA union
semantics、KV precision 或 Attention algorithm。實驗路徑維持 default-off，production
baseline 行為不變。

`003_qa_1_i011` correctness gate 完整通過，但最佳的 1-GQA-group/chunk（g1）在三組
fresh matched D2–D32 pairs 都比 control 慢。平均 TPOT 為 **117.359 → 121.821
ms/token**，即 **+4.463 ms/token（+3.80%）**。因此本方向判定為 negative result：
不擴展到另外兩題、不更新 `CURRENT_IMPLEMENTATION.md`，也不把 pipeline flag 併入
current candidate。

profiler 顯示 g1 確實讓 CPU gather 與 selected-KV H2D 重疊，但 current baseline 本來
已有 4-chunk gather/H2D pipeline；新增的 GPU union/index chunk handoff 在 CPU 開始前就
已完成，沒有形成額外 overlap，反而增加 event、Python chunk loop、bitmap decode 與 H2D
enqueue overhead。

## 實驗條件與 Selection boundary

- Model：Llama-3.2-3B-Instruct，BF16，FlashAttention-2。
- Request：`003_qa_1_i011`。
- Context：32K；Batch size 1；`p=.90`；32 fixed Decode steps。
- Formal TPOT：D1 warm-up，D2–D32 synchronized CPU wall mean。
- Current execution candidate：GPU compact boolean-bitmap GQA union、token-level batched
  new-KV D2H、CPU flat gather、4-chunk selected-KV H2D、direct Attention layout、early
  GPU metadata、layer RoPE、fused Quest score、quant metadata reuse。
- Correctness run 另開 trace/profiling；其 TPOT 不作正式 performance 結論。

Selection core 的固定邊界為：

```text
Quest B0
→ INT4 approximate QK / candidate scoring
→ ranking / sort
→ Softmax
→ cumulative probability
→ Top-p threshold
→ selected membership / allocated count 完成
→ [Selection barrier]
```

pipeline 路徑在 barrier 後才執行：

```text
整層 batched exact GPU GQA union
→ chunked boolean-bitmap D2H
→ CPU flatnonzero / flat row-index preparation
→ CPU selected-KV gather
→ separate-stream selected-KV H2D
→ default-stream wait + current K/V placement
→ varlen Attention
```

post-Selection CUDA stream 明確 `wait_event(selection_done)`；CPU 等待由該 stream 產生
的 chunk completion event；selected-KV H2D stream 也等待 Selection event；Attention
所在 default stream 最後等待 H2D completion。下一層 Selection 又必須等本層 Attention
完成，因此沒有 `Selection(N+1)` 與 `Gather/H2D(N)` overlap。

## Correctness gate

條件：003、32K、p=.90、32 Decode steps；比較 control、g1、g2。trace run 另外含一個
post-timing diagnostic token，因此 row counts 為 33 steps。

- **22,176** 個 per-Query Selection rows exact；整段 trace SHA-256 同為
  `5a722eabaa39aeb9d1ba8cb28a93e0cb3c5cf24cdeb400719116184368630c06`。
- **7,392** 個 exact GQA union rows exact；SHA-256 同為
  `dc4f999844f83c49cdf94aa47d7ab14f8dbb9a4c4a6e0c982aa33dfb202ec81a`。
- D1/D2/D32 共 **84** 個 Attention K/V + valid-length rows exact；SHA-256 同為
  `9fb6fd1f0b039fff68a0e4a826423898b655ce33c92611a45bbf59ef0e54b7dc`。
- **7,392** 個 new-KV rows exact；SHA-256 同為
  `10a2715d316a302df333649840a9a885f0e0b1b4f7d230a4e5420e93c1d9263f`。
- latest per-Q/group position hashes三路一致；final logits SHA-256 同為
  `1ba23407b4d13fc3baa570c53661a0407b7a36ef79ea0c445ee89a5294ffe045`。

這表示 scheduling 修改沒有改變 Quest/QK/ranking/Top-p、GQA union、selected rows、
Attention inputs、valid lengths、new-KV writeback 或 logits。

## Chunk granularity sweep

第一組 fresh run 同一 source snapshot：

| Path | Groups/chunk | TPOT | 相對同輪 control | Gather/H2D overlap |
|---|---:|---:|---:|---:|
| control | existing 4 row-balanced chunks | **115.733 ms** | — | 未另建共同 timeline |
| g1 | 1 | 121.624 ms | +5.891 ms | 25.086 ms；gather 的72.5% |
| g2 | 2 | 133.810 ms | +18.077 ms | 23.378 ms；gather 的68.2% |
| g4 | 4 | 137.116 ms | +21.384 ms | 17.742 ms；gather 的53.5% |
| g8 | 8 | 147.456 ms | +31.723 ms | 3.983 ms；gather 的13.3% |

g1 最快，因最早開始 H2D 且 gather/H2D overlap 最大；但每層需8個group chunks，K/V
各一次，共16個selected-KV H2D calls，而既有control為4個row-balanced chunks、8 calls。
g2雖與control同為8 calls/layer，但固定按GQA group分chunk，payload不平衡且fill/drain較長。
沒有任一 granularity 優於control。

第一版曾把GPU union的gather/scatter也按group切成小kernels；g1 union CUDA因此膨脹至
11.424 ms/token。根據profiler，第二版改成整層一次batched union，只保留bitmap D2H
chunking，g1 union降至三輪平均2.852 ms。這已排除最明顯的naïve small-kernel問題，
但TPOT仍穩定回歸。

## Fresh matched TPOT

三輪順序交錯；每輪使用自己的fresh control，不以historical 123.103或140.691 ms/token
作分母。

| Repetition | Control | g1 pipeline | Delta | Relative |
|---:|---:|---:|---:|---:|
| 1 | 115.733 ms | 121.624 ms | +5.891 ms | +5.09% |
| 2 | 117.961 ms | 122.286 ms | +4.325 ms | +3.67% |
| 3 | 118.383 ms | 121.554 ms | +3.172 ms | +2.68% |
| **Mean** | **117.359 ms** | **121.821 ms** | **+4.463 ms** | **+3.80%** |

三輪方向一致，因此不是單次order/thermal artifact。pipeline沒有降低TPOT；最佳版本仍使
TPOT增加3.80%。因003沒有穩定正收益，依預先gate不跑001/002三題formal。

## Selection latency與post-Selection exposed latency

下表是三組formal TPOT之後各一個D33 diagnostic token的平均；CUDA Event與CPU wall
會重疊，不能相加成TPOT。

| Metric（28 layers aggregate） | Control | g1 pipeline |
|---|---:|---:|
| Selection core wall：Quest開始→membership完成 | 33.734 ms | 33.654 ms |
| Selection core CUDA Event | 27.801 ms | 29.235 ms |
| membership完成→Attention input ready wall | 70.582 ms | 78.247 ms |
| post-Selection GPU union CUDA | baseline既有路徑，未列入新event | 2.852 ms |
| bitmap index D2H CUDA | baseline blocking scope，未列入新event | 1.005 ms |
| CPU bitmap decode | 5.952 ms | 8.417 ms |
| CPU selected-KV gather | 36.842 ms | 34.266 ms |
| selected-KV H2D CUDA intervals | 38.722 ms | 39.356 ms |
| selected-KV H2D enqueue wall | 3.971 ms | 8.124 ms |

Selection core wall兩路只差−0.081 ms（−0.24%），且post stream的event dependency保證所有
union/gather/H2D都從membership完成後才開始；因此Selection latency仍完整暴露，沒有被
pipeline隱藏。CUDA Event差+1.435 ms是三個diagnostic tokens的run-to-run/device variation；
兩路執行相同Selection kernels且exact membership trace一致，不能把該差值當演算法改變。

pipeline後真正的post-Selection exposed wall不是下降，而是 **70.582 → 78.247 ms**。
component scopes有overlap，不能把decode、gather與H2D直接相加；可用的critical-path答案是
78.247 ms exposed wall與正式TPOT回歸+4.463 ms/token。

## Timeline與overlap

g1三個diagnostic tokens平均（28 layers aggregate）：

- CPU gather與H2D overlap：**24.285 ms**；約為gather time的**70.9%**、H2D time的
  **61.7%**。
- bitmap decode與前一chunk H2D overlap：**5.673 ms**。
- post-Selection GPU union/index D2H與CPU decode/gather/H2D overlap：**0 ms**。
- 每層平均fill（Selection barrier→first selected-KV H2D start）：約**0.416 ms**。
- 每層平均steady gather/H2D overlap：約**0.867 ms**。
- 每層平均drain（last gather end→last H2D end，負值截0）：約**0.052 ms**。
- timeline標記的平均layer span：約**2.369 ms/layer**；其外還有host transition、metadata、
  current-K/V placement與new-KV bookkeeping，所以不可用28倍layer span取代完整exposed wall。

關鍵觀察是GPU union與所有bitmap D2H通常在CPU處理第一個chunk前就完成。這些工作只有
約3.857 ms CUDA aggregate，而且host從GPU enqueue轉入CPU loop已有足夠間隔；把它切chunk
不會增加實際concurrency。唯一明顯的overlap仍是CPU gather↔selected-KV H2D，而baseline
已經實作同類型的4-chunk pipeline。

## Memory

- 新增persistent pinned CPU bitmap staging：**259,200 bytes（約253.1 KiB／0.247 MiB）**。
- 新增GPU data buffer：**0 bytes**；重用既有GPU group-membership bitmap與Attention buffers。
- 另外建立CUDA streams/events，但本輪沒有可靠的runtime-object memory accounting，不能把
  未量到的driver/event metadata宣稱為0。

## 研究判斷與限制

本輪否定的是「在目前PyTorch/control layout下，把post-Selection工作再按GQA group做
pipeline會改善TPOT」這個假設，不是否定一般的pipeline概念。current baseline已把真正大的
CPU gather與H2D overlap；剩餘GPU union/index handoff太短，沒有新的可隱藏區間。

限制：只在003做三組repeated matched pairs；因predefined gate未通過，沒有三題結果。
timeline每個formal run只額外量一個D33 diagnostic token；其instrumentation不進入正式
D2–D32 TPOT。沒有Nsight Systems全process trace，也沒有量driver stream/event metadata。

最終決策：保留default-off experimental code供source review，不改production behavior，
不更新`CURRENT_IMPLEMENTATION.md`，並在此方向停止；不自動開始Selection algorithm修改。

## Reproducibility

- Local Git base：`dee81b223e64cfc8b6bd58e36d9f8079bae58691` 加default-off工作樹修改。
- Correctness：`results/twilight_post_selection_pipeline_v1/correctness_v2/`
- Fresh matched/chunk sweep：`results/twilight_post_selection_pipeline_v1/pilot_v2_003/`
- Analysis：`results/twilight_post_selection_pipeline_v1/analysis/summary.json`
- Runner：`scripts/run_twilight_post_selection_pipeline_v1.py`
- Analyzer：`scripts/analyze_twilight_post_selection_pipeline_v1.py`
- Implementation：`source/headinfer/headinfer/twilight_offload_cache.py`

Measured performance source SHA-256：

- `source/headinfer/headinfer/twilight_offload_cache.py`:
  `d7d0efe889ca379e6890280cf614195b4913da4c7f13a598b31dce1bd319e26d`
- `scripts/run_ruler_partial_h2d_tpot_case_v1.py`:
  `ffaa2584b93b0a06da51a857a662d267254ac23ef573eeaea0899d12ef97b829`
- `scripts/run_twilight_post_selection_pipeline_v1.py`:
  `ce1239950fca9041e4c98a0a95668242164f6f6f5b90bffc5fc4c7546c206028`

Analyzer SHA-256：
`295ef9e88b23d96d1315f5fea588065bce5cd1d16fb2ed5bb495a6c29f40afde`。

raw JSON/log/trace留在本機，不同步到公開repository。
