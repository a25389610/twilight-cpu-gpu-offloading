# Current-candidate previous-token resident-KV cost breakdown

日期：2026-09-22
性質：current execution path 的 matched diagnostic；不採用為新 baseline

條件：`meta-llama/Llama-3.2-3B-Instruct`、BF16、FlashAttention 2、Batch 1、三題32K、
`p=.90`、32 fixed Decode steps（D1 warm-up、D2–D32 TPOT）。兩邊都使用 `triton_prepare`、
GPU compact GQA union、CPU flat gather、4-chunk gather/H2D、direct attention layout、layer RoPE、
token-level new-KV D2H batching。

## 結論

此次重跑使用目前正式的 GPU compact GQA union + token-level new-KV D2H baseline；control/reuse
唯一 execution 差異是 existing default-off `--twilight-previous-token-resident-cache`。三題 matched
D2–D32 TPOT control mean **120.732 ms/token**，reuse mean
**128.963 ms/token**，差 **+8.231 ms/token**
（ratio-of-means speedup **-6.82%**；負值是變慢）。

這不改變 Selection/membership 或 assembled Attention K/V。003 smoke gate 的 selected-membership SHA、
GQA-union SHA、D1/D2/D32 Attention K/V trace、P4/D1/D2/D32/final logits 全部 exact。trace run 的
TPOT 不使用於此結論。

## Formal TPOT 與 exposed wall

| request | control TPOT ms | reuse TPOT ms | reuse-control ms | speedup | control selection→Attention-ready ms | reuse | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 119.593 | 130.198 | 10.605 | -8.868 | 69.463 | 72.365 | 2.901 |
| 002_vt_i002 | 119.563 | 126.007 | 6.444 | -5.390 | 71.002 | 71.718 | 0.715 |
| 003_qa_1_i011 | 123.041 | 130.684 | 7.643 | -6.212 | 72.937 | 73.116 | 0.179 |

平均 selection membership完成到 Attention input ready 的 exposed wall 為
**71.134 → 72.400 ms/token**
（delta **+1.265 ms**）。這是同一個 post-selection
parent wall scope；不是把互相overlap的CUDA/CPU component相加。

## CPU/GPU movement（每題一個 post-timing diagnostic token的三題平均）

| component | direction | control MiB | reuse MiB | delta MiB | control CUDA ms | reuse CUDA ms | delta CUDA ms |
|---|---|---:|---:|---:|---:|---:|---:|
| selected-KV CPU-to-GPU H2D | CPU→GPU | 495.520 | 39.073 | -456.446 | 37.676 | 3.978 | -33.698 |
| resident mapping indices | CPU→GPU | 0.000 | 22.612 | 22.612 | 0.000 | 5.293 | 5.293 |
| resident hit gather/copy | GPU→GPU | 0.000 | 456.337 | 456.337 | 0.000 | 7.571 | 7.571 |
| miss scatter into attention layout | GPU→GPU | 0.000 | 39.073 | 39.073 | 0.000 | 0.442 | 0.442 |
| current-union resident snapshot | GPU→GPU | 0.000 | 495.410 | 495.410 | 0.000 | 5.395 | 5.395 |

`resident mapping indices`是四個int64 index tensor的CPU→GPU transfer。hit/miss/snapshot 是GPU→GPU
logical payload；它們不是PCIe bytes。Current token-level new-KV D2H留在兩邊相同的baseline path，
不是 resident-cache 的新增成本。

## 省下的成本 / 新增的成本（active time，不可相加為TPOT）

| item | kind | control ms | reuse ms | delta ms | scope |
|---|---|---:|---:|---:|---|
| CPU selected-KV gather wall | saved | 37.868 | 3.770 | -34.098 | CPU wall; overlaps allowed |
| selected-KV H2D active | saved | 37.676 | 3.978 | -33.698 | CUDA Event; overlaps allowed |
| hit/miss CPU mapping | added | 0.000 | 28.454 | 28.454 | CPU wall: searchsorted, hit/miss partitions, index concatenation |
| resident index H2D | added | 0.000 | 5.293 | 5.293 | CUDA Event; index payload separately reported |
| resident hit gather/copy | added | 0.000 | 7.571 | 7.571 | GPU CUDA Event |
| miss scatter | added | 0.000 | 0.442 | 0.442 | GPU CUDA Event |
| resident snapshot | added | 0.000 | 5.395 | 5.395 | GPU CUDA Event; full union copied every token |
| CPU position snapshot clone | added | 0.000 | 0.738 | 0.738 | CPU wall |
| cache update total wall | net | 49.951 | 53.213 | 3.262 | CPU wall parent; contains overlapping work, not additive |
| pack-ready synchronization wait | net | 0.056 | 0.062 | 0.006 | CPU wall wait |

最大新增 GPU active component 是 resident hit gather/copy，其後是每token完整resident snapshot；
CPU端最明確新增的是 hit/miss `searchsorted` mapping。即使selected-KV H2D與CPU gather都下降，
這些新增項和同步/control路徑仍使exposed wall與TPOT變慢。

## Memory accounting

| item | mean MiB | min | max | note |
|---|---:|---:|---:|---|
| persistent resident K/V | 577.713 | 569.588 | 589.902 | actual GPU object allocation |
| temporary mapping indices | 1.128 | 1.111 | 1.149 | logical live GPU payload, not allocator peak |
| temporary hit-gather payload | 22.837 | 22.258 | 23.476 | logical live GPU payload, not allocator peak |
| persistent CPU selected-position metadata | 7.741 | 7.510 | 7.944 | pageable CPU tensors |
| additional pinned CPU memory | 0.000 | 0.000 | 0.000 | resident prototype incremental allocation |
| final CUDA allocated delta | 580.352 | 571.627 | 594.499 | matched process end-state |
| peak CUDA allocated delta | 0.000 | 0.000 | 0.000 | whole-process allocator high-water delta |
| peak CUDA reserved delta | 0.000 | 0.000 | 0.000 | whole-process reserved high-water delta |

Persistent resident K/V是實際GPU object allocation。temporary mapping/hit-gather rows是由tensor shape
導出的logical live payload，不宣稱等於allocator bytes；因此另列matched `final/peak allocated/reserved`
delta。resident prototype本身新增 pinned CPU allocation是 **0 MiB**；它保留的selected-position
metadata是pageable CPU tensor。既有112 KiB token-batched new-KV pinned staging在control/reuse兩邊都有，
不列為cache增量。

## 限制與研究決策

- 每題只有一組fresh matched pair；formal TPOT的比較只對這一輪有效，不能與歷史TPOT混為同一分母。
- component/transfer資料各題只來自D2–D32後的一個diagnostic token，且CPU wall/CUDA Event可overlap。
- `post_selection_exposed_wall_seconds`使用Selection membership boundary到Attention-ready的parent scope；
  它是要避免component相加的exposed wall證據，但不是Nsight全系統timeline。
- 本輪未修改Selection、selected membership、Attention、KV precision、CPU gather或cache algorithm；
  只增加metrics-gated diagnostic counters。

研究決策：此previous-token prototype仍不應併入current baseline。最大的新增成本是CPU mapping加上
GPU hit-copy/snapshot與index metadata transport；後續若重新設計cache，必須先消除這些本體成本，
而不是僅增加保留token數。

## Artifacts

- `results/twilight_previous_token_resident_cost_v2/manifest.json`
- `results/twilight_previous_token_resident_cost_v2/formal/<request>/<control|reuse>/result.json`
- `results/twilight_previous_token_resident_cost_v2/smoke/003_qa_1_i011/`（local-only huge correctness traces）
- `results/twilight_previous_token_resident_cost_v2/analysis/`
- `scripts/run_twilight_previous_token_resident_cost_v2.py`
- `scripts/analyze_twilight_previous_token_resident_cost_v2.py`
