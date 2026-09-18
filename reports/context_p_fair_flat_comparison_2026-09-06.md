# Full-flat／Quest-flat／Twilight-GQA 公平 Timing 比較

## 結論

四個 Context 的 `Full-flat` 正式 TPOT 已經量過，每個 Context 都有與
`Quest-flat`、`Twilight-GQA` 相同的三筆 requests，因此本次不需補跑，也沒有使用舊版
448 H2D／224 Attention calls 的 Full TPOT 代替。

三者共同條件：Llama-3.2-3B-Instruct、BF16、Batch=1、fixed token decode、32 decode
steps；D1 作 warm-up，正式 TPOT 是 D2–D32 共31 tokens的逐token synchronized wall time
平均。Prefill與model load不計入TPOT。三筆requests的`request_id`與`prompt_sha256`逐項一致。

`Transferred MB/token`使用decimal MB（bytes / 1,000,000）；`Physical KV %`使用每筆
sparse實際K+V H2D bytes除以配對的Full-flat實際K+V H2D bytes，再對三筆取平均。

## 正式 End-to-end Decode TPOT

| Context | Method | Physical KV % | Transferred MB/token | TPOT ms/token |
|---:|---|---:|---:|---:|
| 4K | Full-flat | 100.00 | 379.159 | 63.677 |
| 4K | Quest-flat | 43.85 | 164.807 | 79.296 |
| 4K | Twilight-GQA p=.70 | 28.89 | 106.806 | 104.010 |
| 4K | Twilight-GQA p=.75 | 32.70 | 120.984 | 105.372 |
| 4K | Twilight-GQA p=.80 | 37.46 | 138.707 | 109.405 |
| 4K | Twilight-GQA p=.85 | 43.43 | 161.064 | 122.380 |
| 4K | Twilight-GQA p=.90 | 51.54 | 191.616 | 119.157 |
| 4K | Twilight-GQA p=.95 | 63.84 | 238.384 | 126.951 |
| 8K | Full-flat | 100.00 | 890.820 | 102.039 |
| 8K | Quest-flat | 26.97 | 240.042 | 90.084 |
| 8K | Twilight-GQA p=.70 | 20.99 | 186.297 | 170.336 |
| 8K | Twilight-GQA p=.75 | 24.72 | 219.408 | 182.948 |
| 8K | Twilight-GQA p=.80 | 29.38 | 260.849 | 177.834 |
| 8K | Twilight-GQA p=.85 | 35.46 | 315.025 | 198.174 |
| 8K | Twilight-GQA p=.90 | 43.94 | 390.454 | 209.623 |
| 8K | Twilight-GQA p=.95 | 57.19 | 508.604 | 240.900 |
| 16K | Full-flat | 100.00 | 1,841.163 | 169.959 |
| 16K | Quest-flat | 20.81 | 383.173 | 114.351 |
| 16K | Twilight-GQA p=.70 | 12.31 | 226.131 | 187.526 |
| 16K | Twilight-GQA p=.75 | 14.64 | 268.956 | 193.119 |
| 16K | Twilight-GQA p=.80 | 17.58 | 323.094 | 221.844 |
| 16K | Twilight-GQA p=.85 | 21.45 | 394.274 | 228.465 |
| 16K | Twilight-GQA p=.90 | 26.86 | 493.810 | 246.358 |
| 16K | Twilight-GQA p=.95 | 35.53 | 653.370 | 284.685 |
| 32K | Full-flat | 100.00 | 3,709.622 | 309.152 |
| 32K | Quest-flat | 17.90 | 663.929 | 162.621 |
| 32K | Twilight-GQA p=.70 | 6.24 | 231.289 | 201.077 |
| 32K | Twilight-GQA p=.75 | 7.46 | 276.500 | 196.804 |
| 32K | Twilight-GQA p=.80 | 9.01 | 334.073 | 223.188 |
| 32K | Twilight-GQA p=.85 | 11.06 | 410.303 | 227.050 |
| 32K | Twilight-GQA p=.90 | 14.01 | 519.476 | 273.839 |
| 32K | Twilight-GQA p=.95 | 18.73 | 694.784 | 300.068 |

## Post-timing diagnostic

下表不是正式TPOT的互斥分解。每筆正式timing完成後，另跑一個diagnostic token：Selection
與CPU欄為wall timer，H2D與Attention為CUDA event interval sum。它們可用於定位元件成本，
不可彼此相加或除以正式TPOT當成完整百分比。

`CPU合併／整理`對Quest是CPU gather/pack；對Twilight是GQA group union加CPU gather/pack；
Full-flat沒有sparse selected-index union/gather，因此為0。

| Context | Method | Selection ms | CPU合併／整理 ms | H2D ms | Attention ms |
|---:|---|---:|---:|---:|---:|
| 4K | Full-flat | 0.000 | 0.000 | 27.513 | 4.120 |
| 4K | Quest-flat | 15.739 | 7.568 | 12.060 | 2.484 |
| 4K | Twilight-GQA p=.70 | 32.925 | 16.643 | 7.971 | 2.336 |
| 4K | Twilight-GQA p=.75 | 32.613 | 17.033 | 8.936 | 2.287 |
| 4K | Twilight-GQA p=.80 | 32.810 | 18.908 | 10.261 | 2.452 |
| 4K | Twilight-GQA p=.85 | 34.365 | 24.525 | 11.873 | 2.871 |
| 4K | Twilight-GQA p=.90 | 33.089 | 26.339 | 13.957 | 2.437 |
| 4K | Twilight-GQA p=.95 | 32.640 | 30.208 | 17.089 | 2.307 |
| 8K | Full-flat | 0.000 | 0.000 | 63.431 | 5.589 |
| 8K | Quest-flat | 14.741 | 11.750 | 17.321 | 2.472 |
| 8K | Twilight-GQA p=.70 | 78.639 | 25.614 | 13.366 | 2.678 |
| 8K | Twilight-GQA p=.75 | 83.315 | 30.843 | 15.781 | 2.752 |
| 8K | Twilight-GQA p=.80 | 77.544 | 37.102 | 18.416 | 2.434 |
| 8K | Twilight-GQA p=.85 | 78.509 | 42.183 | 22.343 | 2.760 |
| 8K | Twilight-GQA p=.90 | 77.950 | 53.446 | 27.911 | 2.608 |
| 8K | Twilight-GQA p=.95 | 77.081 | 70.868 | 35.650 | 2.739 |
| 16K | Full-flat | 0.000 | 0.000 | 131.574 | 7.875 |
| 16K | Quest-flat | 17.599 | 21.717 | 27.392 | 2.995 |
| 16K | Twilight-GQA p=.70 | 88.599 | 31.513 | 16.308 | 2.818 |
| 16K | Twilight-GQA p=.75 | 87.929 | 32.801 | 19.108 | 2.523 |
| 16K | Twilight-GQA p=.80 | 95.802 | 44.348 | 23.401 | 2.644 |
| 16K | Twilight-GQA p=.85 | 91.006 | 51.717 | 27.972 | 2.946 |
| 16K | Twilight-GQA p=.90 | 87.375 | 66.438 | 34.791 | 2.855 |
| 16K | Twilight-GQA p=.95 | 87.316 | 92.364 | 46.004 | 3.367 |
| 32K | Full-flat | 0.000 | 0.000 | 265.252 | 13.209 |
| 32K | Quest-flat | 25.152 | 39.415 | 47.249 | 4.664 |
| 32K | Twilight-GQA p=.70 | 99.223 | 29.558 | 16.706 | 2.460 |
| 32K | Twilight-GQA p=.75 | 94.158 | 30.760 | 19.590 | 2.363 |
| 32K | Twilight-GQA p=.80 | 99.389 | 43.256 | 23.816 | 3.328 |
| 32K | Twilight-GQA p=.85 | 94.607 | 49.424 | 28.924 | 2.638 |
| 32K | Twilight-GQA p=.90 | 95.436 | 74.633 | 36.897 | 3.188 |
| 32K | Twilight-GQA p=.95 | 95.426 | 94.168 | 48.968 | 3.424 |

詳細CPU拆分（`cpu_union_ms`與`cpu_gather_pack_ms`）、MiB與sample standard deviation保存於
同目錄的`fair_flat_diagnostic.csv`與`fair_flat_main.csv`。

## 限制

- 三種方法有相同requests與protocol，但不是同一輪interleaved重測；Full-flat／Quest-flat
  於上午完成，Twilight-GQA於下午完成。因此相差只有數ms的設定不可過度解讀。
- 三者都是28次layer-level Attention calls/token；Full-flat使用native GQA dense
  `flash_attn_func`，Quest-flat與Twilight-GQA使用ragged/varlen sparse Attention。這是各方法
  合理的原生資料表示，不是同一kernel family。
- Full-flat TPOT在4K／16K／32K高於先前production Full，原因是layer-flat路徑失去部分
  H2D／Attention overlap；本表刻意使用Full-flat，目的是讓execution granularity與physical
  byte denominator一致。

## Raw evidence

- Full-flat：`results/context_p_ruler_39_v1/timing/full_flat_denominator/<context>/*.json`
- Quest-flat：`results/context_p_ruler_39_v1/timing/main/<context>/quest/*.json`
- Twilight-GQA：`results/context_p_twilight_gqa_group_ruler39_v1/timing/<context>/twilight_p*/*.json`
