# Twilight CPU union／packing與RAM小型pilot

## 範圍與控制

本輪是execution工程改善，不改Twilight QK、Softmax、Top-p、B0、sink/recent或GQA union集合。
8K與32K各使用既有cohort的`003_qa_1_i011`一題，p=.90，32 fixed-token decode、5個額外diagnostic
tokens。三種模式：baseline、flat gather、flat gather + bitmap union，共6 cases。
三者均使用先前的`triton_prepare` QK backend；不是與舊PyTorch Selection簡報數字直接比較。

## Source audit與改動

- 原有gather以`index_select(..., out=pinned_buffer_slice)`直接完成取出與連續打包，沒有另一次
  完整KV packing copy。雙buffer的H2D completion等待保留，D2H source record_stream不變。
- `--twilight-cpu-flat-gather`把每層8組K/V的16次index_select改成2次，global row indices
  寫入既有可重用buffer；每個entry的offset使用allocated capacity，而不是logical length。
- `--twilight-cpu-bitmap-union`使用可重用NumPy bool membership array，標記各Q-head所選token，
  再以flatnonzero輸出升序、無重複indices。scratch每cache一份capacity bytes，輸出獨立持有storage。
  CPU int64與index bounds檢查保留；不做淘汰、不改selection頻率。
- 兩旗標預設皆False。CPU full BF16 KV slab配置與H2D payload不變。

## 數據（ms/token）

Union/gather為5個post-timing diagnostic tokens的wall mean；TPOT是D2–D32的31-token wall mean。
不要把diagnostic components相加當TPOT。單輪、固定測試順序，未鎖clocks。

| Context | Mode | Union | Gather/pack | TPOT mean | TPOT median |
|---|---|---:|---:|---:|---:|
| 8K | baseline | 35.36 | 36.50 | 213.50 | 209.05 |
| 8K | flat | 34.66 | 35.07 | 207.62 | 207.85 |
| 8K | bitmap + flat | 8.86 | 35.45 | 182.25 | 181.32 |
| 32K | baseline | 36.77 | 46.47 | 241.08 | 242.04 |
| 32K | flat | 37.28 | 44.77 | 240.28 | 239.75 |
| 32K | bitmap + flat | 12.09 | 44.23 | 213.83 | 213.36 |

整體mean下降約14.6%／11.3%；主要可歸因的觀察是union大幅縮短，單獨flat gather效益很小。
Synthetic hot-cache gather在6 threads下降約40%／35%，但真實request未重現同等幅度，因此
不能用microbenchmark宣稱已解決gather瓶頸。numpy_take亦未勝過6-thread flat index_select。

## Correctness與量測限制

- 四組variant vs baseline比較，共20對diagnostic steps的selected indices/logits hashes相同；
  逐step B0、union history count及H2D bytes相同；16對P4/D1/D2/D32 checkpoint tensors bit-exact。
- `test_twilight_offload_cache_v1.py`通過；新union helper測試覆蓋empty、duplicates、boundary、
  scratch reuse與invalid index；microbenchmark gather輸出相同。
- 本輪啟用`--host-memory-trace`，checkpoint採樣在per-token latency timer外，但會擾動token之間
  排程，且overall decode elapsed包含部分採樣成本。因此不當成未instrumented正式TPOT。
- 首個8K baseline有cold-start major faults與D2–D32最大295.89ms outlier；其後variants皆warm，
  8K mean改善幅度可能受執行順序影響。median仍下降，但須warm、交錯順序repeated trials確認。
- 未跑4K/16K、其他tasks/p、accuracy或連續requests壓力測試；不能宣稱全域bit-exact或無leak。

## RAM trace

所有6 cases在P4→D1 pinned reserved皆增加268,435,456 bytes（256 MiB），D1→D32不再增加。
此與首次decode建立layer-flat pinned雙buffer一致，不是當年逐head完整KV重配置的約7 GiB cliff。

| Context／mode | P4 RSS GiB | D1 RSS GiB | D32 RSS GiB | D1→D32 pinned reserved增量 |
|---|---:|---:|---:|---:|
| 8K baseline | 2.669 | 3.202 | 3.219 | 0 |
| 8K bitmap+flat | 2.678 | 3.175 | 3.191 | 0 |
| 32K baseline | 5.678 | 6.183 | 6.199 | 0 |
| 32K bitmap+flat | 5.678 | 6.178 | 6.194 | 0 |

所有checkpoints VmSwap=0；D1→D32 major faults不增加。RSS仍小幅增加，不能稱絕對flat。
沒有證據支持「舊allocator cliff／swap是本輪steady TPOT主要原因」，但這不是排除所有memory leak。
此32K request實際prompt不是32768整數邊界，不能代替精確32768→32769的舊boundary重現測試。

## Artifacts與下一步

- `results/twilight_cpu_pack_v1/{manifest,bitmap_manifest,microbenchmark,analysis}.json`
- 同目錄六份case JSON、logs、per-token CSV與logits；manifest保存commands與主要source hashes。
- `source/headinfer/headinfer/cpu_token_union.py`、`twilight_offload_cache.py`
- `scripts/{run,analyze}_twilight_cpu_pack*_v1.py`與`benchmark_twilight_cpu_pack_v1.py`

下一步：先以warm、交錯baseline/variant順序、關閉RAM trace的固定3 requests確認收益；再針對
剩餘gather/packing的實際訪存與CPU scheduling細分，而不是繼續僅減少呼叫數。
若要排除跨request leak，另做cache建立/釋放的重複小型壓力測試。
後續Twilight計算方法研究需共享同一改善後execution baseline，報告絕對ms與TPOT，不能只以
QK占比變大作為研究效益證據。本輪不改演算法、不覆寫舊正式表。
