# Selection indices 非同步邊界診斷

## 方法與範圍

32K qa_1 i011、p=.90，沿用 early metadata、RoPE-only、triton_prepare、bitmap union、flat
gather、direct layout；full QKV/native gather 關閉。兩個 process 各32 decode+5 diagnostic
tokens，一個原路徑、一個 process-local instrumentation。正式 source 未修改。

每個 diagnostic 入口以 CUDA Event synchronize 建立 CPU/GPU anchor；各 phase 不加同步，
保留原 blocking Tensor.cpu()，沿用原 token-end resolve。每層 prepare_layer_selection 入口
record start event；Tensor.cpu 前 record ready event，返回後 record end event。
入口 event 比正式 selector_wall_started 稍早，包含入口檢查，不能稱完全相同 boundary。

CPU/GPU anchor 最大不確定度0.03171ms；跨clock排隊延遲為估計，包含event提交成本、既有
stream依賴，不能稱純GPU busy time或指出是哪個kernel。28層加總亦累積alignment誤差。
D2H device interval含ready event至copy後event，可能包含CPU allocation/enqueue/return空檔，
不是純DMA。原生CUPTI kernel/memcpy activity仍未取得，本輪不偽造精確DMA拆分。

## 實測（每token 28層累計，5 samples）

| 項目 | Mean ms | Median ms | Min–Max ms |
|---|---:|---:|---:|
| 入口CPU至GPU start event延遲估計 | 24.72 | 25.16 | 21.81–26.59 |
| GPU start至indices ready event | 47.75 | 47.09 | 46.73–49.31 |
| indices ready至copy後event | 5.29 | 5.27 | 5.16–5.53 |
| blocking cpu() CPU wall | 55.15 | 55.33 | 51.58–57.73 |

以上不是可相加TPOT components；blocking wall與GPU工作重疊。每層indices tensor
[24,8194] int64，1,573,248 bytes；每token44,050,944 bytes，含counts與sentinel padding，
不是CPU full KV的D2H。ready相對CPU selection入口平均2.588ms/layer。

第一diagnostic第二層（ms，相對diagnostic anchor）：

```text
CPU Selection入口 6.049 → cpu()呼叫 6.897 ──────────→ 返回 9.495
GPU                     start 7.265 → indices ready 9.330 → copy後event 9.520
```

此層入口排隊估計1.216ms，Selection device interval2.065ms，D2H呼叫device interval
0.190ms，blocking wall2.598ms。不能將ready以前的全部等待歸因indices搬運。

## 一致性與擾動

5對diagnostic selected indices/logits hashes、4個checkpoint tensors全部exact，逐step
B0/union/H2D/D2H相同。腳本py_compile通過。普通TPOT baseline173.92ms、timeline process
173.62ms；此時instrumentation未啟用。Diagnostics baseline173.28ms、timeline174.19ms，
觀察差+0.90ms/+0.52%，單對process包含變異，非已隔離的純timer overhead。

## CPU可提前工作與下一步候選

Source顯示update_layer_gqa_group_ragged的entry/shape validation、length preview/bookkeeping、
固定buffer capacity確認不依賴本token indices；可作候選，但不得提前讀尚未完成D2H的KV，
或改變bookkeeping依賴。Buffer地址/固定metadata亦可預備。這些工作量尚未獨立量測，不能
宣稱足以覆蓋等待。Union、union lengths、cu_seqlens內容、row gather與實際H2D長度均依賴
selected indices，不能在結果ready前直接執行。

結論：55ms不是純transfer，單純改成nonblocking後立即等待無法保證收益。下一步仍是候選：
先測可獨立bookkeeping的實際重疊空間，再決定是否值得同層KV-group pipeline；本輪不改
演算法/排程、不承諾可省24.72或55.15ms。未辨識所有既有queued kernels。

Artifacts：results/twilight_index_wait_v1/{baseline,timeline,boundaries,summary}.json、
timeline.chrome.json、兩個command JSON與checkpoint tensors。
Scripts：scripts/profile_twilight_index_wait_v1.py、scripts/analyze_twilight_index_wait_v1.py。
