# GPU metadata提前建立：單變因排程pilot

## 改動與控制

使用者授權在大H2D前建立GPU metadata。新增`--twilight-early-gpu-metadata`，default False。
基底為triton_prepare、bitmap union、flat gather、direct layout與RoPE-only；完整QKV與native
gather關閉。只把同一cu_seqlens建立由GPU new-token copy之後，移到union lengths計算完成後、
CPU gather及H2D之前。tensor內容/dtype/cumsum與搬運API不变，沒有刪除安全wait或增加同步。
原版與variant都新增同一diagnostic-only metadata wall timer，普通TPOT不開metrics。

32K qa_1 i011、p=.90，各32 fixed decode及5 diagnostics，四process兩輪交換順序；沒有accuracy/grid。

## 結果（ms/token）

| Trial | Mode | TPOT mean | TPOT median | Metadata wall | Selection wall | H2D device |
|---|---|---:|---:|---:|---:|---:|
| 1 | baseline | 181.91 | 181.06 | 36.18 | 53.18 | 37.66 |
| 1 | early | 173.66 | 172.54 | 2.64 | 76.56 | 37.57 |
| 2 reverse | baseline | 185.94 | 181.47 | 35.88 | 52.13 | 37.60 |
| 2 reverse | early | 173.81 | 172.63 | 2.75 | 73.32 | 37.60 |

兩輪mean下降8.24/12.14ms（4.53%/6.53%）；median下降8.52/8.85ms（4.70%/4.87%）。
第二輪baseline mean受較慢tokens影響，不能只挑6.53%作代表。此仍單request/兩輪 preliminary，
較早不同process的215ms或236ms不是本輪baseline，不可跨run拼接累計加速。

Metadata呼叫減少約33ms，但不是省掉33ms：Selection wall反而增加約21–23ms，
append_and_d2h_enqueue由48.42/47.89ms降到12.12/13.66ms。工作排程改變後，阻塞位置也改變。
Selection wall不具獨立純selector算術意義，可能包含先前queued projection/Attention債務；
不能把73–77ms當selector算法變慢的證據。H2D device穩定37.6ms、bytes不變，沒有聲稱搬運變快。
本輪未重新取得完整async kernel trace，以上是wall位置變化與TPOT實測，不做精確overlap歸因。

## Correctness、安全性與限制

兩輪10對diagnostic indices/logits hashes、8對P4/D1/D2/D32 checkpoint tensors均bit-exact；
逐step B0、union history count、H2D/D2H bytes相同。Twilight helper tests通過。
預設False，正式表與既有artifacts不覆寫；CPU full KV、packing buffers與KV內容不變。
cu_seqlens Tensor生命週期保留到Attention，原buffer completion/events/record_stream機制不變。

下一決策：先擴展同32K固定3題的matched repeats，再考慮其他context/p。不得用新的Selection
wall比例宣稱QK算術占TPOT變高；如要分解需重新量同一非同步path。

Source：`source/headinfer/headinfer/twilight_offload_cache.py`、runner新增旗標。
Artifacts：`results/twilight_early_metadata_v1/`（raw/logits/CSV/logs、manifest source hashes/commands、summary）。
Scripts：`run_twilight_early_metadata_v1.py`、`analyze_twilight_early_metadata_v1.py`。
