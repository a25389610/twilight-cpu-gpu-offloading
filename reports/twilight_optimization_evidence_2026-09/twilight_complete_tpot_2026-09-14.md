# 最新32K TPOT完整host accounting與獨立GPU phase量測

## 條件與結果

32K qa_1 i011、p=.90，沿用fused Quest、skip-unused-host-views、4chunk pipeline、
metadata reuse、triton_prepare等既有候選；hybrid gather關閉。沒有修改正式演算法/source。
CPU-only scopes、GPU Events scopes、原始runner baseline各32 decode＋5 diagnostics，依序執行。

| Run | 正常D2–D32 TPOT ms | 5-token diagnostic wall ms | 相對baseline diagnostic |
|---|---:|---:|---:|
| 原始runner baseline | 141.068 | 144.558 | — |
| CPU互斥scopes | 141.827 | 155.879 | +7.83% |
| GPU Events＋scopes | 142.112 | 184.287 | +27.48% |

本輪正常值約141–142ms，異於上次163.443ms，沒有新優化，不能將差額歸因本輪工作。
Scopes僅diagnostic啟動，正常階段wrapper inactive；原始baseline完全無新wrappers。
差額含profiler、process與token變異，非嚴格同token overhead calibration。
不覆寫原正式grid，不宣稱已排除runtime漂移。

## 完整、互斥CPU wall分區

同一perf_counter，scope stack將每段分配給最內層呼叫。每token從原diagnostic_started到
原diagnostic_wall_seconds，保留原末端cache/device synchronize，不插入逐phase同步。
以下可加總為155.879ms；這是CPU執行緒經過時間，包含dispatch及隱式等待，不是CPU純算術。
GPU非同步計算可能在這些區間同時執行，不能再加上GPU表。

| 分區 | mean ms | % |
|---|---:|---:|
| CPU gather index_select | 36.705 | 23.55 |
| Selection呼叫其餘host工作 | 24.110 | 15.47 |
| final indices blocking cpu() | 21.875 | 14.03 |
| 新KV D2H呼叫及其餘排程（互斥兩項合併） | 11.980 | 7.69 |
| KV控制其餘工作 | 11.767 | 7.55 |
| CPU bitmap union | 10.260 | 6.58 |
| Attention tensor/控制其餘工作 | 8.435 | 5.41 |
| Q/K/V/O projection呼叫 | 6.706 | 4.30 |
| H2D呼叫 | 4.164 | 2.67 |
| Norm呼叫 | 3.261 | 2.09 |
| CPU row offsets、CPU/GPU metadata呼叫 | 2.870 | 1.84 |
| Attention kernel呼叫 | 2.781 | 1.78 |
| 原有explicit synchronize等待 | 2.683 | 1.72 |
| MLP呼叫 | 2.058 | 1.32 |
| RoPE呼叫 | 1.994 | 1.28 |
| 新token GPU buffer copy呼叫 | 1.985 | 1.27 |
| Model/decoder/layer/runner控制 | 1.554 | 1.00 |
| Residual add呼叫 | 0.596 | 0.38 |
| Embedding呼叫 | 0.072 | 0.05 |
| LM head呼叫 | 0.024 | 0.02 |
| 合計（未rounded） | 155.879 | 100.00 |

具體界線：
- Selection scope是prepare_layer_selection整個呼叫，排除子scope cpu()，包含其後CPU ragged
  unpack/cache寫入；比舊twilight_selector_wall_seconds終點更晚，不能直接當舊42ms同定義。
- KV控制為update_layer_gqa_group_ragged剩餘區間：length/bookkeeping、selected positions讀取、
  buffer slices與segments lists建立、完成event與metrics、回傳組裝等，已排除union、gather、
  row offsets、tensor metadata、copy、schedule_d2h及明確sync子scope；不是11.8ms未知GPUkernel。
- Attention控制為self_attn.forward剩餘區間：head迴圈、weight/tensor views、reshape/transpose、
  query組裝、cu_seqlens等與回傳準備；已排除具名projections、RoPE、Selection、cache、Attention。
- Model/decoder/layer/runner四類剩餘host scopes全部保留，原始未named最外Other_control僅0.268ms。
- D2H兩子項：copy呼叫2.021ms、其餘_schedule_d2h9.959ms，並非純傳輸12ms。
- 以上控制類仍是多操作bundle，不聲稱已拆到每個Python語句、driver API或每個GPU kernel。

## GPU phase lane（不同process，不能加入上表）

每個scope首尾在原stream記Events，無phase sync。全模型小呼叫Events會擾動dispatch；
下列為184.287ms diagnostic下的interval，不是141ms正常執行的精確kernel busy time。

| GPU phase | ms/token |
|---|---:|
| Selection整個呼叫interval（含indices子區間） | 43.869 |
| Selected KV H2D | 39.400 |
| MLP | 11.476 |
| Q/K/V/O projections | 11.393 |
| 新token GPU buffer copy | 3.858 |
| Attention | 3.478 |
| Norm | 2.898 |
| GPU metadata | 2.576 |
| RoPE | 2.248 |
| LM head | 2.197 |
| Residual | 0.349 |
| Embedding | 0.123 |
| 新KV D2H（eviction stream，另列） | 3.428 |

Selection內indices cpu()對應device interval5.164ms，非純DMA，不能再加在43.869ms上。
Main stream從第一到最後Event span182.544ms；扣除nested interval後phase分區閉合。
未覆蓋phase gap58.678ms不是額外GPU算術，亦未證明為GPU idle。
用CPU/GPU anchor對齊後，其時間對應CPU union10.307、gather9.686、D2H copy發起7.932、
D2H其餘排程6.863、KV控制6.726、projection發起4.618、Attention控制3.339ms等。
完整各項在summary.json；anchor最大不確定度0.047ms。時間對應非因果歸因，含instrumentation。
這回答大gap當下host在做什麼，但不假裝已有CUPTI逐kernel trace。

## 低擾動baseline的Selection粗項

Baseline diagnostic144.558ms：Selection wall41.996ms；Quest3.326、INT4 preparation＋QK16.561、
Top-p6.660ms（CUDA）；indices blocking wall22.998ms。FP32 QK/Softmax本輪未再獨立拆。
其他baseline粗項：gather36.972、union9.972、append/D2H13.732ms wall；H2D38.066、Attention3.686ms
CUDA。它們非互斥，不能用差值估Other，亦不拿來填入CPU-exclusive表。

## 驗證與限制

10對diagnostic selected/logits hashes、8checkpoint tensors、逐token B0/union/H2D/D2H exact；
兩組各5 CPU partitions無缺口、無重疊、逐token完整閉合；GPU main partition閉合。
前次manifest四個source hashes仍相同；runner全exit0、status=ok。只單request，未跑accuracy。
verification-before-completion gate依新分析assertions通過後才報結果。

完整性指host wall沒有遺失區間，不等於所有kernel/driver細項可完全獨立歸因。
GPU密集Events擾動27.48%，只作結構診斷；CPU表亦有7.83%觀察擾動，不縮放成正常141ms的100%。
本輪支持下一個候選包含CPU union與控制/發起鏈，不只gather和QK；未驗證任何新優化收益。

## Artifacts

- scripts/profile_twilight_complete_20260914.py
- scripts/analyze_twilight_complete_20260914.py
- results/twilight_complete_20260914/summary.json
- results/twilight_complete_20260914/baseline.json、baseline.log（完整命令）
- results/twilight_complete_20260914/{cpu,gpu}/：raw、command、intervals.json、timeline.chrome.json
- results/twilight_complete_20260914_cpu.log、twilight_complete_20260914_gpu.log
