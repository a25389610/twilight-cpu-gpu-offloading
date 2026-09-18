# 最新4-chunk CPU gather/packing診斷

## 問題與方法

使用者要求檢查約60ms CPU gather/packing可如何以純工程方式改善。本輪只新增process-local
診斷script，未修改production、selection規則或buffer配置。依systematic-debugging先定位成本。
使用最新fused Quest、4-chunk pipeline、metadata reuse，32K qa_1 i011、p=.90。
原始runner baseline與profile各32decode+5coarse diagnostics，未加逐phase CUDA同步。

Profile僅在metrics enabled後計時CPU BF16 index_select與CPU int64 torch.add，第一個diagnostic
token capture 112個K chunk的indices clones（V使用相同indices）；模型完成後以numpy計算runs。
原H2D仍非同步。Calls以perf_counter wall為主，另列process_time，兩者不可相加。

## 實測

| 項目 | ms/token |
|---|---:|
| Baseline正常TPOT | 202.66 |
| Profile開啟diagnostic前TPOT | 202.41 |
| Baseline diagnostic wall | 200.61 |
| Profile diagnostic wall | 204.11 |
| Baseline gather/packing | 59.46 |
| Profile gather/packing | 62.74 |
| Profile index_select calls | 55.17 |
| Profile row offset torch.add | 1.27 |
| Parent wall其餘（同clock相減） | 6.29 |

Index_select佔profile parent87.94%，offset約2.03%。其餘6.29ms包括views、Python control、
scalar writes、slice建立與本輪wrappers/capture；**不是6.29ms皆可移除**。
Capture clone平均攤至5tokens約0.326ms，包含在其餘內。不同diagnostic wall差3.51ms/1.75%
混合process變異與instrumentation，不稱純overhead。

每token224個index_select（28layers×4chunks×K/V）及224個offset adds。
平均payload528.83MB/token；index_select process CPU時間330.62ms/token，為多threads CPU
時間總和，不可與wall相加，也不能判斷全是DRAM搬運/飽和或全部是thread overhead。
普通TPOT202ms不當成新優化成果，與歷史206/171/144ms漂移仍不可跨run歸因。

## 實際連續區段

第一個diagnostic token、112 chunks，共1,041,433 rows（K/V共享同一組row indices）。
355,733個runs，mean2.93 tokens、median1、max255。
Chunk邊界會切斷run；包含每group新token預留gap row，不能外推所有steps/tasks。

| Run至少包含tokens | 此類run涵蓋的rows比例 | Run數 |
|---:|---:|---:|
| 4 | 59.25% | 70,510 |
| 8 | 35.70% | 21,535 |
| 16 | 20.05% | 5,602 |
| 32 | 12.01% | 1,558 |
| 64 | 8.24% | 615 |
| 128 | 5.68% | 239 |

這是payload coverage，不是預期節省時間。即使整段copy仍須讀寫相同bytes。
不得用Python為35萬runs各發起一次copy；也不得搬整頁未選中的tokens進Attention。

## Source audit與建議

1. `twilight_offload_cache.py`目前即時index_select的out直接是persistent pinned flat buffer，
   gather/packing同一次寫入，沒有第二次完整CPU temporary copy。既有direct layout已跳過
   GPU整份history cat。不能再把這些当未做優化。
2. flat gather分支仍無条件建立每group host_key/host_value views（每token448次helper calls），
   但該分支改用_host_key_rows/_host_value_rows，不讀這兩個views。可移至non-flat分支；
   這是確認存在的冗餘，未獨立量時間，不承諾省數十ms。
3. 低風險候選：移除上述unused views及檢查重複slice/control，維持資料流與copy完成event。
4. 較值得最小copy消融：compiled hybrid run-copy（例如threshold8/16）＋零散row gather。
   同時處理K/V，保留4段pipeline及現有pinned輸出。Run偵測自身成本必須算入，保留output bit-exact。
   只改copy策略，不能改union/selected tokens或固定整頁copy引入多餘Attention tokens。
5. 先前簡單OpenMP native逐row copy沒有穩定收益；降低threads亦未改善gather。因此不能
   把換C++/加減threads當已知解法。本次continuity訊號不支持承諾gather減半。
6. 若hybrid copy無實測收益，應先查同一working set下CPU memory/copy及排程條件，而不是
   再建第二套完整KV layout或以跨token cache策略冒充單純程式優化。

## 驗證與下一決策

兩case 5對diagnostic selected/logits hashes、4checkpoints exact，B0/union/H2D/D2H一致；
三份cache/kernel source hashes前後不變；1120 index_select與1120 offset adds符合預期。
只診斷，尚未實作hybrid copy。下一步若授權，先unused views單變因清理，再獨立run-copy
microbenchmark/paired TPOT gate，不把局部wall下降等同TPOT下降。

Artifacts：`results/twilight_gather_chunks_audit_v1/{manifest,analysis,baseline,profile}.json`、logs、
checkpoint tensors；`scripts/diagnose_twilight_gather_chunks_v1.py`。未重跑accuracy grid。
