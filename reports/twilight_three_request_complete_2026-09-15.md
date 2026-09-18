# Twilight 32K p=.90：歷史簡報、舊路徑重建與最新配置三題完整比較

## 證據口徑

09/09老師看到的baseline固定為273.84ms；不使用245.06ms。09/15補測09/14最新配置，未改演算法。
三題完全沿用niah_multikey_3 i011、vt i002、qa_1 i011。每case 32 fixed decode，正常TPOT取D2–D32共31tokens；另5個diagnostic tokens，不重跑accuracy。
每版本/題分開跑coarse、Selection detail、CPU互斥scopes、GPU Events，共24成功cases。第二題反轉版本順序；非多trial穩態評估。
**未找到舊完整source snapshot。舊路徑重建是目前保留legacy分支＋關閉後續優化，不等同09/09原始commit或當時trace。** 歷史缺項仍為N/A；重建值另欄。
歷史artifacts沒有selected hashes，不能宣稱历史selected集合逐元素驗證。可驗證歷史4個logits checkpoints、B0/union數量、H2D/D2H；新重建/最新與各instrumentation間另比完整hashes。
CPU用同perf_counter互斥分區；GPU用原stream Events另列。沒有逐phase新增同步，只有diagnostic入口anchor與既有token結尾同步。CPU純工作、driver發起與隱式等待不能只靠此timer完全分離。

## 正常TPOT與profiler擾動

| 模式 | 重建舊路徑正常TPOT | 最新正常TPOT | 重建舊診斷wall | 最新診斷wall |
|---|---:|---:|---:|---:|
| coarse | 276.773 | 140.691 | 272.791 | 143.173 |
| detail | 273.766 | 139.509 | 276.024 | 144.574 |
| cpu | 273.470 | 140.965 | 281.627 | 155.344 |
| gpu | 268.140 | 136.716 | 305.458 | 191.198 |

單位ms/token。正常數字在新增profiling啟動前取得；CPU/GPU wrappers的inactive dispatch仍可能影響正常階段，因此主結果只用原runner coarse。
不要把detail/cpu/gpu診斷分布縮放成coarse TPOT，也不把cross-process差額全當精確timer overhead。

| request | 09/09原始TPOT | 重建舊正常TPOT | 最新正常TPOT |
|---|---:|---:|---:|
| 001_niah_multikey_3_i011 | 257.184 | 280.901 | 143.162 |
| 002_vt_i002 | 295.694 | 271.195 | 139.390 |
| 003_qa_1_i011 | 268.640 | 278.222 | 139.521 |
- 重建舊 detail診斷相對自己的coarse診斷：+1.19%。
- 重建舊 cpu診斷相對自己的coarse診斷：+3.24%。
- 重建舊 gpu診斷相對自己的coarse診斷：+11.98%。
- 最新 detail診斷相對自己的coarse診斷：+0.98%。
- 最新 cpu診斷相對自己的coarse診斷：+8.50%。
- 最新 gpu診斷相對自己的coarse診斷：+33.54%。

## 粗分項：保留老師看到的baseline

| 項目 | 09/09簡報 | 重建舊路徑 | 最新三題 |
|---|---:|---:|---:|
| 正常TPOT | 273.84 | 276.773 | 140.691 |
| Selection wall | 95.440 | 106.680 | 41.611 |
| CPU union | 34.522 | 34.736 | 9.863 |
| CPU gather/packing | 40.111 | 35.073 | 37.141 |
| H2D CUDA | 36.900 | 36.725 | 37.275 |
| Attention CUDA | 3.190 | 3.105 | 3.570 |
| indices blocking wall | 75.852 | 85.845 | 22.792 |

舊CPU資料處理74.633ms=union34.522＋gather40.111。上述CPU wall/CUDA intervals有重疊，不能相加為TPOT。
新direct layout比重建舊每token多112KiB預留new-token slots，不是改Top-p；B0/union集合及logits新配對exact。

## 完整互斥CPU wall：所有時間都有所屬scope

歷史09/09未保存此表，全部歷史細項N/A。以下為補測兩版本，非原273.84ms的精確分拆。
| CPU主執行緒區間 | 重建舊ms | 重建舊% | 最新ms | 最新% |
|---|---:|---:|---:|---:|
| CPU gather index_select | 33.108 | 11.76 | 37.061 | 23.86 |
| final indices cpu()/等待 | 85.360 | 30.31 | 21.752 | 14.00 |
| Selection GPU工作發起/其餘host | 22.018 | 7.82 | 20.680 | 13.31 |
| KV其餘bookkeeping/buffer/list/event控制 | 10.690 | 3.80 | 10.274 | 6.61 |
| CPU union | 33.599 | 11.93 | 10.106 | 6.51 |
| 新KV D2H排程其餘工作 | 8.695 | 3.09 | 9.226 | 5.94 |
| Attention其餘tensor/weight views/組裝控制 | 7.523 | 2.67 | 8.235 | 5.30 |
| Q/K/V/O projection呼叫 | 6.908 | 2.45 | 6.489 | 4.18 |
| Selection後CPU ragged unpack/cache寫入 | 4.428 | 1.57 | 4.266 | 2.75 |
| H2D發起 | 1.251 | 0.44 | 3.953 | 2.54 |
| Norm呼叫 | 2.981 | 1.06 | 3.086 | 1.99 |
| 原有explicit同步等待 | 2.685 | 0.95 | 2.837 | 1.83 |
| Attention呼叫 | 2.718 | 0.96 | 2.751 | 1.77 |
| RoPE呼叫 | 11.974 | 4.25 | 2.064 | 1.33 |
| GPU metadata tensor建立呼叫 | 38.440 | 13.65 | 2.047 | 1.32 |
| MLP呼叫 | 1.821 | 0.65 | 1.920 | 1.24 |
| CPU positions concat（含sink/recent） | 2.657 | 0.94 | 1.881 | 1.21 |
| 新token GPU copy呼叫 | 0.000 | 0.00 | 1.806 | 1.16 |
| 新KV D2H copy呼叫 | 1.718 | 0.61 | 1.762 | 1.13 |
| layer層級控制 | 0.837 | 0.30 | 0.867 | 0.56 |
| CPU row offsets | 0.000 | 0.00 | 0.720 | 0.46 |
| Residual add呼叫 | 0.604 | 0.21 | 0.570 | 0.37 |
| runner邊界/其餘控制 | 0.301 | 0.11 | 0.331 | 0.21 |
| decoder層級控制 | 0.267 | 0.09 | 0.265 | 0.17 |
| CPU metadata tensor建立 | 0.283 | 0.10 | 0.234 | 0.15 |
| 模型輸出等控制 | 0.069 | 0.02 | 0.072 | 0.05 |
| Embedding呼叫 | 0.065 | 0.02 | 0.067 | 0.04 |
| LM head呼叫 | 0.018 | 0.01 | 0.021 | 0.01 |
| GPU history+new-token cat呼叫 | 0.608 | 0.22 | 0.000 | 0.00 |
| 合計（未rounded） | 281.627 | 100 | 155.344 | 100 |

計時範圍从原diagnostic_started到原diagnostic_wall_seconds；scope最內層優先，每一段只計一次。數值含instrumentation。
KV控制包含length/preview/advance、讀取cached positions、buffer slices/segments、完成event、metrics、回傳；已排除表中具名子scope。
Attention控制包含head迴圈、weight/tensor views、reshape/transpose、query/metadata組裝等，已排除projection/RoPE/cache/Attention。
raw label CPU_union_concat涵蓋KV scope內所有CPU torch.cat，包括_selected_positions的sink/selected/recent串接（:740）與舊union前cat（:995），不是僅union內成本。
此表是經過時間，不是CPU純算術；例如MLP呼叫返回後GPU仍可能在工作，其完成等待會出現在其他scope。

## GPU模型與傳輸分項

歷史09/09完整GPU細項N/A。以下屬GPU診斷run；Events含可能host enqueue gaps，不是純kernel busy time，也不能加到CPU表。
| GPU phase | 重建舊ms | 最新ms |
|---|---:|---:|
| Attention呼叫 | 3.429 | 3.801 |
| Embedding呼叫 | 0.111 | 0.108 |
| GPU history+new-token cat呼叫 | 4.043 | 0.000 |
| GPU metadata tensor建立呼叫 | 1.104 | 2.256 |
| H2D發起 | 36.915 | 38.324 |
| final indices cpu()/等待 | 5.123 | 5.192 |
| LM head呼叫 | 2.051 | 2.049 |
| MLP呼叫 | 11.365 | 11.478 |
| 新KV D2H copy呼叫 | 3.652 | 3.627 |
| 新token GPU copy呼叫 | 0.000 | 3.388 |
| Norm呼叫 | 2.583 | 2.636 |
| Q/K/V/O projection呼叫 | 13.263 | 11.417 |
| Residual add呼叫 | 0.595 | 0.348 |
| RoPE呼叫 | 12.934 | 2.366 |
| Selection GPU工作發起/其餘host | 106.043 | 46.949 |

Selection interval含indices子interval，不重複加；D2H在eviction stream另列。

| Main-stream partition | 重建舊ms | 最新ms |
|---|---:|---:|
| Attention呼叫 | 3.429 | 3.801 |
| Embedding呼叫 | 0.111 | 0.108 |
| GPU history+new-token cat呼叫 | 4.043 | 0.000 |
| GPU metadata tensor建立呼叫 | 1.104 | 2.256 |
| H2D發起 | 36.915 | 38.324 |
| final indices cpu()/等待 | 5.123 | 5.192 |
| LM head呼叫 | 2.051 | 2.049 |
| MLP呼叫 | 11.365 | 11.478 |
| 新token GPU copy呼叫 | 0.000 | 3.388 |
| Norm呼叫 | 2.583 | 2.636 |
| Q/K/V/O projection呼叫 | 13.263 | 11.417 |
| Residual add呼叫 | 0.595 | 0.348 |
| RoPE呼叫 | 12.934 | 2.366 |
| Selection GPU工作發起/其餘host | 100.920 | 41.757 |
| Uncovered_phase_gap | 109.430 | 64.283 |

Main partition使用最內層interval排除nested雙計；phase gap非已證實GPU idle，下面只列時間上host正在做什麼，非因果歸因。

| Phase gap對應CPU活動 | 重建舊ms | 最新ms |
|---|---:|---:|
| CPU gather index_select | 33.185 | 12.533 |
| CPU union | 33.697 | 10.186 |
| 新KV D2H copy呼叫 | 8.204 | 8.053 |
| 新KV D2H排程其餘工作 | 7.341 | 7.264 |
| KV其餘bookkeeping/buffer/list/event控制 | 8.577 | 6.010 |
| Q/K/V/O projection呼叫 | 6.070 | 4.980 |
| Attention其餘tensor/weight views/組裝控制 | 3.155 | 3.697 |
| H2D發起 | 0.865 | 2.784 |
| CPU positions concat（含sink/recent） | 2.938 | 1.920 |
| 新token GPU copy呼叫 | 0.000 | 1.415 |
| GPU metadata tensor建立呼叫 | 0.050 | 0.882 |
| Selection GPU工作發起/其餘host | 0.471 | 0.804 |
| Attention呼叫 | 0.687 | 0.787 |
| CPU row offsets | 0.000 | 0.686 |
| RoPE呼叫 | 2.725 | 0.659 |
| Norm呼叫 | 0.460 | 0.507 |
| MLP呼叫 | 0.417 | 0.444 |
| 原有explicit同步等待 | 0.323 | 0.280 |
| CPU metadata tensor建立 | 0.000 | 0.237 |
| Residual add呼叫 | 0.112 | 0.073 |
| layer層級控制 | 0.125 | 0.064 |
| decoder層級控制 | 0.022 | 0.019 |
| Selection後CPU ragged unpack/cache寫入 | 0.005 | 0.000 |
| 模型輸出等控制 | 0.000 | 0.000 |
| runner邊界/其餘控制 | 0.000 | 0.000 |
| LM head呼叫 | 0.000 | 0.000 |

## Selection完整細分：資料準備與研究候選分開

本表使用獨立detail run，CUDA Events無逐phase同步；除最後兩列wall外都是device intervals，不能混加。歷史09/09未保存純QK/Softmax細分，舊欄為重建測量。

| 步驟 | 分類 | 重建舊ms | 最新ms |
|---|---|---:|---:|
| Query stack/cast | 資料準備 | 1.023 | 0.497 |
| Quest metadata/page score | 第一輪重要性估計 | 9.435 | 2.962 |
| Quest page Top-k | 第一輪選擇 | 0.439 | 0.445 |
| B0 page→token展開 | indices準備 | 0.327 | 0.520 |
| INT4 metadata stack/reuse | 資料準備 | 4.067 | 0.212 |
| Candidate packed K/scale/min gather | 資料準備 | 3.811 | 融合於另一列／不走此路徑 |
| INT4 unpack/FP32 codes | 資料準備 | 22.353 | 融合於另一列／不走此路徑 |
| Affine dequant/FP32 K materialization | 資料準備 | 38.767 | 融合於另一列／不走此路徑 |
| Fused candidate preparation | 資料準備 | 融合於另一列／不走此路徑 | 9.311 |
| FP32 QK matmul | 研究候選：score計算 | 7.061 | 7.045 |
| 1/sqrt(head_dim) scaling | 研究候選：score scaling | 0.112 | 0.138 |
| argsort | 研究候選：排序選擇 | 3.235 | 3.231 |
| sorted logits gather | 資料準備 | 0.212 | 0.214 |
| FP32 Softmax | 研究候選：weight計算 | 0.150 | 0.150 |
| cumsum＋stack | 研究候選：累積機率 | 2.917 | 2.910 |
| searchsorted/threshold/count | 研究候選：Top-p判定 | 0.336 | 0.334 |
| final B1 gather/mask/sort/cat | indices準備 | 5.684 | 5.709 |
| Selection wall | CPU wall，與上方device重疊 | 108.157 | 44.196 |
| final indices blocking wall | CPU wall，與上方device重疊 | 82.103 | 20.747 |

INT4 preparation只加metadata與gather/unpack/dequant/materialization各互斥子段，不含QK/scaling。新版fused prepare無法再可靠拆成獨立gather/dequant時間，不硬拆。
FP32 QK timer只包torch.matmul及相鄰unsqueeze/squeeze view，不含candidate準備/scaling/Softmax/Top-p。
Softmax輸入輸出FP32 [8,3,8192]；QK為[8,3,8192,128] × [8,3,128,1]，FP32 K為96MiB/layer暫存。
Final indices blocking在GPU工作排隊後呼叫.cpu()，是等待＋傳輸，不是純DMA；CPU ragged unpack另見CPU表。

## 研究候選佔比

下列比例是device phase時間除以同detail run完整diagnostic wall，僅描述時間尺度，不是互斥TPOT占比、Amdahl上限或可保證節省量。

| 研究範圍 | 重建舊ms | 重建舊比值 | 最新ms | 最新比值 |
|---|---:|---:|---:|---:|
| 純QK | 7.061 | 2.56% | 7.045 | 4.87% |
| QK＋scaling＋Softmax | 7.323 | 2.65% | 7.333 | 5.07% |
| 上述＋argsort/cumsum/Top-p threshold | 13.812 | 5.00% | 13.808 | 9.55% |

INT4 preparation本身：重建舊 68.997ms，最新 9.523ms；不列入上述研究候選算術。
若只改算術而selection集合不變，不可期待gather/H2D一起減少；若新方法改變選中數量/品質，須另外做quality與end-to-end ablation。

## 驗證、限制與artifacts

{'pair_checks': 27, 'status': 'PASS', 'historical_selected_hashes': 'unavailable', 'new_pairs_hashes': 'exact', 'production_source_hashes': 'unchanged', 'cpu_partitions': 'all contiguous and closed'}
正常coarse每題只跑一輪，本報不是完整accuracy或多trial穩態評估。失敗的CLI backend名稱與diagnostic字串建構smoke已修復，未改production；manifest保留失敗attempt。
歷史原始缺少的完整timers不能被重建值覆寫；273.84ms僅當會議基準，不能用新分區硬拆。
本輪沿用verification-before-completion gate；新配對hash/checkpoint、count/bytes、CPU閉合与source hashes通過才輸出報告。
- results/twilight_three_requests_20260915/manifest.json：命令、source hashes、return codes。
- 同目錄 summary.json、component_statistics.csv：mean/median/min/max。
- 各request/version/mode/result.json、run.log、command.json；CPU/GPU另含intervals.json及timeline.chrome.json。
- scripts/run_twilight_three_requests_20260915.py、profile_twilight_three_requests_20260915.py、analyze_twilight_three_requests_20260915.py。
