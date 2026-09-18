# 保留非同步的互斥CPU wall accounting

32K qa_1 i011、p=.90，最新early metadata/RoPE-only/triton_prepare/bitmap union/flat gather/direct
layout配置不變。profile與baseline各32 fixed decode+5 diagnostics，無accuracy/grid。

## 方法

Process-local wrappers以scope stack將CPU時間歸到當下最內層分類。切換scope時保存相鄰
perf_counter區間，沒有逐phase同步；既有Event/Stream/Device同步只包timer，不新增等待。
以runner原diagnostic_started與diagnostic_wall_seconds作完整邊界，透過記憶體內兩個callback
插入點，不修改正式runner檔案。GPU anchor只在diagnostic入口、計時外同步，token中不新增同步。
GPU僅量Selection/H2D/Attention/MLP phase Events，為獨立lane，不加進CPU表。

## 結果

- 正常TPOT profile174.02、baseline174.49ms（D2–D32），profiling未啟用。
- Diagnostic profile175.91、baseline173.33ms，觀察增幅2.58ms/1.49%，單對process含變異，
  非純timer overhead。明顯不同於上輪serialized232.70ms，不以跨run差額精確分離同步成本。
- 5對indices/logits hashes、4checkpoint tensors exact，逐step B0/union/H2D/D2H相同。
- 每個token驗證segments相鄰、無重疊/缺口、非負，總計等於runner diagnostic wall。

| CPU互斥分類 | Mean ms | % |
|---|---:|---:|
| indices cpu() blocking | 51.66 | 29.37 |
| CPU gather operator | 32.63 | 18.55 |
| Selection其餘host呼叫 | 25.82 | 14.68 |
| KV其餘控制/整理 | 13.39 | 7.61 |
| CPU union | 10.29 | 5.85 |
| 其他控制區間 | 9.13 | 5.19 |
| 新KV D2H發起（扣除已分類同步） | 8.31 | 4.72 |
| QKVO呼叫 | 6.33 | 3.60 |
| 既有顯式sync等待 | 3.31 | 1.88 |
| Attention呼叫 | 3.15 | 1.79 |
| Norm呼叫 | 3.08 | 1.75 |
| MLP呼叫 | 2.60 | 1.48 |
| GPU metadata呼叫 | 2.01 | 1.14 |
| RoPE呼叫 | 1.99 | 1.13 |
| H2D發起 | 1.89 | 1.07 |
| CPU metadata呼叫 | 0.23 | 0.13 |
| Embedding呼叫 | 0.08 | 0.04 |
| LM head呼叫 | 0.02 | 0.01 |
| Total（未rounded） | 175.91 | 100 |

Selection host25.82不是純Python或純launch，仍可能含隱式等待及instrumentation，且包含
cpu()後ragged unpack；不等同舊selector wall邊界。CPU gather只包index_select operator，
其餘row-index/packing整理在KV控制；不能直接拿舊gather bundle相減當優化。
同步等待3.31不含indices cpu()（另列51.66），也不涵蓋其他API的隱式同步。

GPU獨立phase intervals mean：H2D37.75、MLP10.78、Attention1.88、Selection55.07ms。
Selection GPU event包整個prepare_layer_selection呼叫，含copy/host return gap，不是純selector
kernels；trace中的Selection_host_excluding_cpu_wait名稱在GPU lane僅為相同wrapper標籤，
GPU interval沒有扣除cpu()。GPU lane只有上述四類，未量全部projections/Norm/RoPE/kernel，
不能把未覆蓋空間稱GPU idle。Anchor最大不確定度41.67us。

## 結論與下一決策

在保留非同步排程下取得完整互斥host分佈，沒有強制同步造成的232ms總時間。CPU主要區間
為等indices、gather與Selection host呼叫，但51.66ms不能稱可直接移除的transfer成本。
H2D CPU發起1.89ms與GPU interval37.75ms說明host表不能代表GPU算術/搬運成本。
下一步若優化仍需對準相依鏈、做單變因TPOT；不依host百分比直接估GPU kernel收益。

Artifacts: results/twilight_async_wall_v1/{profile,baseline,intervals,summary}.json、
timeline.chrome.json與commands/checkpoints；scripts/profile_twilight_async_wall_v1.py、
scripts/analyze_twilight_async_wall_v1.py。正式source與既有performance表未修改。
