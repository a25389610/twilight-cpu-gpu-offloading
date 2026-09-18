# CPU gather → H2D 分批pipeline pilot

## 問題、假設與實作

CPU整層gather完成才發起整層H2D形成串接成本。假設將flat pinned buffer分成2/4段，
每段gather後立即在原current stream發起H2D，能讓CPU下一段gather與前段DMA重疊。
不改Quest/B0/Top-p/GQA union、KV內容、順序或Attention算法。新增opt-in
`--twilight-gather-h2d-chunks N`，預設0；本pilot要求flat gather+direct layout+GQA，
不與native gather混用。未新增stream或逐chunk sync。

Pinned source與GPU destination均為原持久buffer的不相交slice；CPU僅寫下一slice。
重用pinned slot前仍等待原completion event，最後一段H2D後record同一slot完成事件；
GPU新token覆寫與Attention仍排在所有copies之後。CPU source KV原host-ready等待保留，
既有D2H record_stream未更動。無新增完整KV buffer或額外persistent pinned allocation；
本輪未做新的RSS長時間trace，不能聲稱排除所有memory leak。

## Matched小規模實驗

32K qa_1 i011、p=.90，最新triton_prepare/bitmap union/flat gather/direct/RoPE-only/early metadata。
6 processes：第一輪0→2→4，第二輪4→2→0；各32 fixed decode與5 diagnostics。
主TPOT是未instrumented D2–D32，沒有accuracy/grid。

| 分段 | Trial1 mean / median ms | Trial2 mean / median ms |
|---|---:|---:|
| 0 baseline | 166.39 / 166.65 | 165.94 / 166.24 |
| 2 | 152.49 / 152.00 | 154.60 / 152.69 |
| 4 | 149.47 / 148.89 | 149.06 / 149.57 |

4段mean分別下降16.92/16.88ms，約10.17%/10.17%。只在本request驗證，不能把舊175ms
當本輪baseline，也不能稱已完成所有Context/requests優化。保留default0，不覆寫正式grid。

4段不是gather變快：gather bundle33.10→38.89ms、32.98→37.98ms；H2D Event總和
37.42→37.89ms、37.37→37.82ms。Selection wall73.42→52.14ms、72.09→51.62ms包含
排程/上游等待變化，不能稱selector算術加速。收益以正常TPOT衡量。
Gather timer在pipeline路徑扣除內嵌H2D enqueue wall（同clock），不把該呼叫重複列入gather。

## Correctness與timeline

4組variant/baseline對照共20對diagnostic indices/logits hashes、16對checkpoint tensors exact；
逐step B0/union/H2D/D2H相同，H2D仍533,213,696 bytes/token（首diagnostic）。
Twilight helper tests PASS，py_compile PASS，manifest source hashes核對不變。

另跑4段非同步timeline：普通TPOT149.73ms、instrumented diagnostic161.64ms；不可將
此diagnostic當正常performance或把差額作純overhead（token位置不同）。5steps觀察CPU gather
operator37.40ms與H2D phase38.85ms的交集約26.54ms/token，支持確有phase重疊。
Anchor最大不確定度50.68us，CUDA Event interval可能含host發起空檔；這不是純DMA/DRAM
硬體trace。Timeline也通過5對hashes、4checkpoints與bytes/counts驗證。
交集26.54ms不等同TPOT可省26.54ms，分段額外operator/launch與其他排程成本存在。

## 結論與下一決策

此單request pilot有效：同一演算法與payload，4段pipeline約149ms，比matched baseline約166ms
快10.2%。優先採4段作後續驗證候選，不再宣稱native gather有效。
下一步固定3 requests/其他context的matched gate，確認chunk大小通用性與持續buffer安全；
尚未提升全域預設、未做大規模accuracy。若其他場合無收益，可直接設0回原排程。

Source: source/headinfer/headinfer/twilight_offload_cache.py，runner新旗標。
Artifacts: results/twilight_chunk_pipeline_v1/（manifest/commands/source hashes/raw/checkpoints/summary）
與results/twilight_chunk_pipeline_timeline_v1/（intervals/summary/timeline.chrome.json）。
Scripts: run_twilight_chunk_pipeline_v1.py、analyze_twilight_chunk_pipeline_v1.py、
profile_twilight_chunk_pipeline_v1.py、analyze_twilight_chunk_timeline_v1.py。
