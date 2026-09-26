# 最新 Twilight + previous-token resident cache：TPOT、Selection 與 VRAM breakdown

日期：2026-09-26（Asia/Taipei）  
範圍：目前最快的 **GPU bitmap hit/miss mapping + mapped CPU pinned KV read** opt-in 路徑。完整 historical KV 留在 CPU；Full GPU KV 只作容量對照，不作 TPOT baseline。

## 問題、方法與量測口徑

目的是判定目前 Selection 是否已成值得改進的 exposed bottleneck，並分辨 resident reuse 的時間和 VRAM 代價。固定 Llama-3.2-3B-Instruct、BF16、FlashAttention 2、RTX 5060 Ti 16 GB、Batch 1、三題 32K（001_niah_multikey_3_i011、002_vt_i002、003_qa_1_i011）、Twilight-GQA dynamic Top-p `p=.90`、B0=8192、32 fixed Decode steps。正式 TPOT 為同步 CPU wall 的 D2–D32 共 31 token；D1 warm-up，Prefill/model load 不計。

正式 run 沒有同步 component profiling 或 Selection/Attention trace。最後 source 快照每題三輪（rep3–5），另保留在加入診斷探針之前的 rep1–2。Normal-overlap timeline 是每題 **D33–D35 三個 post-timing diagnostic token**，同一 token origin 記錄 CUDA Event 與 host spans；互斥表用 interval ownership 將重疊區間只歸給一類，未覆蓋時間列入 runtime/control residual。這是**診斷執行的 exposed-time 分類**，不等於 kernel-only active time 或某機制移除後可保證省下的 TPOT。詳細 Selection 的 CUDA Event 是另一組 D33–D35 diagnostic run；parent/child 和 CPU/GPU scope 不相加。VRAM inventory 在 D32 後取 unique backing storage，Selection 暫存另在 D33 取樣。

## 1. Fresh formal TPOT

| Request | rep3 | rep4 | rep5 | 平均 ms/token |
|---|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 74.802 | 74.900 | 74.544 | **74.748** |
| 002_vt_i002 | 74.196 | 74.683 | 74.752 | **74.544** |
| 003_qa_1_i011 | 74.612 | 74.794 | 73.977 | **74.461** |
| 三題平均 | | | | **74.584** |

rep1–2（診斷探針加入前、同一 execution 設定）的三題平均是 **73.038 ms/token**；先前 matched 新路徑均值 **73.787 ms/token**。最後快照三輪為 **74.584 ms/token**，比先前值高約 1.08%；造成這個跨輪差異的環境／source instrumentation 影響尚未分離，故不挑最低輪作「目前速度」。各輪 checkpoint/final logits SHA 一致。

## 2. 同一 token origin 的互斥 TPOT 診斷分布

Normal-overlap diagnostic wall 平均 **74.500 ms/token**，與最後快照 formal **74.584** 接近（-0.084 ms，-0.11%），但每題仍有差異，**下表百分比只以 74.500 diagnostic wall 為分母**。GPU Event interval 可包含 host submission 造成的 queue gap；host 控制與 GPU 活動交疊時，interval ownership 只給一類。`runtime/control residual` 也包含未標記 GPU 工作、排程空隙與 clock alignment 誤差。

| 互斥 exposed-time 類別 | ms/token | diagnostic wall % | 邊界 |
|---|---:|---:|---|
| KV Selection：query prep + Quest 第一輪 + Twilight 第二輪 | **26.721** | **35.87%** | Selection core CUDA Event 起點至 membership decision 終點 |
| Selection 後整理：GPU GQA union + 8 組長度 handoff | 2.083 | 2.80% | union 1.150；GPU reduction/D2H handoff 0.933 |
| Previous-token cache mapping、snapshot、state、新 token slot、control | 8.480 | 11.38% | 不含下一列的 fused K/V assembly |
| Resident hit + mapped CPU miss read + Attention-layout assembly | 3.877 | 5.20% | **同一 CUDA kernel**；不能把 bus read 與 hit copy 硬拆 |
| Attention | 2.606 | 3.50% | FlashAttention varlen call interval |
| 其他 model compute | **23.703** | **31.82%** | QKV/RoPE prep、O projection、MLP、Norm、embedding、lm_head |
| new-KV writeback | 1.436 | 1.93% | GPU staging／D2H 的 exposed ownership |
| 其他 runtime/control residual | 5.593 | 7.51% | 其餘 token wall；不能全稱 CPU overhead |
| **diagnostic wall** | **74.500** | **100%** | 每題 D33–D35，共 9 diagnostic tokens |

Selection exposed 26.721 ms 與其他 model compute 23.703 ms 是本輪最大的兩類；cache 相關（mapping／snapshot／state／fused assembly／control）合計 **12.357 ms、16.59%**。這些是 diagnostic interval 的分類，不能當成可加的 TPOT 改善預測。原始互斥 token rows 與分類規則見 `results/twilight_latest_breakdown_20260926_v1/analysis/{timeline_per_token.csv,tpot_breakdown_diagnostic.csv}`。

## 3. Selection：真正的演算法階段與詳細子項

Normal-overlap **同輪同 origin** 的 Selection 26.721 ms 可分成：query prep **0.667 ms（Selection 2.49%；diagnostic wall 0.89%）**、Quest 第一輪 **3.536 ms（13.23%；4.75%）**、Twilight 第二輪 **22.518 ms（84.27%；30.23%）**。若只將這些 diagnostic ms 除以 fresh formal 74.584 ms，Quest 為 **4.74%**、Twilight 為 **30.19%**；這是**跨 run 的參考比值**，不是 formal run 直接量到的精確占比，更不是演算法改善的保證上限。正常 overlap 的同時計時百分比應採 4.75% 與 30.23%。

以下子項來自另一組詳細 Selection diagnostic：wall **79.034 ms/token**，高於 fresh formal 4.450 ms（+5.97%）；Selection core parent **26.912 ms**。表中 `% Selection` 用 26.912 作分母，`% 全體` 只用 **79.034 diagnostic wall** 作分母。parent 與 child 重疊，不可整欄相加。

| Selection 子項（實際 timer 邊界） | ms/token | % Selection | % detailed diagnostic wall |
|---|---:|---:|---:|
| Query FP32 stack／prepare | 0.280 | 1.04 | 0.35 |
| Quest min/max stack + fused metadata score | 2.413 | 8.97 | 3.05 |
| Quest Top-K page | 0.451 | 1.68 | 0.57 |
| Quest B0 candidate index expansion | 0.344 | 1.28 | 0.43 |
| **Quest B0 parent（含上三項及間隙）** | **3.306** | **12.28** | **4.18** |
| INT4 metadata stack／reuse | 0.028 | 0.10 | 0.04 |
| Triton `triton_prepare` fused INT4 gather/unpack/dequant + FP32 estimated-K materialization | **8.720** | **32.40** | **11.03** |
| FP32 approximate QK matmul | **7.081** | **26.31** | **8.96** |
| QK scale | 0.122 | 0.46 | 0.15 |
| **INT4 preparation + QK parent** | **16.157** | **60.04** | **20.44** |
| argsort／rank | 3.221 | 11.97 | 4.08 |
| sorted logits gather | 0.216 | 0.80 | 0.27 |
| FP32 softmax | 0.149 | 0.55 | 0.19 |
| cumsum | 2.981 | 11.08 | 3.77 |
| Top-p threshold/search + desired count | 0.279 | 1.04 | 0.35 |
| **Top-p parent（含以上子項及間隙）** | **6.997** | **26.00** | **8.85** |
| 最終 membership／allocated count decision | 0.025 | 0.09 | 0.03 |
| **Selection core parent** | **26.912** | **100.00** | **34.05** |

`triton_prepare` 的 INT4 candidate gather/unpack/dequant/materialize 是融合範圍，沒有可靠的內部分項 timer；Quest metadata score timer 也同時包含 min/max stack 與 fused score。GQA union 與長度 handoff 屬 Selection **後處理**，沒有混入 Quest/Twilight 演算法兩個數字。詳細 raw 表：`analysis/{selection_stage_common_origin.csv,selection_detailed_active.csv}`。

後處理採 normal-overlap 同輪互斥 exposed time；若把 Selection core 與後處理合稱 selection pipeline，分母為 **28.804 ms/token**：

| Selection 後處理 | ms/token | selection pipeline % | diagnostic wall % |
|---|---:|---:|---:|
| GPU GQA union／bitmap | 1.150 | 3.99 | 1.54 |
| 8 組 group-length reduction／D2H handoff | 0.933 | 3.24 | 1.25 |
| **後處理合計** | **2.083** | **7.23** | **2.80** |

此處的 pipeline 分母與上表 detailed Selection core 的 26.912 ms 分母不同，不應混算。

## 4. Previous-token resident cache 成本與 reuse control

以下為 normal-overlap 的 28-layer/token CUDA Event elapsed，`CPU host span` 是提交／等待範圍，可能包含先前排入 GPU 的 Selection 等待，**不能與 CUDA 欄或 formal TPOT 相加**。

| 當前實際 component | CUDA Event ms/token | CPU host span ms/token | diagnostic wall % |
|---|---:|---:|---:|
| GPU current/previous bitmap hit/miss mapping | 3.452 | 2.454 | 4.63 |
| 8 組 group-length reduction／D2H handoff | 0.933 | 14.200 | 1.25 |
| **fused resident hit K/V + mapped CPU miss K/V + layout assembly** | **3.877** | 0.529 | **5.20** |
| current new-token slot copy | 0.978 | 2.528 | 1.31 |
| resident K/V snapshot update | 1.331 | 0.448 | 1.79 |
| position reference + resident bitmap state clone | 0.201 | 0.322 | 0.27 |
| cache parent（含上列除 group handoff，**不可再加**） | **12.357** | 10.591 | **16.59** |
| cache parent 內未分配的 control／queue gap（由互斥表推得） | 2.517 | — | 3.38 |

`14.200 ms` group-length CPU span 主要跨越等待已排入 GPU 的工作，不能解讀為獨立的 14.2 ms CPU 管理成本。fused kernel 無法把 resident hit copy、mapped miss PCIe read 和 layout write 個別拆成獨立 kernel time；其 mapped miss **logical payload** 為 45.175／49.113／46.134 MiB/token，平均 **46.807 MiB/token**，不等於實際 PCIe transaction bytes。new-KV writeback 診斷：GPU staging/schedule 1.247 ms、D2H Event 0.015 ms、CPU host wait 0.002 ms、CPU scatter 0.045 ms；有 overlap，互斥 exposed 類別為 1.436 ms。細節見 `analysis/{resident_cache_details.csv,new_kv_writeback_details.csv}`。

另做一個**不改 production source 的 default-off diagnostic control**：以相同 GPU bitmap mapping／mapped CPU read／resident snapshot 路徑，將 previous membership 改為預先配置的全零 bitmap，使 selected history 全部從 CPU 讀取。三題單次 preliminary：

| Request | no-hit control | resident hit reuse | 差值 | no-hit mapped logical MiB/token |
|---|---:|---:|---:|---:|
| 001 | 110.935 | 74.802 | -36.133 | 512.222 |
| 002 | 109.719 | 74.196 | -35.524 | 487.907 |
| 003 | 112.565 | 74.612 | -37.953 | 523.286 |
| **平均** | **111.073** | **74.536** | **-36.537（-32.89%）** | **507.805** |

三題 checkpoint logits exact，no-hit 的 D2–D32 共 868 layer rows/request 全為 0 hit。這個 ablation 顯示 reuse 對**此 mapped-read 路徑**的淨效益；control 仍做無用的 resident snapshot，所以差值不能稱為純 cache 管理成本，也只有每題一組，標為 preliminary。raw：`results/twilight_latest_breakdown_20260926_v1/nohit_control/` 和 `analysis/nohit_control_preliminary.csv`。

## 5. Full 32K GPU KV 與目前方法 VRAM 容量

實際 Llama-3.2-3B config：28 layers、8 KV heads、head_dim 128、BF16 2 bytes；32,768 tokens 的 K+V 理論容量 = `28×8×32768×128×2(K,V)×2 bytes` = **3,758,096,384 bytes = 3.500 GiB = 3,584 MiB**。獨立 process 載入相同 BF16 model 後，實際配置兩個 CUDA tensor 並寫入，兩者 unique storage 與 CUDA allocated 增量均為 **3,584 MiB**。這是容量實驗，未跑 Full GPU TPOT。

| CUDA allocator 指標 | model only／共同 weights | model + Full 32K GPU KV（容量） | 目前方法（D32，三題平均） |
|---|---:|---:|---:|
| allocated MiB | 6,128.348 | 9,712.348 | 7,628.675 |
| peak allocated MiB | 6,128.348 | 9,712.348* | 9,131.930† |
| reserved MiB | 6,174.000 | 9,758.000 | 9,933.333 |
| peak reserved MiB | 6,174.000 | 9,758.000* | 9,933.333† |
| 扣共同 model-only allocated 增量 MiB | — | **3,584.000** | **1,500.327** |

`*` Full 欄只含 model 載入 + Full K/V allocation，沒有 Decode workspace，因此不能把兩欄 peak 當成同條件 runtime peak 比較。`†` 目前方法 peak 包含 Prefill 及 Decode，三題各自 peak allocated 為 9,100.126／9,159.301／9,136.363 MiB。allocator reserved 是 caching/fragmentation，不能當 active tensor bytes；目前方法部分 request 的 reserved 甚至高於 Full 容量實驗。以**持續 allocated 增量**比較，目前方法比完整 32K GPU KV 少約 **2,083.673 MiB**，但兩者不是相同 Decode workload。

目前方法的完整 historical KV 是 **CPU pinned**，三題實際 prompt+capacity 的 slab 為 3,497.266／3,573.938／3,543.750 MiB，平均 **3,538.318 MiB**；另有 CPU pinned flat/pack/new-KV staging **224.547 MiB**。這些不算 VRAM。

## 6. 目前方法的 GPU storage 來源（D32 snapshot）

以 `untyped_storage().data_ptr()` 和 backing storage bytes 去重 view／alias；下表是三題平均的 **unique cache GPU storage**，分母 **1,486.886 MiB**，不含共同 model weights。它不等於 CUDA allocator allocated；D32 process 相對 model-only 的 allocated 增量平均 **1,500.327 MiB**，差 **13.441 MiB** 是目前無法安全歸類為 KV 的其他 live allocation／runtime。暫存 workspace 在下表之外。

| Unique GPU storage component | MiB | cache storage % |
|---|---:|---:|
| Previous-token resident K/V | **577.827** | **38.86** |
| Twilight INT4 codes | **441.784** | **29.71** |
| Twilight INT4 scale + minimum | 27.612 | 1.86 |
| Quest min + max metadata | **220.390** | **14.82** |
| Attention-layout selected K/V buffer | 99.762 | 6.71 |
| selected-history staging K/V buffer（已配置，mapped fast path 不需由此 H2D） | 99.750 | 6.71 |
| GPU pack staging | 12.469 | 0.84 |
| previous-token membership bitmaps | 6.909 | 0.46 |
| current GQA membership bitmap | 0.247 | 0.02 |
| new-token GPU K/V staging | 0.109 | 0.01 |
| 其他可追蹤 cache GPU storage | 0.026 | <0.01 |
| **合計** | **1,486.886** | **100** |

Sink/recent tokens 沒有獨立、完整 32K 的 GPU KV allocation；它們已包含於當步 selected K/V、resident K/V 和 CPU historical slab。GQA index／hit-miss mapping 中間 tensor 大多是短暫的，D32 snapshot 未將其冒充常駐 storage。Selection/Top-p 暫存也非持續 resident：D33 診斷在 Top-p 結束點仍 live 的 local GPU temporary unique storage 平均 **110.107 MiB**（主要是 FP32 estimated K 約 96 MiB）；每 layer Selection 區間的 allocator peak 相對該 layer 入口平均 **116.518 MiB**。這是順序執行的 per-layer temporary，不乘以 28，也不能與含 Prefill 的 process peak 相加。raw：`vram_probe/*/inventory.json`、`analysis/{vram_components.csv,vram_process.csv,selection_workspace_probe.csv}`。

## Correctness、限制、下一決策

- 三題 formal 各輪和 normal／detailed diagnostic 的 checkpoint/final logits SHA 相同；003 在最後 source 的 native resident 與 GPU mapped resident 另核對 D1/D2/D32 Attention K/V trace、new-KV trace、checkpoint/final logits exact。Selection 既有完整 trace gate（先前 source 快照）為 21,504 entries exact；本輪另檢查 D1/D2/D32 GPU GQA bitmap 84 layer rows（見 raw correctness artifact）。Formal fixed-token logits gate 不等同其他 Context、`p`、模型或自由生成品質全驗證。
- normal-overlap 只取各題 D33–D35；同一 token origin 的 CUDA Event/CPU wall alignment 與 interval priority 會影響互斥歸屬。尤其 mapped CPU read 融合於 kernel、CPU group-length host wait 與先前 GPU 工作交疊，不能虛構獨立 PCIe read 時間或把 active scopes 相加。
- 目前 source 的 CUDA helper 使用 `sm_120`；其他 GPU、Context、`p`、batch 及品質 cohort 未驗證。本輪只 profiling 與容量量測，沒有改 Selection 演算法或自行繼續優化。

**目前三個最大 latency 類別**：Selection 26.721 ms（35.87% diagnostic wall）、其他 model compute 23.703 ms（31.82%）、previous-token cache path 12.357 ms（16.59%，含 fused mapped assembly）。**最大三個 VRAM storage 類別**：resident K/V 577.827 MiB、Twilight INT4 codes/scale/min 469.396 MiB、Quest min/max 220.390 MiB。下一個研究決策由使用者決定；本輪完成後停止，不自行優化。

## Artifact 與 source provenance

- Formal raw：`results/twilight_latest_breakdown_20260926_v1/formal/rep{1,2,3,4,5}/`；最後 source 的主要統計採 rep3–5。`analysis/formal_tpot.csv`。
- Normal/detailed raw：`results/twilight_latest_breakdown_20260926_v1/diagnostic_final/`，含每題 JSON、normal-overlap Chrome trace；`analysis/{summary.json,timeline_per_token.csv,tpot_breakdown_diagnostic.csv,selection_stage_common_origin.csv,selection_detailed_active.csv,resident_cache_details.csv,new_kv_writeback_details.csv}`。
- 容量 raw：`results/twilight_latest_breakdown_20260926_v1/vram/full_gpu_kv_32k.json`；目前方法 `vram_probe/*/inventory.json`，含 storage aliases/data pointers，僅保留本地。
- No-hit raw：`results/twilight_latest_breakdown_20260926_v1/nohit_control/`。Correctness raw：`correctness003/`、`bitmap_gate003/`。
- 最後正式 source SHA-256：`scripts/run_ruler_partial_h2d_tpot_case_v1.py` `840754ff99596eb68649185519ec260f3c61f866c22b036b751fbc7b875d35f6`；`source/headinfer/headinfer/twilight_offload_cache.py` `225f3947e2672cd689666238c84a01ad961a4f049b2c8b1c41f3c19ee450f0a9`；`resident_gpu_mapping.py` `f92a3b5135bcbd4fb5279d6ce679df2f9003f6214fdcaf322183be06d440aec2`；`resident_zero_copy.py` `c7553f75cc2fbee7bc3a19fe20809d5635c012a36acacd40a1aff7a3b9b5cc21`；`resident_zero_copy.cu` `ccf6b381ddc23c3deac8289fa89e00692259468b0a086504cc2844bfb95a37d8`。diagnostic_final 的 source hash 在其 manifest；後加的 workspace probe 僅在 diagnostic flag 下執行，故 source hash 不同，沒有將兩者的 timer 直接相減。
