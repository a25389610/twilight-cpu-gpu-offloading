# Direct INT4 QK 低 VRAM 路徑：TPOT、Selection、resident cache 與 VRAM breakdown

日期：2026-09-28（Asia/Taipei）

## 範圍、source 與計時口徑

本輪只量測，不修改 production Selection／Attention 演算法。從先前 **65.309 → 52.104 ms/token** 的 `results/twilight_official_gap_20260927_v1/direct_qk_matched/rep1/<request>/direct/command.json` 逐題還原 command，只改 output path；matched exact 對照唯一切換 `--twilight-qk-backend triton_prepare`，Direct 用 `triton`。其餘原旗標包含 previous-token resident cache、GPU bitmap mapping、GPU mapped CPU KV read、low-VRAM alias/lazy/on-demand、Top-p CUDA Graph、precomputed GPU mapping，均保持原樣。三題 32K、Llama-3.2-3B-Instruct BF16、RTX 5060 Ti 16 GB、Batch 1、dynamic `p=.90`、Quest B0=8192、32 fixed Decode steps。完整 historical KV 在 CPU pinned memory。

原候選與本輪共同 source SHA-256：runner `b28eebd01391635ddddd44547091237956231fa32862410b001e4a1893e4f4eb`；`twilight_offload_cache.py` `8491eb046506ffef127bcc02b1ea877be11ad0310bf84312567235fb1753e16e`；`twilight_fused_qk.py` `ed75777a4a7c7733341024bba3fc760c846d58f9ade96d0959a26eff840988e7`。逐題 seed command hash、全部實際命令及其餘 source hash 見 raw `results/twilight_direct_qk_breakdown_20260928_v1/manifest.json`。

**Formal**：D1 warm-up；D2–D32 共 31 token 的 synchronized wall，模型載入與 Prefill 排除，沒有 component profiling。**Normal diagnostic**：每題 D33–D35 三個 post-timing token，共 9 個；沿用 09/26 `twilight_latest_latency_vram_breakdown_2026-09-26.md` 的 common-origin CUDA Event／host timeline 與 priority interval ownership，重疊時間只歸一類。**Detailed Selection diagnostic**：另三題各三 token，CUDA Event parent/child；不是 normal timeline 的同一批 token。下面的 `%` 僅以所在 diagnostic wall 為分母，不能稱作 formal TPOT 的直接百分比或保證可省時間。

## 1. Fresh matched formal TPOT

| Request | rep1 exact → Direct | rep2 exact → Direct | 兩輪 exact | 兩輪 Direct | matched 改善 |
|---|---:|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 68.879 → 55.574 | 69.032 → 55.710 | 68.956 | **55.642** | 19.31% |
| 002_vt_i002 | 68.937 → 55.861 | 69.325 → 56.054 | 69.131 | **55.957** | 19.06% |
| 003_qa_1_i011 | 69.025 → 56.372 | 69.624 → 56.060 | 69.324 | **56.216** | 18.91% |
| **六組平均** | | | **69.137** | **55.939** | **-13.198 ms/token；19.09%** |

六個 matched pair 均較快，但**本輪沒有重現 52.104 ms/token 的絕對值**。前一量測時段 65.309 → 52.104、09/28 前一次 68.292 → 55.042、本輪 69.137 → 55.939；三輪的相對／絕對 paired 差約維持 19–20%／13 ms，兩臂絕對 TPOT 同向漂移。漂移原因未確認，不混算不同時段，也不把歷史最快值當成目前實測。Raw：`analysis/formal_tpot.csv`、`formal/rep{1,2}/`。

## 2. Normal-overlap TPOT 大項：互斥 exposed-time ownership

Normal diagnostic wall 平均 **62.338 ms/token**，比本輪 formal Direct **55.939** 高 **6.399 ms（11.44%）**；下表百分比全部以 **62.338** 為分母，不能移作 formal TPOT 百分比。它可與 09/26 約 74 ms 報告的**同一 ownership 方法**比較，但 execution flags 和 source 快照不同，跨報告差值不能單獨歸因於 Direct QK。

| 互斥 exposed 類別 | ms/token | diagnostic wall % | 實際邊界 |
|---|---:|---:|---|
| KV Selection：query prep + Quest + Twilight | **12.939** | **20.76%** | Selection core 起至 membership decision |
| Selection 後處理：GQA union + group handoff | 2.284 | 3.66% | GPU union 1.092；group lengths／D2H 1.192 |
| Previous-token cache management | **6.291** | **10.09%** | cache update 的 mapping pop、new slot、snapshot、state 與其餘 control |
| resident hit + mapped CPU miss + layout assembly | **6.151** | **9.87%** | 同一 fused CUDA kernel interval，不拆 PCIe／hit copy |
| Attention | 3.000 | 4.81% | FlashAttention varlen |
| 其他 model compute | **24.652** | **39.55%** | QKV/RoPE、O projection、MLP、Norm 等 |
| new-KV writeback | 1.147 | 1.84% | GPU staging／D2H exposed ownership |
| runtime/control residual | 5.873 | 9.42% | 未被上述覆蓋的 wall，含未標記工作／排程間隙 |
| **diagnostic wall** | **62.338** | **100.00%** | 3 requests × D33–D35 |

Ownership 每個 token 的類別加總等於該 token wall。`runtime/control residual` 不是全數純 CPU work；CUDA Event interval 可含 queue gap。Raw：`analysis/{timeline_per_token,tpot_breakdown,tpot_major_breakdown}.csv`。

## 3. Selection detailed breakdown

### 3.1 與大項表同輪同 origin 的 exposed 階段

Selection core **12.939 ms**；若連後處理，整個 selection pipeline **15.223 ms（24.42% normal diagnostic wall）**。

| 階段 | exposed ms/token | Selection core % | normal diagnostic wall % |
|---|---:|---:|---:|
| Query preparation | 1.008 | 7.79% | 1.62% |
| **Quest 第一輪** | **3.994** | **30.87%** | **6.41%** |
| **Twilight 第二輪** | **7.937** | **61.34%** | **12.73%** |
| Selection core | **12.939** | **100%** | **20.76%** |
| 後處理 GPU GQA union／bitmap | 1.092 | core 之外 | 1.75% |
| group-length／index D2H handoff | 1.192 | core 之外 | 1.91% |

### 3.2 獨立 detailed CUDA Event（parent/child 不相加）

這組 diagnostic wall **67.980 ms/token**，高於 formal **12.042 ms**；Selection core parent **13.718 ms**。下表 `% 整體` 的分母只能是 **67.980 detailed diagnostic wall**。Quest parent、QK parent、Top-p parent 含各自 queue gap；子項與 parent 重疊。

| 真實 timer 邊界 | CUDA Event ms/token | Selection core % | detailed wall % |
|---|---:|---:|---:|
| Query FP32 stack／prepare | 0.522 | 3.81% | 0.77% |
| Quest min/max stack + fused score | 2.882 | 21.01% | 4.24% |
| Quest Top-K page | 0.422 | 3.08% | 0.62% |
| Quest B0 candidate expansion | 0.489 | 3.56% | 0.72% |
| **Quest 第一輪 parent** | **4.005** | **29.20%** | **5.89%** |
| INT4 metadata stack／reuse | 0.034 | 0.25% | 0.05% |
| **Direct INT4 QK fused stage** | **3.929** | **28.64%** | **5.78%** |
| QK scale | 0.096 | 0.70% | 0.14% |
| INT4 QK parent（含上列與間隙） | 4.369 | 31.85% | 6.43% |
| **Top-p CUDA Graph replay** | **4.522** | **32.97%** | **6.65%** |
| **Top-p parent（含 replay／間隙）** | **4.578** | **33.37%** | **6.73%** |
| final membership decision | 0.023 | 0.17% | 0.03% |
| **Selection core parent** | **13.718** | **100%** | **20.18%** |
| 後處理 GQA union／bitmap（另於 core 外） | 1.593 | 11.62% of core *reference* | 2.34% |

Direct stage 把 candidate gather、INT4 unpack/dequantize、approximate QK 融在**同一 Triton kernel**；沒有獨立且可靠的內部時間。Top-p 現由 CUDA Graph replay，不能把 graph 裡的 argsort／softmax／cumsum 虛構為獨立 exposed 子項。Quest min/max timer 同時包含 metadata stack 與 fused score。實際可用於演算法研究的量級：**Quest 3.994 ms／normal wall 6.41%；整個 Twilight 第二輪 7.937 ms／12.73%**。Direct QK 本身的 3.929 ms、Top-p parent 的 4.578 ms來自另一組 detailed run，分別占該 run Selection core 28.64%／33.37%、detailed wall 5.78%／6.73%；不能加到上面的 7.937 ms 或當作 formal 可省上限。Raw：`analysis/{selection_common_origin,selection_detailed}.csv`。

## 4. Previous-token cache 與 no-hit control

下表 CUDA Event／host span 可能彼此交疊，parent 與 child 也不能相加。最後一欄僅是單一 Event interval 除以 normal diagnostic wall；**cache path 合計改用上節互斥 ownership**。

| 實際 component | CUDA Event ms/token | host span ms/token | 計時與歸屬 |
|---|---:|---:|---|
| current／previous bitmap mapping（cache update 取用段） | 0.080 | 0.093 | 真正 mapping 已於下列 handoff 預先執行 |
| GPU mapping + group lengths／control handoff | 1.192 | 8.092 | 前移的 `map_membership` + lengths GPU→CPU；host span 含等待先前 GPU 工作 |
| fused resident hit K/V + mapped CPU miss K/V + Attention layout | **6.151** | 0.633 | 單一 kernel，不硬拆 bus read／hit copy／layout write |
| current new-token slot copy | 1.108 | 2.730 | GPU K/V 寫入 Attention layout |
| resident K/V snapshot update | 1.360 | 0.492 | 當步 selected K/V 複製到下一 token resident |
| position reference + bitmap state update | 0.224 | 0.407 | reference／clone bookkeeping |
| cache metadata/control parent | 12.442 | 10.020 | **parent；覆蓋上面部分子項，不再加總** |
| new-KV writeback exposed ownership | **1.147** | — | 另列的大項；GPU stage Event 0.950、D2H Event 0.017、CPU scatter wall 0.052，互有 overlap |

互斥分類中的**狹義 cache management** 是 **6.291 ms（10.09% normal wall）**；加上屬 selection 後處理、但包含預先 mapping 的 group handoff 為 **7.483 ms（12.00%）**；再含不可拆的 hit/miss/layout fused kernel，完整 **previous-token cache + KV preparation path 是 13.634 ms（21.87%）**。new-KV writeback 另列 1.147 ms，不重複加到前面。這些是診斷 exposed ownership，並非「若拿掉 cache 就必然能省」的因果量。

沿用 default-off `run_twilight_nohit_control_v1.py`，在**同一 Direct QK execution path** 把 previous bitmap 強制全零；GPU mapping、mapped CPU read、resident snapshot 仍執行，三題各一個 matched pair（preliminary）：

| Request | resident Direct TPOT | all-miss control TPOT | reuse 淨收益 | all-miss logical mapped payload |
|---|---:|---:|---:|---:|
| 001 | 55.523 | 92.854 | 37.331 | 512.256 MiB/token |
| 002 | 55.965 | 91.117 | 35.152 | 488.048 MiB/token |
| 003 | 55.731 | 94.025 | 38.294 | 523.262 MiB/token |
| **平均** | **55.740** | **92.665** | **36.926 ms/token（相對 all-miss -39.85%）** | — |

no-hit 的 D2–D32 共 868 layer rows/request 均 0 hit；與相鄰 resident run 的 checkpoint/final logits hash 相同。control 仍負擔 snapshot，因此收益是**本 mapped-read 系統中 reuse 的淨 end-to-end 效果**，不是純粹的 cache 管理成本。每題只一組，標為 preliminary。Raw：`analysis/{cache_details,new_kv_details,nohit_preliminary}.csv`、`nohit/`。

## 5. VRAM：Full 32K GPU KV 容量與 Direct path

Llama config 為 28 layers、8 KV heads、head_dim 128、BF16 2 bytes。Full 32K K+V：`28×8×32768×128×2×2 = 3,758,096,384 bytes = 3,584 MiB = 3.500 GiB`。本輪獨立 process 實配兩個 CUDA tensor，unique storage／allocated 增量均 **3,584 MiB**。這只是容量 baseline，**未跑 Full GPU TPOT**。

| CUDA allocator／容量指標 | model only／共同 weights | model + Full 32K GPU KV | Direct D32（三題均值） |
|---|---:|---:|---:|
| live allocated | 6,128.348 MiB | 9,712.348 MiB | **7,474.695 MiB** |
| D2–D32 Decode peak allocated | — | 未執行 Decode | **7,500.352 MiB** |
| process peak allocated（含 Prefill/D1） | 6,128.348 MiB | 9,712.348 MiB * | **9,119.525 MiB** |
| D32 reserved | 6,174.000 MiB | 9,758.000 MiB | **7,734.000 MiB** |
| process peak reserved | 6,174.000 MiB | 9,758.000 MiB * | **9,918.000 MiB** |
| 扣 model-only 的 D32 allocated 增量 | — | **3,584.000 MiB／3.500 GiB** | **1,346.347 MiB／1.315 GiB** |
| cache object unique GPU backing storage | — | K/V **3,584.000 MiB** | **1,324.817 MiB／1.294 GiB** |

`*` Full 欄的 peak 只有模型載入加 Full K/V allocation，沒有 Decode workspace，不能與 Direct 的 process peak 當作同 workload 比較。以 **unique KV/cache backing** 比較，Direct 為 Full KV 的 **36.96%**、少 **63.04%**；若採較保守的 **allocator D32 incremental**（含不完全可歸類的 runtime／Graph bytes），則為 **37.57%**、少 **62.43%**。兩種口徑不混合。Full raw：`full_gpu_kv_32k.json`；Direct raw：`vram/rep1/<request>/direct/inventory.json`。

Direct historical GPU KV 沒有完整常駐；CPU pinned historical KV 三題平均 **3,538.318 MiB**，其他 pinned staging **224.547 MiB**，皆不算 VRAM。

### 5.1 Direct D32 unique GPU storage（`data_ptr` 去重）

以下分母 **1,324.817 MiB**，是 cache object 中 unique backing storage；alias/view 不重複計算。與 allocator incremental **1,346.347 MiB** 相差 **21.530 MiB**。Graph static input/output 另外在 diagnostic wrapper 找到 **2.250 MiB**，不在 cache inventory 遞迴範圍；其餘約 **19.280 MiB** 為未歸類的 live allocator bytes，可能含 Graph private 與其他 runtime，不可全部稱作 KV 或 Graph private。

| unique cache GPU component | MiB | cache unique % | 備註 |
|---|---:|---:|---|
| **Previous-token resident K/V** | **577.858** | **43.62%** | 下一 token hit reuse |
| **Twilight INT4 codes** | **441.784** | **33.35%** | 全部量化 K codes |
| INT4 scale + minimum | 27.612 | 2.08% | 兩者各 13.806 |
| **Quest min + max metadata** | **220.390** | **16.64%** | 兩者各 110.195 |
| Attention-layout selected K/V | 49.881 | 3.77% | 真正供 Attention 的 K/V buffer |
| previous-token bitmaps + current GQA bitmap | 7.156 | 0.54% | 6.909 + 0.247 |
| new-token GPU K/V staging | 0.109 | 0.01% | 小型當步 staging |
| 其他 cache object GPU storage | 0.026 | <0.01% | 可追蹤小項 |
| **cache object unique 合計** | **1,324.817** | **100%** | 不含 Graph 靜態 storage |
| Top-p Graph static input/order/desired（另計） | **2.250** | — | `0.750 + 1.500 + <0.001 MiB`，三題相同 |

`selected-history staging K/V` 在本路徑與 Attention-layout K/V **alias 同一 backing storage**，沒有另列約 100 MiB；短暫 mapping indices、Top-p workspace 不冒充 D32 persistent。Graph private allocation 本輪沒有獨立可歸因的 allocator tag，只保留在上述 residual，不杜撰數值。Raw：`analysis/{vram_process,vram_storage,graph_static_storage}.csv`、`graph_inventory/`。

### 5.2 Direct QK 是否降低 Selection temporary VRAM？

同 source、同三題、相同 diagnostic workspace probe，`triton_prepare` exact vs Direct：

| per-layer Selection 指標 | exact `triton_prepare` | Direct `triton` | 差值 |
|---|---:|---:|---:|
| Top-p 結束點仍 live 的 local temporary unique storage | 107.857 MiB | **11.857 MiB** | **-96.000 MiB** |
| Selection allocator peak 相對 layer 入口 | 106.601 MiB | **10.666 MiB** | **-95.935 MiB** |
| D2–D32 Decode peak allocated（整個 process） | 7,586.595 MiB | **7,500.352 MiB** | **-86.243 MiB** |
| D32 live allocated | 7,469.931 MiB | **7,474.695 MiB** | **+4.765 MiB** |
| process peak allocated（含 Prefill/D1） | 9,119.525 MiB | 9,119.525 MiB | 0 |

因此 FP32 estimated-K 大型 temporary 確實不再建立，**Selection per-layer peak 實測降低約 96 MiB**；這不是 D32 persistent 節省，D32 live 反而平均多 4.765 MiB，與先前 matched formal 的增量一致。allocator Decode peak 少約 86 MiB；整個 process peak 仍由較早的 Prefill/D1 主導。Raw：`analysis/selection_workspace.csv`、兩臂 `vram/` inventory。

## 正確性、限制與回答

- 六組 exact／Direct 的 P4 checkpoint logits hash 相同；Direct 的 D2/D32 與 exact 不同，符合先前數值近似的已知 tradeoff。Direct formal、normal timeline、detailed Selection、VRAM、Graph inventory 的三題 checkpoint/final logits hash 均一致；no-hit 與其 paired resident Direct 也一致。另於 003 加相同 correctness trace flags 比對 trace-only 與 trace+normal profiling：共同 D1–D32 的 Quest Selection **21,504/21,504**、GQA union **7,168/7,168**、Attention K/V **84/84**、new-KV key/value hash **7,168/7,168**，以及 formal/checkpoint/final logits 全相同。初版直接比整份 trace 曾失敗：normal 額外執行 D33–D35，且 D32 new-KV 的 `consumer_decode_step`／`verified_before_next_read` 因 D33 而更新；此差異不是 K/V data 差異。失敗原始 `correctness003/analysis.json` 與修正共同步數口徑 `analysis_common_steps.json` 均保留。這支持 profiling 沒有改變**已測 003 共同步數**的 Selection、Attention K/V 與 logits；正式跑本身未啟用逐 element tracing，不外推其他 request 的逐 row bit-exact。Direct 相對 exact 的品質範圍另見 130 題報告。
- Normal timeline 和 detailed Selection 是兩組不同診斷，不能混用分母或把 CPU wall 與 CUDA Event 直接相加。前移 mapping 在 handoff interval 內；fused mapped read 沒有 PCIe-only timer。no-hit 三題各一組是 preliminary。
- **最大三個 latency 負擔**：其他 model compute **24.652 ms（39.55% normal diagnostic wall）**、Selection core **12.939 ms（20.76%）**、cache management + fused KV assembly **12.442 ms（19.96%）**；若把前移 mapping handoff 算入 cache preparation，後者是 **13.634 ms（21.87%）**，但 handoff 已在主表後處理列，不能再對主表加總。
- **真正 Selection 演算法空間**：Quest 第一輪 exposed **3.994 ms（6.41% normal wall）**、Twilight 第二輪 **7.937 ms（12.73%）**；Direct QK fused kernel **3.929 ms（5.78% detailed wall）**、Top-p parent **4.578 ms（6.73% detailed wall）**。這些不是 formal TPOT 的保證加速上限。
- **cache 代價／收益**：狹義管理 **6.291 ms**；連前移 handoff **7.483 ms**；連 mapped hit/miss layout assembly 的完整 preparation **13.634 ms**。同 path all-miss preliminary 的 reuse 淨 TPOT 收益 **36.926 ms/token**。
- **最大三個 VRAM component**：resident K/V **577.858 MiB**、Twilight INT4 codes+scale/min **469.396 MiB**、Quest min/max **220.390 MiB**。Direct QK 明確降低 temporary peak，但不減少這三個 persistent metadata/cache 主體。

本輪量測後停止；未做新的 optimization。分析與 driver：`scripts/{run_twilight_direct_qk_breakdown_20260928.py,analyze_twilight_direct_qk_breakdown_20260928.py,run_twilight_graph_inventory_20260928.py}`；全部 raw JSON／CSV／command／log 位於 `results/twilight_direct_qk_breakdown_20260928_v1/`，不納入公開鏡像。
