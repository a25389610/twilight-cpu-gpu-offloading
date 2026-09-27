# Twilight mapped-resident 路徑的實作加速：Graph submission 與 GPU mapping handoff

日期：2026-09-27（Asia/Taipei）。本輪目標是在不改 Quest／Twilight Selection 規則、Attention K/V 與 logits，且不保留完整 GPU historical KV 的前提下，降低目前最快 CPU offloading 路徑的 Decode TPOT。所有修改均為 default-off opt-in；沒有修改 Top-p、INT4／Quest metadata 精度、selected membership 或 Attention chunk 數。

## 問題、假設與固定條件

09/26 的最新 formal 三題平均約 74–75 ms/token，normal diagnostic 中 Selection 約 26.7 ms、其他 model compute 約 23.7 ms、resident/mapped KV 約 12.4 ms。現行程式仍逐層執行 8 個 KV group 的 Q/K/V 小投影、逐 head 1D `cumsum`，並在 GQA membership 後先求一次 group lengths，GPU resident mapper 又掃同一 bitmap。假設：保留原本 PyTorch kernel 與數值順序、減少 host submission 和重複 count／同步邊界，可降低正式 TPOT。CUDA Event/CPU wall 只用於定位，不作可加節省預測。

固定 Llama-3.2-3B-Instruct、BF16、FlashAttention 2、RTX 5060 Ti 16GB、Batch=1、32K、Quest B0=8192、Twilight dynamic Top-p `p=.90`、原三題 `001_niah_multikey_3_i011`／`002_vt_i002`／`003_qa_1_i011`、32 個 fixed Decode steps。D1 為 warm-up，正式 TPOT 是未插同步 profiling 的 D2–D32 共 31 token 同步 CPU wall。frozen baseline 是 09/26 GPU bitmap + previous-token resident + GPU mapped CPU KV read，加上已驗證的三項 VRAM flags（history staging alias、lazy GPU pack、on-demand Attention capacity）。完整 historical KV 一直留在 CPU pinned slab。每項消融固定同一 source hash、同一 execution command，只有該項 flag 不同，request 與執行順序交錯；不同消融的 control 均值不得跨列相減。

## 修改與單機制消融

1. `--twilight-projection-graph`：每層仍呼叫原本 8 組 Q/K/V Linear 與同一整層 RoPE；CUDA Graph 重播相同 kernel，將每步 hidden state／cos／sin 複製到 graph static inputs。它**沒有**啟用先前 failed-exact 的整層 QKV GEMM。Graph 在 D1 建立，capture 時間不計 D2–D32。
2. `--twilight-exact-scan-graph`：保留 24 個原本的 1D `torch.cumsum` kernel，以單一 Graph replay 提交；是過渡候選。
3. `--twilight-exact-top-p-graph`：進一步把原本的 `torch.argsort`、gather、softmax、24 個 1D cumsum、`torch.searchsorted` 與 clamp 放在同一 Graph；不與 scan-only flag 同時啟用。保留原 numerical order、tie handling 與 Top-p decision。
4. `--twilight-resident-precompute-gpu-mapping`：在 GQA bitmap 建立後先執行原 GPU mapper，用 mapper 的 `current_lengths` 作原本必要的 CPU length handoff；Cache consumer 重用 mapping outputs，省去額外的 `membership.sum` 和稍後第二次 mapping。只改資料準備時序，不改 hit/miss 定義或 mapped-read assembly kernel。

| 同 source 配對機制 | 配對數 | Control → candidate ms/token | 差值 | VRAM／判斷 |
|---|---:|---:|---:|---|
| 只省 mapper 未消費的 miss arrays／scan | 6 | 73.389 → 73.127 | -0.262 (-0.36%) | 有兩組變慢，方向不一致；**撤回** |
| 逐 head `cumsum` Graph | 6 | 74.454 → 72.875 | -1.578 (-2.12%) | 六組皆快；D32 allocated 約 +1.07 MiB；過渡候選 |
| grouped QKV／RoPE Graph | 6 | 74.682 → 72.565 | -2.117 (-2.83%) | 六組皆快；D32 allocated 約 +238.30 MiB |
| QKV Graph + scan Graph | 9 | 74.156 → 70.542 | -3.614 (-4.87%) | 九組皆快；組合可行 |
| 在 QKV Graph 下，完整 Top-p Graph 取代 scan Graph | 6 | 70.073 → 68.883 | -1.190 (-1.70%) | 六組皆快；額外 D32 allocated 約 +0.75 MiB |
| 在 QKV + Top-p Graph 下，預先算 GPU mapping | 9 | 69.629 → 66.873 | -2.756 (-3.96%) | 九組皆快；無額外 D32 allocated |

小型 operator pilot 中，完整 INT4 prepare＋FP32 QK Graph 保持 exact，但每層僅約 0.670 → 0.646 ms，單一 Graph 額外持有約 8.9 MiB；若為 28 層獨立保留，速度／VRAM 交換尚不理想，本輪未接入正式路徑。QKV Graph shared memory pool 的 003 單輪只有 CUDA allocator reserved 下降，live allocated 未降，未採用。這些 pilot 不當作正式 TPOT 結論。

## 最終 matched formal TPOT

最終版本同時啟用 **QKV Graph + 完整 Top-p Graph + 預先算 GPU mapping**；以下與原 frozen baseline 在**最終相同 source**下三題各三輪配對。九組逐 pair 差值均為負，範圍 -8.442 至 -7.155 ms/token；不能把上表不同 source／輪次的收益加總成此表。

| Request | Frozen baseline | 最終版本 | 差值 |
|---|---:|---:|---:|
| 001_niah_multikey_3_i011 | 73.871 | 66.533 | -7.338 |
| 002_vt_i002 | 74.491 | 66.687 | -7.804 |
| 003_qa_1_i011 | 74.548 | 66.607 | -7.941 |
| **三題 × 三輪平均** | **74.303** | **66.609** | **-7.694（-10.36%）** |

這是同一 CPU offloading execution condition 的 end-to-end Decode TPOT 改善，不是 Full GPU KV baseline，也不含模型載入、Prefill 與 D1 Graph capture。先前不同快照量到的 73.787／74.584／74.907 ms/token 不與此 fresh control 混算。

## Correctness 與 VRAM

九組正式 pair 的三題 checkpoint/final logits SHA 與每層 resident history/hit/miss rows 相同。最終 source 的 003 獨立 diagnostic gate：Quest Selection trace **21,504/21,504**、Twilight budget trace **21,504/21,504**、GQA union **7,168/7,168**、D1/D2/D32 Attention K/V **84/84**、new-KV **7,168/7,168** 與 logits 全部 exact。Selection-trace run 會切到 CPU 可見的診斷路徑；Attention K/V gate 是另一組保持 mapped-read fast path 的 run，兩者不作正式 TPOT。

三題平均的獨立 memory diagnostic，在 D1 後重設 CUDA peak 計數：

| GPU memory 指標 | Baseline MiB | 最終 MiB | 差值 MiB |
|---|---:|---:|---:|
| D32 live allocated | 7,468.114 | 7,708.664 | +240.550 |
| D2–D32 Decode peak allocated | 7,586.138 | 7,825.207 | +239.069 |
| 含 Prefill/D1 process peak allocated | 9,119.525 | 9,119.525 | 0 |
| unique cache GPU backing storage | 1,324.786 | 1,324.786 | 0 |

額外 live VRAM 主要是 Graph static inputs／outputs 與 private allocations；相對 16 GB 裝置約 +0.235 GiB，沒有加入完整 32K GPU KV mirror。Allocator `reserved` 在這次 diagnostic 中約 9,916.667 → 8,515.333 MiB，但它反映 allocator／Graph pool 行為，**不能把 reserved 的下降解讀成 active KV 或實體容量節省**。GPU mapped CPU KV read、CPU pinned historical slab 與 Selection metadata 精度維持原樣。

## 瓶頸位置與量測限制

003 的另一組 D33–D35 common-origin normal timeline：baseline diagnostic wall 80.626、最終 72.252 ms/token；兩者比正式 TPOT 高，故下列百分比**僅以各自 diagnostic wall 為分母**。互斥 ownership 中，最終 Selection query prep + Quest + Twilight 為 **22.934 ms/token（31.74%）**；其他 model compute **22.240 ms（30.78%）**。因此在此診斷樣本中，Selection 仍是最大的單一 exposed 類別，但只比 model compute 稍大。baseline 的 Selection 為 27.626 ms（34.26%），其他 model compute 25.922 ms。這些不是 formal TPOT 直接量出的百分比，也不是未來演算法收益的保證上限。

預先算 mapping 把工作移進原 `group_length_handoff` interval；CUDA Event 亦可包含 GPU queue gap。最終 trace 中舊 `cache_gpu_hit_miss_map` timer 變小、`mapped_host_read_fused_assembly` interval 變大，**不可據此推論 mapped CPU read 的實際 kernel 或 PCIe 流量變差**，也不可將各子項跨版本逐列相減。可靠的整體效果採上述 matched formal TPOT。

此結論限 RTX 5060 Ti 16GB、這三題 32K、Batch 1、BF16、`p=.90`、B0=8192、fixed 32-step Decode。尚未驗證其他 Context／`p`／GPU／模型、長自由生成或完整品質 cohort；Graph 在 D1 建立，D1 latency 與多形狀 Graph capture memory 不包含於正式 TPOT。這輪是 execution implementation optimization，尚不是新的 Twilight Selection 演算法貢獻。

## Artifacts、source 與下一決策

- 本地 raw／commands／source hashes：`results/twilight_system_optimization_20260927_v1/{map_matched,scan_graph_matched,projection_graph_matched,combined_matched,top_p_graph_matched,precompute_map_matched,final_matched,final_gates,final_timeline003,analysis}/`。正式表重算自 `analysis/{matched_pairs.csv,matched_summary.json}`；correctness／VRAM／timeline 見 `analysis/{final_vram_by_request.csv,final_gates_vram_timeline.json}`。raw JSON/CSV、logits 與 trace 不進公開 Markdown mirror。
- 分析器：`scripts/{analyze_twilight_system_optimization_20260927_v1.py,analyze_twilight_system_final_gates_v1.py}`。實作：`source/headinfer/headinfer/{mp.py,twilight_projection_graph.py,twilight_offload_cache.py,twilight_exact_top_p_graph.py}` 與通用 runner。scan-only Graph helper保留供已測 ablation；未採用 fast mapper 與 shared-pool path 均已撤回。
- 最終 source SHA-256：runner `73d1d8137b02447b3ddc327c217f08b9611652963eb32745a7afefc3026b874b`；`mp.py` `805b40565c797ffb4c388a9c0937396ff996be1ebd1263735040d32ad4092e9f`；`twilight_offload_cache.py` `8491eb046506ffef127bcc02b1ea877be11ad0310bf84312567235fb1753e16e`。完整 hash manifest 在 raw artifact。
- 下一決策：先以其他 Context／`p` 與小型品質 cohort 界定這個 opt-in 速度／VRAM／correctness 適用範圍；若要投入 Twilight 演算法創新，使用新的 matched formal TPOT 與同 origin timeline 評估收益。不要把 graph／handoff 的系統加速稱為演算法改良。
