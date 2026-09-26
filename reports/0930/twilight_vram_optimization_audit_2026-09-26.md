# Twilight GPU mapped resident 路徑：Decode VRAM optimization audit

日期：2026-09-26（Asia/Taipei）。本報告針對既有最快的 Twilight + previous-token resident cache + GPU mapped CPU KV read 路徑；**未修改 Quest／Twilight Selection 演算法、INT4 精度、Quest metadata 精度或 Attention chunk 數**。完整 historical KV 仍在 CPU pinned slab。研究問題是：在三題 32K 的既定 correctness 與 TPOT 範圍內，能否刪除不再消費的 GPU storage，並將短期工作 buffer 改成安全的按需容量。

## 凍結條件與判讀界線

Llama-3.2-3B-Instruct、BF16、FlashAttention 2、RTX 5060 Ti 16GB、Batch 1、原三題 `001_niah_multikey_3_i011`／`002_vt_i002`／`003_qa_1_i011`，Twilight dynamic Top-p `p=.90`、B0=8192、32 fixed Decode steps。D1 warm-up，正式 TPOT 僅 D2–D32 共 31 token，formal run 不插入同步 memory/Selection profiling。每個單項以**同一 source hash、同一 execution flags，僅一個 VRAM flag 不同**的 baseline/candidate 包夾配對；各做三題 × 兩輪，最終組合追加至三題 × 四輪。不同 ablation 的 baseline 均值因量測時段波動，**不可跨列相減**。

VRAM inventory 是另一個 diagnostic run：D32 同步後用 `untyped_storage().data_ptr()` 對 cache tensor 的 backing storage 去重；D1 後重設 CUDA peak 計數，故可分別報 **D2–D32 Decode peak** 與 **含 Prefill/D1 的 process peak**。D33 每 layer 診斷 Selection temporary peak，不能與 D32 live storage 相加，也不是 formal TPOT。`reserved` 是 allocator 保留量，並非 live tensor bytes。

## 1. Source／lifetime audit：所有找到的候選

| GPU storage／方向 | 原本三題平均 | 用途、consumer 與安全邊界 | 是否改 execution path／本輪決策 |
|---|---:|---|---|
| **selected-history staging K/V** | **99.750 MiB** | `Quest` flat path 仍配置兩個 GPU history tensor；mapped fast path 的 D2–D32 由 fused kernel 直接讀 CPU pinned miss 並寫 Attention layout，沒有讀這份 staging。D1 direct-layout 也把 K/V 寫 Attention buffer。以 Attention storage 的 view 作相容 alias，保留舊路徑欄位但不新增 backing storage。 | opt-in alias；同一 Attention 算法／kernel；**採納** |
| **舊 GPU pack K/V** | **12.469 MiB** | `Quest` 一般 per-head pack path 使用；目前 GQA flat mapped path 無 runtime consumer。延後配置為 `None`；`state_snapshot()` 修正為對可選 tensor 作 unique-storage accounting。 | opt-in lazy；不觸碰當前 Attention；**採納** |
| **Attention-layout selected K/V** | **99.762 MiB** | 每層同一工作 buffer，曾按所有 Query head 上限預配；目前三題的 GQA union 遠小於該上限。先配置約半容量；每層取得實際 `total_attention` 後若不足就配置足量新 buffer，不截斷 K/V。 | opt-in on-demand capacity；不增加 Attention chunk；**採納** |
| resident K/V capacity headroom | **577.827 MiB** | 每層保存 previous selected K/V，當前 5% + 64 rows 餘裕。改為剛好 `rows` 可省約 21.56 MiB，但可能頻繁重新配置；兩輪 TPOT 方向相反。 | opt-in tight capacity；**不納入 final**，速度證據不足 |
| GQA current／previous bitmap 與 indices | **7.156 MiB** | current bitmap 約 0.247 MiB；previous bitmap 逐層必須保留到下一步 mapping。將前後 bitmap 當成同一可寫 storage 會覆蓋 hit/miss 來源；packed bitmap 需更動 mapper kernel。 | 本輪不做變更；維持 exact mapping |
| Quest min/max metadata | **220.391 MiB** | 448 個 min/max backing storages，逐層 score 需要；未發現同內容的 GPU 副本。降 dtype 會影響 Selection。 | 保留 |
| Twilight INT4 codes／scale／minimum | **469.395 MiB** | 28 層 quantized K；`_layer_quant_metadata` 與 per-entry `_quant_key_*` 是同 storage alias，inventory 已去重，沒有第二份 469 MiB 可刪。 | 保留原表示與 Selection |
| resident K/V 與 Attention-layout alias | **577.827 + 99.762 MiB** | Attention assembly 讀 previous resident 同時寫 current layout，之後 snapshot current；兩者生命週期重疊，直接 alias 會覆蓋仍待讀的 previous rows。 | 不安全，保留分離 storage |
| Selection／Top-p temporary | **D33 per-layer peak delta 116.518 MiB** | 主要為 FP32 estimated-K；它是單層短生命週期 PyTorch temporary，allocator 已跨 layer 循環 reuse，並非乘 28 的 persistent VRAM。要再縮通常需改 fused Selection 算法。 | 本輪不改演算法 |
| new-token GPU K/V staging、其他 cache allocation | **0.109 + 0.026 MiB** | token-batched D2H 使用前者；後者多為小型 control tensor。 | 保留，收益極小 |

CPU pinned historical slab 平均約 **3,538 MiB**；host flat／pack staging 不計 VRAM。本輪未把 CPU RAM 節省算作 GPU 容量收益。

## 2. 單機制 ablation

三題平均的 VRAM 使用 **D32 live allocated** 與 **D2–D32 Decode peak allocated**；單項 memory run 各題一次，TPOT 為 matched 三題各兩輪。所有 candidate 的三題 formal checkpoint/final logits 與 selected-position SHA 相同；003 另有 D1/D2/D32 Attention K/V 共 84 rows、new-KV 共 7,168 rows exact。memory runs 不用作正式 TPOT。

| 單項機制 | baseline → candidate unique cache storage MiB | baseline → candidate D32 allocated MiB | baseline → candidate Decode peak MiB | matched baseline → candidate TPOT ms/token | 判斷 |
|---|---:|---:|---:|---:|---|
| history staging alias | 1,486.886 → 1,387.136 | 7,628.675 → 7,528.650 | 7,746.577 → 7,646.592 | 74.796 → 74.980（+0.25%） | 約省 99.75 MiB；小幅差異落在 run 波動範圍，保留 |
| lazy GPU pack | 1,486.886 → 1,474.417 | 7,628.675 → 7,617.784 | 7,746.577 → 7,735.749 | 74.570 → 74.320（-0.34%） | 約省 12.47 MiB；未證明真實加速，保留 |
| Attention on-demand capacity | 1,486.886 → 1,437.005 | 7,628.675 → 7,578.288 | 7,746.577 → 7,696.416 | 73.676 → 73.335（-0.46%） | 約省 49.88 MiB；安全 grow guard，保留 |
| tight resident capacity | 1,486.886 → 1,465.329 | 7,628.675 → 7,610.355 | 7,746.577 → 7,728.975 | 74.027 → 74.307（+0.38%） | 約省 21.56 MiB；兩輪配對方向相反，**暫不採納** |

其餘要求的每項 memory gate 如下；`reserved` 與 `peak reserved` 在各列相同，且不代表 live tensor：

| Memory diagnostic | process peak allocated MiB | reserved／peak reserved MiB | D33 Selection temporary peak delta MiB |
|---|---:|---:|---:|
| baseline | 9,131.930 | 9,933.333 | 116.518 |
| history staging alias | 9,131.930 | 9,933.333 | 116.464 |
| lazy GPU pack | 9,119.525 | 9,916.667 | 116.728 |
| Attention on-demand capacity | 9,131.930 | 9,933.333 | 116.621 |
| tight resident capacity | 9,131.930 | 9,933.333 | 116.768 |

沒有把短期 Selection workspace 當 persistent 節省。baseline/candidate 每一組的 source hashes、commands、individual request TPOT、storage aliases 與 memory stages 在本地 raw artifacts；baseline inventory 另以最終 source 重跑三題，D32 bytes 與最初 inventory 逐題完全相同。

## 3. 最終組合：alias history + lazy pack + on-demand Attention

未加入 tight resident。組合後逐題四輪 matched formal：

| Request | baseline ms/token | 組合 ms/token | candidate − baseline |
|---|---:|---:|---:|
| 001_niah_multikey_3_i011 | 74.523 | 74.539 | +0.016 |
| 002_vt_i002 | 74.666 | 75.104 | +0.438 |
| 003_qa_1_i011 | 74.899 | 75.078 | +0.179 |
| **三題 × 四輪均值** | **74.696** | **74.907** | **+0.211 ms（+0.28%）** |

前兩輪均值 +0.387 ms，後兩輪 +0.035 ms；將每輪三題先平均、以四輪為單位估算的 95% t interval 約 **[-0.216, +0.638] ms**，包含零，樣本仍少。此條件下**未觀察到明顯、穩定的 TPOT regression**，但測得均值略慢，不能稱為「更快」或數學上完全不變；002 的四輪平均 +0.438 ms 仍是限制。若研究要求嚴格的「均值不可增加 0 ms」，則此三項組合尚未達到該更嚴標準。沒有從這些小差值推論效能因果。

| GPU memory 指標（三題平均） | frozen baseline | 最終組合 | 差值 |
|---|---:|---:|---:|
| **D32 live CUDA allocated** MiB | 7,628.675 | **7,468.114** | **-160.561** |
| **D2–D32 Decode peak allocated** MiB | 7,746.577 | **7,586.138** | **-160.439** |
| 含 Prefill/D1 process peak allocated MiB | 9,131.930 | 9,119.525 | -12.405 |
| CUDA reserved／peak reserved MiB | 9,933.333 | 9,916.667 | -16.666 |
| **unique cache GPU storage** MiB | 1,486.886 | **1,324.786** | **-162.100（-10.90%）** |
| D33 per-layer Selection temporary peak delta MiB | 116.518 | 116.684 | +0.166（diagnostic noise） |

Allocator live delta **160.561 MiB** 與 unique cache storage delta **162.100 MiB** 不相等：前者包含 allocator 中其他 live objects，後者僅數 cache 物件的 unique backing storage。`reserved` 僅少 16.666 MiB，因 Prefill 等較早階段已使 caching allocator 持有大區塊；**Decode allocated peak** 有實質下降，但整個 process 的最高峰仍主要在 Prefill/D1。本輪沒有把二者混稱。

目前剩下最大的五個 GPU memory component（D32 unique storage，三題平均；INT4 codes／scale／minimum 合併表示）：

| Component | MiB | 占 1,324.786 MiB |
|---|---:|---:|
| previous-token resident K/V | **577.827** | **43.62%** |
| Twilight INT4 K representation | **469.395** | **35.43%** |
| Quest min/max metadata | **220.391** | **16.64%** |
| Attention-layout selected K/V working buffer | **49.881** | **3.77%** |
| GQA current／previous bitmap 與 indices | **7.156** | **0.54%** |

最後組合的三題 formal checkpoint/final logits 與 selected-position SHA 均 exact；003 D1/D2/D32 GQA membership bitmap **84/84** rows exact，Attention K/V **84/84** trace rows exact，new-KV **7,168/7,168** rows exact。Selection 算法、selected K/V、Attention call 結構保持相同。驗證僅涵蓋此 GPU、三題 32K、`p=.90`、固定 32-step Decode；其他 Context、`p`、模型、Batch 或長自由生成尚未驗證。

## Artifact 與 source provenance

- 本地 raw：`results/twilight_vram_audit_20260926_v1/{formal,memory,correctness,bitmap_gate003,analysis}/`。`analysis/{summary.json,formal_matched_by_request.csv,formal_matched_mean.csv,memory_by_request.csv,memory_mean.csv}`。raw JSON/CSV、storage data pointers、logits 與 trace 不納入公開鏡像。
- 單項與 final benchmark driver：`scripts/run_twilight_vram_audit_v1.py`；分析器：`scripts/analyze_twilight_vram_audit_v1.py`。實作在 `scripts/run_ruler_partial_h2d_tpot_case_v1.py`、`source/headinfer/headinfer/{quest_offload_cache.py,twilight_offload_cache.py}`，四個 VRAM audit flags 預設關閉。既有 `resident_gpu_mapping.py` 與 `resident_zero_copy.py/.cu` 未更動。
- 完成組合測試時的 SHA-256：runner `8ebbcedeb1eb066809f78a0623aedc6e324f1b4f068223ae2216e18319ec4020`；Quest cache `682702d3e5c65dbfb554f3f7ae5f97f1bd75eb8f3d05d24a3ad8e6c7f8c811fd`；Twilight cache `02b3267d3c15b2662e3d845de33d623078829e43343a8f48d58b484d1a4b98cf`。每次 matched run 另存完整 source-hash JSON。

**下一個研究決策**：使用者先審閱 160.561 MiB Decode live allocation 節省與 +0.28% 測得 TPOT 均值差。本輪至此停止，不繼續改 Twilight Selection 演算法。
