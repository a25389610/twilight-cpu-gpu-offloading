# Twilight 目前執行路徑（截至 2026-09-28）

## 目前首選：32K 低 VRAM direct QK 近似路徑

使用者已接受已測 130 題 task-level 品質相同、其中一題未計分文字不同的
取捨。於 Llama-3.2-3B-Instruct、BF16、RTX 5060 Ti 16 GB、Batch 1、
32K、Twilight dynamic Top-p `p=.90`、Quest `B0=8192` 條件，首選
previous-token resident cache + GPU mapped CPU KV read 的低 VRAM 組合，
再指定 **`--twilight-qk-backend triton`**。完整 32K historical KV 留在
CPU pinned slab；此 backend 融合 INT4 prepare 與 QK，屬 approximate。
當前 source 三題各兩輪 matched formal TPOT **68.292 → 55.042
ms/token（-19.40%）**，六組都改善；D32 live allocated 平均 +4.765 MiB、
process peak allocated paired 差為 0。09/27 前一量測時段的最快數字
**65.309 → 52.104 ms/token** 保留為歷史已測結果，不與當前六組混算。
完整命令見
`results/twilight_direct_qk_current_confirm_20260928_v1/rep1/003_qa_1_i011/direct/command.json`，
口徑見 `reports/twilight_direct_qk_preferred_path_2026-09-28.md`。

Current-source 32K RULER 13×10 paired quality：Macro 76.1796 兩臂相同、
0/130 官方分數退步、129/130 EOS 前答案 token exact。唯一不同的 FWE
題三個詞相同，但括號頻次數字改變且未納入 scorer。Selection/logits
不 bit-exact；品質範圍與 raw 見
`reports/twilight_direct_qk_ruler130_quality_2026-09-28.md`。需要 exact
研究對照時，把同一 command 的 backend 改回 `triton_prepare`。CLI 的
CLI 預設仍為 `pytorch`，未變更既有實驗語義；`triton_prepare` 可明確指定為本次 matched exact 對照。
此首選身份只適用上述已測 32K 設定，不代表跨 Context／`p`／模型驗證。

## 正式 cache-off control

`scripts/run_ruler_partial_h2d_tpot_case_v1.py` 在指定 `--twilight-top-p`、
且未指定 `--twilight-previous-token-resident-cache` 時，會自動套用目前的
optimized nonresident profile。舊的 cache-off 組合若與此 profile 衝突，
會在載入模型前報錯。`top_p`、Context 與 request 仍是實驗變因；
**約 121 ms/token 只對已量測的 32K、p=.90、Batch 1、D2–D32 三題成立**。

固定執行設定：`short_head_count=0`、dynamic budget、layer-batched
selection、GQA group、`triton_prepare`、CPU flat gather、GPU compact
GQA union、direct Attention layout、layer RoPE、early GPU metadata、
4-chunk gather/H2D、quant metadata reuse、fused Quest score、
skip unused host views，以及 token-batched new-KV D2H。
`cpu_bitmap_union=True` 延續已量測 control 的設定；GPU compact union
啟用時，正常執行不走 CPU union 的實作分支。

已量測的 fresh matched cache-off 三題 TPOT：121.770、119.602、
122.796 ms/token，平均 121.390 ms/token。量測條件、correctness 與
source snapshot 見 `reports/twilight_resident_cache_optimization_2026-09-26.md`
及 `results/twilight_resident_optimized_current_v1/`。這次路徑整理
只更動 runner 的參數解析，沒有重新量測 TPOT，因此上述數字是
整理前的基準，不是新 benchmark。

## Previous-token resident cache

加上 `--twilight-previous-token-resident-cache` 時，runner **不套用**上述
cache-off profile；既有 cache 路徑與 ablation 參數全數保留，包括
`torch`／`native`／`native_compact` hit/miss mapping、`torch`／`triton`
hit copy，以及 `torch`／`triton` snapshot。正式 optimized candidate 使用
先前 formal optimized candidate 使用 `native_compact` mapping + `triton`
hit copy；snapshot 用 `torch`。
cache-on 目前仍是 default-off candidate，不能把它當成跨條件 production
baseline。

09/26 後續系統優化的一個已測候選，在上述 cache-on 設定再加上
`--twilight-resident-snapshot-backend contiguous` 和
`--twilight-resident-position-snapshot-backend reference`。它保存 Attention
layout 作為 per-layer resident snapshot，並保留已解碼 selected-position
Tensor 的 reference。三題 32K `p=.90`、兩輪 matched pair 的舊→新正式
TPOT 是 **95.159 → 92.262 ms/token**；六組均改善、輸出 exact。
細節與限制見 `reports/twilight_resident_contiguous_snapshot_2026-09-26.md`。
`--twilight-resident-pack-hit-indices` 是未採用的 opt-in ablation。

**09/26 當時最快、仍符合 CPU offloading VRAM 前提的 opt-in 候選**，在
上述 `contiguous` snapshot + `reference` position 路徑再使用
`--twilight-resident-mapping-backend gpu_bitmap` 與
`--twilight-resident-zero-copy`。完整 historical KV 保留在 CPU pinned
slab；GPU 只保留前一步 selected resident K/V，對 miss 從 mapped CPU
slab 直接讀取。最終 source 同條件三題各兩輪 matched formal TPOT
**91.890 → 73.787 ms/token（-19.70%）**，六組皆改善，peak GPU
allocated 各組不變、已測輸出 exact。miss logical payload 仍經 PCIe，
不是零傳輸。詳見 `reports/twilight_resident_zero_copy_2026-09-26.md`；
此路徑仍 default-off，未跨 Context／`p`／GPU／quality cohort 驗證。

**09/27 最新最快 opt-in 候選**以 GPU mapped resident 路徑與已測三項
VRAM flags（`--twilight-vram-alias-history-staging`、
`--twilight-vram-lazy-gpu-pack`、
`--twilight-vram-ondemand-attention-capacity`）為 frozen baseline，
再加上：

```text
--twilight-projection-graph
--twilight-exact-top-p-graph
--twilight-resident-precompute-gpu-mapping
```

同一最終 source、三題 32K、`p=.90`、Batch 1、三輪 matched formal
TPOT **74.303 → 66.609 ms/token（-10.36%）**，九組皆快；Selection、
Attention K/V、new-KV 與已測 logits exact。D32 live GPU allocated 約
**+240.550 MiB**，D2–D32 peak 約 **+239.069 MiB**，但 unique cache
GPU storage 不變，完整 historical KV 仍留在 CPU pinned slab。
`--twilight-exact-scan-graph` 是較早的有效單項，但已由完整 Top-p Graph
取代，兩個 flags 不可同時使用。新組合仍 default-off、只在上述條件驗證；
正式命令可參考
`results/twilight_system_optimization_20260927_v1/final_matched/rep1/003_qa_1_i011/final/command.json`。
詳細消融與限制見 `reports/twilight_system_graph_handoff_2026-09-27.md`。

**09/27 低 VRAM opt-in 替代路徑**沿用同一 frozen baseline，只加
`--twilight-exact-top-p-graph` 與
`--twilight-resident-precompute-gpu-mapping`，**不加**
`--twilight-projection-graph`。同一 source 的三臂、三題各三輪 matched
formal TPOT 為 baseline **71.735**、低 VRAM **65.529**、完整 Graph
**63.426 ms/token**；低 VRAM 改善 **8.65%**，而 D32 live allocated
只 **+1.817 MiB**、Decode peak 只 **+0.457 MiB**。它比完整 Graph
少用 **238.734 MiB** D32 live VRAM，但慢 **2.103 ms/token**。
九組 formal logits hash 和 003 Selection／Attention K/V／new-KV trace
exact。此結果只對已測三題 32K `p=.90` 成立；低 VRAM 正式命令可參考
`results/twilight_low_vram_speed_20260927_v1/matched/rep1/003_qa_1_i011/low_vram/command.json`，
報告見 `reports/twilight_low_vram_speed_tradeoff_2026-09-27.md`。

## 舊實驗檔案

舊的 `scripts/run_twilight_*_v1.py`、`scripts/analyze_twilight_*_v1.py`、
reports 與 raw artifacts 保留作為研究證據；其中企圖從現行通用 runner
執行舊 cache-off 組合的腳本，現在會明確報錯。`twilight_offload_cache.py`
底層舊實作尚與 cache-on ablation 共用，尚未安全分離，因此不能把
原始碼中仍可看到的分支視為目前正式 cache-off 路徑。
