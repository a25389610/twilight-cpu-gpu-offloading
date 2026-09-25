# Twilight resident cache：連續 snapshot 與 selected-position reference

日期：2026-09-26（Asia/Taipei）  
性質：保持 Twilight-inspired Selection、GQA union、Attention K/V 與輸出 exact 的系統路徑優化。

## 問題與假設

先前最快的 previous-token resident selected-KV cache 已用 `native_compact`
CPU hit/miss mapping 與 Triton hit copy，但每步仍將 Attention layout 的完整
selected history 用兩次 `torch.index_select` 複製到 per-layer resident K/V。
003 的單 token 診斷中，snapshot 約 5 ms CUDA active，且需另外傳送
`history_destinations` index。假設保留 Attention layout（每 KV group 歷史
後面多一個 new-token 位置）作為下一步 resident layout，即可改用連續 K/V
copy，並省去該 index H2D；下一步 mapping 的 hit source offset 加上 group
gap，維持完全相同的歷史 token 對應。

第二個獨立假設：GPU bitmap 解碼後的 CPU selected-position Tensor 已有自己的
storage；resident cache 可保存 Tensor reference，省去每 KV group 的 CPU
`clone`，而不改變 selected positions。

## 實驗條件與計時

- Llama-3.2-3B-Instruct、BF16、FlashAttention 2、RTX 5060 Ti 16 GB、
  PyTorch 2.7.0+cu128／CUDA 12.8；Batch 1。
- 固定三題 32K cohort：`001_niah_multikey_3_i011`、`002_vt_i002`、
  `003_qa_1_i011`；Twilight-GQA dynamic Top-p `p=.90`，B0 8192-token。
- 32 fixed Decode steps；D1 warm-up，正式 TPOT 是 D2–D32 的 31 個 token
  同步 CPU wall 平均。Prefill、model load 和 diagnostic token 均不計入。
- 同份 source 快照兩輪交錯順序 fresh matched pair；兩邊都用先前最快的
  `native_compact` mapping、Triton hit copy、GPU compact GQA union、
  `triton_prepare`、4-chunk gather/H2D 和 token-batched new-KV D2H。
  唯一 execution 差異：舊路徑 `torch` snapshot + CPU position `clone`；
  新路徑 `contiguous` snapshot + CPU position `reference`。
- Formal run 不插 Selection/Attention trace。correctness smoke、D33 component
  diagnostic、common-origin timeline 與 formal TPOT 分開。

## Correctness

- `native_compact` attention-layout hit source offset 通過 150 組隨機映射核對。
- 003 完整 32K smoke：selected positions、GQA union、D1/D2/D32 Attention
  K/V trace、new-KV trace、P4/D1/D2/D32/final logits 逐項 exact。CPU
  selected-position reference 另以同規格 smoke 對 clone 核對 exact。
- 最新快照的六組 formal pair：prompt、selection/GQA SHA、resident hit/miss
  trace、physical KV transfer、checkpoint/final logits 均一致；
  `scripts/analyze_twilight_resident_current_best_v1.py` 從 raw JSON 和
  `command.json` 重新核對，結果為 `validated`。
- Selection 演算法、B0/B1、Top-p、precision、Attention valid lengths 與
  selected-KV miss-only H2D payload 未變；此輪沒有額外品質近似。

## 整體 fresh matched TPOT

| 題目 | 第 1 輪舊→新 ms/token | 第 2 輪舊→新 ms/token |
|---|---:|---:|
| 001_niah_multikey_3_i011 | 94.804 → 92.588 | 94.851 → 92.278 |
| 002_vt_i002 | 94.500 → 92.190 | 94.545 → 92.546 |
| 003_qa_1_i011 | 96.191 → 91.463 | 96.065 → 92.505 |

六組平均 **95.159 → 92.262 ms/token**，降低 **2.898 ms/token
（3.05%）**；六組皆改善，paired 差值中位數 **-2.441 ms/token**。
這是**同輪 matched**
比較，不拿先前其他 GPU 狀態下的 94.483 或本輪其他快照約 83 ms/token
直接相減。

## 單機制結果、診斷與未採用方向

- 連續 snapshot 單機制 ablation：第一組三題均改善
  **94.832 → 92.550 ms/token**。反向順序重測的三題有兩題改善、一題
  `001` 出現 +8.213 ms 的偏慢值；第三組三題再度全改善
  **85.539 → 83.659 ms/token**。全部 9 組中 8 組改善，paired 差值
  中位數 -2.035 ms/token；不能刪除異常組後宣稱全數穩定。
- D33 snapshot CUDA active 在三題由 **5.041/4.875/5.778 ms** 降至
  **1.578/1.412/1.599 ms**。因 Attention layout 每 group 多一列，
  snapshot logical bytes 僅增加 114,688 bytes/token；
  `history_destinations` 不再 H2D，003 resident index H2D
  **16,659,344 → 8,329,672 bytes**，copy calls **112 → 84**。
  這些是 post-timing D33 active/bytes，不可加總或當正式 TPOT 節省。
- CPU selected-position reference 單機制 ablation：兩組交錯三題 pair
  均改善。第一組平均 **83.579 → 82.778**，第二組
  **83.969 → 82.683 ms/token**。D33 position snapshot CPU wall 約
  **0.69–0.73 → 0.02 ms/token**，僅是診斷子項。
- Packed hit-index H2D 試作通過 150 組 mapping 隨機核對及 003 完整
  correctness smoke，但三題 formal 首輪 **82.616 → 86.686 ms/token**；
  003 有 +11.520 ms 異常值，反向重測又較快 1.603 ms。001/002 首輪
  也未改善；因此保留 opt-in ablation，**不納入目前最快候選**。
- GPU bitmap→CPU sorted positions 的另一版兩階段 native decoder 在
  合成 8×32K map 上為 **0.223 vs NumPy 0.072 ms/layer**，未接入
  runtime；不推論正式 TPOT。

## 目前瓶頸與限制

最新新版本 003 的同輪正式 TPOT **90.987 ms/token**；正式計時後三個
common-origin diagnostic token 的 wall 平均 **99.334 ms**。其非重疊大類：
Selection exposed **27.796 ms**、Selection 完成至 Attention 開始
**41.294 ms**、已標記 model compute active **28.744 ms**、未歸屬
**1.500 ms**。這些比例與時間只描述 diagnostic run，不是正式 TPOT
分解；不能與先前其他 run 的 timeline 直接相減。post-selection 的
CPU bitmap decode 約 **5.855 ms active wall**、selected-KV CPU gather
**3.898 ms active wall**、H2D **4.443 ms CUDA active**，彼此和 parent
可能重疊。Selection 外成本仍大，尚未完成瓶頸轉移。

本輪僅測 Batch 1、32K、`p=.90`、三題固定 token Decode；未驗證其他
Context、`p`、batch、模型、GPU 或完整 RULER quality cohort。position
reference 的安全性已於本輪路徑驗證，其他 selection backend 或未來
in-place 改寫 positions 時需重驗 storage ownership。偶發單次 TPOT
偏慢原因尚未確定；報告保留原值。

## Artifacts 與下一決策

- 整體兩輪配對：`results/twilight_resident_current_best_v1/{manifest.json,formal,analysis/summary.json}`。
- 最終 source 快照 003 完整 trace：
  `results/twilight_resident_current_best_v1/smoke_final/003_qa_1_i011/`。
- snapshot、reference、packed ablation：
  `results/twilight_resident_{contiguous_snapshot,position_reference,packed_indices}_v1/`。
- 新版 common-origin 診斷：
  `results/twilight_resident_current_best_v1/normal_overlap/003_qa_1_i011/`。
- 原始碼：`source/headinfer/headinfer/{twilight_offload_cache.py,resident_mapping.py,resident_mapping.cpp}`；
  runner：`scripts/run_ruler_partial_h2d_tpot_case_v1.py`；驗證器：
  `scripts/analyze_twilight_resident_current_best_v1.py`。

Formal source snapshot 的 SHA-256：`twilight_offload_cache.py`
`dba3a8172b0eb74065a8abcdaceebf06c3b31909f12b40150e0c4ecd66c3b9cc`、
`resident_mapping.py`
`4e7e4ec5eb01dc503dd964e873f7e514748d5907ed2e95feb52f097d12c2b47c`、
`resident_mapping.cpp`
`ead760b8cdc635e7e88a597b22322ad4f883406cd71b18e01cd75dd0767a5d77`、
`resident_hit_copy.py`
`e037e57c56fbcaa67aa58844264eabe8c0a8538f9e3bf69771ba5d409e4b2f2e`、
case runner
`4255185d6b3236f3223e3fccabe100b415ad60e63d5d0c0ed83f5bc0538710ea`。

保留新組合為 default-off **optimized candidate**。下一步先用 fresh matched
Context×`p` 小矩陣與品質 cohort 確認適用範圍，再針對 post-selection
index handoff、host control 與 gather/H2D 做單機制 ablation；每步維持
Selection/Attention/logits correctness gate。
