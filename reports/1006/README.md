# 本週研究報告：2026-09-30～2026-10-06

此資料夾集中18份獨立實驗報告、1份文獻調查與1份總週報，共20份；另附本README。這些都是canonical檔案的副本，原始報告仍保留在研究workspace。

可將整個資料夾放到GitHub；下方目錄採用相對連結。整理日期：2026-10-06。

## 閱讀說明

- 同名報告是不同實驗階段的紀錄，條件與source版本可能不同；比較效能時使用各報告的同批基準。
- 最新B0品質結果見`twilight_b0_4096_2048_ruler130_2026-10-05.md`；最新runtime B0=8192尚未補130題品質。
- 報告中的`results/`、`scripts/`、`source/`及本機絕對路徑是canonical workspace的artifact指標，本資料夾未附raw JSON/CSV/log/PT、模型、程式或環境。
- 總週報是本週進度的時間序列，早期「尚未完成」狀態需搭配後面的完成紀錄閱讀。
- 本資料夾是分享用快照，不是第二套canonical週報；本次僅整理檔案，未commit或push。

## 總週報

| 報告檔案 | 內容 |
| --- | --- |
| [2026-09-30.md](2026-09-30.md) | 本週唯一 canonical 週報的副本 |

## 你的方法：效能與品質

| 報告檔案 | 內容 |
| --- | --- |
| [retroinfer_inspired_decode_fusion_2026-10-05.md](retroinfer_inspired_decode_fusion_2026-10-05.md) | 參考RetroInfer的模型運算融合 |
| [tpot_execution_optimization_2026-10-05.md](tpot_execution_optimization_2026-10-05.md) | Execution fusion、metadata reuse與CUDA Graph |
| [tpot_cache_control_optimization_2026-10-05.md](tpot_cache_control_optimization_2026-10-05.md) | Cache control與buffer swap |
| [tpot_free_acceleration_followup_2026-10-05.md](tpot_free_acceleration_followup_2026-10-05.md) | CUDA QK／Quest与union續優化 |
| [tpot_b0_comparison_2026-10-05.md](tpot_b0_comparison_2026-10-05.md) | B0＝8192／4096／2048正式TPOT比較 |
| [twilight_b0_4096_2048_ruler130_2026-10-05.md](twilight_b0_4096_2048_ruler130_2026-10-05.md) | B0＝4096／2048各130題RULER品質 |
| [twilight_p070_tpot_vram_2026-10-05.md](twilight_p070_tpot_vram_2026-10-05.md) | p＝0.70的TPOT與VRAM |

## Baseline比較與瓶頸分析

| 報告檔案 | 內容 |
| --- | --- |
| [full_runtime_comparison_2026-10-05.md](full_runtime_comparison_2026-10-05.md) | Full runtime對照 |
| [baseline_tpot_profile_2026-10-05.md](baseline_tpot_profile_2026-10-05.md) | 本研究／RetroInfer／FreeKV的TPOT分佈與profiling |
| [retroinfer_ruler32k_tpot_vram_runtime_2026-10-05.md](retroinfer_ruler32k_tpot_vram_runtime_2026-10-05.md) | RetroInfer的TPOT、VRAM與單題耗時 |
| [retroinfer_ruler130_quality_2026-10-05.md](retroinfer_ruler130_quality_2026-10-05.md) | RetroInfer的130題品質 |
| [retroinfer_vs_twilight_2026-10-05.md](retroinfer_vs_twilight_2026-10-05.md) | RetroInfer與Twilight結果比較 |
| [retroinfer_budget_diagnostic_2026-10-05.md](retroinfer_budget_diagnostic_2026-10-05.md) | RetroInfer retrieval budget診斷 |
| [freekv_ruler32k_tpot_vram_quality_2026-10-05.md](freekv_ruler32k_tpot_vram_quality_2026-10-05.md) | FreeKV的130題品質、TPOT與VRAM |

## Source code移植與可行性

| 報告檔案 | 內容 |
| --- | --- |
| [flexicache_local_feasibility_2026-09-30.md](flexicache_local_feasibility_2026-09-30.md) | FlexiCache |
| [retroinfer_local_feasibility_2026-10-05.md](retroinfer_local_feasibility_2026-10-05.md) | RetroInfer |
| [specache_local_feasibility_2026-10-05.md](specache_local_feasibility_2026-10-05.md) | SpeCache beta |
| [freekv_local_feasibility_2026-10-05.md](freekv_local_feasibility_2026-10-05.md) | FreeKV保守移植及async限制 |

## 文獻調查

| 報告檔案 | 內容 |
| --- | --- |
| [twilight_overlap_literature_survey_2026-09-30.md](twilight_overlap_literature_survey_2026-09-30.md) | 與本研究重疊的相關論文survey |
