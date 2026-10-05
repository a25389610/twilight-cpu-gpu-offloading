# RetroInfer budget提高：兩組各10題品質診斷與同輪TPOT

日期：2026-10-05。完整固定20題診斷，非候選完整130題測試；正式TPOT三題各兩輪。

## 問題與假設

RetroInfer原設定在multikey_3/multiquery為20/92.5，低於目前Twilight Direct p=.90的40/95。使用者要求提高budget，先跑這兩個任務各10題並看TPOT。假設提高retrieval可補回遺漏真實KV，提高estimation可補近似資訊；並不預設品質會單調上升。

## 方法與條件

- Llama-3.2-3B-Instruct、BF16、RTX5060Ti16GB、Batch1；CPUcore4、cacheratio=.05、graphsoff，Fullprefill。
- 4arms：base=.018/.232；r030=.03/.232；r050=.05/.232；e350=.018/.35（依序retrieval/estimation）。逐項只改一個budget。
- 與完整130題相同frozen32Kcohort，兩個指定tasks完整各10題，沒有依正誤篩選request。quality沿用原budget/greedy/EOS與scorer。base20題byte-identical沿用已驗證的130題結果，候選新增60題；各request候選順序交錯。
- formal重新跑base與三個候選：相同三個timingrequests，各arm兩輪共24runs，arm順序輪轉/反向；fixedtoken1、32forwards、D1warm-up、D2–D32共31tokens synchronizedCPUwall，排除load/prefill，無componentprofile。formal先跑，quality後跑，qualitylatency不納入formal。
- 使用parameteradapter包住未修改的`benchmark_retroinfer_ruler_case.py`，只覆寫config中的retrieval/estimation。官方Python/C++/CUDA source及3Bconfig未修改；immutablemanifest凍結case/adapter/shell/scorer及官方trackedhashes。

## 主要結果

| Arm | Retrieval | Estimation | multikey_3 /100 | multiquery /100 | Formal TPOT ms/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.018 | 0.232 | 20.00 | 92.50 | 24.568 |
| r030 | 0.03 | 0.232 | 30.00 | 95.00 | 24.979 |
| r050 | 0.05 | 0.232 | 40.00 | 95.00 | 27.976 |
| e350 | 0.018 | 0.35 | 20.00 | 92.50 | 25.145 |
| 你的Direct p=.90 | — | — | 40.00 | 95.00 | 歷史55.939；非本輪配對 |

兩task等權診斷平均：base56.25、r03062.50、r05067.50、e35056.25；你的相同20題也是67.50。不能把67.50誤認成完整RULER130題總分。

| Arm | 相對base較好 / 同分 / 較差題數 | 六次formal TPOT範圍 ms |
| --- | ---: | ---: |
| base | 0 / 20 / 0 | 24.075–26.621 |
| r030 | 2 / 18 / 0 | 24.759–25.652 |
| r050 | 3 / 17 / 0 | 27.615–28.484 |
| e350 | 1 / 18 / 1 | 24.268–28.052 |

**觀察**：r050在這兩個tasks平均追平你的方法，20題3better/17same/0worse，TPOT平均27.976。r030改善較少但平均24.979。e350一題+25、一題-25，task平均無淨改善。這是有限固定subset的實測，非一般化/完整品質結論。

## 逐題結果

### niah_multikey_3：完整10題

| Request | 原設定 | r030 | r050 | e350 | 你的Direct p=.90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 051_niah_multikey_3_i014 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 052_niah_multikey_3_i013 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 053_niah_multikey_3_i006 | 0.00 | 0.00 | 100.00 | 0.00 | 100.00 |
| 054_niah_multikey_3_i003 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 055_niah_multikey_3_i007 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 056_niah_multikey_3_i016 | 0.00 | 100.00 | 100.00 | 0.00 | 100.00 |
| 057_niah_multikey_3_i005 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 058_niah_multikey_3_i000 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 059_niah_multikey_3_i001 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 060_niah_multikey_3_i002 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### niah_multiquery：完整10題

| Request | 原設定 | r030 | r050 | e350 | 你的Direct p=.90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 071_niah_multiquery_i002 | 75.00 | 100.00 | 100.00 | 100.00 | 75.00 |
| 072_niah_multiquery_i003 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 073_niah_multiquery_i010 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 074_niah_multiquery_i008 | 100.00 | 100.00 | 100.00 | 75.00 | 75.00 |
| 075_niah_multiquery_i004 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 076_niah_multiquery_i006 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 077_niah_multiquery_i011 | 75.00 | 75.00 | 75.00 | 75.00 | 100.00 |
| 078_niah_multiquery_i012 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 079_niah_multiquery_i013 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 080_niah_multiquery_i007 | 75.00 | 75.00 | 75.00 | 75.00 | 100.00 |


## 驗證與成本

- 60新quality+24新formal全部returncode0/statusok；另外20base沿用有origin/hash記錄。
- 獨立重算80qualityscores，核對requestID/promptsha/count、generationbudget、token/latency長度、EOS或budget、finite結果與actualconfig；13taskcohort未改。
- formal逐run重算31latencymean，核對32steps、mode、promptsha/ID、retrieval/estimation/cache/core/graphs；所有4arms共用相同adapter/case，官方trackedsource/confighash保持一致。
- 新增84個processwall合計1438.94秒（23.98分鐘），不含沿用20題及orchestration/report空檔，不是TPOT。
- base/e350各有較慢單次（26.621/28.052ms），全部保留；小幅TPOT差不解讀成穩定因果收益。r050六次27.615–28.484ms，比base增加約3.408ms（13.87%）。
- 本輪沒有memoryinventory，不報候選cacheVRAM新數字；retrieval增加可能改變executionbuffer等容量，不能沿用base421.339MiB為r050實測。

## 結論、限制與下一決策

r050是最值得補完整130題的候選：已補回目前最關注兩task的平均差距，仍保有約28ms的本輪standaloneTPOT。不能宣布它完整130題達76.18，也不能假設其餘110題不變。提高estimation不單調，品質不可由budget公式保證。

20題已用於調參，屬exploratorydiagnosis，不是獨立held-outtest。完整130題若續跑也是現有cohort驗證；正式generalization需額外未用於調參資料。本輪Twilight沒有freshpaired測量，所以不能用28vs55.939主張嚴格matchedspeedup。

下一決策：若使用者繼續，優先補r050剩餘110題完整品質與獨立VRAM，再按共同品質門檻做freshTwilight/RetroInfer正式pair；不自動啟動下一輪。

## Artifacts與重現

- 本地raw：`results/retroinfer_budget_diagnostic_20261005_v1/`（manifest、quality/base/r030/r050/e350、formal/arm/rep/request、quality.csv、formal.csv、summary.json、commands/processwall/log及reuseprovenance）。
- scripts：`benchmark_retroinfer_budget_case.py`、`run_retroinfer_budget_local.sh`、`run_retroinfer_budget_diagnostic_20261005.py`、`analyze_retroinfer_budget_diagnostic_20261005.py`。在專案root：`python scripts/run_retroinfer_budget_diagnostic_20261005.py`；核算`python scripts/analyze_retroinfer_budget_diagnostic_20261005.py`。
- basequalityraw：`results/retroinfer_ruler130_quality_20261005_v1/`；Twilightqualityraw：`results/twilight_direct_qk_ruler130_quality_20260927_v2/`。
- publicmirror準備Markdown與四個新scripts；rawJSON/CSV/log/prompt/模型/compiledextensions未納入，未commit/push。
