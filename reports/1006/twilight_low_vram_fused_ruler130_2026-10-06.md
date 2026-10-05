# 低VRAM shared-workspace＋fused snapshot：32K RULER130

完成與核對時間：2026-10-06T04:04:03.695330+08:00

## 問題／假設與控制

使用者要求TPOT分佈完成後直接測最新約29–30ms、低VRAM版本的130題，確認移除約1GiB預留成本是否維持原版品質。RTX5060Ti16GB／Llama-3.2-3B-Instruct／BF16／Batch1／B0=4096／p=.90，Quest→exact INT4QK→dynamicTop-p→GQAunion，原FlashAttention使用bounded max_seqlen／GPU cu_seqlens。各layer小resident quota、共用Attentionworkspace、超quota selectedKV從CPU補回；newtoken insertion與snapshot合併。Production defaults未改。

130個fresh候選processes；同frozen32Kcohort13tasks×10、prompt IDs／references／generation budgets，greedy到EOS或budget。對照已保存原B0=4096／p=.90的130題結果；不是重跑原版的fresh paired timing，也不是與FreeKV／RetroInfer品質配對。

## 已驗證的品質

| 版本 | RULER macro /100 | 滿分題 | 生成tokens |
| --- | ---: | ---: | ---: |
| 保存原版4096 | 75.359077 | 88/130 | 2615 |
| 低VRAM fused | 75.359077 | 88/130 | 2620 |

候選相對原版 better／same／worse=0／130／0；output token exact=129/130。

| Task | 原版 | Fused | 差值 |
| --- | ---: | ---: | ---: |
| niah_single_1 | 100.000000 | 100.000000 | +0.000000 |
| niah_single_2 | 100.000000 | 100.000000 | +0.000000 |
| niah_single_3 | 100.000000 | 100.000000 | +0.000000 |
| niah_multikey_1 | 100.000000 | 100.000000 | +0.000000 |
| niah_multikey_2 | 100.000000 | 100.000000 | +0.000000 |
| niah_multikey_3 | 30.000000 | 30.000000 | +0.000000 |
| niah_multivalue | 100.000000 | 100.000000 | +0.000000 |
| niah_multiquery | 95.000000 | 95.000000 | +0.000000 |
| vt | 78.000000 | 78.000000 | +0.000000 |
| cwe | 0.000000 | 0.000000 | +0.000000 |
| fwe | 86.668000 | 86.668000 | +0.000000 |
| qa_1 | 50.000000 | 50.000000 | +0.000000 |
| qa_2 | 40.000000 | 40.000000 | +0.000000 |

## 速度與記憶體的分離證據

前輪同批原版→fused正式TPOT：33.048044→29.885338ms；三題兩輪／186samples每arm、fixedtoken1 D2–D32 synchronized wall、無profiler／greedy。Final nonweight live allocation 1158.615560→1089.201335MiB。含KV／metadata／workspace／logits，不是純KV、不含opaque driver/library。

另一批最新三方法正式TPOT：本研究30.097575、RetroInfer22.250318、FreeKVconservative30.278378ms。非equal-quality／equal-VRAM，FreeKV仍有未解crossprocess logits變異。不同batch不相減算新收益。

本輪130題qualityprocess wall加總59.65分鐘，含import／NVCC/JIT／model load／prefill／generation／checkpoints，不能當formalTPOT或persistent-model inference速度。

130題自然生成結束時nonweight PyTorch live allocation：平均1105.466、min1047.522、max1178.926MiB。各request實際context／response與D32 timingcases不同；不是memorypeak，也不是fresh memorypaired ablation。

## 驗證與限制

130process returncodes/status、130reference_resultsha、cohortmanifest／current與snapshotsourcehash、實際prompt tensorhash、request／BF16／Batch1／B0／p、EOS只在尾端／budget、commands僅adapter/output差異、原baseline所有executionenv flags保持，fusedopt-in／GPUlengths／native非shadow核對。從IDs重新decode與官方scorer重算，與batchsummary相同；saved checkpoint tensors finite共1082個。

結果僅涵蓋這130題與設定；不宣稱所有輸入bit-exact／通用品質不變，也不代表模型或context generalization。BoundedFA schedule有小浮點差異，quota改實體residency但不降低logicalselection；前輪shadow gate／syntheticpayload支持該範圍正確性，不能替代所有長generation／beam／batch驗證。沒有重用本輪quality timing宣稱與另一篇方法的同品質速度優勢。

## Artifacts／公開檔案

本地root：results/tpot_slim_fused_20261006/quality130/；manifest.json、source_snapshot/、requests/<request>/result.json／result.logits.pt／per_tokenCSV／command／env／processlog、quality_analysis.json、quality_verified.csv、quality_verification.json、verification.log。完整逐題輸出在outputs_130.md，只保留本地。

本輪應同步：

- reports/1007/twilight_low_vram_fused_ruler130_2026-10-06.md
- scripts/tpot_execution_opt/run_slim_fused_quality130.py
- scripts/tpot_execution_opt/verify_slim_fused_quality130.py
- scripts/tpot_execution_opt/report_slim_fused_quality130.py
- scripts/tpot_execution_opt/finish_slim_fused_quality130.py

未納入rawJSON/CSV/log/PT／完整prediction/reference／weights／trace／source snapshots／binary／externalcheckouts。Scripts需既有localruntime／模型／cohort／原版reference，不是standaloneinstaller。Production source/defaults未改；publicmirrorprepare核對bytes，未commit／push。

## 下一決策

依本130題品質與此前正式速度／memory，判斷是否採用opt-in低VRAM版；若output或score不同，保留差異逐題分析，不能直接標示無損。完整模型／長context generalization與equal-quality baselines仍需另外設計。
