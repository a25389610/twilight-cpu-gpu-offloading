# UUID Estimation Cluster特徵：3失敗／3成功案例探索對照

2026-10-07。使用者希望比較logit spread／variance、cluster exponential mass誤差、weighted-V numerator誤差，以及cluster在全量attention中的權重。目的：找出能解釋先前Exact Estimation救回UUID的條件；不預設失敗題所有指標都更大。

## 樣本與baseline

- 原本official RULER UUID20題中已診斷的0、2、3，對照原index排序前3個Full／本研究／RetroInfer原本都答對的1、4、5。pair0→1、2→4、3→5。沒有依本輪cluster指標挑題。
- 六題相同Llama-3.2-3B-Instruct BF16 Batch1，原32K prompt。local RetroInfer retrieval .018／estimation .232／cache .05、seed2025、graph=False。每題fresh process/index，同題內baseline/control/exact共用index。
- Full原生成history作teacher forcing。失敗題取已固定的首次分歧第38／37／36個生成token；成功對照取配對相同absolute生成步數。這是步數配對，並非UUID相同字元位置或相同Query，未控制所有內容／答案位置變因。
- 捕捉當步28layers、8KV heads、每KV head對應3Q heads的所有非空estimation clusters；每個cluster／Q head保留一列。重複層/head/cluster不是獨立題目，樣本量是每組3題。
- baseline逐層實際execution buffer tape、history／分區固定的approx replay必須logits max abs0、Q相同；才換全部estimation為同cluster真KV。多層介入後下游Q可變。完整generation另從prefill生成128/EOS；history/selection演變且LRU未完整恢復，故為補充。
- historical rescue label屬整題，不代表CSV每一群是造成失敗的群。本輪未做單cluster next-token intervention，不能稱已找到rescued cluster。

## 指標定義與計算

s_i=q·K_i/√d、s_c=q·C/√d。FP32 dot product，FP64統計／exponent／weighted sums，共同max shift防overflow。spread=max−min；variance使用population除m。cluster真mass Z=sum exp(s_i)，近似mass Za=m exp(s_c)，MassError=abs(Z−Za)/Z。記錄logZ／logZa保留原scale。

Weighted-V error為||sum exp(s_i)V_i−exp(s_c)sumV||／||sum exp(s_i)V_i||，對應使用者OutputError，但屬未normalized numerator。分母clamp1e-30，同時保存共同shift後絕對誤差與真numerator norm，不能跨head把shift後絕對量當原scale。

額外記錄：

1. cluster true mass／全部真KV的Z；全部包含steady、retrieval、estimation、dropped。Q／KV取RetroInfer baseline當步狀態，並非另一個Full runtime自己的Q／KV，這是same-Q all-real-KV attention權重。
2. within-cluster normalized output差：true numerator/Z vs summedV/m。
3. 只把一群exact化、其他群保留approx的normalized attention output relative變化；重新計算共同分母。這是在baseline固定Q的attention-local FP64計算，不是單cluster完整model因果介入。
4. stored BF16 centroid logit與真群mean logit差；有限精度會使理想Jensen下界有小偏差。
5. answer_tokens：原答案span逐token在該cluster的數量；answer-containing estimator群另外列，不把membership比例當attention mass。

## 六題重現結果

| 原index | historical類別 | 記錄step | 原版完整生成score | 全estimation Exact score | 當步baseline／exact命中Full token |
|---|---|---:|---:|---:|---|
| 0 | 先前失敗、全estimation救回 | 38 | 0 | 100 | False／True |
| 1 | 先前答對 | 38 | 100 | 100 | True／True |
| 2 | 先前失敗、全estimation救回 | 37 | 0 | 100 | False／True |
| 3 | 先前失敗、全estimation救回 | 36 | 0 | 100 | False／True |
| 4 | 先前答對 | 37 | 100 | 100 | True／True |
| 5 | 先前答對 | 36 | 100 | 100 | True／True |

## 全部estimation clusters的每題中位數

每一列先在該題內取中位數，避免大題／cluster數較多的題主導比較。

| index | spread | variance | mass error | weighted-V error | cluster full mass | 單cluster normalized output變化 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.13971 | 1.33574 | 0.461673 | 0.59896 | 6.43607e-05 | 0.000158574 |
| 1 | 3.94213 | 1.21603 | 0.438021 | 0.57938 | 9.82785e-05 | 0.000218351 |
| 2 | 3.93022 | 1.21018 | 0.4329 | 0.573629 | 9.08804e-05 | 0.000220988 |
| 3 | 4.10078 | 1.3242 | 0.461212 | 0.598811 | 8.80565e-05 | 0.0002202 |
| 4 | 3.86129 | 1.16334 | 0.423623 | 0.565777 | 8.36622e-05 | 0.00019862 |
| 5 | 3.94824 | 1.22201 | 0.437892 | 0.575085 | 8.63441e-05 | 0.000196425 |

## 含答案token的estimation clusters：每題中位數

| index | spread | mass error | weighted-V error | cluster full mass | 單cluster normalized output變化 |
|---|---:|---:|---:|---:|---:|
| 0 | 5.1434 | 0.605654 | 0.708567 | 9.67055e-05 | 0.000248907 |
| 1 | 4.68866 | 0.526599 | 0.644976 | 0.000125331 | 0.000302 |
| 2 | 5.54584 | 0.645232 | 0.76107 | 8.62318e-05 | 0.000274995 |
| 3 | 4.79283 | 0.553048 | 0.680035 | 0.000128287 | 0.000328736 |
| 4 | 5.21137 | 0.573907 | 0.689099 | 0.000131082 | 0.000369136 |
| 5 | 4.48136 | 0.507672 | 0.640973 | 0.000114231 | 0.000315137 |

## 配對比較與判讀

| 指標 | 失敗題全群中位數較高（3pairs） | 含答案群中位數較高（3pairs） |
|---|---:|---:|
| logit_spread | 3/3 | 3/3 |
| logit_variance | 3/3 | 3/3 |
| mass_error | 3/3 | 3/3 |
| weighted_v_relative_error | 3/3 | 3/3 |
| full_attention_mass | 2/3 | 1/3 |
| single_cluster_normalized_output_error | 2/3 | 1/3 |

這些只描述此6題的一步，不等於分類閾值或可泛化weakness condition。成功題也可有大spread／mass error／weighted-V error；相對cluster誤差與normalized output、final logits之間不必單調。即使部分中位數較高，也需要新題／新index檢查及單群介入；沒有因上述觀察直接建predictor／改production policy。

## 驗證／限制

- 六題共1,898,400cluster×Q-head rows；原prompt hash／config／fixed-step／control／finite、672layer/Q-head combinations、每組內cluster身份無重複、mass比例／總mass<=1、source／inputsSHA核對。
- FP64 approximate reference與native BF16單head最大abs差：0.0153084（guardrail <.04）；非兩者bit-identical聲明。
- 初版逐cluster GPU scalar export過慢，停止保留不完整attempt；改head批次export，前1000rows與原版最大abs差2.842170943040401e-14，未改公式／選取。failed attempt source/log/partialCSV保留本地，不當完整run。
- 原existing exact kernel verification沿用；本輪control每題logits差0。未量formal TPOT／VRAM，診斷額外KV／FP64／CPU資料記錄不屬baseline效率。
- sample historical outcome-selected，3fail／3success，同模型/context/task，一次fresh index，缺少heldout、隨機seed重複與other-task generalization；不能把百萬row當百萬独立實驗。

## 下一決策

先依每head normalized output變化與真mass，從失敗及成功兩類各選候選群做單cluster fixed-history／fixed-index替換，觀察下一token logit margin。需要同時測高分失敗群、高分成功群與對照群，避免只展示救回案例。只有可重現且能在未參與探索的新題區分危險群，才進入低成本估計指標／方法設計；此下一階段尚未執行。

## Artifacts／公開範圍

本地`results/uuid_cluster_features_20261007/`：cases.json、manifest.json、recording_verification.json、summary.json、六case的result.json／clusters.csv、logs、attempt_scalar_recording/。cluster CSV／rawJSON／原prompt／全文prediction／KV／weights／external checkout均不公開。

可同步`reports/1014/uuid_cluster_features_2026-10-07.md`與`scripts/uuid_cluster_features_20261007/{prepare.py,case.py,features.py,run.sh,analyze.py,report.py}`。本輪production source未改，未commit/push。
