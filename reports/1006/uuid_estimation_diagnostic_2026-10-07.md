# RetroInfer UUID失敗：KV分區與精確estimation診斷（3題）

日期2026-10-07。從新官方RULER UUID pilot取index0／2／3：Full與本研究對、RetroInfer錯。這是已知失敗案例的機制診斷，不是新的隨機品質樣本。研究假設：cluster estimation可能使精確UUID複製失誤；不能先假定答案未取回或所有近似都不可靠。

## 設定與控制

- 同Llama-3.2-3B-Instruct、BF16、Batch1、原32K prompt hash、128budget、greedy/EOS；沿用本local RetroInfer port，retrieval=.018／estimation=.232／cache=.05，core4、graph=False、torch seed2025。Production source未修改；診斷程式僅process內monkeypatch。
- 每題fresh process／fresh index，在prefill保留原始逐token Key／Value及官方cluster membership；驗證每個KV head中每個非steady token恰好屬一個cluster。答案與整筆record位置使用原prompt tokenizer offsets，是真token位置，不是字元比例。
- 用Full的生成token作teacher forcing，逐步找fresh RetroInfer首次argmax不同的位置；本輪均與saved首次分歧一致。每題只建立一次index，後續對照共用它，不重新clustering。
- 當步固定history、開始前steady KV／context／static length、每層cluster分區，以及baseline實際execution KV buffer／有效length。先重播approximate control，逐層Q完全相同、logits max abs差0；才將estimation的cluster centroid／summed Value換成同cluster所有真實逐tokenKV。retrieval／steady buffer與dropped集合不變，使用同weighted-flash kernel的unweighted真KV路徑及相同merge。
- 單步exact介入涵蓋當步全部28layers；上游attention改變後，下游Q會跟著變，這是介入傳播，不宣稱兩個完整forward的每層Q都相同。另有baseline各層Q／KV固定的attention-local FP32近似／精確對照，報告輸出差异。
- 完整生成另將state回到prefill，分别原版與精確estimation生成，沿用同index／budgets；這兩條軌跡的history、Q與後續選取會演變，不能把它說成整段固定分區的因果隔離。GPU block-cache/LRU未恢復初始狀態，完整生成是補充品質觀察；核心因果證據來自taped buffer的單步對照。

## 首次分歧與單步對照

位置為生成序列1-based；概率是相同Full history下的next-token概率，非答案整體概率。

| 題index | 首次分歧 | 正確token | 原版argmax | 精確estimation argmax | 正確token概率原版 | 精確後 |
|---|---:|---|---|---|---:|---:|
| 0 | 38 | `4` | `2` | `4` | 0.25918682% | 99.56404% |
| 2 | 37 | `e` | `ce` | `ce` | 0.00003466% | 40.29377% |
| 3 | 36 | `5` | `6` | `5` | 1.03304368% | 99.98176% |

原版與精確後correct-minus-original-wrong logit margin：

- index0：-5.1875 → 10.6875。
- index2：-14.8750 → -0.3750。
- index3：-4.0625 → 9.7500。

## KV分區：答案token

分母是「答案token × 28layers × 8KVheads」在該首次分歧步驟的membership次數；不是整個cache比例、不是attention probability mass、也不是答案整筆只有一種zone。不同head／layer的同一token可屬不同zone。

| 題index | steady | retrieval真KV | estimation摘要 | dropped |
|---|---:|---:|---:|---:|
| 0 | 0.00% | 9.49% | 53.71% | 36.80% |
| 2 | 0.00% | 1.65% | 25.03% | 73.32% |
| 3 | 0.00% | 5.67% | 56.66% | 37.67% |

逐head/layer數量、整筆record分區、estimation的實際attention mass、attention-local輸出relative L2保存在raw result。文字／UUID相似不等於證實同cluster。

## 同index完整生成（補充觀察）

| 題index | 原版官方score | 精確estimation官方score |
|---|---:|---:|
| 0 | 0 | 100 |
| 2 | 0 | 100 |
| 3 | 0 | 100 |

三題原版0／精確100。index2當步精確後仍argmax錯，但correct probability與margin大幅改善；從prefill開始精確生成則答對，可能涉及更早step的hidden-state／KV/history差異。本輪未進一步隔離哪個前序step，不把完整生成改善等同當步單一token已翻轉。

## 驗證、失敗與限制

- 第一次只恢復context／steady並固定cI，approximate control logits仍有max abs0.21875，未通過gate，沒有採用該次結果。原因未確認；不能由此認定upstream bug／race或LRU造成。加入actual execution-buffer tape後，三題baseline control logits max abs皆0，且control Q一致。失敗log保留case_0_setup_control_failure.log。
- exact kernel真KV路徑另以兩組不等長KV heads（小序列與約8K）對照相同BF16輸入的FP32 softmax，output relative L2<1%、LSE max abs<.01；raw kernel_verification.json保存實值。這驗證mask／GQA與numerical實作，不是品質或性能benchmark。
- 三題status／prompt身份／generated decode／官方scorer重算／EOS／fixed-zone control／finite／sources核對；每題一次fresh index，不保證其他index或requests均相同。
- 核心結果：在這三個已知失敗案例中，estimation approximation會實質改變答案token概率；單步固定分區2/3翻轉正確、完整精確生成3/3通過。不是單纯把更多dropped KV加入計算。這支持進一步驗證「精確UUID複製對cluster estimation的敏感性」。
- 仍有部分答案KV dropped；結果不證明selection無影響，也不證明所有錯誤只由estimation造成。這是本3B local port、已選失敗題，不代表RetroInfer論文所有模型或RULER全題型的通用弱點。
- 精確estimation需要更多真KV、H2D與計算，診斷版不得當免費修正、正式TPOT或VRAM對照；未量測其性能。

## 下一決策

優先在未用於診斷的剩餘官方UUID題確認原版／本研究品質差距，並以新的失敗和成功案例重複本因果對照。若要提出方法，再研究何種廉價訊號能偵測有害estimation；目前尚未提出或訓練predictor。

## Artifacts與公開

results/uuid_estimation_diag_20261007/{cases.json,source_manifest.json,verification.json,kernel_verification.json,case_*.log,<case>/result.json}；scripts/uuid_estimation_diag_20261007/{prepare_cases.py,case.py,run.sh,verify_kernel.py,analyze.py}。依賴source見manifest以及既有benchmark_retroinfer_ruler_case.py／external RetroInfer checkout。

應同步reports/1014/uuid_estimation_diagnostic_2026-10-07.md與上述5scripts；rawJSON/log／全文prompt與predictions／KV tensors／weights／externalcheckout未納入。Production source/defaults未修改，未commit/push。
