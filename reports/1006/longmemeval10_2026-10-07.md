# LongMemEval衍生32K：10題三方法品質pilot

日期2026-10-07。目的：把UUID近似診斷延伸到自然語言資訊更新／時間推理，確認是否有值得追查的品質差距；未預设RetroInfer較差。使用者只授權10題，本輪共30fresh串行推論。

## 資料與建構

- 原始官方cleaned S資料xiaowu0162/longmemeval-cleaned，revision98d7416c24c778c2fee6e6f3006e7a073259d48f，原檔500題。每類排除abstention後，依SHA256(seed20261007+question_id)排序取前5，推論前固定；沒有依各方法輸出換題。
- 保留官方answer_session_ids的全部完整對話與原日期，其他完整session依固定hash順序加入直到prompt <=32256tokens，再依日期排序。question／reference／question_date不變；未截掉證據turn，未改寫、翻譯或插入answer；evidence labels只用來縮短資料，不提供給模型runtime／selector。
- 這是使用oracle evidence labels建構的32K衍生診斷資料，不是官方約115K LongMemEval_S；縮短會改變干擾、evidence位置與難度，不能報為官方S benchmark分數或一般自然語言代表性。
- 相同Llama-3.2-3B-Instruct、BF16、Batch1、同一prompt ids/hash、greedy/EOS，上限256。實際10prompt 32003–32255tokens。Custom direct-reading prompt含timestamped USER／ASSISTANT history与question，外層canonical Llama3 user／assistant模板。
- Full：既有canonical GroupedPinnedSlabOffloadedCache＋mp_headinfer；本研究：B0=4096、p=.90、最新slim fused；RetroInfer：local portretrieval=.018／estimation=.232／cache=.05，非上游其他設定。各題fresh process，輪替三方法順序，GPU串行；runtime並不同，不是單一機制ablation。

## 評分口徑

- 官方evaluate_qa.py的category rules／reference作依據，腳本commit/hash見evaluation_provenance.json。未呼叫付費API或官方GPT-4o judge，沒有獨立人類評審。
- Codex單一reviewer先閱讀method-anonymized responses，固定blind_ratings.json後才開blind_mapping.json。這是輔助語意判讀，非官方benchmark分數／正式獨立blind study。完整判讀與理由保留。
- 時間類約3.29weeks對reference3weeks，本輪暫接受為約3週並標borderline；需獨立評分確認。原回答內日期減法有誤，不隱藏其推理問題。Full及本研究各有此borderline，RetroInfer同題0.71weeks，與3weeks差距超過一單位，不接受。
- 另一題問誰先成為父母，三方法主結論皆Rachel（reference Alex），即使正文提Alex／January，仍判主答案錯；不能以答案名字出現就評對。
- 原runner內RULER兼容string-match欄位只是執行器產物，不作本次LongMemEval品質分數，避免誤讀。

## 結果（preliminary）

| 方法 | 暫定語意判讀 | 明確正確 | 待複核邊界題 | knowledge update（5題） | temporal（5題） |
|---|---:|---:|---:|---:|---:|
| Full attention | 5/10（50%） | 4/10 | 1 | 4/5 | 1/5 |
| 本研究 | 5/10（50%） | 4/10 | 1 | 4/5 | 1/5 |
| RetroInfer | 4/10（40%） | 4/10 | 0 | 4/5 | 0/5 |

严格按明確正確計，三者均4/10。暫定Full與本研究對／RetroInfer錯只有1題：['gpt4_61e13b3c']，正是邊界評分題；反向0題。此結果不支持擴大宣稱一般自然語言下estimation的穩定品質弱點。

## 逐題結果

| ID | 題目重點 | Full | 本研究 | RetroInfer |
|---|---|---|---|---|
| 01493427 | 新增明信片25張 | 對 | 對 | 對 |
| 031748ae | 開始4位／現在5位工程師 | 對 | 對 | 對 |
| c6853660 | 咖啡杯數限制增加 | 對 | 對 | 對 |
| 1cea1afa | 目前600 followers | 對 | 對 | 對 |
| 10e09553 | 較早釣魚7尾（均答9） | 錯 | 錯 | 錯 |
| gpt4_fe651585 | Alex先成為父母（均答Rachel） | 錯 | 錯 | 錯 |
| 0db4c65d | 讀書至活動18/19天 | 錯 | 錯 | 錯 |
| gpt4_61e13b3c | 兩次活動相隔3週 | 待複核（暫對） | 待複核（暫對） | 錯 |
| a3045048 | 買禮物至生日7/8天 | 錯 | 錯 | 錯 |
| eac54add | 四週前與首位客戶簽約 | 錯 | 錯 | 錯 |

## 實際耗時與執行驗證

- full：10次process合計4.93分鐘，平均29.60秒。
- twilight：10次process合計4.92分鐘，平均29.54秒。
- retroinfer：10次process合計2.84分鐘，平均17.04秒。

30次推論process wall合計 **12.70分鐘**，包含import／模型載入／prefill／decode，不是formal TPOT，不用此表對steady decode排序。下載277MB來源、資料整理及語意判讀另計。30次皆正常退出、EOS結束，0次256token截斷。

- 30outputs重新decode、request/hash／token count／budget／方法設定／Full finite／本研究checkpoint finite／RetroInfer final finite與source hashes核對；10題完整evidence sessions與日期、時間排序及question/ref對齊驗證。結果raw與審核artifact已保存。

## 結論與下一決策

本批knowledge update三者同為4/5，temporal四題三者共同失敗，剩一題評分邊界。這顯示本3B、prompt與衍生資料設定下存在共同品質限制；不能由共同失敗歸因estimation，也未隔離模型能力、prompt或長context效應。10題太少且只有一次，沒有自然語言exact-estimation ablation。

應先由獨立review／官方judge確認邊界題，並用相同題目的evidence-only短context Full對照檢查模型是否能完成任務；短context的改善只能協助分離長context效應，仍不能單獨證明estimation原因。這些下一輪未自動執行。不因未找到預期差距而換題，保留本負面／不確定pilot。

## Artifacts與公開

results/longmemeval10_20261007/{dataset_revision.json,evaluation_provenance.json,cohort_manifest.json,run_manifest.json,artifact_verification.json,blind_responses.json,blind_mapping.json,blind_ratings.json,summary.json,cohort/,runs/}。下載來源與full prompt在本地，不納入public。

應同步reports/1014/longmemeval10_2026-10-07.md＋scripts/longmemeval10_20261007/{prepare.py,run.py,blind_review.py,report.py}；既有Full／RetroInfer／slim fused runners為依賴。RawJSON/log/PT、完整prompt／predictions／dataset／weights／externalcheckout未納入。Production source/defaults未改，未commit/push。
