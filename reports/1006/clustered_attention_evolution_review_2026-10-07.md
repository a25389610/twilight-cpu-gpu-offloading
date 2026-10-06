# Clustered Attention 深讀：後續進展與 RetroInfer 的剩餘誤差條件

日期：2026-10-07。類型：原文／supplement／新文獻／本地source查核及數學例子；沒有新LLM／GPU benchmark。研究層級2–5。使用者疑問：2020基本聚類方法的失敗能否用來判斷2025／2026的RetroInfer？

## 結論與前輪解讀修正

不能把basic clustered attention的SQuAD崩潰直接套到RetroInfer。2020同篇的improved版本就已透過精確top-k dot products修正大部分問題。後續研究提升了重要entry偵測、normalization／output error保證、動態Key retrieval；RetroInfer的steady＋retrieval精確區與estimation尾部補償也有合理性。

應研究「estimation cluster內的query-specific權重差異是否與Value差異共同造成有害摘要誤差」，不是預設近似都不好。成功與失敗案例皆需分析。一般自然語言的普遍缺陷仍未確認。

## 1. Fast Transformers with Clustered Attention 完整核心方法

Apoorv Vyas／Angelos Katharopoulos／François Fleuret，NeurIPS2020。[main](https://proceedings.neurips.cc/paper_files/paper/2020/file/f6a8dd1c954c8506aadc764cc32b895e-Paper.pdf)、[supplement](https://proceedings.neurips.cc/paper_files/paper/2020/file/f6a8dd1c954c8506aadc764cc32b895e-Supplemental.pdf)。本轮取得main10頁／supp9頁，讀取算法、證明、實驗設定與補充分析；沒有執行作者repo。

### 背景與basic

原本N個Query各自對N個Keys做attention，成本quadratic。它將Query分成C群，群中心Qc對全部原始Keys計算softmax與加權Value，再把同一個attention output分給群內Query。不是把Keys或Values平均；Keys／Values仍原始。固定C時成本O(NC(Dk+Dv))，hash與clustering也有成本。

使用random-projection sign／LSH，把Query轉bit signatures，在Hamming space做K-means減少clustering開銷；centroids由Query向量計算。多heads与residual使同群Query的完整layer representations不必相同，不意味後續layer只能一直合併cluster。

Proposition1以Query-to-centroid距離與K spectral norm界定attention weight L2差；標準scaled logits另含1/√d。不是attention output／最終QA score的保證；乘V、後續network與argmax仍可能放大影響。

### improved的精確含義

對每個Query cluster由centroid attention選top-k Keys（主實驗通常k32）。每個原Query重新計算對這些Keys的dot products，得到top-k內相對softmax，再乘回centroid估計的top-k總mass mhat。剩下Keys仍沿用centroid attention。注意：精確的是top-k的原Query-Key dot products與條件相對權重，top-k對全部Keys的總mass仍是近似；不是整個attention已精確。

Supplement §2將true top-k mass m與估計mhat比較：重新分配後top-k的L1誤差=|m−mhat|，不大於basic版本在同集合的L1誤差，因此整列L1誤差不增加。這個證明沒有保證zero error、output token或task accuracy一定改善。

### 實驗的完整取捨

- WSJ／Switchboard ASR包含從頭訓練與替換attention的cross-evaluation；應分開trained-with與evaluated-with，不能pool。
- RoBERTa approximation：GLUEmax128、SQuADmax384、25clusters；SQuAD F1 .904／.006／.876（Full／basic／improved）。RTE .682／.498／.704。短序列Full反而比clustered variants快。
- Supplement masked-copy：4layer6heads、每head32dim、輸入64–512tokens、15/30/60/100clusters。Improved在測試組合全部完成任務；basic少clusters長序列時下降。這是「補回精確局部運算有效」的反證，不能只摘basic failure。
- WSJ oracle-top32-only比improved差，支持尾部近似可能比直接丟棄尾部好。

## 2. 較新研究確實有進步，但保證仍有範圍

這些是相關路線，未宣稱全部為Clustered Attention的直接後代。

### KDEformer — ICML2023

[原文](https://proceedings.mlr.press/v202/zandieh23a/zandieh23a.pdf)。估計softmax denominator並採樣matrix product，提供attention output的spectral norm誤差保證；practical版本用LSH抽出heavy entries以降低residual stable rank。比只看attention matrix entries的誤差更接近真正輸出，但不等於整個LLM任務保證。不同error tolerance／rank／input條件影響所需成本。

### HyperAttention — ICLR2024

[原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/ab5aa940590399350401c57cdf52ce78-Paper-Conference.pdf)。LSH找heavy entries，剩餘部分sampling，支援causal masking；理論快算法依賴特定column norms與heavy-entry移除後row-norm ratios，practical實作簡化sampling。不能把spectral保證當任意固定budget沒有品質損失。

§4.1 Table1 pretrained chatglm3-6b-32k逐步替換最後layers、無finetuning，作者報告QA比summary／code更敏感：Full0layers→all28layers時single-QA96.07→53.30，summary60.29→60.45。表中是作者跨dataset category彙整metric，並非0–100標準accuracy；其他類別有>100，不能寫成96.07%答對率。這是新的自然語言task sensitivity證據，仍不是RetroInfer測試。其prefill attention改動不同於本localport的decode estimation。

### Squeezed Attention — ACL2025（arXiv2024）

[正式ACL原文](https://aclanthology.org/2025.acl-long.1568.pdf)。對fixed-context Keys聚類，centroids只做query-aware selection，被選cluster展開真KV做sparse attention；不是用summedV作尾部estimation。Hierarchical indexing保留fine clusters同时減少lookup成本。

AppendixI／Table7：LWM-Text-Chat-1M，LongBench的TREC／2WikiMQA／MultifieldQA subset平均score：Full43.07；90%pruning時1%centroids19.55、2%42.19、5%42.97。Cluster太粗的問題2025仍可觀察，但這主要是selection resolution，不是estimation因果。§8明言可達sparsity取決context／task且hyperparameters需設定，沒有普遍保證。不能由此宣稱cluster數增加必然使本port更好。

## 3. RetroInfer 已避免什麼，仍剩什麼

本次额外查最新版[arXivv3，2026-04-27／PVLDB19(5)](https://arxiv.org/html/2505.02922v3)，不只依2025v1。§4.2 Eq2–4仍為centroid＋cluster size＋summedV與Jensen lower bound。沒有在這些公式看到normalizedoutput small-error／final-token一致性的保證。原文實驗支持所測model/task的effectiveness，应保留。

| 設計 | 2020basic／improved | RetroInfer本地已核source |
|---|---|---|
| 分群對象 | Queries | Keys（centering＋segmented clustering） |
| 真正Query是否保留 | basic用centroid；improved在top-k補原Query | 每step用真正Query |
| 精確重要資訊 | improved top-k內精確dot products | steady／retrieval真KV attention |
| 其他資訊 | centroid-query對原KV的attention | estimation同cluster共用權重＋sumV；本localbudget還有dropped |
| 誤差關注 | Query替換／top-k mass | 群內token權重差與Key-Value配對丟失 |

本localsource cache_hub/retroinfer_cache.py：~542centering、~554存centroids/value_sum/cluster_size；~754sparse_attention，以head-group的softmax分數sum後topk做共同cluster ranking，~772estimation weighted_flash、~802exact retrieval＋steady merge。source hash存manifest。sum與average對固定group_size排序等價。論文v3描述使用normalized inner products平均共同排名；本輪依實際source描述，不把二者細節混稱完全相同。

保護效果：重要cluster真KV取回、steady保留sink／recent、centering改善clustering、estimation補回原本可能丟棄尾部。不能说只是2020basic的重包裝。新的index或kernel速度改善也不自动恢復estimation被摘要掉的細節。

## 4. 收斂的數學與機制假設

令一個estimation cluster內s_i=q·k_i/√d、C=mean(k)、Vbar=mean(v)。原numerator為Σ exp(s_i)v_i，approx為exp(mean(s))Σv_i。可以分解成：

N_true−N_approx = (Z_true−Z_approx) Vbar + m Cov(exp(s),v)。

這是本輪直接代數推導（population covariance、vector v），不是引用新論文定理。兩種誤差：

1. cluster總mass誤差：Jensen只給Zapprox≤Ztrue，不保證gap小。
2. 群內配對誤差：即使校正mass，缺少exp(s)與Value的關聯仍可能錯。

如果群內logits幾乎一致，或Values幾乎一致，或者該cluster實際影響很小，摘要可很準／對答案無實質影響。Key空間平均距離小並非任意Query方向spread小的充分品質判斷；需測s_i的query-specific spread，例如variance/range，不只文字similarity。

最值得測的三條件：estimation群內權重不均＋承載不同資訊的Values＋足夠的真實attention貢獻。Next-token margin可影響最後是否翻轉；大output error也未必使答案錯，需品質驗證。

### 相同摘要、不同正確輸出的存在性檢查

scalar cluster-only例子：q=1、K=[4,0]。V=[1,-1]與V=[-1,1]皆有同C=2、sumV=0、size2；summary-based output皆0，true outputs±tanh(2)=±.96402758。只靠這三項summary无法辨別兩個Key-Value配對。腳本以Python double驗證。

此為數學構造，非模型KV、非真實dataset、非「RetroInfer一定會遇到」證據；如cluster被取回，或占比小，問題可避免。Full cache在CPU仍保留，資訊只在未展開的estimation path不可見，不是全系統永久丟失。

## 5. 新候選：GQA共同排名（次優先、尚未驗證）

本3B每KVhead對應3Queryheads。若某cluster只有一個head非常需要，其他head不需要，共同排名可能讓它落入estimation；單head實际尖峰會被平均排名掩蓋。這是source導出的假設，不是確認原因，也不是一定发生。應先分離estimation機制，再考慮改selection；否则两者混淆。

## 6. 與本地已有證據與下一決策

已核reports/uuid_estimation_diagnostic_2026-10-07.md：已選3個失敗case，固定buffers／zones單步2/3argmax修正，支持本localport確實存在estimation敏感案例；不是一般自然語言普遍失敗。完整生成3/3恢復不能当全部step固定partition因果。LongMemEval10明確正確三方法同4/10，仍未看到穩定自然語言差距。

研究問題更新為：哪些estimation clusters的摘要误差會使上下文事實配對不可靠？先測機制條件，再討論能否低成本偵測；不先做predictor或稱novelty。

最小後續（尚未跑）：

- 優先單一document grounded extraction／QA，先以3B短contextFull確認能力；可從SWDE／FDA／LongBench單文件QA尋找，數據／格式先核對。避免因多跳／日期算術讓Full都錯。
- 固定cohort與評分，實際自然長度與32Kderived分開；完整報所有題与Full通過子集，保留成功／反向案例。
- 比較Full／本研究／RetroInfer品質。再對有差異及成功case做同index/history/partition的exact-estimation，查spread、mass误差、weightedValue差與token outcome。
- 若exact-estimation恢复、且新的成功失敗樣本符合機制條件，才能擴大／提出改善；若不恢复或無穩定差距，就接受近似在該設定有效，回到selection／模型能力等解釋。

## Artifacts與公開

results/clustered_attention_review_20261007/ 保存5份primary PDFs、text、source_manifest.json（來源URL＋SHA）。scripts/clustered_attention_review_20261007/summary_counterexample.py為數學check，不含模型或題目資料。

準備public mirror：reports/1014/clustered_attention_evolution_review_2026-10-07.md＋scripts/clustered_attention_review_20261007/summary_counterexample.py。RawPDF／text／JSON、KV、prompt、weights与externalrepo不納入；source未修改。Canonical週報同turn更新；未commit/push。文獻与数學查核，不作新TPOT／VRAM／benchmark claims。
