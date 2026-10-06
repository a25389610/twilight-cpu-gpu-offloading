# 近似 attention 的限制：文獻查核與 RetroInfer 驗證方向

日期：2026-10-07。類型：針對研究問題的文獻整理，非新 GPU benchmark，也非窮盡式 systematic review。研究層級4–5：候選不足與所需證據。

## 問題與結論

教授希望釐清近似 attention 的品質風險。近似並非 RetroInfer 獨創；相關研究已有 query clustering、kernel／低秩近似、linear attention、sparse＋low-rank 混合等。不同方法的限制不能直接互相套用。FlashAttention 的數學運算是 exact attention（仍有浮點數誤差），不應歸為本文的演算法近似。

目前最接近 RetroInfer 的待驗證問題是：在 estimation cluster 裡，當 query 對個別 token 的權重差異很大，且那些 token 的 Value 承載不同答案時，centroid＋summed Value 是否損失必要辨識能力？此為假設，不由文字相似度或近似比例直接推出。

## 優先文獻與證據邊界

### 1. Fast Transformers with Clustered Attention — NeurIPS 2020

作者 Apoorv Vyas、Angelos Katharopoulos、François Fleuret。原文 §3、§4.3／Table4：聚類的是 Query，共用 centroid attention；improved 版本重新計算每個 Query 與 top-k Keys 的精確 dot product。預訓練 RoBERTa 推論近似，GLUE maximum128、SQuAD384、25clusters：SQuAD F1 Full .904、clustered .006、improved .876；RTE accuracy .682／.498／.704。這是明確的自然語言品質損失案例，也顯示修正關鍵 token 的精確計算能大幅恢復品質。

限制：不是 RetroInfer 的 Key clustering＋retrieval／estimation zoning，數字不能當 RetroInfer 預測。文章摘要較樂觀，應以 Table4 分清 basic／improved。

來源：[原論文](https://proceedings.neurips.cc/paper_files/paper/2020/file/f6a8dd1c954c8506aadc764cc32b895e-Paper.pdf)、[作者專案](https://clustered-transformers.github.io/)。

### 2. Scatterbrain: Unifying Sparse and Low-rank Attention — NeurIPS 2021

作者 Beidi Chen、Tri Dao、Eric Winsor、Zhao Song、Atri Rudra、Christopher Ré。§3.1：sparse 與 low-rank 的適合範圍隨 attention temperature／entropy 不同；低 entropy 的尖峰 attention 更適合 sparse，高 entropy 更適合 low-rank。結合兩者可降低近似誤差。

對 RetroInfer 啟示：應查真正尖峰 token 是否留在 retrieval，還是落在 estimation。混合精確／近似本身有合理性，不是天然缺陷；錯置重要 token 才是候選風險。此對應為本研究推論。

來源：[NeurIPS](https://papers.nips.cc/paper_files/paper/2021/hash/9185f3ec501c674c7c788464a36e7fb3-Abstract.html)。原文 code 連結 https://github.com/HazyResearch/scatterbrain 本次開啟轉址到 HazyResearch/fly，尚未驗證目前 repo 是否可完整重現原論文。

### 3. The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry — ICLR 2024

作者 Michael Zhang、Kush Bhatia、Hermann Kumbong、Christopher Ré。指出先前 linear attention 缺少與表現相關的低 entropy／spiky weights 及 dot-product monotonicity，並以可學習 feature maps 模仿 softmax。

對 RetroInfer 啟示：檢查 cluster 內尖峰權重、token ordering 與 attention output error。RetroInfer 並非 Hedgehog 所分析的 kernel feature map，不能直接說它也違反相同性質。作者後續 LoLCATs repo 支援 Hedgehog feature map，不在此宣稱等同原始完整 artifact。

來源：[ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ebba182cb97864368fdb6ae00773a5e4-Abstract-Conference.html)、[後續作者程式](https://github.com/HazyResearch/lolcats)。

### 4. The Devil in Linear Transformer — EMNLP 2022

作者 Zhen Qin 等。指出 kernel-based linear attention 的 unbounded gradients 與 attention dilution；後者是長序列權重過度分散、忽略鄰近結構。

限制：梯度問題關於訓練，不能解釋固定權重 RetroInfer 的 decode。Attention dilution 也須測量，不可直接套用。適合作為誤差分析背景。

來源：[ACL Anthology](https://aclanthology.org/2022.emnlp-main.473/)、[作者程式](https://github.com/OpenNLPLab/Transnormer)。

### 5. Zoology: Measuring and Improving Recall in Efficient Language Models — ICLR 2024

作者 Simran Arora、Sabri Eyuboglu 等。17種 attention／gated-convolution 模型的分析中，82% 的 perplexity gap 可由 associative recall 解釋；提出 MQAR，並提供真實語料 recall 分析。

價值：支持分開量測「依靠上下文的事實配對」與一般語言流暢度；不是只有 UUID 才算 recall。限制：核心比較是模型架構，不是同一 Llama 插入 RetroInfer estimation；不能拿架構下界證明本 port 品質差。

來源：[ICLR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html)、[作者程式](https://github.com/HazyResearch/zoology)。

### 6. Simple linear attention language models balance the recall-throughput tradeoff（BASED）— ICML 2024

作者 Simran Arora、Sabri Eyuboglu、Michael Zhang 等。分析 state size／recall tradeoff，結合 linear＋sliding-window attention。原文 Table1 與 AppendixE.3 使用 SWDE 網頁属性抽取、FDA 文件欄位抽取與 SQuAD 閱讀理解，說明普通 perplexity／common-sense averages 可遮蔽 recall 差異。

這是最有用的真實任務來源：商品／學校／電影 HTML 裡抽指定欄位；真實文件裡找指定屬性。較少依賴時間算術，可減少目前 LongMemEval temporal 的能力混淆。原文有 small-model prompt adaptations（FDA1920token chunks、SQuAD經 GPT4 改寫），不能直接稱其設定為32K官方benchmark。

來源：[ICML proceedings](https://proceedings.mlr.press/v235/arora24a.html)、[原文v2](https://arxiv.org/html/2402.18668v2)、[作者程式及 dataset links](https://github.com/HazyResearch/based)。

### 7. Fast Attention Requires Bounded Entries — NeurIPS 2023

作者 Josh Alman、Zhao Song。Q/K/V entry bound、d=O(log n)、1/poly(n) additive accuracy 等特定設定下，給出快近似算法與依 SETH 的 worst-case 下界。

價值：近似保證需要輸入與誤差假設。限制：分析整個 n×n attention problem；單步 decode 是單 Query，還有 index preprocessing，不能拿 quadratic lower bound 宣稱 RetroInfer 單步必定失敗。不是一般自然語言 benchmark。

來源：[NeurIPS](https://papers.neurips.cc/paper_files/paper/2023/hash/c72861451d6fa9dfa64831102b9bb71a-Abstract-Conference.html)。

補充理論：[Fundamental Limitations on Subquadratic Alternatives to Transformers](https://openreview.net/pdf?id=T2d0geb6y0)，Josh Alman／Hantao Yu，ICLR2025：條件性 worst-case document-pair similarity 下界，不保證現成3B模型能解任務，不直接適用單 Query decode。

## RetroInfer 的誤差機制：由公式推導，非新 benchmark

[RetroInfer 原文 §4 Eq2–4](https://arxiv.org/html/2505.02922v1) estimation 用 centroid Key、summed Value、cluster size。令 cluster 內 s_i=q·k_i/√d：

- 真實 cluster numerator：Σ exp(s_i) v_i。
- 近似 numerator：exp(mean(s)) Σ v_i。
- 真實 mass：Σ exp(s_i)；近似 mass：m exp(mean(s))。

Jensen 保證近似 mass 不大於真實 mass，不保證相對誤差小，也不保證 normalized attention output／生成 token 相同。就算 mass 已校正，若 exp(s_i) 與 v_i 有關聯，共用平均 Value 仍會損失逐 token 加權差異。

數學示例（cluster-only、scalar V）：logits[4,0]、V[1,-1]，真實輸出 tanh(2)≈.964，centroid＋sumV 輸出0。只是存在性示例；沒有證明這種 cluster 會在指定自然語言資料出現。

本研究也有 token selection／drop，僅 selected KV attention 使用原始KV，不是等同 Full attention。

## 與本地觀察對齊

已讀 reports/uuid_estimation_diagnostic_2026-10-07.md：3個已選失敗案例的 taped-buffer 固定分區 single-step 對照2/3修正 argmax；同index完整生成3/3修正，但 history／selection會演變，不能等同全程固定分區因果。已讀 reports/longmemeval10_2026-10-07.md：明確正確三方法均4/10，一題Full／本研究邊界評分；没有一般自然語言弱點的清楚新證據。既有觀察不因文獻而改寫。

## 下一個關鍵決策／最小驗證

優先檢查真實文件欄位抽取（BASED的SWDE／FDA），而非再加時間推理難度。先確認來源、使用條款／官方資料、完整證據與3B Full短context能力，固定題目與評分；再做配對長context3方法。

原始自然長度與32K加干擾的衍生版分開報告。以Full correctness作能力分層而非偷偷刪失敗題；保存全題並報Full通過subset。追查Full對Retro錯案例，固定index／history／zones對比exact-estimation，成功案例也取對照。至少查 estimation 内logit spread、Value差異、真實／近似mass、局部output error及最終品質。觀察無差距也接受。

本輪只完成文獻查核與待驗證方向，未啟動GPU、修改runtime、宣稱新的品質／TPOT／VRAM結果。

## Artifact／公開範圍

本文件與 canonical reports/weekly/2026-10-07.md 為文字紀錄。Public mirror只準備 reports/1014/approximate_attention_limits_survey_2026-10-07.md；本輪沒有新增 scripts／source，沒有納入raw datasets／JSON／log／prompt／KV／weights，未commit或push。
