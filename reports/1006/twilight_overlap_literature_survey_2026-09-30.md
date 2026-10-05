# Twilight CPU–GPU sparse KV 路徑：高重疊文獻調查

調查日期：2026-09-30（Asia/Taipei）  
文件性質：原論文／官方程式庫的定位調查；**沒有重現任何論文數字，也沒有完成 novelty 證明**。

## 1. 比較對象與判讀方式

本地目前首選實驗路徑（詳見 `TWILIGHT_CURRENT_PATHS.md`）：Llama-3.2-3B-Instruct、Batch 1、32K、RTX 5060 Ti 16GB；完整歷史 KV 位於 CPU pinned slab；每步用 Quest 候選加 Twilight Top-p 選出當步集合，GPU 保留前一步 selected KV，當步 hit 重用、miss 透過 GPU mapped host memory 讀取。另有融合 INT4 prepare/QK 的 approximate 路徑。已測 matched 3 requests × 2 rounds 的正式 D2–D32 TPOT 為 68.292→55.042 ms/token；這不是與任何論文同條件的直接比較。

以下「重疊」分三軸：**S**＝當步選重要 KV 的方法、**R**＝跨 token 重用／GPU residency、**O**＝CPU–GPU offloading／recall。請勿將 R 的 LRU／previous-token 命中誤當 S 的 attention importance。也須區分「當步重選後，舊 KV 只作搬運 hit」與「當步直接沿用舊 selected set」；後者改變 attention 使用集合。

Code 狀態只表示本輪是否找到論文連結或可信官方公開程式庫；「未確認」**不等於不存在**，「有」也**不等於**能在本地 3B／16GB 環境直接跑。優先度依研究主題相似度排序，非論文品質排名。

## 2. 最需要先處理的論文

| 優先 | 論文與已確認重疊 | 和本地方法的關鍵差異 | Source code／本地 baseline 判斷 |
|---|---|---|---|
| 1 | [LiteCache（原名 CLO；同一 arXiv ID）](https://arxiv.org/html/2511.14510)：S/R/O。利用相鄰 query 相似度，決定是否沿用 GPU cached top-k KV；GPU-centric 同步、零拷貝／預取及 CUDA Graph。 | query 相似即直接用舊集合，miss 則整組重取；本地每步照當步 Twilight selector 選出集合，再逐項 hit/miss，並用 direct INT4 QK。**這是目前最直接的主題碰撞之一。** | 論文 [匿名程式連結](https://anonymous.4open.science/r/LiteCache-888D)；舊 CLO [公開 repo](https://github.com/CommediaJW/CLO)。兩者是否對應同一最新版 source 尚未逐 commit 核對；不能算兩篇論文。原文測 Llama3-8B／Qwen2.5-14B、A40／H100；移植本地需改模型及 backend。 |
| 2 | [FlexiCache](https://arxiv.org/html/2511.00868)：S/R/O。Quest 類 min/max page score、top-K、host offload；stable heads 重排週期較長，僅傳新進 top-K pages。 | stable heads 中間步直接沿用舊 top-K；unstable heads 的完整 KV 留 GPU。本地完整 historical KV 在 CPU，每步選擇，GPU previous-token resident 是 data-movement policy。 | [官方碼](https://github.com/NazmulTakbir/FlexiCache)；README 使用 H100 94GB、至少 256GB RAM、vLLM 與離線 head profile，直接在 16GB 3B 跑的可行性未驗證。 |
| 3 | [HiSparse](https://arxiv.org/html/2608.07009)：R/O，S 接外部 indexer，包括 Quest。完整 KV 在 CPU、固定 GPU buffer、當步選擇的 hit/miss、LRU、host fetch；不改 indexer 輸出。 | 與本地 **exact current-step hit/miss** 結構最像；其主貢獻是 bounded cache + fused resolve + 部分模型的 exact inter-layer prefetch。本地是 previous-step snapshot／GPU mapped host read／Twilight Top-p+INT4，不是一般 LRU。 | [SGLang 上游實作與文件](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hisparse_guide.mdx)；論文含 Quest 實驗，但文件現成啟動路徑偏 native sparse model，**不能假設換成 Llama-3.2-3B 就可直接當 Quest baseline**。 |
| 4 | [RetroInfer](https://arxiv.org/html/2505.02922)：S/R/O。自己設計 wave index 與 GPU wave buffer；報告相鄰步 critical KV 的時間局部性及 cache hit。 | 它是向量檢索式選擇與 cache／CPU-GPU 執行；本地採 Quest→Twilight、previous-step set 與 GPU mapped misses，兩者 selection 語義不同。 | [官方碼](https://github.com/microsoft/RetrievalAttention)；README 測試環境 CUDA 12.4、獨立 FlashAttention/FlashInfer，移植成本中高。 |
| 5 | [AsyncTLS](https://arxiv.org/html/2604.07815)：S/R/O。先 block 篩選再 token 篩選；前一步 block 結果用於本步 token 選擇，current block 結果用於預取下一步，僅傳增量。 | 其 previous-step 資訊**參與當步 selection**，而本地前一步 KV 主要作 residency；本地另有 Twilight Top-p、direct INT4。 | 本輪未找到可核對的官方 code；無法列為立即可跑的 baseline。 |
| 6 | [FreeKV](https://arxiv.org/html/2505.13109)：S/R/O。Quest 類 page summary selection、完整 CPU KV、前一步 recalled set 供本步推測性 attention，query 不相似時做 head-wise correction；transfer overlap。 | 本地每步先確定當步集合，再判斷是否已在 GPU；FreeKV 多數步 attention 直接使用上一集合，存在獨立的品質取捨。 | [官方碼](https://github.com/sjtu-zhao-lab/FreeKV)；需編譯 FlashAttention、修改 FlashInfer／RAFT dependency，模型與 16GB 適配未驗證。 |
| 7 | [LocalKV](https://link.springer.com/chapter/10.1007/978-981-92-4805-6_6)：S/R/O。從上一步選中 token 的鄰域尋找當步重要 token，融合 sparse FlashAttention，async CPU→GPU prefetch。 | 它避免每步全域選擇，依賴 temporal + spatial locality；本地仍做全域 Quest/Twilight selection，舊集合只作 resident 命中。公開頁只提供摘要，不能把具體 cache 細節當已確認。 | 本輪未找到官方 code；Springer 頁標 APPT 2026、2026-08-24 online，正式引用年份顯示 2027。 |
| 8 | [SpeCache](https://arxiv.org/html/2503.16163)：S/R/O。高精度完整 KV 在 CPU，GPU 低 bit KV 副本估重要度，推測下一步需要的高精度 KV 並預取。 | GPU 持續放低 bit 的**全域副本**，本地以 Quest page metadata + INT4 candidate scoring，另保留前一步 selected high-precision KV；預取訊號不同。它已碰到「低 bit importance + full CPU KV + selective fetch」的組合。 | 本輪未找到論文指向的官方可用 repo；可作必要 Related Work，暫非即跑 baseline。 |
| 9 | [ParisKV](https://arxiv.org/html/2602.07721)：S/O。GPU 上 collision candidate search + **4-bit quantized inner-product reranking**；million-token 時完整 KV 可留 pinned CPU，UVA 按需取 final top-k，含 Batch 1 實驗。 | 這篇直接威脅「低 bit Key 直接評分 + CPU offload」的主張。它是 PolarANN/coarse collision + 4-bit RaBitQ 式 rerank、固定 top-k；本地是 Quest B0 + Twilight Top-p + resident previous set。 | [官方碼](https://github.com/amy-77/ParisKV)；有實際程式，但需先核對其 UVA 路徑與本地 3B/GQA／16GB。 |
| 10 | [SPIN](https://arxiv.org/html/2604.26837)：R/O，S 作外部 plugin。Index/Select 由稀疏方法提供，SPIN 管 Offload/Retrieve；GPU mandatory pages + 額外 buffer，以 bucketed LRU 留跨步 KV。 | 本地只留前一步 selected set，沒有動態 per-request buffer 與 bucketed LRU；SPIN 自身不提出 Twilight 式重要度公式。它的「previous-step only」已作 ablation，因此不能把此 baseline 說成未被研究。 | 本輪未找到作者官方 source repo；概念／ablation 必引，暫非即跑 baseline。 |
| 11 | [ECHO](https://www.usenix.org/conference/osdi26/presentation/liu-guangda)：R/O，S 來自 native sparse model。GPU graph-friendly cache 與利用 index score 可預測性做 lossless prefetch；將 recall 與 indexer 重疊。 | native sparse attention／DeepSeek-V3.2，非把 training-free Twilight 接上 dense Llama；研究主軸偏高併發 throughput。 | [官方 artifact](https://github.com/sjtu-zhao-lab/ECHO)／[公開鏡像](https://github.com/MachineLearningSystem/26OSDI-ECHO)；README 需 8×H20 96GB 與 1.5TB RAM，本地 16GB 不宜當直接 runnable baseline。 |
| 12 | [PulseInfer](https://arxiv.org/html/2609.34555)：S/O。完整或多數 KV 在 CPU，SoloHead selection 降低 per-head I/O 碎片，gather-scatter engine、adaptive admission、layer scheduling 處理變動的 recall 量。 | 著重服務排程與 I/O coalescing，非 previous-step resident set 或 Twilight INT4；但其「小傳輸與調度抵銷 sparse gain」問題與本地 TPOT 分解相近。 | 2026-09-28 初版；本輪未找到作者官方 code。 |
| 13 | [OasisKV](https://arxiv.org/abs/2608.08097)：S/R/O。draft/lookahead token 預測下步重要 block，從 CPU/remote memory 預取，保留 GPU sparse working set。 | 預測依賴 speculative decoding draft token，本地使用真實當步 query 重選與前一步 resident hit；實驗偏大規模 serving。 | 本輪未找到官方 code；先作 Related Work。 |

## 3. Selection／低 bit 的必要對照

| 論文 | 與本地重疊及差異 | Source code |
|---|---|---|
| [Twilight](https://arxiv.org/abs/2502.02770) | 基礎 selection：Quest 類候選後以量化 Key 再計分，Top-p 動態決定 budget；原方法本身不等於本地 CPU-resident + previous-token buffer + GPU mapped miss 系統。 | [官方碼](https://github.com/tsinghua-ideal/Twilight)。本地的 fused direct INT4 QK 不是可直接歸給官方 kernel 的量測。 |
| [Quest](https://arxiv.org/abs/2406.10774) | page min/max upper-bound + fixed budget 的 S 基線；原實作不是完整 CPU offload 系統。 | [官方碼](https://github.com/mit-han-lab/Quest)。 |
| [RaBitQCache](https://arxiv.org/abs/2606.31519) | binary/INT4 estimator + adaptive Top-p，是低精度 S 直接競爭者；未以完整 CPU KV + previous-token cache 作核心。 | [官方碼](https://github.com/Sakuraaa0/RaBitQCache)，README 含 accuracy 與 vLLM efficiency path；主要已驗證 Llama-3.1-8B/70B 等。 |
| [Double-P](https://arxiv.org/abs/2602.05191) | cluster-level Top-p 粗估，再做 token-level Top-p；直接競爭階層式 budget/selection，但不主打 CPU residency。 | 本輪未找到官方 code。 |
| [Self-Indexing KVCache](https://arxiv.org/abs/2603.14224) | 1-bit compressed Key 同時作 sparse retrieval index，與 direct low-bit score 的 S 問題重疊；非 previous-token offload 系統。 | 找到 [公開實驗 repo](https://github.com/LfieLike/selfindexingkv)，但論文頁未指向它；README 自述設定分散、attention kernel 僅實驗用途，官方歸屬／完整性仍需核對。 |
| [LServe](https://arxiv.org/abs/2502.14866) | reusable page selector、hierarchical paging，屬跨步 selection reuse；並非同一 CPU full KV offload 路徑。 | [官方 OmniServe 程式](https://github.com/mit-han-lab/omniserve)。 |

## 4. 更早的基礎／不同模型方向

| 論文 | 用途與差異 | Source code |
|---|---|---|
| [InfiniGen](https://arxiv.org/abs/2406.19707)（Twilight 前） | 由 partial 下一層 query/key 估重要 KV，從 CPU 預取；「selective host recall」的重要前作，不採本地 Twilight/previous-step hit。 | [官方碼](https://github.com/snu-comparch/InfiniGen)。 |
| [ArkVale](https://openreview.net/pdf?id=4oAt5L4lYe)（Twilight 前） | 可 recall 的 page-based GPU/CPU KV 管理與 page summary；是 retrieval offload 前作。 | [官方碼](https://github.com/pku-liang/ArkVale)。 |
| [ShadowKV](https://arxiv.org/abs/2410.21465)（Twilight 前） | GPU 低秩 Key／landmark 選取與 CPU Value offload；selection representation 不同。 | [官方碼](https://github.com/ByteDance-Seed/ShadowKV)。 |
| [NOSA/NOSI](https://arxiv.org/abs/2510.13602) | 以訓練後 native sparse attention 約束 CPU–GPU transfer，並提供 offload 系統；不能只把本地 Llama 換成其 selector 而維持相同模型。 | [官方碼及訓練後模型](https://github.com/thunlp/NOSA)。 |

其餘已檢視但目前較低優先的方向：HGCA/ScoutAttention 把一部分 attention 直接交給 CPU；FFD/LOCKS 主要是 selector 或 kernel；Text2JSON 研究偏品質失敗案例與 evaluation。這些可補 Related Work／quality gate，但不是當前 previous-token resident + current-step selector 的首輪直接系統 baseline。

## 5. 對「是否撞題」的目前結論

**已確認：** previous-token／adjacent-query locality、full CPU KV + GPU bounded cache、current-step hit/miss、低 bit Key importance，均已有各自或部分整合的 prior art。尤其 LiteCache、FlexiCache、HiSparse、RetroInfer、AsyncTLS、SpeCache、ParisKV 必須明確對照；不能聲稱單獨提出任一大概念。

**尚未確認：** 本地特定 execution path（每步 Twilight Top-p 重選、前一步 resident set 僅作 exact selected-set data-movement hit、GPU mapped host misses、direct INT4 QK、Batch 1／16GB）是否已有完全同機制、同目標的發表工作；本輪文獻搜尋無法證明 novelty 的不存在命題。也未驗證外部 source 在 Llama-3.2-3B／5060 Ti 上可運作。

**最快的公平比較路線：**

1. 先做 *method-level* 精確矩陣：每步是否重選？attention 是否使用當步選中集合？GPU resident budget/VRAM 相同嗎？miss 是 staged H2D 還是 mapped read？哪些模型／硬體可用？
2. *可重現 baseline* 優先看 FlexiCache、RetroInfer、FreeKV、ParisKV 程式，以及 HiSparse 的 SGLang integration；先證明能以同一模型、同一 32K requests、同一 Batch 1、同一品質門檻測量，再比較正式 D2–D32 TPOT。
3. 若移植成本過高，至少在本地系統做 matched ablation：無 cache、previous-token、同 GPU budget 的 FIFO/LRU；current-step selection vs query-similarity skip vs periodic rerank；`triton_prepare` exact 對照 vs direct INT4。報 TPOT、selected set、quality、H2D bytes、GPU mapped-read logical payload、VRAM、管理成本。
4. 不可拿論文的最高 throughput speedup／多 GPU server TPOT 直接與本地 55.042 ms/token 排名。
