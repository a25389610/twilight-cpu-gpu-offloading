# HeadInfer／Quest 目前研究方向

更新日期：2026-09-04  
狀態：後續研究討論、實驗設計與實作的 canonical direction

## 2026-09-05 Canonical Accuracy擴充至130題

Full、Quest與24個Twilight-GQA settings已擴至13 tasks × 10題，共130題／setting；目前正式
accuracy應引用此版本。32K Full／Quest Macro為76.6540／74.6412；Twilight
`.70/.75/.80/.85/.90/.95`為75.9104／76.0258／76.0258／76.1796／76.1796／76.0258。
所有32K Twilight仍略低於Full，`.85`是仍比Full快的settings中Macro最接近者；p與accuracy
仍不單調。完整證據見`reports/context_p_twilight_gqa_group_ruler130_2026-09-05.md`。

## 2026-09-04 Canonical Accuracy擴充至65題

Full、Quest與24個Twilight-GQA settings已由39題擴至13 tasks × 5題，共65題／setting；Timing
不重跑，仍使用獨立3-request正式結果。32K Full／Quest Macro為74.9232／73.0514；Twilight
`.70/.75/.80/.85/.90/.95`為73.0514／73.2822／73.2822／73.2822／73.5898／73.2822。
所有32K Twilight Macro仍低於Full，且p與accuracy不單調。正式accuracy討論應優先引用65題
結果；完整證據見`reports/context_p_twilight_gqa_group_ruler65_2026-09-04.md`。

## 2026-09-04 Twilight-GQA-Group正式Context × p Grid（canonical baseline）

已freeze pilot通過的CPU exact-union backend，完成4K／8K／16K／32K ×
`p=.70/.75/.80/.85/.90/.95`正式grid：72個新timing與936個新accuracy artifacts；Full與Quest
沿用相同frozen requests的既有正式artifacts。從現在起，此GQA-group grid是canonical Twilight
結果；舊per-Q head-wise grid只作historical comparison。

主要結論：Physical KV在4K／8K／16K／32K的`.70→.95`分別為
28.89→63.84%、20.99→57.19%、12.31→35.53%、6.24→18.73%。只有32K
`.70/.75/.80/.85`比Full快，沒有Twilight setting比Quest快。32K `.70/.75` TPOT為
201.08／196.80 ms，但Selection仍為99.22／94.16 ms，是最大單一diagnostic成本；Union為
15.32／17.49 ms。完整證據見
`reports/context_p_twilight_gqa_group_ruler39_2026-09-04.md`與
`results/context_p_twilight_gqa_group_ruler39_v1/summary.json`。

## 2026-09-04 Twilight GQA Group-wise Alignment Pilot（已完成）

Twilight NeurIPS 2025 Appendix B.2 audit確認：GQA應將同一KV Head所對應Query Heads的selected
tokens取union，shared K/V只載入一次，head-wise dynamism轉為group-wise dynamism。舊CPU-offload
Twilight為per-Q gather/H2D，並非paper-aligned canonical GQA execution。

已完成opt-in `Twilight-GQA-Group`與4點pilot。4K/.95與8K/.95的Physical KV從
141.42%／119.67%降至63.84%／57.19%；32K `.70/.75`從10.69%／12.68%降至6.24%／7.46%。
四點H2D降低41.18%–54.75%，TPOT降低4.82%–13.23%，但32K Selection仍約109–111 ms。
39題accuracy中4K／8K `.95`與head-wise完全相同；32K `.70` Macro −0.3846、`.75` −3.5897，
顯示group-wise語義不能假設quality bit-exact。

舊Context×p grid不再適合作為canonical Twilight結果，後續需要用GQA-group重跑；但依目前停止點
尚未啟動完整grid。完整證據見`reports/twilight_gqa_group_alignment_pilot_2026-09-04.md`。

## 2026-09-04 Context × p Head-wise歷史結果

Backend Freeze後的正式grid已完成：4K／8K／16K／32K，Twilight dynamic
`p=.70/.75/.80/.85/.90/.95`，另含各Context的Full與Quest fixed-5% baseline。Timing為每組
3個共同且與accuracy不重疊的requests；accuracy為13 RULER tasks × 3題，共39題。

核心證據是32K：Full、Quest、Twilight `.70/.75/.80/.85/.90/.95`的TPOT分別為
264.87、162.62、214.16、226.20、241.28、268.89、292.77、343.28 ms；Physical KV Transfer
分別為100%、17.90%、10.69%、12.68%、15.30%、18.87%、24.23%、33.47%。Twilight `.70`
雖比Quest少搬約40.3%的KV bytes，仍因Selection 109.66 ms（Quest 25.15 ms）而較慢。

目前結論：dynamic B1已能降低CPU gather/H2D payload，但Batch=1的Twilight Selection固定成本
抵銷大部分saving；短Context更明顯。32K `.75`在39題cohort得到76.4531 Macro RULER、27/39，
Full為74.4018、26/39，但逐題只有1 better／37 same／1 worse，不可宣稱普遍優於Full，也不可
由39題決定精確最佳p。下一個系統問題應優先是降低dynamic selection／synchronization成本。
完整證據見`reports/context_p_dynamic_twilight_ruler_39_2026-09-04.md`與
`results/context_p_ruler_39_v1/summary.json`。

## 2026-09-04 Phase 2 Backend Freeze（最高優先）

Phase 2已檢查Quest／Twilight layer-flat的CPU gather／packing。實驗性的layer-global
two-`index_select` path雖將CPU gather calls 448→56並消除224個index temporary，但在四個
32K requests、每個method共12組paired trials中，Quest有11/12較慢、Twilight有10/12較慢；
沒有發現可能造成>10% TPOT變化的generic glue-code artifact。因此不採用此候選，正式grid
凍結既有per-KV-head direct-to-final-pinned gather。

正式baseline固定為：

- Full TPOT／Accuracy：production pipelined CPU-offload path；
- Physical denominator：matched Full-flat actual K+V H2D artifact = 100%；
- Quest：fixed 5%、layer-batched selection、layer-flat ragged execution、無GQA union；
- Twilight：dynamic Top-p raw B1、layer-batched selection、layer-flat ragged execution、無GQA union。

Backend版本、runtime、source SHA-256與artifact定義保存於
`results/phase2_sparse_gather_stabilization_v1/backend_freeze_manifest.json`。workspace沒有Git
repository，因此無commit可記錄。後續正式Context × p grid已在此凍結backend上完成；grid
執行期間沒有修改backend。

## 2026-09-04 Phase 0／1規劃背景（由上方Freeze決策更新）

本階段暫停新增 Adaptive-K 公式，先回答老師指定的 `p × Context Length` 問題：在
Batch=1、full historical KV位於CPU的情況下，Twilight dynamic Top-p如何影響實際K/V
H2D payload、H2D／Attention latency、TPOT與RULER quality。

Phase 0原先規劃的方法為：

- `Full-flat`：CPU-resident Full KV，layer-level K/V H2D與Full Attention；Phase 2後只保留作
  physical H2D denominator，正式TPOT改用production pipelined Full；
- `Quest-flat`：canonical per-Query-Head fixed 5% Top-K，不做GQA union；
- `Twilight-dynamic-flat`：raw Top-p B1直接控制gather、H2D與Attention，不經matched allocator。

已完成的正式grid為Context `4K/8K/16K/32K`、Twilight `p=.70/.75/.80/.85/.90/.95`，另加每個
Context各一個Full與Quest baseline。正式`KV usage %`定義為同Context下實際sparse K+V
H2D payload bytes除以實際Full-flat payload bytes；per-Q logical visibility只作diagnostic。

開始grid前必須先完成三個backend的execution alignment與correctness parity，並將正式TPOT
與diagnostic profiling分開。本階段不做GQA union、Selection Frequency、shared selection、
CPU/GPU scheduler、block-size sweep、Predictor或chunked Prefill。下方2026-08-29 Adaptive-K
內容保留為研究歷史與後續候選，不是目前立即執行項目。

## 一句話定位

> 在 Batch 1～Small Batch 的資源受限 CPU–GPU 平台上，利用 Quest 已有的低成本
> Query-aware Page Score，設計 Extremely Lightweight 的 Head-wise Adaptive KV
> Block Selection，使減少的 KV transfer 真正轉化成 TPOT 改善，而不是被
> selection overhead 抵銷。

## 研究場景

- 主要模型：`meta-llama/Llama-3.2-3B-Instruct`
- 主要 context：32K，後續視可行性延伸至 64K 或更長
- 主要硬體：Consumer GPU、一般個人電腦／Edge CPU、有限 VRAM 與 CPU resources
- 主要 batch size：1；後續最多補 Batch 2／4
- 完整歷史 KV 主要駐留 CPU；Decode 時只把必要 KV 搬到 GPU
- 首要系統指標：TPOT；同時報告 quality、H2D/D2H bytes、VRAM、RSS 與額外 overhead

## 已停止作為預設主線的方向

`Adaptive Head-wise Offloading based on Lightweight Importance Predictor` 不再是預設
研究假設。後續不得因舊報告存在而直接進入 Predictor training、offline label
construction 或 Fluxion-style exhaustive supervision。

目前優先順序是：

1. Training-free
2. Lightweight
3. Query-aware
4. Simple score statistics／heuristics
5. 先證明 Batch=1 consumer platform 的端到端系統價值

Fluxion 保留為相關研究與比較背景，不再作為主要實作對象；已知限制包括未公開
source code、offline labeling／training 成本高，以及其 server-oriented hybrid pipeline
與本地系統不同。

## 基礎方法：Quest CPU–GPU KV offloading

目前 prototype 的 execution path 是：

```text
CPU pinned slab 保存完整歷史 K/V
→ GPU current Query
→ Quest Key min/max Page scoring
→ 每個 Query Head 固定 Top-K Pages
→ selected indices GPU→CPU
→ CPU non-contiguous gather／packing
→ selected K/V H2D
→ GPU sparse head-wise attention
```

Quest fixed Top-K 可能同時造成：

- Over-selection：score 已明顯衰減的 Head 仍被迫選滿 K，浪費 gather／H2D／attention。
- Under-selection：score 分布平坦的 Head 需要更多 Pages，固定 K 卻截掉仍重要的 KV。

## 目前核心研究問題

能否只利用 Quest 原本已計算出的 Page scores，以 O(number of candidate／selected
pages) 或更低成本的 score statistic／threshold／gap，決定每個 Head 當下需要的 K，
並在 Batch=1 resource-constrained CPU–GPU offloading 中降低總 TPOT且控制品質退化？

候選訊號目前只是假設，尚未決定：

- sorted-score gap
- top／tail score ratio
- relative score threshold
- score decay rate
- score concentration
- 其他不需要第二次 QK estimation 的簡單統計

研究不得把「Adaptive K」本身當作 novelty；需要驗證的是 Extremely Lightweight
Adaptive K 在 Small-Batch CPU–GPU offloading 的成本效益。

## 與 Twilight、Ada-KV、FlexiCache、ScoutAttention 的邊界

- Twilight：以 Quest 等 base selector 先選保守候選，再用 INT4 Key 做 QK re-scoring、
  softmax 與 Top-p。它支持 adaptive budget 的動機，但第二層 estimation 可能不適合
  Batch=1 resource-constrained offloading；本研究暫不直接照搬。
- Ada-KV：支持 different heads need different budgets，但主要是 KV eviction／compression，
  不是 Quest Page Score 加 CPU–GPU transfer。
- FlexiCache：主要處理 Page selection 的跨 decode-step stability、更新頻率與增量搬移，
  即「多久重選／重搬」，不是「每個 Head 應選多少」。
- ScoutAttention：主要處理 CPU/GPU cooperative attention、placement 與 overlap，不直接
  解 fixed Top-K budget。

## 已確認的本地系統證據

32K、Batch=1：

- Full：H2D 約 3,538.94 MiB/token，TPOT 約 266.81 ms/token。
- Quest fixed 5%：logical KV visibility 約 5.96%，H2D 約 630–640 MiB/token，
  歷史 3-request TPOT median 約 241.26 ms/token。
- Conditional Greedy @112：TPOT 約 184.11 ms/token；它是 request-specific Offline
  Oracle evidence，不是 deployable online method，也不得改名為 Static112。

因此已確認：減少 KV transfer bytes 不會自動等比例改善 TPOT。

2026-08-29 preliminary profiling 進一步確認：

- Quest selector GPU aggregate：78.55 ms/token
- H2D：51.63 ms/token
- 224 次 head-wise attention：29.05 ms/token
- CPU gather／packing：24.29 ms/token
- CPU selected-index construction：14.40 ms/token

目前可以確認「整體 selector」是最大的單一已量測 cost center；但 selector 內部的
Page score computation、Top-K、sort 與 event／launch overhead 尚未做互斥拆分，不能
直接宣稱其中任何單一子步驟是最大原因。

2026-08-29 五 task Twilight-inspired matched-H2D preliminary experiment 新增以下證據：

- 在 H2D 逐 token 完全等於 Quest fixed 5%（macro 629.67 MiB/token）時，Top-p-derived
  variable per-Head allocation 將 Macro RULER 76→100、Mean Hidden-L2 14.42→7.24、
  answer-trajectory PPL 1.254→1.218。
- 相同控制下 TPOT 234.36→507.61 ms；selector wall 109.29→304.68 ms，且 variable
  per-Head pipeline 使 H2D copies 448→1,344、attention intervals 224→672。
- 因此「fixed Top-K 有品質改善空間」獲得 preliminary 支持，但完整 Twilight-style
  second-stage estimator 不符合目前 Batch=1 TPOT 目標。
- 此 prototype 使用 PyTorch INT4 QK／Top-p 且含 token-level refinement，不是官方 fused
  Twilight reproduction；品質收益仍需 Page-level ablation 分離 per-Head K 與 within-page
  selection。
- 同五 requests 的 fresh Full CPU-offload TPOT 為 mean 278.26 ms、median 277.60 ms、
  CV 1.18%，H2D 3,520.06 MiB/token；因此同輪 Quest speedup 是 1.187×，而 Twilight
  matched 為 Full 的 1.824× latency。
- 2026-08-30 另以 synchronized phase boundaries 完成 45-token 互斥 serialized work
  breakdown；逐 token 所有類別精確加總 100%。Full 最大類別為 H2D 56.95%；Quest
  最大類別為 Selection 30.90%，高於 H2D 14.57%；Twilight 最大類別為 Selection
  45.54%，Attention 15.00% 與 H2D 14.56% 次之。此百分比刻意移除 overlap，只代表
  serialized work composition，不是正常 TPOT 的 causal share。
- 2026-08-30 將 matched-5% Twilight RULER 擴展至 frozen unfiltered 13 tasks × 3 requests：
  Full 74.4018、Quest 67.7351、Twilight 73.3762；fully correct 為 26／23／26。Twilight
  相對 Quest 3/39 較好、36/39 相同、0/39 較差，改善集中於 `niah_multikey_3` 兩筆與
  `vt` 一筆；因此 adaptive allocation 有 quality upper-bound value，但五筆 selected
  Full-correct cohort 高估了改善的廣泛程度。
- 2026-08-30 進一步擴展至 13 tasks × 5 requests（65 題，選題不使用 Full score）：Full
  73.8975、Quest 71.5129、Twilight 73.2822；fully correct 為 42／41／42。Twilight 相對
  Quest 4/65 較好、59/65 相同、2/65 較差；相對 Full 為 0/65 較好、63/65 相同、2/65
  較差。改善主要仍在 `niah_multikey_3`／`vt`，但 `niah_multiquery` 有 Quest trajectory
  較高的反例，因此後續 Adaptive-K 必須同時檢查正向案例與 regression set。
- 2026-08-30 再擴展至 13 tasks × 10 requests（130 題）：Full／Quest／Twilight 為
  76.5002／74.1283／76.8975，fully correct 87／85／90。Twilight 相對 Quest 8 題較高、
  120 題相同、2 題較低；Full-correct retention 為 87/87。Task-stratified 20,000-iteration
  bootstrap 的 Twilight−Quest 95% interval 為 [−0.4615,+5.8077]，仍跨 0；因此可作目前
  主要 preliminary quality cohort，但不能宣稱統計穩定的普遍優勢。後續 Page-level
  Adaptive-K 應以這 8 個正向、2 個反向案例與 87 個 Full-correct requests 作 regression gate。

## 下一個 evidence gate

下一個 evidence gate 改為 Page-level matched-budget Adaptive-K ablation；在提出公式以前，
仍需完成 selector 內部的低侵入 profiling：

1. Page min/max score computation
2. `torch.topk`
3. selected indices sort
4. GPU→CPU index materialization／synchronization
5. CPU position construction
6. 每 token、layer、KV Head／GQA group 的分布

正式 TPOT measurement 必須關閉 instrumentation；profiling 只在額外 diagnostic tokens
執行。同步等待時間與被等待的 CUDA interval 必須分開呈現，不可重複相加。

通過此 evidence gate 後，才依瓶頸選擇第一個最小 ablation：

- scoring dominant：Adaptive K 不得加入第二次 QK estimation；優先 fused／shared scoring。
- Top-K／sort dominant：測試 threshold／gap scan 或 sorting-free selection。
- gather dominant：Adaptive K 必須共同考慮 Page locality／block-major prepack。
- H2D dominant：才支持更 aggressive 的 variable budget。
- GQA duplication dominant：測試 group-wise/shared Page selection，但必須另做 quality gate。

第一個方法候選不得沿用完整 Twilight INT4 QK／token-level Top-p pipeline；應只使用
Quest 已計算的 Page Scores，在每個 GQA group 維持與 fixed 5% 相同總 Page budget，僅
重新分配 per-Head Page counts。如此才能單獨回答 Head-wise Adaptive K 是否保留品質收益，
並避免將三倍 copy／attention dispatch 誤當成方法不可避免的成本。

## 證據與命名規則

- 舊 Predictor／Fluxion／Oracle artifacts 保留，不刪除、不改寫成新方法結果。
- Conditional Greedy 是 label-leaked request-specific Offline Oracle，不是 deployable policy。
- 本地 Quest prototype 是 GPU-executed selected-KV Partial-H2D prototype，不是官方 Quest
  fused CUDA performance，也不是 Fluxion hybrid CPU/GPU pipeline reproduction。
- 單一 request 或單次 profiling 必須標示 preliminary。
