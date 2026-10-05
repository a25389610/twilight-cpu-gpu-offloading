# RetroInfer 與目前 Twilight previous-token cache：本機 artifact 比較

日期：2026-10-05。Artifact-only 分析，未執行新 GPU benchmark。

## 問題與對照身份

使用者要求比較剛完成的 RetroInfer 與目前研究。對照採用已接受的 approximate 路徑：Direct INT4 QK + Quest B0=8192 + Twilight dynamic Top-p p=.90 + previous-token resident cache + GPU bitmap mapping + GPU mapped CPU KV read + low-VRAM flags / Top-p Graph。不是早期120ms cache-off版，也不是 exact triton_prepare版。

品質使用相同 Llama-3.2-3B-Instruct、BF16、Batch1、frozen RULER 32K 130 requests、原 generation budget、greedy/EOS與 scorer。RetroInfer budget .018/.232/.05、CPU core4、graphs off；其框架與 Attention approximation 和 Twilight 不同，沒有 equal-quality/equal-VRAM tuning。

## 重算結果

| 指標 | 目前 Twilight + previous-token cache | RetroInfer |
| --- | ---: | ---: |
| 完整130題平均分數 /100 | 76.179615 | 74.859077 |
| 保存的 formal TPOT（ms/token） | 55.938551 | 24.259923 |
| Unique GPU cache backing storage（MiB） | 1324.816757 | 421.339019 |
| 相對固定32K Full BF16 KV3584MiB | 36.96% | 11.76% |

品質逐題配對：RetroInfer **3題較好、123題同分、4題較差**；整體低 1.320538 points。同分不代表答案字串或logits相同；本輪未對每題生成文字做exact主張。

| Task（各10題） | Twilight | RetroInfer |
| --- | ---: | ---: |
| niah_single_1 | 100.00 | 100.00 |
| niah_single_2 | 100.00 | 100.00 |
| niah_single_3 | 100.00 | 100.00 |
| niah_multikey_1 | 100.00 | 100.00 |
| niah_multikey_2 | 100.00 | 100.00 |
| niah_multikey_3 | 40.00 | 20.00 |
| niah_multivalue | 100.00 | 100.00 |
| niah_multiquery | 95.00 | 92.50 |
| vt | 82.00 | 84.00 |
| cwe | 0.00 | 0.00 |
| fwe | 83.34 | 86.67 |
| qa_1 | 50.00 | 50.00 |
| qa_2 | 40.00 | 40.00 |

主要差異是 niah_multikey_3 40→20與multiquery95→92.5；RetroInfer在vt82→84與fwe83.335→86.668較高。CWE與兩組QA同分，因此不能把RetroInfer這些低分單獨當成其selection退化證據；仍缺Full quality anchor。

## 時間與memory的可比範圍

兩組 formal 使用相同三題、相同prompt hashes、fixed token1、32forwards、D1warm-up、31個D2–D32同步CPU wall，排除load/prefill與component profiling。Twilight raw是09/28breakdown最新保存六次結果55.939；先前52.104與55.042不同批，不混算。RetroInfer raw是10/05六次24.260。兩組未在同時段交錯重測；比值 2.306×、latency低56.63%只描述歷史測量比值，不能宣稱fresh paired speedup、equal-quality優勢或cache單元因果收益。

memory採三題獨立inventory的unique cache GPU backing去重，alias不重複。RetroInfer比Twilight少903.478MiB（68.20%）。這不是整個process顯存，也不含模型、runtime/Graph等；兩框架實際權重storage不同：RetroInfer複製embed/lm_head，不能拿totalVRAM直接推cache優劣。這裡37%與12%包含各自selection metadata，而非只算residentKV。

## 方法重疊與差異（本機source可確認）

- 重疊：完整history KV於CPU、GPU保留部分可reuse的KV、query-dependent selection、miss由CPU補取；兩者均服務有限VRAM長context。
- Twilight：Quest min/max粗篩，GPU全history INT4 Key估計query-key重要度，再dynamicTop-p/GQA union；保留previous-token selectedKV，miss mappedCPU讀取。Direct QK為已接受近似實作，不聲稱logitsexact。
- RetroInfer：對KV建立clusters，以query對centroid分數挑cluster；retrieve zone讀真實KV，estimation zone以centroid/value summary做weighted attention近似並合併；GPU cache以cluster/page管理與LRU重用。retrieval .018是centroid選取比例，不能直接等同1.8%實際KV tokens；cache .05也不是所有cache-relatedstorage只有5%。
- persistentmemory：Twilight resident577.858MiB + INT4codes/scale/min469.396 + Questmetadata220.390，占大部分1324.817MiB。RetroInfer resident176.750MiB + index/summaries222.085，占大部分421.339MiB；摘要部分也參與attention estimation，語義不等同只丟棄沒選到的KV。
- 因selection、attention approximation、GPUKV容量、kernels/framework與graphs都不同，不能把24vs56ms差值歸因LRU比previous-token更好；亦不能把1.32point小差異當一般化或統計顯著品質優勢。

## 研究判斷與下一決策

**已確認**：RetroInfer是高度相關、可在現有硬體實跑的baseline；目前兩設定下，你的方法score稍高，RetroInfer保存的latency與cachestorage較低。整體不能宣稱你勝出，也不能從這組trade-off宣稱研究被完全取代。

**尚未確認**：相同品質門檻與GPUbudget下的比較、fresh同時段效能差、完整Full quality anchor、優勢來自selector或執行框架何者。

**下一決策**：若研究目的是證明整體系統競爭力，先定義Full為anchor的品質容忍門檻，再調整RetroInfer retrieval/estimation budget並重測同cohort與fresh交錯timing。若目的是previous-token reuse的獨立貢獻，固定selector/attention與cachebudget後比較previous-token/LRU；不直接把兩個完整框架比較當reuseablation。本輪未自動展開新的budget sweep。

## Evidence與重現

- 分析：`python scripts/compare_retroinfer_twilight_20261005.py`，由專案root執行。逐題重新scoring，assert130組request IDs、prompt hashes/count、budget與成功狀態一致；formal核對固定步數與prompt hashes；memory由rawinventories讀取。
- comparisonraw：`results/retroinfer_vs_twilight_20261005_v1/{summary.json,per_request.csv,artifact_hashes.json}`。
- qualityraw：`results/twilight_direct_qk_ruler130_quality_20260927_v2/`、`results/retroinfer_ruler130_quality_20261005_v1/`。
- timing/memoryraw：`results/twilight_direct_qk_breakdown_20260928_v1/`、`results/retroinfer_ruler_20261005_v1/`。
- 機制source：外部`../../external/retroinfer-20261005/cache_hub/retroinfer_cache.py`；本機`source/headinfer/headinfer/twilight_offload_cache.py`，以及09/28breakdown與10/05RetroInfer報告。
- publicmirror只準備本報告與分析script，raw未納入；未commit/push。
