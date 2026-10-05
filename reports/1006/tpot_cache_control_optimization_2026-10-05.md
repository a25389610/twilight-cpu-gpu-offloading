# 2026-10-05：selection 之外的 cache control／snapshot 優化

本輪新增改善：同批前輪 v5 **44.301443 → 40.772061 ms/token，減少 3.529382 ms（7.9667%）**。相對更早原版本 48.110928 ms 的累積改善為 7.338867 ms（15.2541%），不可當成本輪新增收益。40.772061 ms 是目前已完成的結果，尚無後續版本低於此基準的正式證據。六組配對全部較快。cache backing相對v5平均多109.970540MiB，應保留速度／容量取捨。本報告承接 `reports/tpot_execution_optimization_2026-10-05.md` 的 v5 execution preset。

## 問題、假設與保留條件

使用者指出目前 TPOT 與 RetroInfer／FreeKV 的差距大，不能僅以 selection 解釋。先查核既有優化、source 與已量結果，再測 cache-control 候選。控制條件為 RTX 5060 Ti 16GB、Llama-3.2-3B-Instruct BF16、Batch1、32K cohort、p=.90、B0=8192、Direct INT4 tile16；保留原 logical selection、GQA union 與模型算術。基底是先前 merged-QKV experimental 路徑，本次不宣稱相對舊 grouped runtime 全輸入 exact。

Formal 為 fixed-token1、32 forwards、D2–D32 synchronized CPU wall，無 profiler；每 arm 三題兩輪186 samples。Frozen requests 分別31938／32639／32363 tokens，與先前相同。Diagnostic memory、payload gate、offline replay 與 formal 分開。

## 本次額外參考 source

- ShadowKV：`external/shadowkv-20261005`，commit `e51904cdeab7d4d34013370f09f2cf5fcd655e15`。讀取 `models/kv_cache.py`、`kernels/gather_copy.cu`、`map.cuh`、`copy.cuh`。GPU產生hit/miss offsets／counts，區分 GPU hit 與 host miss 的搬移；選取chunk數固定，不能直接套到本方法dynamic token Top-p。
- SGLang HiSparse：下載 pinned commit `bab04cd7913ff50d1bf1a677a2d10e5015dc4c3b` 的 `python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh`、`python/sglang/kernels/ops/kvcache/hisparse.py`、guide，保存於 `external/hisparse-source-20261005` 與manifest。Source使用persistent device buffer／slot mapping、GPU hit判斷及miss填入，避免每一步重建完整resident快照。並非在本環境跑完整SGLang／HiSparse。
- 延續RetroInfer／FreeKV source查核，參考其metadata與buffer分工。本次是本方法的adapter實驗，不可稱完整移植這些系統。

## 候選與 source 證據

1. **GPU cu_seqlens**：原consume在CPU建立 `torch.tensor(group_total_lengths).cumsum(0).tolist()`，再建device tensor。改由GPU current_lengths以單一Triton integer scan建9元素cu_seqlens，避免CPU構造及小型H2D；selected KV／attention運算不變。
2. **Buffer ownership swap**：原assembly先寫attention scratch，再把完整內容複製至resident。改為只寫new token，直接令attention scratch成為該layer的下一步resident；該layer的舊resident成為下一次共用scratch。source／target必須disjoint，同ordered compute stream，returned tensor仍指向正確資料。沒有減少hit assembly或CPU misses，僅免snapshot copy。
3. **Reusable mapping workspace**：重用bitmap scans的counts／positions／sources；miss-length輸出另配置，避免pending diagnostic records被下一layer覆寫。限同一ordered stream。
4. **Separated assembly**：GPU hit與compact host miss分開kernel，保持row order；增加launch成本，需實測，不能只從cache hit ratio預測收益。

## 已完成的最小證據

### Host-length offline replay（非可部署方法）

第一題32K，868個entry:length keys（31 steps×28layers）預錄後全部重播；baseline44.329399→replay41.272038ms/token，單配對差3.057361ms。仍執行selection／mapping，只免CPU等待這份長度。checkpoint／final logits與resident trace相同。這是diagnostic，不是一般化加速或所有overhead的upper bound，不能與其他消融跨run相加。第一次取得動態生成source失敗，修正adapter source取得方式後完成，原失敗資料保留。

### Cache control 初步單題消融

| 版本 | ms/token |
| --- | ---: |
| v5基底 | 44.296294 |
| mapping pool only | 44.578351 |
| GPU cu_seqlens only | 40.781897 |
| swap only | 43.557421 |
| pool＋GPU cu＋swap | 40.454185 |

五個fresh processes，checkpoint／final logits及完整resident trace一致。單題結果皆preliminary；pool沒有顯示效益，最終候選關閉pool。

第二批單題matched pilot：v5 44.160796、GPU cu＋swap40.153491、再加split40.966022ms。三arm checkpoint／final logits與resident trace一致。split沒有顯示收益，最終候選關閉split。

### Correctness 與失敗保留

- 八種length／previous stride的mapping micro gate：valid source／position／miss rows與原mapping一致，GPU cu一致，持久miss counts不被覆寫。
- QA128 gate：v5對pool＋GPU cu＋swap，prompt、checkpoint／final logits、selection、GQA union、Attention K/V、CPU new-KV與resident trace全部相同。Selection／Attention是D1／D2／D32／D128 checkpoints，CPU new-KV逐step；不是每step Attention皆capture或130題品質。
- 初始split32-step payload gate相同，但split曾在D2才編譯，timing受影響，不用於效率結論。將編譯移至cache初始化後做第二批pilot；編譯修改不改kernel算術。最終split關閉。

## 最終 matched formal：原版本／v5／新候選

18個fresh processes，三題×兩輪×三arms，各186 D2–D32 samples。raw CSV重算與manifest／retained source snapshot hash核對通過；固定token1、p=.90、B0=8192、無profiler。原版本包含既有merged-QKV／pointwise；新候選為v5＋GPU cu＋swap，pool／split皆關閉。

| 版本 | 六run平均 ms/token | run mean範圍 |
| --- | ---: | ---: |
| 原版本 | 48.110928 | 47.606160–48.546836 |
| 前輪v5 | 44.301443 | 44.005142–44.917027 |
| 新候選 | 40.772061 | 40.189440–41.941336 |

| 配對 | 原版本 | v5 | 新候選 |
| --- | ---: | ---: | ---: |
| rep1 / 001_niah_multikey_3_i011 | 48.131008 | 44.360295 | 40.189440 |
| rep1 / 002_vt_i002 | 47.606160 | 44.116512 | 40.573985 |
| rep1 / 003_qa_1_i011 | 48.505511 | 44.319672 | 40.751878 |
| rep2 / 001_niah_multikey_3_i011 | 48.040881 | 44.005142 | 40.291399 |
| rep2 / 002_vt_i002 | 47.835172 | 44.090011 | 41.941336 |
| rep2 / 003_qa_1_i011 | 48.546836 | 44.917027 | 40.884327 |

三arm prompt／checkpoint／final logits與完整resident count/payload trace一致。新版本相對原版本低7.338867ms（15.2541%），相對v5低3.529382ms（7.9667%）。不是將各批次消融相加，不能扣歷史46.511ms。D2包含fresh-process首次dispatch／既有Triton cache載入；保留全部D2，未為結果刪除較慢樣本。未清除編譯cache，不能解讀為首次安裝的cold compile latency。

## 最終 QA128 payload gate

另外兩個fresh diagnostic processes，v5對最終pool=0／split=0的GPU cu＋swap。D1／D2／D32／D128 selection／GQA union／Attention K/V，逐step CPU new-KV、resident全trace、checkpoint／final logits全部相同。2688 selection records、896 unions、112 Attention K/V、28672 CPU new-KV、3584 resident records；28448 new-KV records在下一consumer前驗證，最後224 entries沒有下一consumer。不是130題greedy品質或每step完整Attention capture。Gate timing有診斷hook，不混入formal。

## 最終 memory：六個獨立processes

另三題各v5／新候選，inventory與formal分開；MiB，D32 live allocated，不是reserved／NVML／prefill peak。模型unique權重各6127.841064MiB。cache backing依storage去重，包含metadata／INT4／resident／cache-owned workspace；runtime／graph allocation算入全部live與nonweight live。

| 三題平均 | v5 | 新候選 | 差值 MiB |
| --- | ---: | ---: | ---: |
| cache_unique | 1324.790390 | 1434.760930 | 109.970540 |
| 全部live allocated | 7493.943848 | 7598.911947 | 104.968099 |
| 扣模型權重後live | 1366.102783 | 1471.070882 | 104.968099 |
| decode peak allocated | 7519.558268 | 7626.583333 | 107.025065 |

Logical selected KV相同，physical backing增加：交換buffer時，較大的scratch會跨layer輪替，on-demand capacity可能擴張。不能把免copy解讀成容量不增加。此preset適用於接受上述約105MiB額外live的情況；固定equal-VRAM比較需另做budget控制。

### 保留較低容量的 GPU-cu-only 選項

`--preset gpu-cu`保留snapshot，只把cu_seqlens留在GPU。第一批單題formal-style pilot為44.296294→40.781897ms（preliminary，未三題兩輪）。另單題D32 memory對照：cache_unique 1308.558014MiB，cache delta 0.000000MiB、live delta 0.000000MiB；checkpoint／final logits與resident trace一致。該memory run的時間不是正式TPOT。此選項沒有與最快候選同批完整六pair正式比較，不宣稱其Pareto優越或全三題memory已驗證。

## 與其他offloading系統仍有差距的解讀

本次已證實cache control與snapshot能改善整體TPOT；不能以模型名稱／offloading／reuse相同，推論每步資料量與同步成本相同。先前同cohort v5第一輪raw resident trace核對：三題每token selected logical KV約512.216／487.847／523.387MiB，mapped CPU miss logical payload約45.170／49.157／46.171MiB。高hit ratio仍有大量GPU hit assembly與CPU miss read；這是logical bytes，不是實測PCIe transactions。Snapshot免除一份整批copy，原hit assembly與host misses仍存在。

RetroInfer本機設定retrieval=.018、estimation=.232、cache=.05，以cluster／WaveBuffer管理；與本方法p=.90的動態selected sets不同。尚未獨立量其matched PCIe miss bytes，不能直接拿上述logical bytes或跨runGPU time比值歸因全部剩餘差距。這次沒有重新建立與兩篇同品質的Pareto curve。

GPU host-length `.cpu().tolist()` handoff仍在。HiSparse式stable physical slots／GPU index-aware attention或GPU length-aware ragged attention是較大架構候選，需控制capacity、VRAM、attention數值與quality。初步offline replay支持有改善空間，但不能預告可達25–30ms或把各diagnostic收益相加。

## Artifacts 與使用限制

相對路徑以HeadInfer reproduction root為準：

- `results/tpot_length_oracle_20261005/analysis.json`、lengths／coverage／原失敗資料。
- `results/tpot_cache_control_20261005/{probe.json,gate_analysis.json,pilot_analysis.json,split/gate_analysis.json}`。
- `results/tpot_cache_final_20261005/{pilot_analysis.json,pilot_manifest.json,pilot_source_snapshot/,formal/...}`。
- Source：`scripts/tpot_execution_opt/benchmark_{length_oracle,cache_control,swap,split}_case.py`、`resident_split.cu`、drivers／analyzers。
- 最終候選入口為 `scripts/tpot_execution_opt/run_cache_optimized_case.py`，`baseline`＝原基底、`v5`＝前輪改善、`gpu-cu`＝v5＋GPU cu、`optimized`＝v5＋GPU cu＋swap。實驗性process-local adapter，依賴prepared source tree／本機model／CUDA env，非獨立installer。production source未改。

尚未130題、尚無25–30ms證據。GPU host-length handoff仍在；paged／GPU-index-aware attention的slot設計尚未移植。已核對deduplicated cache與全部nonweight live；swap的容量代價已列於上表。

## 執行入口與鏡像

在prepared HeadInfer reproduction root：

```bash
/home/paul/miniconda3/envs/headinfer_repro/bin/python \
  scripts/tpot_execution_opt/run_cache_optimized_case.py \
  --request-dir results/context_p_ruler_39_v1/cohort/32768/timing_requests/001_niah_multikey_3_i011 \
  --output results/my_cache_optimized/result.json \
  --preset optimized --decode-steps 32
```

`--preset gpu-cu`可使用較低容量候選；其三題兩輪收益尚未正式驗證。全部adapter仍experimental，未切換production預設。CLI語法／help、37個以上Python source AST、report與mirror byte equality核對；不把help當作GPU效能驗證。正式推論驗證依上述fresh processes。

已prepare public mirror `reports/1007/tpot_cache_control_optimization_2026-10-05.md`與`scripts/tpot_execution_opt/`的Python／CUDA／README。未修改production `source/`、未commit／push。raw JSON／CSV／log／PT／trace、external checkout、模型、環境與compiled binary不納入；可公開腳本仍依賴prepared local research tree。
