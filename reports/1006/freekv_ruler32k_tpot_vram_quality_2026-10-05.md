# FreeKV conservative dispatch：相同 32K RULER 130 題、TPOT 與 VRAM

日期：2026-10-05。研究層級：6，外部 baseline 在固定 operating point 的實際評估。

## 結論

**130/130 題完成；RULER 平均 70.359/100。正式 TPOT 六次平均 32.632ms/token。GPU cache 相關 unique backing storage平均 1603.488MiB（1.566GiB），不含模型權重。**

這是 **FreeKV local port with conservative dispatch**，保留 previous-query selection／per-head correction，但停用原 background CPU worker dispatch overlap；不是原論文完整非同步效率重現。原 async path 有未解決的 resident payload 差異；本次沒有使用它的數字。不能僅以此點TPOT與未 matched quality的舊結果判斷方法勝負。

## 問題、假設與控制

使用者要求同樣130題的完整outputs、TPOT及VRAM。假設是：已通過合成測試的保守port可在真實frozen requests下保持資料一致性並完成評估。以每題end-state resident payload／inverse mapping、EOS／budget、finite及scorer重算為gate；formal timing另跑，避免quality D2H／audit干擾。

- RTX5060Ti16GB、driver580.173.02、CUDA12.8、PyTorch2.7.0+cu128、HF4.45.2、FlashInfer.2.4、BF16、Batch1。
- 模型 `meta-llama/Llama-3.2-3B-Instruct`，snapshot `0cb88a4f764b7a12671c53f0838cd831a0843b95`。
- 官方 [FreeKV repository](https://github.com/sjtu-zhao-lab/FreeKV)，commit `2c8a7d25c9f3c7c15ce15b2f84cd03f477bd7469`；GQA3／build相容性、page mapping／prefill stream／mask生命週期修正及保守dispatch詳見同週 `freekv_local_feasibility_2026-10-05.md`。
- 固定 page_size32、budget2048 tokens（64pages）、sink512、recent512、corr=.8、group_size1、spec_ret開、CPU HND、cuda_cpy、GPU pool1536MiB／CPU pool4096MiB、graphs off。中途未調參。
- 130題＝13tasks×10，沿用研究既有 frozen 32K cohort的prompt IDs、request IDs、generation budget、EOS與scorer；dataset `SaylorTwift/RULER-32768-llama-3.2-tokenizer` revision `e748e0cd1872b4bbaa6d5ed9c6fbcf6068951c42`。標籤32K不表示每個prompt剛好32768tokens。
- 130題皆freshprocess、greedy到EOS或原budget；沒有重用pilot outputs。保存完整prediction、IDs與reference／score。

## 130 題品質與輸出

| Task | 題數 | 平均 score／100 |
|---|---:|---:|
| niah_single_1 | 10 | 100.000 |
| niah_single_2 | 10 | 100.000 |
| niah_single_3 | 10 | 100.000 |
| niah_multikey_1 | 10 | 100.000 |
| niah_multikey_2 | 10 | 80.000 |
| niah_multikey_3 | 10 | 0.000 |
| niah_multivalue | 10 | 97.500 |
| niah_multiquery | 10 | 82.500 |
| vt | 10 | 68.000 |
| cwe | 10 | 0.000 |
| fwe | 10 | 96.667 |
| qa_1 | 10 | 50.000 |
| qa_2 | 10 | 40.000 |
| **13-task平均** | **130** | **70.359** |

所有task各10題，因此macro score也等於130題request mean。127題EOS、3題用盡128-token budget：`062_niah_multivalue_i016`、`065_niah_multivalue_i001`、`075_niah_multiquery_i004`；三題都保留，沒有排除。總生成2916IDs。

完整本機輸出：研究root的 `results/freekv_ruler130_20261005_v1/outputs_130.md`，每題有prediction／references／score／tokens／EOS；`quality.csv`可逐列查詢；原始JSON另有完整greedy IDs與latencies。這些raw輸出不加入GitHub文字報告mirror。

品質process wall加總 **37.77分鐘**，包含每題load/import及end-state audit；cache init＋prefill＋decode加總 **27.29分鐘**，不含load／audit。這是130題phase的逐process加總，不包含另9次timing／memory runs，也不是聲稱可達到的model-reuse時間。

## Formal TPOT

相同三題timing-only disjoint prompts：niah_multikey_3／vt／qa_1，長度31938／32639／32363；兩輪freshprocess，第二輪順序反轉。

| Request | Rep | D2–D32 TPOT（ms） | D1（ms） |
|---|---:|---:|---:|
| 001_niah_multikey_3_i011 | 1 | 32.330 | 47.057 |
| 001_niah_multikey_3_i011 | 2 | 32.423 | 46.367 |
| 002_vt_i002 | 1 | 33.511 | 45.971 |
| 002_vt_i002 | 2 | 32.887 | 45.974 |
| 003_qa_1_i011 | 1 | 32.282 | 47.468 |
| 003_qa_1_i011 | 2 | 32.360 | 47.214 |

平均 **32.632ms/token**，individual mean **32.282–33.511**。

Timer是synchronized CPU wall `model.forward`，fixed token ID1、32次decode，D1排除後31次為分母。包含runtime的position metadata準備與模型／selection／recall；沒有每步logits D2H、finite audit、額外Python component counters或profiler。prefill先執行，D1處理首輪recall；不用quality的EOS長度做formal分母。數字適用本地保守dispatch；不當成原FreeKV background overlap效能。

## VRAM：排除權重，分清backing capacity與有效頁

三題各跑一次獨立memory run，在D32做InferState reachable tensors的backing storage去重。平均：

| 類別 | GPU MiB |
|---|---:|
| gpu_kv_digest_pool_capacity | 1536.000 |
| gpu_selection_and_metadata | 1.488 |
| gpu_attention_workspace | 64.000 |
| gpu_recall_workspace | 2.000 |
| **Cache-related unique backing總和** | **1603.488** |

**1536MiB pool完整計入實際CUDA配置**，即使頁尚未用、或prefill eviction延後回收；KV與digest共用這個pool，不能再將兩者active bytes加到unique總和造成double count。active resident KV頁是224MiB、digest頁112MiB（合計336MiB），但這是pool內有效頁容量，**不是此版本實際GPU VRAM只占336MiB**。尚未測最小可行pool，不能推算縮pool一定保持同樣效能／correctness。

| 另一種統計口徑 | 平均 MiB |
|---|---:|
| Decode end allocated − model weight storage | 1612.892 |
| D2–D32 peak allocated − weights | 1613.632 |
| Prefill peak allocated − weights | 3904.485 |

實際CUDA model parameter unique storage **6127.834MiB**，三run一致。end allocated減權重不是純KV；包含outputs／runtime其他暫存。prefill peak含temporary，不能稱KV大小。三run total prefill peak含權重為9.774–9.819GiB；上述是PyTorch allocated／unique storage，非driver全機VRAM數字。reachable inventory不宣稱涵蓋外部driver context或opaque CUDA allocations。

## Correctness、驗證與限制

- 130 quality＋3 memory的end-state payload與inverse mapping audits全部0差異；每run排除最新兩頁，檢查CPU-backed resident KV。這是end-state gate，不證明所有中間attention值逐步exact。
- 130筆prompt/request/file hashes、EOS／budget、輸出長度、process returncodes及scores全部獨立核對；6筆formal分母、3組unique storages核對通過。
- Benchmarkcase、shell、scorer及9個FreeKV source／extension hashes從frozen manifest核對一致；沒有在batch中修改source。
- 不推論剩餘低分來自stale selection、budget、模型backend或特定修正；需要ablation及Full reference才可歸因。先前dense只有token agreement，非logits exact。
- 未恢復原CPU worker overlap、沒有同品質／同VRAM的Pareto curve；不能把較快TPOT與較低品質混成全面勝出。

## Artifacts、執行與下一決策

研究root：`headinfer/headinfer_reproduction/`。

- `results/freekv_ruler130_20261005_v1/`：manifest、逐題formal/memory/quality、command／process wall／log、quality.csv、formal.csv、summary.json、validation.log、outputs_130.md。
- `scripts/freekv_local/benchmark_freekv_ruler_case.py`、`run_freekv_ruler_local.sh`、`run_freekv_ruler130.py`、`analyze_freekv_ruler130.py`。新增4scripts；原5port scripts沿用。

從workspace root執行 `external/freekv-20261005/.venv/bin/python headinfer/headinfer_reproduction/scripts/freekv_local/run_freekv_ruler130.py`；已有完整validated results會resume skip，不重跑。mirror中的scripts可設 `FREEKV_WORKSPACE` 指向本機workspace。

本地canonical週報 `reports/weekly/2026-09-30.md` 更新；本report及4新增scripts準備於public mirror `reports/1007/`／`scripts/freekv_local/`。raw JSON/CSV/log/predictions/IDs、env、模型、submodules與binary未納入；沒有commit／push。

下一決策：以這個 **70.359分／32.632ms／1.566GiB實際cache backing** 的operating point討論品質門檻，再決定調budget／corr或修原async runtime。尚未做這些新實驗；當前比較名稱需明示conservative dispatch。
