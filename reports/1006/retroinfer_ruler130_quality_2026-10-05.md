# RetroInfer：本機 3B / RULER 32K 完整 130 題品質測試

日期：2026-10-05。單次完整 cohort 評估；13 題沿用同設定已驗證 pilot，117 題本次新增。

## 問題與實驗目的

先前 13 題 pilot 的 63.21 分不能代表完整 130 題。本次依使用者要求完成相同 frozen cohort，確認目前 RetroInfer 本機設定的 task quality；不是驗證品質相當的效能優勢。

## 條件與來源

- RTX 5060 Ti 16GB，Batch 1，`meta-llama/Llama-3.2-3B-Instruct`，BF16；官方 Llama Python API，新模型 config，CPU thread pool 4。
- 官方 source：`microsoft/RetrievalAttention`，SHA `03f912c6e917c380d9d90c5ec85bb0f161ba53ef`；weighted attention SHA `56d96228ada74d6df806b0083bf018d0d57f57e9`。tracked source hashes 在完成時再次核對一致。
- retrieval=.018、estimation=.232、cache ratio=.05、graphs off、CPU offloading enabled，Full prefill。`n_centroids` 依官方 config generator 隨實際 prompt 長度決定。
- frozen dataset：`SaylorTwift/RULER-32768-llama-3.2-tokenizer`，revision `e748e0cd1872b4bbaa6d5ed9c6fbcf6068951c42`；13 tasks × 10 requests，直接使用原 prompt IDs，不重新 tokenize 或補齊至 32768。
- cohort：`results/context_p_twilight_gqa_group_ruler130_v1/cohort/32768/`。沿用每題原 `generation_budget`、greedy decoding、EOS=128009；min 2 forwards 的原 runner 條件；使用相同 `official_ruler_score` 及每題 references/scorer。score 是字串匹配比例分數，不能解讀為全對題數百分比。
- 每題 fresh process。13 個 pilot result byte-identical 沿用，另有 `reuse.json` 記錄來源與 hash；未重跑或挑選替代題。
- 環境為先前 feasibility 的獨立 overlay：torch 2.7+cu128、transformers 4.49、Triton 3.4；保留 torch/Triton package pin mismatch。詳見同週 feasibility 報告。

## 完整結果

**整體分數：74.859077 / 100（四捨五入 74.86）。** 每 task 等權平均；因每 task 都是 10 題，亦等於 130 題 score 平均。

| Task | 題數 | 平均分數 / 100 |
| --- | ---: | ---: |
| niah_single_1 | 10 | 100.00 |
| niah_single_2 | 10 | 100.00 |
| niah_single_3 | 10 | 100.00 |
| niah_multikey_1 | 10 | 100.00 |
| niah_multikey_2 | 10 | 100.00 |
| niah_multikey_3 | 10 | 20.00 |
| niah_multivalue | 10 | 100.00 |
| niah_multiquery | 10 | 92.50 |
| vt | 10 | 84.00 |
| cwe | 10 | 0.00 |
| fwe | 10 | 86.67 |
| qa_1 | 10 | 50.00 |
| qa_2 | 10 | 40.00 |

- 完成 130/130，13 tasks 各 10 題，無失敗或缺失；130 題均遇 EOS，沒有題目撞到 generation budget。
- 生成 IDs 合計 2625 tokens（含停止 special token），不代表各題固定生成長度。

## 實際執行時間與範圍

- 本次新增 117 題的 fresh-process wall 合計：1942.437 s（32.37 分鐘）。
- 完整 130 題含先前 13 題的 fresh-process wall 加總：2160.516 s（36.01 分鐘）。這是各題 process 時間加總，不是本次 turn 的連續經過時間；不包含 orchestrator、檢查、報告整理與中斷空檔。
- cache init + prefill/prepare + decode 的 130 題累計：1495.153 s（24.92 分鐘），不含 model load/import。各題均單獨載入模型，未測 shared-model batch runner。
- 本次是 quality 評估；不可把可變長度 greedy latency 當成 fixed-token formal TPOT。先前三題兩輪 formal TPOT 24.260 ms/token / 獨立 cache unique 421.339 MiB 的 scope 仍依 pilot 報告，不能宣稱此為 130 題重測平均。

## 驗證、失敗與處理

- 獨立 analyzer 核對 130 個 request ID、prompt hash/count、manifest 中 request JSON/PT 的 file hashes、generation budget、EOS、輸出長度、success/returncode 及 config。
- 全部 130 題依相同 scorer 重新計分，與逐題原 score 完全相同；macro 與 request average 相同。
- 全部官方 tracked source 與 3B config hashes 保持一致；case/shell/scorer hashes 保持一致。
- orchestration 在第 10 題後曾因把不同 prompt 長度的 `n_centroids` 要求完全相同而中斷。修正驗證器允許此官方衍生欄位不同，保持 selection/cache/thread 參數一致，接續原 result；沒有修改生成程式、官方演算法或重跑已完成答案。

## 結論與限制

**實驗觀察**：目前本機 RetroInfer 3B 設定在這個完整 cohort 得 74.86 分，且任務差異很大。六個 NIAH tasks 得 100，`niah_multikey_3` 20、`cwe` 0、QA 50/40，是需要保留的品質弱點。

**尚未確認**：不能把失分直接歸因於 clustering、selection 或 cache reuse；尚未在本輪補 fresh Full quality 對照。這也不是原論文模型/硬體/benchmark 的精確重現，亦不能由 standalone 24ms 直接宣稱比目前 Twilight 方法更快且品質相當。

**下一決策**：在完全相同 cohort/scorer 上核對 Full 與目前 previous-token cache 的品質結果，先確認共同品質門檻，再討論速度/VRAM trade-off；若要調整 RetroInfer budget，需另立設定與結果，不覆寫本輪。

## Artifacts 與重現

- raw artifacts（僅本地）：`results/retroinfer_ruler130_quality_20261005_v1/`；包含 immutable manifest、130 個 `requests/<name>/result.json`、command/process wall/log、13 個 reuse provenance、`quality.csv`、`summary.json`。
- 執行：`python scripts/run_retroinfer_ruler130_quality.py`；完整輸出可 resume，核算：`python scripts/analyze_retroinfer_ruler130_quality.py`。在 `headinfer/headinfer_reproduction/` 下執行。
- case runner：`scripts/benchmark_retroinfer_ruler_case.py`；shell：`scripts/run_retroinfer_ruler_local.sh`。
- manifest SHA256：`0309ec720284863c76b14b8d95681e56b11f080a7948a0c1ed22277d600502a4`。
- analyzer SHA256：`373aec5a3c007b5ac2c0b0216a4ed289e76dd8776c22ecdc203eb3b96d5d9622`。
- GitHub mirror 準備本 Markdown 與新增兩支 orchestration/analysis scripts；不包含 raw JSON/CSV/log、prompt data、模型、binary 或 `.venv`，未 commit/push。
