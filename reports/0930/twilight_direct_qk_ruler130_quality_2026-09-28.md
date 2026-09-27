# Direct INT4 QK：32K RULER 130 題 paired 品質驗證

日期：2026-09-28（Asia/Taipei；實驗於 09/27 開始、09/28 完成）  
狀態：130/130 題、260 個 quality runs 完成；非 formal TPOT 實驗。

## 問題、假設與比較範圍

先前在低 VRAM Twilight + previous-token resident cache + GPU mapped CPU KV read 路徑，將 `triton_prepare` 的 FP32 estimated-K materialization + matmul 切換成 experimental direct fused INT4 QK，三題兩輪 matched formal TPOT **65.309 → 52.104 ms/token（-20.22%）**，但 Selection 與 logits 不 exact。本輪要檢查這個數值差異是否在既有 32K RULER cohort 造成可觀察的答案或任務分數變化。先前 13 題 probe 太小；舊 130 題品質結果屬於其他 source 快照，不能拿來和此候選直接配對。

本輪使用 frozen 32K cohort 的 **13 tasks × 10 requests = 130 題**，Llama-3.2-3B-Instruct、BF16、Batch 1、Twilight dynamic Top-p `p=.90`、Quest `B0=8192`、同一 resident mapped-KV execution path。每題 exact 對照和 direct candidate 都用當前 source 重新執行。兩臂 command 除 output path 外僅差 `--twilight-qk-backend triton_prepare` / `triton`；arm 順序逐題交錯。每題 greedy 生成至第一個 EOS 或原 cohort 的 generation budget（30、32、50、120、128），以 tokenizer `skip_special_tokens=True` 解碼第一個 EOS 前綴，使用專案既有 `official_ruler_score` 和 request 內 frozen references/scorer 重新計分。

本輪為 quality-only。`--greedy-probe-stop-at-eos` 是 default-off 診斷開關；第一題與早先跑滿 128 steps 的結果核對，兩臂 P4 hash 和 EOS 前 token 各自 exact。該開關避免 EOS 後無效 Decode；正式 fixed-token TPOT 路徑未使用。舊版滿 budget 的 22 題不納入本輪 130 分析，保留於 `results/twilight_direct_qk_ruler130_quality_20260927_v1/` 作 pilot；第一次 EOS-stop 的 final-logits key 失敗也保留在 `v2_failed_eos_checkpoint/`，已修正並全數重跑為單一 v2 source。

## 主結果

| 指標 | exact `triton_prepare` | direct fused `triton` | paired 差異 |
|---|---:|---:|---:|
| Macro RULER | 76.1796 | 76.1796 | 0.0000 point |
| Fully correct requests | 89/130 | 89/130 | 0 |
| Official score better / same / worse | — | 0 / 130 / 0 | 0 題退步 |
| EOS 前答案 token 完全相同 | — | 129/130（99.23%） | 1 題不同 |
| 解碼後答案字串完全相同 | — | 129/130（99.23%） | 1 題不同 |
| 兩臂皆到達 EOS | — | 130/130 | — |
| Baseline 滿分題退步 | — | 0/89 | — |

13 個 task 的 paired Macro 差值全為 0。20,000 次 task-stratified paired request bootstrap 的此資料集觀察差值 95% interval 為 `[0, 0]`，因為每題官方分數都相同；**這不表示其他未測 prompt 的真實退步機率是零**。

| task | 每臂 Macro RULER | official score same | 答案 token exact |
|---|---:|---:|---:|
| niah_single_1 | 100.000 | 10/10 | 10/10 |
| niah_single_2 | 100.000 | 10/10 | 10/10 |
| niah_single_3 | 100.000 | 10/10 | 10/10 |
| niah_multikey_1 | 100.000 | 10/10 | 10/10 |
| niah_multikey_2 | 100.000 | 10/10 | 10/10 |
| niah_multikey_3 | 40.000 | 10/10 | 10/10 |
| niah_multivalue | 100.000 | 10/10 | 10/10 |
| niah_multiquery | 95.000 | 10/10 | 10/10 |
| vt | 82.000 | 10/10 | 10/10 |
| cwe | 0.000 | 10/10 | 10/10 |
| fwe | 83.335 | 10/10 | **9/10** |
| qa_1 | 50.000 | 10/10 | 10/10 |
| qa_2 | 40.000 | 10/10 | 10/10 |

## 唯一不同的答案：`105_fwe_i012`

兩版都選出相同三個 coded words、同在第 39 個 generated token 到達 EOS，官方 RULER score 都是 100，但從答案第 11 個 token 起括號內的頻次數字不同：

| | exact baseline | direct QK |
|---|---|---|
| 第 1 個詞 | `tqsvko (appears 34 times)` | `tqsvko (appears 44 times)` |
| 第 2 個詞 | `jwzcvg (appears 33 times)` | `jwzcvg (appears 43 times)` |
| 第 3 個詞 | `unhqst (appears 32 times)` | `unhqst (appears 42 times)` |

該 request 的 frozen references 只有三個 coded words，scorer 為 `ruler_string_match_all`；括號內次數**不在官方評分範圍內**。因此「130/130 官方分數相同」不能寫成「130/130 輸出完全相同」。本次沒有以這些生成的次數作獨立 correctness 保證。

## 驗證與口徑限制

- `cohort/32768/manifest.json` 包含 130 unique requests，13 tasks 各 10 題；cohort SHA256 `3cec214fae547d06e41a35cd149bdf9f66e0dba1a06463881000e6eb9030b3d0`。260 個 result 均為 `ok`，request ID/prompt hash/decode policy/backend/步數核對通過，130 題兩臂皆有 EOS。逐項檢查 130 組 command，除 output path 只差 QK backend。
- v2 source SHA256：runner `b28eebd01391635ddddd44547091237956231fa32862410b001e4a1893e4f4eb`，cache `8491eb046506ffef127bcc02b1ea877be11ad0310bf84312567235fb1753e16e`，fused QK `ed75777a4a7c7733341024bba3fc760c846d58f9ade96d0959a26eff840988e7`。direct QK 與 exact QK 實作未在這輪修改。
- 後續 orchestrator 的 process wall 為 4,641.6 秒，另有先行第一題 pilot 約 35.8 秒；合計約 4,677 秒（78.0 分鐘）。這是兩臂逐題、重新 Prefill 的 quality-run wall，**不是 TPOT**。先前 formal 52.104 ms/token 是舊 runner 快照的三題 fixed-token matched 結果，本輪沒有重新量 formal TPOT。
- 此證據支持「在這個 frozen 32K RULER 130 題 cohort，direct QK 未造成官方 task score 退步，EOS 前答案有 129/130 完全相同」。它**不支持** Selection、Attention K/V、logits exact：先前 003 trace 已證明 Selection membership 和後續 logits 不 exact；也不代表其他 context、`p`、模型、資料集或未測 prompt 的品質必然相同。
- 若研究主張是 task-level 近似品質可接受，這是強化的 cohort 內證據；若主張 output logic 完全不變，這個 candidate 仍不合格。後續是否接受一題未計分數字差異，需要以事先明定的輸出容忍規則判斷，不能只看 RULER Macro。

## Artifacts 與下一決策

- 主 raw：`results/twilight_direct_qk_ruler130_quality_20260927_v2/{manifest.json,summary.json,per_request.csv,per_request.json,run.log,requests/}`。`per_request.json` 包含 130 題兩臂預測文字；`requests/` 保留每臂 command、result、logits 和 process log，僅留本地。
- 執行與獨立分析：`scripts/run_twilight_direct_qk_ruler130_quality_v1.py`、`scripts/analyze_twilight_direct_qk_ruler130_quality_v1.py`。
- 先前三題正式速度與 numerical gate：`reports/twilight_direct_qk_gap_pilot_2026-09-27.md`。

下一個研究決策：若允許 task-level 近似，此版本可進入事先定義容忍門檻的 broader quality/generalization 測試；若必須 exact Selection/logits，仍保留 `triton_prepare` 為正式路徑，繼續尋找 exact 融合方法。**本輪不自動將 direct QK 升為 exact default，也不將品質 run wall 冒充 TPOT。**
