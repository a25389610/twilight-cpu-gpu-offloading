# 32K 低 VRAM Twilight：direct QK 首選近似路徑決策與當前 source TPOT 確認

日期：2026-09-28（Asia/Taipei）

## 決策範圍

使用者已明確接受此設定下 task-level 品質維持、少量未計分文字差異的取捨。**在已測 Llama-3.2-3B-Instruct、BF16、RTX 5060 Ti 16 GB、Batch 1、32K、Twilight dynamic Top-p `p=.90`、Quest `B0=8192`、previous-token resident cache + GPU mapped CPU KV read 條件下，direct fused INT4 QK 是目前首選的低 VRAM 快速近似路徑。**它仍需明確使用 `--twilight-qk-backend triton`；CLI 預設仍為 `pytorch`，本次 matched exact 研究對照則明確使用 `--twilight-qk-backend triton_prepare`。完整 historical KV 仍在 CPU pinned memory，沒有 full GPU KV mirror。

此決策不表示 Selection、Attention K/V 或 logits bit-exact，也不外推到其他 Context、`p`、模型或品質資料集。論文若使用此路徑，應標為 approximate implementation 並與 exact 路徑分開報告。

## 當前 source 的 formal TPOT

為確認品質診斷 runner 加入 default-off EOS-stop 開關後的當前 source，重新執行三題各兩輪、arm 順序交錯的 matched formal TPOT。兩臂 command 除 output path 只差 `--twilight-qk-backend triton_prepare`／`triton`；均不開 greedy/profiling flags。D1 warm-up，D2–D32 共 31 個 fixed Decode tokens，以 synchronized wall 計時，模型載入與 Prefill 排除。

| 32K request | exact `triton_prepare` | direct `triton` | 差值 | 相對改善 |
|---|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 68.739 | 54.574 | -14.165 | 20.61% |
| 002_vt_i002 | 68.206 | 55.033 | -13.173 | 19.31% |
| 003_qa_1_i011 | 67.930 | 55.518 | -12.411 | 18.27% |
| **六組平均** | **68.292** | **55.042** | **-13.250 ms/token** | **19.40%** |

六組 paired 差值都改善；D32 live CUDA allocated 兩臂差平均 **+4.765 MiB**，process peak allocated paired 差均為 **0 MiB**。P4 logits hash 相同，D2/D32 logits hash 不同，與既有 nonexact gate 一致。額外 003 第三組在同一時段為 **68.667 → 55.404 ms/token**，方向與絕對值一致，但不併入上表六組平均。

09/27 前一 source/量測時段的六組 matched formal 平均是 **65.309 → 52.104 ms/token**，差 **-13.205 ms/token（-20.22%）**。當前 source 兩臂絕對 TPOT 均約高 3 ms，差值仍約 13 ms；這個絕對值漂移的原因未確認，不將兩批 run 混成同一平均。因此目前應把 **55.042 ms/token** 稱為最新當前 source 的 matched 數字；**52.104 ms/token** 保留為先前已測最快數字，不保證每次機器狀態都重現。

Raw：`results/twilight_direct_qk_current_confirm_20260928_v1/{manifest.json,summary.json,summary.csv,rep1/,rep2/}`；額外 003 `rep3/`。正式 runner/source SHA256 和逐組 command/result 保存在 manifest 與 raw 中。第一次誤用 003 的 Full-flat denominator 作為所有 request seed command，已在執行初期中止並保留 `results/twilight_direct_qk_current_confirm_20260928_v1_failed_seed/`；正式六組改用各 request 自己的既有 command。

## 品質接受的證據與明確代價

同一 low VRAM execution path 的 current-source 32K RULER frozen cohort，13 tasks × 10 題，exact/direct 各重新 greedy 生成：Macro **76.1796 → 76.1796**、官方逐題 **0 better／130 same／0 worse**、EOS 前答案 token **129/130 exact**。唯一不同的 `105_fwe_i012` 三個 coded words 相同，但括號次數 `34/33/32` → `44/43/42`；RULER scorer 不評次數，兩臂均 100。先前 003 Selection trace 已見一個 Top-p 邊界 token 首次不同，六組後續 logits hash 不 exact。品質 cohort 因此支持已測任務品質無退步，但不構成輸出 exact 或一般化保證。詳細見 `reports/twilight_direct_qk_ruler130_quality_2026-09-28.md`、`reports/twilight_direct_qk_gap_pilot_2026-09-27.md`。

## 使用與下一個研究決策

- 需要最快的已測 32K low VRAM task-level 路徑：使用上述 resident mapped-KV flags，並加 `--twilight-qk-backend triton`。完整可重跑正式命令見 `results/twilight_direct_qk_current_confirm_20260928_v1/rep1/003_qa_1_i011/direct/command.json`。
- 需要 bit-exact Selection/logits 對照：同一命令改用 `--twilight-qk-backend triton_prepare`；兩臂只差此 backend。CLI 不改成自動 direct，以保留舊 formal 實驗與 exact control 的明確身份。
- 此輪已完成使用者要求的 130 題品質檢查與當前 source formal 確認。下一個研究方向若改變 Selection 演算法，應在此明確標記的 approximate 系統路徑上重新測 paired TPOT 與品質，並保留 exact control。
