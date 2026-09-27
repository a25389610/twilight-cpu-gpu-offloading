# Twilight fused direct QK 的 TPOT 加速上限與 correctness audit（2026-09-27）

## 問題與假設

先前記得原生 Twilight operator 可在 10 ms 內完成，但那是與本系統不同口徑的 isolated operator benchmark，不包含 Llama 模型計算、Quest、Top-p、GQA union、CPU mapped KV read、Attention、resident 更新及控制流程。本輪在維持完整 historical KV 位於 CPU、使用低 VRAM previous-token resident 路徑的條件下，檢查 Twilight 第二輪 INT4 candidate prepare 與 QK 實作是否仍有顯著 TPOT 空間。

現行 exact 對照 `triton_prepare` 先以 Triton unpack/dequantize 形成 FP32 estimated K，再以 `torch.matmul` 算 QK。已存在的實驗性 `triton` backend 把 INT4 prepare 和 QK 合成單一 Triton kernel，避免 FP32 candidate K materialization 及獨立 matmul。兩者輸入資料與後續 Top-p、GQA、CPU mapped read、Attention 路徑相同；但 FP32 reduction order 不同，必須獨立檢查 Selection 與 logits。

## 固定實驗條件與證據口徑

- Llama-3.2-3B-Instruct、BF16、RTX 5060 Ti 16 GB、Batch 1、32K、Twilight dynamic Top-p `p=.90`、Quest `B0=8192`、三題 `001_niah_multikey_3_i011`、`002_vt_i002`、`003_qa_1_i011`。
- 低 VRAM exact baseline 保留 `--twilight-exact-top-p-graph --twilight-resident-precompute-gpu-mapping`、previous-token resident cache、GPU mapped CPU KV read，以及既有 VRAM audit 的 alias/lazy/on-demand 選項。唯一性能 ablation 是 `--twilight-qk-backend triton_prepare` → `triton`。
- 正式 TPOT 是每次 unprofiled 執行的 synchronized wall time，D1 warm-up、D2–D32 共 31 tokens。每題兩組 fresh matched pair，第二輪交錯 arm 順序；模型載入與 Prefill 不計入 TPOT。這是同 source/同條件的 candidate 比較，不和不同快照的 65.529 ms/token 混算。
- CUDA Event Selection 數字來自另外的 003 診斷執行，各 3 個 post-timing tokens、每 token 28 個 layer-batched Selection 呼叫。CUDA intervals 可定位 kernel 差異，但不當作正式 TPOT 的可加分解。
- raw：`results/twilight_official_gap_20260927_v1/direct_qk_matched/`，每組保留 command、result、process log、per-token CSV、logits；彙總 `summary.json`、`summary.csv`。逐項核對六組 command，除 output path 外只差 QK backend。003 profile：`profile003/`、`direct_qk_profile003/`，彙總 `profile_summary.json`。大型 Selection trace、greedy probe 及 quality score 見同一個 results 根目錄；raw 僅留本地。

## Formal TPOT：六組 matched pair

| request | rep 1 baseline → direct | rep 2 baseline → direct | 兩輪 baseline → direct | paired 差值 |
|---|---:|---:|---:|---:|
| 001 NIAH multikey 3 | 65.294 → 52.042 | 64.897 → 51.942 | 65.095 → 51.992 | -13.103 ms/token |
| 002 VT | 65.206 → 52.250 | 65.488 → 51.984 | 65.347 → 52.117 | -13.230 ms/token |
| 003 QA 1 | 65.376 → 52.174 | 65.593 → 52.231 | 65.485 → 52.202 | -13.283 ms/token |
| **六組平均** | | | **65.309 → 52.104** | **-13.205 ms/token（-20.22%）** |

六組差值均為負，範圍 -12.955 至 -13.504 ms/token。這證明此 fused direct QK 在本機本設定可顯著降低 TPOT，**但不構成 exact 替換方案**。

### 003 Selection diagnostic，與 formal TPOT 分開

| CUDA Event interval mean | exact `triton_prepare` | direct `triton` |
|---|---:|---:|
| INT4 prepare | 8.898 ms/token | 融於 direct QK |
| FP32 QK matmul | 6.938 ms/token | 融於 direct QK |
| Fused INT4 prepare + QK | — | 3.246 ms/token |
| Selection core | 25.785 ms/token | 12.705 ms/token |
| Quest 第一輪 | 3.787 ms/token | 3.716 ms/token |
| Top-p | 4.613 ms/token | 4.652 ms/token |

diagnostic 的 formal D2–D32 分別為 68.288 與 52.250 ms/token，已與上表獨立標示。prepare + matmul 是相鄰子階段，不應把跨 CPU/GPU 或與 Selection core 重疊的數值再相加。數字支持瓶頸集中於本地 `triton_prepare` 的 FP32 materialization 與 matmul，而非原生 operator 的時間可直接當成本系統 TPOT。

## Correctness、品質與限制

- 六組 P4 logits hash 全同；001/002 的 D1 仍同，D2/D32 與 final 不同；003 從 D1 起不同。direct 不是 bit-exact 的 Selection / logits 路徑。
- 003 首個 Selection membership mismatch 出現在 **D1 layer 8、KV group 5**：只差一個 history position（baseline 包含 29504，direct 不包含）；此前 D1 layers 0–7 exact。後續 layer 及 token 的大範圍 Selection 差異受到 hidden state 連鎖變化影響，不可全歸因於當層 QK 計算誤差。
- 同一合成輸入的 QK scalar max abs 誤差約 `1.5e-5`、mean abs 約 `1e-6`；這種小誤差仍可能讓 Top-p 閾值附近的 token 改變 membership。003 已測 checkpoint logits max abs：P4 0、D1 0.140625、D2 0.109375、D32 0.125；四個 checkpoint top-1 相同，不能據此宣稱完整輸出 exact。
- 額外 default-off `--greedy-probe` 僅用於 answer 檢查，不用於正式 TPOT。13 個 32K RULER request/answer pairs 覆蓋三題及另外十個 task type；以第一個 EOS (`128009`) 截斷後，**13/13 的 greedy answer token IDs 與 exact baseline 相同**。初版 `quality_summary.json` 的分數使用簡化 reference 比對；其後以專案既有 `official_ruler_score` 重新計分，13/13 兩臂官方 task score 也相同，且這 13 題的簡化分數碰巧與官方分數一致；重新計分 raw 見 `quality_official_rescore.json`。002 及另外三例在 EOS 後的固定步數 token diverge。這是一題一筆的 preliminary quality probe；CWE、QA2 的兩臂分數本來都是 0，不能推論整體品質不降，也不能替代 Selection/Attention/logits exact gate。

## VRAM

下表是 formal runs 的兩輪平均，process peak 包含模型載入/Prefill；D32 live 也含模型權重，非純 KV 增量。兩臂完整 historical KV 仍在 CPU pinned memory，未加 full GPU KV mirror。

| request | D32 CUDA allocated baseline → direct | 差值 | process peak allocated baseline → direct | peak reserved baseline → direct |
|---|---:|---:|---:|---:|
| 001 | 7453.486 → 7463.759 MiB | +10.272 MiB | 9087.739 → 9087.739 MiB | 9542 → 9542 MiB |
| 002 | 7473.132 → 7474.268 MiB | +1.136 MiB | 9146.864 → 9146.864 MiB | 10122 → 10122 MiB |
| 003 | 7481.902 → 7484.788 MiB | +2.885 MiB | 9123.971 → 9123.971 MiB | 10090 → 10090 MiB |

六組 D32 live allocated 差值平均 +4.765 MiB；process peak allocated 與 peak reserved 無差。這個 experimental fused kernel 沒有以額外數百 MiB VRAM 換取速度。這裡尚未做單獨的 Decode temporary peak unique-storage inventory，因此不能聲稱所有時間點的暫存 VRAM 完全相同。

## 其他單機制嘗試與下一決策

- INT4 prepare Triton tile 8/16/32/64/128、warps 4/8：standalone bit-exact，但約 0.416–0.446 ms/layer，未找到顯著加速。`matmul`、`bmm`、`baddbmm`、einsum 的等價 QK standalone 約 0.291–0.294 ms/layer，亦無明顯收益。
- 合成輸入上試 direct QK FMA reduction 變體；最佳 scalar bit-exact fraction 仍僅約 30.6%。整合 `triton_fma16` 的 003 pilot D1/D2 logits 不 exact，已撤回該 source 實驗，未納入 matched TPOT。單輪非 matched TPOT 不可當收益。
- 原生 Twilight Top-p kernel 對合成 4D arbitrary logits 的 24 rows × 5 seeds，與本地 exact Top-p 的 count/membership 不同；原生實作對最高 probability 的位置有假設，不能直接替換目前的 Top-p。這也說明「原生 10 ms」無法等價推回本系統。

**決策：** exact 低 VRAM 路徑仍使用 `triton_prepare`。direct `triton` 保留實驗性 opt-in，提供約 13.2 ms/token 的大幅 TPOT 空間與低 VRAM 成本，但 Selection、logits gate 未過。若研究目標容許數值近似，下一步應用更大 request cohort 和明確品質容忍門檻驗證；若要求 exact Selection，下一步應尋找能保留 FP32 matmul reduction 結果的融合/重用方法，或針對 Top-p 閾值邊界做可證明正確的 fallback，先驗證 correctness 再量正式 TPOT。這些方向目前都尚未證實可達相同速度。

source provenance：`source/headinfer/headinfer/twilight_offload_cache.py` SHA256 `8491eb046506ffef127bcc02b1ea877be11ad0310bf84312567235fb1753e16e`；`source/headinfer/headinfer/twilight_fused_qk.py` SHA256 `ed75777a4a7c7733341024bba3fc760c846d58f9ade96d0959a26eff840988e7`。現有 source 的 experimental direct backend 未修改。runner `scripts/run_ruler_partial_h2d_tpot_case_v1.py` 只加入 default-off `--greedy-probe`；正常 formal run 不受其執行邏輯影響。
