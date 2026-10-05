# 10/05 最新 CUDA execution preset 的 B0=8192／4096／2048 比較

## 問題與事前判斷條件

只減少 Quest 候選 token budget B0，是否能將約38ms的目前版本降到接近30ms？固定 p=.90、GPU-cu／buffer swap、union fusion、explicit-order CUDA QK／Quest、原attention與模型。事前130題啟動門檻：candidate mean≤32ms、相對同批8192低至少15%、六個request×rep配對皆快，並通過same-B0 implementation gate。

## 正式條件

RTX5060Ti16GB／Llama-3.2-3B-Instruct／BF16／Batch1，三個 frozen 32K timing prompts（niah_multikey_3、vt、qa_1），兩輪×三arms，18fresh processes。每次32fixedtoken1 forwards，排除D1，D2–D32 synchronized CPU wall；每arm186samples，無profiler、無payload capture，交錯arms與第二輪反向題目順序。不同B0預期selection與logits不同，不以跨B0 bit-exact作quality標準。

## 正式結果

| B0 | 平均 TPOT ms | 同批改善 ms | 改善 % | selected history MiB/token | CPU miss logical MiB/token | hit ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 37.428709 | 0.000000 | 0.0000 | 507.817 | 46.833 | 90.76% |
| 4096 | 33.799000 | 3.629709 | 9.6977 | 293.693 | 28.922 | 90.14% |
| 2048 | 33.059525 | 4.369184 | 11.6734 | 172.519 | 17.003 | 90.13% |

Logical selected/miss bytes來自resident history/miss rows×512B（BF16 K+V、head_dim128），不是PCIe transactions或memory inventory。Mapped CPU reads下DMA counter可能為0，不能因此聲稱無CPU misses。Hit ratio先於每process按rows加權，再對六runs取平均。

### 六組配對

| Pair | 8192 ms | 4096 ms | 2048 ms |
| --- | ---: | ---: | ---: |
| rep1/001_niah_multikey_3_i011 | 39.241272 | 36.099729 | 35.049887 |
| rep1/002_vt_i002 | 39.393123 | 36.037796 | 35.065096 |
| rep1/003_qa_1_i011 | 36.207291 | 32.511702 | 32.035376 |
| rep2/001_niah_multikey_3_i011 | 36.289933 | 32.551796 | 32.120275 |
| rep2/002_vt_i002 | 36.315492 | 32.789161 | 32.230044 |
| rep2/003_qa_1_i011 | 37.125145 | 32.803816 | 31.856471 |

## 驗證與研究判斷

新budget的QK synthetic micro：8192／4096／2048都與原Triton reference bit-exact。Formal returncodes／status、B0／p／decode policy、原始CSV重算、相同prompt、env與command僅B0/output不同，以及stage frozen source與current hashes均核對。resident hit+miss=history、logicalbytes=row×512檢查通過。

另128fixedtoken1 forwards比較CUDA與reference，**同一B0**的checkpoint/final logits、selection／union／Attention K/V／newKV／resident traces相同。Payload capture D1／D2／D32／D128，並非all-input proof或130題quality。

**沒有任何candidate達到事前接近30ms門檻，因此依使用者條件未啟動130題。** 降低B0有改善，但這輪否定「只降B0即可到約30ms」在本cohort／設定下的假設。不能將舊runtime的RULER分數貼到本輪B0=4096或2048；本輪沒有自然greedy品質結果。

Selected/miss減量與TPOT未等比例下降是觀察，不是原因證明。沒有對各B0做component profiling，不能定量歸因剩餘時間；本次也未量VRAM。下一決策是另評估重用selection結果或cache/attention介面成本，尚未在本輪實作。

另20個formal/gate logits artifacts的82個checkpoint tensors皆finite；`gate_verification.json`記錄bounded gate與source核對。`run_b0_quality.py`已準備，但門檻未通過，沒有執行；它不是已完成quality結果。

## Artifacts與公開範圍

本地`results/tpot_b0_comparison_20261005/`：formal／micro／gate、stage manifests/source snapshots、analysis與verification；rawCSV/results/logits/process logs保留本地。Source是`scripts/tpot_execution_opt/{run_b0_comparison,probe_b0_qk,analyze_b0_comparison,report_b0_comparison,verify_b0_gate,run_b0_quality}.py`，依賴既有CUDA execution adapters與canonical source/runtime，非standalone installer。

報告prepare到GitHub mirror `reports/1007/`，本輪scripts同步到mirror；raw JSON/CSV/log/PT、模型、source snapshot、compiler/binary與external checkout未納入。Production defaults未改，未commit／push。
