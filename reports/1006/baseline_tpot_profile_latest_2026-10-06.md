# 最新低VRAM版本 vs RetroInfer／FreeKV：32K TPOT 分佈

日期：2026-10-06；canonical週報2026-09-30；GitHub mirror reports/1007。研究層級5–6：量測selection／cache／model／host間的工作差距；本輪沒有更改演算法或跑130題。

## 固定條件與變因

RTX5060Ti16GB、Llama-3.2-3B-Instruct、BF16、Batch1；相同frozen timing prompts（multikey_3 i011／vt i002／qa_1 i011，31938／32639／32363 tokens）。

- 本研究：目前opt-in shared bounded Attention workspace＋D1-sized resident quota＋fused current-token insertion/snapshot，B0=4096、p=.90、Quest→exact Direct INT4 QK→dynamic Top-p→GQA union。原FlashAttention、GPU cu_seqlens、compact GPU bitmap mapping與mapped CPU miss reads；保留dense Norm/MLP子圖／exact Top-p子圖，非完整model decode graph。Production defaults未改。
- RetroInfer：retrieval=.018、estimation=.232、cache=.05、core4；官方local port，完整decode CUDA graph off。Formal正常建立index；diagnostic trace驗證完整KV hash後replay paired control index，隔離clustering變異。
- FreeKV：budget2048／sink512／recent512／page32／corr=.8／spec_ret=True、GPU pool1536MiB；conservative CPU dispatch、原background worker overlap停用，完整CUDA graph off。不是完整原論文async效率重現。
- Selector、retrieval數量與品質不同，**不是equal-quality／equal-VRAM對照**。沒有在本輪把兩篇論文換成你的selector，不能將各component差值當直接可移植收益。

## 未插樁正式TPOT分佈

三題×兩輪×三methods，共18 fresh processes、每method186 D2–D32 samples；32 fixedtoken1 forwards，D1 initialization/warming excluded，保留D2；synchronized CPU wall，無profiler／record_function／replay。交錯methods並反轉第二輪順序。Pooled token quantiles不作confidence interval；request／token非獨立樣本。

| 方法 | Mean ms | Median ms | P90 ms | P95 ms | run mean範圍 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本研究 slim fused | 30.098 | 29.733 | 32.382 | 32.907 | 28.994–31.812 |
| RetroInfer local port | 22.250 | 22.023 | 22.895 | 23.375 | 21.913–22.708 |
| FreeKV conservative dispatch | 30.278 | 29.581 | 31.772 | 34.132 | 29.222–31.446 |

這是新batch的正式數字；先前本研究29.885338ms是另一組已驗證batch，兩者不互相覆寫，也不把batch波動稱為新的optimization。

## GPU工作量分佈（diagnostic）

每method三題、D3–D5，共9 profiled tokens；paired control/trace、Kineto kernel/memcpy/memset真正activity durations，按launch correlation最內層CPU phase分類。沒有加intra-token sync。**以下不是正式TPOT的互斥分解**，不可比例縮放到formal mean。Across-stream sum可能重疊，GPU busy union另列。

| GPU activity ms/profiled token | 本研究 | RetroInfer | FreeKV conservative |
| --- | ---: | ---: | ---: |
| 模型kernels | 16.269 | 15.960 | 16.839 |
| Selection／correction | 5.026 | 1.224 | 2.245 |
| KV/cache／搬移／組裝 | 5.069 | 1.409 | 5.927 |
| Attention | 1.414 | 0.868 | 0.692 |
| 未分類 | 0.186 | 0.047 | 0.061 |
| Activity sum | 27.964 | 19.508 | 25.763 |
| GPU busy union | 27.963 | 19.508 | 25.369 |

Cache activity含GPU hit copy、CPU mapped miss reads、bitmap／snapshot／layout等，不全是PCIe搬移；mapped host reads不出現在一般H2D DMA表，也不能由該表推論PCIe bytes=0。Model包括可辨識projection／MLP／norm／RoPE／embedding／LM head，未覆蓋activity留未分類。

## CPU互斥scope（diagnostic）

同一main-thread取最內層scope，包含在scope內等待GPU／memory／launch；不是CPU純計算，也不能與GPU表相加。

| CPU exclusive ms/profiled token | 本研究 | RetroInfer | FreeKV conservative |
| --- | ---: | ---: | ---: |
| Model API | 7.286 | 7.366 | 9.898 |
| Selection | 14.040 | 8.104 | 7.732 |
| KV/cache | 9.062 | 3.369 | 9.713 |
| Attention | 3.096 | 3.378 | 0.957 |
| 顯式同步 | 2.197 | 2.169 | 2.096 |
| 未標記host | 9.865 | 2.844 | 4.884 |

## 同一trace的時間帳

這裡才以同一個window確認wall=GPU busy union＋GPU inactive。Inactive表示trace捕捉的kernel／memcpy／memset皆未活動，包含host control／dispatch／同步／排程／profiler overhead，不能全稱為Python成本或可移除成本。尤其不能把formalTPOT減diagnosticGPU工作量叫overhead。

| Diagnostic window ms/profiled token | 本研究 | RetroInfer | FreeKV conservative |
| --- | ---: | ---: | ---: |
| Wall | 45.547 | 27.230 | 35.281 |
| GPU busy union | 27.963 | 19.508 | 25.369 |
| GPU inactive | 17.584 | 7.722 | 9.912 |

Inactive所在CPU scope：

| scope ms/profiled token | 本研究 | RetroInfer | FreeKV |
| --- | ---: | ---: | ---: |
| model | 3.499 | 1.760 | 3.282 |
| selection | 4.960 | 1.349 | 1.163 |
| cache | 4.263 | 1.799 | 3.270 |
| attention | 1.841 | 2.084 | 0.766 |
| explicit_wait | 0.106 | 0.050 | 0.053 |
| unscoped | 2.915 | 0.680 | 1.377 |

## Payload／logits與工具限制

Control／trace三題D2與D32的saved logits有限性核對；trace只從D3啟動。Gate如下，maxabs為D2／D32差：

| 方法／request | exact | D2 maxabs | D32 maxabs |
| --- | --- | ---: | ---: |
| retroinfer／001_niah_multikey_3_i011 | True | 0.000000 | 0.000000 |
| retroinfer／002_vt_i002 | True | 0.000000 | 0.000000 |
| retroinfer／003_qa_1_i011 | True | 0.000000 | 0.000000 |
| freekv／001_niah_multikey_3_i011 | False | 0.500000 | 0.687500 |
| freekv／002_vt_i002 | False | 0.164062 | 1.552734 |
| freekv／003_qa_1_i011 | False | 1.031250 | 1.687500 |
| twilight／001_niah_multikey_3_i011 | True | 0.000000 | 0.000000 |
| twilight／002_vt_i002 | True | 0.000000 | 0.000000 |
| twilight／003_qa_1_i011 | True | 0.000000 | 0.000000 |

FreeKV若D2已不同，該差異發生在profiler activation之前，不能歸因profiling；前輪未插樁repeat也有差異，本輪沒有定位根因。其control/trace末態resident audit皆0mismatched／0inverse errors，排除最新兩頁，**不證明逐步attention正確**。FreeKVcomponent／latency只作local implementation觀察，不能作完整correctness已確認官方baseline。RetroInferindex replay只在diagnostic，不作greedy品質gate。本研究完整130題尚未跑；前輪13task smoke答案相同不替代130題。

CUPTI12.9.79 library僅在trace subprocess以LD_PRELOAD載入；formal不載入。9traces各有>100GPUkernels，source current/snapshot hashes／prompt hash／returncodes／rawlatency重算與paired gates核對。Diagnostic window含profiler overhead，不能當正常TPOT。完整model CUDA graph設定與各方法子圖差異如上，未稱same backend。

Timeline analysis第一版因epoch-sized float timestamp分段的sub-ns precision累積差（最大1.220703e-7ms/token）觸發time-account assert；保留failed log／partial summary。V2先將timestamp轉為relative microseconds，再以同樣嚴格1e-7ms tolerance核對，九traces均通過。沒有放寬gate或重跑GPU來挑數字；v2 analyzer hash寫入summary。

## 觀察與下一決策

- 模型GPU工作：本研究16.269、RetroInfer15.960、FreeKV16.839ms/profiled token。
- Selection GPU工作：本研究5.026、RetroInfer1.224、FreeKV2.245ms/profiled token。
- KV/cache GPU工作：本研究5.069、RetroInfer1.409、FreeKV5.927ms/profiled token。
- Attention GPU工作：本研究1.414、RetroInfer0.868、FreeKV0.692ms/profiled token。

這些工作量差異可指出下一個hotspot，但不是可保證TPOT節省的上界；仍需相同選取與品質的單一ablation。先檢查新版assembly與selection內部kernel，而不是假設模型本體慢一倍或把所有residual稱CPU overhead。本輪只量測，不修改研究方法／baseline，不啟動130題。

## Artifacts與公開檔案

Artifact root：results/baseline_tpot_profile_latest_20261006/；manifest.json含source／prompt／CUPTIhash；source_snapshot/、control/、trace/、formal/、summary.json、formal.csv。每trace有trace_summary.json，完整kernel／DMA明細與互斥CPU分類。Scripts依賴既有local模型、官方external checkouts／venv與frozen cohort，不是standalone installer。

本輪應同步：

- reports/1007/baseline_tpot_profile_latest_2026-10-06.md
- scripts/baseline_profile_latest/profile_case.py
- scripts/baseline_profile_latest/run_profiles.py
- scripts/baseline_profile_latest/analyze_profiles.py
- scripts/baseline_profile_latest/analyze_profiles_v2.py
- scripts/baseline_profile_latest/report_profiles.py

未納入raw JSON/CSV/log/PT／index／timeline／weights／CUPTI binary／source snapshots／external checkouts；沒有本輪production source變動。準備public mirror後byte核對，未commit／push。
