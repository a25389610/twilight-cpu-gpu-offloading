# 最新三方法比較：本研究／FreeKV／RetroInfer

核對日期2026-10-06；RTX5060Ti16GB／Llama-3.2-3B-Instruct／BF16／Batch1／32K。

## 正式TPOT、decode live VRAM及130題品質

TPOT與memory由最新同批三題×兩輪×三methods的18freshprocesses重算；fixedtoken1、32forwards、D1排除、D2–D32 synchronized CPU wall、各method186samples。Quality是各方法最新保存的同frozen130cohort（13tasks×10）評估，與formal分開；本輪重新核對390筆request/prompt hashes、設定與官方scorer，不新增benchmark。

| 方法 | TPOT mean ms | median ms | P95 ms | 扣權重 MiB | 扣權重 GiB | RULER /100 | 滿分題 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 本研究 slim fused | 30.097575 | 29.733300 | 32.907092 | 1089.201335 | 1.063673 | 75.359077 | 88/130 |
| FreeKV conservative dispatch | 30.278378 | 29.580730 | 34.132226 | 1612.891602 | 1.575089 | 70.359000 | 79/130 |
| RetroInfer local port | 22.250318 | 22.022713 | 23.374521 | 447.478190 | 0.436990 | 74.859077 | 87/130 |

Memory統一為D32 final live PyTorch allocated減實際model weights，包含KV／metadata／workspace／logits等，不是純KV、不含driver/opaque allocations，也不是prefill peak／reserved。FreeKV1536MiB pool全部算入，不用active pages取代實際配置。RetroInfer loader有額外embed/lm_head權重backing，實際weight bytes不同，逐方法實際扣除。

| 方法 | 實際權重 MiB | 含權重D32 allocated MiB | 含權重GiB |
| --- | ---: | ---: | ---: |
| 本研究 slim fused | 6127.833984 | 7217.035319 | 7.047886 |
| FreeKV conservative dispatch | 6127.833984 | 7740.725586 | 7.559302 |
| RetroInfer local port | 6879.333984 | 7326.812174 | 7.155090 |

## GPU activity分佈（diagnostic，ms/profiled token）

每method三題D3–D5、共9tokens，Kineto kernel/memcpy/memset以launch correlation分類；沒有新增intra-token sync。**不能將本表加總成formalTPOT，不與CPU表相加，FreeKV跨stream部分重疊。**

| 項目 | 本研究 | FreeKV | RetroInfer |
| --- | ---: | ---: | ---: |
| 模型kernels | 16.269445 | 16.839229 | 15.959504 |
| Selection／correction | 5.025615 | 2.244925 | 1.224455 |
| KV/cache／搬移／組裝 | 5.068653 | 5.926532 | 1.408823 |
| Attention | 1.414351 | 0.691638 | 0.868121 |
| 未分類 | 0.185963 | 0.060651 | 0.047291 |

本研究assemble_zero_copy約4.391ms/profiled token、fusedsnapshot約.496ms；不是純H2D。Mapped host reads無一般DMAbytes不代表PCIe沒有讀取。Model activity接近，但selection／cache對RetroInfer較多；差值不是可保證的TPOT收益。

## CPU互斥scope（diagnostic，ms/profiled token）

包含scope內GPU等待／copy發送／同步／host control，不是CPU純計算。

| 項目 | 本研究 | FreeKV | RetroInfer |
| --- | ---: | ---: | ---: |
| Model API | 7.286245 | 9.898147 | 7.366270 |
| Selection | 14.040063 | 7.732073 | 8.103644 |
| KV/cache | 9.062189 | 9.712902 | 3.368538 |
| Attention | 3.096178 | 0.957198 | 3.378479 |
| 顯式同步 | 2.197256 | 2.096420 | 2.169008 |
| 未標記host | 9.865182 | 4.884305 | 2.844289 |

## 同一diagnostic timeline的時間帳

同windowwall=GPU busy union＋GPU inactive；inactive含profiler／host／launch／排程等，不能全稱CPU運算，也不能移植為formaloverhead。

| 項目 ms/profiled token | 本研究 | FreeKV | RetroInfer |
| --- | ---: | ---: | ---: |
| 插樁wall | 45.547112 | 35.281047 | 27.230229 |
| GPU busy union | 27.963430 | 25.368838 | 19.508194 |
| GPU inactive | 17.583682 | 9.912208 | 7.722034 |

## 各task品質（每task10題）

| Task | 本研究 | FreeKV | RetroInfer |
| --- | ---: | ---: | ---: |
| cwe | 0.000000 | 0.000000 | 0.000000 |
| fwe | 86.668000 | 96.667000 | 86.668000 |
| niah_multikey_1 | 100.000000 | 100.000000 | 100.000000 |
| niah_multikey_2 | 100.000000 | 80.000000 | 100.000000 |
| niah_multikey_3 | 30.000000 | 0.000000 | 20.000000 |
| niah_multiquery | 95.000000 | 82.500000 | 92.500000 |
| niah_multivalue | 100.000000 | 97.500000 | 100.000000 |
| niah_single_1 | 100.000000 | 100.000000 | 100.000000 |
| niah_single_2 | 100.000000 | 100.000000 | 100.000000 |
| niah_single_3 | 100.000000 | 100.000000 | 100.000000 |
| qa_1 | 50.000000 | 50.000000 | 50.000000 |
| qa_2 | 40.000000 | 40.000000 | 40.000000 |
| vt | 78.000000 | 68.000000 | 84.000000 |

## Operating point與限制

- 本研究：B0=4096／p=.90，shared bounded Attention workspace＋D1residentquota＋fusedsnapshot，原FlashAttention、GPUcu；最新完整130結果75.359077、88滿分，129/130IDs與保存原版相同，130scores都同。先前29.885338ms是另一matchedbatch，本表用新同批30.097575ms，不將波動稱新改版。
- FreeKV：budget2048／sink512／recent512／page32／corr=.8／spec_ret=True、GPUpool1536MiB，conservative dispatch，停用原backgroundoverlap；不是完整官方async效率重現。CrossprocessD2在profiler啟動前已有差異，根因未定位。Endresidentaudit0差異不證明每步attention，因此70.359與efficiency是保存localport觀察，不是品質repeatability已確認。
- RetroInfer：retrieval=.018／estimation=.232／cache=.05／core4、完整CUDAgraphoff；diagnosticindexreplay、formal正常建index；quality有13個前導pilot＋117新processes，沿用同cohort與設定。未為matching你的75.359品質調budget。
- 三者selector與有效KV數不同、不是equal-quality／equalVRAM或samebackend。FreeKV與你formalmean相差.181ms，不宣稱統計顯著勝負；RetroInfer較低VRAM／較快，但平均品質少.5point仍需Pareto比較。不能用組件相減推論causal收益。

## Artifacts與完整輸出

相對canonical研究root headinfer/headinfer_reproduction：

- results/baseline_tpot_profile_latest_20261006/{summary.json,comparison_verified.json,formal/,control/,trace/}：本表formal／memory／diagnostic。
- results/tpot_slim_fused_20261006/quality130/{quality_verification.json,quality_verified.csv,outputs_130.md}：本研究130品質。
- results/freekv_ruler130_20261005_v1/{summary.json,quality.csv,outputs_130.md,quality/}：FreeKV130品質。
- results/retroinfer_ruler130_quality_20261005_v1/{summary.json,quality.csv,requests/}：RetroInfer130品質；每request/result.json有prediction與IDs。

本輪應同步reports/1007同名report與scripts/baseline_profile_latest/consolidate_comparison.py；rawJSON/CSV/log/PT／完整prediction／weights／trace／snapshots／externalcheckout未納入。沒有改production source／重跑GPU／commit／push。
