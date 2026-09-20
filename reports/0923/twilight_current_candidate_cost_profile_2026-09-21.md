# Twilight current candidate TPOT 成本組成 profiling baseline

日期：2026-09-21  
性質：diagnostic / profiling baseline；不是新的 execution candidate  
結論適用範圍：`003_qa_1_i011`、32K、`p=.90`、Batch 1、32 fixed decode steps

## 摘要

- 三次完全未開 profiling 的 fresh D2–D32 TPOT 為 **121.812 / 122.739 /
  120.528 ms/token**；mean **121.693 ms/token**，median 121.812，range
  120.528–122.739，sample standard deviation 1.110 ms。
- normal-overlap profile 顯示：Selection core 約 **27.893 ms/token**；從每層
  Top-p membership 完成到 Attention 開始的 post-selection exposed interval 合計約
  **73.989 ms/token**；named Transformer model compute active union 約
  **27.308 ms/token**。這些取自 diagnostic tokens，且不同 parent scope 不能相加。
- CPU selected-KV gather active **38.832 ms/token**、selected-KV H2D active
  **39.086 ms/token**；兩者共同時鐘下重疊 **26.746 ms/token**。亦即約 68.9%
  gather active time、68.4% H2D active time互相重疊；各自仍有約 **12.085 / 12.340
  ms/token** 沒被對方覆蓋。
- selected-KV H2D 與 named GPU model compute 的交集為 **0.000 ms/token**。目前
  H2D 在 GPU default-stream compute 前完成，並沒有藏在 Attention/MLP 下。
- Selection 內最大 active parent 是 INT4 preparation + approximate QK，約
  **16.845 ms/token（Selection core 的 60.0%）**；其後是 Top-p/ranking parent
  6.937 ms 與 Quest parent 3.682 ms。
- 本輪沒有修改 Selection algorithm、KV layout、gather、projection 或 pipeline；
  沒有更新 `CURRENT_IMPLEMENTATION.md`。

## 實驗條件與證據口徑

| 項目 | 條件 |
|---|---|
| Model | `meta-llama/Llama-3.2-3B-Instruct`, BF16, FlashAttention 2 |
| Request | `003_qa_1_i011`；實際 prompt 32,363 tokens |
| Twilight | `p=.90`, Quest block 16, Sink 64, Recent 256 |
| Current path | `triton_prepare`, GPU compact GQA union, CPU flat gather, 4-chunk gather/H2D, direct attention layout, token-batched new-KV D2H |
| Decode | D1 warm-up；formal TPOT 使用 D2–D32，共 31 tokens |
| Hardware/runtime | RTX 5060 Ti；PyTorch 2.7.0+cu128；CUDA 12.8 |
| Source base | HeadInfer `dee81b223e64cfc8b6bd58e36d9f8079bae58691`，另加 default-off diagnostic instrumentation |

量測分為三組，不能混用：

1. **Formal TPOT**：三次完整 fresh process，沒有任何 profile flag。
2. **Normal-overlap timeline**：正常 streams、正常 4-chunk gather/H2D pipeline，不在
   component boundary synchronize；一個 diagnostic token 結束後才 resolve CUDA
   Events。共 3 tokens。
3. **Detailed component active time**：沿用既有 `--profile-breakdown` 與 Selection
   CUDA Events，取得 Selection subphase；因既有 profiler 在 Selection boundary 會
   synchronize，只用來報 component active time，不作主要 critical-path timeline。

normal-overlap diagnostic wall mean 是 130.872 ms，比 fresh TPOT mean 高 9.179 ms
（+7.54%），所以 timeline run 的 wall time不作正式效能宣稱。另試過
PyTorch/Kineto trace，但 CUPTI 回報 `CUPTI_ERROR_INVALID_DEVICE`，且 token wall 膨脹至
約 181.6 ms；該 trace **完全排除**於下列結果。輔助的 exclusive serialized token 為
207.309 ms，也不作主要結論。

所有 run 的 P4/D1/D2/D32 與 final logits SHA-256 一致；final logits hash 為
`1ba23407b4d13fc3baa570c53661a0407b7a36ef79ea0c445ee89a5294ffe045`。

## 表一：Component Active Time

這張表允許重疊，parent/child 也可能巢狀；**不得加總成 TPOT，也不 normalized 成
100%**。`count` 是每個 token 的 invocation/event count。

### Model execution

| Component | ms/token | clock | count | 是否可能 overlap |
|---|---:|---|---:|---|
| Q/K/V projections + RoPE/tensor preparation（合併 scope） | 6.993 | GPU CUDA Events | 28 layers；Q/K/V 各 224 small Linear calls | normal execution |
| Attention | 2.247 | GPU CUDA Events | 28 | 與 parent scope可重疊 |
| O projection | 2.241 | GPU CUDA Events | 28 | 同上 |
| MLP | 11.370 | GPU CUDA Events | 28 | 同上 |
| normalization | 2.033 | GPU CUDA Events | 57 | 同上 |
| embedding | 0.348 | GPU CUDA Events | 1 | 同上 |
| lm_head | 2.075 | GPU CUDA Events | 1 | 同上 |
| Named model-compute active union | **27.308** | GPU common timeline | — | 已對上述 GPU intervals 取 union |

目前不能在不顯著擾動 execution 的前提下，把 6.993 ms 可靠拆成獨立 Q、K、V、RoPE
與 tensor plumbing；因此保留合併 parent，而沒有硬湊細項。Residual/Python/tensor
plumbing 亦沒有獨立可靠 timer。

### Selection

| Component | ms/token | clock | count | 備註 |
|---|---:|---|---:|---|
| Selection core parent | **28.084** | GPU CUDA Events | 28 layers | detailed profile；與 normal timeline 的 27.893 接近 |
| query preparation | 0.399 | GPU CUDA Events | 28 | child |
| Quest parent | **3.682** | GPU CUDA Events | 28 | 包含下列 Quest children |
| ├─ page metadata / score | 2.469 | GPU CUDA Events | 28 | child |
| ├─ Quest Top-k | 0.581 | GPU CUDA Events | 28 | child |
| └─ B0 token expansion | 0.476 | GPU CUDA Events | 28 | child |
| INT4/QK parent | **16.845** | GPU CUDA Events | 28 | 包含下列 children |
| ├─ INT4 metadata stack | 0.097 | GPU CUDA Events | 28 | child |
| ├─ unpack/dequant/estimated-K preparation | **9.347** | GPU CUDA Events | 28 | child |
| ├─ approximate QK matmul | **6.840** | GPU CUDA Events | 28 | child |
| └─ QK scale | 0.149 | GPU CUDA Events | 28 | child |
| Top-p parent | **6.937** | GPU CUDA Events | 28 | 包含下列 children |
| ├─ ranking / argsort | 3.241 | GPU CUDA Events | 28 | child |
| ├─ sorted-logits gather | 0.214 | GPU CUDA Events | 28 | child |
| ├─ Softmax | 0.149 | GPU CUDA Events | 28 | child |
| ├─ cumsum | 2.910 | GPU CUDA Events | 28 | child |
| ├─ threshold/search | 0.279 | GPU CUDA Events | 28 | child |
| └─ membership decision | 0.032 | GPU CUDA Events | 28 | child |

### Post-selection / offloading

| Component | ms/token | clock | count | 是否 overlap |
|---|---:|---|---:|---|
| GPU GQA union | 1.111 | GPU CUDA Events | 28 | normal execution |
| index bitmap D2H | 1.511 | D2H CUDA Events | 28 | 後接CPU decode |
| CPU bitmap decode / flat-index construction | 6.154 | CPU wall | 28 | 可與device工作部分交錯 |
| CPU selected-KV gather | **38.832** | CPU wall | 112 chunks | 與H2D重疊26.746 ms |
| selected-KV H2D | **39.086** | H2D CUDA Events | 112 chunks | 與gather重疊26.746 ms |
| new-KV GPU staging | 0.246 | GPU CUDA Events | 28 | active time |
| new-KV D2H | 0.019 | D2H CUDA Events | 1 batched interval | 幾乎不可見 |
| new-KV host scatter | 0.056 | CPU wall | 2 copies | active time |

Attention metadata/layout preparation、H2D enqueue/pack/control 與等待目前沒有可靠的互斥
leaf timer。從 Selection 完成到 Attention 開始的 73.989 ms parent，扣除 union、index
D2H、CPU decode 與 gather/H2D interval union 後尚有約 **14.041 ms/token**；這只能標為
混合 residual upper-level interval，不能全部命名成 layout 或 Python overhead。

## 表二：Critical-path / overlap summary

| 指標 | ms/token | 解讀 |
|---|---:|---|
| Fresh formal TPOT | **121.693** | 3次 uninstrumented D2–D32 mean |
| Selection exposed interval | **27.893** | normal-overlap CUDA timeline；Selection 未被 post work 隱藏 |
| Post-selection exposed interval | **73.989** | 各層 membership done → Attention GPU start 的合計 |
| Named model-compute active union | **27.308** | QKV/RoPE prep + Attention/O/MLP/norm/embedding/lm_head |
| gather ↔ H2D overlap | **26.746** | common-clock interval intersection |
| gather active、未被 H2D 覆蓋 | **12.085** | 不是可直接相加的TPOT saving |
| H2D active、未被 gather 覆蓋 | **12.340** | 不是可直接相加的TPOT saving |
| selected-KV H2D ↔ named GPU model compute | **0.000** | H2D目前沒有藏在Attention/MLP下 |
| new-KV D2H ↔ other GPU work | **0.001** | 幾乎零；但本身只有0.019 ms |
| token wall未分配 upper bound | **1.682** | 含Python/control/event overhead；非唯一component |

三個 normal-overlap tokens 的 Selection 為 28.548 / 28.446 / 26.686 ms，post-selection
exposed 為 74.063 / 73.928 / 73.976 ms，named model compute 為 26.938 / 27.649 /
27.337 ms。post interval非常穩定；Selection第三個token略低，但整體排序沒有改變。

## 每-token timeline

以下是 28 layers 累積後的簡化關係，不按字元寬度精確縮放：

```text
GPU/default  | QKV+RoPE | Selection | union | index D2H | H2D C0 C1 C2 C3 | Attention | O | MLP |
CPU          |          |           | decode | gather C0 | gather C1 | gather C2 | gather C3 |
new-KV D2H   |                                                                    | 0.019 ms |
                         ^ Selection membership barrier
```

每層 Selection 結束後才進入 union/index/decode/gather/H2D；沒有用後段工作隱藏下一層
Selection。4個chunk形成 gather/H2D pipeline，因此兩者有26.746 ms重疊；但 Attention
必須等該層 selected-KV ready，H2D與後續model compute沒有重疊。

## 研究問題回答

### 1. Selection core由哪些subphase主導？

INT4/QK parent 16.845 ms，約為 Selection core 的60.0%，其中 estimated-K preparation
9.347 ms與QK matmul 6.840 ms最明顯。Top-p parent 6.937 ms（24.7%），內部主要是
argsort 3.241與cumsum 2.910 ms。Quest parent 3.682 ms（13.1%）。因此若之後研究
Selection，本輪證據優先指向 INT4 preparation/QK，其次才是 ranking/cumsum。

### 2. Q/K/V projection + RoPE + tensor plumbing有多大？

合併 active time 6.993 ms/token，約為 fresh TPOT 的5.75%；每 token 的確有Q、K、V
各224次 small Linear calls（672次總計），但目前沒有證據顯示它比28 ms Selection或
74 ms post-selection exposed interval更優先。它值得後續獨立matched ablation，但不是
本輪前3名瓶頸。

### 3. CPU gather還有多少沒有被H2D overlap？

active 38.832 ms，其中26.746 ms與H2D重疊，約68.9%；剩餘約12.085 ms沒有被H2D
覆蓋。這是interval remainder，不能直接宣稱移除後TPOT就下降12.085 ms。

### 4. selected-KV H2D還有多少真正exposed？

相對CPU gather而言，H2D有12.340 ms沒被gather覆蓋；此外與GPU model compute的交集
為0，因此它仍位於每層Attention之前。真正能轉成TPOT的上限必須用matched scheduling
實驗驗證，不能直接把active remainder視為saving。

### 5. 正常Transformer compute佔多少？

已命名GPU model intervals取union後為27.308 ms，約為fresh TPOT的22.44%。其中MLP
11.370 ms最大；QKV/RoPE prep 6.993 ms；Attention 2.247、O projection 2.241、
normalization 2.033、lm_head 2.075 ms。這些比例只用fresh TPOT作尺度參考，不是互斥
百分比圓餅。

### 6. 是否有明顯idle/synchronization/Python gap？

沒有證據支持一個單獨、巨大的全token idle gap。normal profile wall扣除Selection、
post-selection與named model intervals後的upper bound為1.682 ms；但profile本身有9.179 ms
額外wall overhead，所以不能把1.682 ms命名為純Python。較明確的排程限制是：
selected-KV H2D與GPU model compute交集為0，以及post-selection parent內仍有14.041 ms
尚不能唯一歸屬的control/layout/enqueue/wait interval。

### 7. 下一步最值得改善的三個方向

依**真正 exposed latency**排序，而非立即實作承諾：

1. **Selection INT4 preparation/QK，再看argsort/cumsum**：Selection完整暴露，且
   INT4/QK占Selection約60%。
2. **縮短selected-KV gather/H2D的未重疊尾端，或安全地讓H2D與GPU compute重疊**：
   gather/H2D彼此已重疊約26.7 ms，但各自仍有約12 ms remainder，H2D與model compute
   現在完全沒有交集。
3. **index handoff/decode與post-selection residual control/layout**：index D2H 1.511 +
   CPU decode 6.154 ms，另有不能唯一歸屬的14.041 ms post parent residual，適合先做更
   細但低擾動的因果ablation。

per-GQA projection small calls暫列第二梯隊，因合併active time只有6.993 ms；除非matched
implementation證明能大量降低launch/control overhead，不應只因call count很高就列第一優先。

## Artifacts

- Formal fresh TPOT：
  `results/twilight_current_cost_profile_v1/fresh_tpot/003_qa_1_i011/rep{1,2,3}.json`
- Detailed active time：
  `results/twilight_current_cost_profile_v1/component_active/detailed_3tokens.json`
- Normal-overlap evidence：
  `results/twilight_current_cost_profile_v1/normal_overlap/lean_3tokens.json`
- Common-clock Chrome traces：
  `results/twilight_current_cost_profile_v1/normal_overlap/lean_trace_token{1,2,3}.json`
- Auxiliary exclusive profile：
  `results/twilight_current_cost_profile_v1/exclusive_aux/result.json`
- Derived tables：
  `results/twilight_current_cost_profile_v1/analysis/{summary.json,component_active_time.csv,critical_path_summary.csv,per_token_timeline.csv}`
- Reproducible analysis：`scripts/analyze_twilight_current_cost_profile_v1.py`

Diagnostic source SHA-256：

- runner：`1ebcfd065038137067f257dcf92153cd5266f7dba1810ea7098fb51a3cf9b345`
- `mp.py`：`0a9737ae15e14043e7db00f9911bf52a60e5a28df8155ec578c28637aede6b1f`
- `twilight_offload_cache.py`：`1b2a18e9b7978858b85b24df452d36bdc971066b245bf1286fb599cee0c74757`
- `quest_offload_cache.py`：`2ac05f1b6bcab00b0ba09f0df266c628b61262320af03b8212277c9394863517`

## 限制

- 目前只跑 `003_qa_1_i011`；三個profile tokens足以建立本機優先順序，但不是跨request
  generalization，也不是confidence interval。
- CUDA Event是active interval；CPU wall是host interval。不同clock domain只在已校準的
  common token origin timeline中做交集，不把所有component相加。
- Selection detailed run有既有boundary synchronization；只用於subphase active time。
- normal-overlap instrumentation仍增加約7.54% wall，因此所有critical-path interval是
  diagnostic evidence，不取代formal TPOT。
- 本輪沒有accuracy重跑；以相同P4/D1/D2/D32/final logits hashes檢查execution結果未變。

