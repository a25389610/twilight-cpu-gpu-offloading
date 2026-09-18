# Twilight 官方 Source Code Inspection（2026-08-29）

## 目的

確認 Twilight 是否有公開 source code、它是否能用來驗證 Quest fixed Top-K 的
over-selection／under-selection 問題，以及官方實作能否直接代表本研究的 Batch=1
CPU–GPU KV offloading pipeline。

## Source provenance

- Paper：*Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning*
  （NeurIPS 2025；arXiv:2502.02770）。
- 官方 repo：`https://github.com/tsinghua-ideal/Twilight`
- Main repo commit：`ce6c050d63907840cb49df64200aad901dd7ea76`
- `flash-topk-attention` submodule commit：
  `d8803b29961c44d77a747636ad4282bd7a9094af`
- 本地唯讀 checkout：`/home/paul/Desktop/SOC/Codex/external/twilight`

## 方法對照

Twilight 並未取代 Quest selector，而是在其後增加第二階段：

1. Quest Key min/max Page Score 先選一個保守的固定候選集合 `B0`。
2. 對 `B0` 內 token 使用 INT4 Key 估計更細的 QK logits。
3. 對 estimated logits 做 Top-p pruning，讓每個 Head／Query 得到不同的最終 `B1`。
4. 對可變長的 `B1` 執行 ragged sparse attention。

repo 的 `config_quest_twi.json` 設定為 Page size 16、`B0=8192 tokens`、INT4 estimator、
Top-p `p=0.85`、前兩層不稀疏。這是一份 repo config，不能直接視為所有模型的
paper default；paper 對 Llama 2／3 使用的候選 threshold 是 `p=0.95`。

## Source inspection 結果

### 1. Accuracy path 可以驗證 adaptive budget，但不能量 TPOT

`twilight/pyimpl/attention.py` 明確註記 Python implementation 只用於 accuracy，並會先
計算 full attention weights。decode path 隨後執行 Quest mask、Key quantization、
estimated QK 與 Top-p mask。因此這條路可以觀察每個 Head 的最終 budget 與品質，但其
計算量不代表部署時的 Twilight latency。

### 2. Efficiency breakdown 是 component microbenchmark，不是完整 dataflow

`benchmark/efficiency/bench_breakdown.py` 分別量測：

- Quest sparse GEMV + Top-K；
- INT4 sparse GEMV；
- Top-p；
- ragged attention。

但它使用隨機 KV、metadata 與 logits；Top-p wrapper 的 logits 並不是前一個 INT4
SpGEMV wrapper 的 output，ragged-attention metadata 也在 benchmark 初始化時先建好。
程式最後將各 component latency 相加，因此它是近似 cost model，不是每個 decode token
真正串接 Quest → INT4 QK → Top-p → attention 的 end-to-end 執行。

其預設測試是 Batch 16／32／64、32 heads、30K sequence，不等於本地
Llama-3.2-3B、Batch=1、24 Query Heads／8 KV Heads、CPU-offload 場景。

### 3. `llama_patch.py` 仍是 prototype，不是可直接採用的 end-to-end model path

目前檔案會跳過 prefill、配置隨機 cache／metadata；Quest 得到的 `label_index` 沒有接到
後續 INT4 estimator，而 Top-p 的回傳結果也未用來建立最終 attention indices。最後
attention 使用的是 `arange(kv_seq_len)`。因此不能據此宣稱官方 Twilight model-level
pipeline 已完成或已在本機重現。

### 4. Kernel source 有實際借鑑價值

`flash-topk-attention` submodule 含 Quest min/max sparse GEMV、RAFT Top-K、INT4 sparse
GEMV；Twilight repo 也含 Top-p CUDA implementation。它們可用來建立 Batch=1 operator
ablation，但目前不是單一 fused Quest+Twilight selector，仍會新增 kernel launch、QK
estimation 與 Top-p selection 成本。

## 對目前研究問題的判斷

**已知事實：** Twilight 的可變 `B1` 確實直接處理 Quest 每個 Head 固定 Top-K 的限制。

**本地已知觀察：** Quest fixed 5% prototype 的 selector aggregate 已約 78.55 ms/token，
高於 H2D 51.63 ms/token；因此完整 Twilight 第二階段很可能加重目前 bottleneck。

**尚未驗證假設：** Twilight 減少的最終 KV／attention／H2D 成本，是否足以抵銷
Batch=1 下新增的 INT4 QK、Top-p、metadata 與同步成本。paper 的 A100 large-batch
speedup 不能回答此問題。

## 最小、可歸因的驗證流程

### Gate A：只驗證 fixed Top-K 品質問題是否改善

先選一個既有 32K request，不進行 TPOT 宣稱：

1. Page size 固定 16；Quest 先取保守 `B0`。
2. 同時測 exact-QK Top-p（concept upper bound）與 INT4-QK Top-p（Twilight-like）。
3. sweep `p={0.85, 0.90, 0.95, 0.98}`，Llama 3 優先檢視 `0.95`。
4. 記錄每 layer／head／decode step 的 final selected tokens、mean／median／P95／max。
5. 比較 Hidden-L2、answer-trajectory PPL 與 RULER answer。
6. 對每個 adaptive run 加一個「相同平均 final KV budget」的 Quest fixed-K control；否則
   無法區分品質改善來自 adaptivity，或只是實際拿了更多 KV。

### Gate B：只驗證 Batch=1 新增 selection cost

用本地實際 shape（Batch=1、24 Q Heads、8 KV Heads、head dim 128、約 32K）分別量：

- Quest Page Score；
- Top-K(`B0`)；
- INT4 QK over `B0`；
- Top-p indices／counts；
- ragged GQA attention 與 metadata build。

輸入應優先使用 Gate A 的真實 logits／budget distribution，不只用官方 benchmark 的
random logits。此 gate 仍不宣稱 CPU-offload TPOT。

### Gate C：最後才整合 CPU–GPU offload

只有 Gate A 顯示 matched-budget quality 改善，且 Gate B 的新增成本合理時，才接到本地
`QuestTopKOffloadedCache`。正式比較 fixed Quest、Twilight-like Adaptive Top-p，以及只從
Quest Page Score 推導的 lightweight Adaptive-K；三者需同 request、同平均 H2D/KV budget。

## 環境限制

2026-08-29 最後核對的本地環境為 PyTorch `2.11.0+cu128`、Transformers `5.5.4`、
RTX 5060 Ti（compute capability 12.0），且沒有 `flashinfer`。Twilight requirements pin
PyTorch `2.5.0`、Transformers `4.45.2`、FlashInfer `0.2.0.post1`、FlashAttention
`2.6.3`。本次沒有改動現有環境，也尚未確認官方 CUDA extensions 對 `sm_120` 的相容性。

## 目前結論

官方 source code 足以確認 Twilight 的算法結構並借用 kernels，但不足以直接重現本地
Batch=1 CPU-offload end-to-end 結果。下一步應先做 Gate A 的 matched-budget quality
test；只有證明 adaptive Top-p 確實改善 fixed Top-K，才值得支付 Gate B/C 的整合成本。
