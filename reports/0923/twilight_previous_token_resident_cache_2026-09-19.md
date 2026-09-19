# Twilight previous-token GPU resident selected-KV 實驗

日期：2026-09-19  
分類：correctness-first systems ablation；Selection 演算法與 exact token sets 不變

## 結論

**previous-token-only cache 在目前這個 prototype 不值得併入 baseline。** 它確實把 D2–D32 的 selected-KV payload 從 **507.805 MiB/token** 降到 **46.807 MiB/token**（row-weighted hit/reduction **90.78%**），但三題 matched control 的平均 TPOT 由 **134.186** 上升為 **141.160 ms/token**，ratio-of-means 為 **-5.20%**，三題都變慢。

這否定的是目前的「CPU `searchsorted` mapping + GPU index upload + hit gather/scatter + 每 token resident snapshot」實作，不是否定 temporal reuse 本身。byte upper bound 仍成立，但額外 index/control/D2D 工作吞掉了收益。

## Correctness gate

- `003_qa_1_i011`，32K、p=.90、32 fixed decode steps。
- 21,504 筆 per-Query-head selection trace exact。
- 7,168 筆 layer × KV-head GQA union trace exact。
- D1/D2/D32 的 28 layers，共 84 組 assembled attention K/V SHA-256 exact，valid lengths exact。
- P4/D1/D2/D32 logits tensor 與 SHA-256 exact，final logits SHA exact。
- 三題 formal run 的 checkpoints 與 post-timing selected-position SHA 亦皆 exact。

因此通過 correctness gate；trace run 的 TPOT 因 SHA/D2H 與 per-token event instrumentation **不作正式效能數字**。

## Formal D2–D32 TPOT

| Request | Control (ms/token) | Reuse (ms/token) | Δ ms | Speedup | D2–D32 hit | Miss payload MiB/token |
|---|---:|---:|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 146.837 | 150.058 | +3.221 | -2.19% | 91.18% | 45.175 |
| 002_vt_i002 | 126.754 | 139.005 | +12.251 | -9.66% | 89.93% | 49.113 |
| 003_qa_1_i011 | 128.969 | 134.416 | +5.448 | -4.22% | 91.18% | 46.134 |

- Control：mean/median/range = **134.186 / 128.969 / 126.754–146.837 ms/token**。
- Reuse：mean/median/range = **141.160 / 139.005 / 134.416–150.058 ms/token**。
- Per-request speedup mean/median/range = **-5.36% / -4.22% / -9.66%–-2.19%**；負值代表變慢。
- 這次 fresh matched control 平均是 134.186 ms/token，而 historical current baseline 是 140.691 ms/token；正式判斷使用同一輪 paired control，不拿舊 TPOT 當分母。

## Row-weighted reuse 與搬移量

- D2–D32 共 96,718,596 selected history rows：hit 87,803,497、miss 8,915,099。
- Full current GQA union 1-step reuse：**90.78%**。
- Current selected-KV payload：**507.805 MiB/token**。
- Actual miss-only selected-KV payload：**46.807 MiB/token**。
- Selected-KV byte reduction：**90.78%**。這是 byte reduction，不代表 TPOT 同比例下降。

## D33 component diagnostic（不進正式 TPOT）

三題平均：

- selected-KV H2D：495.520 → 39.073 MiB；CUDA Events 37.284 → 4.016 ms。
- CPU gather wall：34.577 → 3.819 ms。
- 新增 index H2D：22.612 MiB、4.333 ms。
- 新增 resident hit copy：456.337 MiB logical payload、8.194 ms。
- 新增 miss scatter：39.073 MiB、0.446 ms。
- 新增 current-union snapshot：495.410 MiB、5.447 ms。
- cache update CPU wall：65.352 → 70.365 ms。

以上 CPU wall 與 CUDA Event scopes 有 overlap，**不可相加成 TPOT**。資料顯示主要問題不是 miss H2D，而是 exact order mapping、index transport、GPU gather/scatter 與每 token snapshot 的新成本。

## VRAM / memory accounting

- Model parameters（actual tensor bytes）：**5.266 GiB**。
- Full CPU historical K/V slab（actual allocation，三題平均）：**3.455 GiB**。
- Quest page min/max metadata（actual，三題平均）：**220.4 MiB**。
- INT4 packed K + scale/min metadata（actual，三題平均）：**469.4 MiB**。
- Existing reusable GPU execution buffers（actual object size）：**212.0 MiB**。
- New resident cache allocated object size（三題平均）：**577.7 MiB**；final `memory_allocated` matched delta **580.4 MiB**。
- Theoretical full 32K BF16 GPU K+V：**3.500 GiB**。
- `triton_prepare` 的 materialized FP32 estimated-K 暫存 shape `[8,4,8192,128]`，theoretical **128 MiB/layer invocation**（layer-sequential transient，不是 28 倍常駐）。
- PyTorch observed peak allocated/reserved 在 paired control 與 reuse 間皆無增加；這表示既有 prefill/Selection transient high-water mark 已覆蓋 resident allocation，**不能**把 peak 與上述 object sizes相加。final allocated delta 才反映 resident 常駐成本。
- 未取得可靠的 process-scoped NVML peak time series，因此不報 NVML peak。

## 實作範圍與限制

- 只新增 default-off `--twilight-previous-token-resident-cache`。
- D1 full gather/H2D 暖 cache；D2–D32 才重用 previous union。
- Selection、CPU bitmap GQA union、`group_positions`、valid lengths、attention order/dtype/value 均未改。
- 未做 GPU union、compact GPU indices、Selection↔H2D pipeline、Sink/Recent 常駐、多步/LRU、direct INT4→QK 或 full-layer QKV batching。
- 本次只有三個固定 32K requests，各一組 paired run，不是 repeated-trial confidence interval。

## Research decision

停止這個 previous-token-only prototype，不改 production baseline。若之後再研究 resident reuse，前提應是先移除 CPU `searchsorted`/index H2D 與 full snapshot，例如讓 positions/union 在 GPU 直接產生 compact reuse mapping；否則只增加 2/4/8-step retention 會擴大 VRAM 與 lookup 成本，沒有證據能改善目前負 TPOT 結果。

## Artifacts

- Formal manifest：`results/twilight_previous_token_resident_v1/formal/manifest.json`
- Formal per-run JSON/logits/per-token CSV：`results/twilight_previous_token_resident_v1/formal/<request>/<control|reuse>/`
- Correctness smoke：`results/twilight_previous_token_resident_v1/smoke/003_qa_1_i011/`
- Analysis：`results/twilight_previous_token_resident_v1/analysis/summary.json`、`per_request.csv`
- Source snapshot base commit：`dee81b223e64cfc8b6bd58e36d9f8079bae58691`；exact modified-file SHA-256 在 formal manifest。
