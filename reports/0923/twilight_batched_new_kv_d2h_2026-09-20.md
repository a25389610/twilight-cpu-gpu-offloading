# Twilight token-level batched new-KV D2H（2026-09-20）

## 結論

本輪只修改 Decode 新 K/V 寫回完整 CPU historical slab 的 execution path，沒有修改
Quest B0、INT4 approximate QK、Softmax/Top-p、GPU GQA union、CPU selected-KV
gather、4-chunk selected-KV H2D、direct-layout Attention 或 FlashAttention。

最佳版本是 default-off 的 token-level batching：每層先把 8 個 KV heads 的當前
K/V 寫入一個很小的 contiguous GPU staging buffer；第 28 層完成後，K 與 V 各做一次
nonblocking D2H 到 pinned-host staging，token boundary 等待完成後再以兩個 CPU
strided copies 寫入原本的 pinned CPU slab。完整 historical BF16 K/V 仍只以 CPU
slab 為主要儲存位置。

三題 fresh matched D2–D32 row-weighted mean TPOT 為 **128.892 → 123.103
ms/token**，TPOT 下降 **4.49%**，等價 throughput speedup **4.70%**。三題皆改善。

## 問題、假設與 granularity

GPU-union current candidate 的 delta-only new-KV writeback 每 token 為224 KV entries，
每entry各建立ready/done dependency並做K/V D2H，合計448 tiny D2H calls及448個
operational events；logical payload卻只有114,688 bytes/token。假設control overhead
而非PCIe payload主導。

新增default-off `--twilight-batched-new-kv-d2h`與
`--twilight-new-kv-d2h-granularity {layer,token}`。003無trace pilot：

| Path | TPOT | D2H calls/token | ready/done events/token |
|---|---:|---:|---:|
| per-entry control | 130.163 ms | 448 | 224 / 224 |
| layer-level | 125.629 ms | 56 | 28 / 28 |
| token-level | **124.281 ms** | **2** | **1 / 1** |

因此正式三題採token-level。兩個batching variants都只配置112 KiB GPU staging及
112 KiB pinned CPU staging；沒有額外GPU temporary buffer。

## Correctness gate

條件：`003_qa_1_i011`、32K、p=.90、32 fixed decode steps。Correctness run開啟
GPU-vs-CPU union validation、完整per-Q selection trace、new-KV host-row trace與
D1/D2/D32 Attention K/V hash；trace TPOT不作效能主張。

- 32 × 28 × 8 = **7,168**組host new-K/V rows，control與token-batched全部exact；
  trace SHA-256同為`b778a1a437a26978cb6115dd5b7c5d7c175113e3d0a0e3ecbafa0bcf30137037`。
- D1–D31共**6,944** rows在下一Decode step真正進入host gather前再次驗證unchanged；
  D32的224 rows在run結束同步後驗證寫回exact。
- 完整per-Q selected-history trace SHA-256同為
  `bf43671505b17fd65e0e396ed2dcf8f05e24205342d0578f388604a3042cdf01`。
- D1/D2/D32共84組Attention K/V＋valid lengths trace SHA-256同為
  `2cc6cf1836cadc1f5816024753929169c3db7b44b1484495639c0d3954b0202a`。
- P4/D1/D2/D32 logits tensors exact；final logits SHA-256同為
  `1ba23407b4d13fc3baa570c53661a0407b7a36ef79ea0c445ee89a5294ffe045`。
- 三題formal pairs的final/checkpoint logits、Selection/group hashes、diagnostic
  selected/group hashes、selected-KV H2D bytes、union/valid rows及new-KV bytes皆exact。

## 正式 TPOT

共同條件：Llama-3.2-3B-Instruct BF16、FlashAttention-2、Batch 1、32K、p=.90、
32 fixed Decode steps，D1 warm-up、D2–D32正式統計；GPU compact GQA union、
`triton_prepare`、CPU flat gather、4-chunk selected-KV H2D、direct Attention layout、
early GPU metadata、layer RoPE、fused Quest score及quant metadata reuse均保持一致。

| Request | GPU-union control | Token-batched new-KV D2H | TPOT decrease | Throughput speedup |
|---|---:|---:|---:|---:|
| `001_niah_multikey_3_i011` | 129.005 ms | 122.931 ms | 4.71% | 4.94% |
| `002_vt_i002` | 125.707 ms | 120.774 ms | 3.92% | 4.08% |
| `003_qa_1_i011` | 131.965 ms | 125.604 ms | 4.82% | 5.06% |
| **93-token row-weighted overall** | **128.892 ms** | **123.103 ms** | **4.49%** | **4.70%** |

Overall median/range：control **128.412 / 122.959–141.678 ms**；optimized
**122.639 / 117.364–150.340 ms**。optimized有一個較高單token outlier，但三題
mean皆改善。每題只有一組fresh matched pair，尚未建立多trial confidence interval。

## Component profile

以下是三題各一個post-TPOT diagnostic token之平均；CUDA Event、CPU wall與total TPOT
scopes重疊，不可相加宣稱TPOT savings。

| Metric | Control | Token-level batching |
|---|---:|---:|
| New-KV payload | 114,688 bytes | 114,688 bytes |
| D2H copy calls/token | 448 | 2 |
| ready events/token | 224 | 1 |
| done events/token | 224 | 1 |
| new-KV D2H enqueue/scheduling CPU wall | 10.227 ms | 0.083 ms |
| new-KV D2H CUDA intervals | 1.730 ms | 0.009 ms |
| host-ready wait wall | 0.238 ms | 0.002 ms |
| GPU staging CUDA intervals | 0 | 0.235 ms |
| GPU staging host/enqueue wall | 0 | 1.785 ms |
| pinned-host→final-slab scatter wall | 0 | 0.066 ms |

正式runner每token都呼叫`cache.synchronize()`；D2H與host scatter在token計時結束前完成，
不是跨token pipeline。control原本可讓早期layer tiny D2H與後續layer compute overlap；
token batching等最後一層才啟動兩次D2H，反而減少intra-token overlap。TPOT仍改善，支持
tiny-copy/event/control overhead假設；但不能把10.227−0.083 ms直接當TPOT savings。

## Memory、限制與目前決策

- persistent GPU staging：114,688 bytes（112 KiB）；
- GPU temporary staging：0 bytes；
- persistent pinned CPU staging：114,688 bytes（112 KiB）；
- 三題`max_memory_allocated` delta皆114,688 bytes；
- `max_memory_reserved` delta為0/2 MiB/0，2 MiB是allocator reservation granularity。

完整CPU slab大小/layout不變，沒有selected或historical KV GPU residency。三題correctness
及fresh matched TPOT皆通過，因此token-level path升為目前execution candidate；flag仍
default-off，舊control未刪除。這是execution optimization，不是Selection algorithm創新。

限制：每題單一matched pair；component每題一個diagnostic token；沒有process-scoped
NVML peak；結果只適用目前Batch-1、32K、逐token同步的runner/hardware配置。

## Reproducibility

- Local Git base：`dee81b223e64cfc8b6bd58e36d9f8079bae58691`加default-off工作樹修改。
- Formal manifest：`results/twilight_batched_new_kv_d2h_v1/formal/manifest.json`
- Correctness：`results/twilight_batched_new_kv_d2h_v1/correctness/`
- Pilot：`results/twilight_batched_new_kv_d2h_v1/pilot_003/`
- Analysis：`results/twilight_batched_new_kv_d2h_v1/analysis/`
- Runner/analyzer：`scripts/{run,analyze}_twilight_batched_new_kv_d2h_v1.py`

Measured source SHA-256：

- `quest_offload_cache.py`: `2ac05f1b6bcab00b0ba09f0df266c628b61262320af03b8212277c9394863517`
- `twilight_offload_cache.py`: `687b61a0811b7ab42f463fa78b0f7915362d385768b46fdf778f2a0887278134`
- `mp.py`: `17cc266001bfa9fa0440ec8371d11cbe75db33cf11cee631d58e6593e5547806`
- `run_ruler_partial_h2d_tpot_case_v1.py`: `4981330d4e0e2a6518f1e57cc57ec6c3a7dcd9a32db8b9689e3ac97a24ed9128`
- `run_twilight_batched_new_kv_d2h_v1.py`: `0e96e5f785ad974cef88b632f480ecb1117b35dee384a4d91038214cebeead75`

正式 command：

```bash
CUDA_VISIBLE_DEVICES=0 /home/paul/miniconda3/envs/headinfer_repro/bin/python \
  scripts/run_twilight_batched_new_kv_d2h_v1.py \
  --modes control token \
  --output-dir results/twilight_batched_new_kv_d2h_v1/formal

/home/paul/miniconda3/envs/headinfer_repro/bin/python \
  scripts/analyze_twilight_batched_new_kv_d2h_v1.py
```

公開同步不含raw JSON/CSV/logits/trace/log。
