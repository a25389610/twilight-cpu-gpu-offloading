# Twilight fused candidate preparation pilot — 2026-09-09

## 實作與研究界線

已實作可切換的本地Triton backend，參考Twilight custom sparse GEMV在kernel內讀取並解碼
packed K的方式，保留目前affine INT4相鄰nibble layout、per-token scale/minimum與FP32語義。
這是本地execution optimization，不是官方kernel原封不動移植，也不是新selection演算法。

官方checkout `ce6c050d63907840cb49df64200aad901dd7ea76`、submodule
`d8803b29961c44d77a747636ad4282bd7a9094af`的`compute_qk`使用`scale=1,zero=0`，metadata
load被註解；不能直接作為本地affine INT4的等價替換。先前「官方支援scale/min」的概括須以此限制
解讀；本輪沒有修改official checkout。

## 兩條實驗路徑

- `triton_prepare`：candidate positions直接讀取stacked packed K/scale/min，在同一kernel完成
  gather、nibble unpack、FP32 cast與affine dequantization，僅寫出一份FP32 estimated K，
  接回原本`torch.matmul`及scaling。`enable_fp_fusion=False`保留分開mul/add的rounding。
- `triton`：連QK reduction也融合，只寫logits。Synthetic QK誤差約1.1e-5，但第一輪真實
  requests的60個diagnostic hashes皆未通過bitwise gate，模型checkpoint最大差異0.15625。
  48個checkpoint的top-1仍相同；這不等於證明RULER品質相同或降低。保留為experimental，
  不作exact-preserving改善結論。
- 預設仍為`pytorch`；目前支援head_dim=128、layer-batched selection、同GPU contiguous tensors。
  Triton採lazy import，原PyTorch路徑不需載入新kernel。

```text
原路徑：metadata stack → candidate tensors gather → low/high tensors
      → FP32 codes → multiply temporary → FP32 estimated K → torch.matmul
改善後：metadata stack → fused candidate gather/unpack/dequant → FP32 estimated K → torch.matmul
```

Quest、B0、sink/recent、Top-p、final indices、GQA union、CPU KV gather與H2D維持原語義。
metadata stack與一份FP32 estimated K仍存在，因此本輪沒有完全消除K materialization。

## 實驗與保留紀錄

- 固定p=.90，4K/8K/16K/32K各沿用3個timing requests，每題32個fixed-token decode steps，
  另量5個diagnostic steps；兩個backend共24 cases、60對diagnostic steps。
- Selection為CPU wall；INT4 bundle為CUDA Event interval，包含metadata stack、candidate準備、
  QK與scaling。TPOT為未啟用diagnostic timers的D2–D32 synchronized wall。不同clock不可相加。
- 各pair交錯執行、單輪preliminary，未鎖GPU clocks。模型載入/Prefill與D1不計入TPOT。
- `twilight_fused_qk_v1`保存direct-QK試驗；`twilight_fused_prepare_v1`保存第一版prepare試驗。
  第一版N為constexpr，4K qa_1 B0由2656→2672時會新編譯，Selection曾達110 ms；8K曾達128 ms。
- 最終版使用runtime N避免B0跨頁觸發shape specialization，結果另存`twilight_fused_prepare_v2`。
  H仍為specialization parameter，初次模型執行可能包含JIT成本；deployment需保留warm-up。

## 最終結果

最終24 cases全部完成；12組paired requests的60個diagnostic steps中，selected indices與
logits hashes全部相同、H2D bytes全部相同，48個P4/D1/D2/D32 logits checkpoints亦全部
bit-exact。Kernel synthetic prepared K與QK bit-exact，Twilight helper test通過，manifest的
source hashes與交付程式一致。測試環境為RTX 5060 Ti、PyTorch 2.7.0+cu128、Triton 3.3.0。

下表為本輪matched means（ms/token），並非替換既有跨六p的簡報數字。

| Context | Selection原版 | fused prepare | Selection降低 | 原TPOT | 新TPOT | TPOT降低 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 35.82 | 24.55 | 31.5% | 143.36 | 127.37 | 11.1% |
| 8K | 89.32 | 40.33 | 54.8% | 248.38 | 200.69 | 19.2% |
| 16K | 99.44 | 43.46 | 56.3% | 283.99 | 227.35 | 19.9% |
| 32K | 106.01 | 50.57 | 52.3% | 297.64 | 239.45 | 19.6% |

INT4 bundle CUDA mean（metadata stack到QK scaling）：4K 20.674→6.989 ms、8K
60.735→16.584 ms、16K 73.547→17.737 ms、32K 75.088→19.228 ms。它不能與CPU wall相加。
完整mean/median/min/max見`comparison.csv`。改善版Selection median依序為
24.617/40.476/43.347/50.454 ms；min–max依序22.443–26.207/37.929–42.682/
42.603–44.429/49.308–51.807 ms。

runtime N修正後，4K qa_1在2656→2672 B0跨頁時Selection為23.23 ms，與鄰近token相當；
8K改善版所有15 samples最大42.68 ms，沒有重現第一版約128 ms的JIT尖峰。

## 純資料準備與temporary memory

Synthetic operator測試：每層8 KV groups × 3 Q heads/group、D=128，使用隨機B0位置、
BF16 scale/min與FP32 Q，5 warm-ups、15 repeats，表列CUDA Events median。
本表排除metadata stack、Top-p及CPU-offload，不能直接當作整個模型Selection。

| History/B0 | 原candidate準備 | fused準備 | 準備+QK原版 | 準備+QK改善版 | 原temporary peak | 改善temporary peak |
|---|---:|---:|---:|---:|---:|---:|
| 4096/2976 | 0.618 ms | 0.101 ms | 0.693 ms | 0.150 ms | 118.25 MiB | 35.15 MiB |
| 8192/7445 | 1.782 ms | 0.298 ms | 2.014 ms | 0.502 ms | 298.08 MiB | 88.68 MiB |
| 16384/8192 | 1.996 ms | 0.363 ms | 2.391 ms | 0.586 ms | 325.50 MiB | 96.75 MiB |
| 32768/8192 | 2.001 ms | 0.381 ms | 2.396 ms | 0.602 ms | 325.50 MiB | 96.75 MiB |

Temporary peak使用`torch.cuda.max_memory_allocated()`減去呼叫前allocation，量準備+QK的
增量峰值，包含其outputs，排除既有inputs，不代表完整模型VRAM或allocator reserved memory。
8K synthetic資料準備時間降低83.3%，temporary allocated peak降低70.2%。

## 研究判斷

1. 支持「本地PyTorch資料準備存在可消除的materialization成本」：保持實際B1與H2D不變，
   Selection仍降低31–56%。原prototype高成本不能視為Twilight演算法固有成本。
2. 保留原matmul的融合版本已具備本輪exact correctness證據；直接融合QK的版本仍需處理
   reduction/tie-boundary差異，不能將其更快的latency當成同樣語義已驗證的收益。
3. 本輪是一般kernel融合與integration改善，尚不足以宣稱新研究方法或官方Twilight效能重現。
4. 此為p=.90、3 requests/context單輪fixed-token pilot，未重跑RULER accuracy/grid，也尚未
   驗證全部p、其他head_dim、其他GPU或長時間生成；預設保留PyTorch，改善版以flag選用。

## 執行方法與artifacts

在原case command加入`--twilight-qk-backend triton_prepare`，其餘Twilight與layer-batched參數沿用。
完整pilot與檢查指令（在headinfer_reproduction目錄）：

```bash
/home/paul/miniconda3/envs/headinfer_repro/bin/python scripts/run_twilight_fused_qk_pilot_v1.py --backend triton_prepare --output-dir results/twilight_fused_prepare_v2
/home/paul/miniconda3/envs/headinfer_repro/bin/python scripts/analyze_twilight_fused_pilot_v1.py --results results/twilight_fused_prepare_v2
/home/paul/miniconda3/envs/headinfer_repro/bin/python scripts/test_twilight_fused_qk_v1.py
```

- Source：`source/headinfer/headinfer/twilight_fused_qk.py`、`twilight_offload_cache.py`
- Runner：`scripts/run_ruler_partial_h2d_tpot_case_v1.py`
- 最終raw JSON/log/checkpoints、manifest、comparison CSV/JSON：`results/twilight_fused_prepare_v2/`
- Synthetic operator timing與temporary allocated memory：`results/twilight_fused_qk_v1/operator.json`
