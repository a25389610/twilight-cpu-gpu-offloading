# Quest page score 中間 tensor 融合 pilot

## 問題與範圍

使用者要求純工程優化，保留 Twilight selection 規則。只將 page extrema 選擇、FP32 cast、
乘法及sum融合成 local Triton kernel。保留page_min/page_max stack；沒有重用/凍結原本會
變動的Quest metadata，未修改B0、Top-k、INT4、Softmax、Top-p、sink/recent或KV布局。
新旗標 `--twilight-fused-quest-score` 預設關閉；不是官方Twilight kernel。

原路徑：stack metadata → BF16 extrema tensor → FP32 extrema → FP32 product → sum。
新路徑：stack metadata → fused load/where/cast/multiply/reduce → FP32 page scores。
沒有完整extrema/cast/product輸出，metadata stack及最終scores仍配置。

## 浮點與適用條件

本kernel模擬目前torch2.7 CUDA contiguous dim128 sum的順序：各lane累加
d,d+32,d+64,d+96，再以shuffle-down offsets1,2,4,8,16合併。
關閉FP fusion，不以不同tl.sum reduction樹取代原sum。適用head_dim128、至少16 output rows、
FP32 query、contiguous同CUDA device metadata；已驗證metadata BF16/FP16/FP32。
其他PyTorch版本、shape或特殊非有限輸入未驗證，不宣稱通用bit-exact。

## 實驗

32K qa_1 i011、p=.90；現有4-chunk gather/H2D、metadata reuse、triton_prepare、bitmap union、
flat gather、direct layout、RoPE-only、early metadata保持不變。4 processes：
base→fused→fused→base。每process先32decode，正常TPOT取D2–D32，再5個detailed diagnostic tokens。
沒有逐phase新增同步，不把diagnostic加總當TPOT。只有單request，屬preliminary。

| Trial | Normal TPOT base→fused ms/token | 改善 | Median base→fused | Quest score Event base→fused |
|---|---:|---:|---:|---:|
| 1 | 208.49 → 206.26 | 2.23ms / 1.07% | 208.45 → 204.44 | 9.40 → 4.53ms |
| 2（反序） | 209.87 → 205.67 | 4.20ms / 2.00% | 209.50 → 204.60 | 9.71 → 4.50ms |

Quest metadata-score區段降低51.7%/53.7%；此timer仍含metadata stack與device interval內
可能的host enqueue空檔，不是單一kernel純算術時間。Selection wall49.82→48.52、
49.93→48.08ms，沒有與Quest區段等量下降；後續其他Events也變動。
不以不同clock做相減歸因，不聲稱省5ms GPU就一定省5ms TPOT。

本輪baseline208–210ms，不是歷史144/171ms；既有runtime漂移仍未解決。
只作同時段ABBA對照，不把206ms稱優於歷史144ms的新最佳成績。

## 驗證

- 57組synthetic score tensors與Top-k indices bit-exact：6種pages×3種metadata dtype×3種
  query分布，加zero/positive/negative tied scores。
- 真實request額外correctness-only replay：**1036次逐layer page score tensors bit-exact**。
  q=[8,3,128]、metadata=[8,2002..2005,128] BF16；包含32decode+5diagnostics。
- ABBA兩組共10對diagnostic selected/logits hashes、8個checkpoint tensors exact；逐step
  B0/union/H2D/D2H一致。三個production/runner/kernel source hashes在實驗前後一致。
- 既有Twilight helper與metadata reuse helper PASS；py_compile PASS。
- 首次correctness-only runner正常退出，但runpy遇SystemExit而未保存最後統計；加捕捉正常
  exit後另存correctness_only_v2.json及real_score_parity.json。兩次correctness-only會逐call
  算reference/sync，**均不納入效能比較**，保留原檔。

## 結論與下一決策

相同scores/selection下確實減少Quest局部成本，單request两輪正常TPOT有約1–2%下降。
這是純工程融合，不是新selector算法；不宣稱重大端到端加速或跨context成功。
保留opt-in，不改正式grid/預設；下一步固定3requests matched驗證，必要時再獨立分析
metadata stack與較快selector之後的enqueue等待，不混入候選數量/精度改動。

## Artifacts

- `source/headinfer/headinfer/twilight_fused_quest.py`
- `source/headinfer/headinfer/twilight_offload_cache.py`
- `scripts/run_ruler_partial_h2d_tpot_case_v1.py`
- `scripts/{test,run}_twilight_fused_quest_v1.py`
- `scripts/validate_twilight_fused_quest_request_v1.py`
- `results/twilight_fused_quest_v1/`：manifest、summary、4raw cases/logs/checkpoints、
  correctness-only results、real_score_parity.json。
