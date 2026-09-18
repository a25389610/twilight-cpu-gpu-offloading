# Selection INT4 metadata重用消融

## 問題、假設與單變因修改

使用者要求改善Selection資料整理與indices建構。先選一個已知重複工作：prepare_layer_selection
每token將packed INT4 K、scale、minimum重新torch.stack成layer tensor。Source目前只在
_build_metadata建構quant metadata，decode沒有更新這些tensor；保留此既有策略，不新增或
改變quantization規則。新增opt-in --twilight-reuse-quant-metadata（預設False）。

第一次Selection（D1）建立layer連續tensor後快取，將各entry原參照改成該tensor的views，
後續直接重用；不長期保留原逐entry獨立storage。_build_metadata任一entry重建時invalidate
所有涵蓋該entry的layer cache。沒有更動Quest/QK/Softmax/Top-p或final indices。
本輪只驗證batch1固定decode路徑，不宣稱beam/reorder或其他cache生命周期已通過。

## Matched實測

基底為最新4段gather/H2D pipeline，32K qa_1 i011、p=.90。兩輪正反順序base/reuse、reuse/base，
共4process，各32 fixed decode+5 diagnostics；無accuracy/grid。正式D2–D32 TPOT未開profiling。

| Trial | Mode | TPOT mean ms | median ms | Selection wall ms | INT4 QK bundle device ms |
|---|---|---:|---:|---:|---:|
| 1 | base | 148.52 | 147.64 | 51.65 | 19.25 |
| 1 | reuse | 144.44 | 143.99 | 45.62 | 15.01 |
| 2 reverse | base | 148.99 | 147.41 | 50.43 | 19.28 |
| 2 reverse | reuse | 144.29 | 143.64 | 45.46 | 15.06 |

TPOT mean分別減少4.08/4.70ms（2.75%/3.15%）。INT4 bundle少約4.23ms，符合移除metadata
stack；不是FP32 QK算術加速。Selection wall另包含等待，不能把其差額全稱GPU節省。

## Correctness與記憶體

10對diagnostic indices/logits hashes、8對P4/D1/D2/D32 tensors exact，逐step
B0/union/H2D/D2H相同。helper測試驗證三次Selection cache指標不變、entry tensors與parent
共用storage、重建一個entry後cache失效且重新Selection與baseline相同。既有Twilight helper
PASS、py_compile PASS、manifest source hashes核對相同。
4個process peak_gpu_allocated_bytes皆9,580,056,576；這是整run峰值，不能據此細推steady
VRAM節省。D1首次stack仍有暫時重疊分配，沒有宣稱排除所有VRAM/RSS尖峰或跨request leak。

## 結論與下一步

最新此request TPOT約144ms，metadata重用為exact-preserving工程改善候選，default仍False。
只完成Selection資料整理這一項；final indices gathering/masking/sorting與44MB傳输格式未改。
下一個獨立消融可針對final indices materialization，但應保留數值/集合/傳輸可核對的gate，
不要把本輪視為新的Top-p研究算法或泛化到所有contexts。

Artifacts: results/twilight_metadata_reuse_v1/（commands/logs/raw/checkpoints/source hashes/summary）。
Scripts: run_twilight_metadata_reuse_v1.py、analyze_twilight_metadata_reuse_v1.py、
test_twilight_metadata_reuse_v1.py。Source: twilight_offload_cache.py與case runner旗標。
