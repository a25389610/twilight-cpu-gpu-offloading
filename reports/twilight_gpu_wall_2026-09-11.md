# GPU-side phase breakdown：保留非同步

## 方法與範圍

最新32K qa_1 i011、p=.90 exact-tested配置；兩個process各32 decode+5 diagnostics。
擴充process-local profiler的CUDA Events至QKVO、Norm、RoPE、Residual、LM head、Embedding、
GPU metadata、新tokenGPU copy及eviction stream的實際Tensor.copy_呼叫；保留原算法、copy API、
stream依賴與正式source。僅diagnostic入口anchor同步，沒有新增逐phase同步。

每個stream Events統一對齊anchor。Main stream依事件邊界分段，nested indices cpu()的device
interval從Selection外層扣出；未覆蓋區間另列gap。Eviction D2H獨立列出，不與main相加。
主stream span為首個受量測Event至最後Event，並非整個CPU TPOT。CUDA Event interval
包含可能的host enqueue空檔，不是純kernel busy time或純DMA；gap也不表示GPU一定idle。

## 五個diagnostic tokens平均

| Main stream component | ms | % of main span |
|---|---:|---:|
| Selection（扣除indices cpu() device interval） | 48.67 | 25.31 |
| selected KV H2D | 37.78 | 19.65 |
| QKVO projections | 11.35 | 5.90 |
| MLP | 11.21 | 5.83 |
| indices D2H呼叫區間 | 5.30 | 2.76 |
| 新token GPU copy | 3.57 | 1.86 |
| GPU metadata呼叫 | 3.50 | 1.82 |
| LM head | 2.17 | 1.13 |
| Attention | 2.09 | 1.09 |
| RoPE | 2.09 | 1.09 |
| Norm | 1.63 | 0.85 |
| Residual add | 0.26 | 0.14 |
| Embedding | 0.15 | 0.08 |
| 未覆蓋timeline gap | 62.49 | 32.50 |
| Main stream observed span（未rounded） | 192.25 | 100 |

已分類interval合計129.76ms，不可稱純GPU計算總量。Selection仍包含host提交空檔及
cpu()後host返回/event空檔，不是純Quest/QK/Top-p。Indices5.30ms不是舊CPU阻塞51ms，
是indices ready event至copy後event的device區間。D2H另一stream實際copy呼叫interval
合計0.765ms/token，可与main stream重疊，不加入上表。Anchor最大誤差44.47us。

## Latency變異與correctness

| Process | 正常TPOT（profiler前）ms | Diagnostic wall ms |
|---|---:|---:|
| profile | 182.87 | 193.91 |
| baseline | 166.38 | 165.95 |

兩process正常TPOT事先已差16.49ms，因此diagnostic差16.85%不能當純instrumentation overhead。
Profile process diagnostic比其先前正常mean高約6.03%，但token位置不同亦不能視為隔離估計。
不將166.38ms宣稱新優化；演算法/production code沒變，也不拿193.91ms覆寫正式TPOT。

5對indices/logits hashes與4checkpoint tensors exact，B0/union/H2D/D2H相同。
每tokenmain-stream partition閉合。腳本py_compile通過。無完整CUPTI kernel activity，
無法把62.49ms分成GPU真idle、未量測kernel、host instrumentation或driver成本。

## 結論

主要已量測GPU phases為Selection與H2D；模型projections與MLP也有實質device成本。
CPU呼叫時間小不代表GPU工作小；CPU發起/等待表與GPU interval表不能相加。
本輪可供研究定位，但跨process變異與Event擾動使其不能直接當175ms版本的精確百分比。
下一步若要估算kernel優化收益，需matched重复與可用的kernel activity trace/單機制消融，
不能把gap當可全部移除時間。

Artifacts：results/twilight_gpu_wall_v1/（raw/checkpoints/commands/intervals/summary、timeline.chrome.json）。
Scripts：scripts/profile_twilight_async_wall_v1.py新增gpu_profile/gpu_baseline modes；
scripts/analyze_twilight_gpu_wall_v1.py。
