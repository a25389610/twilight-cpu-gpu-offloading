# 專用CPU gather與Direct Attention Layout實作pilot

## 範圍

依使用者授權實作兩個獨立opt-in工程選項；不改Twilight selection、QK、Top-p、sink/recent。
基底是triton_prepare + bitmap union + flat gather。32K qa_1 i011、p=.90，32 fixed decode，
各5 post-timing diagnostics；四模式正序與反序共8 cases，未跑accuracy/grid。

## 實作

- `--twilight-cpu-native-gather`：C++ OpenMP static row scheduling，在同一迴圈複製K/V，
  ctypes呼叫；使用現有6 CPU threads與pinned out buffers。初始化時以g++編譯，排除於TPOT。
  Python檢查CPU/contiguity/dtype/shape，C++檢查row bounds，不作數值轉換。
- `--twilight-direct-attention-layout`：CPU pack預留每KV group一個new-token位置，H2D直接
  寫入GPU Attention buffer，GPU以同stream copy_補新token，跳過完整history的兩個cat。
  保留host雙buffer完成事件與delta D2H來源生命週期。預留位置先搬已初始化row 0，並在Attention
  前覆寫；不存在未初始化slot被Attention讀取。每token多114688bytes（112KiB）H2D。
- 原GPU history buffers仍保留配置，本輪只改資料流，不宣稱降低VRAM。
- 兩者皆要求flat gather + GQA group execution，default False。沒有把新版本升為預設。

## TPOT（D2–D32 mean ms）

| Mode | 第一輪（baseline→native→direct→both） | 第二輪（反向） |
|---|---:|---:|
| 已有優化baseline | 222.97 | 215.07 |
| Native CPU gather | 213.15 | 212.16 |
| Direct layout | 213.48 | 211.63 |
| Both | 213.36 | 212.64 |

第一輪baseline偏慢；期間也曾啟動helper tests，不能用其差額做強因果結論。
第二輪測試順序反轉、無其他測試並行，direct比baseline mean少3.44ms（1.60%），
median214.14→212.24ms；仍只有一題、兩輪且未鎖clocks，屬小幅preliminary訊號。
不可宣稱本輪得到穩定10ms收益。

## Component觀察（5 diagnostics mean ms）

| Mode | 第一輪CPU gather | 第二輪CPU gather |
|---|---:|---:|
| baseline | 45.16 | 43.68 |
| native | 43.41 | 43.87 |
| direct | 44.49 | 37.92 |
| both | 44.37 | 44.16 |

Native沒有穩定降低gather，不能因TPOT較低就說專用CPU copy成功解決瓶頸。
Direct第二輪gather較低、第一輪未重現，不能直接歸因；改布局也改row offset與CPU訪存。
append_and_d2h_enqueue timer在direct有小幅下降，但涵蓋GPU排隊等待，不能當pure repack。
兩項合併沒有額外收益，因此暫不推薦native或both作新的預設baseline。

## Correctness與驗證

- 兩輪六組variant-vs-baseline的30對diagnostic indices/logits hashes完全相同，
  B0與union history count相同；24對P4/D1/D2/D32模型checkpoint tensors bit-exact。
- Native H2D bytes不變；direct/both每step精確增加114688bytes，符合預留slot計畫，
  有效歷史KV集合不變。這不是完全matched-bytes ablation，必須明示額外payload。
- `test_cpu_kv_gather_v1.py`測empty/singleton/duplicates/invalid indices與K/V exactness；
  `test_twilight_offload_cache_v1.py`通過。最初native單元測試發現ctypes pointer argument order
  錯誤，已在整合前修正，之後單元測試及所有真實模型檢查通過。
- 未驗證其他context/tasks/p、long decode、concurrent calls或其他CPU；不宣稱全面bit-exact。

## Artifacts與下一決策

- `results/twilight_native_layout_v1/`與`results/twilight_native_layout_reverse_v1/`：
  raw JSON/logits/CSV/logs、manifest commands與source hashes、summary。
- `scripts/run_twilight_native_layout_v1.py`及`analyze_twilight_native_layout_v1.py`。
- source：`cpu_kv_gather.{py,cpp}`、`twilight_offload_cache.py`；runner新增兩旗標。

結論：已完成安全可切換的兩個prototype，但沒有像bitmap union一樣的大幅收益。
Direct layout可保留為小幅改善候選，先擴到同32K固定3題repeated測試再決定是否採用；
專用native gather尚無穩定operator效益，不繼續堆疊複雜度。所有正式簡報與預設保持原狀。
