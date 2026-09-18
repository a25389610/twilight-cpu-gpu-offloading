# 逐layer QKV／RoPE工程改善：correctness gate與拆分

## 授權、版本與實作

使用者要求先改善逐KV-group Q/K/V projection與RoPE，之後才看GPU metadata。
本輪metadata保持不變，基底為triton_prepare + bitmap union + flat gather + direct layout。
32K qa_1 i011、p=.90、32 fixed decode、5 post-timing diagnostics，未跑accuracy/grid。

- `--twilight-layer-projection`：初始化時保留完整Q/K/V weight detach views，decode各layer
  各做一次完整Q/K/V，再做一次RoPE、切回8個KV groups。無每token權重cat/copy，prefill路徑不變。
- `--twilight-layer-rope`：維持原逐group QKV GEMM，只將Q/K cat後做一次RoPE，再切回groups。
  這是第一個版本未過exact gate後的單變因拆分。兩旗標default False，只支援Llama bias-free
  layer-batched Twilight；不修改Selection或Top-p。

## 完整QKV版本：速度改善但未過等價gate

Baseline219.60ms、整layer版本198.44ms（D2–D32 TPOT mean）。P4 bit-exact，D1/D2/D32
logits max abs差0.125/0.125/0.1875，checkpoint top1皆相同；5-step indices/logits hashes不同，
H2D bytes也不同，B0相同。因此保留experimental，不稱為品質不變的工程加速，也不等於已證實
accuracy下降。完整projection改变GEMM形狀，不能只憑公式相同假設BF16結果相同。

## RoPE-only版本：第一輪

反序（先variant再baseline）：baseline215.93ms、RoPE-only205.37ms，下降10.56ms（4.89%）。
P4/D1/D2/D32四個checkpoint tensors bit-exact；5 diagnostic indices/logits hashes、B0及H2D
bytes皆相同。這隔離出可保持原結果的RoPE batching收益，不宣稱QKV已完成等價batching。
按程式路徑，RoPE apply每token224→28次；Q/K/V各仍224次。未用instrumented timing取代TPOT。

## 交換順序第二輪

正序（先baseline再variant）：215.72→207.45ms，下降8.27ms（3.83%）。
再次通過4對checkpoint tensors、5對indices/logits hashes、B0/H2D exact gate。
兩輪方向一致，約3.8–4.9% preliminary收益；尚非多request confidence interval。
第二輪artifact：`results/twilight_layer_rope_v2/`。

## 限制與下一決策

單一request/p/模型、未鎖clocks，需交換順序重測與之後固定3題驗證。Flag保持opt-in，不覆寫正式表。
完整QKV若要繼續研究，需另做局部projection數值歸因／品質gate，不能把top1相同當全面品質證明。
GPU metadata25.51ms仍是inclusive建立/搬運/等待，留下一步獨立處理，不與本輪混合。

Artifacts：`results/twilight_layer_projection_v1/`（失敗exact gate）、
`results/twilight_layer_rope_v1/`（exact-pass），commands/source hashes/raw/checkpoints/summary保留。
Source：`source/headinfer/headinfer/mp.py`、`scripts/run_ruler_partial_h2d_tpot_case_v1.py`。
Runner/analysis：`run_twilight_layer_projection_v1.py`、`analyze_twilight_layer_projection_v1.py`。
