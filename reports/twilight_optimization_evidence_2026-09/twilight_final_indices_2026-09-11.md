# Final indices gather/mask融合：未證明TPOT收益

## 問題、實作與guardrail

使用者授權嘗試改善final indices。以4段pipeline+quant metadata reuse為基底，新增
opt-in --twilight-fused-final-indices，預設False。Local Triton kernel融合ranked positions
gather與active mask/sentinel填入，只建立一份masked positions，保留原torch.sort與
counts/desired/sorted positions的torch.cat、int64 transfer格式及blocking cpu()。
不改QK/Softmax/Top-p、選中集合或順序。N/M/sentinel為runtime參數，避免decode length逐步
改變觸發不同constexpr specialization；首次kernel使用在D1，不算入D2–D32 TPOT。

Synthetic helper在N=17/2976/8192、三個width、zero/full/random counts下，比對masked與sorted
tensor exact，共9種shape/width；既有Twilight helper也PASS。

## 正反兩輪32K pilot

32K qa_1 i011 p=.90，各32 fixed decode+5 diagnostics；base/fused、fused/base共4process。
同時打開diagnostic-only detailed selection Events，正常TPOT期間metrics關閉。

| Trial | Mode | TPOT mean ms | median ms | final indices bundle device ms |
|---|---|---:|---:|---:|
| 1 | base | 143.34 | 143.29 | 5.82 |
| 1 | fused | 144.60 | 141.64 | 5.35 |
| 2 reverse | base | 143.62 | 142.68 | 5.90 |
| 2 reverse | fused | 143.81 | 143.80 | 5.42 |

Final indices bundle穩定減少0.466/0.479ms（約8%），但TPOT mean反而增加1.253/0.189ms；
median一輪較好一輪較差。不能稱TPOT加速，不以第一輪較快median選擇性報告成功。
剩餘5.35–5.42ms仍含sort與transfer tensor materialization，未獨立拆出sort，不宣稱全部是sort。
Detailed Selection wall base47.16/46.86ms、fused46.01/46.00ms，有重疊等待，不能預測TPOT。

## 驗證與結論

兩輪10對diagnostic selected indices/logits hashes、8checkpoint tensors exact；逐step
B0/union/H2D/D2H相同。Helper/py_compile通過，manifest source hashes重新核對一致。
正式cache defaults不變，最新推薦配置仍是4段pipeline+metadata reuse，fused final indices
僅experimental，不加入最佳版本或覆寫performance grid。

此實驗表明減少final gather/mask的一些temporaries確實縮短該bundle，但不足以證明
end-to-end收益。下一步若繼續應先拆sort/cat成本或擴大matched測量以界定微小效應；
不繼續堆疊未驗證修改，不改selection算法。本輪未重跑accuracy/grid。

Source: source/headinfer/headinfer/twilight_fused_indices.py、twilight_offload_cache.py、case runner。
Artifacts: results/twilight_final_indices_v1/（manifest/commands/source hashes/raw/checkpoints/summary）。
Scripts: run_twilight_final_indices_v1.py、analyze_twilight_final_indices_v1.py、
test_twilight_fused_indices_v1.py。
