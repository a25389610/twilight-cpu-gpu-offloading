# CPU gather兩項純工程改善：views清理小幅訊號，hybrid copy未改善

## 範圍與實作

使用者授權分別測移除unused host views、compiled連續區段copy＋零散gather。
32K qa_1 i011、p=.90，基底含fused Quest、metadata reuse及4-chunk pipeline，32decode+
5coarse diagnostics，不重跑accuracy。測試順序base→views→hybrid及hybrid→views→base。

- `--twilight-skip-unused-host-views`：flat gather分支不建立沒有被讀取的host_key/host_value
  views；non-flat仍照原流程。每token跳過448次helper呼叫及相關slice，無payload變更。
- `--twilight-cpu-run-gather`：新增local C++ OpenMP helper，每個thread分配不重疊output rows，
  偵測連續run；>=16 tokens整段memcpy，其餘逐row copy，K/V同一次compiled呼叫。
  偵測、thread scheduling及copy均在gather計時內，編譯在cache初始化、TPOT外。
- Hybrid每chunk返回時CPU writes已完成，才沿用原stream發起H2D；下一chunk可與前段H2D
  重疊。保留原pinned slots、completion events、新token預留位置、delta D2H與Attention順序。
- 兩旗標皆預設False、獨立測，沒有將views與hybrid綁在同一variant，也未測combined。
  Hybrid要求flat/direct/GQA/chunk pipeline，不能與舊native gather同時開啟。

## 所有量測（ms/token）

正常TPOT為D2–D32 mean，diagnostic gather/H2D是後續5tokens平均，互相可重疊。

| Trial | Mode | Normal TPOT | TPOT median | Gather wall | H2D Event |
|---|---|---:|---:|---:|---:|
| 1 | base | 203.04 | 202.76 | 60.27 | 39.57 |
| 1 | views | 164.12 | 163.94 | 47.40 | 38.65 |
| 1 | hybrid | 166.81 | 166.44 | 50.78 | 38.84 |
| 2（反序） | base | 164.69 | 164.66 | 49.82 | 38.60 |
| 2（反序） | views | 162.16 | 161.99 | 48.00 | 38.75 |
| 2（反序） | hybrid | 166.29 | 166.83 | 50.60 | 38.77 |

第一輪baseline203.04、第二輪同一baseline164.69差38.34ms，runtime漂移仍存在。
**不能把第一輪203→164/167的下降全稱為優化效果，也不取兩輪均值強算效益。**
第二輪各mode處於較接近的時間水準：

- Views正常TPOT164.69→162.16，少2.53ms/1.54%；gather49.82→48.00，少1.81ms。
  Median164.66→161.99；支持小幅工程改善候選，但單request不足以證明穩定普遍收益。
- Hybrid正常TPOT164.69→166.29，增加1.59ms/0.97%；gather49.82→50.60增加0.79ms。
  第一輪hybrid也比views慢。不能因第一輪相對203ms更快就稱run-copy成功。
- 第二輪diagnostic wall base163.72、views164.72、hybrid164.68ms，與正常TPOT分開，
  不用其差額硬歸因或反推critical-path節省。

## Correctness與驗證

- C++ helper 21種pattern/thread cases：空、singleton、連續、random、倒序、duplicates、
  不同run長度與thread邊界，K/V bit-exact；output guard rows不變、非法indices被拒絕。
- 四組variant-vs-base共20對diagnostic selected/logits hashes、16個P4/D1/D2/D32
  checkpoint tensors完全一致；每step B0/union/H2D/D2H一致。
- 六process均exit0，production/runner/helper source hashes前後一致。
- 既有Twilight helper與新增run-gather helper重驗PASS，py_compile PASS。
- 沒有改selection、數值精度或增加完整CPU KV副本；沒有新的長時RSS驗證，不宣稱排除leak。

## 結論與下一決策

完成兩個可切換工程prototype。Views清理可保留為小幅改善候選；hybrid threshold16沒有
證明效益，保持關閉、不推薦加入最佳配置。實際indices短runs多且bytes未減少，可能使掃描/
排程抵銷copy收益，但尚未拆出這些成本，不能當已證實原因。
下一步若採views，先擴固定3requests matched repeats；不繼續盲目調threshold或新增完整KV
layout，也不將runtime漂移誤認成大量程式改善。未覆寫正式grid、未提高旗標預設。

## Artifacts

- `results/twilight_gather_improvements_v1/{manifest,summary}.json`與6個case JSON/log/checkpoints。
- `scripts/run_twilight_gather_improvements_v1.py`、`scripts/test_cpu_kv_run_gather_v1.py`。
- `source/headinfer/headinfer/cpu_kv_run_gather.{cpp,py}`。
- `source/headinfer/headinfer/twilight_offload_cache.py`與runner新增兩個opt-in flags。
