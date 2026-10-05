# 最新 Direct INT4 QK＋previous-token cache：32K p=.70 preliminary TPOT / VRAM

日期：2026-10-05。三題各一組 fresh p=.70/.90 formal pair，另三題獨立p=.70 memory inventory；非完整品質測試。

## 問題與原先假設

使用者詢問目前系統p=.70時的TPOT與VRAM。過去六個p值的130題品質與舊timing不是目前Direct/resident版本，因此不以舊TPOT比例或p比例換算。假設低p能減少residentKV與assembly成本，但全historyINT4/Questmetadata、候選QK計算與modelcompute大多保留，效能不會按selectedKV量等比例下降。

## 實驗條件與處理

RTX5060Ti16GB、Llama-3.2-3B-Instruct、BF16、Batch1、QuestB0=8192、DirectINT4QK backend triton、previous-tokenresident、GPUbitmapmapping、mappedCPUread、lowVRAMflags、Top-pGraph與precomputedmapping。完整命令來自`results/twilight_direct_qk_breakdown_20260928_v1/formal/rep1/<request>/direct/command.json`；配對命令只改p與output，第一/第三題先.70，第二題先.90。source未修改，manifest凍結5個核心檔案hash。

原三題32K frozen prompt，每次32個fixedtoken1 forwards；D1warm-up、D2–D32共31tokens synchronizedCPUwall，排除load/prefill，不插componentprofiling。Memory另跑inventory，不把其latency併入formal。

## TPOT

| Request | p=.70 ms/token | p=.90 ms/token |
| --- | ---: | ---: |
| 001_niah_multikey_3_i011 | 49.512 | 51.938 |
| 002_vt_i002 | 51.364 | 52.163 |
| 003_qa_1_i011 | 48.880 | 52.610 |
| **三題平均** | **49.919** | **52.237** |

p=.70較同輪.90少2.319ms/token（4.44%）；三pair都變快。每題只有一次，屬preliminary；不與歷史55.939或52.104相減。結果約49.9ms，非以p比例推算。

## VRAM（三題memory平均）

- cache object unique GPU backing：**1015.886MiB＝0.992GiB**；三題範圍1002.512–1032.517MiB。相對固定32KFullKV3584MiB為**28.35%**。
- cache unique包含residentKV、INT4Key、Questmetadata、selectedattentionbuffers與mappingmetadata；不含模型、cache外的Graph/private/runtimestorage。alias不重複。
- D32 liveallocated扣model-onlybaseline增量：1036.404MiB（1.012GiB），含cache之外runtime/Graph，不能全稱KV。
- 整個process D32 torch CUDA liveallocated：7164.752MiB（6.997GiB），包含model、buffers與cache；不等於nvidia-smi或allocatorreserved，也不是prefillpeak。
- model-onlybaseline含模型與既有buffers，故「扣baseline」不等於只扣3B權重。

| cache unique component | MiB |
| --- | ---: |
| Previous-token residentKV | 268.927 |
| 全history INT4codes/scale/min | 469.395 |
| Questmin/max | 220.391 |
| Attention selectedKV working | 49.881 |
| Mapping/other/newtoken | 7.292 |

相對09/28同設定p=.90歷史cacheunique1324.817MiB，此次.70少308.931MiB（23.32%），resident577.858→268.927MiB，其餘metadata大致固定；memory.90未在本輪重跑，標為歷史容量比較。

## 驗證、結論與限制

九個process均success。分析核對核心sourcehash、formalpair命令僅p/output不同、promptsha相同、32steps/31timingtokens/fixedpolicy、backend/resident/p值，memory分類加總等於unique總額。尚未重複各request以確認variation，也沒有本sourcep=.70的130題quality。過去p=.70分數75.9104是舊版本，不能自動貼到本輪49.919ms的點。

**實驗觀察**：目前p=.70約50ms、cache-related約1GiB；低p明顯省residentmemory，但TPOT改善只有約4.44%。**未驗證原因**：固定selection/modelcompute可能限制收益；本輪無componentprofile，不作因果確認。

**下一決策**：若要正式與RetroInfer同品質比較，先補本sourcep=.70完整130題quality，再視差異大小重複formalpair；本輪停止，不自動展開p sweep。

## Artifacts

`results/twilight_p070_probe_20261005_v1/`：manifest、formal/<request>/070或090、memory/<request>/070、commands/result/processwall/log/inventory、summary。

`scripts/run_twilight_p070_probe_20261005.py`與`scripts/analyze_twilight_p070_probe_20261005.py`保存重跑/獨立核算。公開mirror準備本報告與兩支scripts；raw/模型/log未納入，未commit/push。
