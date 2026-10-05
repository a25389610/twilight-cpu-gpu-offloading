# Cache–attention 介面優化：32K／B0=4096／p=.90

實驗日期：2026-10-06；local canonical weekly：2026-09-30；public mirror：reports/1007。

## 問題與控制

原版本每層將 selected lengths 同步到 CPU，配置連續 Attention KV，並組裝 previous-token cache hits。驗證改善 cache 與 attention ABI 能否降低 TPOT。固定 Llama-3.2-3B-Instruct／BF16／Batch=1／RTX5060Ti16GB、32K cohort、B0=4096、p=.90、Quest→Twilight→GQA union 選取規則，沿用已驗證 GPU-cu／swap／union／exact CUDA QK+Quest execution preset。Production defaults未改。

## 兩個原型

1. Fixed slots：GPU KV slots保留hit位置，miss-only mapped-host fill，indexed Triton split decode attention直接讀取slot。刪除hit payload組裝，但引入slot metadata、不同attention arithmetic及更多capacity。
2. Bounded GPU lengths：保留原FlashAttention與contiguous assembly，GPU cu_seqlens決定有效rows；host max_seqlen與storage使用安全上界，移除每層lengths_gpu.cpu().tolist()。上界為 min(context, 3×ceil(B0/page_size)×page_size+sink+historical_recent)，attention另含current token。這是capacity，不是實際保留的logical KV量；沒有降低p／B0或重用selection結果。

第二個原型是先前被否定 GPU-length 方向的明確 revision：舊prototype使用full-context bound且bit-exact gate失敗；本輪使用B0-derived bound，明示max_seqlen會改FlashAttention schedule／rounding，加入shadow payload及natural-generation checks，不將舊失敗改寫成已通過。

## 正式TPOT

每個原型各12 fresh processes、三題×兩輪×兩 arms、每arm186個samples。32 fixedtoken1 forwards、D1 excluded，D2–D32 synchronized CPU wall，無profiler／shadow／payload capture；snapshot/commands/env保存。各原型使用自己同批baseline，不能將兩批差值相加。D1及whole-process成本保留在raw；slot kernel warming/初始化在D1，未丟掉D2 samples。

| 原型 | 同批baseline ms | candidate ms | 少多少 ms | 改善 | 更快pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed slots＋indexed attention | 33.893619 | 31.326523 | 2.567096 | 7.5740% | 6/6 |
| Bounded lengths＋原FlashAttention | 32.987535 | 28.998166 | 3.989370 | 12.0936% | 6/6 |

## GPU memory：有代價

Final live PyTorch allocated減去model parameter bytes，包含KV、selection metadata、workspace、logits等，**不是純KV**。不包含opaque driver/library allocations；reserved與prefill-included peak不作decode KV footprint。各batch model parameter bytes相同。

| 原型 | baseline nonweight MiB | candidate nonweight MiB | 額外 MiB |
| --- | ---: | ---: | ---: |
| Fixed slots | 1158.615560 | 2566.170247 | 1407.554688 |
| Bounded lengths | 1158.615560 | 2158.383952 | 999.768392 |

本prototype為避免每層CPU同步而按安全上界配置resident/attention backing；容量明顯高於actual selected KV。雖未更改邏輯選取，不能宣稱VRAM免費、同VRAM加速，或直接套用舊130題品質分數。

## Correctness與數值限制

Synthetic六steps檢查slot eviction、unique slots、miss fill與KV逐值相同；indexed attention對FP32→BF16 reference maxabs≤0.000244141。兩種原型各有32K／128-forward shadow gate：D1／D2／D32共84個layer attention captures的K/V與baseline相同，resident hit/miss、新KV、checkpoint/final logits相同；shadow返回原FlashAttention output維持相同query軌跡。每種3556個native attention calls皆finite；shadow timing不作正式TPOT。

- Indexed shadow attention maxabs=0.01562500，max per-call meanabs=0.00009619；不要求native bit-exact，shadow checkpoint exact不能替代native quality。
- Bounded FlashAttention shadow attention maxabs=0.01562500，max per-call meanabs=0.00003481；不要求native bit-exact，shadow checkpoint exact不能替代native quality。

Native正式case saved checkpoints finite；source frozen/current hashes、returncodes、CSV重算、prompt／commands／env paired協議核對。Fixed-slot harness原始attention欄位繼承flash_attention_2字串，**native decode實際為Triton**，以adapter source／本報告為準；bounded原型確實沿用FlashAttention。

## Native natural-generation smoke

僅優先對bounded原型做三tasks各一題，fresh paired processes、frozen per-request generation budget、greedy to EOS，官方RULER scorer重新計分。不是130題，不能把三題作完整品質結論。

| Task | baseline score | candidate score | answer token exact |
| --- | ---: | ---: | --- |
| niah_multikey_3 | 0.000000 | 0.000000 | True |
| vt | 60.000000 | 60.000000 | True |
| qa_1 | 0.000000 | 0.000000 | True |

Fixed slots未做native task-quality sweep，不能採用shadow gate的輸出當native答案。

## 保留失敗與初步實驗

Cold slot prototypes含JIT／初始化，D2–D32被污染，未當加速結果；posstride誤設constexpr造成逐token JIT後已修正。Pilot期間synthetic micro有一次與slot process初始化重疊，該pilot完全排除正式證據。Earlier warm single-pair 29.69ms不是final slot mean31.33ms。最終三題兩輪均序列執行。未從GPU component timer相減推導收益，亦未宣稱去掉CPU wait是全部速度差的唯一原因：schedule、capacity、attention實作也改變。

## 結論／下一決策

介面有實際TPOT headroom，較小的bounded-length variant值得後續評估；目前是速度換buffer預留量，並伴隨浮點schedule差異，不能作零VRAM／零品質代價的免費加速宣稱。下一步先縮減buffer預留與檢查overflow fallback，再決定是否跑完整130題；production default維持已完成品質量測的版本。

## Artifacts與入口

- Fixed slots：results/tpot_slot_interface_20261006/final/{formal,gate,verification.json}；synthetic在results/tpot_slot_interface_20261006/final_micro/。
- Bounded lengths：results/tpot_bounded_lengths_20261006/final/{formal,gate,verification.json,quality_smoke/}。
- Opt-in matched runner：scripts/tpot_execution_opt/run_bounded_lengths.py formal --tag <new-tag> --gpu-lengths；固定使用B0=4096／p=.90，<new-tag>選新目錄避免覆寫。需canonical local model/runtime/cohort，非standalone installer。
- GPU slots runner：scripts/tpot_execution_opt/run_slot_interface.py formal --tag <new-tag> --gpu-lengths。

## 公開準備範圍

本報告及下列新scripts/CUDA source prepare到public mirror，未commit／push。Raw JSON/CSV/log/PT、models、source snapshots、traces、binary、external checkout、完整prediction/reference未納入；本地canonical週報不取代。

本輪應同步檔案：

- `reports/1007/tpot_cache_attention_interface_2026-10-06.md`
- `scripts/tpot_execution_opt/benchmark_slot_interface_case.py`
- `scripts/tpot_execution_opt/slot_interface.py`
- `scripts/tpot_execution_opt/probe_slot_interface.py`
- `scripts/tpot_execution_opt/run_slot_interface.py`
- `scripts/tpot_execution_opt/run_slot_quality_smoke.py`
- `scripts/tpot_execution_opt/slot_miss.cu`
- `scripts/tpot_execution_opt/benchmark_bounded_lengths_case.py`
- `scripts/tpot_execution_opt/run_bounded_lengths.py`
- `scripts/tpot_execution_opt/run_bounded_quality_smoke.py`
- `scripts/tpot_execution_opt/verify_interface_experiment.py`
- `scripts/tpot_execution_opt/report_interface_experiment.py`

Production source／既有baseline adapters未在本輪修改；raw與排除範圍如上。詳細prepared清單保留於本地 `results/tpot_bounded_lengths_20261006/final/publishable_files.json`。
