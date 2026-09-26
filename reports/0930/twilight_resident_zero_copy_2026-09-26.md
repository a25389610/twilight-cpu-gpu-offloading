# Twilight resident cache：CPU pinned KV 的 GPU mapped read

日期：2026-09-26（Asia/Taipei）  
性質：維持完整 KV CPU offloading 容量前提的 Selection 外系統優化。

## 問題、假設與失敗方向

先前最佳 previous-token resident cache 的 32K TPOT 約 91–92 ms/token，仍有 CPU bitmap decode、hit/miss mapping、miss KV gather、index H2D 與 selected-KV H2D。假設讓 GPU 直接用 selection bitmap 形成 hit/miss mapping，並從 mapped CPU pinned KV slab 讀取 miss rows，可以降低這些管理成本，同時保留與原路徑相同的 Attention K/V 和 logits。

曾試作完整 32K historical KV GPU mirror；雖有單輪 003 的 80.91 ms/token preliminary 值，卻額外佔用整段 KV 的 VRAM，**違反本研究的 CPU offloading 容量前提**，已移除該 source／runner 路徑。這個值不納入本報告的有效 offloading 比較，也不能作為本次加速的證據。

## 方法與記憶體邊界

完整 historical KV 繼續存放在現有 CPU pinned slab。GPU 只保存前一 token 已選中的 resident K/V，沒有完整 32K KV GPU mirror。新 opt-in `gpu_bitmap` mapping 在 GPU 上由 current／previous membership bitmap 產生 hit source、位置與 group lengths；fused CUDA kernel 對 hit 讀取 GPU resident K/V，對 miss 透過 `cudaHostGetDevicePointer` 映射的 CPU pinned slab 讀取 K/V，直接組成相同的 Attention layout。CPU fast path 只接收每層 8 個 GQA group lengths；Twilight Selection、Top-p、B0/B1、Attention 演算法與 fixed Decode token 設定沒有改動。

`selected_kv_h2d_bytes=0` 只代表不再發出原本的**顯式** selected-KV H2D copy；miss 的 GPU mapped host reads 仍須經過 PCIe。本次每 token 的 mapped miss KV **logical payload** 依 request 為 45.175、49.113、46.134 MiB，平均 46.807 MiB，與舊路徑的顯式 miss H2D logical payload 逐層相等；這不是 PCIe transaction counter，不能宣稱 bus bytes 為零或完全相等。

## 實驗條件與正式計時

- Llama-3.2-3B-Instruct、BF16、FlashAttention 2、RTX 5060 Ti 16 GB、Batch 1；三題 32K cohort `001_niah_multikey_3_i011`、`002_vt_i002`、`003_qa_1_i011`。
- Twilight-GQA dynamic Top-p `p=.90`、B0 8192、32 fixed Decode steps。D1 含 kernel compilation／warmup；正式 TPOT 取未插完整 trace 的 D2–D32 共 31 tokens，同步 CPU wall；排除 Prefill、model load。
- 同份凍結 source，三題各兩輪交錯執行 old/new matched pair。old 是 `native_compact` CPU mapping + Triton hit copy + contiguous resident snapshot + position reference；new 只將 mapping／miss assembly 換成 opt-in `gpu_bitmap` + mapped CPU pinned read。其他 Selection、model、runner 設定一致。
- `results/twilight_resident_zero_copy_v1/matched_final/analysis.json` 由 raw `result.json`／`result.per_token.csv` 重算 TPOT，核對 prompt、模型設定、checkpoint/final logits、每層 hit/miss 數量與 logical payload。所有 6 組通過。

| Request | 第 1 輪 old→new ms/token | 第 2 輪 old→new ms/token |
|---|---:|---:|
| 001_niah_multikey_3_i011 | 92.161 → 73.759 | 92.649 → 73.606 |
| 002_vt_i002 | 90.842 → 73.922 | 92.257 → 73.880 |
| 003_qa_1_i011 | 91.195 → 73.438 | 92.233 → 74.114 |

六組平均 **91.890 → 73.787 ms/token**，paired 平均降低 **18.103 ms/token（-19.70%）**；六組皆改善。這是新機制相對同輪 resident control 的邊際收益，不能把它與其他快照的百分比相加。初次 generic mapper 凍結前的另一組 6 pair 為 91.824 → 73.739 ms/token（-19.70%），趨勢相同，但正式主表採最終 source 快照。

## Correctness、VRAM 與未採用 ablation

- 最終 6 組 formal pair 的 prompt／設定、checkpoint/final logits SHA-256 相同；D2–D32 的每層 resident history、hit、miss rows 相同，共每組 31×28 trace rows。新路徑不在 CPU 重建 group positions，故 formal `twilight_group_positions_sha256=None` 是刻意的 fast-path instrumentation 差異，不能用它宣稱 positions 未檢查。
- 003 的另一次完整 diagnostic gate：Selection trace 21,504 entries、GQA union trace 7,168 entries、D1/D2/D32 Attention K/V trace 84 entries、P4/D1/D2/D32/final logits 均 exact。diagnostic trace 改變執行時間，不能作正式 TPOT。
- 所有 6 組 matched pair 的 **peak GPU allocated bytes old/new 相同**。new 的 final GPU allocated 比 old 約多 6.8–7.0 MiB；resident selected K/V 約 577.7 MiB，本來就存在於 old/new 兩邊。peak reserved bytes 相差 0–2 MiB，受 allocator reservation 影響。這些指標支持本次沒有額外完整 32K historical KV GPU mirror；未量測物理 PCIe transactions。
- 另試專用 compact GPU mapper，6 pair 為 91.697 → 73.772 ms/token（-19.55%），相較 generic mapper 無一致 TPOT 增益，已還原，未納入最終候選。

## 結論、限制與下一決策

在已測 Batch 1、32K、`p=.90` 三題 fixed Decode，GPU bitmap mapping 加 mapped CPU KV read 保住 CPU offloading 的 VRAM 前提，且對最佳 resident control 的正式 TPOT 有約 **19.7%** 改善，已測輸出 exact。這是 default-off 的硬體相依候選：目前 CUDA helper 使用 `nvcc -arch=sm_120`，只在 RTX 5060 Ti 路徑驗證；其他 GPU、Context、`p`、batch、模型及完整 RULER quality cohort 尚未驗證。CPU pinned memory mapping 的實際 PCIe transaction 數、selection exposed 比例和跨平台效益也尚未量測；不從此次 TPOT 差值推論各 component 的可加節省。

下一步以相同容量邊界做 Context×`p` 的 matched TPOT、GPU allocated／reserved、mapped logical payload 與 correctness／quality 驗證；再取得新候選同輪 common-origin timeline，確認 Selection 是否真正成為最大 exposed 成本。其他 GPU 需先處理 architecture-specific CUDA build，再驗證 zero-copy 支援與效能。

## Artifacts 與 source provenance

- 正式結果：`results/twilight_resident_zero_copy_v1/matched_final/{manifest.json,analysis.json,rep1,rep2}/`，raw JSON／CSV 只在本地。
- 完整 003 diagnostic gate：`results/twilight_resident_zero_copy_v1/diagnostic003/`，大型 trace 只在本地。
- 未採用 ablation：`results/twilight_resident_zero_copy_v1/matched_compact/analysis.json`。
- Source：`source/headinfer/headinfer/{twilight_offload_cache.py,resident_gpu_mapping.py,resident_zero_copy.py,resident_zero_copy.cu}`；runner／validator：`scripts/{run_ruler_partial_h2d_tpot_case_v1.py,analyze_twilight_resident_zero_copy_v1.py}`。
- 最終 source SHA-256：`twilight_offload_cache.py` `decd79a8c19b5433366a12b6640e9c9ca4b0c9193a9ddfc14a60a619363cbf7b`；`resident_gpu_mapping.py` `f92a3b5135bcbd4fb5279d6ce679df2f9003f6214fdcaf322183be06d440aec2`；`resident_zero_copy.py` `c7553f75cc2fbee7bc3a19fe20809d5635c012a36acacd40a1aff7a3b9b5cc21`；`resident_zero_copy.cu` `ccf6b381ddc23c3deac8289fa89e00692259468b0a086504cc2844bfb95a37d8`；case runner `57b3aacee141d91092c4bb24340fa5fd14216a8bcd1f6118c724c2df31c81fce`。
