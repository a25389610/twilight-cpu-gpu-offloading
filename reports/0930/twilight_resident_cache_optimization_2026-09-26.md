# Twilight previous-token resident cache 成本優化

日期：2026-09-26（Asia/Taipei）  
性質：correctness-first matched systems ablation；原 Selection、GQA union、Attention K/V、KV precision 不變。

## 問題與原先假設

09/22 的 previous-token resident cache 能將 selected-KV CPU gather/H2D 大幅降低，但當時的 CPU hit/miss mapping、index H2D、GPU hit copy 與每 token snapshot 讓 TPOT 由 120.732 升至 128.963 ms/token（三題單組 matched pair；歷史比較，不能當本輪分母）。本輪測試：在不改輸出邏輯下，合併 CPU mapping 與 GPU hit-copy 工作是否能讓 resident reuse 真正降低 TPOT。

## 條件與計時口徑

- 模型 `meta-llama/Llama-3.2-3B-Instruct`、BF16、FlashAttention 2；RTX 5060 Ti 16 GB、PyTorch 2.7.0+cu128、CUDA 12.8。
- Batch 1、三題固定 32K cohort：`001_niah_multikey_3_i011`、`002_vt_i002`、`003_qa_1_i011`；Twilight-GQA dynamic Top-p `p=.90`、8192-token B0。
- 32 fixed Decode steps；D1 warm-up，正式 TPOT 是 D2–D32 共 31 tokens 的同步 CPU wall mean。Prefill、model load、D1 不計入 TPOT。
- 兩邊共同使用 `triton_prepare`、GPU compact GQA union、CPU flat gather、4-chunk H2D、direct Attention layout、layer RoPE、token-level new-KV D2H。唯一主要差異是 cache-off control 與下述 default-off optimized resident cache。
- Formal TPOT run 不插 Selection/Attention trace；correctness smoke 與 post-timing D33 component diagnostic 分開，不把 diagnostic component 當正式 TPOT。

## 改動與 correctness

1. CPU mapping：以 bounded token position 作直接索引，native C++ 單次產出 hit source/destination、miss source/destination 與 history destination；取代每 KV head 的 `torch.searchsorted`、多個 PyTorch 遮罩與串接。native library 在 model timing 前編譯。
2. GPU hit copy：Triton 單 kernel 複製 K/V resident hits 到 Attention layout，避免兩次 `index_select` 的中間 hit-gather 張量及後續 `index_copy_`。hit count 是 runtime 參數，避免對每個 token 重新編譯。
3. Hit source/destination 索引用 `int32` 傳到 GPU；miss destination 與 history destination 維持 `int64`，以符合現有 PyTorch consumer。C++ 對 `int32` 範圍做檢查。
4. `snapshot` 與 `miss scatter` 在最終候選維持原路徑。以上選項皆 default-off；不啟用 resident cache 時，原控制路徑保持原設定。

150 組隨機 CPU mapping 測試逐項核對五條輸出索引；Triton hit-copy 多個大小的 K/V 逐元素 exact。完整 `003` 32K smoke 的 selected-membership SHA、GQA-union SHA、D1/D2/D32 Attention K/V trace、P4/D1/D2/D32/final logits 及 new-KV trace 均與原 resident 路徑 exact。最終三題 formal pair 的 checkpoint/final logits 與 post-timing selected-position/GQA-union SHA 皆 exact。

## 最終 fresh matched TPOT：cache-off → optimized cache

此表使用 **目前程式快照**的每題一組 fresh matched pair；不以 09/22 歷史 TPOT 當分母。

| request | cache-off control | optimized cache | 差值 | TPOT 降幅 |
|---|---:|---:|---:|---:|
| 001_niah_multikey_3_i011 | 121.770 | 93.425 | -28.345 | 23.28% |
| 002_vt_i002 | 119.602 | 95.156 | -24.447 | 20.44% |
| 003_qa_1_i011 | 122.796 | 94.868 | -27.929 | 22.74% |
| 三題平均 | **121.390** | **94.483** | **-26.907** | **22.17%** |

在 metrics-only 記帳修正前的同一 execution 設計，另有每題兩組交錯順序 matched pair：六組平均 **121.341 → 95.041 ms/token**（-21.67%）。這是另一個 source 快照的重複性支持，不能把其 ms 與上表混成單一 paired 統計。

## 單機制 ablation 與失敗嘗試

以下每行都有其**各自同輪 matched control**，行與行不可直接相減成元件的可加 TPOT 貢獻。

| 唯一改動 | 同輪 control → candidate TPOT ms/token | 判斷 |
|---|---:|---|
| 原 resident mapping → native fused direct index | 133.576 → 105.053（三題平均，-21.35%） | 三題一致有效 |
| native mapping 下 PyTorch hit copy → Triton hit copy | 102.326 → 95.528（-6.64%） | 三題一致有效 |
| 已融合 hit copy 下 PyTorch snapshot → Triton snapshot | 96.277 → 96.051（-0.23%） | 003 變慢；不採用 |
| native mapping 下 `int64` hit index → `int32` hit index | 95.595 → 94.760（-0.87%） | 三題小幅一致有效 |

第一版 Triton hit-copy 曾把逐 token 變動的 hit count 設為 compile-time constant；001 有 868 次 hit-copy 呼叫、843 種不同 count，導致 001/002 TPOT 超過 1.6 秒/token。改為 runtime count 後，100 個不同 count 的小測試於 0.242 秒完成；重新跑三題 matched formal 才採用上表結果。這個失敗版本的 TPOT 不代表 Triton kernel 正常穩態效能。

## 正式 reuse 與單 token 診斷

正式 D2–D32 三題 row-weighted previous-token hit 為 **90.782%**。對應的 selected history KV payload 是 **507.805 MiB/token**；實際 miss-only selected-KV H2D 為 **46.807 MiB/token**。這是 selected-KV payload 的下降，未包含新增 index H2D，也不能直接推論同百分比 TPOT 節省。

以下只取 `003` formal timing 後的一個 diagnostic token（D33）；CPU wall 與 CUDA Event active scopes 可重疊，**不可相加為 TPOT**。

| component | cache-off | optimized | scope |
|---|---:|---:|---|
| selected-KV CPU gather | 40.127 ms | 3.856 ms | CPU wall |
| selected-KV H2D | 38.788 ms | 3.763 ms | CUDA Event |
| resident CPU mapping | 0 | 4.328 ms | CPU wall |
| resident index H2D | 0 | 3.883 ms | CUDA Event |
| resident hit K/V copy | 0 | 3.386 ms | CUDA Event |
| resident miss scatter | 0 | 0.517 ms | CUDA Event |
| resident full snapshot | 0 | 5.168 ms | CUDA Event |
| membership-complete → Attention-ready | 75.123 ms | 48.377 ms | 同一 parent CPU wall |

D33 selected-KV H2D logical bytes 為 **533,213,696 → 40,916,992 bytes**；optimized 另有 **16,659,344 bytes** resident index CPU→GPU。hit copy **492,182,016 bytes**、snapshot **533,099,008 bytes** 是 GPU→GPU logical payload，不是 PCIe bytes。正式三題 resident K/V allocated object mean **577.713 MiB**；matched final CUDA allocated delta mean **580.352 MiB**，CPU pageable selected-position metadata mean **7.738 MiB**。Triton hit copy 不配置先前 PyTorch hit-gather 暫存張量。

## 結論與限制

在此 Batch-1 32K `p=.90` 三題配置中，**optimized previous-token resident cache 已由負收益變成一致的 TPOT 正收益**，且已測的輸出邏輯維持 exact。這是系統實作的條件式結果；尚未測 4K/8K/16K、其他 `p`、batch、模型、GPU 或 RULER quality cohort，也不能宣稱全域最快。

- 目前程式快照的 formal TPOT 每題只有一組 fresh pair；另一快照有每題兩組 pair，兩者分開陳述。
- D33 diagnostic 只是一個 token；元件 active time、parent exposed wall 與正式 TPOT 必須分別解讀。
- snapshot 仍有約 5 ms GPU active cost。這輪 Triton snapshot ablation 未顯示穩定 TPOT 收益，因此不採用；下一步若要消除此成本，應先設計不破壞 resident/Attention exact order 的 buffer ownership 實驗。
- `native_compact` 與 Triton hit copy 需要本地 C++ compiler/Triton；正式計時排除 native compile/model load，但部署環境仍須具備相依套件。

下一個研究決策：保留 optimized cache 為 default-off candidate，先在其他 Context/`p` 做 matched TPOT 與品質範圍驗證，再決定是否成為 production baseline；不要把本輪三題結果直接外推。

## Artifacts

- 目前程式快照：`results/twilight_resident_optimized_current_v1/{manifest.json,formal,analysis}/`；分析命令 `scripts/analyze_twilight_resident_optimized_v1.py`。
- 重複配對：`results/twilight_resident_optimized_v1/{manifest.json,formal,diagnostic}/`。
- Mapping ablation：`results/twilight_resident_mapping_native_v1/`；hit-copy ablation：`results/twilight_resident_hit_copy_triton_fixed_v1/`；未採用 snapshot：`results/twilight_resident_snapshot_triton_v1/`；compact-index ablation：`results/twilight_resident_compact_index_v1/`。
- 主要程式：`source/headinfer/headinfer/{twilight_offload_cache.py,resident_mapping.py,resident_mapping.cpp,resident_hit_copy.py}`；runner：`scripts/{run_twilight_resident_optimized_v1.py,run_ruler_partial_h2d_tpot_case_v1.py}`。
- 目前程式快照 source SHA-256 見 `results/twilight_resident_optimized_current_v1/manifest.json`。例如 `resident_mapping.cpp` 為 `b9c8fcc264a36de5c6d7da946c8bb3971a9684ad8287e6d5c798247326138e00`。
