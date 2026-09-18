# Twilight CPU–GPU Offloading Optimization Evidence

本資料夾整理 2026-09 期間本地 `Twilight-inspired` CPU-offload prototype 的
execution optimization、profiling 與負結果，目的是避免把已完成或已否定的方向重新列為
未嘗試工作。

## 研究範圍與限制

- 本地實作是 correctness-first `Twilight-inspired` PyTorch CPU-offload prototype，
  不是官方 Twilight efficiency kernel 的直接效能重現。
- 報告中的 CPU wall、CUDA Event 與正常未插樁 TPOT 具有不同 clock domain 或
  instrumentation scope；除非報告明確定義為互斥分區，否則不可直接相加或相減。
- 多數小型 pilot 使用 Llama-3.2-3B、Batch 1、32K、`p=.90` 與 RTX 5060 Ti；
  單一 request 或單輪結果均保留其 preliminary 限制。
- 本資料夾收錄可閱讀的研究報告；報告引用的 raw JSON、CSV、log、checkpoint 與
  timeline artifacts 仍保存在本地 reproduction workspace，未全部複製至 GitHub。

## 證據索引

| 主題 | 報告 | 已確認事項 |
|---|---|---|
| CPU bitmap union／flat gather | `twilight_cpu_pack_ram_2026-09-10.md` | CPU exact bitmap union 已實作；`torch.unique` union 大幅下降，flat gather 單獨收益小。 |
| Padded final indices | `twilight_index_wait_2026-09-10.md` | 每 token 約 44.05 MB indices transfer；blocking `.cpu()` 同時包含 queued GPU work 等待，並非純 D2H。 |
| Final-index fusion | `twilight_final_indices_2026-09-11.md` | Triton gather/mask fusion 縮短局部 device bundle，但未證明 TPOT 改善。 |
| CPU gather／packing attribution | `twilight_gather_investigation_2026-09-10.md` | 多數 gather wall 位於 `index_select`；目前不能斷言已達 DRAM 或 PCIe ceiling。 |
| Gather continuity audit | `twilight_gather_chunks_audit_2026-09-14.md` | 實際 selected rows 以短 run 為主；單純 threshold／run-copy 不具大幅收益證據。 |
| OpenMP／run-copy prototypes | `twilight_gather_improvements_2026-09-14.md` | unused views 有小幅訊號；hybrid run-copy 沒有改善。 |
| Direct Attention layout | `twilight_native_layout_2026-09-10.md` | 已避免完整 history GPU `cat`；native C++ gather 無穩定收益。 |
| Gather→H2D overlap | `twilight_chunk_pipeline_2026-09-11.md` | 4-chunk pipeline 在 matched pilot 約改善 10.2%，payload 與 selection 不變。 |
| RoPE／layer projection | `twilight_layer_projection_rope_2026-09-10.md` | RoPE batching 通過 exact gate；full-layer QKV batching 未通過 bit-exact gate。 |
| GPU metadata scheduling | `twilight_early_metadata_2026-09-10.md` | 提前建立 `cu_seqlens` 有 preliminary TPOT 訊號，但不能把 timer 移位誤認成刪除相同工作。 |
| Quant metadata reuse | `twilight_metadata_reuse_2026-09-11.md` | 重用固定 INT4 metadata 通過 exact gate並降低 TPOT。 |
| Fused INT4 preparation／direct QK | `twilight_fused_prepare_2026-09-09.md` | `triton_prepare` 通過 exact gate；direct INT4→QK prototype 因 reduction parity 未通過。 |
| Fused Quest score | `twilight_fused_quest_2026-09-14.md` | 模仿目前 reduction order 的 Quest kernel 通過 exact gate，屬工程融合而非新 selection 演算法。 |
| 完整 TPOT accounting | `twilight_complete_tpot_2026-09-14.md` | CPU 互斥 wall 與獨立 GPU Events 的最新 scope 定義及限制。 |
| 最新三題總結 | `twilight_three_request_complete_2026-09-15.md` | 最新 direct path、舊路徑重建、normal TPOT、Selection 細分與 correctness gate。 |

## 尚未完成的主要方向

- 在 GPU 直接形成 GQA union 或 compact representation，避免 per-Query-head padded
  indices 全量回傳 CPU。
- 將 Sink＋Recent KV 保持 GPU-resident，並在離開 recent window 時才批次 D2H；
  必須補長 decode、eviction 與 buffer-lifetime correctness。
- 量測獨立 pinned-memory H2D hardware ceiling，再決定是否值得修改 K/V host layout。
- 使 direct INT4→QK kernel 通過目前 PyTorch matmul reduction／Top-p boundary parity。
- UVA／mapped pinned memory 直接存取 CPU KV；尚未在目前 consumer platform 實測。
