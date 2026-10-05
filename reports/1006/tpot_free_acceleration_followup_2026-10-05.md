# 10/05 TPOT 續優化：保留 selection 的 CUDA QK／Quest 與 union fusion

## 問題與實驗目的

使用者要求把目前約 40.77 ms 的 Llama-3.2-3B CPU–GPU offloading 版本繼續加速，目標接近本地 FreeKV。這輪以已完成的 GPU-cu／buffer-swap 版本為基準，保留 Twilight p=.90、B0=8192、previous-token resident reuse 與原 attention。沒有透過減少 KV budget 換速度。

## 正式 TPOT

三個 frozen 32K requests（niah_multikey_3、vt、qa_1），每題兩輪、三個 arms，共 18 個 fresh processes。Batch 1、BF16、RTX 5060 Ti；fixed token ID=1，32 個 decode forwards；排除 D1，計算 D2–D32 CPU wall end-to-end latency。每 arm 共 186 個 samples；無 profiler／payload capture，三 arms 交錯執行。

| 版本 | 平均 TPOT ms |
| --- | ---: |
| 本輪同批 GPU-cu／swap 基準 | 40.572783 |
| integer union fusion＋Triton Quest tile16 | 39.376610 |
| union fusion＋explicit-order CUDA QK／Quest | 38.027293 |

最佳版本比同批基準少 2.545490 ms（6.2739%），六個 request×rep 配對皆較快。最佳 run means 為 36.150930–38.802155 ms，基準 38.370728–42.289879 ms；有跨 process 漂移，不把最快單次當常態。相對中間版本再少 1.349317 ms。歷史 48 ms 與前輪 40.772061 ms 僅供版本脈絡，不用來計算本輪新增收益。

**尚未達到 30 ms；不是與 FreeKV／RetroInfer 相同品質、selector 與 budget 的公平勝負結論。**

## 實作改變

1. GQA selected-token union：原多個 gather／mask／scatter 操作改成兩個 integer Triton kernels，保留 protected tokens 與 sentinel。避免動態 length 的 specialization 編譯。
2. Direct INT4 QK：從本機參考 Triton PTX 的 reduction tree 推導 CUDA 實作；8 lanes/token，保留 16 項 sequential accumulation，再依指定 shuffle tree 合併。explicit RN mul/add，FMA disabled，減少原跨 warp shared-memory 工作。
3. Quest page scores：同 page extrema 在三個 GQA Query heads間重用，保留原逐 chunk 累積與 cyclic shuffle 次序。沒有換 selector 或近似公式。

Production default 未修改，提供 process-local experimental preset `cuda`。支援已驗證的 BF16、head_dim128、8 KV groups、3 Query heads/group、B0=8192、sm_120；CUDA QK candidate count 必須可被 4 整除。其它模型／Triton layout／硬體須重新 gate，不能推論通用 bit-exactness。

## Correctness 與限制

所有正式配對 checkpoint／final logits hashes 與逐步 resident history／hit／miss trace 相同。另做 QA 128 fixed-token forwards 的 diagnostic gate，selection、GQA union、Attention K/V、CPU new-KV、resident traces 與 checkpoint/final logits 相同；selection／attention payload capture 是 D1／D2／D32／D128，並非每一步完整 payload proof。沒有 130 題 RULER 品質分數，fixed-token throughput 也不是自然 EOS request completion time。

## 顯存：本輪沒有新增 PyTorch allocation

另六個 fresh memory processes（三題×baseline/CUDA），每題的 cache backing、D32 live allocation、nonweight live 與 peak allocation 均相等；inventory process 的計時不混入正式 TPOT。

| Scope | 基準與 CUDA 都是 MiB |
| --- | ---: |
| cache unique GPU storage | 1434.760930 |
| D32 live allocated | 7598.911947 |
| D32 live minus model parameter/buffer storage | 1471.070882 |
| peak allocated | 7626.583333 |

這是 PyTorch allocated/storage inventory，不是 NVML process VRAM；不包含 driver/context 或 opaque CUDA code/library allocation。沒有抹去前輪 buffer swap 約 +110 MiB cache backing 的既有代價；只證明本輪新增 kernels 未增加上述 storage。

## 為何仍與其它系統有差距

對本輪基準做一次 D3–D5 GPU profiling：model 16.387974、selection 9.338877、cache 6.009727、attention 1.503423 ms/token；GPU busy union 33.426005 ms，instrumented wall 53.243427 ms。這是三 tokens 的 diagnostic，不能加總或與正式 40.57 ms 相減。這不是最佳 CUDA 版本的 profile。

目前 assemble_zero_copy 約 5.701426 ms，QK 約 2.431774 ms；原 CPU 控制與 tensor 操作也有成本。歷史 cache 8.158 ms 不代表目前版本。RetroInfer 的 1.4 ms 來自不同 execution／retrieval 工作量，不能直接宣稱相同 KV 搬移能省到 1.4 ms。PCIe 當下 Gen4×8 是 link observation，未量實際 traffic utilization。

## 未採用的實驗

- 固定 CTA mapped-copy：payload exact，但 90% hits 的 micro 約 0.135–0.157 ms，未優於原約 0.135 ms；all-hit 改善不足以代表實際 workload。
- GPU lengths／完整 context capacity headroom：單題 40.391027→37.442877 ms，但 D2/D32 logits exact gate 失敗，後續 selection counts 改變；且非 equal VRAM。未部署，不算免費加速。
- mapping scan tile：valid mapping exact，替代 block/warp 未穩定加速。
- QK warp1/2/8：FP32 非 exact，未使用。
- 初次 union pilot 44.482 ms 包含動態 specialization 編譯；原失敗 artifact 保留，修正後重新測量。

## 可執行入口

從 canonical research root 執行：

```bash
/home/paul/miniconda3/envs/headinfer_repro/bin/python \
  scripts/tpot_execution_opt/run_followup_optimized_case.py \
  --request-dir results/context_p_ruler_39_v1/cohort/32768/timing_requests/001_niah_multikey_3_i011 \
  --output results/my_followup/result.json --preset cuda --decode-steps 32
```

入口記錄 env／command／source hashes，NVCC 編譯在模型載入與 prefill 前。`baseline` 可回到本輪比較基準；`optimized` 是中間 union＋tile16。現有 merged-QKV experimental runtime 的算術與更早 grouped runtime 並非 bit-exact，這輪 exactness 是相對已完成 GPU-cu／swap 基準。

## Artifacts 與公開範圍

本地：`results/tpot_followup_20261005/final_cuda/` 的 formal／gate／memory、各 stage manifest／source snapshot、analysis 與 verification；`validated_combo/` 是先完成的兩-arm study。其它 micro、failed headroom、current_profile 都保留在同一 results root。

Source：`scripts/tpot_execution_opt/` 的 followup runner、verification analyzer、union／Quest／QK adapters、CUDA kernels、micro／gate／profile scripts 與 README-followup.md。此鏡像是 source/report collection，不是 standalone installer，需原 canonical runtime、模型、requests 與本機環境。

GitHub-friendly 報告準備到 `reports/1007/`，相關 `.py`／`.cu`／README 準備到 mirror scripts；不納入 raw JSON/CSV/log/PT、profiler trace、compiler PTX/IR、binary library、model 或 external checkout。未 commit／push。

## 下一決策

優先處理 GPU-length handoff 與 mapped CPU KV assembly 的實際 critical path，但必須維持有效 max_seqlen 的 attention 算術、equal physical capacity 與 gate；目前 headroom 失敗版本不能沿用為成果。若要省更多 selection 時間，需另開 quality-controlled 方法研究；本輪沒有以改 p 或 B0 冒充 execution 加速。
