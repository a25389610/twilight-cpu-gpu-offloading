# RetroInfer：32K RULER TPOT、KV VRAM 與 130 題時間估算 — 2026-10-05

## 問題、原先假設與已確認範圍

使用者要求沿用目前研究的 32K RULER 題目，量 RetroInfer TPOT、排除 3B 權重後的 KV 相關 GPU memory、單題時間，並估算 130 題品質測試需要多久。原先只有重複句子的可行性驗證，無真實 RULER TPOT / VRAM / 任務時間。

本次完成 **3 題 fixed timing × 2 fresh-process reps、3 題獨立 memory audit，以及 13 tasks × 1 題 greedy quality/runtime pilot**。尚未執行完整 130 題，不將 13 題分數當作 130 題總分。

## Source、題庫與固定條件

- RetroInfer SHA `03f912c6e917c380d9d90c5ec85bb0f161ba53ef`；weighted attention SHA `56d96228ada74d6df806b0083bf018d0d57f57e9`。官方 tracked source hashes 在本輪前後一致。
- 本機 RTX 5060 Ti 16GB，Llama-3.2-3B-Instruct、BF16、Batch 1、greedy、Full prefill、4 CPU worker threads、CUDA graphs off。
- PyTorch 2.7.0+cu128、Triton 3.4.0、FlashInfer 0.2.4、FlashAttention 2.8.3.post1。獨立環境保留 torch/Triton pin mismatch；套件相容性 gate 見 `retroinfer_local_feasibility_2026-10-05.md`。
- 官方參數：cache_ratio=.05、retrieval_budget=.018、estimation_budget=.232。沒有調參以配合本輪答案。
- 固定題庫 `results/context_p_twilight_gqa_group_ruler130_v1/cohort/32768/`，dataset `SaylorTwift/RULER-32768-llama-3.2-tokenizer`，revision `e748e0cd1872b4bbaa6d5ed9c6fbcf6068951c42`；13 tasks 各 10 題。
- 直接讀既有 `prompt_ids.pt`，逐題驗證 comma-separated token hash 與 `prompt_token_count`。不重 tokenize、不改 chat template、不 padding 成精確 32768。
- 「32K」為相同 context 題庫標籤；本輪 timing 真實 input 31938 / 32639 / 32363 tokens，quality pilot 31886–32651 tokens。
- Timing 使用與既有研究相同的三題 timing-only requests，與 accuracy cohort 分離。Quality pilot 從每個 task 取 frozen order 第一題，未依 RetroInfer 得分挑選。

## Formal synchronized TPOT

與現有 protocol 一致：Prefill 後每次輸入 fixed token ID 1；執行 D1–D32，D1 warm-up 排除。每個 forward 前後 CUDA synchronize，CPU `perf_counter` wall；31 個 D2–D32 平均。timer 內沒有每步 logits D2H、finite check、KV audit、component profiler 或 scorer。最後 finite check 與 JSON write 在計時外。

| Timing request | 兩輪平均 ms/token |
|---|---:|
| 001_niah_multikey_3_i011 | 24.299 |
| 002_vt_i002 | 24.074 |
| 003_qa_1_i011 | 24.408 |

六次平均 **24.260 ms/token**，六次個別 mean 範圍 **23.903–24.856 ms/token**。這是新環境適配實作在上述條件的實測，非官方 A100 論文數字。並非 greedy 作答 TPOT；greedy query 的選取與輸出長度可能不同。

尚未與目前 Twilight 方法做 fresh paired rerun 或 matched-quality / equal-memory 比較。兩個推論框架的非 attention 計算與 weight layouts 不同，不能將全部速度差異歸因於 KV reuse。

## GPU memory：排除權重，分清 live cache 與 peak

三題獨立 memory audit 在 fixed D32 之後遞迴列出 cache Tensor，按 `(device, storage.data_ptr, storage.nbytes)` 去重。aliases / views 只算一次；CPU tensors 不納入 GPU 合計。下表為三題平均 MiB（1 MiB=2^20 bytes）。

| Cache 相關 GPU backing storage | MiB |
|---|---:|
| resident KV | 176.750 |
| steady KV | 10.938 |
| execution KV buffer | 9.474 |
| cluster index / centroid / summed Value / cluster-size metadata | 222.085 |
| selected-cluster estimation workspace | 1.837 |
| other selection / metadata workspace | 0.256 |
| **Cache unique GPU storage 合計** | **421.339** |

- 常駐與工作區全部納入的 cache unique 合計 **421.339 MiB = 0.411 GiB**；三題範圍 416.544–425.624 MiB。
- 精確 32K 完整 BF16 K+V 容量為 `28×8×32768×128×2×2=3,758,096,384 bytes=3584 MiB`。上述 cache backing 為此固定容量的 **11.76%**。本輪實際 prompts 略短於 32K；此分母是容量參考，不是本輪 Full GPU TPOT 或品質結果。
- 實際權重 unique GPU backing **6879.334 MiB**。逐一盤點官方 Llama 的 embed、lm_head、fused layer weights 與 norms，不用模型名稱估算「3B × 2 bytes」。3B config 設 tie_word_embeddings；官方實作分別搬 embed 與 lm_head 到 GPU，形成額外權重副本，因此其實際權重 backing 與一般 HF tied-weight loader 不必相同。
- Decode end allocated 減權重平均 **447.478 MiB**；D2–D32 peak allocated 減權重平均 **448.218 MiB**。
- End allocated 減權重與 cache unique 差 **26.139 MiB**，包含 RoPE / positions、輸入、logits 及其他未歸類配置；其中 model-only 非權重 allocation **17.261 MiB**。不把差額全稱為 KV。
- Prefill peak allocated 減權重平均 **2847.931 MiB = 2.781 GiB**；包含 prefill activations、分群與其他 temporary，**不是 steady KV cache 大小**。
- 這是 PyTorch allocator 與可見 Tensor storage 口徑；GPU driver/context、非 PyTorch allocation 與桌面其他程序未納入 cache unique。

## 單題真實 RULER 作答與品質 pilot

Quality 使用每題原先 budget、greedy、tokenizer EOS 128009。與既有 quality runner 一致，至少執行兩步，生成至 EOS 或 frozen budget；輸出包含 Prefill 產生的第一 token。計時含 cache allocation / init、Full prefill、分群、prepare、decode 與同步，排除模型載入、process imports、結果寫檔；另保存 parent subprocess wall 供估算每題重新啟動的成本。

Scoring 直接使用現有研究腳本的 `official_string_match_all` / `official_string_match_part` / `official_ruler_score` 函式定義；另重新計分驗證所有 pilot rows。P4 與最後 logits finite；未宣稱所有 tokens logits 逐步一致。

| Task | 真實 prompt tokens | 生成/budget | Score /100 | 作答秒（不含載入） | 完整 process 秒 |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 32484 | 7/128 | 100.00 | 11.851 | 16.956 |
| niah_single_2 | 32174 | 6/128 | 100.00 | 11.163 | 16.308 |
| niah_single_3 | 32197 | 29/128 | 100.00 | 12.083 | 17.187 |
| niah_multikey_1 | 32228 | 6/128 | 100.00 | 11.166 | 16.254 |
| niah_multikey_2 | 32582 | 7/128 | 100.00 | 11.391 | 16.621 |
| niah_multikey_3 | 31886 | 26/128 | 0.00 | 11.608 | 16.777 |
| niah_multivalue | 32223 | 29/128 | 100.00 | 11.720 | 16.873 |
| niah_multiquery | 32261 | 29/128 | 75.00 | 11.761 | 16.828 |
| vt | 32642 | 15/30 | 80.00 | 11.619 | 16.785 |
| cwe | 32397 | 45/120 | 0.00 | 12.266 | 17.325 |
| fwe | 31960 | 31/50 | 66.67 | 11.725 | 16.826 |
| qa_1 | 32552 | 21/32 | 0.00 | 11.716 | 16.936 |
| qa_2 | 32651 | 3/32 | 0.00 | 11.292 | 16.401 |

- 作答平均 **11.643 s/question**，範圍 **11.163–12.266 s**。
- 每題新 process 並載入模型，平均 **16.775 s/question**。
- 13 題 macro mean **63.21/100**；這是 13 tasks × 1，**不是 130 題分數**。沒有同批 fresh Full / Twilight 品質對照，不能將錯誤全歸因於 RetroInfer selector。
- 使用官方默認 retrieval/estimation budget 即能得到速度數字，但品質是否與現有研究相當尚未確認。

## 130 題時間估算

130 cohort 每 task 10 題。本輪各 task 1 題實測秒數乘 10，加總：

| 執行方式 | 預估總時間 | 證據狀態 |
|---|---:|---|
| **目前 runner：每題 fresh process / model** | **36.35 分鐘** | 13 題 stratified 線性外推，未跑完整 130 |
| 同一 process / model reuse，排除重複 imports 與 model load | 25.25 分鐘 | 以作答時間加一次 model load 推算；共享模型 runner 尚未測試 |

這不是完成時間保證。每 task 僅一題，題目難度、輸出長度、GPU clock 與其他負載會造成變動。全 cohort budgets 合計 12880 tokens，但 pilot 實際生成大多提早 EOS；不能只用「130 × 128 × TPOT」估算。模型評分採字串 match，成本相較模型運算很小；整套 launch/I/O 成本以 parent process wall 反映。

## Artifacts 與重跑

相對 `headinfer/headinfer_reproduction/`：

- `results/retroinfer_ruler_20261005_v1/manifest.json`：frozen source / cohort hashes 與 pilot selection。
- `formal/rep1,rep2/*/result.json`、`memory/*/result.json`、`quality/*/result.json`：逐題 raw結果。
- `formal.csv`、`memory.csv`、`quality_pilot.csv`、`summary.json`、`analysis_provenance.json`。
- 各 job `command.json`、`process.log`、`process_wall.json`；總 log `results/retroinfer_ruler_pilot_20261005.log`。
- 前導第一題保留於 `results/retroinfer_ruler_20261005/`，不納入六次 formal 平均。
- Scripts：`benchmark_retroinfer_ruler_case.py`、`run_retroinfer_ruler_local.sh`、`run_retroinfer_ruler_pilot.py`、`analyze_retroinfer_ruler_pilot.py`。

```bash
python headinfer/headinfer_reproduction/scripts/run_retroinfer_ruler_pilot.py
python headinfer/headinfer_reproduction/scripts/analyze_retroinfer_ruler_pilot.py
```

Runner 支援相同 manifest 的 resume；已有有效結果會跳過。欲 fresh repeated experiment 需建立新輸出位置並保存新 manifest，不覆寫舊結果。

## 下一決策

目前已能回答本機 TPOT、KV-related memory 與品質測試時間。正式比較前，先完成相同 frozen 130 題的品質對照，確認近似 budget 是否達到共同品質門檻，再解讀速度 / VRAM 優勢。Graphs off 是本輪驗證過的 configuration，不代表 RetroInfer 的最佳可達速度。
