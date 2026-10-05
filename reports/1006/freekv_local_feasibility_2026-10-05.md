# FreeKV：RTX 5060 Ti／Llama-3.2-3B 的本機移植與可行性驗證

日期：2026-10-05。研究層級：5–6，外部 baseline 的可行性與 correctness gate。

## 結論與範圍

**官方 kernels 已在本機編譯並執行。保守 CPU dispatch 的 FreeKV port 完成兩次 32768-token context＋64 次 decode，resident KV 資料檢查通過。原始 asynchronous worker 路徑仍未通過資料一致性檢查，不能宣稱完整重現論文效率。**

保守版本保留 previous-query speculative selection、per-head cosine correction、官方 page scoring／TopK／CPU-HND→GPU-NHD recall kernels；背景 CPU worker dispatch 改為同步派發。這是 `FreeKV local port with conservative dispatch`，不是未修改的官方 runtime。沒有 RULER score、formal TPOT、130 題品質結果或與 Twilight／RetroInfer 的公平速度比較。

## 問題、假設與控制

使用者要求在現有環境實際執行 FreeKV。要驗證的假設是：官方 selector、cache 與 recall 能適配 3B 的 GQA ratio 3、sm_120，以及本機 RAM／VRAM 容量。

先做 native HF Full 與 FreeKV dense token 對照，再做同步 retrieval／強制 correction 對照。最後檢查 CPU-backed resident pages 的實際內容與 inverse mapping，而非只以輸出 finite／沒有 crash 當成功。全部使用同一 BF16 checkpoint、Batch 1、固定合成 prompt；品質對照以同一 reference token 序列逐步餵入。

## Source 與環境

- 官方 repository：[sjtu-zhao-lab/FreeKV](https://github.com/sjtu-zhao-lab/FreeKV)，commit `2c8a7d25c9f3c7c15ce15b2f84cd03f477bd7469`。
- 外部 checkout：workspace 的 `external/freekv-20261005/`；獨立 `.venv`，繼承既有 PyTorch，沒有修改 HeadInfer／RetroInfer／SpeCache 環境。
- RTX 5060 Ti 16GB、driver 580.173.02、CUDA 12.8、Python 3.10.20、PyTorch 2.7.0+cu128、Transformers 4.45.2、FlashInfer 0.2.4、Triton 3.4.0、CMake 3.31.6。
- 模型：`meta-llama/Llama-3.2-3B-Instruct`，snapshot `0cb88a4f764b7a12671c53f0838cd831a0843b95`；28 layers、24 Q heads、8 KV heads、head_dim 128。
- FlashInfer／RAFT 使用官方 submodule commits；README 要求的 BF16、GQA 5/7 patch 加上本模型的 GQA 3 dispatch。

## 移植與 correctness 修正

1. 固定 CMake Python executable／headers／library、pybind11 FindPython／SOABI；使用 system NVTX。原建置曾選到 Python 3.13，產生無法在 3.10 載入的 extension。最終載入 `freekv_cpp.cpython-310-x86_64-linux-gnu.so`，API 存在且實際模型測試完成。
2. 新增 GQA group size 3；保留 README 的 BF16／5／7 patch。沒有改寫 scoring、TopK 或 recall CUDA 演算法。
3. `KvCache._decode_alloc_1_page`：完成頁從最後 GPU slot 複製到 recent slot 後，原 `cc2gp` 沒有依實際 `c2p` 更新。第一步量到 **448 個 inverse mapping 不一致**；修正兩個對應關係後歸零。
4. prefill backup 增加 `prefill_backup_stream.wait_stream(compute_stream)`。只修 mapping 後仍在少數 layer 的 page 0／1 出現內容差異；加上 append→backup 相依後，4K 同步／強制 correction resident audits 歸零。
5. 非同步 correction mask 改成 owned clone，避免下一層覆寫共用 buffer。**這項修正不足以讓原 async path 通過檢查**；不能將所有剩餘差異歸因於 mask。
6. 外部 runner 的 conservative dispatch wrapper：官方 `estimate_select_recall` 在 CPU 同步派發；非同步 recall 分支以原 blocking kernel 處理尚未 corrected 的 heads，並記錄完成 events。previous-query selection／correction 規則保留，但 CPU background dispatch overlap 停用。

`infer_state.py`、CUDA `select.cuh`／`estimate.cuh`／`recall_cpuhnd_2buf.cu`／`recall_corr.cu` 與 upstream byte-identical。adapter／kv_cache 的修改完整保存在 patch script 與 `final_port.patch`。保守 wrapper 是 runner 的 runtime adaptation，需和上游 patch 一起揭露。

## 實驗觀察

### Dense 與 retrieval 對照

| 對照 | Context／decode | 相同 argmax tokens | 最大 logits absolute difference |
|---|---:|---:|---:|
| Native HF Full → FreeKV dense | 256／32 | 32/32 | 4.546875 |
| Native HF Full → FreeKV dense | 32768／8 | 8/8 | 2.625 |
| 修正後同步 retrieval → 強制所有 heads correction | 4096／64 | 64/64 | 0.984375 |

Logits **非 exact**，以上只證明這幾組合成輸出的 token agreement。未定位所有浮點差異的原因，也沒有以這些結果宣稱模型品質等價。Native 32K 初次輸出所有位置 logits 時 OOM；改用 HF 內建 `num_logits_to_keep=1` 後成功，attention 未改變。失敗 log 保留。

### 保守版本的兩次 32K 試跑

設定：page_size 32、KV budget 2048 tokens（64 pages）、sink 512、recent 512、per-KV-head group_size 1、CPU HND、spec_ret 開、corr=.8；GPU pool capacity 1536 MiB、CPU pool capacity 4096 MiB。原 pred 的 6GiB／20GiB pools 依本機資源縮小。

| 指標 | 第一次 | 第二次 |
|---|---:|---:|
| Context／decode calls | 32768／64 | 32768／64 |
| captured／final logits finite | 是 | 是 |
| correction layer checks／triggers | 1764／651 | 1764／658 |
| resident audit page-head blocks | 13888 | 13888 |
| audit payload bytes | 227540992 | 227540992 |
| mismatched elements／inverse mapping errors | 0／0 | 0／0 |
| 全流程 peak allocated（含權重／temporary） | 10060.330 MiB | 10060.330 MiB |
| decode end allocated（含權重） | 7740.743 MiB | 7740.743 MiB |
| model parameter bytes | 6127.834 MiB | 6127.834 MiB |
| CPU pool 使用頁容量 | 3591 MiB | 3591 MiB |
| prefill call wall，diagnostic | 9810.579 ms | 9844.047 ms |
| decode model-call mean wall，diagnostic | 34.909 ms | 34.517 ms |

**Timing 定義**：64 次 synchronized `model.forward`，包含第一 decode call；呼叫外做 finite／argmax `.item()`／logits D2H，每步前後全裝置同步，另有 Python call counters。不是正式 steady TPOT，不可拿來與已有 47.944ms／RetroInfer 24ms 排名。CPU worker overlap 已停用。Correction 次數是 layer-level trigger，不是 corrected head 百分比。

**Memory 定義**：GPU pool backing capacity 是 1536 MiB，不是最小必要 KV footprint。pool 已分配頁容量為 1419.5 MiB，包含延後回收的 prefill pages；兩者均不可當成純 selected KV。上表 allocated／peak 包含權重、workspaces、輸出及 temporary；沒有完成與現有方法同口徑的 cache unique VRAM audit。CPU pool 是 pinned，使用頁容量不等於實際有效 token bytes。

Resident audit 在 host workers join／global CUDA synchronize 後，依 `cc2gp` 比對所有已完成 CPU-backed pages，排除最新兩頁；檢查約 217 MiB。這是 end-state 內容一致性，不能證明每個中間 attention step、未測文本／batch 的 correctness 或 selection quality。

### 原 async 路徑的失敗結果

| Runtime revision | end-state mismatched elements |
|---|---:|
| mapping＋prefill ordering 修正，2 workers | 2791092 |
| 同版本，1 worker | 244304 |
| 再加 mask clone，2 workers | 2021760 |
| mask clone，1 worker | 57294 |

這些 run 能完成 decode、輸出 finite，但 payload audit 失敗；不使用其約 28ms 的 diagnostic 數字做速度主張。不同 run 的 output／correction pattern 不完全固定，以上不是只改單一變因的 performance ablation，也不能由差異比例推算各問題的因果貢獻。

**尚未確認**：剩餘差異可能涉及 shared recall buffers、worker enqueue/event ordering 或其他資料依賴；尚未定位，不記成真正原因。

## 執行方式

從 workspace root：

```bash
headinfer/headinfer_reproduction/scripts/freekv_local/setup_freekv_local.sh
headinfer/headinfer_reproduction/scripts/freekv_local/run_freekv_local.sh \
  --mode free --context 32768 --decode 64 \
  --out headinfer/headinfer_reproduction/results/freekv_port_20261005/new_safe.json
```

預設 `--safe-dispatch` 開。`--no-safe-dispatch` 僅供追查已知失敗的原 async path；沒有通過 correctness gate，不作正式效率測量。

## Artifacts、驗證與下一決策

- 本地 raw：`headinfer/headinfer_reproduction/results/freekv_port_20261005/`，包括失敗／成功 logs、JSON、captured logits PT、`provenance.json`、`final_provenance.json`、`final_port.patch`、第三方 patches、`validation_summary.json`／`validation.log`。
- Scripts：`scripts/freekv_local/` 的 setup、run shell、patch、probe、analyzer，共 5 個檔案。語法檢查與 artifact analyzer 通過。
- Analyzer 核對八組成功 artifacts 的 prompt／shape／finite／token agreement、四組 resident payload audits，以及原 async failures，結論限定為 conservative port feasibility。
- 公開 Markdown 準備至 mirror `reports/1007/`；5 scripts 準備至 `scripts/freekv_local/`。raw JSON/PT/log、venv、模型、submodules、compiled binaries 不納入。沒有 commit／push。

下一個決策：**先定位並修復原 async recall 的資料依賴，再決定是否做正式效率 baseline。** 保守版本可先用少量真實 RULER requests 做品質診斷；目前沒有正式 RULER 分數，不直接進入130題或聲稱效率勝負。
