# RetroInfer／FreeKV／本研究：32K TPOT 分佈與 CPU–GPU profiling

日期：2026-10-05。研究層級：5–6（建立證據、最小量測與原因分析）。

## 問題與假設

使用者要求了解為何 RetroInfer 與 FreeKV 比目前 previous-token cache 方法快。
候選解釋是 dense model execution、selection 工作量、KV 組裝／搬移，以及 host control／同步的差異。
本輪量測不修改方法或 retrieval budget、不重跑130題、不把 profiling 差值當成可直接省下的 TPOT。

## 固定條件與兩種量測

- RTX 5060 Ti 16GB，Llama-3.2-3B-Instruct，BF16，Batch=1；RetroInfer／FreeKV完整decode CUDA graphs off，本研究保留既有 exact Top-p 子圖（非完整model graph）。
- 相同 frozen timing prompts：multikey_3 i011／VT i002／QA_1 i011，实际 token 數31938／32639／32363。
- Formal：每 arm 三題×兩輪 fresh process，交錯順序；32次 fixed-token-1 decode，D1排除，D2–D32 synchronized CPU wall，186個 token samples／arm。不是 greedy generation latency，也不是130題 latency。
- Diagnostic：三題各一組 control／trace，trace只捕捉D3–D5，避開D2 checkpoint logits D2H；9 profiled tokens／arm。不加 intra-token synchronize。保留既有程式同步。
- 新建39個成功process：9 control＋9 trace＋18 formal＋3 FreeKV repeat controls。
- GPU table是 Kineto 真正 kernel／memcpy／memset activity duration；CPU table是同一main-thread的互斥 scope elapsed time，包含scope內等待。GPU/CPU兩表不可相加，不可比例縮放到formalTPOT。
- Diagnostic traces有首次 profiler啟動與 instrumentation overhead，少量token結果只作hotspot證據。捕捉window平均：本研究 62.971、RetroInfer 27.509、FreeKV 36.083 ms/token，非正式TPOT。

### 各方法設定

- 本研究：最新 opt-in merged QKV＋gate/up＋pointwise fusion，Quest B0=8192 → Direct INT4 QK → Twilight dynamic Top-p=.90 → previous-token resident／GPU bitmap／mapped CPU reads；保留原 cache／selector。
- RetroInfer：retrieval=.018、estimation=.232、cache=.05、core=4；Wave Index＋Wave Buffer，官方 local port；formal每次正常建index。
- FreeKV：budget2048／sink512／recent512／page32／corr=.8／spec_ret=True，GPU pool1536MiB；**conservative CPU dispatch**，原background worker overlap停用，保留官方selection／correction／recall。不是原論文完整async效率重現。
- Retrieval語義、精確KV數與品質不同；本輪未建立equal-quality或equal-VRAM比較。前輪130題品質不能當作本輪新測品質；本研究fusion候選仍未完整130驗證。

## 1. 未插樁正式 TPOT 分佈

單位ms/token。分位數是186個 token latency的pooled分位數；token與request非獨立樣本，不當confidence interval。

| 方法 | Mean | Median | P90 | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| 本研究 merged-QKV／p=.90 | 46.511 | 45.065 | 50.665 | 52.060 | 43.531 | 56.098 |
| RetroInfer | 21.870 | 21.533 | 22.589 | 22.823 | 21.274 | 28.127 |
| FreeKV conservative dispatch | 30.548 | 28.692 | 35.460 | 43.031 | 27.527 | 60.718 |

六個run mean範圍：本研究 44.824–50.022；RetroInfer 21.750–22.079；FreeKV 28.800–36.427。

本輪 fresh條件下RetroInfer／FreeKV均較快，FreeKV有明顯長尾。這些新數值不覆寫前輪24.260／32.632／47.944ms的各自歷史run，不能把跨run差异稱成新optimization收益。

## 2. GPU activity 分佈

單位ms／profiled token，三題D3–D5平均。模型類別涵蓋已標記projection／MLP／norm／embedding／LM head與可識別RoPE；未覆蓋kernel列未分類。

| GPU activity category | 本研究 | RetroInfer | FreeKV保守版 |
|---|---:|---:|---:|
| 模型 kernels | 16.409 | 15.982 | 16.860 |
| Selection／correction | 10.600 | 1.222 | 2.245 |
| KV/cache／搬移／組裝 | 8.158 | 1.401 | 5.940 |
| Attention | 1.529 | 0.864 | 0.692 |
| 未分類 kernels／DMA | 0.186 | 0.047 | 0.060 |

GPU busy union跨stream去重：本研究 36.882、RetroInfer 19.516、FreeKV 25.436 ms/token。
Activity sum則為 36.882／19.516／25.797；FreeKV不同stream部分重疊，sum不是critical path。

**實驗觀察：** 最新融合後，三套模型kernel約16–17ms，MLP約10.3–10.4ms，QKV約2.19–2.35ms，LM head約1.92–1.93ms。較大的GPU工作差異在selection與cache，而不是模型本體少算一半。

- 本研究selection約10.60ms；RetroInfer約1.22ms；FreeKV約2.25ms。
- 本研究cache約8.16ms；RetroInfer約1.40ms；FreeKV約5.94ms。
- 本研究attention約1.53ms；RetroInfer約.86ms；FreeKV約.69ms。

這是測量工作量的差異；未做selector交換或equal-budget ablation，不能把差值直接當可移植收益，也不能把低retrieval budget的效率優勢等同相同品質優勢。

### 46.511 ms 與 GPU 表的時間帳：10/05 補充查核

前述前三項 GPU activity 相加是16.409＋10.600＋8.158＝35.167 ms；加上attention1.529與未分類0.186後，總和36.882 ms。這不是正式46.511 ms的完整互斥分解。Formal與diagnostic是不同process／instrumentation／token window；46.511−36.882＝9.629 ms只能算跨run數字差，不能標示為已量出的CPU overhead。

進一步直接分析同一份diagnostic timeline，三題D3–D5平均滿足：**62.971 ms wall＝36.882 ms GPU busy union＋26.089 ms GPU-inactive interval**。把GPU-inactive intervals與最內層main-thread CPU scope相交，位置分佈如下：

| 空檔發生時的CPU scope | ms/profiled token |
|---|---:|
| Selection | 7.667 |
| KV/cache | 7.966 |
| Model API | 4.876 |
| Attention | 2.076 |
| 顯式同步 | 0.081 |
| 未標記scope | 3.423 |

這是空檔所在位置，不是因果歸因：CPU控制／metadata／launch、同步、排程與profiler overhead可能落在上述scope；不能把全部空檔叫純CPU運算或可消除成本，也不能把26.089 ms移植到未插樁46.511 ms。正式46.511 ms尚缺同一run、低干擾timeline的完整分解。GPU-inactive定義僅限本trace所捕捉的kernel／memcpy／memset活動。

Artifact：`results/baseline_tpot_profile_20261005_v5/twilight_gpu_inactive_scope.json`；逐題驗證wall＝busy＋inactive，誤差小於1e-7 ms/token。此補充未新增GPU benchmark、未更改研究程式。

## 3. CPU互斥 scope 分佈

單位ms／profiled token；scope內可能正在等待GPU或搬移，**不是CPU純計算時間**。例如model scope呼叫GPU線性層，selection的blocking D2H可等待先前工作。

| CPU exclusive category | 本研究 | RetroInfer | FreeKV保守版 |
|---|---:|---:|---:|
| Model API scopes | 12.222 | 7.602 | 10.272 |
| Selection scopes | 19.612 | 8.058 | 7.707 |
| KV/cache scopes | 16.133 | 3.235 | 9.912 |
| Attention scopes | 3.151 | 3.451 | 0.984 |
| 顯式同步 | 2.043 | 2.218 | 2.104 |
| 未覆蓋 scope 的 host 時間 | 9.810 | 2.946 | 5.104 |

本研究selection scope約19.61ms，resident assembly scope約11.67ms；FreeKVrecall scope約9.35ms、correction scope約5.54ms；RetroInfercoarse selection scope約8.06ms。
未分類host時間、launch overhead、runtime control與各scope重疊關係仍需timeline解讀；不能以CPU/GPU相減認定某段是純Python成本。

## 4. 資料搬移與特製 kernel

- RetroInfer使用 `concat_gather_copy` 直接從mapped pinned CPU來源組裝GPUexecution buffer，約1ms/token的kernel activity（包含GPU hits／CPU misses組裝）。一般H2D DMA表沒有大量KV搬移，不代表沒有PCIe讀取；本輪未量實際mapped-read PCIe bytes。
- 本研究 `assemble_zero_copy` 約5.7–6.3ms/token；cache類別另含resident／metadata／copies。該kernel同時處理hit与miss，不能將全部時間稱為CPUmiss transfer。
- FreeKV pinned H2D平均 **5.461ms／2077.6calls／32.447MiB per profiled token**。大量page-sized copy是實測，recall的CPU／GPU成本皆較明顯；尚未用copy-granularity ablation確認可省多少。
- 本研究DtoD DMA平均 **1.940ms／532.0calls／539.965MiB per profiled token**。只涵蓋DMA，可涉及INT4候選／metadata／resident資料；CUDA copy kernels不在這個byte總和內，不能全部當previous-token snapshot成本。
- GPU correlation以CUDA runtime／driver correlation指向launch API，再取最內層CPUphase；無correlation覆蓋的activities留未分類。GPU user_annotation跨度不是kernel duration，分析時排除。
- RetroInfer diagnostic AST的一個attention子標記同時涵蓋estimation與retrieved attention，報告統一只用Attention broad category，沒有把此fine label當精確兩階段拆分。

## 5. Correctness／變異與工具限制

### 已通過與未通過的不同gate

- 本研究三題control／trace：D2與D32 logits bit-exact。
- RetroInfer正常freshprocess clustering的最終hash不同，不能用它直接判定instrumentation改結果。原Tritonclustering含atomic reductions／reverse-index ordering，這是source觀察，未證明它是本輪全部變異的原因。
- 為隔離prefill變異，RetroInfer診斷control保存每layer index輸出；trace先驗證完整Key／Value輸入hash一致，再replay同份index。三題D2／D32 logits bit-exact。Replay只用於診斷，formal沒有replay；不宣稱完整greedy序列exact gate。
- FreeKV三題control／trace **不exact**：profiler尚未啟動的D2已不同，max abs .21875–.875；D32 .28418–2.17969。
- FreeKV再補未開profiler的fresh control，也有如下變異：

| Timing request | D2 max abs差 | D32 max abs差 |
|---|---:|---:|
| 001_niah_multikey_3_i011 | 0.468750 | 1.125000 |
| 002_vt_i002 | 0.265625 | 0.312500 |
| 003_qa_1_i011 | 0.386719 | 0.562500 |

- 上述D2/D32比較的argmax均相同，但不保證其他steps／greedy序列／品質。真正原因尚未定位，不把它直接稱為正常浮點誤差或race，也不歸因profiler。
- 全部9個FreeKV diagnostic/control/repeat end-state resident audits皆0mismatched elements／0inverse errors；排除最新兩頁，只證明該範圍的end resident payload，**不能證明每一步attention已完整正確**。
- 因此FreeKV保守版的本輪latency可作local implementation觀察，component與品質解讀須保留未解變異；不能稱為完整correctness已確認的官方效率baseline。前輪130題70.359是保存的一次測試結果，尚未量品質repeatability。

### Profiling工具修復與失敗保留

原CUPTI12.8對本機回報INVALID_DEVICE，第一輪只有CPUtrace；獨立小probe也沒有CUDAevents。
僅下載並解開 `nvidia-cuda-cupti-cu12==12.9.79`，trace subprocess以LD_PRELOAD／LD_LIBRARY_PATH載入；獨立probe出現CUDAevents，最終9traces皆有>100個GPUkernels。
未改Torch／model環境或driver。Formal不載新CUPTI。library hash在manifest，來源wheel與解開library由本地tooling artifact保留。
早期v1缺CUDAtrace、v2import namespace錯誤、v3keyword-only clustering參數错误、v4 harness缺SystemExit後metadata與中途sourcehash改動停批，均保留，未混入v5正式結果。

## 6. 目前判斷與下一決策

**已確認：** 三題freshformal仍有明顯TPOT差距；最新fusion下模型GPUkernel量接近，selection與cache是值得優先查的部分。RetroInfer以cluster selection、低retrieval budget與mapped buffer assembly完成較少GPU工作。

**尚未確認：** 少做多少工作能在本研究同品質條件下成立；降低B0、Top-p或改ANN selector對品質的代價；CPUdispatch與kernel改寫各自能省多少；FreeKV跨runlogits變異的根因。

**下一個最小實驗建議：** 若繼續優化本研究，先把10.6ms selection分成Quest／候選materialization／INT4QK／Top-p／index整理，與8.16ms cache內的UVA assembly／copies分開。保持p=.90與選取語義，優先對最大固定成本做單一ablation。若繼續使用FreeKV作強baseline，先定位D2 pre-profile變異，再談原async恢復；本輪不自行做修復或130題。

## Artifacts與重現

本地artifact root：`results/baseline_tpot_profile_20261005_v5/`。

- `manifest.json`：160個當時source hashes、prompt檔案hash、CUPTIlibraryhash與scope。
- `formal/rep1|rep2/.../result.json`與`formal.csv`：18個未插樁formal。
- `control/`、`trace/`、`control_repeat/`：配對診斷、9份trace、resident audit、D2/D32 tensor與hash、FreeKVrepeat。
- `trace/.../trace_summary.json`：CPU互斥scope、GPUactivities／union、DMA、kernel明細。
- `summary.json`、`analysis.log`：獨立重算latency／分位數／source／prompthash／gates。
- `tpot_distribution.png`／`.pdf`：正式分佈與診斷分佈分開的圖。
- `scripts/baseline_profile/`：profile_case、run_profiles、analyze_profiles、repeat_freekv_controls、plot_profiles。

Scripts依賴本機已準備的research checkout、兩個external官方local ports與venv、模型cache與frozen prompts；不是獨立安裝器。鏡像中的scripts供核對／複製回research tree使用；重現trace前需要12.9.79 CUPTI解開於script指定路徑。

公開mirror準備：`reports/1007/baseline_tpot_profile_2026-10-05.md`與上述5scripts。未納入rawJSON/CSV/log/PT、frozenindex、CUPTIwheel/library、trace、weights/env/binaries／PNG/PDF。本輪沒有production source改動、沒有commit／push。
