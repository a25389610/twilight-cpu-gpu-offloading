# GPU-resident Full Attention：HeadInfer與RetroInfer runtime對照

日期：2026-10-05。使用者要求只補Full runtime comparison，本輪沒有sparsebudget/quality測試或productionoptimization。

## 問題與假設

目前RetroInfer約24–28ms，Twilight Direct/resident約50–56ms，但兩者forward/framework不同。假設部分差距來自denseforward底層runtime，而非selection/cache。現有Full-flat是CPUoffloading，不適合作為GPU-resident runtime anchor。

## 方法與條件

相同RTX5060Ti16GB、Llama-3.2-3B-Instruct、BF16、Batch1、同三個32K frozen timingrequests。原始promptIDs驗證sha，實際prompt31938/32639/32363tokens，兩臂cachecapacity=prompt+33，完整歷史K/V在GPU，沒有pruning、quantization或CPUmiss。prefill以各自nativeFull實作執行，所有prefill準備與move排除formal。

- HeadInfer arm：原環境Transformers模型，保留現有`headinfer.mp`groupedQ/K/Vprojection與逐group RoPE，FullGQA layer-level flashattention；新test-onlyGPUcacheadapter以預分配GPUtensor保存完整KV，prefill來自nativeHF DynamicCache，copy逐元素exact驗證。不是未patch的vanillaHF runtime，也不是CPUFull-flat；不用候選selection、mappedread、Top-pGraph或projectionGraph。
- RetroInfer arm：官方`Full_Flash_Attn`與`flash_attn_cache`，每層完整GPUKV，官方prefill/decode/move，無WaveIndex/WaveBuffer/sparseapproximation，無CUDAGraph。
- 模型不重新訓練，兩套existing核心source無修改。test-only新scripts記錄在manifest；GPUcache形狀[1,capacity,8,128]×28layers×K/V，兩臂容量與storagebytes逐題完全相同。
- 三題各兩輪，runtime順序交錯、第二輪request反向；12formalruns，freshprocess。32fixedtoken1forwards，D1warm-up，D2–D32共31tokens synchronizedCPUwall；無componentprofile、無每步logitsD2H，末尾finite與cachevalidation在timer外。Gate的greedy/CPUcheckpoint不納入formal。

## Correctness與可行性

256token合成prompt＋32greedytokens，HeadInfer GPUFull adapter與nativeHF outputs32/32相同，與RetroInferFull outputs32/32相同。這是tokenIDsgate，不聲稱logits/attentionallclose或32K完整品質一致。

32K每個formalrun成功，prefill與final logitsfinite、全部56KVtensors在GPU、KVunique bytes符合shape公式、最終cachelength=prompt+32。三題actualcache平均3537.880MiB（3.455GiB），兩臂完全相同；因actualprompt不足32768，不強制寫成3584MiB。

第一個gate失敗：測試包裝器切回nativeHF時未restoreHeadInfer分組後的projectionweights/layerindex，產生projectionshapeerror。核對`mp.py`的weightdataswitch後，僅修正testreference復原狀態，重新跑完整gate與正式cohort；失敗raw保留`results/full_runtime_comparison_20261005_v1_failed_gate_restore/`，無formal數據混入。

## 正式TPOT結果

| Request（兩輪平均） | HeadInfer GPUFull ms/token | RetroInfer GPUFull ms/token |
| --- | ---: | ---: |
| 001_niah_multikey_3_i011 | 37.953 | 27.665 |
| 002_vt_i002 | 38.091 | 27.828 |
| 003_qa_1_i011 | 38.083 | 27.790 |
| **六次平均** | **38.042445** | **27.761058** |

HeadInfer六次範圍37.788–38.165ms，RetroInfer27.620–27.870ms。RetroInfer dense runtime平均少10.281ms，latency低27.03%；throughput比值1.370×。六組同request/rep全部較快。這是兩個runtimedense路徑的比較，非selector收益。

## 顯存與環境

| Runtime | FullKVunique MiB | 實際模型parameter storage MiB（patch前） | D32liveallocated MiB |
| --- | ---: | ---: | ---: |
| HeadInfer | 3537.880 | 6127.834 | 9680.358 |
| RetroInfer | 3537.880 | 6879.334 | 10460.642 |

Fullcachebytes相同，但模型/runtimestorage不同，所以不能要求totalVRAM相同。RetroInfer的embed/lm_head載入方式有額外physicalstorage。這裡權重在HeadInferpatch前量，以uniquebacking去重，不拿patch後分組Parameter.numel當完整physicalweightbaseline。liveallocated不是reserved/nvidia-smi或prefillpeak。

兩臂torch2.7+cu128、flash-attn2.8.3.post1相同；HeadInfertransformers4.45.2/Triton3.3，RetroInferoverlaytransformers4.49/Triton3.4。保留各自實際既有runtime環境，沒有強制共用backend或compiler。版本與forward差異都屬本次runtime差距範圍，未單獨消融根因。

## 結論與限制

**實驗觀察**：即使都做FullGPUattention、完全不做selection/offload，RetroInferruntime仍約快10.28ms（1.37×throughput）。因此兩個sparse系統約2×歷史差距不能全部歸因RetroInferselector或cachepolicy。

**不能推論**：不能把Full10.28ms直接從sparse差值扣掉，當作剩餘selection/cachecost；也不能宣稱runtime解釋了幾成sparse差距。不同KVworkload、bufferlayout、projection/attentionkernels、overlap與packageversions仍可能交互影響。Gate只驗256/32，沒有測Full130題品質或context以外generalization。

**研究意義**：目前denseanchor已確認兩套executionframework有顯著速度差異。FullGPU允許使用約3.45GiBKV，與sparse低VRAM路徑資源需求不同；若把denseanchor與歷史sparse數字放一起，只能作背景，非fresh四臂消融。

**下一決策**：完成使用者限定的Fullruntime對照後停止。若後續比較完整系統，可保留這兩個denseanchors；若要歸因selection/placement，需要在commonbackend或單框架內設計控制實驗。沒有啟動retrieval=.05剩餘110題或優化。

## Artifacts與重現

`results/full_runtime_comparison_20261005_v1/`：manifest、environment、gate/synthetic256、formal/rep/request/runtime、command/log/processwall、12results、formal.csv、summary.json。失敗gate在獨立failed目錄。

`scripts/benchmark_full_runtime_case_20261005.py`、`run_full_runtime_comparison_20261005.py`、`analyze_full_runtime_comparison_20261005.py`。專案root執行`python scripts/run_full_runtime_comparison_20261005.py`，核算`python scripts/analyze_full_runtime_comparison_20261005.py`。

獨立analysis驗證sourcehash、兩gateIDs、12success、所有prompt/capacity/storagepair、32steps與31latency重算。publicmirror準備本report與三支scripts；raw/PT/log/模型與binaries未納入，無production source改動，未commit/push。

## 10/05 source audit 修正與加速候選

本次重新核對 `headinfer.mp` 發現：`layer_rope`、`layer_projection` 與 projection Graph 都受 `use_layer_batched_selection` 條件控制。test-only GPUFull cache 只啟用 `layer_full_attention_execution`，沒有 layer-batched selection，因此 wrapper 雖設 `_twilight_layer_rope=True`，實際 Full 測試仍走逐group RoPE。原報告「保留 layerRoPE」描述有誤，已更正；raw 結果及 source 不變。38.042445 ms 是此 grouped-RoPE Full 路徑，不完全代表目前已啟用 layerRoPE 的 sparse 計算路徑。因此 runtime 差異觀察仍成立，但10.281387 ms不能直接當作目前 sparse 路徑可消除的底層成本。

Source 已確認的差異：HeadInfer 每層8個KV groups，各呼叫Q/K/V projection（24次Linear）；RetroInfer `model_hub/llama.py::wqkv` 使用合併權重、一次 `F.linear`。RetroInfer MLP 合併 gate/up projection，並用 FlashInfer `silu_and_mul`，RMSNorm與RoPE也用 FlashInfer融合kernel；HeadInfer既有Transformers4.45.2的RMSNorm由float32 cast/pow/mean/rsqrt/multiply等operations組成，MLP gate/up分開。這些是已確認的程式結構差異；尚未量測各自解釋多少TPOT，不宣稱都是瓶頸。

先前證據：`reports/twilight_layer_projection_rope_2026-09-10.md` 中整層Q/K/V projection未過bit-exact gate，D1/D2/D32 logits與selection hashes改變，不等於已證實quality下降；RoPE-only通過exact gate並已採用。`reports/twilight_low_vram_speed_tradeoff_2026-09-27.md` 中grouped QKV Graph在當時matched組合比no-QKV-Graph快2.103ms/token，代價為238.734MiB D32 live allocated；不能當作目前版本的新增保證收益。

下一個最小驗證：先建立與目前layerRoPE設定一致的Full adapter，確認旗標實際路徑、greedy/logits guardrail及matched TPOT；之後再獨立消融projection或norm/MLP融合，測速度、storage與品質。此turn只做source audit與文件修正，沒有修改production、重跑benchmark或啟動優化。
