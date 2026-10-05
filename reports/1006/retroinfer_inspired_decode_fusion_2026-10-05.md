# 參考 RetroInfer 的 Decode fusion：matched TPOT 與 correctness

日期：2026-10-05（Asia/Taipei）。狀態：formal、operator/model gates、13題品質診斷與獨立memory audits已核對。依使用者指示未跑完整130題。

## 問題、假設與控制

使用者授權參考RetroInfer优化目前程式。此前GPUFull runtime對照有10.28ms差距，但Full測試的layerRoPE旗標未實際生效，不能把此差距當作目前sparse可省成本。本次直接測preferred sparse路徑，假設減少dense計算的dispatch及逐元素kernel能降低TPOT。

RTX5060Ti16GB、Llama-3.2-3B-Instruct BF16、Batch1、相同3個32K frozen prompts；QuestB0=8192、Twilightdynamic p=.90、DirectINT4QK、previous-tokenresident、GPUbitmap/mappedCPUread、lowVRAMflags、layerRoPE、exactTopPGraph與precomputedmapping維持相同。沒有降低p、retrievalbudget、候選數或generation長度。原環境torch2.7+cu128/Transformers4.45.2/Triton3.3，不安裝FlashInfer或切換RetroInfer環境。

## 三個arm

- baseline：目前preferred路徑。
- gateup：保留8組Q/K/V計算；將RMSNorm尾端（保留PyTorchvariance reduction）、MLP SiLU乘法與RoPE逐元素操作用Triton融合，保留BF16中間rounding；合併每layer的MLP gate/up projection。原gate/up Parameters改為一次合併storage的views，避免保留第二份權重。prefill沿用原函式。
- merged_qkv：在gateup上，Decode每layer一次mergedQKV Linear，替代8組×3次Linear；沿用現有layer-projection分支。Q/K/V Parameters重指向mergedweight的views。數值可能改變，標為experimental；未設預設。

参考source：`../../external/retroinfer-20261005/model_hub/llama.py`的`wqkv`、`mlp`、`layernorm`與RoPE實作。這是依其execution設計自行適配，未搬入WaveIndex/WaveBuffer，也不宣稱研究novelty。

## Correctness gates與失敗證據

3 seeds的normtail、SiLU-multiply、RoPE operator probes逐element相同。擴大至65,280個finiteBF16 gate值及randomup時，初版SiLU有500個subnormal輸入差異；PTX顯示`div.rn.ftz.f32`會flushdenormal，改成inline`div.rn.f32`後重跑0差異。失敗probe與舊source保留在`results/decode_pointwise_20261005_v1/`，不是忽略極小數差異。

修正後gateup：3個32Krequests的P4/D1/D2/D32 checkpoints hashes、D33–D35 selectedKV/GQAunion/logits hashes皆與baseline相同。這是已測request/steps的bit-exact gate，非全模型或全部input的形式證明。

mergedQKV pilot（003）：P4exact，D1/D2/D32 logits maxabs差0.125/0.125/0.109375；checkpoint top1相同，3個diagnostic步selected/GQAunion hashes相同，但logits hashes不同。因此不能作等價工程加速宣稱，需RULER驗證。

## 正式TPOT

3題×2輪×3arms，freshprocess，runtime順序輪替、第二輪題序反轉。32fixedtoken1 forwards，D1warm-up，formal為D2–D32共31token同步CPUwall平均。prefill、qualitygreedy、diagnostics、memoryinventory不納入；無componentprofiling或每步logitsD2H。逐run CSV重新核算31latencies，核對promptsha、decodepolicy、backend與command單變因。

| Arm | 平均ms/token | 六run範圍ms | 比baseline快的pair |
|---|---:|---:|---:|
| baseline | 52.788535 | 51.828–55.842 | — |
| gateup | 51.363084 | 51.195–51.552 | 6/6 |
| merged_qkv | 47.944378 | 47.509–48.526 | 6/6 |

gateup平均少1.425451ms（2.70%）；mergedQKV組合平均少4.844157ms（9.18%）。新增mergedQKV相對gateup少3.418706ms。baseline其中一run55.842ms偏高，保留全部結果與range；不只挑最佳run，不與舊55.94ms跨run直接算加速。

## 品質診斷：13 tasks各1題（不是完整130）

原版fresh baseline13題全都重現historical Direct的greedy IDs、checkpoint/final logits hashes；因此historical baseline可作本次已測13題的品質對照。mergedQKV同樣greedy到firstEOS或各題budget；兩版13題平均均63.205385，13/13逐題分數相同，0改善／0變差。12/13生成tokenIDs相同。這個63.205分是13題subset，不是既有130題76.179615的重測，也不能從兩者差值推論品質下降。

| Task | 原版 | mergedQKV |
|---|---:|---:|
| cwe | 0.00 | 0.00 |
| fwe | 66.67 | 66.67 |
| niah_multikey_1 | 100.00 | 100.00 |
| niah_multikey_2 | 100.00 | 100.00 |
| niah_multikey_3 | 0.00 | 0.00 |
| niah_multiquery | 75.00 | 75.00 |
| niah_multivalue | 100.00 | 100.00 |
| niah_single_1 | 100.00 | 100.00 |
| niah_single_2 | 100.00 | 100.00 |
| niah_single_3 | 100.00 | 100.00 |
| qa_1 | 0.00 | 0.00 |
| qa_2 | 0.00 | 0.00 |
| vt | 80.00 | 80.00 |

唯一IDs改變：`111_qa_1_i001`，20→19tokens，兩個回答措辭不同但都0分。logits差異是真實存在，不能稱bit-exact或「所有輸出不變」。13題各task只有1題，且部分baseline0分；這只是preliminary品質診斷，不能證實完整130題／所有request品質不變。使用者明確要求不用馬上做130題，本輪沒有啟動其餘117題。

## 獨立VRAM audits

3requests×3arms，memory flags不進formaltiming。數字為三題平均；cache相關unique GPU storage包含residentKV、Questmetadata、TwilightINT4副本、mapping與attentionbuffers，不含模型權重。liveallocated是PyTorch CUDA allocated，不是reserved／nvidia-smi／整個process peak。

| Arm | Cache相關unique MiB | D32 liveallocated MiB | Model unique parameter/buffer MiB | Decode peak allocated MiB |
|---|---:|---:|---:|---:|
| baseline | 1324.816757 | 7474.695312 | 6127.841064 | 7500.351888 |
| gateup | 1324.816757 | 7473.852376 | 6127.841064 | 7499.508952 |
| merged_qkv | 1324.790390 | 7472.230957 | 6127.841064 | 7498.131999 |

三arm的modelunique backing bytes相同：weightpacking使用共享storageviews，沒有保留第二份完整權重。gateup的cacheunique逐題完全相同；mergedQKV的cache大小有微小差異，符合BF16變化可能影響selection/count的限制，不能把它稱為所有KV選擇逐步exact。沒有增加已測D32steadyVRAM；不把此結果推廣為startup或任意prefillpeak不變。

## 結論、限制與下一決策

**已確認**：相同p=.90／preferredsparse設定下，融合MLP/pointwise平均快2.70%；再合併QKV整體快9.18%，六pair均較快。cache相關VRAM約1324.8MiB（1.294GiB），沒有因第二份權重增加steadyallocation。13題品質診斷分數持平。

**尚未確認**：mergedQKV不是bit-exact，沒有完整130題品質證明，也沒有其他model/context/batch驗證。baseline其中一run偏高，所有run均保留；結果不等於RetroInfer dense runtime差距全部消除，也不是common-quality完整系統優勝或novelty。

**採用方式**：兩個版本保留opt-in，不改原版預設。較保守版本為gateup（已測model/selection exact）；較快版本為mergedQKV（13題quality preliminary）。本輪依使用者指示停止130題；後續若決定作正式baseline，才補完整quality。

**wrapper整理**：原runner會以SystemExit結束，wrapper末尾額外mode marker寫入不可達；已刪除這兩個無法執行的寫入，模型推論已執行部分byte-identical。raw JSON以savedcommand與既有fusion/layerprojection fields核對arm。`wrapper_cleanup.json`記錄前後hash；analysis核對原source_snapshot並確認current差異只限這段不可達尾端，沒有重寫raw數據。

## Artifacts與重現

`results/decode_pointwise_20261005_v1/`：初版operator/gates/正式12runs與FTZ失敗證據，source_snapshot保留初版。
`results/decode_gateup_20261005_v1/`：修正operatorprobe與3題×2arms gate。
`results/decode_retro_dense_20261005_v1/`：manifest/source_snapshot、pilot、正式18runs、formal_summary，qualitypilot_summary、memory_summary與逐request結果保存於同root。

Source新增`source/headinfer/headinfer/decode_pointwise.py`；原case runner新增opt-in `--decode-pointwise-fusion norm|mlp|rope|all`，defaultNone。wrapper與run/analyze scripts位於`scripts/*decode*20261005.py`。沒有改預設方法或舊數據，未commit/push。

### 重現與本次公開檔案

原版：沿用相同preferredcase command。gateup：case script改為`benchmark_decode_gateup_case_20261005.py`並加`--decode-pointwise-fusion all`。mergedQKV：script改為`benchmark_decode_retro_dense_case_20261005.py`並加`--decode-pointwise-fusion all --twilight-layer-projection`；其餘flags與prompt固定。不可同時啟用QKVprojectionGraph。

兩個wrappers均為Llama BF16／Batch1 Decode opt-in；本輪只驗3B。新的fresh實驗可用`run_decode_retro_dense_20261005.py --output-root results/<new-run> formal`，避免修改frozen v1；不要用新source覆寫已保存的實驗。既有v1驗證用`analyze_decode_retro_dense_20261005.py formal|memory|qualitypilot`。

Publicmirror：`reports/1007/retroinfer_inspired_decode_fusion_2026-10-05.md`；`scripts/run_ruler_partial_h2d_tpot_case_v1.py`、`run_decode_pointwise_20261005.py`、`run_decode_gateup_20261005.py`、`run_decode_retro_dense_20261005.py`、`analyze_decode_retro_dense_20261005.py`、`probe_decode_pointwise_20261005.py`、`benchmark_decode_gateup_case_20261005.py`、`benchmark_decode_retro_dense_case_20261005.py`；`source/headinfer/headinfer/decode_pointwise.py`。未納入raw JSON/CSV/log/PT、模型、checkpoints、CUDAcache或binaries；未commit/push。

RetroInfer reference repository：https://github.com/microsoft/RetrievalAttention ，checkout SHA `03f912c6e917c380d9d90c5ec85bb0f161ba53ef`。本次没有修改其官方source或套件環境。
