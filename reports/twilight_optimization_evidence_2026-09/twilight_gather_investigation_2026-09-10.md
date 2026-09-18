# CPU gather與重複KV整理調查

## 範圍

最新bitmap union + flat gather + triton_prepare，32K qa_1 i011，p=.90，32 decode及5 diagnostics。
僅新增process-local diagnostic wrappers與離線replay，未修改正式cache／模型實作。
在diagnostic CPU BF16 index_select前後量perf_counter/process_time；GPU final cat用CUDA Events。
capture第一個diagnostic token的56組source views、indices clones與out views，模型完成後重播。
CPU完整slab仍保留，非先前隨機hot-cache microbenchmark。Threads sweep固定1/2/4/6，
每項1 warmup+3 repeats，非模型TPOT。clone與wrapper會擾動父timer，不能視為零overhead。

## 結果

- 未啟用wrappers的D2–D32 TPOT（同process前段）：215.04ms。
- diagnostic gather bundle mean44.30ms；其中56個index_select wall總和39.99ms（90.27%）。
  差額4.31ms包括indices offset construction、views、Python loop及本次instrumentation，
  不是無instrumentation下可直接消掉的overhead。
- index_select的process CPU總時間230.04ms/token，是process多threads CPU時間總和，不能
  與wall相加，也不能單靠它判定DRAM bandwidth是否飽和或是否busy-spin。
- 平均payload528.71MB/token；payload/wall13.22GB/s，名義source read+out write26.44GB/s。
  後者不是DRAM實測traffic，未計cache line、write allocate與cache命中，也沒有量平台頻寬上限。
- 第一step實際indices重播mean：1 thread65.13ms、2 threads44.25ms、4 threads38.06ms、
  6 threads32.55ms。6-thread重播比實際39.99ms短，可能含cache、CPU/GPU交錯與排程差異，
  不能把7.44ms差額全稱為可移除thread overhead。未鎖clocks、固定sweep順序，單request限制。
- GPU history+new token cat：56 calls/token、CUDA interval mean4.75ms，output528.83MB/token。
  名義read+write約1.058GB；存在Python launch與Event排程差異，4.75ms不能直接当TPOT收益。
- 5個diagnostic indices/logits hashes與既有coarse版本完全相同。
- 首次replay遇到inference tensor寫入限制；已在同inference_mode重播成功。失敗log保留為
  `results/twilight_gather_replay_inference_error.log`，不是模型correctness失敗。

## Source流程與複製邊界

1. CPU full BF16 slab → CPU pinned flat buffer：index_select的out直接完成gather+pack，
   沒有「取出完整temporary後再CPU copy」的第二份完整payload。
2. CPU pinned flat buffer → GPU history buffer：K/V各1次H2D/layer。
3. GPU history buffer + GPU new-token KV → GPU attention buffer：K/V各1次cat/layer，
   為每個KV group插入一個new-token row，的確再次複製整份history。
4. cu_seqlens建立目前經CPU tensor/cumsum/tolist，再建GPU int32 tensor；mp.py還建立
   group_queries與cu_seqlens_q。這些是小metadata/dispatch候選，尚無獨立成本量測。

## 結論與優化候選

已確認大多wall在gather operator內，不是外部Python呼叫；不能進一步斷言90%皆純RAM搬運。
降低threads並無優勢；先前flat gather呼叫合併在真實模型只有小幅收益也支持不再只減call count。
下一個最小CPU實驗：同一實際indices、全slab、K/V雙plane與多layer working set，對照專用
fused K/V row-copy helper與現有index_select；逐項測chunk/thread scheduling，驗證bit-exact。
需要負結果與continuous-copy對照，若已接近此機器實測copy極限就停止追求巨大降幅。

GPU候選：CPU packing時為每group預留new-token slot，直接H2D到attention-layout buffer，
GPU只寫入new token，避免全history cat。這是不改selection/attention集合的layout改動，
但須保持slot覆寫、buffer lifetime、H2D完成依賴；不得讓未初始化slot進入Attention。
若連gap一起H2D，新增bytes為28 layers*8 groups*2 K/V*128 dim*2 bytes=114688/token，
約本輪payload的0.022%；必須揭露，不能聲稱H2D bytes完全不變。若堅持原bytes則需別的copy布局，
可能增加transfer calls。此方案尚未實作，4.75ms device interval不是guaranteed TPOT加速。

優先順序：CPU gather專用copy最小ablation（收益待測）；GPU避免整份cat（約幾ms級候選）；
最後再做metadata預配置。不要以降低CPU完整KV容量或恢復舊per-head eviction作為此問題的解法。

Artifacts：`results/twilight_gather_investigation_v1/{case,analysis}.json`，分析包含逐call shape/bytes/
wall/CPU時間、replay repeats與command；`scripts/diagnose_twilight_gather_v1.py`。
