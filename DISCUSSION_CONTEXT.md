# Discussion context

This public repository is a curated discussion subset of the local
`headinfer_reproduction` workspace. It includes the Twilight CPU-GPU
offloading implementation, selected profiling and test runners, key
configurations, and selected evidence reports. It intentionally excludes model
caches, raw experiment outputs, logs, and historical files that are not needed
for the current discussion.

## What this project is

This work examines CPU-GPU KV-cache offloading for long-context, batch-1 LLM
decoding. The present implementation is a `Twilight-inspired` PyTorch
prototype in HeadInfer; it is **not** an implementation of an official Twilight
kernel. The research focus is whether a query-aware selection policy can reduce
the physical KV transfer and attention cost without letting selection,
synchronization, or CPU gathering erase the system-level benefit.

## Current implementation focus

- `source/headinfer/headinfer/twilight_offload_cache.py`: cache lifecycle and
  CPU/GPU offloading policy.
- `source/headinfer/headinfer/twilight_fused_quest.py`: fused per-layer
  selection and gather path.
- `source/headinfer/headinfer/twilight_fused_qk.py` and
  `twilight_fused_indices.py`: GPU scoring and selected-index handling.
- `scripts/profile_twilight_three_requests_20260915.py`: three-request
  profiling harness.
- `reports/twilight_three_request_complete_2026-09-15.md`,
  `reports/twilight_async_wall_2026-09-10.md`, and
  `reports/twilight_gpu_wall_2026-09-11.md`: uploaded evidence for current
  measurements and timing-scope interpretation.

## Important interpretation guardrails

- The request-level head configuration is frozen; do not describe the
  prototype as per-decode dynamic Quest Top-K.
- Conditional Greedy at `@112` is an offline, label-leaked oracle, not
  `Static112` and not a deployable baseline.
- End-to-end TPOT must remain separate from diagnostic CPU-wall scopes and CUDA
  Event timing. A blocking `.cpu()` can include queued GPU work and index
  transfer; it is not automatically pure D2H time.
- Correctness and efficiency are separate: asynchronous D2H source storage
  needs proper stream lifetime handling before a speed claim is trusted.

## Deliberately excluded materials

The local workspace contains multi-gigabyte Hugging Face caches and raw
experiment results. They are omitted so this repository remains suitable for
code and design discussion; the repository is not a complete reproducibility
artifact.

## Useful questions for an online reviewer

1. Trace the source-to-source flow from GPU Q/K scoring through selected-index
   synchronization, CPU KV gathering, and attention reconstruction.
2. Identify CPU-GPU synchronization points that could dominate batch-1 decode
   latency.
3. Propose one minimal, controlled experiment to isolate the cost of scoring,
   Top-K/sort, final-index GPU-to-CPU transfer, and CPU gather.
4. Review whether the cache lifetime and `record_stream()` usage is safe under
   asynchronous D2H.
