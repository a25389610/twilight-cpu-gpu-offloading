# HeadInfer reproduction

The current canonical research direction is documented in
`CURRENT_RESEARCH_DIRECTION.md`. As of 2026-08-29, the primary direction is a
training-free, Quest-based, extremely lightweight adaptive KV-block selection
study for Batch-1 resource-constrained CPU-GPU offloading. Older Predictor and
Fluxion reports remain historical evidence, not the active default hypothesis.

Weekly Wednesday-to-Tuesday research summaries are stored in
`reports/weekly/`. The active report is
`reports/weekly/2026-08-26.md`.

Reproduce after the NVIDIA driver is operational:

```bash
conda activate headinfer_repro
pip install -e source/headinfer
pip install flash-attn --no-build-isolation
python scripts/smoke_test.py
python scripts/compare_correctness.py --baseline results/smoke/baseline.json --headinfer results/smoke/headinfer.json --output results/smoke/correctness.json
python scripts/benchmark_context_sweep.py
```

Results are intentionally separated by process, because `mp_headinfer()` patches a Transformers attention class globally and changes projection-weight views in place. `benchmark_context_sweep.py` records a failed baseline and continues with HeadInfer, allowing a real-prefill OOM boundary to be established.

The experimental fixed-capacity pinned-slab cache and its 32K comparison can
be checked with:

```bash
python scripts/test_pinned_slab.py
python scripts/benchmark_pinned_slab_32k.py
python scripts/benchmark_pinned_slab_context_sweep.py
```

The benchmark launches baseline, original HeadInfer, and slab modes in
separate processes. It compares post-prefill, D1, D2, and D32 logits and
records host allocator, RSS, GPU memory, logical cache length, exposed active
range, and greedy-token results. The completed CUDA results and stream-race
diagnostic are documented in `reports/pinned_slab_experiment_status.md` and
`reports/pinned_slab_32k_report.md`. The fixed-input 8K/16K/32K/64K RAM,
overall decode throughput, and per-token latency comparison is in
`reports/pinned_slab_power2_context_sweep_report.md`.

The direct CUDA event and allocator-lifetime repair for the original
`OffloadedCache`, including deterministic red-green tests and a real 32K
correctness rerun, is documented in
`reports/offloaded_cache_stream_fix_report.md`.

The 128K pinned-slab test, including the one-shot activation OOM and successful
four-by-32K chunked prefill plus D1 result, is documented in
`reports/pinned_slab_128k_report.md`.
