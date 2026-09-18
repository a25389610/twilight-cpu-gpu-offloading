# Twilight CPU-GPU Offloading Discussion Context

This repository is a curated, private discussion mirror of the local
`headinfer_reproduction` workspace. It contains implementation code, runners,
configuration, and research reports, but intentionally excludes model caches,
raw experiment outputs, and logs.

## Current scope

- Model and setting: Llama-3.2-3B-Instruct, Batch 1, RTX 5060 Ti 16 GB, CPU-resident historical KV.
- Mechanism: selected historical KV is gathered on the CPU and transferred to the GPU for Decode attention.
- Implementation: `source/headinfer/headinfer/twilight_offload_cache.py` and its fused selection helpers.
- Current question: profile and reduce the cost of Twilight-style selection and GPU-to-CPU final-index synchronization before proposing adaptive-K policy changes.

## Important terminology

`Twilight-inspired` means this local PyTorch CPU-offload prototype reproduces a selection structure for research. It is not the official Twilight efficiency implementation and does not use its custom low-bit sparse GEMV execution path.

End-to-end TPOT must be discussed separately from diagnostic CPU-wall scopes and CUDA Event timing. The current canonical weekly report is `reports/weekly/2026-09-16.md`.

## Suggested questions for ChatGPT

- "Trace the Decode request path from `twilight_offload_cache.py` through CPU gather, index synchronization, H2D, and attention."
- "Compare normal TPOT and the CPU-wall/CUDA-Event diagnostic scopes; identify which values must not be added."
- "Review a proposed change for cache correctness, stream lifetime, and whether it changes the frozen selection policy."

## Deliberately excluded data

`results/`, `logs/`, `.hf_cache/`, virtual environments, Python bytecode, and credential-like local files are ignored. Reports retain the evidence summary and artifact paths; raw artifacts stay local.
