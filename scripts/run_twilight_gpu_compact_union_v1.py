#!/usr/bin/env python3
"""Run matched current-baseline versus GPU compact-GQA-union experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/paul/miniconda3/envs/headinfer_repro/bin/python")
CASE_RUNNER = ROOT / "scripts/run_ruler_partial_h2d_tpot_case_v1.py"
ALL_REQUESTS = (
    "001_niah_multikey_3_i011",
    "002_vt_i002",
    "003_qa_1_i011",
)
SOURCE_FILES = (
    ROOT / "source/headinfer/headinfer/quest_offload_cache.py",
    ROOT / "source/headinfer/headinfer/twilight_offload_cache.py",
    ROOT / "source/headinfer/headinfer/cpu_token_union.py",
    CASE_RUNNER,
    Path(__file__).resolve(),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(request: str, mode: str, output: Path) -> list[str]:
    result = [
        str(PYTHON),
        str(CASE_RUNNER),
        "--request-dir",
        str(
            ROOT
            / "results/context_p_ruler_39_v1/cohort/32768/timing_requests"
            / request
        ),
        "--short-head-count",
        "0",
        "--decode-steps",
        "32",
        "--profile-breakdown",
        "--profile-breakdown-steps",
        "1",
        "--twilight-top-p",
        "0.90",
        "--twilight-budget-mode",
        "dynamic",
        "--quest-layer-batched-selection",
        "--twilight-gqa-group",
        "--twilight-qk-backend",
        "triton_prepare",
        "--full-flat-h2d-reference-json",
        str(
            ROOT
            / "results/context_p_ruler_39_v1/timing/full_flat_denominator/32768"
            / f"{request}.json"
        ),
        "--local-files-only",
        "--twilight-cpu-flat-gather",
        "--twilight-cpu-bitmap-union",
        "--twilight-direct-attention-layout",
        "--twilight-layer-rope",
        "--twilight-early-gpu-metadata",
        "--twilight-gather-h2d-chunks",
        "4",
        "--twilight-reuse-quant-metadata",
        "--twilight-fused-quest-score",
        "--twilight-skip-unused-host-views",
        "--output",
        str(output),
    ]
    if mode == "gpu_union":
        result.append("--twilight-gpu-compact-gqa-union")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requests", nargs="+", choices=ALL_REQUESTS, default=list(ALL_REQUESTS)
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/twilight_gpu_compact_union_v1/formal",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {
        str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_FILES
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"source_sha256": fingerprints, "runs": []}
    )
    if manifest["source_sha256"] != fingerprints:
        raise RuntimeError("source snapshot changed after experiment started")

    for request_index, request in enumerate(args.requests):
        for repetition in range(1, args.repetitions + 1):
            normal_order = ("control", "gpu_union")
            modes = (
                normal_order
                if (request_index + repetition) % 2
                else tuple(reversed(normal_order))
            )
            for mode in modes:
                run_dir = args.output_dir / request / f"rep{repetition}" / mode
                run_dir.mkdir(parents=True, exist_ok=True)
                output = run_dir / "result.json"
                cmd = command(request, mode, output)
                (run_dir / "command.json").write_text(
                    json.dumps(cmd, indent=2) + "\n"
                )
                if output.exists() and json.loads(output.read_text()).get("status") == "ok":
                    continue
                started = time.time()
                with (run_dir / "run.log").open("a") as log:
                    completed = subprocess.run(
                        cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                    )
                manifest["runs"].append(
                    {
                        "request": request,
                        "repetition": repetition,
                        "mode": mode,
                        "returncode": completed.returncode,
                        "elapsed_seconds": time.time() - started,
                        "command": cmd,
                    }
                )
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                if completed.returncode:
                    raise RuntimeError(
                        f"failed: {request} repetition {repetition} {mode}"
                    )
                print(f"PASS {request} rep{repetition} {mode}", flush=True)

    current = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_FILES}
    if current != fingerprints:
        raise RuntimeError("source snapshot changed during experiment")
    print("ALL FORMAL RUNS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
