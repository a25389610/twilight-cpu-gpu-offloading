#!/usr/bin/env python3
"""Paired 32K RULER-130 greedy quality for exact-prepare vs direct QK.

Each request uses its frozen generation budget as an upper bound and stops
at EOS. These process wall times are never formal TPOT evidence.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "results/context_p_twilight_gqa_group_ruler130_v1/cohort/32768"
SEED_COMMAND = ROOT / (
    "results/twilight_official_gap_20260927_v1/direct_qk_matched/"
    "rep1/003_qa_1_i011/baseline/command.json"
)
OUTPUT = ROOT / "results/twilight_direct_qk_ruler130_quality_20260927_v2"
SOURCE_FILES = [
    ROOT / "scripts/run_ruler_partial_h2d_tpot_case_v1.py",
    ROOT / "source/headinfer/headinfer/twilight_offload_cache.py",
    ROOT / "source/headinfer/headinfer/twilight_fused_qk.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_for(request: Path, budget: int, arm: str, result: Path) -> list[str]:
    cmd = json.loads(SEED_COMMAND.read_text())
    cmd[cmd.index("--request-dir") + 1] = str(request)
    cmd[cmd.index("--decode-steps") + 1] = str(budget)
    cmd[cmd.index("--output") + 1] = str(result)
    ref_index = cmd.index("--full-flat-h2d-reference-json")
    del cmd[ref_index : ref_index + 2]
    cmd += ["--greedy-probe", "--greedy-probe-stop-at-eos"]
    cmd[cmd.index("--twilight-qk-backend") + 1] = (
        "triton_prepare" if arm == "baseline" else "triton"
    )
    return cmd


def complete(result: Path, request: dict, arm: str) -> bool:
    if not result.exists():
        return False
    try:
        payload = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "ok"
        and payload.get("request_id") == request["request_id"]
        and payload.get("prompt_sha256") == request["prompt_sha256"]
        and payload.get("decode_steps") == request["generation_budget"]
        and payload.get("decode_policy") == "greedy_probe"
        and payload.get("twilight_qk_backend")
        == ("triton_prepare" if arm == "baseline" else "triton")
        and payload.get("greedy_probe_stop_at_eos") is True
        and 2 <= len(payload.get("greedy_token_ids") or [])
        <= request["generation_budget"]
        and (
            128009 in (payload.get("greedy_token_ids") or [])
            or len(payload.get("greedy_token_ids") or [])
            == request["generation_budget"]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=130)
    args = parser.parse_args()
    requests = []
    for directory in sorted((COHORT / "requests").iterdir()):
        if directory.is_dir():
            metadata = json.loads((directory / "request.json").read_text())
            requests.append((directory, metadata))
    tasks = collections.Counter(item[1]["task"] for item in requests)
    if len(requests) != 130 or len(tasks) != 13 or any(n != 10 for n in tasks.values()):
        raise AssertionError(f"unexpected frozen cohort: {len(requests)} {tasks}")
    if len({item[1]["request_id"] for item in requests}) != 130:
        raise AssertionError("duplicate request_id")
    expected_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_FILES}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / "manifest.json"
    metadata = {
        "classification": "quality-only paired 32K RULER 13x10",
        "cohort_manifest": str(COHORT / "manifest.json"),
        "cohort_manifest_sha256": sha256(COHORT / "manifest.json"),
        "seed_command": str(SEED_COMMAND),
        "seed_command_sha256": sha256(SEED_COMMAND),
        "source_sha256": expected_hashes,
        "request_count": 130,
        "generation_policy": "greedy to first EOS or request budget",
        "formal_tpot": False,
    }
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise AssertionError("resume manifest differs from frozen inputs/source")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    selected = requests[: args.limit]
    started = time.time()
    for index, (request_dir, request) in enumerate(selected, start=1):
        arms = ("baseline", "direct") if index % 2 else ("direct", "baseline")
        for arm in arms:
            destination = OUTPUT / "requests" / request_dir.name / arm
            destination.mkdir(parents=True, exist_ok=True)
            result = destination / "result.json"
            cmd = command_for(request_dir, request["generation_budget"], arm, result)
            (destination / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            if complete(result, request, arm):
                print("SKIP", index, request_dir.name, arm, flush=True)
                continue
            current_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_FILES}
            if current_hashes != expected_hashes:
                raise AssertionError("source changed during quality study")
            print("START", index, request_dir.name, arm, flush=True)
            tick = time.time()
            with (destination / "process.log").open("w") as log:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(
                "DONE", index, request_dir.name, arm, proc.returncode,
                round(time.time() - tick, 1), "elapsed", round(time.time() - started, 1),
                flush=True,
            )
            if proc.returncode or not complete(result, request, arm):
                raise RuntimeError(f"failed or invalid result: {result}")
    print("COMPLETE", len(selected), "requests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
