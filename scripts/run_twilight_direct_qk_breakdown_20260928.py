#!/usr/bin/env python3
"""Reproduce the saved 52.104 ms/token direct-QK commands for diagnostics."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "results/twilight_official_gap_20260927_v1/direct_qk_matched/rep1"
OUT = ROOT / "results/twilight_direct_qk_breakdown_20260928_v1"
REQUESTS = ("001_niah_multikey_3_i011", "002_vt_i002", "003_qa_1_i011")
SOURCE = (
    "scripts/run_ruler_partial_h2d_tpot_case_v1.py",
    "source/headinfer/headinfer/twilight_offload_cache.py",
    "source/headinfer/headinfer/twilight_fused_qk.py",
    "source/headinfer/headinfer/resident_gpu_mapping.py",
    "source/headinfer/headinfer/resident_zero_copy.py",
)
STAGES = ("formal", "normal", "selection", "vram", "nohit")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_for(stage, request, arm, rep):
    src = SEED / request / "direct/command.json"
    cmd = json.loads(src.read_text())
    assert cmd[cmd.index("--twilight-qk-backend") + 1] == "triton"
    cmd[cmd.index("--twilight-qk-backend") + 1] = (
        "triton_prepare" if arm == "exact" else "triton")
    dest = OUT / stage / f"rep{rep}" / request / arm
    dest.mkdir(parents=True, exist_ok=True)
    cmd[cmd.index("--output") + 1] = str(dest / "result.json")
    if stage == "normal":
        cmd += ["--profile-normal-overlap", "--profile-breakdown-steps", "3",
                "--profile-normal-overlap-trace", str(dest / "timeline.trace.json")]
    elif stage == "selection":
        cmd += ["--profile-breakdown", "--profile-breakdown-steps", "3",
                "--twilight-detailed-selection-profile"]
    elif stage == "vram":
        cmd += ["--vram-inventory-output", str(dest / "inventory.json"),
                "--vram-selection-workspace-probe"]
    elif stage == "nohit" and arm == "nohit":
        cmd[1] = str(ROOT / "scripts/run_twilight_nohit_control_v1.py")
    return cmd, dest


def plan(stage):
    if stage == "formal":
        for rep in (1, 2):
            for request in (REQUESTS if rep == 1 else tuple(reversed(REQUESTS))):
                for arm in (("exact", "direct") if rep == 1 else ("direct", "exact")):
                    yield request, arm, rep
    elif stage == "nohit":
        for request in REQUESTS:
            for arm in ("direct", "nohit"):
                yield request, arm, 1
    else:
        for request in REQUESTS:
            for arm in (("exact", "direct") if stage == "vram" else ("direct",)):
                yield request, arm, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=STAGES)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {name: digest(ROOT / name) for name in SOURCE}
    seed_hashes = {r: digest(SEED / r / "direct/command.json") for r in REQUESTS}
    manifest = {"source_sha256": hashes, "seed_command_sha256": seed_hashes,
                "seed_root": str(SEED), "formal_scope": "D2-D32, 31 fixed tokens"}
    mpath = OUT / "manifest.json"
    if mpath.exists():
        prior = json.loads(mpath.read_text())
        if prior != manifest:
            raise AssertionError("source or seed command changed since first stage")
    else:
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    for stage in args.stages:
        for request, arm, rep in plan(stage):
            assert {name: digest(ROOT / name) for name in SOURCE} == hashes
            cmd, dest = command_for(stage, request, arm, rep)
            (dest / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            result = dest / "result.json"
            if result.exists() and json.loads(result.read_text()).get("status") == "ok":
                print("SKIP", stage, request, arm, rep, flush=True)
                continue
            started = time.time()
            print("START", stage, request, arm, rep, flush=True)
            with (dest / "process.log").open("w") as handle:
                p = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            print("DONE", stage, request, arm, rep, p.returncode,
                  round(time.time() - started, 1), flush=True)
            if p.returncode:
                raise RuntimeError(f"failed {stage} {request} {arm}: {dest}")


if __name__ == "__main__":
    main()
