#!/usr/bin/env python3
"""Current-source matched formal TPOT confirmation for preferred direct QK."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / (
    "results/twilight_official_gap_20260927_v1/direct_qk_matched/"
    "rep1"
)
OUTPUT = ROOT / "results/twilight_direct_qk_current_confirm_20260928_v1"
REQUESTS = (
    "001_niah_multikey_3_i011",
    "002_vt_i002",
    "003_qa_1_i011",
)
SOURCE_FILES = (
    ROOT / "scripts/run_ruler_partial_h2d_tpot_case_v1.py",
    ROOT / "source/headinfer/headinfer/twilight_offload_cache.py",
    ROOT / "source/headinfer/headinfer/twilight_fused_qk.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_FILES}
    manifest = {
        "classification": "three-request two-repetition matched formal fixed-token TPOT",
        "source_sha256": hashes,
        "seed_command_sha256": {
            request: sha256(SOURCE_DIR / request / "baseline/command.json")
            for request in REQUESTS
        },
        "primary_tpot": "D2-D32 synchronized wall, D1 warm-up",
        "quality_probe_flags_enabled": False,
    }
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise AssertionError("manifest changed during resume")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    for rep in (1, 2):
        for request in REQUESTS:
            for arm in (("baseline", "direct") if rep == 1 else ("direct", "baseline")):
                destination = OUTPUT / f"rep{rep}" / request / arm
                destination.mkdir(parents=True, exist_ok=True)
                result = destination / "result.json"
                cmd = json.loads(
                    (SOURCE_DIR / request / "baseline/command.json").read_text()
                )
                cmd[cmd.index("--output") + 1] = str(result)
                cmd[cmd.index("--twilight-qk-backend") + 1] = (
                    "triton_prepare" if arm == "baseline" else "triton"
                )
                if "--greedy-probe" in cmd or "--greedy-probe-stop-at-eos" in cmd:
                    raise AssertionError("formal command contains quality probe")
                (destination / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
                if result.exists():
                    saved = json.loads(result.read_text())
                    if (
                        saved.get("status") == "ok"
                        and saved.get("D2_D128_tpot", {}).get("tokens") == 31
                        and saved.get("twilight_qk_backend")
                        == ("triton_prepare" if arm == "baseline" else "triton")
                    ):
                        print("SKIP", rep, request, arm, flush=True)
                        continue
                if {str(file.relative_to(ROOT)): sha256(file) for file in SOURCE_FILES} != hashes:
                    raise AssertionError("source changed during matched test")
                print("START", rep, request, arm, flush=True)
                started = time.time()
                with (destination / "process.log").open("w") as log:
                    proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                print("DONE", rep, request, arm, proc.returncode, round(time.time() - started, 1), flush=True)
                if proc.returncode:
                    raise RuntimeError(f"formal run failed: {result}")
    print("COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
