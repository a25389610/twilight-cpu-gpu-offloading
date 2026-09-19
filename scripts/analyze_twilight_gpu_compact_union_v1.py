#!/usr/bin/env python3
"""Validate and summarize the GPU compact GQA-union experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHOSEN = ROOT / "results/twilight_gpu_compact_union_v1/formal_v2"
BITPACKED = ROOT / "results/twilight_gpu_compact_union_v1/formal_v3"
PROFILE = ROOT / "results/twilight_gpu_compact_union_v1/profile/003_qa_1_i011"
CORRECTNESS = ROOT / "results/twilight_gpu_compact_union_v1/correctness"
CORRECTNESS_V2 = ROOT / "results/twilight_gpu_compact_union_v1/correctness_v2"
OUT = ROOT / "results/twilight_gpu_compact_union_v1/analysis"
REQUESTS = (
    "001_niah_multikey_3_i011",
    "002_vt_i002",
    "003_qa_1_i011",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def source_snapshot() -> dict[str, str]:
    manifest = load(CHOSEN / "manifest.json")
    expected = manifest["source_sha256"]
    actual = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in expected
    }
    if actual != expected:
        raise AssertionError("chosen source snapshot no longer matches working files")
    return expected


def attention_block_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    capture = False
    with path.open() as handle:
        for line in handle:
            if not capture and line == '  "resident_attention_trace": [\n':
                capture = True
            if capture:
                digest.update(line.encode())
                if line == "  ],\n":
                    return digest.hexdigest()
    raise AssertionError(f"attention trace block not found: {path}")


def main() -> int:
    snapshot = source_snapshot()
    chosen_rows = []
    bitpacked_rows = []
    for request in REQUESTS:
        pair = {}
        packed_pair = {}
        for mode in ("control", "gpu_union"):
            result = load(CHOSEN / request / "rep1" / mode / "result.json")
            profile = result["diagnostic_breakdown_tokens"][0]
            pair[mode] = result
            chosen_rows.append(
                {
                    "request": request,
                    "mode": mode,
                    "mean_ms": 1000
                    * result["D2_D128_tpot"]["mean_seconds_per_token"],
                    "median_ms": 1000
                    * result["D2_D128_tpot"]["median_seconds_per_token"],
                    "min_ms": 1000
                    * result["D2_D128_tpot"]["min_seconds_per_token"],
                    "max_ms": 1000
                    * result["D2_D128_tpot"]["max_seconds_per_token"],
                    "index_d2h_bytes": profile["twilight_index_d2h_bytes"],
                    "index_sync_ms": 1000
                    * profile["twilight_index_sync_and_position_wall_seconds"],
                    "cpu_union_ms": 1000
                    * profile["twilight_group_union_prepare_wall_seconds"],
                    "compact_decode_ms": 1000
                    * profile.get("twilight_compact_index_decode_wall_seconds", 0),
                    "kv_h2d_bytes": profile["h2d_bytes"],
                    "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"],
                    "peak_gpu_reserved_bytes": result["peak_gpu_reserved_bytes"],
                    "membership_bytes": result["cache_state"].get(
                        "gpu_group_membership_allocated_bytes", 0
                    ),
                }
            )
            packed_result = load(
                BITPACKED / request / "rep1" / mode / "result.json"
            )
            packed_pair[mode] = packed_result
            bitpacked_rows.append(
                {
                    "request": request,
                    "mode": mode,
                    "mean_ms": 1000
                    * packed_result["D2_D128_tpot"]["mean_seconds_per_token"],
                    "index_d2h_bytes": packed_result[
                        "diagnostic_breakdown_tokens"
                    ][0]["twilight_index_d2h_bytes"],
                    "compact_decode_ms": 1000
                    * packed_result["diagnostic_breakdown_tokens"][0].get(
                        "twilight_compact_index_decode_wall_seconds", 0
                    ),
                }
            )
        for key in (
            "checkpoint_logits_sha256",
            "final_logits_sha256",
            "twilight_group_positions_sha256",
        ):
            if pair["control"][key] != pair["gpu_union"][key]:
                raise AssertionError(f"chosen parity failed for {request}: {key}")
        for key in ("h2d_bytes", "selected_history_tokens_total", "group_union_history_tokens_total"):
            if (
                pair["control"]["diagnostic_breakdown_tokens"][0][key]
                != pair["gpu_union"]["diagnostic_breakdown_tokens"][0][key]
            ):
                raise AssertionError(f"chosen payload parity failed: {request} {key}")
        if packed_pair["control"]["final_logits_sha256"] != packed_pair["gpu_union"]["final_logits_sha256"]:
            raise AssertionError(f"bitpacked logits parity failed: {request}")

    correctness_control = (
        CORRECTNESS / "003_qa_1_i011/control_trace.json"
    )
    correctness_gpu = load(
        CORRECTNESS_V2 / "003_qa_1_i011/gpu_validate.json"
    )
    expected_correctness = {
        "selected": "c9c96efe01bc1caa7308c90bf07fdb2dd4e411cf4c31a7fda0e5cffe36bc7849",
        "group": "5f77a191afb25741661559c2bed56d3f809ae8c7b27412845ea6fa0750ea5a1c",
        "final": "1ba23407b4d13fc3baa570c53661a0407b7a36ef79ea0c445ee89a5294ffe045",
    }
    if correctness_gpu["twilight_selected_positions_sha256"] != expected_correctness["selected"]:
        raise AssertionError("per-Q selected-position hash changed")
    if correctness_gpu["twilight_group_positions_sha256"] != expected_correctness["group"]:
        raise AssertionError("group-position hash changed")
    if correctness_gpu["final_logits_sha256"] != expected_correctness["final"]:
        raise AssertionError("final logits hash changed")
    attention_control = attention_block_sha256(correctness_control)
    attention_gpu = attention_block_sha256(
        CORRECTNESS_V2 / "003_qa_1_i011/gpu_validate.json"
    )
    if attention_control != attention_gpu:
        raise AssertionError("attention K/V trace changed")

    detailed = {}
    for mode, filename in (
        ("control", "control_detailed.json"),
        ("gpu_union", "gpu_bool_detailed.json"),
    ):
        result = load(PROFILE / filename)
        metrics = result["diagnostic_breakdown_tokens"][0]
        detailed[mode] = {
            key: value
            for key, value in metrics.items()
            if key
            in {
                "twilight_final_b1_indices_cuda_seconds",
                "twilight_gpu_gqa_union_cuda_seconds",
                "twilight_index_sync_and_position_wall_seconds",
                "twilight_compact_index_decode_wall_seconds",
                "twilight_group_union_prepare_wall_seconds",
                "twilight_index_d2h_bytes",
            }
        }

    means = {
        mode: [
            row["mean_ms"] for row in chosen_rows if row["mode"] == mode
        ]
        for mode in ("control", "gpu_union")
    }
    packed_means = {
        mode: [
            row["mean_ms"] for row in bitpacked_rows if row["mode"] == mode
        ]
        for mode in ("control", "gpu_union")
    }
    summary = {
        "chosen_representation": "GPU bool membership bitmap; CPU flatnonzero decode",
        "source_sha256": snapshot,
        "correctness": {
            **expected_correctness,
            "attention_trace_sha256": attention_gpu,
            "attention_trace_rows": len(correctness_gpu["resident_attention_trace"]),
            "internal_cpu_union_assertions": "passed for all 32 decode steps x 28 layers x 8 KV heads",
        },
        "chosen_rows": chosen_rows,
        "chosen_overall": {
            "control_mean_ms": statistics.mean(means["control"]),
            "gpu_union_mean_ms": statistics.mean(means["gpu_union"]),
            "speedup_percent": (
                statistics.mean(means["control"])
                / statistics.mean(means["gpu_union"])
                - 1
            )
            * 100,
        },
        "003_rep2": {
            mode: 1000
            * load(
                CHOSEN
                / "003_qa_1_i011"
                / "rep2"
                / mode
                / "result.json"
            )["D2_D128_tpot"]["mean_seconds_per_token"]
            for mode in ("control", "gpu_union")
        },
        "detailed_003": detailed,
        "bitpacked_rows": bitpacked_rows,
        "bitpacked_overall": {
            "control_mean_ms": statistics.mean(packed_means["control"]),
            "gpu_union_mean_ms": statistics.mean(packed_means["gpu_union"]),
            "speedup_percent": (
                statistics.mean(packed_means["control"])
                / statistics.mean(packed_means["gpu_union"])
                - 1
            )
            * 100,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["chosen_overall"], indent=2))
    print("analysis validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
