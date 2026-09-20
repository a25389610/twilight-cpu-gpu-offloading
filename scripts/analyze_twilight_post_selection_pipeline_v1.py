#!/usr/bin/env python3
"""Summarize the diagnostic-only post-Selection pipeline experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import statistics
from pathlib import Path
from typing import Iterable


MODES = ("control", "g1", "g2", "g4", "g8")
TRACE_RANGES = {
    "new_kv": (
        b'  "twilight_new_kv_trace": ',
        b'  "resident_diagnostic_only": ',
    ),
    "attention": (
        b'  "resident_attention_trace": ',
        b'  "twilight_detailed_selection_profile": ',
    ),
    "selection": (
        b'  "quest_selection_trace": ',
        b'  "twilight_budget_trace": ',
    ),
    "union": (
        b'  "twilight_group_union_trace": ',
        b'  "peak_gpu_allocated_bytes": ',
    ),
}


def merge(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, stop in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], stop))
        else:
            result.append((start, stop))
    return result


def duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(stop - start for start, stop in merge(intervals))


def overlap(
    left: Iterable[tuple[float, float]], right: Iterable[tuple[float, float]]
) -> float:
    a, b = merge(left), merge(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def timeline_summary(timeline: list[dict]) -> dict[str, float]:
    totals = {
        "layer_span_ms": 0.0,
        "cpu_decode_ms": 0.0,
        "cpu_gather_ms": 0.0,
        "selected_kv_h2d_ms": 0.0,
        "gather_h2d_overlap_ms": 0.0,
        "decode_with_gather_or_h2d_overlap_ms": 0.0,
        "post_gpu_with_cpu_or_h2d_overlap_ms": 0.0,
        "fill_to_first_h2d_ms": 0.0,
        "drain_after_last_gather_ms": 0.0,
    }
    for layer in timeline:
        chunks = layer["chunks"]
        decode = [
            (item["cpu_decode_start_ms"], item["cpu_decode_end_ms"])
            for item in chunks
        ]
        gather = [
            (item["cpu_gather_start_ms"], item["cpu_gather_end_ms"])
            for item in chunks
        ]
        h2d = [
            (item["selected_kv_h2d_start_ms"], item["selected_kv_h2d_end_ms"])
            for item in chunks
        ]
        union = [
            (item["union_start_ms"], item["union_end_ms"])
            for item in chunks
            if "union_start_ms" in item
        ]
        index = [
            (item["index_d2h_start_ms"], item["index_d2h_end_ms"])
            for item in chunks
        ]
        cpu_or_h2d = decode + gather + h2d
        totals["layer_span_ms"] += max(stop for _, stop in cpu_or_h2d + union + index)
        totals["cpu_decode_ms"] += duration(decode)
        totals["cpu_gather_ms"] += duration(gather)
        totals["selected_kv_h2d_ms"] += duration(h2d)
        totals["gather_h2d_overlap_ms"] += overlap(gather, h2d)
        totals["decode_with_gather_or_h2d_overlap_ms"] += overlap(
            decode, gather + h2d
        )
        totals["post_gpu_with_cpu_or_h2d_overlap_ms"] += overlap(
            union + index, cpu_or_h2d
        )
        totals["fill_to_first_h2d_ms"] += min(start for start, _ in h2d)
        totals["drain_after_last_gather_ms"] += max(
            0.0, max(stop for _, stop in h2d) - max(stop for _, stop in gather)
        )
    totals["layers"] = len(timeline)
    totals["gather_overlap_fraction"] = (
        totals["gather_h2d_overlap_ms"] / totals["cpu_gather_ms"]
    )
    totals["h2d_overlap_fraction"] = (
        totals["gather_h2d_overlap_ms"] / totals["selected_kv_h2d_ms"]
    )
    return totals


def raw_section_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("rb") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_READ
    ) as view:
        for name, (start_marker, stop_marker) in TRACE_RANGES.items():
            start = view.find(start_marker)
            stop = view.find(stop_marker, start + len(start_marker))
            if start < 0 or stop < 0:
                raise ValueError(f"missing {name} trace in {path}")
            digest = hashlib.sha256()
            cursor = start + len(start_marker)
            while cursor < stop:
                chunk_stop = min(stop, cursor + 8 * 1024 * 1024)
                digest.update(view[cursor:chunk_stop])
                cursor = chunk_stop
            result[name] = digest.hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--correctness-dir", type=Path, required=True)
    parser.add_argument("--request", default="003_qa_1_i011")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot_request = args.pilot_dir / args.request
    repetitions = sorted(pilot_request.glob("rep*"))
    if not repetitions:
        raise ValueError("no pilot repetitions found")

    runs: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for repetition in repetitions:
        for mode in MODES:
            path = repetition / mode / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result.get("status") != "ok":
                    raise ValueError(f"non-ok result: {path}")
                runs[mode].append(result)

    control = runs["control"]
    g1 = runs["g1"]
    paired_count = min(len(control), len(g1))
    paired = []
    for index in range(paired_count):
        control_ms = control[index]["D2_D128_tpot"]["mean_seconds_per_token"] * 1000
        g1_ms = g1[index]["D2_D128_tpot"]["mean_seconds_per_token"] * 1000
        paired.append(
            {
                "repetition": index + 1,
                "control_ms_per_token": control_ms,
                "g1_ms_per_token": g1_ms,
                "delta_ms_per_token": g1_ms - control_ms,
                "delta_percent": 100 * (g1_ms / control_ms - 1),
            }
        )

    metric_keys = (
        "twilight_selection_core_wall_seconds",
        "twilight_selection_core_cuda_seconds",
        "post_selection_exposed_wall_seconds",
        "post_selection_union_cuda_seconds",
        "post_selection_index_d2h_cuda_seconds",
        "post_selection_index_wait_wall_seconds",
        "twilight_compact_index_decode_wall_seconds",
        "host_gather_pack_wall_seconds",
        "h2d_cuda_seconds",
        "h2d_enqueue_wall_seconds",
        "diagnostic_token_wall_seconds",
    )

    def diagnostic_means(items: list[dict]) -> dict[str, float]:
        return {
            key.replace("_seconds", "_ms"): 1000
            * statistics.mean(
                item["diagnostic_transfer_metrics_one_post_timing_token"].get(
                    key, 0.0
                )
                for item in items
            )
            for key in metric_keys
        }

    first_rep_sweep = {}
    for mode in MODES:
        if not runs[mode]:
            continue
        item = runs[mode][0]
        first_rep_sweep[mode] = {
            "tpot_ms_per_token": item["D2_D128_tpot"]["mean_seconds_per_token"]
            * 1000,
            "timeline": (
                timeline_summary(
                    item["diagnostic_transfer_metrics_one_post_timing_token"][
                        "post_selection_timeline"
                    ]
                )
                if mode != "control"
                else None
            ),
        }

    correctness_root = args.correctness_dir / args.request / "rep1"
    trace_hashes = {
        mode: raw_section_hashes(correctness_root / mode / "result.json")
        for mode in ("control", "g1", "g2")
    }
    correctness_exact = {
        trace: len({trace_hashes[mode][trace] for mode in trace_hashes}) == 1
        for trace in TRACE_RANGES
    }

    control_mean = statistics.mean(
        item["D2_D128_tpot"]["mean_seconds_per_token"] * 1000
        for item in control[:paired_count]
    )
    g1_mean = statistics.mean(
        item["D2_D128_tpot"]["mean_seconds_per_token"] * 1000
        for item in g1[:paired_count]
    )
    summary = {
        "request": args.request,
        "formal_token_range": "D2-D32",
        "paired": paired,
        "paired_summary": {
            "pairs": paired_count,
            "control_mean_ms_per_token": control_mean,
            "g1_mean_ms_per_token": g1_mean,
            "delta_mean_ms_per_token": g1_mean - control_mean,
            "delta_percent_ratio_of_means": 100 * (g1_mean / control_mean - 1),
            "all_pipeline_faster": all(
                row["delta_ms_per_token"] < 0 for row in paired
            ),
        },
        "first_repetition_chunk_sweep": first_rep_sweep,
        "diagnostic_mean": {
            "control": diagnostic_means(control[:paired_count]),
            "g1": diagnostic_means(g1[:paired_count]),
        },
        "correctness": {
            "trace_hashes": trace_hashes,
            "all_trace_sections_exact": all(correctness_exact.values()),
            "trace_sections_exact": correctness_exact,
            "selection_rows": 22176,
            "group_union_rows": 7392,
            "attention_rows": 84,
            "new_kv_rows": 7392,
        },
        "memory": {
            "additional_pinned_cpu_bytes": runs["g1"][0]["cache_state"][
                "post_selection_pinned_index_bytes"
            ],
            "additional_gpu_buffer_bytes": runs["g1"][0]["cache_state"][
                "post_selection_additional_gpu_buffer_bytes"
            ],
        },
        "decision": (
            "negative_on_003_do_not_expand_to_three_requests_or_update_current"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["paired_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
