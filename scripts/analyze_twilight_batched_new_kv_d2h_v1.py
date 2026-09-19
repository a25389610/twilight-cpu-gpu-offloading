#!/usr/bin/env python3
"""Validate and summarize the batched new-K/V D2H matched experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUESTS = (
    "001_niah_multikey_3_i011",
    "002_vt_i002",
    "003_qa_1_i011",
)
MODES = ("control", "token")
EXACT_FIELDS = (
    "request_id", "prompt_token_count", "prompt_sha256", "model", "dtype",
    "attention", "batch_size", "fixed_token_id", "decode_steps",
    "twilight_top_p", "twilight_budget_mode", "twilight_qk_backend",
    "twilight_gpu_compact_gqa_union", "final_logits_sha256",
    "checkpoint_logits_sha256", "twilight_selected_positions_sha256",
    "twilight_group_positions_sha256", "diagnostic_selected_positions_sha256",
    "diagnostic_group_positions_sha256", "diagnostic_logits_sha256",
)
METRICS = (
    "d2h_bytes", "d2h_copy_calls", "new_kv_d2h_ready_events",
    "new_kv_d2h_done_events", "new_kv_d2h_enqueue_wall_seconds",
    "new_kv_d2h_cuda_seconds", "host_ready_wait_wall_seconds",
    "new_kv_gpu_staging_wall_seconds", "new_kv_gpu_staging_cuda_seconds",
    "new_kv_host_scatter_wall_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/twilight_batched_new_kv_d2h_v1/formal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/twilight_batched_new_kv_d2h_v1/analysis")
    return parser.parse_args()


def read_latencies(path: Path) -> list[float]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return [float(row["latency_seconds"]) * 1000.0 for row in rows[1:]]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_request = []
    all_latencies = {mode: [] for mode in MODES}
    component_rows = {mode: [] for mode in MODES}
    peak_deltas = []
    for request in REQUESTS:
        results = {}
        for mode in MODES:
            run_dir = args.input_dir / request / "rep1" / mode
            result = json.loads((run_dir / "result.json").read_text())
            if result.get("status") != "ok":
                raise AssertionError(f"{request} {mode} did not finish")
            results[mode] = result
            all_latencies[mode].extend(read_latencies(run_dir / "result.per_token.csv"))
            component_rows[mode].append(result["diagnostic_breakdown_tokens"][0])
        control, token = results["control"], results["token"]
        for field in EXACT_FIELDS:
            if control[field] != token[field]:
                raise AssertionError(f"{request}: exact field mismatch: {field}")
        if token["twilight_new_kv_d2h_granularity"] != "token":
            raise AssertionError("optimized run is not token-level batching")
        if control["twilight_batched_new_kv_d2h"]:
            raise AssertionError("control unexpectedly enabled batching")
        control_ms = control["D2_D128_tpot"]["mean_seconds_per_token"] * 1000.0
        token_ms = token["D2_D128_tpot"]["mean_seconds_per_token"] * 1000.0
        peak_delta = token["peak_gpu_allocated_bytes"] - control["peak_gpu_allocated_bytes"]
        peak_deltas.append(peak_delta)
        per_request.append({
            "request": request,
            "control_tpot_ms": control_ms,
            "token_batched_tpot_ms": token_ms,
            "tpot_decrease_percent": 100.0 * (1.0 - token_ms / control_ms),
            "throughput_speedup_percent": 100.0 * (control_ms / token_ms - 1.0),
            "peak_gpu_allocated_delta_bytes": peak_delta,
            "final_logits_sha256": control["final_logits_sha256"],
        })
    overall = {}
    for mode in MODES:
        values = all_latencies[mode]
        overall[mode] = {
            "tokens": len(values), "mean_tpot_ms": statistics.mean(values),
            "median_tpot_ms": statistics.median(values), "min_tpot_ms": min(values),
            "max_tpot_ms": max(values),
        }
    overall["tpot_decrease_percent"] = 100.0 * (1.0 - overall["token"]["mean_tpot_ms"] / overall["control"]["mean_tpot_ms"])
    overall["throughput_speedup_percent"] = 100.0 * (overall["control"]["mean_tpot_ms"] / overall["token"]["mean_tpot_ms"] - 1.0)
    components = {
        mode: {metric: statistics.mean([row.get(metric, 0.0) for row in component_rows[mode]]) for metric in METRICS}
        for mode in MODES
    }
    first_token = json.loads((args.input_dir / REQUESTS[0] / "rep1/token/result.json").read_text())
    memory = {
        "gpu_persistent_staging_bytes": first_token["cache_state"]["new_kv_gpu_staging_allocated_bytes"],
        "gpu_temporary_buffer_bytes": 0,
        "pinned_cpu_staging_bytes": first_token["cache_state"]["new_kv_pinned_host_staging_allocated_bytes"],
        "peak_gpu_allocated_delta_bytes_by_request": peak_deltas,
    }
    summary = {"status": "FINAL VERIFY PASS", "per_request": per_request, "overall": overall, "components_three_request_mean": components, "memory": memory}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "per_request.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_request[0].keys())
        writer.writeheader()
        writer.writerows(per_request)
    lines = ["# Batched new-K/V D2H summary", "", "| Request | Control TPOT | Token-batched TPOT | TPOT decrease | Throughput speedup |", "|---|---:|---:|---:|---:|"]
    for row in per_request:
        lines.append(f"| `{row['request']}` | {row['control_tpot_ms']:.3f} ms | {row['token_batched_tpot_ms']:.3f} ms | {row['tpot_decrease_percent']:.2f}% | {row['throughput_speedup_percent']:.2f}% |")
    lines.extend(["", f"Overall 93-token row-weighted mean: {overall['control']['mean_tpot_ms']:.3f} → {overall['token']['mean_tpot_ms']:.3f} ms/token; TPOT -{overall['tpot_decrease_percent']:.2f}%, throughput +{overall['throughput_speedup_percent']:.2f}%.", "", "Component timers are diagnostic and overlapping; they are not additive TPOT savings."])
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
