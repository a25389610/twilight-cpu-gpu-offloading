#!/usr/bin/env python3
"""Validate paired direct-QK quality with the project's official RULER scorer."""

from __future__ import annotations

import collections
import csv
import json
import random
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_ruler_conditional_greedy_teacher_2k_v1 import official_ruler_score

COHORT = ROOT / "results/context_p_twilight_gqa_group_ruler130_v1/cohort/32768"
OUTPUT = ROOT / "results/twilight_direct_qk_ruler130_quality_20260927_v2"


def load(path: Path):
    return json.loads(path.read_text())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    location = fraction * (len(ordered) - 1)
    lo = int(location)
    hi = min(lo + 1, len(ordered) - 1)
    weight = location - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", local_files_only=True
    )
    eos_id = int(tokenizer.eos_token_id)
    rows = []
    for request_dir in sorted((COHORT / "requests").iterdir()):
        if not request_dir.is_dir():
            continue
        request = load(request_dir / "request.json")
        arms = {
            arm: load(OUTPUT / "requests" / request_dir.name / arm / "result.json")
            for arm in ("baseline", "direct")
        }
        for arm, payload in arms.items():
            assert payload["status"] == "ok"
            assert payload["request_id"] == request["request_id"]
            assert payload["prompt_sha256"] == request["prompt_sha256"]
            assert payload["decode_steps"] == request["generation_budget"]
            assert payload["decode_policy"] == "greedy_probe"
            assert payload["twilight_qk_backend"] == (
                "triton_prepare" if arm == "baseline" else "triton"
            )
            assert payload["greedy_probe_stop_at_eos"] is True
            assert 2 <= len(payload["greedy_token_ids"]) <= request["generation_budget"]
            assert (
                eos_id in payload["greedy_token_ids"]
                or len(payload["greedy_token_ids"]) == request["generation_budget"]
            )
        outputs = {}
        for arm, payload in arms.items():
            full_ids = [int(value) for value in payload["greedy_token_ids"]]
            eos_position = next(
                (index for index, value in enumerate(full_ids) if value == eos_id), None
            )
            ids = full_ids if eos_position is None else full_ids[: eos_position + 1]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            score = float(
                official_ruler_score(text, request["references"], request["scorer"])
            )
            outputs[arm] = {
                "ids": ids,
                "text": text,
                "score": score,
                "eos": eos_position is not None,
            }
        baseline, direct = outputs["baseline"], outputs["direct"]
        first_difference = next(
            (
                index + 1
                for index, (left, right) in enumerate(zip(baseline["ids"], direct["ids"]))
                if left != right
            ),
            None,
        )
        if first_difference is None and len(baseline["ids"]) != len(direct["ids"]):
            first_difference = min(len(baseline["ids"]), len(direct["ids"])) + 1
        rows.append(
            {
                "request": request_dir.name,
                "task": request["task"],
                "request_id": request["request_id"],
                "prompt_sha256": request["prompt_sha256"],
                "generation_budget": request["generation_budget"],
                "baseline_generated_tokens_to_eos": len(baseline["ids"]),
                "direct_generated_tokens_to_eos": len(direct["ids"]),
                "baseline_eos": baseline["eos"],
                "direct_eos": direct["eos"],
                "answer_token_exact": baseline["ids"] == direct["ids"],
                "prediction_text_exact": baseline["text"] == direct["text"],
                "first_answer_token_difference": first_difference,
                "baseline_score": baseline["score"],
                "direct_score": direct["score"],
                "score_delta": direct["score"] - baseline["score"],
                "baseline_prediction": baseline["text"],
                "direct_prediction": direct["text"],
            }
        )
    if len(rows) != 130 or len({row["request_id"] for row in rows}) != 130:
        raise AssertionError(f"incomplete or duplicate 130 cohort: {len(rows)}")
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    if len(by_task) != 13 or any(len(group) != 10 for group in by_task.values()):
        raise AssertionError("task balance is not 13 x 10")
    rng = random.Random(20260927)
    bootstrap = []
    for _ in range(20_000):
        sample = [
            rng.choice(group)["score_delta"]
            for group in by_task.values()
            for _ in range(len(group))
        ]
        bootstrap.append(statistics.fmean(sample))
    task_rows = []
    for task, group in sorted(by_task.items()):
        task_rows.append(
            {
                "task": task,
                "baseline_macro": statistics.fmean(row["baseline_score"] for row in group),
                "direct_macro": statistics.fmean(row["direct_score"] for row in group),
                "delta": statistics.fmean(row["score_delta"] for row in group),
                "better": sum(row["score_delta"] > 0 for row in group),
                "same": sum(row["score_delta"] == 0 for row in group),
                "worse": sum(row["score_delta"] < 0 for row in group),
                "answer_token_exact": sum(row["answer_token_exact"] for row in group),
            }
        )
    summary = {
        "classification": "quality-only current-source matched paired 32K RULER 13x10",
        "formal_tpot": False,
        "requests": len(rows),
        "tasks": len(by_task),
        "baseline_macro": statistics.fmean(row["baseline_score"] for row in rows),
        "direct_macro": statistics.fmean(row["direct_score"] for row in rows),
        "paired_macro_delta": statistics.fmean(row["score_delta"] for row in rows),
        "better": sum(row["score_delta"] > 0 for row in rows),
        "same": sum(row["score_delta"] == 0 for row in rows),
        "worse": sum(row["score_delta"] < 0 for row in rows),
        "answer_token_exact": sum(row["answer_token_exact"] for row in rows),
        "prediction_text_exact": sum(row["prediction_text_exact"] for row in rows),
        "both_eos": sum(row["baseline_eos"] and row["direct_eos"] for row in rows),
        "baseline_full_correct": sum(row["baseline_score"] == 100 for row in rows),
        "baseline_full_correct_regressed": sum(
            row["baseline_score"] == 100 and row["direct_score"] < 100
            for row in rows
        ),
        "task_stratified_paired_bootstrap": {
            "iterations": len(bootstrap),
            "seed": 20260927,
            "mean_delta_95_percent_interval": [
                percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)
            ],
        },
        "per_task": task_rows,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "per_request.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    )
    with (OUTPUT / "per_request.csv").open("w", newline="") as handle:
        fields = [key for key in rows[0] if not key.endswith("_prediction")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
