#!/usr/bin/env python3
"""Focused helper tests for the Twilight matched-budget offload prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "headinfer"))

from headinfer.twilight_offload_cache import (
    TwilightMatchedBudgetOffloadedCache,
    matched_budget_counts_tensor,
    affine_int4_dequantize,
    affine_int4_pack,
    matched_budget_counts,
)


def make_selection_cache(
    *,
    p: float,
    batched: bool,
    mode: str,
    flat: bool = False,
    stabilized: bool = False,
    gqa_group: bool = False,
) -> TwilightMatchedBudgetOffloadedCache:
    cache = TwilightMatchedBudgetOffloadedCache(
        num_cache_entries=2,
        max_cache_len=32,
        head_dim=4,
        num_query_heads_per_kv=3,
        dtype=torch.bfloat16,
        device="cuda",
        sink_tokens=1,
        recent_tokens=2,
        block_size=1,
        candidate_token_budget=8,
        top_p=p,
        budget_mode=mode,
        layer_batched_selection=batched,
        layer_flat_ragged_execution=flat,
        sparse_gather_stabilized=stabilized,
        gqa_groupwise_execution=gqa_group,
    )
    torch.manual_seed(1234)
    cache._prompt_length = 16
    for entry in range(2):
        cache._logical_lengths[entry] = 16
        keys = torch.randn(1, 1, 16, 4, dtype=torch.bfloat16, device="cuda")
        cache._build_metadata(entry, keys)
    cache.enable_selection_trace()
    cache.enable_budget_trace()
    return cache


def populate_host_cache(cache: TwilightMatchedBudgetOffloadedCache) -> None:
    """Give parity caches identical deterministic CPU-resident history."""
    for entry in range(2):
        base = torch.arange(16 * 4, dtype=torch.float32).view(16, 4)
        key = (base + entry * 100).to(torch.bfloat16)
        value = (base * 0.25 - entry * 10).to(torch.bfloat16)
        cache._host_tensor("key", entry)[0, 0, :16].copy_(key)
        cache._host_tensor("value", entry)[0, 0, :16].copy_(value)


def main() -> int:
    torch.manual_seed(7)
    keys = torch.randn(17, 128, dtype=torch.float32)
    packed, scale, minimum = affine_int4_pack(keys)
    restored = affine_int4_dequantize(packed, scale, minimum)
    assert packed.shape == (17, 64)
    assert restored.shape == keys.shape
    assert torch.isfinite(restored).all()
    per_token_max_error = (restored - keys).abs().amax(dim=-1)
    assert torch.all(per_token_max_error <= scale.float() / 2 + 1e-6)

    counts = matched_budget_counts([10, 20, 30], 33, [50, 50, 50])
    assert sum(counts) == 33
    assert counts[0] < counts[1] < counts[2]
    capped = matched_budget_counts([100, 1, 1], 12, [4, 20, 20])
    assert sum(capped) == 12 and capped[0] == 4

    desired_groups = ([10, 20, 30], [7, 7, 7], [3, 8, 5])
    for desired_values, target, capacity in (
        (desired_groups[0], 33, 50),
        (desired_groups[1], 20, 24),
        (desired_groups[2], 11, 20),
    ):
        desired = torch.tensor([desired_values], dtype=torch.long)
        actual = matched_budget_counts_tensor(desired, target, capacity)
        expected = torch.tensor(
            [matched_budget_counts(desired_values, target, [capacity] * 3)],
            dtype=torch.long,
        )
        assert torch.equal(actual.cpu(), expected)

    if torch.cuda.is_available():
        torch.manual_seed(4321)
        queries = [
            torch.randn(1, 3, 1, 4, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        sequential = make_selection_cache(p=0.85, batched=False, mode="dynamic")
        sequential_positions = [
            sequential._selected_positions(query=query, entry=entry, old_length=16)
            for entry, query in enumerate(queries)
        ]
        batched = make_selection_cache(p=0.85, batched=True, mode="dynamic")
        batched.prepare_layer_selection(entries=[0, 1], queries=queries)
        batched_positions = [
            batched._selected_positions(query=query, entry=entry, old_length=16)
            for entry, query in enumerate(queries)
        ]
        assert all(
            torch.equal(seq, batch)
            for seq_group, batch_group in zip(sequential_positions, batched_positions)
            for seq, batch in zip(seq_group, batch_group)
        )
        assert sequential.selection_trace() == batched.selection_trace()
        for row in batched.budget_trace():
            assert row["budget_mode"] == "dynamic"
            assert row["raw_b1_tokens"] == row["selected_adaptive_tokens"]
            assert 1 <= row["raw_b1_tokens"] <= row["b0_tokens"]
            assert row["selected_history_tokens"] == (
                row["protected_tokens"] + row["selected_adaptive_tokens"]
            )

        matched = make_selection_cache(p=0.85, batched=True, mode="matched")
        matched.prepare_layer_selection(entries=[0, 1], queries=queries)
        for entry, query in enumerate(queries):
            matched._selected_positions(query=query, entry=entry, old_length=16)
        matched_rows = matched.budget_trace()
        for entry in range(2):
            entry_rows = [row for row in matched_rows if row["entry"] == entry]
            assert len(entry_rows) == 3
            assert sum(row["selected_adaptive_tokens"] for row in entry_rows) == 3
            assert any(
                row["raw_b1_tokens"] != row["selected_adaptive_tokens"]
                for row in entry_rows
            )

        # Layer-flat execution must preserve selection and H2D bytes while
        # reducing K/V copy calls from 2 per Query head to 2 per layer.
        before = make_selection_cache(p=0.85, batched=True, mode="dynamic")
        flat = make_selection_cache(
            p=0.85, batched=True, mode="dynamic", flat=True
        )
        populate_host_cache(before)
        populate_host_cache(flat)
        current_keys = [
            torch.randn(1, 1, 1, 4, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        current_values = [torch.randn_like(key) for key in current_keys]
        before.prepare_layer_selection(entries=[0, 1], queries=queries)
        flat.prepare_layer_selection(entries=[0, 1], queries=queries)
        before.enable_metrics()
        flat.enable_metrics()
        before_outputs = []
        for entry in range(2):
            key_out, value_out = before.update(
                current_keys[entry],
                current_values[entry],
                entry,
                {"query_states": queries[entry]},
            )
            lengths = before.get_last_query_head_lengths(entry)
            assert lengths is not None
            before_outputs.extend(
                (
                    key_out[0, head, :length].clone(),
                    value_out[0, head, :length].clone(),
                )
                for head, length in enumerate(lengths)
            )
        flat_key, flat_value, cu_k, max_k, flat_lengths = (
            flat.update_layer_flat_ragged(
                key_states=current_keys,
                value_states=current_values,
                queries=queries,
                entries=[0, 1],
            )
        )
        before.synchronize()
        flat.synchronize()
        offsets = [int(value) for value in cu_k.cpu().tolist()]
        assert tuple(offsets[index + 1] - offsets[index] for index in range(6)) == flat_lengths
        assert max_k == max(flat_lengths)
        for head, ((key_ref, value_ref), length) in enumerate(
            zip(before_outputs, flat_lengths)
        ):
            start, stop = offsets[head], offsets[head + 1]
            assert stop - start == length
            assert torch.equal(key_ref, flat_key[start:stop, 0])
            assert torch.equal(value_ref, flat_value[start:stop, 0])
        before_metrics = before.resolve_metrics()
        flat_metrics = flat.resolve_metrics()
        assert before.selection_trace() == flat.selection_trace()
        assert before_metrics["h2d_bytes"] == flat_metrics["h2d_bytes"]
        assert before_metrics["h2d_copy_calls"] == 12
        assert flat_metrics["h2d_copy_calls"] == 2
        assert flat_metrics["host_gather_index_select_calls"] == 4

        old_flat = make_selection_cache(
            p=0.85, batched=True, mode="dynamic", flat=True
        )
        stabilized_flat = make_selection_cache(
            p=0.85,
            batched=True,
            mode="dynamic",
            flat=True,
            stabilized=True,
        )
        populate_host_cache(old_flat)
        populate_host_cache(stabilized_flat)
        torch.manual_seed(991)
        stable_keys = [
            torch.randn(1, 1, 1, 4, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        stable_values = [torch.randn_like(key) for key in stable_keys]

        def execute_flat(cache: TwilightMatchedBudgetOffloadedCache):
            cache.prepare_layer_selection(entries=[0, 1], queries=queries)
            cache.enable_metrics()
            output = cache.update_layer_flat_ragged(
                key_states=stable_keys,
                value_states=stable_values,
                queries=queries,
                entries=[0, 1],
            )
            cache.synchronize()
            key, value, cu_seqlens, max_seqlen, lengths = output
            return (
                key.clone(),
                value.clone(),
                cu_seqlens.clone(),
                max_seqlen,
                lengths,
                cache.selection_trace(),
                cache.budget_trace(),
                cache.resolve_metrics(),
            )

        old_result = execute_flat(old_flat)
        stabilized_result = execute_flat(stabilized_flat)
        assert torch.equal(old_result[0], stabilized_result[0])
        assert torch.equal(old_result[1], stabilized_result[1])
        assert torch.equal(old_result[2], stabilized_result[2])
        assert old_result[3:7] == stabilized_result[3:7]
        old_metrics, stabilized_metrics = old_result[7], stabilized_result[7]
        assert old_metrics["h2d_bytes"] == stabilized_metrics["h2d_bytes"]
        assert old_metrics["h2d_copy_calls"] == stabilized_metrics["h2d_copy_calls"]
        assert old_metrics["host_gather_index_select_calls"] == 4
        assert stabilized_metrics["host_gather_index_select_calls"] == 2
        assert old_metrics["host_gather_index_temporary_allocations"] == 2
        assert stabilized_metrics.get("host_gather_index_temporary_allocations", 0) == 0

        # Paper-aligned GQA execution must preserve each head's raw Top-p set,
        # transfer exactly one set union per KV group, and never duplicate a KV
        # position inside that group's H2D payload.
        reference = make_selection_cache(
            p=0.85, batched=True, mode="dynamic"
        )
        grouped = make_selection_cache(
            p=0.85, batched=True, mode="dynamic", gqa_group=True
        )
        populate_host_cache(reference)
        populate_host_cache(grouped)
        reference.prepare_layer_selection(entries=[0, 1], queries=queries)
        expected_per_q = [
            reference._selected_positions(
                query=query, entry=entry, old_length=16
            )
            for entry, query in enumerate(queries)
        ]
        expected_unions = [
            torch.unique(torch.cat(positions), sorted=True)
            for positions in expected_per_q
        ]
        grouped.prepare_layer_selection(entries=[0, 1], queries=queries)
        grouped.enable_metrics()
        group_key, group_value, group_cu_k, group_max_k, group_lengths = (
            grouped.update_layer_gqa_group_ragged(
                key_states=current_keys,
                value_states=current_values,
                queries=queries,
                entries=[0, 1],
            )
        )
        grouped.synchronize()
        expected_lengths = tuple(int(union.numel()) + 1 for union in expected_unions)
        assert group_lengths == expected_lengths
        assert group_max_k == max(expected_lengths)
        group_offsets = [int(value) for value in group_cu_k.cpu().tolist()]
        for entry, (union, total_length) in enumerate(
            zip(expected_unions, group_lengths)
        ):
            assert union.numel() == torch.unique(union).numel()
            start, stop = group_offsets[entry], group_offsets[entry + 1]
            expected_key = torch.cat(
                (
                    grouped._host_tensor("key", entry)[0, 0, :16].index_select(0, union).cuda(),
                    current_keys[entry][0, 0],
                ),
                dim=0,
            )
            expected_value = torch.cat(
                (
                    grouped._host_tensor("value", entry)[0, 0, :16].index_select(0, union).cuda(),
                    current_values[entry][0, 0],
                ),
                dim=0,
            )
            assert stop - start == total_length
            assert torch.equal(group_key[start:stop, 0], expected_key)
            assert torch.equal(group_value[start:stop, 0], expected_value)
        grouped_metrics = grouped.resolve_metrics()
        expected_history = sum(int(union.numel()) for union in expected_unions)
        expected_per_q_history = sum(
            int(selected.numel())
            for positions in expected_per_q
            for selected in positions
        )
        expected_bytes = (
            2 * expected_history * 4 * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        assert grouped_metrics["h2d_bytes"] == expected_bytes
        assert grouped_metrics["h2d_copy_calls"] == 2
        assert grouped_metrics["group_union_history_tokens_total"] == expected_history
        assert grouped_metrics["selected_history_tokens_total"] == expected_per_q_history
        assert grouped_metrics["group_union_duplicate_rows_eliminated"] == (
            expected_per_q_history - expected_history
        )
        assert grouped.selection_trace() == reference.selection_trace()
    print("twilight helper tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
