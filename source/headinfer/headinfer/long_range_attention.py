"""Diagnostic long-range attention scoring for HeadInfer KV entries."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LongRangeScoreResult:
    scores: torch.Tensor
    elapsed_seconds: float
    peak_allocated_delta_bytes: int
    query_bytes: int
    transient_logits_bytes: int


def score_long_range_attention(
    query_window: torch.Tensor,
    key_entries: list[torch.Tensor],
    *,
    num_layers: int,
    num_kv_heads: int,
    sink_tokens: int,
    recent_tokens: int,
) -> LongRangeScoreResult:
    """Return mean attention mass outside sink + position-local recent KV.

    ``query_window`` is post-RoPE and ordered as
    ``[layer * num_kv_heads + kv_head, query_pos, q_head_in_group, head_dim]``.
    Each cached Key entry must be post-RoPE ``[1, 1, seq_len, head_dim]``.
    The softmax is evaluated in FP32, one Layer at a time; no attention matrix
    is retained after a Layer score is reduced to eight scalars.
    """
    if query_window.ndim != 4:
        raise ValueError("query_window must be [entries, positions, q_per_kv, dim]")
    entries, positions, q_per_kv, head_dim = map(int, query_window.shape)
    if entries != num_layers * num_kv_heads or len(key_entries) != entries:
        raise ValueError("Query/Key entry count differs from model geometry")
    if positions <= 0 or q_per_kv <= 0 or head_dim <= 0:
        raise ValueError("invalid Query geometry")
    if sink_tokens < 0 or recent_tokens <= 0:
        raise ValueError("invalid sink/recent window")
    first = key_entries[0]
    if first.ndim != 4 or tuple(first.shape[:2]) != (1, 1):
        raise ValueError("Key entries must have shape [1, 1, seq_len, head_dim]")
    seq_len = int(first.shape[-2])
    if int(first.shape[-1]) != head_dim or positions > seq_len:
        raise ValueError("Query/Key sequence or head dimension mismatch")
    if any(tuple(key.shape) != tuple(first.shape) for key in key_entries):
        raise ValueError("Key entries do not share one geometry")

    device = first.device
    query_window = query_window.to(device=device)
    query_bytes = query_window.numel() * query_window.element_size()
    absolute_queries = torch.arange(
        seq_len - positions, seq_len, dtype=torch.long, device=device
    )
    key_positions = torch.arange(seq_len, dtype=torch.long, device=device)
    causal = key_positions.unsqueeze(0) <= absolute_queries.unsqueeze(1)
    long_stop = (absolute_queries + 1 - recent_tokens).clamp_min(sink_tokens)
    long_range = (
        (key_positions.unsqueeze(0) >= sink_tokens)
        & (key_positions.unsqueeze(0) < long_stop.unsqueeze(1))
    )
    valid = causal

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        allocated_before = 0
    started = time.perf_counter()
    layer_scores: list[torch.Tensor] = []
    transient_logits_bytes = num_kv_heads * q_per_kv * positions * seq_len * 4
    scale = 1.0 / math.sqrt(head_dim)
    with torch.inference_mode():
        for layer in range(num_layers):
            start = layer * num_kv_heads
            stop = start + num_kv_heads
            # [8, positions, 3, dim] -> [8, 3, positions, dim]
            queries = query_window[start:stop].permute(0, 2, 1, 3).float()
            keys = torch.cat(key_entries[start:stop], dim=0).squeeze(1).float()
            logits = torch.einsum("ehqd,ekd->ehqk", queries, keys) * scale
            logits.masked_fill_(~valid.unsqueeze(0).unsqueeze(0), -torch.inf)
            probabilities = torch.softmax(logits, dim=-1)
            mass = probabilities.masked_fill(
                ~long_range.unsqueeze(0).unsqueeze(0), 0.0
            ).sum(dim=-1)
            layer_scores.append(mass.mean(dim=(1, 2)))
    scores = torch.cat(layer_scores).float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_delta = max(
            0, int(torch.cuda.max_memory_allocated(device) - allocated_before)
        )
    else:
        elapsed = time.perf_counter() - started
        peak_delta = 0
    if tuple(scores.shape) != (entries,) or not torch.isfinite(scores).all():
        raise AssertionError("long-range score output is incomplete or non-finite")
    if bool(((scores < 0) | (scores > 1)).any()):
        raise AssertionError("attention probability mass must remain in [0, 1]")
    return LongRangeScoreResult(
        scores=scores.cpu(),
        elapsed_seconds=elapsed,
        peak_allocated_delta_bytes=peak_delta,
        query_bytes=query_bytes,
        transient_logits_bytes=transient_logits_bytes,
    )
