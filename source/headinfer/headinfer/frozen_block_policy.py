"""Quality-only fixed prompt-block selection for per-KV-head experiments.

This module does not alter cache storage or measure transfer savings.  It keeps
the Full KV cache resident, slices the K/V tensors visible to each decode-only
attention call, and is therefore intended only for correctness and quality
feasibility studies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping, Optional

import torch


@dataclass(frozen=True)
class FrozenBlockRecord:
    entry: int
    full_length: int
    selected_length: int
    fixed_prompt_positions: int
    dynamic_recent_positions: int
    selected_first: int
    selected_last: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def prompt_cpu_block_bounds(
    prompt_length: int,
    *,
    sink_tokens: int,
    recent_tokens: int,
    block_size: int,
) -> list[tuple[int, int]]:
    """Partition the Prefill CPU-history region into logical token blocks."""
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if sink_tokens < 0 or recent_tokens <= 0 or block_size <= 0:
        raise ValueError("invalid sink/recent/block size")
    cpu_start = min(sink_tokens, prompt_length)
    cpu_stop = max(cpu_start, prompt_length - recent_tokens)
    return [
        (start, min(start + block_size, cpu_stop))
        for start in range(cpu_start, cpu_stop, block_size)
    ]


def exact_output_aware_block_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    sink_tokens: int,
    recent_tokens: int,
    block_size: int,
    query_chunk_size: int = 8,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Score fixed prompt blocks from exact Full-attention output contribution.

    ``queries`` is the RoPE-applied Prefill suffix with shape ``[W, G, D]``.
    ``keys`` and ``values`` are one KV entry with shape ``[L, D]``.  Each
    Query's block-contribution norms are normalized across candidate blocks;
    the final request-level score is the maximum across suffix positions and
    GQA Query Heads.
    """
    if queries.ndim != 3 or keys.ndim != 2 or values.ndim != 2:
        raise ValueError("expected queries [W,G,D] and keys/values [L,D]")
    if keys.shape != values.shape or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("incompatible Query/Key/Value geometry")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    prompt_length = int(keys.shape[0])
    query_positions = int(queries.shape[0])
    if query_positions <= 0 or query_positions > prompt_length:
        raise ValueError("invalid Query suffix length")
    bounds = prompt_cpu_block_bounds(
        prompt_length,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        block_size=block_size,
    )
    if not bounds:
        raise ValueError("CPU-history region contains no candidate blocks")

    cpu_start = bounds[0][0]
    cpu_stop = bounds[-1][1]
    cpu_length = cpu_stop - cpu_start
    num_blocks = len(bounds)
    padded_length = num_blocks * block_size
    padding = padded_length - cpu_length
    device = queries.device
    q = queries.float()
    k = keys.float()
    v_cpu = values[cpu_start:cpu_stop].float()
    if padding:
        v_cpu = torch.nn.functional.pad(v_cpu, (0, 0, 0, padding))
    v_blocks = v_cpu.reshape(num_blocks, block_size, keys.shape[-1])

    absolute_queries = torch.arange(
        prompt_length - query_positions,
        prompt_length,
        dtype=torch.long,
        device=device,
    )
    key_positions = torch.arange(prompt_length, dtype=torch.long, device=device)
    best = torch.zeros(num_blocks, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(float(keys.shape[-1]))
    for group_head in range(int(q.shape[1])):
        scores = torch.matmul(q[:, group_head], k.transpose(0, 1)) * scale
        scores = scores.masked_fill(
            key_positions.unsqueeze(0) > absolute_queries.unsqueeze(1),
            -torch.inf,
        )
        weights = torch.softmax(scores, dim=-1)
        candidate_weights = weights[:, cpu_start:cpu_stop]
        if padding:
            candidate_weights = torch.nn.functional.pad(
                candidate_weights, (0, padding)
            )
        candidate_weights = candidate_weights.reshape(
            query_positions, num_blocks, block_size
        )
        for start in range(0, query_positions, query_chunk_size):
            stop = min(start + query_chunk_size, query_positions)
            outputs = torch.einsum(
                "qbt,btd->qbd", candidate_weights[start:stop], v_blocks
            )
            norms = torch.linalg.vector_norm(outputs, dim=-1)
            relative = norms / norms.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            best = torch.maximum(best, relative.max(dim=0).values)
    if not torch.isfinite(best).all():
        raise AssertionError("non-finite block importance score")
    return best.cpu(), bounds


def removal_aware_gqa_block_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    sink_tokens: int,
    recent_tokens: int,
    block_size: int,
    top_r_queries: int = 4,
    query_chunk_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Score blocks by their exact single-block removal effect before O projection.

    For each Prefill Query and each GQA Query Head, the Full-softmax output
    after removing one candidate block is recovered analytically.  Squared
    removal deltas are summed across the Query Heads sharing this KV entry,
    normalized across blocks within each Query, and aggregated across Queries
    with both maximum and mean-of-top-r statistics.
    """
    if queries.ndim != 3 or keys.ndim != 2 or values.ndim != 2:
        raise ValueError("expected queries [W,G,D] and keys/values [L,D]")
    if keys.shape != values.shape or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("incompatible Query/Key/Value geometry")
    if query_chunk_size <= 0 or top_r_queries <= 0:
        raise ValueError("query_chunk_size and top_r_queries must be positive")
    prompt_length = int(keys.shape[0])
    query_positions = int(queries.shape[0])
    if query_positions <= 0 or query_positions > prompt_length:
        raise ValueError("invalid Query suffix length")
    bounds = prompt_cpu_block_bounds(
        prompt_length,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        block_size=block_size,
    )
    if not bounds:
        raise ValueError("CPU-history region contains no candidate blocks")

    cpu_start = bounds[0][0]
    cpu_stop = bounds[-1][1]
    cpu_length = cpu_stop - cpu_start
    num_blocks = len(bounds)
    padded_length = num_blocks * block_size
    padding = padded_length - cpu_length
    device = queries.device
    q = queries.float()
    k = keys.float()
    v = values.float()
    v_cpu = v[cpu_start:cpu_stop]
    if padding:
        v_cpu = torch.nn.functional.pad(v_cpu, (0, 0, 0, padding))
    v_blocks = v_cpu.reshape(num_blocks, block_size, keys.shape[-1])

    absolute_queries = torch.arange(
        prompt_length - query_positions,
        prompt_length,
        dtype=torch.long,
        device=device,
    )
    key_positions = torch.arange(prompt_length, dtype=torch.long, device=device)
    removal_energy = torch.zeros(
        query_positions, num_blocks, dtype=torch.float32, device=device
    )
    scale = 1.0 / math.sqrt(float(keys.shape[-1]))
    for group_head in range(int(q.shape[1])):
        logits = torch.matmul(q[:, group_head], k.transpose(0, 1)) * scale
        logits = logits.masked_fill(
            key_positions.unsqueeze(0) > absolute_queries.unsqueeze(1),
            -torch.inf,
        )
        weights = torch.softmax(logits, dim=-1)
        candidate_weights = weights[:, cpu_start:cpu_stop]
        if padding:
            candidate_weights = torch.nn.functional.pad(
                candidate_weights, (0, padding)
            )
        candidate_weights = candidate_weights.reshape(
            query_positions, num_blocks, block_size
        )
        for start in range(0, query_positions, query_chunk_size):
            stop = min(start + query_chunk_size, query_positions)
            block_weights = candidate_weights[start:stop]
            block_outputs = torch.einsum("qbt,btd->qbd", block_weights, v_blocks)
            block_mass = block_weights.sum(dim=-1)
            full_output = block_outputs.sum(dim=1)
            if cpu_start:
                full_output = full_output + torch.matmul(
                    weights[start:stop, :cpu_start], v[:cpu_start]
                )
            if cpu_stop < prompt_length:
                full_output = full_output + torch.matmul(
                    weights[start:stop, cpu_stop:], v[cpu_stop:]
                )
            denominator = (1.0 - block_mass).clamp_min(1e-6)
            delta = (
                block_outputs - block_mass.unsqueeze(-1) * full_output.unsqueeze(1)
            ) / denominator.unsqueeze(-1)
            removal_energy[start:stop] += delta.square().sum(dim=-1)

    removal_norm = removal_energy.sqrt()
    relative = removal_norm / removal_norm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    max_scores = relative.max(dim=0).values
    top_r = min(top_r_queries, query_positions)
    top_r_scores = relative.topk(top_r, dim=0).values.mean(dim=0)
    if not torch.isfinite(max_scores).all() or not torch.isfinite(top_r_scores).all():
        raise AssertionError("non-finite removal-aware block importance score")
    return max_scores.cpu(), top_r_scores.cpu(), bounds


def selected_positions_from_blocks(
    bounds: list[tuple[int, int]],
    scores: torch.Tensor,
    *,
    extra_block_count: int,
    sink_tokens: int,
) -> tuple[torch.Tensor, list[int]]:
    """Return sorted sink plus selected fixed CPU-history positions."""
    if scores.ndim != 1 or int(scores.numel()) != len(bounds):
        raise ValueError("score count does not match block bounds")
    if not 0 <= extra_block_count <= len(bounds):
        raise ValueError("extra_block_count is outside candidate range")
    ranked = sorted(
        range(len(bounds)), key=lambda index: (-float(scores[index]), index)
    )
    chosen = sorted(ranked[:extra_block_count])
    pieces = [torch.arange(0, sink_tokens, dtype=torch.long)]
    pieces.extend(
        torch.arange(bounds[index][0], bounds[index][1], dtype=torch.long)
        for index in chosen
    )
    positions = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)
    if positions.numel() and not bool(torch.all(positions[1:] > positions[:-1])):
        raise AssertionError("fixed prompt positions are not strictly increasing")
    return positions, chosen


class FrozenHeadBlockController:
    """Apply a frozen prompt-history position set to selected KV entries."""

    def __init__(
        self,
        original_attention: Callable[..., Any],
        *,
        num_entries: int,
        prompt_length: int,
        recent_tokens: int,
    ) -> None:
        if num_entries <= 0 or prompt_length <= 0 or recent_tokens <= 0:
            raise ValueError("invalid controller geometry")
        self.original_attention = original_attention
        self.num_entries = int(num_entries)
        self.prompt_length = int(prompt_length)
        self.recent_tokens = int(recent_tokens)
        self._fixed: dict[int, torch.Tensor] = {}
        self._active = False
        self._call_count = 0
        self._records: list[FrozenBlockRecord] = []

    @staticmethod
    def _slice_attention_mask(
        attention_mask: Optional[torch.Tensor],
        positions: torch.Tensor,
        full_length: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if int(attention_mask.shape[-1]) != full_length:
            raise AssertionError("attention mask and K/V lengths differ")
        return attention_mask.index_select(-1, positions)

    def start_forward(self, fixed_positions: Mapping[int, torch.Tensor]) -> None:
        if self._active:
            raise RuntimeError("previous forward has not been finished")
        checked: dict[int, torch.Tensor] = {}
        for raw_entry, raw_positions in fixed_positions.items():
            entry = int(raw_entry)
            if not 0 <= entry < self.num_entries:
                raise ValueError("entry outside controller universe")
            positions = raw_positions.detach().to(device="cpu", dtype=torch.long)
            if positions.ndim != 1 or positions.numel() == 0:
                raise ValueError("fixed positions must be a non-empty vector")
            if int(positions[0]) < 0 or int(positions[-1]) >= self.prompt_length:
                raise ValueError("fixed position outside original prompt")
            if positions.numel() > 1 and not bool(
                torch.all(positions[1:] > positions[:-1])
            ):
                raise ValueError("fixed positions must be strictly increasing")
            checked[entry] = positions
        self._fixed = checked
        self._call_count = 0
        self._records = []
        self._active = True

    def finish_forward(self) -> list[FrozenBlockRecord]:
        if not self._active:
            raise RuntimeError("start_forward must be called first")
        self._active = False
        if self._call_count != self.num_entries:
            raise AssertionError(
                f"expected {self.num_entries} attention calls, got {self._call_count}"
            )
        if {record.entry for record in self._records} != set(self._fixed):
            raise AssertionError("intervened entries do not match frozen policy")
        return list(self._records)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._active:
            return self.original_attention(*args, **kwargs)
        if len(args) < 5:
            raise TypeError("expected positional query, key, value, mask, and q_len")
        entry = self._call_count
        self._call_count += 1
        fixed_cpu = self._fixed.get(entry)
        if fixed_cpu is None:
            return self.original_attention(*args, **kwargs)

        query_states, key_states, value_states, attention_mask, q_len = args[:5]
        if int(q_len) != 1 or int(query_states.shape[1]) != 1:
            raise AssertionError("frozen block intervention is decode-only")
        full_length = int(key_states.shape[1])
        if full_length <= self.prompt_length:
            raise AssertionError("policy must be activated only after Full Prefill")
        fixed = fixed_cpu.to(device=key_states.device)
        recent_start = max(0, full_length - self.recent_tokens)
        recent = torch.arange(
            recent_start, full_length, dtype=torch.long, device=key_states.device
        )
        if int(fixed[-1]) >= recent_start:
            positions = torch.unique(torch.cat((fixed, recent)), sorted=True)
        else:
            positions = torch.cat((fixed, recent))
        selected_key = key_states.index_select(1, positions)
        selected_value = value_states.index_select(1, positions)
        selected_mask = self._slice_attention_mask(
            attention_mask, positions, full_length
        )
        modified_args = (
            query_states,
            selected_key,
            selected_value,
            selected_mask,
            q_len,
            *args[5:],
        )
        self._records.append(
            FrozenBlockRecord(
                entry=entry,
                full_length=full_length,
                selected_length=int(positions.numel()),
                fixed_prompt_positions=int(fixed.numel()),
                dynamic_recent_positions=int(recent.numel()),
                selected_first=int(positions[0]),
                selected_last=int(positions[-1]),
            )
        )
        return self.original_attention(*modified_args, **kwargs)
