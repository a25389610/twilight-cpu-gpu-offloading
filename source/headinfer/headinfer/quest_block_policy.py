"""Quality-only Quest-style decode block selection for HeadInfer experiments.

The implementation mirrors Quest's sign-selected per-dimension page maximum:
for positive Query dimensions it uses the page Key maximum, and for negative
dimensions the page Key minimum.  It deliberately keeps the Full cache
resident and slices attention inputs, so its timings are selector overheads,
not an optimized Quest/Fluxion TPOT measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class QuestHeadRecord:
    entry: int
    query_head_in_group: int
    full_length: int
    eligible_history_tokens: int
    candidate_blocks: int
    selected_blocks: int
    selected_history_tokens: int
    visible_tokens: int
    selected_block_indices: tuple[int, ...]
    selection_refreshed: bool
    selection_age: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quest_page_metadata(
    keys: torch.Tensor,
    *,
    sink_tokens: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Build dimension-wise Key min/max metadata for prompt pages."""
    if keys.ndim != 2:
        raise ValueError("keys must have shape [tokens, head_dim]")
    if sink_tokens < 0 or block_size <= 0 or sink_tokens >= int(keys.shape[0]):
        raise ValueError("invalid sink_tokens or block_size")
    bounds = [
        (start, min(start + block_size, int(keys.shape[0])))
        for start in range(sink_tokens, int(keys.shape[0]), block_size)
    ]
    minima = []
    maxima = []
    for start, stop in bounds:
        page = keys[start:stop]
        minima.append(page.amin(dim=0))
        maxima.append(page.amax(dim=0))
    return torch.stack(minima), torch.stack(maxima), bounds


def quest_page_scores(
    query: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
) -> torch.Tensor:
    """Return Quest's upper-bound Query-Key dot-product score per page."""
    if query.ndim != 1 or page_min.ndim != 2 or page_min.shape != page_max.shape:
        raise ValueError("invalid Query or page metadata shape")
    if int(query.shape[0]) != int(page_min.shape[1]):
        raise ValueError("Query and metadata dimensions differ")
    extremum = torch.where(query.unsqueeze(0) > 0, page_max, page_min)
    return (extremum.float() * query.float().unsqueeze(0)).sum(dim=-1)


class QuestFixedBudgetController:
    """Apply per-Query-head Quest Top-K pages during decode-only attention."""

    def __init__(
        self,
        original_attention: Callable[..., Any],
        *,
        base_keys: list[torch.Tensor],
        prompt_length: int,
        sink_tokens: int,
        recent_tokens: int,
        block_size: int,
        budget_fraction: float,
        selection_interval: int = 1,
    ) -> None:
        if not 0.0 < budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be in (0, 1]")
        if selection_interval <= 0:
            raise ValueError("selection_interval must be positive")
        self.original_attention = original_attention
        self.num_entries = len(base_keys)
        self.prompt_length = int(prompt_length)
        self.sink_tokens = int(sink_tokens)
        self.recent_tokens = int(recent_tokens)
        self.block_size = int(block_size)
        self.budget_fraction = float(budget_fraction)
        self.selection_interval = int(selection_interval)
        self._metadata = []
        for cached in base_keys:
            key = cached.squeeze(0).squeeze(0)
            self._metadata.append(
                quest_page_metadata(
                    key,
                    sink_tokens=self.sink_tokens,
                    block_size=self.block_size,
                )
            )
        self._active = False
        self._call_count = 0
        self._records: list[QuestHeadRecord] = []
        self._forward_index = 0
        self._refresh_this_forward = True
        self._selected_page_cache: list[Optional[tuple[torch.Tensor, ...]]] = [
            None
        ] * self.num_entries

    @staticmethod
    def _slice_mask(
        attention_mask: Optional[torch.Tensor],
        positions: torch.Tensor,
        full_length: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if int(attention_mask.shape[-1]) != full_length:
            raise AssertionError("attention mask and K/V lengths differ")
        return attention_mask.index_select(-1, positions)

    def start_forward(self) -> None:
        if self._active:
            raise RuntimeError("previous forward has not finished")
        self._active = True
        self._call_count = 0
        self._records = []
        self._refresh_this_forward = self._forward_index % self.selection_interval == 0

    def finish_forward(self) -> list[QuestHeadRecord]:
        if not self._active:
            raise RuntimeError("start_forward must be called first")
        self._active = False
        if self._call_count != self.num_entries:
            raise AssertionError(
                f"expected {self.num_entries} attention calls, got {self._call_count}"
            )
        records = list(self._records)
        self._forward_index += 1
        return records

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._active:
            return self.original_attention(*args, **kwargs)
        if len(args) < 5:
            raise TypeError("expected positional query, key, value, mask, q_len")
        entry = self._call_count
        self._call_count += 1
        query, key, value, attention_mask, q_len = args[:5]
        if int(q_len) != 1 or int(query.shape[1]) != 1:
            raise AssertionError("Quest controller is decode-only")
        if int(key.shape[0]) != 1 or int(key.shape[2]) != 1:
            raise AssertionError("expected one batch and one KV head per entry")
        full_length = int(key.shape[1])
        if full_length <= self.prompt_length:
            raise AssertionError("controller must run after Full Prefill")

        page_min, page_max, bounds = self._metadata[entry]
        recent_start = max(self.sink_tokens, full_length - self.recent_tokens)
        eligible = [index for index, (_, stop) in enumerate(bounds) if stop <= recent_start]
        eligible_tokens = max(0, recent_start - self.sink_tokens)
        page_budget = min(
            len(eligible),
            max(1, math.ceil(self.budget_fraction * eligible_tokens / self.block_size)),
        )
        sink = torch.arange(0, min(self.sink_tokens, full_length), device=key.device)
        recent = torch.arange(recent_start, full_length, device=key.device)
        outputs = []
        cached_for_entry = self._selected_page_cache[entry]
        refreshed_for_entry: list[torch.Tensor] = []
        for group_head in range(int(query.shape[2])):
            if self._refresh_this_forward or cached_for_entry is None:
                q = query[0, 0, group_head]
                eligible_tensor = torch.tensor(eligible, dtype=torch.long, device=key.device)
                scores = quest_page_scores(
                    q,
                    page_min.index_select(0, eligible_tensor),
                    page_max.index_select(0, eligible_tensor),
                )
                ranked_local = torch.topk(scores, k=page_budget, largest=True, sorted=True).indices
                chosen = eligible_tensor.index_select(0, ranked_local)
                refreshed_for_entry.append(chosen)
            else:
                chosen = cached_for_entry[group_head]
            chosen_sorted = chosen.sort().values
            pieces = [sink]
            for block_index in chosen_sorted.tolist():
                start, stop = bounds[block_index]
                pieces.append(torch.arange(start, stop, device=key.device))
            pieces.append(recent)
            positions = torch.unique(torch.cat(pieces), sorted=True).to(torch.long)
            selected_key = key.index_select(1, positions)
            selected_value = value.index_select(1, positions)
            selected_mask = self._slice_mask(attention_mask, positions, full_length)
            modified = (
                query[:, :, group_head : group_head + 1],
                selected_key,
                selected_value,
                selected_mask,
                q_len,
                *args[5:],
            )
            outputs.append(self.original_attention(*modified, **kwargs))
            selected_history_tokens = sum(
                bounds[index][1] - bounds[index][0] for index in chosen.tolist()
            )
            self._records.append(
                QuestHeadRecord(
                    entry=entry,
                    query_head_in_group=group_head,
                    full_length=full_length,
                    eligible_history_tokens=eligible_tokens,
                    candidate_blocks=len(eligible),
                    selected_blocks=page_budget,
                    selected_history_tokens=selected_history_tokens,
                    visible_tokens=int(positions.numel()),
                    selected_block_indices=tuple(int(x) for x in chosen.tolist()),
                    selection_refreshed=(self._refresh_this_forward or cached_for_entry is None),
                    selection_age=self._forward_index % self.selection_interval,
                )
            )
        if self._refresh_this_forward or cached_for_entry is None:
            self._selected_page_cache[entry] = tuple(refreshed_for_entry)
        return torch.cat(outputs, dim=2)
