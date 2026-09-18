"""Experimental per-KV-head short-window intervention.

This module deliberately wraps the per-head FlashAttention call instead of
changing cache storage.  It is intended for quality/sensitivity experiments:
the full cache remains resident and only the K/V tensors visible to one
attention call are sliced.  Therefore results from this module must not be
reported as H2D or latency improvements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Collection, Optional

import torch


@dataclass(frozen=True)
class ShortWindowRecord:
    target_entry: int
    full_length: int
    selected_length: int
    sink_tokens: int
    recent_tokens: int
    selected_positions_first: int
    selected_positions_last: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class SingleHeadShortWindowController:
    """Apply sink+recent K/V selection to one global KV-head entry.

    HeadInfer calls ``_flash_attention_forward`` once per KV head in global
    layer-major order.  ``start_forward`` resets that order for one model
    forward, and ``finish_forward`` validates both the call count and that the
    requested intervention occurred exactly once.
    """

    def __init__(
        self,
        original_attention: Callable[..., Any],
        *,
        num_entries: int,
        sink_tokens: int,
        recent_tokens: int,
    ) -> None:
        if num_entries <= 0:
            raise ValueError("num_entries must be positive")
        if sink_tokens < 0 or recent_tokens < 0:
            raise ValueError("sink_tokens and recent_tokens must be non-negative")
        if sink_tokens + recent_tokens <= 0:
            raise ValueError("at least one sink or recent token is required")
        self.original_attention = original_attention
        self.num_entries = num_entries
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self._target_entry: Optional[int] = None
        self._call_count = 0
        self._records: list[ShortWindowRecord] = []
        self._active = False

    def start_forward(self, target_entry: Optional[int]) -> None:
        if self._active:
            raise RuntimeError("previous forward has not been finished")
        if target_entry is not None and not 0 <= target_entry < self.num_entries:
            raise ValueError(
                f"target_entry must be in [0, {self.num_entries}), got {target_entry}"
            )
        self._target_entry = target_entry
        self._call_count = 0
        self._records = []
        self._active = True

    def finish_forward(self) -> Optional[ShortWindowRecord]:
        if not self._active:
            raise RuntimeError("start_forward must be called first")
        self._active = False
        if self._call_count != self.num_entries:
            raise AssertionError(
                f"expected {self.num_entries} attention calls, got {self._call_count}"
            )
        expected_records = 0 if self._target_entry is None else 1
        if len(self._records) != expected_records:
            raise AssertionError(
                f"expected {expected_records} interventions, got {len(self._records)}"
            )
        return self._records[0] if self._records else None

    @staticmethod
    def _slice_attention_mask(
        attention_mask: Optional[torch.Tensor],
        positions: torch.Tensor,
        full_length: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.shape[-1] != full_length:
            raise AssertionError(
                "attention mask key length does not match K/V length: "
                f"{attention_mask.shape[-1]} != {full_length}"
            )
        return attention_mask.index_select(-1, positions)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._active:
            # Real Prefill is intentionally Full KV and runs without an active
            # intervention.  Passing it through also keeps the wrapper scoped
            # to the measured q_len=1 forward.
            return self.original_attention(*args, **kwargs)
        if len(args) < 5:
            raise TypeError("expected positional query, key, value, mask, and q_len")

        entry = self._call_count
        self._call_count += 1
        if entry != self._target_entry:
            return self.original_attention(*args, **kwargs)

        query_states, key_states, value_states, attention_mask, q_len = args[:5]
        if int(q_len) != 1 or query_states.shape[1] != 1:
            raise AssertionError(
                "short-window sensitivity intervention is decode-only (q_len=1)"
            )
        if key_states.shape[1] != value_states.shape[1]:
            raise AssertionError("K/V sequence lengths differ")

        full_length = int(key_states.shape[1])
        sink_end = min(self.sink_tokens, full_length)
        recent_start = max(sink_end, full_length - self.recent_tokens)
        sink_positions = torch.arange(
            0, sink_end, dtype=torch.long, device=key_states.device
        )
        recent_positions = torch.arange(
            recent_start, full_length, dtype=torch.long, device=key_states.device
        )
        positions = torch.cat((sink_positions, recent_positions))
        if positions.numel() == 0:
            raise AssertionError("short-window selected no K/V positions")

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
            ShortWindowRecord(
                target_entry=entry,
                full_length=full_length,
                selected_length=int(positions.numel()),
                sink_tokens=self.sink_tokens,
                recent_tokens=self.recent_tokens,
                selected_positions_first=int(positions[0].item()),
                selected_positions_last=int(positions[-1].item()),
            )
        )
        return self.original_attention(*modified_args, **kwargs)


class HeadSetShortWindowController(SingleHeadShortWindowController):
    """Apply the same short-window action to a validated set of KV entries."""

    def __init__(
        self,
        original_attention: Callable[..., Any],
        *,
        num_entries: int,
        sink_tokens: int,
        recent_tokens: int,
    ) -> None:
        super().__init__(
            original_attention,
            num_entries=num_entries,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
        )
        self._target_entries: frozenset[int] = frozenset()

    def start_forward(self, target_entries: Collection[int]) -> None:
        if self._active:
            raise RuntimeError("previous forward has not been finished")
        entries = list(target_entries)
        if len(set(entries)) != len(entries):
            raise ValueError("target_entries contains duplicates")
        if any(not 0 <= entry < self.num_entries for entry in entries):
            raise ValueError(f"target entries must be in [0, {self.num_entries})")
        self._target_entries = frozenset(entries)
        self._target_entry = None
        self._call_count = 0
        self._records = []
        self._active = True

    def finish_forward(self) -> list[ShortWindowRecord]:
        if not self._active:
            raise RuntimeError("start_forward must be called first")
        self._active = False
        if self._call_count != self.num_entries:
            raise AssertionError(
                f"expected {self.num_entries} attention calls, got {self._call_count}"
            )
        if len(self._records) != len(self._target_entries):
            raise AssertionError(
                f"expected {len(self._target_entries)} interventions, "
                f"got {len(self._records)}"
            )
        recorded_entries = {record.target_entry for record in self._records}
        if recorded_entries != set(self._target_entries):
            raise AssertionError(
                f"intervened entries {recorded_entries} != targets "
                f"{set(self._target_entries)}"
            )
        self._target_entry = None
        return list(self._records)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._active:
            entry = self._call_count
            self._target_entry = (
                entry if entry in self._target_entries else None
            )
        return super().__call__(*args, **kwargs)
