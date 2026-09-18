"""Diagnostic-only Full/Short attention-output feature capture.

The wrapper observes the 224 per-KV-entry decode attention calls emitted by
``mp_headinfer``.  For each entry it computes both the ordinary Full output and
the exact sink+recent Short output, records compact local difference features,
and returns the Full output unchanged.  It is an expensive Offline Teacher
feature extractor, not a deployable selector or Partial-H2D implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class FullShortOutputFeatureRecord:
    global_entry: int
    layer_id: int
    kv_head_id: int
    query_head_start: int
    query_head_end_exclusive: int
    full_length: int
    short_length: int
    output_delta_l2: float
    output_delta_mean_abs: float
    output_delta_max_abs: float
    full_output_l2: float
    short_output_l2: float
    relative_output_delta_l2: float
    full_short_cosine: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class FullShortOutputFeatureWrapper:
    """Capture compact Full-vs-Short output features while returning Full."""

    def __init__(
        self,
        original_attention: Callable[..., torch.Tensor],
        *,
        num_layers: int,
        num_kv_heads: int,
        query_heads_per_kv: int,
        head_dim: int,
        sink_tokens: int,
        recent_tokens: int,
    ) -> None:
        if min(num_layers, num_kv_heads, query_heads_per_kv, head_dim) <= 0:
            raise ValueError("model geometry values must be positive")
        if sink_tokens < 0 or recent_tokens < 0:
            raise ValueError("sink/recent sizes must be non-negative")
        if sink_tokens + recent_tokens <= 0:
            raise ValueError("Short window must retain at least one token")
        self.original_attention = original_attention
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.query_heads_per_kv = int(query_heads_per_kv)
        self.head_dim = int(head_dim)
        self.sink_tokens = int(sink_tokens)
        self.recent_tokens = int(recent_tokens)
        self.num_entries = self.num_layers * self.num_kv_heads
        self._active = False
        self._call_count = 0
        self._metadata: list[tuple[int, int]] = []
        self._features: list[torch.Tensor] = []

    @staticmethod
    def _slice_attention_mask(
        attention_mask: Optional[torch.Tensor],
        positions: torch.Tensor,
        full_length: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.shape[-1] != full_length:
            raise AssertionError("attention mask key length differs from K/V")
        return attention_mask.index_select(-1, positions)

    def _short_positions(self, key_states: torch.Tensor) -> torch.Tensor:
        full_length = int(key_states.shape[1])
        sink_end = min(self.sink_tokens, full_length)
        recent_start = max(sink_end, full_length - self.recent_tokens)
        sink = torch.arange(0, sink_end, device=key_states.device)
        recent = torch.arange(recent_start, full_length, device=key_states.device)
        positions = torch.cat((sink, recent)).to(torch.long)
        if positions.numel() == 0:
            raise AssertionError("Short policy selected no K/V positions")
        return positions

    def start_forward(self) -> None:
        if self._active:
            raise RuntimeError("previous feature forward has not been finished")
        self._active = True
        self._call_count = 0
        self._metadata = []
        self._features = []

    def abort_forward(self) -> None:
        self._active = False
        self._call_count = 0
        self._metadata = []
        self._features = []

    def finish_forward(self) -> list[FullShortOutputFeatureRecord]:
        if not self._active:
            raise RuntimeError("start_forward must be called first")
        self._active = False
        if self._call_count != self.num_entries:
            raise AssertionError(
                f"expected {self.num_entries} attention calls, got {self._call_count}"
            )
        if len(self._features) != self.num_entries:
            raise AssertionError("feature count differs from model geometry")
        values = torch.stack(self._features).float().cpu()
        records: list[FullShortOutputFeatureRecord] = []
        for entry, ((full_length, short_length), row) in enumerate(
            zip(self._metadata, values)
        ):
            layer_id, kv_head_id = divmod(entry, self.num_kv_heads)
            query_start = kv_head_id * self.query_heads_per_kv
            records.append(
                FullShortOutputFeatureRecord(
                    global_entry=entry,
                    layer_id=layer_id,
                    kv_head_id=kv_head_id,
                    query_head_start=query_start,
                    query_head_end_exclusive=query_start + self.query_heads_per_kv,
                    full_length=full_length,
                    short_length=short_length,
                    output_delta_l2=float(row[0]),
                    output_delta_mean_abs=float(row[1]),
                    output_delta_max_abs=float(row[2]),
                    full_output_l2=float(row[3]),
                    short_output_l2=float(row[4]),
                    relative_output_delta_l2=float(row[5]),
                    full_short_cosine=float(row[6]),
                )
            )
        self._metadata = []
        self._features = []
        return records

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self._active:
            return self.original_attention(*args, **kwargs)
        if len(args) < 5:
            raise TypeError("expected positional query, key, value, mask, and q_len")
        if self._call_count >= self.num_entries:
            raise AssertionError("received too many attention calls")

        query_states, key_states, value_states, attention_mask, q_len = args[:5]
        if int(q_len) != 1 or query_states.shape[1] != 1:
            raise AssertionError("feature capture is decode-only (q_len=1)")
        if query_states.ndim != 4 or tuple(query_states.shape[2:]) != (
            self.query_heads_per_kv,
            self.head_dim,
        ):
            raise AssertionError(f"unexpected Query geometry {tuple(query_states.shape)}")
        if key_states.shape != value_states.shape or key_states.shape[2:] != (
            1,
            self.head_dim,
        ):
            raise AssertionError("unexpected per-entry K/V geometry")

        positions = self._short_positions(key_states)
        selected_key = key_states.index_select(1, positions)
        selected_value = value_states.index_select(1, positions)
        selected_mask = self._slice_attention_mask(
            attention_mask, positions, int(key_states.shape[1])
        )
        full_output = self.original_attention(*args, **kwargs)
        short_args = (
            query_states,
            selected_key,
            selected_value,
            selected_mask,
            q_len,
            *args[5:],
        )
        short_output = self.original_attention(*short_args, **kwargs)
        if full_output.shape != short_output.shape:
            raise AssertionError("Full/Short attention output shapes differ")

        full = full_output.detach().float()
        short = short_output.detach().float()
        delta = full - short
        full_l2 = torch.linalg.vector_norm(full)
        short_l2 = torch.linalg.vector_norm(short)
        delta_l2 = torch.linalg.vector_norm(delta)
        cosine = torch.sum(full * short) / torch.clamp(full_l2 * short_l2, min=1e-12)
        feature = torch.stack(
            (
                delta_l2,
                delta.abs().mean(),
                delta.abs().max(),
                full_l2,
                short_l2,
                delta_l2 / torch.clamp(full_l2, min=1e-12),
                cosine,
            )
        ).detach()
        if not torch.isfinite(feature).all():
            raise AssertionError("non-finite Full/Short output feature")
        self._metadata.append((int(key_states.shape[1]), int(positions.numel())))
        self._features.append(feature)
        self._call_count += 1
        return full_output
