"""Differentiable decode-only Full/Short attention mixing for diagnostics.

This module is intentionally isolated from the Partial-H2D runtime.  It wraps
the per-KV-entry FlashAttention calls made by ``mp_headinfer`` and, only while
an explicit diagnostic forward is active, computes both the Full and exact
sink+recent outputs before applying one FP32 gate per global KV entry.

It is an Offline Teacher / Oracle mechanism, not a deployable selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import torch
from torch import nn


class _ExactHardMaskSTE(torch.autograd.Function):
    """Return the hard mask bitwise in forward and identity gradient backward."""

    @staticmethod
    def forward(ctx: Any, scores: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
        del ctx
        return hard

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        del ctx
        return grad_output, None


@dataclass(frozen=True)
class DecodeJointGateRecord:
    """Geometry observed for one global KV entry during a diagnostic forward."""

    global_entry: int
    layer_id: int
    kv_head_id: int
    query_head_start: int
    query_head_end_exclusive: int
    full_length: int
    short_length: int
    sink_tokens: int
    recent_tokens: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class DecodeJointGateWrapper(nn.Module):
    """Compute Full and exact Short attention and mix them with 224 soft gates.

    ``mp_headinfer`` calls the wrapped attention function in layer-major,
    KV-head-major order.  Gate ``entry`` therefore follows
    ``entry = layer * num_kv_heads + kv_head``.  One gate is shared by the
    contiguous GQA Query-Head group passed in that attention call.
    """

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
        device: torch.device | str,
        initial_value: float = 1.0,
    ) -> None:
        super().__init__()
        if min(num_layers, num_kv_heads, query_heads_per_kv, head_dim) <= 0:
            raise ValueError("model geometry values must be positive")
        if sink_tokens < 0 or recent_tokens < 0:
            raise ValueError("sink/recent sizes must be non-negative")
        if sink_tokens + recent_tokens <= 0:
            raise ValueError("Short window must retain at least one token")
        if not 0.0 <= initial_value <= 1.0:
            raise ValueError("initial gate value must be in [0, 1]")

        self.original_attention = original_attention
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.query_heads_per_kv = int(query_heads_per_kv)
        self.head_dim = int(head_dim)
        self.sink_tokens = int(sink_tokens)
        self.recent_tokens = int(recent_tokens)
        self.num_entries = self.num_layers * self.num_kv_heads
        self.gates = nn.Parameter(
            torch.full(
                (self.num_entries,),
                float(initial_value),
                dtype=torch.float32,
                device=device,
            )
        )
        self._active = False
        self._call_count = 0
        self._records: list[DecodeJointGateRecord] = []
        self._hard_topk_ste_full_count: int | None = None

    @property
    def active(self) -> bool:
        return self._active

    @torch.no_grad()
    def set_all_gates(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("gate value must be in [0, 1]")
        self.gates.fill_(float(value))

    @torch.no_grad()
    def clamp_gates_(self) -> None:
        self.gates.clamp_(0.0, 1.0)

    def bounded_gate_scores(self) -> torch.Tensor:
        """Return the bounded FP32 scores before optional hard selection."""

        return self.gates.clamp(0.0, 1.0)

    def enable_hard_topk_ste(self, full_head_count: int) -> None:
        """Use an exact hard Top-k mask in forward with identity STE backward."""

        if not 0 < full_head_count < self.num_entries:
            raise ValueError("full_head_count must be in (0, num_entries)")
        self._hard_topk_ste_full_count = int(full_head_count)

    def disable_hard_topk_ste(self) -> None:
        self._hard_topk_ste_full_count = None

    def hard_topk_mask(self) -> torch.Tensor:
        """Return the deterministic binary mask selected from detached scores."""

        if self._hard_topk_ste_full_count is None:
            raise RuntimeError("hard Top-k STE mode is disabled")
        scores = self.bounded_gate_scores()
        order = torch.argsort(scores.detach(), descending=True, stable=True)
        hard = torch.zeros_like(scores)
        hard[order[: self._hard_topk_ste_full_count]] = 1.0
        return hard

    def gate_values(self) -> torch.Tensor:
        """Return gates used by forward, optionally hard with an STE gradient."""

        scores = self.bounded_gate_scores()
        if self._hard_topk_ste_full_count is None:
            return scores
        hard = self.hard_topk_mask()
        return _ExactHardMaskSTE.apply(scores, hard)

    def regularization_loss(self) -> torch.Tensor:
        """DuoAttention-style mean L1 regularization over bounded gates."""

        return self.bounded_gate_scores().abs().mean()

    def mapping_for_entry(self, entry: int) -> tuple[int, int, int, int]:
        if not 0 <= entry < self.num_entries:
            raise ValueError(f"entry must be in [0, {self.num_entries})")
        layer_id, kv_head_id = divmod(entry, self.num_kv_heads)
        query_start = kv_head_id * self.query_heads_per_kv
        return (
            layer_id,
            kv_head_id,
            query_start,
            query_start + self.query_heads_per_kv,
        )

    def start_forward(self) -> None:
        if self._active:
            raise RuntimeError("previous joint-gate forward has not been finished")
        self._active = True
        self._call_count = 0
        self._records = []

    def abort_forward(self) -> None:
        self._active = False
        self._call_count = 0
        self._records = []

    def finish_forward(self) -> list[DecodeJointGateRecord]:
        if not self._active:
            raise RuntimeError("start_forward must be called first")
        self._active = False
        if self._call_count != self.num_entries:
            count = self._call_count
            self._records = []
            raise AssertionError(
                f"expected {self.num_entries} attention calls, got {count}"
            )
        records = list(self._records)
        self._records = []
        return records

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

    def _short_positions(self, key_states: torch.Tensor) -> torch.Tensor:
        full_length = int(key_states.shape[1])
        sink_end = min(self.sink_tokens, full_length)
        recent_start = max(sink_end, full_length - self.recent_tokens)
        sink = torch.arange(
            0, sink_end, dtype=torch.long, device=key_states.device
        )
        recent = torch.arange(
            recent_start, full_length, dtype=torch.long, device=key_states.device
        )
        positions = torch.cat((sink, recent))
        if positions.numel() == 0:
            raise AssertionError("Short policy selected no K/V positions")
        return positions

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self._active:
            return self.original_attention(*args, **kwargs)
        if len(args) < 5:
            raise TypeError("expected positional query, key, value, mask, and q_len")
        if self._call_count >= self.num_entries:
            raise AssertionError(
                f"received more than {self.num_entries} attention calls"
            )

        query_states, key_states, value_states, attention_mask, q_len = args[:5]
        if int(q_len) != 1 or query_states.shape[1] != 1:
            raise AssertionError("joint-gate diagnostic is decode-only (q_len=1)")
        if query_states.ndim != 4 or tuple(query_states.shape[2:]) != (
            self.query_heads_per_kv,
            self.head_dim,
        ):
            raise AssertionError(
                "unexpected per-KV Query geometry: "
                f"{tuple(query_states.shape)}"
            )
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise AssertionError("expected rank-4 K/V tensors")
        if key_states.shape != value_states.shape:
            raise AssertionError("K/V shapes differ")
        if key_states.shape[2:] != (1, self.head_dim):
            raise AssertionError(
                f"unexpected per-entry K/V geometry: {tuple(key_states.shape)}"
            )

        entry = self._call_count
        self._call_count += 1
        full_length = int(key_states.shape[1])
        positions = self._short_positions(key_states)
        selected_key = key_states.index_select(1, positions)
        selected_value = value_states.index_select(1, positions)
        selected_mask = self._slice_attention_mask(
            attention_mask, positions, full_length
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

        alpha = self.gate_values()[entry].to(dtype=full_output.dtype)
        mixed_output = (1 - alpha) * short_output + alpha * full_output

        layer_id, kv_head_id, query_start, query_end = self.mapping_for_entry(entry)
        self._records.append(
            DecodeJointGateRecord(
                global_entry=entry,
                layer_id=layer_id,
                kv_head_id=kv_head_id,
                query_head_start=query_start,
                query_head_end_exclusive=query_end,
                full_length=full_length,
                short_length=int(positions.numel()),
                sink_tokens=self.sink_tokens,
                recent_tokens=self.recent_tokens,
            )
        )
        return mixed_output
