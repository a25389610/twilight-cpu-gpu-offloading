"""Baseline-only Query capture for Offline dataset construction.

The wrapper is deliberately independent of ``mp.py`` and is a strict
pass-through unless ``start_capture`` has been called.  It captures the last
query position after the wrapped FlashAttention call has completed, so the
wrapped call receives exactly the original arguments and return path.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch


class BaselineQueryCaptureWrapper:
    """Capture one last-position Query tensor for every global KV entry."""

    def __init__(
        self,
        original_attention: Callable[..., Any],
        *,
        num_entries: int,
        query_heads_per_kv: int,
        head_dim: int,
        storage_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        if num_entries <= 0:
            raise ValueError("num_entries must be positive")
        if query_heads_per_kv <= 0:
            raise ValueError("query_heads_per_kv must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.original_attention = original_attention
        self.num_entries = int(num_entries)
        self.query_heads_per_kv = int(query_heads_per_kv)
        self.head_dim = int(head_dim)
        self.storage_dtype = storage_dtype
        self._active = False
        self._captured: list[torch.Tensor] = []

    @property
    def active(self) -> bool:
        return self._active

    def start_capture(self) -> None:
        if self._active:
            raise RuntimeError("previous Query capture has not been finished")
        self._captured = []
        self._active = True

    def abort_capture(self) -> None:
        self._active = False
        self._captured = []

    def finish_capture(self) -> torch.Tensor:
        if not self._active:
            raise RuntimeError("start_capture must be called first")
        self._active = False
        if len(self._captured) != self.num_entries:
            count = len(self._captured)
            self._captured = []
            raise AssertionError(
                f"expected {self.num_entries} Query entries, got {count}"
            )
        result = torch.stack(
            [query.squeeze(0) for query in self._captured], dim=0
        ).cpu()
        self._captured = []
        expected = (
            self.num_entries,
            self.query_heads_per_kv,
            self.head_dim,
        )
        if tuple(result.shape) != expected:
            raise AssertionError(
                f"captured Query shape {tuple(result.shape)} != {expected}"
            )
        if not torch.isfinite(result).all():
            raise AssertionError("captured Query contains non-finite values")
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.original_attention(*args, **kwargs)
        if not self._active:
            return result
        if not args:
            raise TypeError("expected query_states as the first positional argument")
        query_states = args[0]
        if not isinstance(query_states, torch.Tensor):
            raise TypeError("query_states must be a torch.Tensor")
        if query_states.ndim != 4:
            raise AssertionError(
                "expected query_states [batch, q_len, query_heads, head_dim], "
                f"got {tuple(query_states.shape)}"
            )
        if query_states.shape[0] != 1:
            raise AssertionError("Offline dataset v1 requires batch size 1")
        if query_states.shape[1] <= 0:
            raise AssertionError("query_states has no sequence positions")
        if query_states.shape[2] != self.query_heads_per_kv:
            raise AssertionError(
                f"query-head count {query_states.shape[2]} != "
                f"{self.query_heads_per_kv}"
            )
        if query_states.shape[3] != self.head_dim:
            raise AssertionError(
                f"Query head dim {query_states.shape[3]} != {self.head_dim}"
            )
        if len(self._captured) >= self.num_entries:
            raise AssertionError(
                f"received more than {self.num_entries} Query entries"
            )
        captured = query_states[:, -1].detach()
        if self.storage_dtype is not None:
            captured = captured.to(dtype=self.storage_dtype)
        self._captured.append(captured.clone())
        return result


class BaselineQueryWindowCaptureWrapper(BaselineQueryCaptureWrapper):
    """Capture the last ``query_positions`` RoPE-applied Queries per entry.

    HeadInfer invokes the wrapped FlashAttention function once for every KV
    entry in layer-major order.  The first argument is already post-RoPE and
    has shape ``[batch, q_len, query_heads_per_kv, head_dim]``.  This wrapper
    remains a strict pass-through and only clones the requested suffix after
    the original attention call returns.
    """

    def __init__(
        self,
        original_attention: Callable[..., Any],
        *,
        num_entries: int,
        query_heads_per_kv: int,
        head_dim: int,
        query_positions: int,
        storage_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> None:
        super().__init__(
            original_attention,
            num_entries=num_entries,
            query_heads_per_kv=query_heads_per_kv,
            head_dim=head_dim,
            storage_dtype=storage_dtype,
        )
        if query_positions <= 0:
            raise ValueError("query_positions must be positive")
        self.query_positions = int(query_positions)

    def finish_capture(self, *, to_cpu: bool = True) -> torch.Tensor:
        if not self._active:
            raise RuntimeError("start_capture must be called first")
        self._active = False
        if len(self._captured) != self.num_entries:
            count = len(self._captured)
            self._captured = []
            raise AssertionError(
                f"expected {self.num_entries} Query entries, got {count}"
            )
        result = torch.stack(
            [query.squeeze(0) for query in self._captured], dim=0
        )
        self._captured = []
        expected = (
            self.num_entries,
            self.query_positions,
            self.query_heads_per_kv,
            self.head_dim,
        )
        if tuple(result.shape) != expected:
            raise AssertionError(
                f"captured Query shape {tuple(result.shape)} != {expected}"
            )
        if not torch.isfinite(result).all():
            raise AssertionError("captured Query contains non-finite values")
        return result.cpu() if to_cpu else result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.original_attention(*args, **kwargs)
        if not self._active:
            return result
        if not args:
            raise TypeError("expected query_states as the first positional argument")
        query_states = args[0]
        if not isinstance(query_states, torch.Tensor) or query_states.ndim != 4:
            raise AssertionError(
                "expected query_states [batch, q_len, query_heads, head_dim]"
            )
        expected_tail = (1, self.query_heads_per_kv, self.head_dim)
        if query_states.shape[0] != expected_tail[0]:
            raise AssertionError("attention scoring requires batch size 1")
        if tuple(query_states.shape[2:]) != expected_tail[1:]:
            raise AssertionError(
                f"unexpected Query geometry {tuple(query_states.shape)}"
            )
        if query_states.shape[1] < self.query_positions:
            raise AssertionError(
                f"q_len {query_states.shape[1]} < capture window {self.query_positions}"
            )
        if len(self._captured) >= self.num_entries:
            raise AssertionError(
                f"received more than {self.num_entries} Query entries"
            )
        captured = query_states[:, -self.query_positions :].detach()
        if self.storage_dtype is not None:
            captured = captured.to(dtype=self.storage_dtype)
        self._captured.append(captured.clone())
        return result
