"""Experimental pinned-slab cache for controlled HeadInfer offload ablations.

This module intentionally stays separate from the repaired reference cache.
It holds model, dtype, host allocation, attention grouping, and dependency
ordering constant while exposing only three experimental controls:

* ``writeback_mode``: full K/V D2H or appended-range-only D2H;
* ``transfer_group_size``: number of consecutive KV heads per transfer;
* ``activation_wait_mode``: CPU-blocking event synchronization or a
  non-blocking compute-stream event dependency.

The implementation is restricted to the sequential, batch-size-1 Llama path
installed by ``headinfer.mp.mp_headinfer``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache

from .slab_cache import SequentialCacheState, SlabGeometry


@dataclass
class _GroupSlotState:
    group_start: Optional[int] = None
    group_count: int = 0
    prefetched_length: int = 0
    prefetch_done: Optional[torch.cuda.Event] = None
    reusable_after: Optional[torch.cuda.Event] = None


class GroupedPinnedSlabOffloadedCache(Cache):
    """Fixed-capacity cache with controlled D2H, grouping, and wait policies."""

    def __init__(
        self,
        *,
        num_cache_entries: int,
        max_cache_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        batch_size: int = 1,
        transfer_group_size: int = 1,
        writeback_mode: str = "delta",
        activation_wait_mode: str = "event",
        layer_full_attention_execution: bool = False,
    ) -> None:
        super().__init__()
        device = torch.device(device)
        if batch_size != 1:
            raise ValueError("only batch_size=1 is supported")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("an operational CUDA device is required")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if transfer_group_size <= 0:
            raise ValueError("transfer_group_size must be positive")
        if num_cache_entries % transfer_group_size:
            raise ValueError(
                "num_cache_entries must be divisible by transfer_group_size"
            )
        if writeback_mode not in {"full", "delta"}:
            raise ValueError("writeback_mode must be 'full' or 'delta'")
        if activation_wait_mode not in {"blocking", "event"}:
            raise ValueError(
                "activation_wait_mode must be 'blocking' or 'event'"
            )

        self.geometry = SlabGeometry(
            num_entries=num_cache_entries,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            head_dim=head_dim,
        )
        self.dtype = dtype
        self.device = device
        self.allocated_capacity = max_cache_len
        self.transfer_group_size = transfer_group_size
        self.writeback_mode = writeback_mode
        self.activation_wait_mode = activation_wait_mode
        self.layer_full_attention_execution = bool(layer_full_attention_execution)

        self._host_slab = torch.empty(
            self.geometry.total_elements,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        group_shape = (
            transfer_group_size,
            batch_size,
            1,
            max_cache_len,
            head_dim,
        )
        self._slot_keys = [
            torch.empty(group_shape, dtype=dtype, device=device),
            torch.empty(group_shape, dtype=dtype, device=device),
        ]
        self._slot_values = [
            torch.empty(group_shape, dtype=dtype, device=device),
            torch.empty(group_shape, dtype=dtype, device=device),
        ]

        self.prefetch_stream = torch.cuda.Stream(device=device)
        self.eviction_stream = torch.cuda.Stream(device=device)
        self.audit_stream = torch.cuda.Stream(device=device)
        self._slots = [_GroupSlotState(), _GroupSlotState()]
        self._host_write_done: List[Optional[torch.cuda.Event]] = [
            None
        ] * num_cache_entries
        self._bookkeeping = SequentialCacheState(num_cache_entries, max_cache_len)
        self._logical_lengths = self._bookkeeping.logical_lengths
        self._initialized = [False] * num_cache_entries

        self._active_slot: Optional[int] = None
        self._active_group_start: Optional[int] = None
        self._active_group_count = 0
        self._active_updates = 0
        self._active_compute_stream: Optional[torch.cuda.Stream] = None
        self._active_write_start = 0
        self._active_write_end = 0
        self._seen_tokens = 0
        self.write_position = 0

        self.metrics_enabled = False
        self.metrics: Dict[str, float] = defaultdict(float)
        self._timing_events: List[
            Tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []

        self.audit_enabled = False
        self.audit: Dict[str, Any] = {}
        self._audit_expected_keys: List[torch.Tensor] = []
        self._audit_expected_values: List[torch.Tensor] = []
        self._audit_keys: Optional[torch.Tensor] = None
        self._audit_values: Optional[torch.Tensor] = None
        self._full_layer_host_keys: Optional[torch.Tensor] = None
        self._full_layer_host_values: Optional[torch.Tensor] = None
        self._full_layer_gpu_keys: Optional[torch.Tensor] = None
        self._full_layer_gpu_values: Optional[torch.Tensor] = None
        self._full_layer_host_write_done: List[Optional[torch.cuda.Event]] = []

    @classmethod
    def from_llama_model(
        cls,
        model: torch.nn.Module,
        *,
        max_cache_len: int,
        transfer_group_size: int = 1,
        writeback_mode: str = "delta",
        activation_wait_mode: str = "event",
        layer_full_attention_execution: bool = False,
    ) -> "GroupedPinnedSlabOffloadedCache":
        config = model.config
        if (
            layer_full_attention_execution
            and transfer_group_size != config.num_key_value_heads
        ):
            raise ValueError(
                "layer-level Full Attention requires one transfer group per "
                "decoder layer"
            )
        num_entries = config.num_hidden_layers * config.num_key_value_heads
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        parameter = next(model.parameters())
        return cls(
            num_cache_entries=num_entries,
            max_cache_len=max_cache_len,
            head_dim=head_dim,
            dtype=parameter.dtype,
            device=parameter.device,
            transfer_group_size=transfer_group_size,
            writeback_mode=writeback_mode,
            activation_wait_mode=activation_wait_mode,
            layer_full_attention_execution=layer_full_attention_execution,
        )

    def _new_event(self, *, timing: bool = False) -> torch.cuda.Event:
        return torch.cuda.Event(
            enable_timing=timing,
            blocking=False,
            interprocess=False,
        )

    def _host_tensor(self, kind: str, entry_idx: int) -> torch.Tensor:
        start, end = self.geometry.bounds(kind, entry_idx)
        return self._host_slab[start:end].view(self.geometry.tensor_shape())

    def _host_group_tensor(
        self, kind: str, group_start: int, group_count: int
    ) -> torch.Tensor:
        first, _ = self.geometry.bounds(kind, group_start)
        _, last = self.geometry.bounds(kind, group_start + group_count - 1)
        return self._host_slab[first:last].view(
            group_count,
            self.geometry.batch_size,
            1,
            self.geometry.max_cache_len,
            self.geometry.head_dim,
        )

    def _group_start(self, entry_idx: int) -> int:
        return (entry_idx // self.transfer_group_size) * self.transfer_group_size

    def _group_count(self, group_start: int) -> int:
        return min(
            self.transfer_group_size,
            self.geometry.num_entries - group_start,
        )

    def reset_metrics(self) -> None:
        self.metrics = defaultdict(float)
        self._timing_events = []

    def enable_metrics(self, enabled: bool = True) -> None:
        self.metrics_enabled = enabled
        if enabled:
            self.reset_metrics()

    def _record_wait(self, stream: torch.cuda.Stream, event: torch.cuda.Event) -> None:
        stream.wait_event(event)
        if self.metrics_enabled:
            self.metrics["cuda_event_wait_calls"] += 1

    def _record_transfer(
        self,
        *,
        name: str,
        stream: torch.cuda.Stream,
        operation,
        byte_count: int,
        copy_calls: int,
    ) -> None:
        if self.metrics_enabled:
            start = self._new_event(timing=True)
            end = self._new_event(timing=True)
            start.record(stream)
        operation()
        if self.metrics_enabled:
            end.record(stream)
            self._timing_events.append((name, start, end))
            self.metrics[f"{name}_bytes"] += byte_count
            self.metrics[f"{name}_copy_calls"] += copy_calls

    def resolve_metrics(self) -> Dict[str, float]:
        torch.cuda.synchronize(self.device)
        for name, start, end in self._timing_events:
            self.metrics[f"{name}_cuda_seconds"] += (
                start.elapsed_time(end) / 1000.0
            )
            self.metrics[f"{name}_event_intervals"] += 1
        self._timing_events = []
        return dict(self.metrics)

    def enable_history_audit(self) -> None:
        if self.transfer_group_size != 1:
            raise ValueError("history audit is intentionally restricted to group size 1")
        shape = (
            1,
            self.geometry.batch_size,
            1,
            self.geometry.max_cache_len,
            self.geometry.head_dim,
        )
        self._audit_keys = torch.empty(
            shape, dtype=self.dtype, device="cpu", pin_memory=True
        )
        self._audit_values = torch.empty(
            shape, dtype=self.dtype, device="cpu", pin_memory=True
        )
        self.audit_enabled = True
        self.audit = {
            "entries_checked": 0,
            "history_key_mismatch_elements": 0,
            "history_value_mismatch_elements": 0,
            "tail_key_mismatch_elements": 0,
            "tail_value_mismatch_elements": 0,
            "history_elements_compared_per_kind": 0,
            "tail_elements_compared_per_kind": 0,
            "append_lengths": [],
            "gpu_write_ranges": [],
            "d2h_source_ranges": [],
        }

    def _audit_active_group(self, compute_done: torch.cuda.Event) -> None:
        if not self.audit_enabled:
            return
        if self._audit_keys is None or self._audit_values is None:
            raise RuntimeError("audit buffers were not allocated")
        if self._active_slot is None or self._active_group_start is None:
            raise RuntimeError("no active group to audit")
        slot = self._active_slot
        group_start = self._active_group_start
        start = self._active_write_start
        end = self._active_write_end
        with torch.cuda.stream(self.audit_stream):
            self.audit_stream.wait_event(compute_done)
            self._audit_keys[..., :end, :].copy_(
                self._slot_keys[slot][..., :end, :],
                non_blocking=True,
            )
            self._audit_values[..., :end, :].copy_(
                self._slot_values[slot][..., :end, :],
                non_blocking=True,
            )
        self.audit_stream.synchronize()

        host_key = self._host_group_tensor("key", group_start, 1)
        host_value = self._host_group_tensor("value", group_start, 1)
        if start:
            key_diff = self._audit_keys[..., :start, :] != host_key[..., :start, :]
            value_diff = (
                self._audit_values[..., :start, :] != host_value[..., :start, :]
            )
            self.audit["history_key_mismatch_elements"] += int(key_diff.sum())
            self.audit["history_value_mismatch_elements"] += int(value_diff.sum())
        expected_key = torch.stack(self._audit_expected_keys, dim=0)
        expected_value = torch.stack(self._audit_expected_values, dim=0)
        actual_key = self._audit_keys[..., start:end, :]
        actual_value = self._audit_values[..., start:end, :]
        self.audit["tail_key_mismatch_elements"] += int(
            (actual_key != expected_key).sum()
        )
        self.audit["tail_value_mismatch_elements"] += int(
            (actual_value != expected_value).sum()
        )
        self.audit["entries_checked"] += 1
        self.audit["history_elements_compared_per_kind"] += start * self.geometry.head_dim
        self.audit["tail_elements_compared_per_kind"] += (
            (end - start) * self.geometry.head_dim
        )
        self.audit["append_lengths"].append(end - start)
        self.audit["gpu_write_ranges"].append([start, end])
        d2h_start = 0 if self.writeback_mode == "full" else start
        self.audit["d2h_source_ranges"].append([d2h_start, end])
        self._audit_expected_keys = []
        self._audit_expected_values = []

    def _retire_active_group(self) -> None:
        if self._active_slot is None or self._active_group_start is None:
            return
        if self._active_updates != self._active_group_count:
            raise RuntimeError("attempted to retire a partially updated group")
        if self._active_compute_stream is None:
            raise RuntimeError("active group has no owning compute stream")

        slot = self._active_slot
        group_start = self._active_group_start
        group_count = self._active_group_count
        main_done = self._new_event()
        main_done.record(self._active_compute_stream)
        self._audit_active_group(main_done)

        start = 0 if self.writeback_mode == "full" else self._active_write_start
        end = self._active_write_end
        element_size = torch.empty((), dtype=self.dtype).element_size()
        bytes_per_kind = (
            group_count
            * self.geometry.batch_size
            * (end - start)
            * self.geometry.head_dim
            * element_size
        )
        with torch.cuda.stream(self.eviction_stream):
            self._record_wait(self.eviction_stream, main_done)

            def copy_group_to_host() -> None:
                self._host_group_tensor(
                    "key", group_start, group_count
                )[..., start:end, :].copy_(
                    self._slot_keys[slot][:group_count, ..., start:end, :],
                    non_blocking=True,
                )
                self._host_group_tensor(
                    "value", group_start, group_count
                )[..., start:end, :].copy_(
                    self._slot_values[slot][:group_count, ..., start:end, :],
                    non_blocking=True,
                )

            self._record_transfer(
                name="d2h",
                stream=self.eviction_stream,
                operation=copy_group_to_host,
                byte_count=2 * bytes_per_kind,
                copy_calls=2,
            )
            host_done = self._new_event()
            host_done.record(self.eviction_stream)

        for entry_idx in range(group_start, group_start + group_count):
            self._host_write_done[entry_idx] = host_done
            self._initialized[entry_idx] = True
        self._slots[slot].reusable_after = host_done
        if self.metrics_enabled:
            self.metrics["evicted_groups"] += 1

        self._active_slot = None
        self._active_group_start = None
        self._active_group_count = 0
        self._active_updates = 0
        self._active_compute_stream = None

    def _schedule_prefetch(
        self, group_start: int, group_count: int, slot_idx: int
    ) -> None:
        lengths = self._logical_lengths[group_start : group_start + group_count]
        if len(set(lengths)) != 1:
            raise RuntimeError("group members do not have a uniform logical length")
        logical_length = lengths[0]
        state = self._slots[slot_idx]
        element_size = torch.empty((), dtype=self.dtype).element_size()
        bytes_per_kind = (
            group_count
            * self.geometry.batch_size
            * logical_length
            * self.geometry.head_dim
            * element_size
        )
        with torch.cuda.stream(self.prefetch_stream):
            if state.reusable_after is not None:
                self._record_wait(self.prefetch_stream, state.reusable_after)
            ready_events = {
                id(event): event
                for event in self._host_write_done[
                    group_start : group_start + group_count
                ]
                if event is not None
            }
            for event in ready_events.values():
                self._record_wait(self.prefetch_stream, event)
            if logical_length:

                def copy_group_to_device() -> None:
                    self._slot_keys[slot_idx][
                        :group_count, ..., :logical_length, :
                    ].copy_(
                        self._host_group_tensor(
                            "key", group_start, group_count
                        )[..., :logical_length, :],
                        non_blocking=True,
                    )
                    self._slot_values[slot_idx][
                        :group_count, ..., :logical_length, :
                    ].copy_(
                        self._host_group_tensor(
                            "value", group_start, group_count
                        )[..., :logical_length, :],
                        non_blocking=True,
                    )

                self._record_transfer(
                    name="h2d",
                    stream=self.prefetch_stream,
                    operation=copy_group_to_device,
                    byte_count=2 * bytes_per_kind,
                    copy_calls=2,
                )
            ready = self._new_event()
            ready.record(self.prefetch_stream)

        state.group_start = group_start
        state.group_count = group_count
        state.prefetched_length = logical_length
        state.prefetch_done = ready
        if self.metrics_enabled:
            self.metrics["prefetched_groups"] += 1

    def _find_prefetched_slot(self, group_start: int, group_count: int) -> Optional[int]:
        lengths = self._logical_lengths[group_start : group_start + group_count]
        if len(set(lengths)) != 1:
            return None
        logical_length = lengths[0]
        for slot_idx, state in enumerate(self._slots):
            if (
                state.group_start == group_start
                and state.group_count == group_count
                and state.prefetched_length == logical_length
            ):
                return slot_idx
        return None

    def _choose_slot(self, excluded: Optional[int] = None) -> int:
        for slot_idx in range(len(self._slots)):
            if slot_idx != excluded:
                return slot_idx
        raise RuntimeError("no staging slot is available")

    def _prefetch_successor(
        self, group_start: int, group_count: int, active_slot: int
    ) -> None:
        successor = (group_start + group_count) % self.geometry.num_entries
        successor_count = self._group_count(successor)
        if self._find_prefetched_slot(successor, successor_count) is not None:
            return
        slot = self._choose_slot(excluded=active_slot)
        self._schedule_prefetch(successor, successor_count, slot)

    def _activate_group(self, group_start: int, group_count: int) -> None:
        self._retire_active_group()
        slot = self._find_prefetched_slot(group_start, group_count)
        if slot is None:
            slot = self._choose_slot()
            self._schedule_prefetch(group_start, group_count, slot)
        state = self._slots[slot]
        if state.prefetch_done is None:
            raise RuntimeError("prefetch completion event was not recorded")
        compute_stream = torch.cuda.current_stream(self.device)
        if self.metrics_enabled:
            wait_start = self._new_event(timing=True)
            wait_end = self._new_event(timing=True)
            wait_start.record(compute_stream)
        if self.activation_wait_mode == "blocking":
            state.prefetch_done.synchronize()
            if self.metrics_enabled:
                self.metrics["blocking_activation_wait_calls"] += 1
        else:
            self._record_wait(compute_stream, state.prefetch_done)
            if self.metrics_enabled:
                self.metrics["nonblocking_activation_wait_calls"] += 1
        if self.metrics_enabled:
            wait_end.record(compute_stream)
            self._timing_events.append(("activation_wait", wait_start, wait_end))

        self._active_slot = slot
        self._active_group_start = group_start
        self._active_group_count = group_count
        self._active_updates = 0
        self._active_compute_stream = compute_stream
        self._prefetch_successor(group_start, group_count, active_slot=slot)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have identical shapes")
        if key_states.device != self.device or value_states.device != self.device:
            raise ValueError("K/V inputs must be on the configured CUDA device")
        if key_states.dtype != self.dtype or value_states.dtype != self.dtype:
            raise ValueError("K/V dtype differs from slab dtype")
        if key_states.shape[0] != 1 or key_states.shape[1] != 1:
            raise ValueError(
                f"expected head-wise [1, 1, seq, dim], got {tuple(key_states.shape)}"
            )
        if key_states.shape[-1] != self.geometry.head_dim:
            raise ValueError("K/V head dimension differs from slab geometry")

        group_start = self._group_start(layer_idx)
        group_count = self._group_count(group_start)
        if (
            self._active_group_start != group_start
            or self._active_updates == self._active_group_count
        ):
            self._activate_group(group_start, group_count)
        if self._active_slot is None:
            raise RuntimeError("group activation did not select a slot")
        local_idx = layer_idx - group_start
        if local_idx != self._active_updates:
            raise RuntimeError(
                f"group update order violated: expected local {self._active_updates}, "
                f"got {local_idx}"
            )

        append_length = key_states.shape[-2]
        old_length, new_length = self._bookkeeping.preview(
            layer_idx, append_length
        )
        if self._active_updates == 0:
            self._active_write_start = old_length
            self._active_write_end = new_length
        elif (
            old_length != self._active_write_start
            or new_length != self._active_write_end
        ):
            raise RuntimeError("group members have inconsistent append ranges")

        advanced = self._bookkeeping.advance(layer_idx, append_length)
        if advanced != (old_length, new_length):
            raise AssertionError("cache bookkeeping changed unexpectedly")
        self.write_position = self._bookkeeping.write_position
        self._seen_tokens = self._bookkeeping.seen_tokens
        self._slot_keys[self._active_slot][
            local_idx, ..., old_length:new_length, :
        ].copy_(key_states)
        self._slot_values[self._active_slot][
            local_idx, ..., old_length:new_length, :
        ].copy_(value_states)
        if self.audit_enabled:
            self._audit_expected_keys.append(key_states.detach().cpu())
            self._audit_expected_values.append(value_states.detach().cpu())
        self._active_updates += 1
        if self.metrics_enabled:
            self.metrics["cache_update_calls"] += 1
            self.metrics["appended_tokens"] += append_length

        active_key = self._slot_keys[self._active_slot][
            local_idx, ..., :new_length, :
        ]
        active_value = self._slot_values[self._active_slot][
            local_idx, ..., :new_length, :
        ]
        if not active_key.is_contiguous() or not active_value.is_contiguous():
            raise AssertionError("active K/V views must be contiguous")
        return active_key, active_value

    def prepare_layer_full_layout(self) -> None:
        """Materialize a sequence-major pinned layout for layer-level Full H2D.

        The normal slab is entry-major and an eight-entry prefix is strided when
        ``logical_length < max_cache_len``.  A direct grouped ``copy_`` therefore
        does not become one contiguous DMA.  This one-time post-Prefill layout
        conversion keeps the eight native GQA KV Heads (no Query-Head expansion)
        while making each layer's historical prefix contiguous.
        """
        if not self.layer_full_attention_execution:
            raise RuntimeError("layer-level Full execution is disabled")
        if self.transfer_group_size <= 1:
            raise ValueError("layer-level Full execution requires grouped KV Heads")
        self.synchronize()
        if len(set(self._logical_lengths)) != 1:
            raise AssertionError("Full KV entries have unequal logical lengths")
        logical_length = self._logical_lengths[0]
        group_count = self.transfer_group_size
        layers = self.geometry.num_entries // group_count
        host_shape = (
            layers,
            self.allocated_capacity,
            group_count,
            self.geometry.head_dim,
        )
        self._full_layer_host_keys = torch.empty(
            host_shape, dtype=self.dtype, device="cpu", pin_memory=True
        )
        self._full_layer_host_values = torch.empty(
            host_shape, dtype=self.dtype, device="cpu", pin_memory=True
        )
        for layer in range(layers):
            for head in range(group_count):
                entry = layer * group_count + head
                self._full_layer_host_keys[layer, :logical_length, head].copy_(
                    self._host_tensor("key", entry)[0, 0, :logical_length]
                )
                self._full_layer_host_values[layer, :logical_length, head].copy_(
                    self._host_tensor("value", entry)[0, 0, :logical_length]
                )
        gpu_shape = (
            self.allocated_capacity,
            group_count,
            self.geometry.head_dim,
        )
        self._full_layer_gpu_keys = torch.empty(
            gpu_shape, dtype=self.dtype, device=self.device
        )
        self._full_layer_gpu_values = torch.empty(
            gpu_shape, dtype=self.dtype, device=self.device
        )
        self._full_layer_host_write_done = [None] * layers

    def update_layer_full(
        self,
        *,
        key_states: List[torch.Tensor],
        value_states: List[torch.Tensor],
        entries: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transfer and append one decoder layer in native 8-KV-Head GQA form."""
        if not self.layer_full_attention_execution:
            raise RuntimeError("layer-level Full execution is disabled")
        if any(
            tensor is None
            for tensor in (
                self._full_layer_host_keys,
                self._full_layer_host_values,
                self._full_layer_gpu_keys,
                self._full_layer_gpu_values,
            )
        ):
            raise RuntimeError("call prepare_layer_full_layout() after Prefill")
        group_count = self.transfer_group_size
        if not (
            len(entries) == len(key_states) == len(value_states) == group_count
        ):
            raise ValueError("one complete decoder-layer KV group is required")
        group_start = entries[0]
        if entries != list(range(group_start, group_start + group_count)):
            raise ValueError("layer Full entries must be contiguous and ordered")
        if group_start % group_count:
            raise ValueError("layer Full entries are not group-aligned")
        expected_shape = (1, 1, 1, self.geometry.head_dim)
        if any(
            key.shape != value.shape or tuple(key.shape) != expected_shape
            for key, value in zip(key_states, value_states)
        ):
            raise ValueError("layer Full decode expects [1, 1, 1, dim] K/V")

        ranges = []
        for entry in entries:
            append_range = self._bookkeeping.preview(entry, 1)
            if self._bookkeeping.advance(entry, 1) != append_range:
                raise AssertionError("cache bookkeeping changed unexpectedly")
            ranges.append(append_range)
            self._initialized[entry] = True
        if len(set(ranges)) != 1:
            raise AssertionError("layer Full entries have unequal append ranges")
        old_length, new_length = ranges[0]
        self.write_position = self._bookkeeping.write_position
        self._seen_tokens = self._bookkeeping.seen_tokens
        layer = group_start // group_count
        stream = torch.cuda.current_stream(self.device)
        host_ready = self._full_layer_host_write_done[layer]
        if host_ready is not None:
            self._record_wait(stream, host_ready)

        assert self._full_layer_host_keys is not None
        assert self._full_layer_host_values is not None
        assert self._full_layer_gpu_keys is not None
        assert self._full_layer_gpu_values is not None
        byte_count = (
            2
            * old_length
            * group_count
            * self.geometry.head_dim
            * key_states[0].element_size()
        )

        def copy_history_to_device() -> None:
            self._full_layer_gpu_keys[:old_length].copy_(
                self._full_layer_host_keys[layer, :old_length], non_blocking=True
            )
            self._full_layer_gpu_values[:old_length].copy_(
                self._full_layer_host_values[layer, :old_length], non_blocking=True
            )

        self._record_transfer(
            name="h2d",
            stream=stream,
            operation=copy_history_to_device,
            byte_count=byte_count,
            copy_calls=2,
        )
        current_keys = torch.stack([key[0, 0, 0] for key in key_states], dim=0)
        current_values = torch.stack(
            [value[0, 0, 0] for value in value_states], dim=0
        )
        self._full_layer_gpu_keys[old_length].copy_(current_keys)
        self._full_layer_gpu_values[old_length].copy_(current_values)

        ready = self._new_event()
        ready.record(stream)
        with torch.cuda.stream(self.eviction_stream):
            self._record_wait(self.eviction_stream, ready)

            def copy_delta_to_host() -> None:
                self._full_layer_host_keys[layer, old_length].copy_(
                    current_keys, non_blocking=True
                )
                self._full_layer_host_values[layer, old_length].copy_(
                    current_values, non_blocking=True
                )

            delta_bytes = (
                2
                * group_count
                * self.geometry.head_dim
                * key_states[0].element_size()
            )
            self._record_transfer(
                name="d2h",
                stream=self.eviction_stream,
                operation=copy_delta_to_host,
                byte_count=delta_bytes,
                copy_calls=2,
            )
            host_done = self._new_event()
            host_done.record(self.eviction_stream)
        current_keys.record_stream(self.eviction_stream)
        current_values.record_stream(self.eviction_stream)
        self._full_layer_host_write_done[layer] = host_done
        if self.metrics_enabled:
            self.metrics["cache_update_calls"] += group_count
            self.metrics["layer_full_update_calls"] += 1
            self.metrics["appended_tokens"] += group_count
        return (
            self._full_layer_gpu_keys[:new_length].unsqueeze(0),
            self._full_layer_gpu_values[:new_length].unsqueeze(0),
        )

    def synchronize(self) -> None:
        self._retire_active_group()
        torch.cuda.current_stream(self.device).synchronize()
        self.prefetch_stream.synchronize()
        self.eviction_stream.synchronize()
        self.audit_stream.synchronize()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._logical_lengths[0 if layer_idx is None else layer_idx]

    def get_max_length(self) -> Optional[int]:
        return self.allocated_capacity

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        del beam_idx
        raise NotImplementedError(
            "batch-size-1 fixed-order ablation cache does not support reordering"
        )

    def __len__(self) -> int:
        return sum(self._initialized)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.geometry.num_entries:
            raise KeyError(layer_idx)
        if self._full_layer_host_keys is not None:
            group_count = self.transfer_group_size
            layer = layer_idx // group_count
            head = layer_idx % group_count
            event = self._full_layer_host_write_done[layer]
            if event is not None:
                event.synchronize()
            length = self._logical_lengths[layer_idx]
            assert self._full_layer_host_values is not None
            return (
                self._full_layer_host_keys[layer, :length, head]
                .unsqueeze(0)
                .unsqueeze(0),
                self._full_layer_host_values[layer, :length, head]
                .unsqueeze(0)
                .unsqueeze(0),
            )
        event = self._host_write_done[layer_idx]
        if event is not None:
            event.synchronize()
        length = self._logical_lengths[layer_idx]
        return (
            self._host_tensor("key", layer_idx)[..., :length, :],
            self._host_tensor("value", layer_idx)[..., :length, :],
        )

    @property
    def key_cache(self) -> List[torch.Tensor]:
        return [
            self[index][0]
            for index in range(self.geometry.num_entries)
            if self._initialized[index]
        ]

    @property
    def value_cache(self) -> List[torch.Tensor]:
        return [
            self[index][1]
            for index in range(self.geometry.num_entries)
            if self._initialized[index]
        ]

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "logical_length": self.get_seq_length(),
            "logical_lengths_uniform": len(set(self._logical_lengths)) == 1,
            "logical_lengths_min": min(self._logical_lengths),
            "logical_lengths_max": max(self._logical_lengths),
            "allocated_capacity": self.allocated_capacity,
            "initialized_entries": sum(self._initialized),
            "expected_entries": self.geometry.num_entries,
            "host_slab_bytes": self._host_slab.numel()
            * self._host_slab.element_size(),
            "host_slab_pinned": self._host_slab.is_pinned(),
            "staging_slot_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in self._slot_keys + self._slot_values
            ),
            "transfer_group_size": self.transfer_group_size,
            "writeback_mode": self.writeback_mode,
            "activation_wait_mode": self.activation_wait_mode,
            "layer_full_attention_execution": (
                self.layer_full_attention_execution
            ),
            "layer_full_layout_prepared": self._full_layer_host_keys is not None,
            "layer_full_host_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    self._full_layer_host_keys,
                    self._full_layer_host_values,
                )
                if tensor is not None
            ),
        }

    from_legacy_cache = None
    to_legacy_cache = None


class StaticHeadwiseShortWindowOffloadedCache(GroupedPinnedSlabOffloadedCache):
    """Decode-only static Full/short-window H2D policy.

    The host slab deliberately retains the complete K/V history for every
    entry.  During decode, entries in ``short_head_entries`` transfer only the
    sink plus enough historical recent tokens that appending the current token
    produces exactly ``sink_tokens + recent_tokens`` visible positions.

    This first system prototype is intentionally restricted to group size 1
    and delta-only writeback.  Those constraints isolate the H2D policy from
    the previously measured grouping and D2H dimensions.
    """

    def __init__(
        self,
        *,
        short_head_entries: List[int] | Tuple[int, ...] | set[int],
        sink_tokens: int,
        recent_tokens: int,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("transfer_group_size", 1) != 1:
            raise ValueError("static short-window H2D requires transfer_group_size=1")
        if kwargs.get("writeback_mode", "delta") != "delta":
            raise ValueError("static short-window H2D requires delta writeback")
        if sink_tokens < 0 or recent_tokens <= 0:
            raise ValueError(
                "sink_tokens must be non-negative and recent_tokens must be positive"
            )
        super().__init__(**kwargs)
        entries = frozenset(int(entry) for entry in short_head_entries)
        if any(not 0 <= entry < self.geometry.num_entries for entry in entries):
            raise ValueError(
                f"short head entries must be in [0, {self.geometry.num_entries})"
            )
        self.short_head_entries = entries
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self._slot_is_short = [False, False]
        self._slot_attention_lengths = [0, 0]
        self._active_is_short = False
        self._active_attention_length = 0
        self._active_compact_write_start = 0
        self._active_compact_write_end = 0

    @classmethod
    def from_llama_model(
        cls,
        model: torch.nn.Module,
        *,
        max_cache_len: int,
        short_head_entries: List[int] | Tuple[int, ...] | set[int],
        sink_tokens: int,
        recent_tokens: int,
        transfer_group_size: int = 1,
        writeback_mode: str = "delta",
        activation_wait_mode: str = "event",
    ) -> "StaticHeadwiseShortWindowOffloadedCache":
        config = model.config
        num_entries = config.num_hidden_layers * config.num_key_value_heads
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        parameter = next(model.parameters())
        return cls(
            num_cache_entries=num_entries,
            max_cache_len=max_cache_len,
            head_dim=head_dim,
            dtype=parameter.dtype,
            device=parameter.device,
            transfer_group_size=transfer_group_size,
            writeback_mode=writeback_mode,
            activation_wait_mode=activation_wait_mode,
            short_head_entries=short_head_entries,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
        )

    def enable_history_audit(self) -> None:
        raise NotImplementedError(
            "the base history audit assumes logical and staging indices match"
        )

    def _schedule_prefetch(
        self, group_start: int, group_count: int, slot_idx: int
    ) -> None:
        use_short = (
            group_count == 1
            and group_start in self.short_head_entries
            and self._initialized[group_start]
        )
        self._slot_is_short[slot_idx] = use_short
        if not use_short:
            self._slot_attention_lengths[slot_idx] = self._logical_lengths[
                group_start
            ]
            super()._schedule_prefetch(group_start, group_count, slot_idx)
            if self.metrics_enabled:
                self.metrics["full_prefetched_entries"] += group_count
            return

        logical_length = self._logical_lengths[group_start]
        sink_end = min(self.sink_tokens, logical_length)
        # The current decode token is appended after this prefetch.  Copy only
        # recent_tokens - 1 historical positions so the returned cache matches
        # sink + recent after that append.
        historical_recent = max(0, self.recent_tokens - 1)
        recent_start = max(sink_end, logical_length - historical_recent)
        regions = []
        if sink_end:
            regions.append((0, sink_end, 0))
        if recent_start < logical_length:
            regions.append((recent_start, logical_length, sink_end))
        selected_length = sink_end + (logical_length - recent_start)

        state = self._slots[slot_idx]
        element_size = torch.empty((), dtype=self.dtype).element_size()
        bytes_per_kind = (
            self.geometry.batch_size
            * selected_length
            * self.geometry.head_dim
            * element_size
        )
        with torch.cuda.stream(self.prefetch_stream):
            if state.reusable_after is not None:
                self._record_wait(self.prefetch_stream, state.reusable_after)
            host_ready = self._host_write_done[group_start]
            if host_ready is not None:
                self._record_wait(self.prefetch_stream, host_ready)

            def copy_selected_to_device() -> None:
                host_key = self._host_tensor("key", group_start)
                host_value = self._host_tensor("value", group_start)
                for source_start, source_end, destination_start in regions:
                    width = source_end - source_start
                    destination_end = destination_start + width
                    self._slot_keys[slot_idx][
                        0, ..., destination_start:destination_end, :
                    ].copy_(
                        host_key[..., source_start:source_end, :],
                        non_blocking=True,
                    )
                    self._slot_values[slot_idx][
                        0, ..., destination_start:destination_end, :
                    ].copy_(
                        host_value[..., source_start:source_end, :],
                        non_blocking=True,
                    )

            if regions:
                self._record_transfer(
                    name="h2d",
                    stream=self.prefetch_stream,
                    operation=copy_selected_to_device,
                    byte_count=2 * bytes_per_kind,
                    copy_calls=2 * len(regions),
                )
            ready = self._new_event()
            ready.record(self.prefetch_stream)

        state.group_start = group_start
        state.group_count = group_count
        state.prefetched_length = logical_length
        state.prefetch_done = ready
        self._slot_attention_lengths[slot_idx] = selected_length
        if self.metrics_enabled:
            self.metrics["prefetched_groups"] += 1
            self.metrics["short_prefetched_entries"] += 1
            self.metrics["short_historical_tokens_transferred"] += selected_length

    def _activate_group(self, group_start: int, group_count: int) -> None:
        super()._activate_group(group_start, group_count)
        if self._active_slot is None:
            raise RuntimeError("group activation did not select a slot")
        self._active_is_short = self._slot_is_short[self._active_slot]
        self._active_attention_length = self._slot_attention_lengths[
            self._active_slot
        ]

    def _retire_active_group(self) -> None:
        if self._active_slot is None or not self._active_is_short:
            super()._retire_active_group()
            self._active_is_short = False
            self._active_attention_length = 0
            return
        if self._active_group_start is None:
            raise RuntimeError("short active group has no entry")
        if self._active_group_count != 1 or self._active_updates != 1:
            raise RuntimeError("short active entry was not updated exactly once")
        if self._active_compute_stream is None:
            raise RuntimeError("short active entry has no owning compute stream")

        slot = self._active_slot
        entry = self._active_group_start
        host_start = self._active_write_start
        host_end = self._active_write_end
        compact_start = self._active_compact_write_start
        compact_end = self._active_compact_write_end
        if host_end - host_start != compact_end - compact_start:
            raise AssertionError("host and compact append widths differ")

        compute_done = self._new_event()
        compute_done.record(self._active_compute_stream)
        element_size = torch.empty((), dtype=self.dtype).element_size()
        bytes_per_kind = (
            self.geometry.batch_size
            * (host_end - host_start)
            * self.geometry.head_dim
            * element_size
        )
        with torch.cuda.stream(self.eviction_stream):
            self._record_wait(self.eviction_stream, compute_done)

            def copy_delta_to_host() -> None:
                self._host_tensor("key", entry)[..., host_start:host_end, :].copy_(
                    self._slot_keys[slot][
                        0, ..., compact_start:compact_end, :
                    ],
                    non_blocking=True,
                )
                self._host_tensor("value", entry)[..., host_start:host_end, :].copy_(
                    self._slot_values[slot][
                        0, ..., compact_start:compact_end, :
                    ],
                    non_blocking=True,
                )

            self._record_transfer(
                name="d2h",
                stream=self.eviction_stream,
                operation=copy_delta_to_host,
                byte_count=2 * bytes_per_kind,
                copy_calls=2,
            )
            host_done = self._new_event()
            host_done.record(self.eviction_stream)

        self._host_write_done[entry] = host_done
        self._initialized[entry] = True
        self._slots[slot].reusable_after = host_done
        if self.metrics_enabled:
            self.metrics["evicted_groups"] += 1
            self.metrics["short_evicted_entries"] += 1

        self._active_slot = None
        self._active_group_start = None
        self._active_group_count = 0
        self._active_updates = 0
        self._active_compute_stream = None
        self._active_is_short = False
        self._active_attention_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have identical shapes")
        if key_states.device != self.device or value_states.device != self.device:
            raise ValueError("K/V inputs must be on the configured CUDA device")
        if key_states.dtype != self.dtype or value_states.dtype != self.dtype:
            raise ValueError("K/V dtype differs from slab dtype")
        if key_states.shape[0] != 1 or key_states.shape[1] != 1:
            raise ValueError(
                f"expected head-wise [1, 1, seq, dim], got {tuple(key_states.shape)}"
            )
        if key_states.shape[-1] != self.geometry.head_dim:
            raise ValueError("K/V head dimension differs from slab geometry")

        group_start = self._group_start(layer_idx)
        group_count = self._group_count(group_start)
        if (
            self._active_group_start != group_start
            or self._active_updates == self._active_group_count
        ):
            self._activate_group(group_start, group_count)
        if self._active_slot is None:
            raise RuntimeError("group activation did not select a slot")
        if layer_idx != group_start or self._active_updates != 0:
            raise RuntimeError("static short-window cache requires head-wise order")

        append_length = key_states.shape[-2]
        old_length, new_length = self._bookkeeping.preview(
            layer_idx, append_length
        )
        self._active_write_start = old_length
        self._active_write_end = new_length
        if self._active_is_short:
            if append_length != 1:
                raise RuntimeError(
                    "short-window transfer is decode-only and requires q_len=1"
                )
            slot_write_start = self._active_attention_length
            slot_write_end = slot_write_start + append_length
        else:
            slot_write_start = old_length
            slot_write_end = new_length
        self._active_compact_write_start = slot_write_start
        self._active_compact_write_end = slot_write_end

        advanced = self._bookkeeping.advance(layer_idx, append_length)
        if advanced != (old_length, new_length):
            raise AssertionError("cache bookkeeping changed unexpectedly")
        self.write_position = self._bookkeeping.write_position
        self._seen_tokens = self._bookkeeping.seen_tokens
        self._slot_keys[self._active_slot][
            0, ..., slot_write_start:slot_write_end, :
        ].copy_(key_states)
        self._slot_values[self._active_slot][
            0, ..., slot_write_start:slot_write_end, :
        ].copy_(value_states)
        self._active_updates = 1
        if self.metrics_enabled:
            self.metrics["cache_update_calls"] += 1
            self.metrics["appended_tokens"] += append_length
            if self._active_is_short:
                self.metrics["short_cache_update_calls"] += 1

        return_length = slot_write_end if self._active_is_short else new_length
        active_key = self._slot_keys[self._active_slot][
            0, ..., :return_length, :
        ]
        active_value = self._slot_values[self._active_slot][
            0, ..., :return_length, :
        ]
        if not active_key.is_contiguous() or not active_value.is_contiguous():
            raise AssertionError("active K/V views must be contiguous")
        return active_key, active_value

    def state_snapshot(self) -> Dict[str, Any]:
        snapshot = super().state_snapshot()
        snapshot.update(
            {
                "h2d_policy": "static_full_or_sink_recent",
                "short_head_count": len(self.short_head_entries),
                "full_head_count": (
                    self.geometry.num_entries - len(self.short_head_entries)
                ),
                "sink_tokens": self.sink_tokens,
                "recent_tokens": self.recent_tokens,
                "host_history": "full_for_all_entries",
            }
        )
        return snapshot
