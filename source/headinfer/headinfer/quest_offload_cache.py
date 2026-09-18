"""Quest-style per-Query-head sparse H2D cache for TPOT experiments.

This is a correctness-first systems prototype.  Full historical K/V lives in
pinned host memory.  Decode ranks fixed-size pages from GPU-resident Key
min/max metadata, packs the chosen host pages, and transfers only the selected
history plus sink/recent tokens.  It targets the sequential batch-size-1 path
installed by :func:`headinfer.mp.mp_headinfer`.
"""

from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache

from .slab_cache import SequentialCacheState, SlabGeometry


class QuestTopKOffloadedCache(Cache):
    """Pinned-host full cache with per-Query-head Quest Top-K page H2D."""

    def __init__(
        self,
        *,
        num_cache_entries: int,
        max_cache_len: int,
        head_dim: int,
        num_query_heads_per_kv: int,
        dtype: torch.dtype,
        device: torch.device | str,
        sink_tokens: int = 64,
        recent_tokens: int = 256,
        block_size: int = 16,
        budget_fraction: float = 0.05,
        selection_interval: int = 1,
        max_selected_history_fraction: Optional[float] = None,
        layer_batched_selection: bool = False,
        layer_flat_ragged_execution: bool = False,
        sparse_gather_stabilized: bool = False,
        per_head_varlen_attention_reference: bool = False,
    ) -> None:
        super().__init__()
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("an operational CUDA device is required")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if sink_tokens < 0 or recent_tokens <= 0 or block_size <= 0:
            raise ValueError("invalid sink/recent/block configuration")
        if not 0.0 < budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be in (0, 1]")
        if selection_interval <= 0:
            raise ValueError("selection_interval must be positive")
        if max_selected_history_fraction is None:
            max_selected_history_fraction = budget_fraction
        if not 0.0 < max_selected_history_fraction <= 1.0:
            raise ValueError("max_selected_history_fraction must be in (0, 1]")
        if num_query_heads_per_kv <= 0:
            raise ValueError("num_query_heads_per_kv must be positive")

        self.geometry = SlabGeometry(
            num_entries=num_cache_entries,
            batch_size=1,
            max_cache_len=max_cache_len,
            head_dim=head_dim,
        )
        self.dtype = dtype
        self.device = device
        self.allocated_capacity = max_cache_len
        self.num_query_heads_per_kv = num_query_heads_per_kv
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.block_size = block_size
        self.budget_fraction = budget_fraction
        self.selection_interval = int(selection_interval)
        self.layer_batched_selection = bool(layer_batched_selection)
        self.layer_flat_ragged_execution = bool(layer_flat_ragged_execution)
        self.sparse_gather_stabilized = bool(sparse_gather_stabilized)
        self.per_head_varlen_attention_reference = bool(
            per_head_varlen_attention_reference
        )
        if self.layer_flat_ragged_execution and not self.layer_batched_selection:
            raise ValueError(
                "layer-flat ragged execution requires layer-batched selection"
            )
        if self.sparse_gather_stabilized and not self.layer_flat_ragged_execution:
            raise ValueError(
                "stabilized sparse gather requires layer-flat ragged execution"
            )
        self.max_selected_history_fraction = max_selected_history_fraction
        self._bookkeeping = SequentialCacheState(num_cache_entries, max_cache_len)
        self._logical_lengths = self._bookkeeping.logical_lengths
        self._initialized = [False] * num_cache_entries
        self._seen_tokens = 0

        self._host_slab = torch.empty(
            self.geometry.total_elements,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        max_pages = math.ceil(max(0, max_cache_len - sink_tokens) / block_size)
        max_selected = min(
            max_cache_len,
            sink_tokens
            + recent_tokens
            + max(1, math.ceil(max_selected_history_fraction * max_pages)) * block_size,
        )
        self.max_selected_tokens_per_query_head = max_selected
        packed_shape = (1, max_selected, num_query_heads_per_kv, head_dim)
        # Two host packs avoid modifying a pinned H2D source before its copy
        # has completed.  The GPU pack is safe to reuse on the same stream.
        self._host_pack_keys = [
            torch.empty(packed_shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(2)
        ]
        self._host_pack_values = [
            torch.empty(packed_shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(2)
        ]
        self._gpu_pack_keys = torch.empty(packed_shape, dtype=dtype, device=device)
        self._gpu_pack_values = torch.empty(packed_shape, dtype=dtype, device=device)
        self._pack_ready: list[Optional[torch.cuda.Event]] = [None, None]
        self._pack_cursor = 0
        self._layer_flat_group_count: Optional[int] = None
        self._host_layer_flat_keys: list[torch.Tensor] = []
        self._host_layer_flat_values: list[torch.Tensor] = []
        self._host_layer_flat_row_indices: list[torch.Tensor] = []
        self._gpu_layer_flat_history_keys: Optional[torch.Tensor] = None
        self._gpu_layer_flat_history_values: Optional[torch.Tensor] = None
        self._gpu_layer_flat_attention_keys: Optional[torch.Tensor] = None
        self._gpu_layer_flat_attention_values: Optional[torch.Tensor] = None
        self._layer_flat_ready: list[Optional[torch.cuda.Event]] = [None, None]
        self._layer_flat_cursor = 0
        slab_rows = self._host_slab.view(
            2,
            self.geometry.num_entries,
            self.geometry.max_cache_len,
            self.geometry.head_dim,
        )
        self._host_key_rows = slab_rows[0].view(-1, self.geometry.head_dim)
        self._host_value_rows = slab_rows[1].view(-1, self.geometry.head_dim)

        self.eviction_stream = torch.cuda.Stream(device=device)
        self._host_write_done: list[Optional[torch.cuda.Event]] = [
            None
        ] * num_cache_entries
        self._page_min: list[Optional[torch.Tensor]] = [None] * num_cache_entries
        self._page_max: list[Optional[torch.Tensor]] = [None] * num_cache_entries
        self._prompt_length: Optional[int] = None
        self._last_query_head_lengths: list[Optional[tuple[int, ...]]] = [
            None
        ] * num_cache_entries
        self._cached_selected_pages: list[Optional[tuple[torch.Tensor, ...]]] = [
            None
        ] * num_cache_entries
        self._layer_prepared_entries: set[int] = set()
        self._capture_selection_trace = False
        self._selection_trace: list[dict[str, Any]] = []

        self.metrics_enabled = False
        self.metrics: Dict[str, float] = defaultdict(float)
        self._timing_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @classmethod
    def from_llama_model(cls, model: torch.nn.Module, **kwargs: Any) -> "QuestTopKOffloadedCache":
        config = model.config
        parameter = next(model.parameters())
        return cls(
            num_cache_entries=config.num_hidden_layers * config.num_key_value_heads,
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            num_query_heads_per_kv=(
                config.num_attention_heads // config.num_key_value_heads
            ),
            dtype=parameter.dtype,
            device=parameter.device,
            **kwargs,
        )

    def _host_tensor(self, kind: str, entry: int) -> torch.Tensor:
        start, end = self.geometry.bounds(kind, entry)
        return self._host_slab[start:end].view(self.geometry.tensor_shape())

    def _event(self, *, timing: bool = False) -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=timing, blocking=False)

    def reset_metrics(self) -> None:
        self.metrics = defaultdict(float)
        self._timing_events = []

    def enable_metrics(self, enabled: bool = True) -> None:
        self.metrics_enabled = enabled
        if enabled:
            self.reset_metrics()

    def _record_copy(
        self, name: str, operation, byte_count: int, *, copy_calls: int = 2
    ) -> None:
        stream = torch.cuda.current_stream(self.device)
        if self.metrics_enabled:
            start, end = self._event(timing=True), self._event(timing=True)
            start.record(stream)
        operation()
        if self.metrics_enabled:
            end.record(stream)
            self._timing_events.append((name, start, end))
            self.metrics[f"{name}_bytes"] += byte_count
            self.metrics[f"{name}_copy_calls"] += copy_calls

    def resolve_metrics(self) -> Dict[str, float]:
        self.synchronize()
        for name, start, end in self._timing_events:
            self.metrics[f"{name}_cuda_seconds"] += start.elapsed_time(end) / 1000.0
            self.metrics[f"{name}_event_intervals"] += 1
        self._timing_events = []
        return dict(self.metrics)

    def _schedule_d2h(
        self,
        entry: int,
        host_start: int,
        host_end: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        source_key = key_states.detach()
        source_value = value_states.detach()
        compute_stream = torch.cuda.current_stream(self.device)
        ready = self._event()
        ready.record(compute_stream)
        with torch.cuda.stream(self.eviction_stream):
            self.eviction_stream.wait_event(ready)
            self._host_tensor("key", entry)[..., host_start:host_end, :].copy_(
                source_key, non_blocking=True
            )
            self._host_tensor("value", entry)[..., host_start:host_end, :].copy_(
                source_value, non_blocking=True
            )
            done = self._event()
            done.record(self.eviction_stream)
        source_key.record_stream(self.eviction_stream)
        source_value.record_stream(self.eviction_stream)
        self._host_write_done[entry] = done
        if self.metrics_enabled:
            elements = 2 * (host_end - host_start) * self.geometry.head_dim
            self.metrics["d2h_bytes"] += elements * source_key.element_size()
            self.metrics["d2h_copy_calls"] += 2

    def _build_metadata(self, entry: int, key_states: torch.Tensor) -> None:
        keys = key_states[0, 0]
        tail = keys[self.sink_tokens :]
        pages = int(tail.shape[0]) // self.block_size
        if pages == 0:
            empty = torch.empty((0, self.geometry.head_dim), dtype=self.dtype, device=self.device)
            self._page_min[entry] = empty
            self._page_max[entry] = empty
            return
        complete = tail[: pages * self.block_size].view(
            pages, self.block_size, self.geometry.head_dim
        )
        self._page_min[entry] = complete.amin(dim=1)
        self._page_max[entry] = complete.amax(dim=1)

    def enable_selection_trace(self, enabled: bool = True) -> None:
        """Capture selected page indices for correctness-only parity runs."""
        self._capture_selection_trace = bool(enabled)
        self._selection_trace = []

    def selection_trace(self) -> list[dict[str, Any]]:
        return list(self._selection_trace)

    def prepare_layer_selection(
        self,
        *,
        entries: list[int],
        queries: list[torch.Tensor],
    ) -> None:
        """Batch Quest score/Top-K for all Query heads in one decoder layer.

        This changes only operator granularity.  Each Query head still receives
        an independent fixed-budget Top-K over the metadata belonging to its
        GQA KV head.  CPU gather, H2D, and attention remain in ``update``.
        """
        if not self.layer_batched_selection:
            return
        if len(entries) != len(queries) or not entries:
            raise ValueError("entries and queries must be non-empty and aligned")
        if len(set(entries)) != len(entries):
            raise ValueError("layer entries must be unique")
        expected_query_shape = (
            1,
            self.num_query_heads_per_kv,
            1,
            self.geometry.head_dim,
        )
        if any(tuple(query.shape) != expected_query_shape for query in queries):
            raise ValueError("unexpected layer Query shape")

        old_lengths = [self._logical_lengths[entry] for entry in entries]
        if len(set(old_lengths)) != 1:
            raise AssertionError("layer KV entries have unequal logical lengths")
        old_length = old_lengths[0]
        sink_end = min(self.sink_tokens, old_length)
        historical_recent = max(0, self.recent_tokens - 1)
        recent_start = max(sink_end, old_length - historical_recent)
        eligible_tokens = max(0, recent_start - sink_end)
        candidate_pages = eligible_tokens // self.block_size
        budget = min(
            candidate_pages,
            max(1, math.ceil(self.budget_fraction * eligible_tokens / self.block_size)),
        )
        metadata = [(self._page_min[entry], self._page_max[entry]) for entry in entries]
        if any(page_min is None or page_max is None for page_min, page_max in metadata):
            raise RuntimeError("prompt metadata is missing")
        candidate_pages = min(
            candidate_pages,
            *(int(page_min.shape[0]) for page_min, _ in metadata if page_min is not None),
        )
        budget = min(budget, candidate_pages)
        if self._prompt_length is None:
            raise RuntimeError("prompt length is missing")
        decode_step = old_length - self._prompt_length + 1
        refresh = (
            any(self._cached_selected_pages[entry] is None for entry in entries)
            or (decode_step - 1) % self.selection_interval == 0
        )
        if not refresh or budget == 0:
            return

        selector_started = time.perf_counter() if self.metrics_enabled else None
        if self.metrics_enabled:
            selector_gpu_start = self._event(timing=True)
            selector_gpu_end = self._event(timing=True)
            selector_gpu_start.record(torch.cuda.current_stream(self.device))
        q = torch.cat([query[0, :, 0] for query in queries], dim=0).float()
        page_min = torch.stack(
            [item[0][:candidate_pages] for item in metadata if item[0] is not None],
            dim=0,
        ).repeat_interleave(self.num_query_heads_per_kv, dim=0)
        page_max = torch.stack(
            [item[1][:candidate_pages] for item in metadata if item[1] is not None],
            dim=0,
        ).repeat_interleave(self.num_query_heads_per_kv, dim=0)
        extrema = torch.where(q.unsqueeze(1) > 0, page_max, page_min)
        scores = (extrema.float() * q.unsqueeze(1)).sum(dim=-1)
        chosen = torch.topk(scores, k=budget, dim=-1, largest=True, sorted=False).indices
        chosen = chosen.sort(dim=-1).values
        if self.metrics_enabled:
            selector_gpu_end.record(torch.cuda.current_stream(self.device))
            self._timing_events.append(("selector", selector_gpu_start, selector_gpu_end))
            selector_sync_started = time.perf_counter()
        chosen_cpu = chosen.cpu()
        if self.metrics_enabled:
            selector_sync_finished = time.perf_counter()
            self.metrics["selector_index_sync_wall_seconds"] += (
                selector_sync_finished - selector_sync_started
            )
            self.metrics["selector_gpu_enqueued_and_sync_wall_seconds"] += (
                selector_sync_finished - selector_started
            )
            self.metrics["selector_refresh_query_heads"] += int(chosen_cpu.shape[0])
            self.metrics["selector_batch_calls"] += 1
            self.metrics["selector_index_sync_calls"] += 1

        offset = 0
        for entry in entries:
            stop = offset + self.num_query_heads_per_kv
            self._cached_selected_pages[entry] = tuple(chosen_cpu[offset:stop].unbind(0))
            self._layer_prepared_entries.add(entry)
            offset = stop
        if self.metrics_enabled:
            self.metrics["selector_total_wall_seconds"] += (
                time.perf_counter() - selector_started
            )

    def _selected_positions(
        self,
        *,
        query: torch.Tensor,
        entry: int,
        old_length: int,
    ) -> list[torch.Tensor]:
        sink_end = min(self.sink_tokens, old_length)
        historical_recent = max(0, self.recent_tokens - 1)
        recent_start = max(sink_end, old_length - historical_recent)
        eligible_tokens = max(0, recent_start - sink_end)
        candidate_pages = eligible_tokens // self.block_size
        budget = min(
            candidate_pages,
            max(1, math.ceil(self.budget_fraction * eligible_tokens / self.block_size)),
        )
        page_min = self._page_min[entry]
        page_max = self._page_max[entry]
        if page_min is None or page_max is None:
            raise RuntimeError("prompt metadata is missing")
        candidate_pages = min(candidate_pages, int(page_min.shape[0]))
        budget = min(budget, candidate_pages)
        positions: list[torch.Tensor] = []
        sink_cpu = torch.arange(sink_end, dtype=torch.long)
        recent_cpu = torch.arange(recent_start, old_length, dtype=torch.long)
        if self._prompt_length is None:
            raise RuntimeError("prompt length is missing")
        decode_step = old_length - self._prompt_length + 1
        refresh = (
            self._cached_selected_pages[entry] is None
            or (decode_step - 1) % self.selection_interval == 0
        )
        refreshed_pages: list[torch.Tensor] = []
        prepared_for_entry = entry in self._layer_prepared_entries
        for group_head in range(self.num_query_heads_per_kv):
            if budget:
                if prepared_for_entry:
                    cached = self._cached_selected_pages[entry]
                    if cached is None:
                        raise AssertionError("prepared selection cache is missing")
                    chosen = cached[group_head]
                    if self.metrics_enabled:
                        selector_cpu_started = time.perf_counter()
                elif refresh:
                    selector_started = time.perf_counter() if self.metrics_enabled else None
                    if self.metrics_enabled:
                        selector_gpu_start = self._event(timing=True)
                        selector_gpu_end = self._event(timing=True)
                        selector_gpu_start.record(torch.cuda.current_stream(self.device))
                    q = query[0, group_head, 0].float()
                    extrema = torch.where(q.unsqueeze(0) > 0, page_max[:candidate_pages], page_min[:candidate_pages])
                    scores = (extrema.float() * q.unsqueeze(0)).sum(dim=-1)
                    chosen = torch.topk(scores, k=budget, largest=True, sorted=False).indices
                    chosen = chosen.sort().values
                    if self.metrics_enabled:
                        selector_gpu_end.record(torch.cuda.current_stream(self.device))
                        self._timing_events.append(
                            ("selector", selector_gpu_start, selector_gpu_end)
                        )
                        selector_sync_started = time.perf_counter()
                    chosen = chosen.cpu()
                    if self.metrics_enabled:
                        selector_sync_finished = time.perf_counter()
                        self.metrics["selector_index_sync_wall_seconds"] += (
                            selector_sync_finished - selector_sync_started
                        )
                        self.metrics["selector_gpu_enqueued_and_sync_wall_seconds"] += (
                            selector_sync_finished - selector_started
                        )
                        selector_cpu_started = selector_sync_finished
                        self.metrics["selector_refresh_query_heads"] += 1
                        self.metrics["selector_index_sync_calls"] += 1
                    refreshed_pages.append(chosen)
                else:
                    cached = self._cached_selected_pages[entry]
                    if cached is None:
                        raise AssertionError("selection cache unexpectedly missing")
                    chosen = cached[group_head]
                    if self.metrics_enabled:
                        selector_cpu_started = time.perf_counter()
                        self.metrics["selector_reuse_query_heads"] += 1
                offsets = torch.arange(self.block_size, dtype=torch.long)
                blocks = (
                    self.sink_tokens
                    + chosen.unsqueeze(1) * self.block_size
                    + offsets.unsqueeze(0)
                ).reshape(-1)
                selected = torch.cat((sink_cpu, blocks, recent_cpu))
                if self._capture_selection_trace:
                    self._selection_trace.append(
                        {
                            "decode_step": decode_step,
                            "entry": entry,
                            "query_head_in_group": group_head,
                            "selected_page_indices": tuple(int(value) for value in chosen.tolist()),
                        }
                    )
                if self.metrics_enabled:
                    self.metrics["selector_cpu_build_wall_seconds"] += (
                        time.perf_counter() - selector_cpu_started
                    )
            else:
                selected = torch.cat((sink_cpu, recent_cpu))
            positions.append(selected)
        if refresh and budget:
            if not prepared_for_entry:
                self._cached_selected_pages[entry] = tuple(refreshed_pages)
        self._layer_prepared_entries.discard(entry)
        return positions

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape != value_states.shape or key_states.shape[:2] != (1, 1):
            raise ValueError("expected identical head-wise [1, 1, seq, dim] K/V")
        append_length = int(key_states.shape[-2])
        old_length, new_length = self._bookkeeping.preview(layer_idx, append_length)
        self._bookkeeping.advance(layer_idx, append_length)
        self._seen_tokens = self._bookkeeping.seen_tokens
        self._initialized[layer_idx] = True

        if append_length > 1:
            if old_length != 0:
                raise ValueError("chunked/multiple prefill updates are unsupported")
            if self._prompt_length is None:
                self._prompt_length = append_length
            elif self._prompt_length != append_length:
                raise RuntimeError("cache entries received unequal prompt lengths")
            self._build_metadata(layer_idx, key_states)
            self._schedule_d2h(layer_idx, old_length, new_length, key_states, value_states)
            return key_states, value_states

        query = None if cache_kwargs is None else cache_kwargs.get("query_states")
        if query is None or query.shape != (
            1, self.num_query_heads_per_kv, 1, self.geometry.head_dim
        ):
            raise ValueError("decode requires post-RoPE query_states")
        update_started = time.perf_counter() if self.metrics_enabled else None
        host_ready = self._host_write_done[layer_idx]
        phase_started = time.perf_counter() if self.metrics_enabled else None
        if host_ready is not None:
            host_ready.synchronize()
        if self.metrics_enabled:
            self.metrics["host_ready_wait_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
        positions = self._selected_positions(query=query, entry=layer_idx, old_length=old_length)
        if self.metrics_enabled:
            self.metrics["selector_total_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
        history_lengths = tuple(int(item.numel()) for item in positions)
        if not history_lengths:
            raise AssertionError("at least one Query head is required")
        max_history_length = max(history_lengths)
        total_lengths = tuple(length + 1 for length in history_lengths)
        max_total_length = max(total_lengths)
        self._last_query_head_lengths[layer_idx] = total_lengths
        slot = self._pack_cursor
        self._pack_cursor = 1 - self._pack_cursor
        phase_started = time.perf_counter() if self.metrics_enabled else None
        if self._pack_ready[slot] is not None:
            self._pack_ready[slot].synchronize()
        if self.metrics_enabled:
            self.metrics["pack_ready_wait_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
        host_key = self._host_tensor("key", layer_idx)[0, 0, :old_length]
        host_value = self._host_tensor("value", layer_idx)[0, 0, :old_length]
        for group_head, selected in enumerate(positions):
            history_length = history_lengths[group_head]
            torch.index_select(
                host_key, 0, selected,
                out=self._host_pack_keys[slot][0, :history_length, group_head],
            )
            torch.index_select(
                host_value, 0, selected,
                out=self._host_pack_values[slot][0, :history_length, group_head],
            )
        if self.metrics_enabled:
            self.metrics["host_gather_pack_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
        element_size = key_states.element_size()
        byte_count = (
            2 * sum(history_lengths)
            * self.geometry.head_dim * element_size
        )

        def copy_selected() -> None:
            if len(set(history_lengths)) == 1:
                history_length = history_lengths[0]
                self._gpu_pack_keys[:, :history_length].copy_(
                    self._host_pack_keys[slot][:, :history_length], non_blocking=True
                )
                self._gpu_pack_values[:, :history_length].copy_(
                    self._host_pack_values[slot][:, :history_length], non_blocking=True
                )
                return
            for group_head, history_length in enumerate(history_lengths):
                self._gpu_pack_keys[:, :history_length, group_head].copy_(
                    self._host_pack_keys[slot][:, :history_length, group_head],
                    non_blocking=True,
                )
                self._gpu_pack_values[:, :history_length, group_head].copy_(
                    self._host_pack_values[slot][:, :history_length, group_head],
                    non_blocking=True,
                )

        phase_started = time.perf_counter() if self.metrics_enabled else None
        h2d_copy_calls = (
            2 if len(set(history_lengths)) == 1 else 2 * len(history_lengths)
        )
        self._record_copy(
            "h2d", copy_selected, byte_count, copy_calls=h2d_copy_calls
        )
        if self.metrics_enabled:
            self.metrics["h2d_enqueue_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
        copied = self._event()
        copied.record(torch.cuda.current_stream(self.device))
        self._pack_ready[slot] = copied
        current_key = key_states[0, 0, 0]
        current_value = value_states[0, 0, 0]
        for group_head, history_length in enumerate(history_lengths):
            self._gpu_pack_keys[0, history_length, group_head].copy_(current_key)
            self._gpu_pack_values[0, history_length, group_head].copy_(current_value)
        self._schedule_d2h(layer_idx, old_length, new_length, key_states, value_states)
        if self.metrics_enabled:
            self.metrics["append_and_d2h_enqueue_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            self.metrics["cache_update_calls"] += 1
            protected = min(self.sink_tokens, old_length) + min(
                self.recent_tokens - 1,
                max(0, old_length - min(self.sink_tokens, old_length)),
            )
            self.metrics["selected_history_tokens_total"] += sum(history_lengths)
            self.metrics["attention_valid_tokens_total"] += sum(total_lengths)
            self.metrics["attention_valid_tokens_min_sum"] += min(total_lengths)
            self.metrics["attention_valid_tokens_max_sum"] += max(total_lengths)
            self.metrics["selected_history_tokens_per_query_head"] += (
                sum(history_lengths) / len(history_lengths)
            )
            self.metrics["selected_adaptive_tokens_total"] += sum(
                max(0, length - protected) for length in history_lengths
            )
            self.metrics["cache_update_total_wall_seconds"] += (
                time.perf_counter() - update_started
            )
        # mp.py transposes these back to [batch, seq, query_heads, dim].
        return (
            self._gpu_pack_keys[:, :max_total_length].transpose(1, 2),
            self._gpu_pack_values[:, :max_total_length].transpose(1, 2),
        )

    def _ensure_layer_flat_buffers(self, group_count: int) -> None:
        """Allocate reusable layer-wide ragged buffers on first decode."""
        if self._layer_flat_group_count is not None:
            if self._layer_flat_group_count != group_count:
                raise ValueError("layer KV group count changed after initialization")
            return
        query_heads = group_count * self.num_query_heads_per_kv
        history_capacity = (
            query_heads * self.max_selected_tokens_per_query_head
        )
        host_shape = (history_capacity, self.geometry.head_dim)
        attention_shape = (
            history_capacity + query_heads,
            1,
            self.geometry.head_dim,
        )
        self._host_layer_flat_keys = [
            torch.empty(host_shape, dtype=self.dtype, device="cpu", pin_memory=True)
            for _ in range(2)
        ]
        self._host_layer_flat_values = [
            torch.empty(host_shape, dtype=self.dtype, device="cpu", pin_memory=True)
            for _ in range(2)
        ]
        self._host_layer_flat_row_indices = [
            torch.empty(history_capacity, dtype=torch.long, device="cpu")
            for _ in range(2)
        ]
        self._gpu_layer_flat_history_keys = torch.empty(
            host_shape, dtype=self.dtype, device=self.device
        )
        self._gpu_layer_flat_history_values = torch.empty(
            host_shape, dtype=self.dtype, device=self.device
        )
        self._gpu_layer_flat_attention_keys = torch.empty(
            attention_shape, dtype=self.dtype, device=self.device
        )
        self._gpu_layer_flat_attention_values = torch.empty(
            attention_shape, dtype=self.dtype, device=self.device
        )
        self._layer_flat_group_count = group_count

    def update_layer_flat_ragged(
        self,
        *,
        key_states: list[torch.Tensor],
        value_states: list[torch.Tensor],
        queries: list[torch.Tensor],
        entries: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[int, ...]]:
        """Gather and transfer one decoder layer as a flat ragged batch.

        Selection remains per Query head and byte-identical to ``update``.
        The only semantic change is execution granularity: all selected
        histories are concatenated in pinned CPU memory, copied with one K and
        one V H2D operation, then resegmented on GPU for varlen attention.
        """
        if not self.layer_flat_ragged_execution:
            raise RuntimeError("layer-flat ragged execution is disabled")
        if not (
            len(entries) == len(key_states) == len(value_states) == len(queries)
            and entries
        ):
            raise ValueError("layer entries, K/V, and Queries must be aligned")
        if len(set(entries)) != len(entries):
            raise ValueError("layer entries must be unique")
        group_count = len(entries)
        self._ensure_layer_flat_buffers(group_count)

        old_lengths: list[int] = []
        new_lengths: list[int] = []
        for entry, key, value in zip(entries, key_states, value_states):
            if key.shape != value.shape or key.shape != (
                1,
                1,
                1,
                self.geometry.head_dim,
            ):
                raise ValueError("layer-flat decode expects [1, 1, 1, dim] K/V")
            old_length, new_length = self._bookkeeping.preview(entry, 1)
            self._bookkeeping.advance(entry, 1)
            self._initialized[entry] = True
            old_lengths.append(old_length)
            new_lengths.append(new_length)
        self._seen_tokens = self._bookkeeping.seen_tokens
        if len(set(old_lengths)) != 1:
            raise AssertionError("layer KV entries have unequal logical lengths")
        old_length = old_lengths[0]

        update_started = time.perf_counter() if self.metrics_enabled else None
        phase_started = time.perf_counter() if self.metrics_enabled else None
        for entry in entries:
            host_ready = self._host_write_done[entry]
            if host_ready is not None:
                host_ready.synchronize()
        if self.metrics_enabled:
            self.metrics["host_ready_wait_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()

        layer_positions: list[list[torch.Tensor]] = []
        for entry, query in zip(entries, queries):
            positions = self._selected_positions(
                query=query, entry=entry, old_length=old_length
            )
            layer_positions.append(positions)
        if self.metrics_enabled:
            self.metrics["selector_total_wall_seconds"] += (
                time.perf_counter() - phase_started
            )

        lengths = tuple(
            int(selected.numel())
            for positions in layer_positions
            for selected in positions
        )
        if not lengths:
            raise AssertionError("at least one Query head is required")
        total_lengths = tuple(length + 1 for length in lengths)
        total_history = sum(lengths)
        total_attention = sum(total_lengths)
        slot = self._layer_flat_cursor
        self._layer_flat_cursor = 1 - self._layer_flat_cursor
        phase_started = time.perf_counter() if self.metrics_enabled else None
        ready = self._layer_flat_ready[slot]
        if ready is not None:
            ready.synchronize()
        if self.metrics_enabled:
            self.metrics["pack_ready_wait_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()

        host_flat_key = self._host_layer_flat_keys[slot]
        host_flat_value = self._host_layer_flat_values[slot]
        history_offset = 0
        if self.sparse_gather_stabilized:
            # Build one layer-global row-index list in a reusable CPU buffer.
            # Every Query Head remains present in the same order, including
            # duplicate GQA selections.  The two index_select calls therefore
            # gather K and V directly into the final pinned flat H2D source;
            # this is an execution cleanup, not GQA union/deduplication.
            row_indices = self._host_layer_flat_row_indices[slot]
            for entry, positions in zip(entries, layer_positions):
                entry_count = sum(int(selected.numel()) for selected in positions)
                stop = history_offset + entry_count
                torch.cat(positions, out=row_indices[history_offset:stop])
                row_indices[history_offset:stop].add_(
                    entry * self.geometry.max_cache_len
                )
                history_offset = stop
            torch.index_select(
                self._host_key_rows,
                0,
                row_indices[:total_history],
                out=host_flat_key[:total_history],
            )
            torch.index_select(
                self._host_value_rows,
                0,
                row_indices[:total_history],
                out=host_flat_value[:total_history],
            )
        else:
            for entry, positions in zip(entries, layer_positions):
                host_key = self._host_tensor("key", entry)[0, 0, :old_length]
                host_value = self._host_tensor("value", entry)[0, 0, :old_length]
                # Concatenating the per-Query-head index lists preserves their
                # exact ragged order.  Duplicate positions remain duplicated.
                entry_selected = torch.cat(positions)
                stop = history_offset + int(entry_selected.numel())
                torch.index_select(
                    host_key, 0, entry_selected,
                    out=host_flat_key[history_offset:stop],
                )
                torch.index_select(
                    host_value, 0, entry_selected,
                    out=host_flat_value[history_offset:stop],
                )
                history_offset = stop
        if history_offset != total_history:
            raise AssertionError("layer-flat history packing length mismatch")
        if self.metrics_enabled:
            self.metrics["host_gather_pack_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            self.metrics["host_gather_layer_calls"] += 1
            if self.sparse_gather_stabilized:
                self.metrics["host_gather_primitive_calls"] += 2
                self.metrics["host_gather_index_select_calls"] += 2
                self.metrics["host_gather_index_concat_calls"] += len(entries)
                self.metrics["host_gather_index_offset_calls"] += len(entries)
                self.metrics["host_gather_index_temporary_allocations"] += 0
                self.metrics["host_flat_pack_write_calls"] += 2
            else:
                self.metrics["host_gather_primitive_calls"] += 2 * len(entries)
                self.metrics["host_gather_index_select_calls"] += 2 * len(entries)
                self.metrics["host_gather_index_concat_calls"] += len(entries)
                self.metrics["host_gather_index_temporary_allocations"] += len(entries)
                self.metrics["host_flat_pack_write_calls"] += 2 * len(entries)

        assert self._gpu_layer_flat_history_keys is not None
        assert self._gpu_layer_flat_history_values is not None
        byte_count = 2 * total_history * self.geometry.head_dim * key_states[0].element_size()

        def copy_layer_history() -> None:
            self._gpu_layer_flat_history_keys[:total_history].copy_(
                host_flat_key[:total_history], non_blocking=True
            )
            self._gpu_layer_flat_history_values[:total_history].copy_(
                host_flat_value[:total_history], non_blocking=True
            )

        phase_started = time.perf_counter() if self.metrics_enabled else None
        self._record_copy("h2d", copy_layer_history, byte_count, copy_calls=2)
        if self.metrics_enabled:
            self.metrics["h2d_enqueue_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            self.metrics["layer_flat_h2d_calls"] += 1
            phase_started = time.perf_counter()
        copied = self._event()
        copied.record(torch.cuda.current_stream(self.device))
        self._layer_flat_ready[slot] = copied

        assert self._gpu_layer_flat_attention_keys is not None
        assert self._gpu_layer_flat_attention_values is not None
        key_segments: list[torch.Tensor] = []
        value_segments: list[torch.Tensor] = []
        history_offset = 0
        for group_index, (key, value) in enumerate(zip(key_states, value_states)):
            current_key = key[0, 0]
            current_value = value[0, 0]
            for head_in_group in range(self.num_query_heads_per_kv):
                flat_head = (
                    group_index * self.num_query_heads_per_kv + head_in_group
                )
                stop = history_offset + lengths[flat_head]
                key_segments.extend(
                    (self._gpu_layer_flat_history_keys[history_offset:stop], current_key)
                )
                value_segments.extend(
                    (self._gpu_layer_flat_history_values[history_offset:stop], current_value)
                )
                history_offset = stop
        torch.cat(
            key_segments,
            dim=0,
            out=self._gpu_layer_flat_attention_keys[:total_attention, 0],
        )
        torch.cat(
            value_segments,
            dim=0,
            out=self._gpu_layer_flat_attention_values[:total_attention, 0],
        )
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(total_lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=self.device,
        )

        for entry, new_length, key, value, entry_lengths in zip(
            entries,
            new_lengths,
            key_states,
            value_states,
            (
                total_lengths[offset : offset + self.num_query_heads_per_kv]
                for offset in range(0, len(total_lengths), self.num_query_heads_per_kv)
            ),
        ):
            self._last_query_head_lengths[entry] = tuple(entry_lengths)
            self._schedule_d2h(entry, old_length, new_length, key, value)

        if self.metrics_enabled:
            self.metrics["append_and_d2h_enqueue_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            self.metrics["cache_update_calls"] += len(entries)
            self.metrics["layer_flat_update_calls"] += 1
            protected = min(self.sink_tokens, old_length) + min(
                self.recent_tokens - 1,
                max(0, old_length - min(self.sink_tokens, old_length)),
            )
            self.metrics["selected_history_tokens_total"] += total_history
            self.metrics["attention_valid_tokens_total"] += total_attention
            self.metrics["attention_valid_tokens_min_sum"] += min(total_lengths)
            self.metrics["attention_valid_tokens_max_sum"] += max(total_lengths)
            self.metrics["selected_history_tokens_per_query_head"] += (
                total_history / len(lengths)
            ) * len(entries)
            self.metrics["selected_adaptive_tokens_total"] += sum(
                max(0, length - protected) for length in lengths
            )
            assert update_started is not None
            self.metrics["cache_update_total_wall_seconds"] += (
                time.perf_counter() - update_started
            )
        return (
            self._gpu_layer_flat_attention_keys[:total_attention],
            self._gpu_layer_flat_attention_values[:total_attention],
            cu_seqlens,
            max(total_lengths),
            total_lengths,
        )

    def get_last_query_head_lengths(self, entry: int) -> Optional[tuple[int, ...]]:
        """Return the valid packed K/V length for each Query head in an entry."""
        return self._last_query_head_lengths[entry]

    def synchronize(self) -> None:
        torch.cuda.current_stream(self.device).synchronize()
        self.eviction_stream.synchronize()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._logical_lengths[0 if layer_idx is None else layer_idx]

    def get_max_length(self) -> Optional[int]:
        return self.allocated_capacity

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        del beam_idx
        raise NotImplementedError("batch-size-1 cache does not support reordering")

    def state_snapshot(self) -> Dict[str, Any]:
        metadata_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensors in (self._page_min, self._page_max)
            for tensor in tensors
            if tensor is not None
        )
        return {
            "logical_length": self.get_seq_length(),
            "allocated_capacity": self.allocated_capacity,
            "initialized_entries": sum(self._initialized),
            "host_slab_bytes": self._host_slab.numel() * self._host_slab.element_size(),
            "host_slab_pinned": self._host_slab.is_pinned(),
            "metadata_gpu_bytes": metadata_bytes,
            "block_size": self.block_size,
            "budget_fraction": self.budget_fraction,
            "selection_interval": self.selection_interval,
            "layer_batched_selection": self.layer_batched_selection,
            "layer_flat_ragged_execution": self.layer_flat_ragged_execution,
            "sparse_gather_stabilized": self.sparse_gather_stabilized,
            "per_head_varlen_attention_reference": (
                self.per_head_varlen_attention_reference
            ),
            "max_selected_history_fraction": self.max_selected_history_fraction,
            "sink_tokens": self.sink_tokens,
            "recent_tokens": self.recent_tokens,
            "num_query_heads_per_kv": self.num_query_heads_per_kv,
        }

    from_legacy_cache = None
    to_legacy_cache = None
