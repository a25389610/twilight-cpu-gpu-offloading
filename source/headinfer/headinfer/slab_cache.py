"""Fixed-capacity pinned-host slab cache for the HeadInfer Llama path.

This is an experimental cache with deliberately narrow scope:

* batch size 1
* greedy decoding
* a fixed maximum cache length
* the head-wise, strictly sequential cache-entry order installed by
  :func:`headinfer.mp.mp_headinfer`

The host K/V storage is allocated once as one flat pinned tensor. Two CUDA
staging slots are used as a ping-pong pair. CUDA events establish these
lifetimes:

1. the main stream records when attention has finished reading an active slot;
2. the eviction stream waits for that event before copying the appended range
   back to the host slab;
3. the prefetch stream waits for both the host write and slot-eviction events
   before reusing a slot;
4. the main stream waits for prefetch completion before exposing an active
   cache view to FlashAttention.

No tensor shape is used as the logical sequence length. Capacity, logical
length, write position, prefetched entry, and residency are tracked explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


@dataclass(frozen=True)
class SlabGeometry:
    """Pure layout calculation, kept CUDA-independent for unit testing."""

    num_entries: int
    batch_size: int
    max_cache_len: int
    head_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_entries", self.num_entries),
            ("batch_size", self.batch_size),
            ("max_cache_len", self.max_cache_len),
            ("head_dim", self.head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def elements_per_entry(self) -> int:
        return self.batch_size * self.max_cache_len * self.head_dim

    @property
    def total_elements(self) -> int:
        return 2 * self.num_entries * self.elements_per_entry

    def bounds(self, kind: str, entry_idx: int) -> Tuple[int, int]:
        if kind not in {"key", "value"}:
            raise ValueError(f"kind must be 'key' or 'value', got {kind!r}")
        if not 0 <= entry_idx < self.num_entries:
            raise IndexError(f"entry_idx {entry_idx} outside [0, {self.num_entries})")
        plane = 0 if kind == "key" else self.num_entries
        start = (plane + entry_idx) * self.elements_per_entry
        return start, start + self.elements_per_entry

    def tensor_shape(self) -> Tuple[int, int, int, int]:
        return (self.batch_size, 1, self.max_cache_len, self.head_dim)


@dataclass
class SequentialCacheState:
    """CUDA-independent logical state for strict head-wise cache updates."""

    num_entries: int
    allocated_capacity: int

    def __post_init__(self) -> None:
        if self.num_entries <= 0 or self.allocated_capacity <= 0:
            raise ValueError("num_entries and allocated_capacity must be positive")
        self.logical_lengths = [0] * self.num_entries
        self.expected_next_entry = 0
        self.write_position = 0
        self.seen_tokens = 0

    def preview(self, entry_idx: int, append_length: int) -> Tuple[int, int]:
        if entry_idx != self.expected_next_entry:
            raise ValueError(
                "strict sequential head-wise update violated: "
                f"expected {self.expected_next_entry}, got {entry_idx}"
            )
        if append_length <= 0:
            raise ValueError("append_length must be positive")
        old_length = self.logical_lengths[entry_idx]
        new_length = old_length + append_length
        if new_length > self.allocated_capacity:
            raise RuntimeError(
                f"cache capacity exceeded: {old_length} + {append_length} > "
                f"{self.allocated_capacity}"
            )
        return old_length, new_length

    def advance(self, entry_idx: int, append_length: int) -> Tuple[int, int]:
        old_length, new_length = self.preview(entry_idx, append_length)
        self.write_position = old_length
        self.logical_lengths[entry_idx] = new_length
        if entry_idx == 0:
            self.seen_tokens += append_length
        self.expected_next_entry = (entry_idx + 1) % self.num_entries
        return old_length, new_length

    @property
    def uniform(self) -> bool:
        return len(set(self.logical_lengths)) == 1


@dataclass
class _SlotState:
    entry_idx: Optional[int] = None
    prefetched_length: int = 0
    prefetch_done: Optional[torch.cuda.Event] = None
    reusable_after: Optional[torch.cuda.Event] = None


class PinnedSlabOffloadedCache(Cache):
    """A fixed-capacity, double-buffered cache for HeadInfer's Llama patch."""

    def __init__(
        self,
        *,
        num_cache_entries: int,
        max_cache_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        device = torch.device(device)
        if batch_size != 1:
            raise ValueError("PinnedSlabOffloadedCache currently supports batch_size=1 only")
        if device.type != "cuda":
            raise ValueError(f"device must be CUDA, got {device}")
        if not torch.cuda.is_available():
            raise RuntimeError("PinnedSlabOffloadedCache requires an operational CUDA driver")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.geometry = SlabGeometry(
            num_entries=num_cache_entries,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            head_dim=head_dim,
        )
        self.dtype = dtype
        self.device = device
        self.allocated_capacity = max_cache_len

        # One host allocation is intentional: allocator power-of-two rounding is
        # paid once rather than independently for every K/V head tensor.
        self._host_slab = torch.empty(
            self.geometry.total_elements,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        slot_shape = self.geometry.tensor_shape()
        self._slot_keys = [
            torch.empty(slot_shape, dtype=dtype, device=device),
            torch.empty(slot_shape, dtype=dtype, device=device),
        ]
        self._slot_values = [
            torch.empty(slot_shape, dtype=dtype, device=device),
            torch.empty(slot_shape, dtype=dtype, device=device),
        ]

        self.prefetch_stream = torch.cuda.Stream(device=device)
        self.eviction_stream = torch.cuda.Stream(device=device)
        self._slots = [_SlotState(), _SlotState()]
        self._host_write_done: List[Optional[torch.cuda.Event]] = [None] * num_cache_entries
        self._bookkeeping = SequentialCacheState(num_cache_entries, max_cache_len)
        self._logical_lengths = self._bookkeeping.logical_lengths
        self._initialized = [False] * num_cache_entries
        self._active_slot: Optional[int] = None
        self._active_entry: Optional[int] = None
        self._active_compute_stream: Optional[torch.cuda.Stream] = None
        self._active_write_start = 0
        self._active_write_end = 0
        self._last_exposed_lengths = [0] * num_cache_entries
        self._seen_tokens = 0
        self.write_position = 0
        self.prefetched_layer: Optional[int] = None
        self.resident_device: Dict[int, str] = {}

    @classmethod
    def from_llama_model(
        cls,
        model: torch.nn.Module,
        *,
        max_cache_len: int,
        batch_size: int = 1,
    ) -> "PinnedSlabOffloadedCache":
        config = model.config
        num_entries = config.num_hidden_layers * config.num_key_value_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        parameter = next(model.parameters())
        return cls(
            num_cache_entries=num_entries,
            max_cache_len=max_cache_len,
            head_dim=head_dim,
            dtype=parameter.dtype,
            device=parameter.device,
            batch_size=batch_size,
        )

    def _host_tensor(self, kind: str, entry_idx: int) -> torch.Tensor:
        start, end = self.geometry.bounds(kind, entry_idx)
        return self._host_slab[start:end].view(self.geometry.tensor_shape())

    def _new_event(self) -> torch.cuda.Event:
        return torch.cuda.Event(blocking=False, interprocess=False)

    def _retire_active(self) -> None:
        """Persist the appended range and make the active slot reusable."""
        if self._active_slot is None or self._active_entry is None:
            return

        slot_idx = self._active_slot
        entry_idx = self._active_entry
        if self._active_compute_stream is None:
            raise RuntimeError("active cache entry has no owning compute stream")
        main_done = self._new_event()
        main_done.record(self._active_compute_stream)

        with torch.cuda.stream(self.eviction_stream):
            self.eviction_stream.wait_event(main_done)
            start, end = self._active_write_start, self._active_write_end
            self._host_tensor("key", entry_idx)[..., start:end, :].copy_(
                self._slot_keys[slot_idx][..., start:end, :],
                non_blocking=True,
            )
            self._host_tensor("value", entry_idx)[..., start:end, :].copy_(
                self._slot_values[slot_idx][..., start:end, :],
                non_blocking=True,
            )
            host_done = self._new_event()
            host_done.record(self.eviction_stream)

        self._host_write_done[entry_idx] = host_done
        self._slots[slot_idx].reusable_after = host_done
        self._initialized[entry_idx] = True
        self.resident_device[entry_idx] = "cpu"
        self._active_slot = None
        self._active_entry = None
        self._active_compute_stream = None

    def _schedule_prefetch(self, entry_idx: int, slot_idx: int) -> None:
        """Load the committed active range into a reusable staging slot."""
        state = self._slots[slot_idx]
        logical_length = self._logical_lengths[entry_idx]
        with torch.cuda.stream(self.prefetch_stream):
            if state.reusable_after is not None:
                self.prefetch_stream.wait_event(state.reusable_after)
            host_ready = self._host_write_done[entry_idx]
            if host_ready is not None:
                self.prefetch_stream.wait_event(host_ready)
            if logical_length:
                self._slot_keys[slot_idx][..., :logical_length, :].copy_(
                    self._host_tensor("key", entry_idx)[..., :logical_length, :],
                    non_blocking=True,
                )
                self._slot_values[slot_idx][..., :logical_length, :].copy_(
                    self._host_tensor("value", entry_idx)[..., :logical_length, :],
                    non_blocking=True,
                )
            ready = self._new_event()
            ready.record(self.prefetch_stream)

        state.entry_idx = entry_idx
        state.prefetched_length = logical_length
        state.prefetch_done = ready
        self.prefetched_layer = entry_idx

    def _find_prefetched_slot(self, entry_idx: int) -> Optional[int]:
        for slot_idx, state in enumerate(self._slots):
            if state.entry_idx == entry_idx and state.prefetched_length == self._logical_lengths[entry_idx]:
                return slot_idx
        return None

    def _choose_slot(self, excluded: Optional[int] = None) -> int:
        for slot_idx in range(len(self._slots)):
            if slot_idx != excluded:
                return slot_idx
        raise RuntimeError("no staging slot is available")

    def _prefetch_successor(self, entry_idx: int, active_slot: int) -> None:
        successor = (entry_idx + 1) % self.geometry.num_entries
        # With a single entry, its new appended range is not committed to the
        # host slab until the next update retires the active attention view.
        # Prefetching it here would therefore read stale host data.
        if successor == entry_idx:
            return
        if self._find_prefetched_slot(successor) is not None:
            return
        slot_idx = self._choose_slot(excluded=active_slot)
        self._schedule_prefetch(successor, slot_idx)

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
            raise ValueError("K/V input dtype differs from slab dtype")
        if key_states.shape[0] != 1 or key_states.shape[1] != 1:
            raise ValueError(f"expected head-wise [1, 1, seq, dim] K/V, got {tuple(key_states.shape)}")
        if key_states.shape[-1] != self.geometry.head_dim:
            raise ValueError("K/V head dimension differs from slab geometry")

        # This call happens after the previous head's FlashAttention invocation
        # has been enqueued on the main stream, so it is the safe retirement
        # boundary for the previously active staging slot.
        self._retire_active()

        append_length = key_states.shape[-2]
        old_length, new_length = self._bookkeeping.preview(layer_idx, append_length)

        slot_idx = self._find_prefetched_slot(layer_idx)
        if slot_idx is None:
            slot_idx = self._choose_slot()
            self._schedule_prefetch(layer_idx, slot_idx)
        state = self._slots[slot_idx]
        if state.prefetch_done is None:
            raise RuntimeError("prefetch event was not recorded")
        torch.cuda.current_stream(self.device).wait_event(state.prefetch_done)

        advanced = self._bookkeeping.advance(layer_idx, append_length)
        if advanced != (old_length, new_length):
            raise AssertionError("cache bookkeeping changed during prefetch")
        self.write_position = self._bookkeeping.write_position
        self._slot_keys[slot_idx][..., old_length:new_length, :].copy_(key_states)
        self._slot_values[slot_idx][..., old_length:new_length, :].copy_(value_states)
        self._seen_tokens = self._bookkeeping.seen_tokens

        self._active_slot = slot_idx
        self._active_entry = layer_idx
        self._active_compute_stream = torch.cuda.current_stream(self.device)
        self._active_write_start = old_length
        self._active_write_end = new_length
        self._last_exposed_lengths[layer_idx] = new_length
        self.resident_device[layer_idx] = str(self.device)

        # The other slot can prefetch the next head while FlashAttention reads
        # the active view on the main stream.
        self._prefetch_successor(layer_idx, active_slot=slot_idx)

        active_key = self._slot_keys[slot_idx][..., :new_length, :]
        active_value = self._slot_values[slot_idx][..., :new_length, :]
        if not active_key.is_contiguous() or not active_value.is_contiguous():
            raise AssertionError("active K/V views must be contiguous for FlashAttention")
        return active_key, active_value

    def synchronize(self) -> None:
        """Commit the last active entry and wait for all cache transfers."""
        self._retire_active()
        torch.cuda.current_stream(self.device).synchronize()
        self.prefetch_stream.synchronize()
        self.eviction_stream.synchronize()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        return self._logical_lengths[layer_idx]

    def get_max_length(self) -> Optional[int]:
        return self.allocated_capacity

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError(
            "PinnedSlabOffloadedCache intentionally supports batch-size-1 greedy decoding only; "
            "beam-search cache reorder is not implemented"
        )

    def __len__(self) -> int:
        return sum(self._initialized)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.geometry.num_entries:
            raise KeyError(layer_idx)
        host_ready = self._host_write_done[layer_idx]
        if host_ready is not None:
            host_ready.synchronize()
        length = self._logical_lengths[layer_idx]
        return (
            self._host_tensor("key", layer_idx)[..., :length, :],
            self._host_tensor("value", layer_idx)[..., :length, :],
        )

    @property
    def key_cache(self) -> List[torch.Tensor]:
        return [self[index][0] for index in range(self.geometry.num_entries) if self._initialized[index]]

    @property
    def value_cache(self) -> List[torch.Tensor]:
        return [self[index][1] for index in range(self.geometry.num_entries) if self._initialized[index]]

    def state_snapshot(self) -> Dict[str, Any]:
        residency = {
            index: ("cpu" if self._initialized[index] else "uninitialized")
            for index in range(self.geometry.num_entries)
        }
        prefetched_layers = []
        for slot_idx, slot in enumerate(self._slots):
            if slot.entry_idx is None:
                continue
            prefetched_layers.append(slot.entry_idx)
            residency[slot.entry_idx] = (
                f"{residency[slot.entry_idx]}+{self.device}:slot{slot_idx}"
            )
        if self._active_entry is not None:
            residency[self._active_entry] = (
                f"{residency[self._active_entry]}+{self.device}:active"
            )
        return {
            "logical_length": self.get_seq_length(),
            "logical_lengths_uniform": self._bookkeeping.uniform,
            "logical_lengths_min": min(self._logical_lengths),
            "logical_lengths_max": max(self._logical_lengths),
            "last_exposed_lengths": list(self._last_exposed_lengths),
            "uninitialized_range_exposed": any(
                exposed != logical
                for exposed, logical in zip(
                    self._last_exposed_lengths,
                    self._logical_lengths,
                )
            ),
            "allocated_capacity": self.allocated_capacity,
            "write_position": self.write_position,
            "prefetched_layer": self.prefetched_layer,
            "prefetched_layers": prefetched_layers,
            "resident_device": residency,
            "initialized_entries": sum(self._initialized),
            "expected_entries": self.geometry.num_entries,
            "active_slot": self._active_slot,
            "active_entry": self._active_entry,
            "host_slab_bytes": self._host_slab.numel() * self._host_slab.element_size(),
            "host_slab_pinned": self._host_slab.is_pinned(),
            "staging_slot_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in self._slot_keys + self._slot_values
            ),
        }

    from_legacy_cache = None
    to_legacy_cache = None
