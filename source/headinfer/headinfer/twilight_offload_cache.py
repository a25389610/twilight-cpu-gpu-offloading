"""Twilight-inspired adaptive token selection for CPU--GPU KV offloading.

This correctness-first Batch=1 prototype composes Quest's conservative page
selector with per-token affine INT4 Key QK estimates and Top-p pruning.  It
supports both the historical matched-H2D control and raw dynamic Top-p budgets.

The default uses PyTorch operators. Opt-in local Triton backends fuse candidate
preparation (triton_prepare) or preparation and QK (triton, experimental due to
reduction-order differences). These are not official Twilight kernels.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Optional

import numpy as np
import torch

from .quest_offload_cache import QuestTopKOffloadedCache


def affine_int4_pack(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack per-token min/max affine INT4 keys along the final dimension."""
    if keys.ndim != 2 or keys.shape[-1] % 2:
        raise ValueError("keys must be [tokens, even head_dim]")
    minimum = keys.amin(dim=-1)
    maximum = keys.amax(dim=-1)
    scale = ((maximum - minimum).float().clamp_min(1e-9) / 15.0).to(keys.dtype)
    codes = torch.round(
        (keys.float() - minimum.float().unsqueeze(-1))
        / scale.float().unsqueeze(-1)
    ).clamp_(0, 15).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scale, minimum


def affine_int4_dequantize(
    packed: torch.Tensor, scale: torch.Tensor, minimum: torch.Tensor
) -> torch.Tensor:
    """Dequantize tensors produced by :func:`affine_int4_pack`."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).flatten(-2).float()
    return codes * scale.float().unsqueeze(-1) + minimum.float().unsqueeze(-1)


def matched_budget_counts(
    desired: list[int], target_total: int, capacities: list[int]
) -> list[int]:
    """Proportionally rescale Top-p counts to an exact positive total budget."""
    if not desired or len(desired) != len(capacities):
        raise ValueError("desired and capacities must be non-empty and aligned")
    if any(value <= 0 for value in desired) or any(value <= 0 for value in capacities):
        raise ValueError("desired counts and capacities must be positive")
    if target_total < len(desired) or target_total > sum(capacities):
        raise ValueError("target_total is infeasible")

    remaining = target_total - len(desired)
    weights = [float(value) for value in desired]
    weight_sum = sum(weights)
    raw_extra = [remaining * weight / weight_sum for weight in weights]
    counts = [1 + min(capacities[i] - 1, int(math.floor(raw_extra[i]))) for i in range(len(desired))]

    while sum(counts) < target_total:
        candidates = [
            i for i in range(len(counts)) if counts[i] < capacities[i]
        ]
        if not candidates:
            raise AssertionError("unable to fill matched budget")
        best = max(
            candidates,
            key=lambda i: (raw_extra[i] - math.floor(raw_extra[i]), desired[i], -i),
        )
        counts[best] += 1
        raw_extra[best] = math.floor(raw_extra[best])

    while sum(counts) > target_total:
        candidates = [i for i in range(len(counts)) if counts[i] > 1]
        if not candidates:
            raise AssertionError("unable to reduce matched budget")
        worst = min(candidates, key=lambda i: (desired[i], i))
        counts[worst] -= 1
    return counts


def matched_budget_counts_tensor(
    desired: torch.Tensor,
    target_total: int,
    capacity: int,
) -> torch.Tensor:
    """Vectorized exact matched-budget allocation for unsaturated GQA groups.

    The deployed Twilight configuration has one shared candidate capacity for
    every Query head and a per-group target no larger than that capacity.  In
    that regime the capacity clamp in :func:`matched_budget_counts` cannot
    activate, so the original largest-remainder rule can be expressed without
    a CPU synchronization.  Stable sorts preserve its tie order:
    fractional remainder, desired count, then lower head index.
    """
    if desired.ndim != 2 or desired.shape[1] <= 0:
        raise ValueError("desired must be [groups, query_heads]")
    heads = int(desired.shape[1])
    if target_total < heads or target_total > capacity:
        raise ValueError("target_total must be between head count and capacity")

    desired_f64 = desired.to(torch.float64)
    remaining = target_total - heads
    raw_extra = remaining * desired_f64 / desired_f64.sum(dim=-1, keepdim=True)
    floor_extra = torch.floor(raw_extra).to(torch.long)
    counts = 1 + floor_extra
    deficit = target_total - counts.sum(dim=-1)

    # Start in natural index order.  The first stable sort supplies the
    # secondary desired-count key; the second supplies the primary fractional
    # remainder key while retaining desired/index tie ordering.
    priority = torch.argsort(desired, dim=-1, descending=True, stable=True)
    fractions = raw_extra - torch.floor(raw_extra)
    ordered_fractions = torch.gather(fractions, -1, priority)
    fractional_order = torch.argsort(
        ordered_fractions, dim=-1, descending=True, stable=True
    )
    priority = torch.gather(priority, -1, fractional_order)
    ranks = torch.empty_like(priority)
    rank_values = torch.arange(heads, device=desired.device).expand_as(priority)
    ranks.scatter_(-1, priority, rank_values)
    counts += ranks.lt(deficit.unsqueeze(-1)).to(counts.dtype)
    return counts


class TwilightMatchedBudgetOffloadedCache(QuestTopKOffloadedCache):
    """Quest B0 + INT4 QK + Top-p with matched or raw-dynamic budgets.

    The class name is retained for artifact compatibility.  ``budget_mode``
    defaults to ``matched`` so all existing callers preserve their behavior.
    """

    def __init__(
        self,
        *,
        candidate_token_budget: int = 8192,
        top_p: float = 0.95,
        matched_budget_fraction: float = 0.05,
        budget_mode: str = "matched",
        gqa_groupwise_execution: bool = False,
        detailed_selection_profile: bool = False,
        qk_backend: str = "pytorch",
        exact_scan_graph: bool = False,
        exact_top_p_graph: bool = False,
        cpu_flat_gather: bool = False,
        cpu_bitmap_union: bool = False,
        cpu_native_gather: bool = False,
        direct_attention_layout: bool = False,
        early_gpu_metadata: bool = False,
        gather_h2d_chunks: int = 0,
        reuse_quant_metadata: bool = False,
        fused_final_indices: bool = False,
        fused_quest_score: bool = False,
        skip_unused_host_views: bool = False,
        cpu_run_gather: bool = False,
        previous_token_resident_cache: bool = False,
        resident_mapping_backend: str = "torch",
        resident_hit_copy_backend: str = "torch",
        resident_snapshot_backend: str = "torch",
        resident_position_snapshot_backend: str = "clone",
        resident_pack_hit_indices: bool = False,
        resident_zero_copy: bool = False,
        resident_precompute_gpu_mapping: bool = False,
        vram_alias_history_staging: bool = False,
        vram_lazy_gpu_pack: bool = False,
        vram_ondemand_attention_capacity: bool = False,
        vram_tight_resident_capacity: bool = False,
        gpu_compact_gqa_union: bool = False,
        gpu_union_validate_cpu: bool = False,
        batched_new_kv_d2h: bool = False,
        new_kv_d2h_granularity: str = "token",
        post_selection_pipeline: bool = False,
        post_selection_groups_per_chunk: int = 1,
        **kwargs: Any,
    ) -> None:
        if candidate_token_budget <= 0:
            raise ValueError("candidate_token_budget must be positive")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 < matched_budget_fraction <= 1.0:
            raise ValueError("matched_budget_fraction must be in (0, 1]")
        if budget_mode not in {"matched", "dynamic"}:
            raise ValueError("budget_mode must be 'matched' or 'dynamic'")
        num_query_heads_per_kv = int(kwargs["num_query_heads_per_kv"])
        layer_batched_selection = bool(kwargs.get("layer_batched_selection", False))
        if qk_backend not in {"pytorch", "triton", "triton_prepare"}:
            raise ValueError("unknown Twilight QK backend")
        if qk_backend != "pytorch" and not layer_batched_selection:
            raise ValueError("fused QK requires layer-batched selection")
        self.qk_backend = qk_backend
        self.exact_scan_graph = bool(exact_scan_graph)
        self._exact_scan_graph = None
        self.exact_top_p_graph = bool(exact_top_p_graph)
        self._exact_top_p_graph = None
        if self.exact_scan_graph and self.exact_top_p_graph:
            raise ValueError("select one exact Top-p graph scope")
        self.cpu_flat_gather = bool(cpu_flat_gather)
        self.cpu_bitmap_union = bool(cpu_bitmap_union)
        self.cpu_native_gather = bool(cpu_native_gather)
        self.direct_attention_layout = bool(direct_attention_layout)
        self.early_gpu_metadata = bool(early_gpu_metadata)
        self.gather_h2d_chunks = int(gather_h2d_chunks)
        self.reuse_quant_metadata = bool(reuse_quant_metadata)
        self.fused_final_indices = bool(fused_final_indices)
        self.fused_quest_score = bool(fused_quest_score)
        self.skip_unused_host_views = bool(skip_unused_host_views)
        self.cpu_run_gather = bool(cpu_run_gather)
        self.previous_token_resident_cache = bool(previous_token_resident_cache)
        self.resident_zero_copy = bool(resident_zero_copy)
        self.resident_precompute_gpu_mapping = bool(resident_precompute_gpu_mapping)
        if self.resident_precompute_gpu_mapping and not self.resident_zero_copy:
            raise ValueError("precomputed GPU mapping requires mapped resident fast path")
        self.vram_alias_history_staging = bool(vram_alias_history_staging)
        self.vram_lazy_gpu_pack = bool(vram_lazy_gpu_pack)
        self.vram_ondemand_attention_capacity = bool(vram_ondemand_attention_capacity)
        self.vram_tight_resident_capacity = bool(vram_tight_resident_capacity)
        if any((self.vram_alias_history_staging, self.vram_lazy_gpu_pack,
                self.vram_ondemand_attention_capacity,
                self.vram_tight_resident_capacity)) and not self.resident_zero_copy:
            raise ValueError("VRAM audit options require the mapped resident fast path")
        if self.resident_zero_copy and not previous_token_resident_cache:
            raise ValueError("resident zero-copy requires resident cache")
        if resident_mapping_backend not in {"torch", "native", "native_compact", "gpu_bitmap"}:
            raise ValueError("unknown resident mapping backend")
        if resident_mapping_backend != "torch" and not previous_token_resident_cache:
            raise ValueError("native resident mapping requires resident cache")
        self.resident_mapping_backend = resident_mapping_backend
        if resident_hit_copy_backend not in {"torch", "triton"}:
            raise ValueError("unknown resident hit copy backend")
        if resident_hit_copy_backend != "torch" and not previous_token_resident_cache:
            raise ValueError("Triton resident hit copy requires resident cache")
        self.resident_hit_copy_backend = resident_hit_copy_backend
        if (resident_mapping_backend in {"native_compact", "gpu_bitmap"}
                and resident_hit_copy_backend != "triton"):
            raise ValueError("compact hit indices require Triton hit copy")
        if resident_snapshot_backend not in {"torch", "triton", "contiguous"}:
            raise ValueError("unknown resident snapshot backend")
        if resident_snapshot_backend != "torch" and not previous_token_resident_cache:
            raise ValueError("alternate resident snapshot requires resident cache")
        if resident_snapshot_backend == "contiguous" and (
            resident_mapping_backend not in {"native_compact", "gpu_bitmap"}
            or resident_hit_copy_backend != "triton"
        ):
            raise ValueError(
                "contiguous snapshot requires native_compact mapping and Triton hit copy"
            )
        self.resident_snapshot_backend = resident_snapshot_backend
        if resident_position_snapshot_backend not in {"clone", "reference"}:
            raise ValueError("unknown resident position snapshot backend")
        if resident_position_snapshot_backend == "reference" and (
            resident_snapshot_backend != "contiguous" or not gpu_compact_gqa_union
        ):
            raise ValueError(
                "reference position snapshot requires contiguous resident and GPU union"
            )
        self.resident_position_snapshot_backend = resident_position_snapshot_backend
        if self.resident_zero_copy and not (
            resident_mapping_backend == "gpu_bitmap"
            and resident_snapshot_backend == "contiguous"
            and resident_position_snapshot_backend == "reference"
            and gpu_compact_gqa_union
        ):
            raise ValueError("resident zero-copy requires GPU bitmap mapping and contiguous GPU union")
        if resident_pack_hit_indices and (
            resident_snapshot_backend != "contiguous"
            or resident_mapping_backend != "native_compact"
        ):
            raise ValueError(
                "packed hit indices require contiguous resident and native_compact mapping"
            )
        self.resident_pack_hit_indices = bool(resident_pack_hit_indices)
        self._native_resident_mapping = None
        if resident_mapping_backend in {"native", "native_compact"}:
            from .resident_mapping import NativeResidentMapping
            self._native_resident_mapping = NativeResidentMapping(
                int(kwargs["max_cache_len"]),
                compact_hits=resident_mapping_backend == "native_compact",
                attention_layout=resident_snapshot_backend == "contiguous",
                pack_hits=self.resident_pack_hit_indices,
            )
        self.gpu_compact_gqa_union = bool(gpu_compact_gqa_union)
        self.gpu_union_validate_cpu = bool(gpu_union_validate_cpu)
        self.batched_new_kv_d2h = bool(batched_new_kv_d2h)
        self.post_selection_pipeline = bool(post_selection_pipeline)
        self.post_selection_groups_per_chunk = int(
            post_selection_groups_per_chunk
        )
        if new_kv_d2h_granularity not in {"layer", "token"}:
            raise ValueError("new KV D2H granularity must be 'layer' or 'token'")
        self.new_kv_d2h_granularity = new_kv_d2h_granularity
        if self.gpu_compact_gqa_union and not gqa_groupwise_execution:
            raise ValueError("GPU compact union requires GQA group execution")
        if self.gpu_union_validate_cpu and not self.gpu_compact_gqa_union:
            raise ValueError("GPU union CPU validation requires GPU compact union")
        if self.post_selection_groups_per_chunk <= 0:
            raise ValueError("post-selection groups per chunk must be positive")
        if self.post_selection_pipeline and not (
            self.gpu_compact_gqa_union
            and gqa_groupwise_execution
            and cpu_flat_gather
            and direct_attention_layout
            and gather_h2d_chunks > 0
            and not cpu_native_gather
            and not cpu_run_gather
            and not previous_token_resident_cache
        ):
            raise ValueError(
                "post-selection pipeline requires the current GPU-union "
                "GQA flat/direct/chunk path without native/run gather or "
                "resident reuse"
            )
        if self.cpu_run_gather:
            if not (cpu_flat_gather and direct_attention_layout and gqa_groupwise_execution and gather_h2d_chunks > 0) or cpu_native_gather:
                raise ValueError('run gather requires flat/direct/GQA chunk pipeline without native gather')
            from .cpu_kv_run_gather import CpuKVRunGather
            self._cpu_run_gather = CpuKVRunGather()
        if self.fused_quest_score and not layer_batched_selection:
            raise ValueError('fused Quest score requires layer-batched selection')
        self._layer_quant_metadata = {}
        if self.gather_h2d_chunks < 0:
            raise ValueError('gather_h2d_chunks must be nonnegative')
        if self.gather_h2d_chunks and not (cpu_flat_gather and direct_attention_layout and gqa_groupwise_execution) :
            raise ValueError('chunk pipeline requires flat gather, direct layout and GQA execution')
        if self.gather_h2d_chunks and cpu_native_gather:
            raise ValueError('chunk pipeline pilot uses PyTorch gather only')
        if self.early_gpu_metadata and not gqa_groupwise_execution:
            raise ValueError('early GPU metadata requires GQA group execution')
        if (cpu_native_gather or direct_attention_layout) and not (cpu_flat_gather and gqa_groupwise_execution):
            raise ValueError('native gather/direct layout requires flat gather and GQA group execution')
        if self.cpu_native_gather:
            from .cpu_kv_gather import CpuKVGather
            self._cpu_kv_gather = CpuKVGather()
        if self.cpu_flat_gather and not gqa_groupwise_execution:
            raise ValueError("cpu_flat_gather requires GQA group execution")
        if self.previous_token_resident_cache and not (
            gqa_groupwise_execution
            and cpu_flat_gather
            and direct_attention_layout
            and gather_h2d_chunks > 0
            and not cpu_native_gather
            and not cpu_run_gather
        ):
            raise ValueError(
                "previous-token resident cache requires the current GQA "
                "flat/direct/chunk path without native/run gather"
            )
        layer_flat_ragged_execution = bool(
            kwargs.get("layer_flat_ragged_execution", False)
        )
        if gqa_groupwise_execution and not layer_batched_selection:
            raise ValueError(
                "GQA group-wise execution requires layer-batched selection"
            )
        if gqa_groupwise_execution and layer_flat_ragged_execution:
            raise ValueError(
                "GQA group-wise and per-Query-head flat execution are exclusive"
            )
        kwargs["budget_fraction"] = matched_budget_fraction
        if budget_mode == "dynamic":
            max_cache_len = int(kwargs["max_cache_len"])
            sink_tokens = int(kwargs.get("sink_tokens", 64))
            block_size = int(kwargs.get("block_size", 16))
            max_pages = math.ceil(max(0, max_cache_len - sink_tokens) / block_size)
            required_pages = min(
                max_pages, math.ceil(candidate_token_budget / block_size)
            )
            kwargs["max_selected_history_fraction"] = (
                required_pages / max_pages if max_pages else 1.0
            )
        else:
            kwargs["max_selected_history_fraction"] = min(
                1.0, matched_budget_fraction * num_query_heads_per_kv
            )
        super().__init__(**kwargs)
        if self.cpu_bitmap_union:
            if not gqa_groupwise_execution:
                raise ValueError("cpu_bitmap_union requires GQA group execution")
            from .cpu_token_union import CpuTokenUnion
            self._cpu_token_union = CpuTokenUnion(self.geometry.max_cache_len)
        self.candidate_token_budget = int(candidate_token_budget)
        self.top_p = float(top_p)
        self.matched_budget_fraction = float(matched_budget_fraction)
        self.budget_mode = budget_mode
        self.gqa_groupwise_execution = bool(gqa_groupwise_execution)
        self.detailed_selection_profile = bool(detailed_selection_profile)
        self._quant_key_packed: list[Optional[torch.Tensor]] = [
            None
        ] * self.geometry.num_entries
        self._quant_key_scale: list[Optional[torch.Tensor]] = [
            None
        ] * self.geometry.num_entries
        self._quant_key_minimum: list[Optional[torch.Tensor]] = [
            None
        ] * self.geometry.num_entries
        self._cached_twilight_positions: list[
            Optional[tuple[torch.Tensor, ...]]
        ] = [None] * self.geometry.num_entries
        self._cached_twilight_group_positions: list[Optional[torch.Tensor]] = [
            None
        ] * self.geometry.num_entries
        self._cached_twilight_per_q_lengths: list[
            Optional[tuple[int, ...]]
        ] = [None] * self.geometry.num_entries
        self._gpu_group_membership: Optional[torch.Tensor] = None
        self._post_selection_host_membership: Optional[torch.Tensor] = None
        self._post_selection_pending: dict[int, dict[str, Any]] = {}
        self._post_selection_host_starts: dict[int, float] = {}
        self._post_selection_timeline_events: list[dict[str, Any]] = []
        self._post_selection_timeline: list[dict[str, Any]] = []
        self._post_selection_stream = (
            torch.cuda.Stream(device=self.device)
            if self.post_selection_pipeline
            else None
        )
        self._selected_kv_h2d_stream = (
            torch.cuda.Stream(device=self.device)
            if self.post_selection_pipeline
            else None
        )
        self._capture_budget_trace = False
        self._budget_trace: list[dict[str, Any]] = []
        self._group_union_trace: list[dict[str, Any]] = []
        self._resident_positions: dict[int, tuple[torch.Tensor, ...]] = {}
        self._resident_zero_copy_pending_counts: list[tuple[dict[str, Any], torch.Tensor]] = []
        self._resident_precomputed_maps: dict[int, tuple] = {}
        self._resident_zero_copy_placeholder_active = False
        self._resident_zero_copy_kernel = None
        self._resident_bitmaps: dict[int, torch.Tensor] = {}
        self._resident_miss_positions_host: Optional[torch.Tensor] = None
        self._resident_keys: dict[int, torch.Tensor] = {}
        self._resident_values: dict[int, torch.Tensor] = {}
        self._resident_capacities: dict[int, int] = {}
        # Diagnostic accounting only.  These are logical live payload bounds
        # for the opt-in prototype's temporary GPU index/gather tensors; they
        # are intentionally not presented as PyTorch allocator peak bytes.
        self._resident_index_logical_peak_bytes = 0
        self._resident_hit_gather_logical_peak_bytes = 0
        self._resident_reuse_trace: list[dict[str, Any]] = []
        self._capture_resident_attention_trace = False
        self._resident_attention_trace_steps: set[int] = set()
        self._resident_attention_trace: list[dict[str, Any]] = []
        self._capture_new_kv_trace = False
        self._new_kv_trace: list[dict[str, Any]] = []
        self._new_kv_trace_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        self._new_kv_trace_pending_read: dict[
            int, tuple[int, int, str, str]
        ] = {}
        self._new_kv_control_pending: dict[int, tuple[int, int]] = {}
        self._batched_new_kv_pending: dict[
            int, tuple[torch.cuda.Event, int, int, tuple[int, ...]]
        ] = {}
        if self.batched_new_kv_d2h:
            staging_shape = (
                self.geometry.num_entries,
                self.geometry.head_dim,
            )
            self._new_kv_gpu_keys = torch.empty(
                staging_shape, dtype=self.dtype, device=self.device
            )
            self._new_kv_gpu_values = torch.empty(
                staging_shape, dtype=self.dtype, device=self.device
            )
            self._new_kv_host_keys = torch.empty(
                staging_shape, dtype=self.dtype, device="cpu", pin_memory=True
            )
            self._new_kv_host_values = torch.empty(
                staging_shape, dtype=self.dtype, device="cpu", pin_memory=True
            )
        else:
            self._new_kv_gpu_keys = None
            self._new_kv_gpu_values = None
            self._new_kv_host_keys = None
            self._new_kv_host_values = None

    def enable_new_kv_trace(self, enabled: bool = True) -> None:
        """Capture exact host-slab new-K/V hashes; diagnostic-only."""
        self._capture_new_kv_trace = bool(enabled)
        self._new_kv_trace = []
        self._new_kv_trace_by_key = {}
        self._new_kv_trace_pending_read = {}

    def new_kv_trace(self) -> list[dict[str, Any]]:
        return list(self._new_kv_trace)

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        value = tensor.detach().to(device="cpu").contiguous()
        return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()

    def _capture_new_kv_host_rows(
        self,
        entries: tuple[int, ...],
        *,
        decode_step: int,
        position: int,
    ) -> None:
        if not self._capture_new_kv_trace:
            return
        for entry in entries:
            key_row = self._host_tensor("key", entry)[0, 0, position]
            value_row = self._host_tensor("value", entry)[0, 0, position]
            key_hash = self._tensor_sha256(key_row)
            value_hash = self._tensor_sha256(value_row)
            record = {
                "decode_step": decode_step,
                "layer": entry // 8,
                "kv_head": entry % 8,
                "entry": entry,
                "position": position,
                "key_sha256": key_hash,
                "value_sha256": value_hash,
                "verified_before_next_read": False,
                "consumer_decode_step": None,
            }
            self._new_kv_trace.append(record)
            self._new_kv_trace_by_key[(decode_step, entry)] = record
            self._new_kv_trace_pending_read[entry] = (
                decode_step,
                position,
                key_hash,
                value_hash,
            )

    def _verify_new_kv_before_read(self, entries: list[int], decode_step: int) -> None:
        if not self._capture_new_kv_trace:
            return
        for entry in entries:
            pending = self._new_kv_trace_pending_read.pop(entry, None)
            if pending is None:
                continue
            prior_step, position, expected_key, expected_value = pending
            actual_key = self._tensor_sha256(
                self._host_tensor("key", entry)[0, 0, position]
            )
            actual_value = self._tensor_sha256(
                self._host_tensor("value", entry)[0, 0, position]
            )
            if actual_key != expected_key or actual_value != expected_value:
                raise AssertionError("new KV host row changed before next-token read")
            record = self._new_kv_trace_by_key[(prior_step, entry)]
            record["verified_before_next_read"] = True
            record["consumer_decode_step"] = decode_step

    def _scatter_batched_new_kv_to_host(
        self,
        *,
        start_entry: int,
        entries: tuple[int, ...],
        position: int,
    ) -> None:
        if self._new_kv_host_keys is None or self._new_kv_host_values is None:
            raise AssertionError("new KV staging buffers are missing")
        count = len(entries)
        if entries != tuple(range(start_entry, start_entry + count)):
            raise AssertionError("batched new KV entries must be contiguous")
        element_stride = self.geometry.max_cache_len * self.geometry.head_dim
        key_offset = start_entry * element_stride + position * self.geometry.head_dim
        value_offset = (
            (self.geometry.num_entries + start_entry) * element_stride
            + position * self.geometry.head_dim
        )
        key_destination = torch.as_strided(
            self._host_slab,
            size=(count, self.geometry.head_dim),
            stride=(element_stride, 1),
            storage_offset=key_offset,
        )
        value_destination = torch.as_strided(
            self._host_slab,
            size=(count, self.geometry.head_dim),
            stride=(element_stride, 1),
            storage_offset=value_offset,
        )
        key_destination.copy_(self._new_kv_host_keys[start_entry:start_entry + count])
        value_destination.copy_(
            self._new_kv_host_values[start_entry:start_entry + count]
        )

    def _flush_batched_new_kv(self, batch_id: int) -> None:
        pending = self._batched_new_kv_pending.pop(batch_id, None)
        if pending is None:
            return
        done, decode_step, position, entries = pending
        normal_timeline = bool(
            getattr(self, "_normal_overlap_timeline_enabled", False)
        )
        normal_host_origin = getattr(
            self, "_normal_overlap_host_origin", None
        )
        wait_started = time.perf_counter() if self.metrics_enabled else None
        normal_wait_started = time.perf_counter() if normal_timeline else None
        done.synchronize()
        normal_wait_finished = time.perf_counter() if normal_timeline else None
        if wait_started is not None:
            self.metrics["host_ready_wait_wall_seconds"] += (
                time.perf_counter() - wait_started
            )
        scatter_started = time.perf_counter() if self.metrics_enabled else None
        normal_scatter_started = time.perf_counter() if normal_timeline else None
        self._scatter_batched_new_kv_to_host(
            start_entry=entries[0], entries=entries, position=position
        )
        normal_scatter_finished = time.perf_counter() if normal_timeline else None
        if scatter_started is not None:
            self.metrics["new_kv_host_scatter_wall_seconds"] += (
                time.perf_counter() - scatter_started
            )
            self.metrics["new_kv_host_scatter_calls"] += 2
        if normal_timeline:
            if (
                normal_host_origin is None
                or normal_wait_started is None
                or normal_wait_finished is None
                or normal_scatter_started is None
                or normal_scatter_finished is None
            ):
                raise AssertionError("normal-overlap new-KV host timing is missing")
            matching = [
                item
                for item in self._normal_overlap_new_kv_events
                if item["decode_step"] == decode_step
                and item["position"] == position
                and "host_wait_start_ms" not in item
            ]
            if len(matching) != 1:
                raise AssertionError("normal-overlap new-KV event match failed")
            matching[0].update(
                {
                    "host_wait_start_ms": (
                        normal_wait_started - normal_host_origin
                    )
                    * 1000.0,
                    "host_wait_end_ms": (
                        normal_wait_finished - normal_host_origin
                    )
                    * 1000.0,
                    "host_scatter_start_ms": (
                        normal_scatter_started - normal_host_origin
                    )
                    * 1000.0,
                    "host_scatter_end_ms": (
                        normal_scatter_finished - normal_host_origin
                    )
                    * 1000.0,
                }
            )
        self._capture_new_kv_host_rows(
            entries, decode_step=decode_step, position=position
        )

    def _flush_all_batched_new_kv(self) -> None:
        for batch_id in sorted(tuple(self._batched_new_kv_pending)):
            self._flush_batched_new_kv(batch_id)

    def _schedule_batched_new_kv_d2h(
        self,
        *,
        entries: list[int],
        old_length: int,
        key_states: list[torch.Tensor],
        value_states: list[torch.Tensor],
    ) -> None:
        if any(
            tensor is None
            for tensor in (
                self._new_kv_gpu_keys,
                self._new_kv_gpu_values,
                self._new_kv_host_keys,
                self._new_kv_host_values,
            )
        ):
            raise AssertionError("new KV staging buffers are missing")
        start_entry = entries[0]
        stop_entry = entries[-1] + 1
        if entries != list(range(start_entry, stop_entry)):
            raise AssertionError("layer entries must be contiguous")
        decode_step = (
            old_length - self._prompt_length + 1
            if self._prompt_length is not None
            else old_length + 1
        )
        pack_started = time.perf_counter() if self.metrics_enabled else None
        pack_event_start = self._event(timing=True) if self.metrics_enabled else None
        if pack_event_start is not None:
            pack_event_start.record(torch.cuda.current_stream(self.device))
        torch.cat(
            [key[0, 0] for key in key_states],
            dim=0,
            out=self._new_kv_gpu_keys[start_entry:stop_entry],
        )
        torch.cat(
            [value[0, 0] for value in value_states],
            dim=0,
            out=self._new_kv_gpu_values[start_entry:stop_entry],
        )
        if self.metrics_enabled:
            pack_event_end = self._event(timing=True)
            pack_event_end.record(torch.cuda.current_stream(self.device))
            self._timing_events.append(
                ("new_kv_gpu_staging", pack_event_start, pack_event_end)
            )
            self.metrics["new_kv_gpu_staging_wall_seconds"] += (
                time.perf_counter() - pack_started
            )
            self.metrics["new_kv_gpu_staging_calls"] += 2

        if self.new_kv_d2h_granularity == "token" and stop_entry < self.geometry.num_entries:
            return
        if self.new_kv_d2h_granularity == "token":
            batch_id = 0
            batch_entries = tuple(range(self.geometry.num_entries))
            batch_start = 0
            batch_stop = self.geometry.num_entries
        else:
            batch_id = start_entry // len(entries)
            batch_entries = tuple(entries)
            batch_start = start_entry
            batch_stop = stop_entry
        if batch_id in self._batched_new_kv_pending:
            self._flush_batched_new_kv(batch_id)

        enqueue_started = time.perf_counter() if self.metrics_enabled else None
        compute_stream = torch.cuda.current_stream(self.device)
        ready = self._event()
        ready.record(compute_stream)
        with torch.cuda.stream(self.eviction_stream):
            self.eviction_stream.wait_event(ready)
            normal_d2h_start = (
                self._event(timing=True)
                if getattr(self, "_normal_overlap_timeline_enabled", False)
                else None
            )
            if normal_d2h_start is not None:
                normal_d2h_start.record(self.eviction_stream)
            if self.metrics_enabled:
                d2h_start = self._event(timing=True)
                d2h_end = self._event(timing=True)
                d2h_start.record(self.eviction_stream)
            self._new_kv_host_keys[batch_start:batch_stop].copy_(
                self._new_kv_gpu_keys[batch_start:batch_stop], non_blocking=True
            )
            self._new_kv_host_values[batch_start:batch_stop].copy_(
                self._new_kv_gpu_values[batch_start:batch_stop], non_blocking=True
            )
            normal_d2h_end = (
                self._event(timing=True) if normal_d2h_start is not None else None
            )
            if normal_d2h_end is not None:
                normal_d2h_end.record(self.eviction_stream)
            if self.metrics_enabled:
                d2h_end.record(self.eviction_stream)
                self._timing_events.append(("new_kv_d2h", d2h_start, d2h_end))
            done = self._event()
            done.record(self.eviction_stream)
        self._new_kv_gpu_keys[batch_start:batch_stop].record_stream(
            self.eviction_stream
        )
        self._new_kv_gpu_values[batch_start:batch_stop].record_stream(
            self.eviction_stream
        )
        self._batched_new_kv_pending[batch_id] = (
            done,
            decode_step,
            old_length,
            batch_entries,
        )
        if normal_d2h_start is not None and normal_d2h_end is not None:
            self._normal_overlap_new_kv_events.append(
                {
                    "decode_step": decode_step,
                    "position": old_length,
                    "entries": len(batch_entries),
                    "d2h_start_event": normal_d2h_start,
                    "d2h_end_event": normal_d2h_end,
                }
            )
        for entry in batch_entries:
            self._host_write_done[entry] = done
        if self.metrics_enabled:
            elements = 2 * len(batch_entries) * self.geometry.head_dim
            self.metrics["d2h_bytes"] += elements * self._new_kv_gpu_keys.element_size()
            self.metrics["d2h_copy_calls"] += 2
            self.metrics["new_kv_d2h_ready_events"] += 1
            self.metrics["new_kv_d2h_done_events"] += 1
            self.metrics["new_kv_d2h_enqueue_wall_seconds"] += (
                time.perf_counter() - enqueue_started
            )

    def enable_selection_trace(self, enabled: bool = True) -> None:
        super().enable_selection_trace(enabled)
        self._group_union_trace = []

    def group_union_trace(self) -> list[dict[str, Any]]:
        return list(self._group_union_trace)

    def resident_reuse_trace(self) -> list[dict[str, Any]]:
        """Return lightweight per-layer hit/miss rows for the opt-in prototype."""
        if self._resident_zero_copy_pending_counts:
            counts = torch.stack([
                item[1] for item in self._resident_zero_copy_pending_counts
            ]).cpu().tolist()
            for (row, _), group_misses in zip(
                self._resident_zero_copy_pending_counts, counts
            ):
                misses = sum(group_misses)
                row["miss_rows"] = misses
                row["hit_rows"] = row["history_rows"] - misses
                row["hit_ratio"] = (
                    row["hit_rows"] / row["history_rows"]
                    if row["history_rows"] else 0.0
                )
                row["mapped_cpu_kv_read_bytes"] = (
                    2 * misses * self.geometry.head_dim * self.dtype.itemsize
                )
            self._resident_zero_copy_pending_counts.clear()
        return list(self._resident_reuse_trace)

    def enable_resident_attention_trace(
        self, enabled: bool = True, *, decode_steps: tuple[int, ...] = (1, 2, 32)
    ) -> None:
        """Capture exact SHA-256 checkpoints of assembled attention K/V.

        This copies selected attention inputs to CPU and is diagnostic-only.
        It must never be used for a TPOT claim.
        """
        self._capture_resident_attention_trace = bool(enabled)
        self._resident_attention_trace_steps = {int(step) for step in decode_steps}
        self._resident_attention_trace = []

    def resident_attention_trace(self) -> list[dict[str, Any]]:
        return list(self._resident_attention_trace)

    def _ensure_resident_layer_capacity(self, layer: int, rows: int) -> None:
        capacity = self._resident_capacities.get(layer, 0)
        if capacity >= rows:
            return
        # D1 establishes the working set.  Small headroom avoids allocator work
        # inside D2--D32 when Top-p union sizes drift by a few rows.
        new_capacity = (
            rows if self.vram_tight_resident_capacity
            else max(rows, math.ceil(rows * 1.05) + 64)
        )
        self._resident_keys[layer] = torch.empty(
            (new_capacity, self.geometry.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._resident_values[layer] = torch.empty_like(self._resident_keys[layer])
        self._resident_capacities[layer] = new_capacity

    def _ensure_audit_attention_capacity(self, rows: int) -> None:
        """Grow the layer-shared Attention workspace only when the union needs it."""
        if not self.vram_ondemand_attention_capacity:
            return
        assert self._gpu_layer_flat_attention_keys is not None
        assert self._gpu_layer_flat_attention_values is not None
        if self._gpu_layer_flat_attention_keys.shape[0] >= rows:
            return
        shape = (rows, 1, self.geometry.head_dim)
        self._gpu_layer_flat_attention_keys = torch.empty(
            shape, dtype=self.dtype, device=self.device
        )
        self._gpu_layer_flat_attention_values = torch.empty(
            shape, dtype=self.dtype, device=self.device
        )
        if self.vram_alias_history_staging:
            self._gpu_layer_flat_history_keys = self._gpu_layer_flat_attention_keys[:, 0]
            self._gpu_layer_flat_history_values = self._gpu_layer_flat_attention_values[:, 0]

    def enable_budget_trace(self, enabled: bool = True) -> None:
        """Capture lightweight B1 counts for diagnostic-only runs."""
        self._capture_budget_trace = bool(enabled)
        self._budget_trace = []

    def budget_trace(self) -> list[dict[str, Any]]:
        return list(self._budget_trace)

    def selected_positions_sha256(self) -> str:
        """Hash the latest exact per-entry/per-Query-head selected positions."""
        digest = hashlib.sha256()
        for entry_index, positions in enumerate(self._cached_twilight_positions):
            if positions is None:
                continue
            digest.update(entry_index.to_bytes(4, byteorder="little", signed=False))
            digest.update(len(positions).to_bytes(4, byteorder="little", signed=False))
            for selected in positions:
                selected_cpu = selected.detach().to(
                    device="cpu", dtype=torch.int64
                ).contiguous()
                digest.update(
                    selected_cpu.numel().to_bytes(
                        8, byteorder="little", signed=False
                    )
                )
                digest.update(selected_cpu.numpy().tobytes())
        return digest.hexdigest()

    def group_positions_sha256(self) -> Optional[str]:
        """Hash the latest exact sorted GQA unions."""
        if self._resident_zero_copy_placeholder_active:
            return None
        digest = hashlib.sha256()
        for entry_index, positions in enumerate(
            self._cached_twilight_group_positions
        ):
            if positions is None:
                continue
            selected_cpu = positions.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
            digest.update(entry_index.to_bytes(4, byteorder="little", signed=False))
            digest.update(
                selected_cpu.numel().to_bytes(8, byteorder="little", signed=False)
            )
            digest.update(selected_cpu.numpy().tobytes())
        return digest.hexdigest()

    def _build_metadata(self, entry: int, key_states: torch.Tensor) -> None:
        # Any rebuild invalidates the corresponding layer cache; never reuse
        # a previous prompt's packed metadata.
        for entries in list(self._layer_quant_metadata):
            if entry in entries:
                del self._layer_quant_metadata[entries]
        super()._build_metadata(entry, key_states)
        keys = key_states[0, 0]
        packed, scale, minimum = affine_int4_pack(keys)
        self._quant_key_packed[entry] = packed
        self._quant_key_scale[entry] = scale
        self._quant_key_minimum[entry] = minimum

    def _record_phase_start(self) -> Optional[torch.cuda.Event]:
        if not self.metrics_enabled:
            return None
        start = self._event(timing=True)
        start.record(torch.cuda.current_stream(self.device))
        return start

    def _record_phase_end(self, name: str, start: Optional[torch.cuda.Event]) -> None:
        if start is None:
            return
        end = self._event(timing=True)
        end.record(torch.cuda.current_stream(self.device))
        self._timing_events.append((name, start, end))

    def _record_detailed_phase_start(self) -> Optional[torch.cuda.Event]:
        if not self.detailed_selection_profile:
            return None
        return self._record_phase_start()

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self._post_selection_timeline_events = []
        self._post_selection_timeline = []

    def resolve_metrics(self) -> Dict[str, Any]:
        metrics = super().resolve_metrics()
        resolved: list[dict[str, Any]] = []
        for item in self._post_selection_timeline_events:
            origin = item["origin"]
            chunks = []
            for chunk in item["chunks"]:
                resolved_chunk = {
                    key: value
                    for key, value in chunk.items()
                    if not key.endswith("_event") and key != "done"
                }
                for prefix in ("union", "index_d2h", "selected_kv_h2d"):
                    start = chunk.get(f"{prefix}_start_event")
                    end = chunk.get(f"{prefix}_end_event")
                    if start is not None and end is not None:
                        resolved_chunk[f"{prefix}_start_ms"] = origin.elapsed_time(
                            start
                        )
                        resolved_chunk[f"{prefix}_end_ms"] = origin.elapsed_time(end)
                chunks.append(resolved_chunk)
            resolved.append(
                {
                    "layer": item["layer"],
                    "decode_step": item["decode_step"],
                    "chunks": chunks,
                }
            )
        self._post_selection_timeline = resolved
        if resolved:
            metrics["post_selection_timeline"] = resolved
        self._post_selection_timeline_events = []
        return metrics

    def prepare_layer_selection(
        self,
        *,
        entries: list[int],
        queries: list[torch.Tensor],
    ) -> None:
        """Batch both Twilight selection stages across one decoder layer."""
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
        fixed_pages = min(
            candidate_pages,
            max(
                1,
                math.ceil(
                    self.matched_budget_fraction
                    * eligible_tokens
                    / self.block_size
                ),
            ),
        )
        b0_pages = min(
            candidate_pages,
            max(1, math.ceil(self.candidate_token_budget / self.block_size)),
        )

        page_metadata = [
            (self._page_min[entry], self._page_max[entry]) for entry in entries
        ]
        quant_metadata = [
            (
                self._quant_key_packed[entry],
                self._quant_key_scale[entry],
                self._quant_key_minimum[entry],
            )
            for entry in entries
        ]
        if any(
            page_min is None or page_max is None
            for page_min, page_max in page_metadata
        ) or any(
            packed is None or scale is None or minimum is None
            for packed, scale, minimum in quant_metadata
        ):
            raise RuntimeError("Twilight prompt metadata is missing")
        candidate_pages = min(
            candidate_pages,
            *(
                int(page_min.shape[0])
                for page_min, _ in page_metadata
                if page_min is not None
            ),
        )
        fixed_pages = min(fixed_pages, candidate_pages)
        b0_pages = min(b0_pages, candidate_pages)

        if candidate_pages == 0 or fixed_pages == 0 or b0_pages == 0:
            empty = tuple(
                torch.empty(0, dtype=torch.long)
                for _ in range(self.num_query_heads_per_kv)
            )
            protected = torch.cat(
                (
                    torch.arange(sink_end, dtype=torch.long),
                    torch.arange(recent_start, old_length, dtype=torch.long),
                )
            )
            for entry in entries:
                self._cached_twilight_positions[entry] = empty
                if self.gpu_compact_gqa_union:
                    self._cached_twilight_group_positions[entry] = protected.clone()
                    self._cached_twilight_per_q_lengths[entry] = tuple(
                        int(protected.numel())
                        for _ in range(self.num_query_heads_per_kv)
                    )
                self._layer_prepared_entries.add(entry)
            return

        selector_wall_started = time.perf_counter() if self.metrics_enabled else None
        normal_timeline = bool(
            getattr(self, "_normal_overlap_timeline_enabled", False)
        )
        normal_host_origin = getattr(
            self, "_normal_overlap_host_origin", None
        )
        normal_selection_host_start = (
            time.perf_counter() if normal_timeline else None
        )
        normal_selection_start = (
            self._event(timing=True) if normal_timeline else None
        )
        if normal_selection_start is not None:
            normal_selection_start.record(torch.cuda.current_stream(self.device))
        normal_query_end = None
        normal_quest_end = None
        normal_selection_end = None
        normal_union_start = None
        normal_union_end = None
        normal_index_start = None
        normal_index_end = None
        normal_decode_host_start = None
        normal_decode_host_end = None
        groups = len(entries)
        query_heads = self.num_query_heads_per_kv
        workspace_probe = bool(getattr(self, "_selection_workspace_probe", False))
        if workspace_probe:
            selection_allocated_start = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        selection_cuda_phase = self._record_detailed_phase_start()
        subphase = self._record_detailed_phase_start()
        q = torch.stack([query[0, :, 0] for query in queries], dim=0).float()
        self._record_phase_end("twilight_query_prepare", subphase)
        if normal_timeline:
            normal_query_end = self._event(timing=True)
            normal_query_end.record(torch.cuda.current_stream(self.device))

        phase = self._record_phase_start()
        subphase = self._record_detailed_phase_start()
        page_min = torch.stack(
            [page_min[:candidate_pages] for page_min, _ in page_metadata if page_min is not None],
            dim=0,
        )
        page_max = torch.stack(
            [page_max[:candidate_pages] for _, page_max in page_metadata if page_max is not None],
            dim=0,
        )
        if self.fused_quest_score:
            from .twilight_fused_quest import fused_quest_score
            page_scores = fused_quest_score(q, page_min, page_max)
        else:
            extrema = torch.where(q.unsqueeze(2) > 0, page_max.unsqueeze(1), page_min.unsqueeze(1))
            page_scores = (extrema.float() * q.unsqueeze(2)).sum(dim=-1)
        self._record_phase_end("twilight_quest_metadata_score", subphase)

        subphase = self._record_detailed_phase_start()
        chosen_pages = torch.topk(
            page_scores, k=b0_pages, dim=-1, largest=True, sorted=False
        ).indices
        self._record_phase_end("twilight_quest_page_topk", subphase)

        subphase = self._record_detailed_phase_start()
        offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.long
        )
        candidate_positions = (
            self.sink_tokens
            + chosen_pages.unsqueeze(-1) * self.block_size
            + offsets
        ).flatten(-2)
        self._record_phase_end("twilight_b0_token_expand", subphase)
        self._record_phase_end("twilight_quest_b0", phase)
        if normal_timeline:
            normal_quest_end = self._event(timing=True)
            normal_quest_end.record(torch.cuda.current_stream(self.device))

        phase = self._record_phase_start()
        subphase = self._record_detailed_phase_start()
        metadata_key = tuple(entries)
        cached_quant = self._layer_quant_metadata.get(metadata_key) if self.reuse_quant_metadata else None
        if cached_quant is None:
            packed = torch.stack(
                [packed for packed, _, _ in quant_metadata if packed is not None], dim=0
            )
            scale = torch.stack(
                [scale for _, scale, _ in quant_metadata if scale is not None], dim=0
            )
            minimum = torch.stack(
                [minimum for _, _, minimum in quant_metadata if minimum is not None], dim=0
            )
            if self.reuse_quant_metadata:
                self._layer_quant_metadata[metadata_key] = (packed, scale, minimum)
                for i, entry in enumerate(entries):
                    self._quant_key_packed[entry] = packed[i]
                    self._quant_key_scale[entry] = scale[i]
                    self._quant_key_minimum[entry] = minimum[i]
        else:
            packed, scale, minimum = cached_quant
        self._record_phase_end("twilight_int4_metadata_stack", subphase)

        if self.qk_backend == "triton_prepare":
            from .twilight_fused_qk import fused_affine_int4_qk
            subphase = self._record_detailed_phase_start()
            estimated_keys = fused_affine_int4_qk(
                q, packed, scale, minimum, candidate_positions, materialize=True)
            self._record_phase_end("twilight_fused_int4_prepare", subphase)
            subphase = self._record_detailed_phase_start()
            logits = torch.matmul(estimated_keys, q.unsqueeze(-1)).squeeze(-1)
            self._record_phase_end("twilight_fp32_qk_matmul", subphase)
        elif self.qk_backend == "triton":
            from .twilight_fused_qk import fused_affine_int4_qk
            subphase = self._record_detailed_phase_start()
            logits = fused_affine_int4_qk(q, packed, scale, minimum, candidate_positions)
            self._record_phase_end("twilight_fused_int4_qk", subphase)
        else:
            subphase = self._record_detailed_phase_start()
            packed_index = candidate_positions.unsqueeze(-1).expand(
                groups,
                query_heads,
                candidate_positions.shape[-1],
                packed.shape[-1],
            )
            selected_packed = torch.gather(
                packed.unsqueeze(1).expand(-1, query_heads, -1, -1),
                2,
                packed_index,
            )
            selected_scale = torch.gather(
                scale.unsqueeze(1).expand(-1, query_heads, -1),
                2,
                candidate_positions,
            )
            selected_minimum = torch.gather(
                minimum.unsqueeze(1).expand(-1, query_heads, -1),
                2,
                candidate_positions,
            )
            self._record_phase_end("twilight_int4_candidate_gather", subphase)

            if self.detailed_selection_profile and self.metrics_enabled:
                subphase = self._record_detailed_phase_start()
                low = selected_packed & 0x0F
                high = (selected_packed >> 4) & 0x0F
                codes = torch.stack((low, high), dim=-1).flatten(-2).float()
                self._record_phase_end("twilight_int4_unpack_to_fp32_codes", subphase)

                subphase = self._record_detailed_phase_start()
                estimated_keys = (
                    codes * selected_scale.float().unsqueeze(-1)
                    + selected_minimum.float().unsqueeze(-1)
                )
                self._record_phase_end(
                    "twilight_int4_affine_dequant_and_fp32_k_materialize", subphase
                )
            else:
                estimated_keys = affine_int4_dequantize(
                    selected_packed, selected_scale, selected_minimum
                )

            subphase = self._record_detailed_phase_start()
            logits = torch.matmul(estimated_keys, q.unsqueeze(-1)).squeeze(-1)
            self._record_phase_end("twilight_fp32_qk_matmul", subphase)

        subphase = self._record_detailed_phase_start()
        logits /= math.sqrt(self.geometry.head_dim)
        self._record_phase_end("twilight_qk_scale", subphase)
        self._record_phase_end("twilight_int4_qk", phase)

        phase = self._record_phase_start()
        if self.exact_top_p_graph:
            if self._exact_top_p_graph is None:
                from .twilight_exact_top_p_graph import ExactTopPGraph
                self._exact_top_p_graph = ExactTopPGraph(self.top_p)
            subphase = self._record_detailed_phase_start()
            order, desired = self._exact_top_p_graph(logits)
            self._record_phase_end("twilight_top_p_graph_replay", subphase)
        else:
            subphase = self._record_detailed_phase_start()
            order = torch.argsort(logits, dim=-1, descending=True)
            self._record_phase_end("twilight_argsort", subphase)

            subphase = self._record_detailed_phase_start()
            ordered_logits = torch.gather(logits, -1, order)
            self._record_phase_end("twilight_sorted_logits_gather", subphase)

            subphase = self._record_detailed_phase_start()
            probabilities = torch.softmax(ordered_logits, dim=-1)
            self._record_phase_end("twilight_fp32_softmax", subphase)
            # Keep the 24 original 1-D scans: a 2-D CUDA scan changes rounding
            # at some Top-p boundaries.
            subphase = self._record_detailed_phase_start()
            if self.exact_scan_graph:
                if self._exact_scan_graph is None:
                    from .twilight_exact_scan_graph import ExactPerHeadScanGraph
                    self._exact_scan_graph = ExactPerHeadScanGraph()
                cumulative = self._exact_scan_graph(probabilities)
            else:
                cumulative = torch.stack(
                    [row.cumsum(dim=0) for row in probabilities.flatten(0, 1)], dim=0
                ).view_as(probabilities)
            self._record_phase_end("twilight_cumsum", subphase)

            subphase = self._record_detailed_phase_start()
            thresholds = torch.full(
                (*cumulative.shape[:-1], 1),
                self.top_p,
                dtype=cumulative.dtype,
                device=self.device,
            )
            desired = torch.searchsorted(
                cumulative.contiguous(), thresholds, right=False
            ).squeeze(-1) + 1
            desired.clamp_max_(order.shape[-1])
            self._record_phase_end("twilight_top_p_threshold", subphase)
        self._record_phase_end("twilight_top_p", phase)
        if workspace_probe:
            persistent_ptrs = set()
            for collection in (
                self._page_min, self._page_max, self._quant_key_packed,
                self._quant_key_scale, self._quant_key_minimum,
            ):
                for tensor in collection:
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        persistent_ptrs.add(tensor.untyped_storage().data_ptr())
            local_storages = {}
            for name, tensor in tuple(locals().items()):
                if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                    continue
                storage = tensor.untyped_storage()
                ptr = storage.data_ptr()
                item = local_storages.setdefault(ptr, {
                    "storage_bytes": storage.nbytes(),
                    "aliases": [],
                    "is_persistent_cache_storage": ptr in persistent_ptrs,
                })
                item["aliases"].append(name)
            self._selection_workspace_probe_records.append({
                "layer": entries[0] // groups,
                "old_length": old_length,
                "selection_start_allocated_bytes": selection_allocated_start,
                "post_top_p_allocated_bytes": torch.cuda.memory_allocated(self.device),
                "selection_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "live_local_temporary_unique_storage_bytes": sum(
                    item["storage_bytes"] for item in local_storages.values()
                    if not item["is_persistent_cache_storage"]
                ),
                "local_storages": list(local_storages.values()),
            })

        subphase = self._record_detailed_phase_start()
        target_total = fixed_pages * self.block_size * query_heads
        allocated = (
            desired
            if self.budget_mode == "dynamic"
            else matched_budget_counts_tensor(
                desired, target_total, int(order.shape[-1])
            )
        )
        self._record_phase_end("twilight_membership_decision", subphase)
        self._record_phase_end("twilight_selection_core", selection_cuda_phase)
        if normal_timeline:
            normal_selection_end = self._event(timing=True)
            normal_selection_end.record(torch.cuda.current_stream(self.device))
        selection_done: Optional[torch.cuda.Event] = None
        if self.post_selection_pipeline or self.metrics_enabled:
            selection_done = self._event()
            selection_done.record(torch.cuda.current_stream(self.device))
        timeline_origin = selection_done
        timeline_host_origin = time.perf_counter()
        if self.metrics_enabled:
            # Diagnostic-only wall boundary. Formal D2--D32 does not synchronize
            # here; the post-selection stream waits on the recorded event instead.
            assert selection_done is not None
            selection_done.synchronize()
            assert selector_wall_started is not None
            self.metrics["twilight_selection_core_wall_seconds"] += (
                time.perf_counter() - selector_wall_started
            )
            timeline_origin = self._event(timing=True)
            timeline_origin.record(torch.cuda.current_stream(self.device))
            timeline_host_origin = time.perf_counter()
            self._post_selection_host_starts[entries[0] // groups] = (
                timeline_host_origin
            )
        max_allocated = (
            int(order.shape[-1])
            if self.budget_mode == "dynamic"
            else target_total - (query_heads - 1)
        )
        allocated_cpu: Optional[torch.Tensor] = None
        desired_cpu: Optional[torch.Tensor] = None
        rows: Optional[torch.Tensor] = None
        group_positions: Optional[list[torch.Tensor]] = None
        transferred_bytes = 0
        if self.gpu_compact_gqa_union and self.post_selection_pipeline:
            if self._post_selection_stream is None:
                raise AssertionError("post-selection stream is missing")
            if timeline_origin is None:
                raise AssertionError("Selection barrier event is missing")
            required_shape = (groups, self.geometry.max_cache_len + 1)
            if (
                self._gpu_group_membership is None
                or tuple(self._gpu_group_membership.shape) != required_shape
            ):
                self._gpu_group_membership = torch.empty(
                    required_shape, dtype=torch.bool, device=self.device
                )
            host_shape = (groups, self.geometry.max_cache_len)
            if (
                self._post_selection_host_membership is None
                or tuple(self._post_selection_host_membership.shape) != host_shape
            ):
                self._post_selection_host_membership = torch.empty(
                    host_shape, dtype=torch.bool, device="cpu", pin_memory=True
                )
            layer_index = entries[0] // groups
            if layer_index in self._post_selection_pending:
                raise AssertionError("unconsumed post-selection layer state")
            chunk_records: list[dict[str, Any]] = []
            chunk_size = min(self.post_selection_groups_per_chunk, groups)
            with torch.cuda.stream(self._post_selection_stream):
                self._post_selection_stream.wait_event(timeline_origin)
                # Build all GQA unions in one batched GPU operation.  Splitting
                # gather/scatter by group made the small CUDA kernels dominate
                # the attempted overlap (especially at one group per chunk).
                # Only the bitmap handoff remains chunked, so the CPU can start
                # decoding chunk 0 while later bitmap slices are copied D2H.
                union_start = self._event(timing=True) if self.metrics_enabled else None
                union_end = self._event(timing=True) if self.metrics_enabled else None
                if union_start is not None:
                    union_start.record(self._post_selection_stream)
                ranked_prefix = torch.gather(
                    candidate_positions,
                    -1,
                    order[..., :max_allocated],
                )
                active = torch.arange(
                    max_allocated, device=self.device
                ).view(1, 1, -1) < allocated.unsqueeze(-1)
                sentinel = torch.full_like(ranked_prefix, old_length)
                scatter_positions = torch.where(active, ranked_prefix, sentinel)
                membership = self._gpu_group_membership[:, : old_length + 1]
                membership.zero_()
                membership.scatter_(1, scatter_positions.flatten(1), True)
                membership[:, :sink_end] = True
                membership[:, recent_start:old_length] = True
                if union_end is not None:
                    union_end.record(self._post_selection_stream)
                for group_start in range(0, groups, chunk_size):
                    group_stop = min(groups, group_start + chunk_size)
                    index_start = self._event(timing=True) if self.metrics_enabled else None
                    index_end = self._event(timing=True) if self.metrics_enabled else None
                    if index_start is not None:
                        index_start.record(self._post_selection_stream)
                    self._post_selection_host_membership[
                        group_start:group_stop, :old_length
                    ].copy_(
                        membership[group_start:group_stop, :old_length],
                        non_blocking=True,
                    )
                    if index_end is not None:
                        index_end.record(self._post_selection_stream)
                    done = self._event()
                    done.record(self._post_selection_stream)
                    chunk_records.append(
                        {
                            "group_start": group_start,
                            "group_stop": group_stop,
                            "done": done,
                            "union_start_event": (
                                union_start if group_start == 0 else None
                            ),
                            "union_end_event": (
                                union_end if group_start == 0 else None
                            ),
                            "index_d2h_start_event": index_start,
                            "index_d2h_end_event": index_end,
                        }
                    )
            for tensor in (candidate_positions, order, allocated):
                tensor.record_stream(self._post_selection_stream)
            self._post_selection_pending[layer_index] = {
                "old_length": old_length,
                "selection_done": timeline_origin,
                "host_origin": timeline_host_origin,
                "origin": timeline_origin,
                "chunks": chunk_records,
            }
            transferred_bytes += groups * old_length
            sync_started = time.perf_counter() if self.metrics_enabled else None

            need_counts = (
                self.metrics_enabled
                or self._capture_budget_trace
                or self._capture_selection_trace
                or self.gpu_union_validate_cpu
            )
            if need_counts:
                count_transfer = torch.stack((allocated, desired), dim=-1).cpu()
                allocated_cpu = count_transfer[..., 0]
                desired_cpu = count_transfer[..., 1]
                transferred_bytes += (
                    count_transfer.numel() * count_transfer.element_size()
                )
            need_per_q_rows = (
                self._capture_budget_trace
                or self._capture_selection_trace
                or self.gpu_union_validate_cpu
            )
            if need_per_q_rows:
                diagnostic_ranked_prefix = torch.gather(
                    candidate_positions,
                    -1,
                    order[..., :max_allocated],
                )
                rows = diagnostic_ranked_prefix.reshape(
                    groups * query_heads, -1
                ).cpu()
                transferred_bytes += rows.numel() * rows.element_size()
            sync_finished = time.perf_counter() if self.metrics_enabled else None
        elif self.gpu_compact_gqa_union:
            if normal_timeline:
                normal_union_start = self._event(timing=True)
                normal_union_start.record(torch.cuda.current_stream(self.device))
            union_subphase = self._record_detailed_phase_start()
            # The consumer needs one sorted union per KV head, not four sorted
            # per-Query-head rows.  Scatter the exact active prefixes into a
            # reusable GPU membership map.  The compact bool map preserves set
            # semantics; CPU flatnonzero reconstructs the exact sorted int64
            # positions required by the unchanged gather path.
            ranked_prefix = torch.gather(
                candidate_positions,
                -1,
                order[..., :max_allocated],
            )
            active = torch.arange(
                max_allocated, device=self.device
            ).view(1, 1, -1) < allocated.unsqueeze(-1)
            sentinel = torch.full_like(ranked_prefix, old_length)
            scatter_positions = torch.where(active, ranked_prefix, sentinel)
            required_shape = (groups, self.geometry.max_cache_len + 1)
            if (
                self._gpu_group_membership is None
                or tuple(self._gpu_group_membership.shape) != required_shape
            ):
                self._gpu_group_membership = torch.empty(
                    required_shape, dtype=torch.bool, device=self.device
                )
            membership = self._gpu_group_membership[:, : old_length + 1]
            membership.zero_()
            membership.scatter_(1, scatter_positions.flatten(1), True)
            membership[:, :sink_end] = True
            membership[:, recent_start:old_length] = True
            self._record_phase_end("twilight_gpu_gqa_union", union_subphase)
            if normal_timeline:
                normal_union_end = self._event(timing=True)
                normal_union_end.record(torch.cuda.current_stream(self.device))
            self._record_phase_end(
                "twilight_selection_before_d2h", selection_cuda_phase
            )
            sync_started = time.perf_counter() if self.metrics_enabled else None
            zero_copy_fast = bool(
                self.resident_zero_copy
                and self._prompt_length is not None
                and old_length > self._prompt_length
                and not self._capture_selection_trace
                and not self._capture_budget_trace
                and not self.gpu_union_validate_cpu
            )
            if zero_copy_fast:
                if normal_timeline:
                    normal_index_start = self._event(timing=True)
                    normal_index_start.record(torch.cuda.current_stream(self.device))
                    normal_decode_host_start = time.perf_counter()
                if self.resident_precompute_gpu_mapping:
                    from .resident_gpu_mapping import map_membership
                    layer_index = entries[0] // groups
                    if layer_index not in self._resident_bitmaps:
                        raise AssertionError("resident bitmap is missing")
                    mapped = map_membership(
                        membership[:, :old_length],
                        self._resident_bitmaps[layer_index],
                    )
                    self._resident_precomputed_maps[layer_index] = mapped
                    lengths_gpu = mapped[3]
                else:
                    lengths_gpu = membership[:, :old_length].sum(
                        dim=1, dtype=torch.int32
                    )
                lengths_cpu = lengths_gpu.cpu().tolist()
                if normal_timeline:
                    normal_decode_host_end = time.perf_counter()
                    normal_index_end = self._event(timing=True)
                    normal_index_end.record(torch.cuda.current_stream(self.device))
                group_positions = [
                    torch.empty(length, dtype=torch.long)
                    for length in lengths_cpu
                ]
                self._resident_zero_copy_placeholder_active = True
                sync_finished = time.perf_counter() if self.metrics_enabled else None
                transferred_bytes += 8 * 4
            else:
                if normal_timeline:
                    normal_index_start = self._event(timing=True)
                    normal_index_start.record(torch.cuda.current_stream(self.device))
                membership_cpu = membership[:, :old_length].cpu()
                if normal_timeline:
                    normal_index_end = self._event(timing=True)
                    normal_index_end.record(torch.cuda.current_stream(self.device))
                sync_finished = time.perf_counter() if self.metrics_enabled else None
                transferred_bytes += membership_cpu.numel() * membership_cpu.element_size()
                decode_started = time.perf_counter() if self.metrics_enabled else None
                if normal_timeline:
                    normal_decode_host_start = time.perf_counter()
                group_positions = [
                    torch.from_numpy(np.flatnonzero(row.numpy()))
                    for row in membership_cpu
                ]
                self._resident_zero_copy_placeholder_active = False
                if normal_timeline:
                    normal_decode_host_end = time.perf_counter()
            if self.metrics_enabled:
                if not zero_copy_fast:
                    assert decode_started is not None
                    self.metrics["twilight_compact_index_decode_wall_seconds"] += (
                        time.perf_counter() - decode_started
                    )

            need_counts = (
                self.metrics_enabled
                or self._capture_budget_trace
                or self._capture_selection_trace
                or self.gpu_union_validate_cpu
            )
            if need_counts:
                count_transfer = torch.stack((allocated, desired), dim=-1).cpu()
                allocated_cpu = count_transfer[..., 0]
                desired_cpu = count_transfer[..., 1]
                transferred_bytes += (
                    count_transfer.numel() * count_transfer.element_size()
                )

            need_per_q_rows = (
                self._capture_budget_trace
                or self._capture_selection_trace
                or self.gpu_union_validate_cpu
            )
            if need_per_q_rows:
                rows = ranked_prefix.reshape(groups * query_heads, -1).cpu()
                transferred_bytes += rows.numel() * rows.element_size()
        else:
            subphase = self._record_detailed_phase_start()
            if self.fused_final_indices:
                from .twilight_fused_indices import fused_gather_mask
                masked_positions = fused_gather_mask(
                    candidate_positions,
                    order,
                    allocated,
                    max_allocated,
                    old_length,
                )
                sorted_selected = masked_positions.sort(dim=-1).values
            else:
                ranked_positions = torch.gather(candidate_positions, -1, order)
                ranked_prefix = ranked_positions[..., :max_allocated]
                active = torch.arange(
                    max_allocated, device=self.device
                ).view(1, 1, -1) < allocated.unsqueeze(-1)
                sentinel = torch.full_like(ranked_prefix, old_length)
                sorted_selected = torch.where(
                    active, ranked_prefix, sentinel
                ).sort(dim=-1).values
            # Counts, Top-p desired counts, and all ragged selected positions
            # share one GPU-to-CPU synchronization for the entire layer.
            transfer = torch.cat(
                (allocated.unsqueeze(-1), desired.unsqueeze(-1), sorted_selected),
                dim=-1,
            ).reshape(groups * query_heads, -1)
            self._record_phase_end("twilight_final_b1_indices", subphase)
            self._record_phase_end(
                "twilight_selection_before_d2h", selection_cuda_phase
            )
            sync_started = time.perf_counter() if self.metrics_enabled else None
            transfer_cpu = transfer.cpu()
            sync_finished = time.perf_counter() if self.metrics_enabled else None
            transferred_bytes += transfer_cpu.numel() * transfer_cpu.element_size()
            allocated_cpu = transfer_cpu[:, 0].view(groups, query_heads)
            desired_cpu = transfer_cpu[:, 1].view(groups, query_heads)
            rows = transfer_cpu[:, 2:]

        decode_step = (
            old_length - self._prompt_length + 1
            if self._prompt_length is not None
            else None
        )
        protected_tokens = sink_end + (old_length - recent_start)
        for group_index, entry in enumerate(entries):
            start = group_index * query_heads
            selected: list[torch.Tensor] = []
            if rows is not None:
                assert allocated_cpu is not None
                for head_index in range(query_heads):
                    count = int(allocated_cpu[group_index, head_index])
                    selected_row = rows[start + head_index, :count]
                    if self.gpu_compact_gqa_union:
                        # Diagnostic GPU-union traces transfer rank order and
                        # normalize it on CPU.  Baseline rows were already
                        # sorted on GPU and must not be sorted a second time.
                        selected_row = selected_row.sort().values
                    selected.append(selected_row)
                    if self._capture_budget_trace:
                        assert desired_cpu is not None
                        source_pages = torch.div(
                            selected_row - self.sink_tokens,
                            self.block_size,
                            rounding_mode="floor",
                        )
                        self._budget_trace.append(
                            {
                                "decode_step": decode_step,
                                "entry": entry,
                                "layer": entries[0] // groups,
                                "kv_head": group_index,
                                "query_head_in_group": head_index,
                                "query_head": group_index * query_heads + head_index,
                                "budget_mode": self.budget_mode,
                                "b0_tokens": int(order.shape[-1]),
                                "raw_b1_tokens": int(desired_cpu[group_index, head_index]),
                                "selected_adaptive_tokens": count,
                                "protected_tokens": protected_tokens,
                                "selected_history_tokens": protected_tokens + count,
                                "eligible_history_tokens": eligible_tokens,
                                "full_history_tokens": old_length,
                                "unique_source_pages": int(
                                    torch.unique(source_pages).numel()
                                ),
                            }
                        )
            if selected:
                self._cached_twilight_positions[entry] = tuple(selected)
            elif self.gpu_compact_gqa_union:
                self._cached_twilight_positions[entry] = None
            if group_positions is not None:
                union = group_positions[group_index]
                self._cached_twilight_group_positions[entry] = union
                if allocated_cpu is not None:
                    self._cached_twilight_per_q_lengths[entry] = tuple(
                        protected_tokens + int(value)
                        for value in allocated_cpu[group_index]
                    )
                else:
                    self._cached_twilight_per_q_lengths[entry] = None
                if self.gpu_union_validate_cpu:
                    sink_cpu = torch.arange(sink_end, dtype=torch.long)
                    recent_cpu = torch.arange(
                        recent_start, old_length, dtype=torch.long
                    )
                    cpu_positions = [
                        torch.cat((sink_cpu, row, recent_cpu)) for row in selected
                    ]
                    expected = (
                        self._cpu_token_union(cpu_positions, old_length)
                        if self.cpu_bitmap_union
                        else torch.unique(torch.cat(cpu_positions), sorted=True)
                    )
                    if not torch.equal(union, expected):
                        raise AssertionError(
                            f"GPU compact GQA union mismatch for entry {entry}"
                        )
            self._layer_prepared_entries.add(entry)

        if self.metrics_enabled:
            assert selector_wall_started is not None
            assert sync_started is not None and sync_finished is not None
            self.metrics["twilight_index_sync_and_position_wall_seconds"] += (
                sync_finished - sync_started
            )
            self.metrics["twilight_index_d2h_bytes"] += transferred_bytes
            self.metrics["twilight_selector_wall_seconds"] += (
                sync_finished - selector_wall_started
            )
            assert allocated_cpu is not None and desired_cpu is not None
            self.metrics["twilight_raw_top_p_tokens_total"] += int(desired_cpu.sum())
            self.metrics["twilight_allocated_tokens_total"] += int(allocated_cpu.sum())
            self.metrics["twilight_allocated_tokens_min_sum"] += int(
                allocated_cpu.min(dim=-1).values.sum()
            )
            self.metrics["twilight_allocated_tokens_max_sum"] += int(
                allocated_cpu.max(dim=-1).values.sum()
            )
            self.metrics["twilight_b0_tokens_total"] += int(
                groups * query_heads * order.shape[-1]
            )
            self.metrics["twilight_groups"] += groups
            self.metrics["twilight_selector_batch_calls"] += 1
            self.metrics["twilight_index_sync_calls"] += 1
            if self.detailed_selection_profile:
                self.metrics["twilight_profile_old_length"] = old_length
                self.metrics["twilight_profile_eligible_tokens"] = eligible_tokens
                self.metrics["twilight_profile_candidate_pages"] = candidate_pages
                self.metrics["twilight_profile_b0_pages"] = b0_pages
                self.metrics["twilight_profile_b0_tokens_per_query_head"] = int(
                    order.shape[-1]
                )
                self.metrics["twilight_profile_groups"] = groups
                self.metrics["twilight_profile_query_heads_per_group"] = query_heads
                self.metrics["twilight_profile_head_dim"] = self.geometry.head_dim
            self.metrics["selector_total_wall_seconds"] += (
                time.perf_counter() - selector_wall_started
            )
        if normal_timeline:
            if normal_host_origin is None or normal_selection_host_start is None:
                raise AssertionError("normal-overlap host origin is missing")
            self._normal_overlap_selection_events.append(
                {
                    "decode_step": decode_step,
                    "layer": entries[0] // groups,
                    "selection_core_start_event": normal_selection_start,
                    "query_prepare_end_event": normal_query_end,
                    "quest_first_round_end_event": normal_quest_end,
                    "selection_core_end_event": normal_selection_end,
                    "gpu_union_start_event": normal_union_start,
                    "gpu_union_end_event": normal_union_end,
                    "index_d2h_start_event": normal_index_start,
                    "index_d2h_end_event": normal_index_end,
                    "selection_host_start_ms": (
                        normal_selection_host_start - normal_host_origin
                    )
                    * 1000.0,
                    "selection_host_end_ms": (
                        time.perf_counter() - normal_host_origin
                    )
                    * 1000.0,
                    "cpu_decode_start_ms": (
                        None
                        if normal_decode_host_start is None
                        else (normal_decode_host_start - normal_host_origin)
                        * 1000.0
                    ),
                    "cpu_decode_end_ms": (
                        None
                        if normal_decode_host_end is None
                        else (normal_decode_host_end - normal_host_origin)
                        * 1000.0
                    ),
                }
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
        prepared_for_entry = entry in self._layer_prepared_entries
        if prepared_for_entry:
            cached = self._cached_twilight_positions[entry]
            if cached is None:
                raise AssertionError("prepared Twilight selection is missing")
            sink_cpu = torch.arange(sink_end, dtype=torch.long)
            recent_cpu = torch.arange(recent_start, old_length, dtype=torch.long)
            positions = []
            decode_step = (
                old_length - self._prompt_length + 1
                if self._prompt_length is not None
                else None
            )
            for group_head, selected in enumerate(cached):
                positions.append(torch.cat((sink_cpu, selected, recent_cpu)))
                if self._capture_selection_trace:
                    self._selection_trace.append(
                        {
                            "decode_step": decode_step,
                            "entry": entry,
                            "query_head_in_group": group_head,
                            "selected_token_indices": tuple(
                                int(value) for value in selected.tolist()
                            ),
                        }
                    )
            self._layer_prepared_entries.discard(entry)
            return positions
        eligible_tokens = max(0, recent_start - sink_end)
        candidate_pages = eligible_tokens // self.block_size
        fixed_pages = min(
            candidate_pages,
            max(
                1,
                math.ceil(
                    self.matched_budget_fraction
                    * eligible_tokens
                    / self.block_size
                ),
            ),
        )
        b0_pages = min(
            candidate_pages,
            max(1, math.ceil(self.candidate_token_budget / self.block_size)),
        )

        page_min = self._page_min[entry]
        page_max = self._page_max[entry]
        packed = self._quant_key_packed[entry]
        scale = self._quant_key_scale[entry]
        minimum = self._quant_key_minimum[entry]
        if any(item is None for item in (page_min, page_max, packed, scale, minimum)):
            raise RuntimeError("Twilight prompt metadata is missing")
        assert page_min is not None and page_max is not None
        assert packed is not None and scale is not None and minimum is not None
        candidate_pages = min(candidate_pages, int(page_min.shape[0]))
        fixed_pages = min(fixed_pages, candidate_pages)
        b0_pages = min(b0_pages, candidate_pages)

        sink_cpu = torch.arange(sink_end, dtype=torch.long)
        recent_cpu = torch.arange(recent_start, old_length, dtype=torch.long)
        if candidate_pages == 0 or fixed_pages == 0 or b0_pages == 0:
            selected = torch.cat((sink_cpu, recent_cpu))
            return [selected.clone() for _ in range(self.num_query_heads_per_kv)]

        ranked_tokens: list[torch.Tensor] = []
        desired_counts: list[int] = []
        candidate_counts: list[int] = []
        offsets = torch.arange(self.block_size, device=self.device, dtype=torch.long)
        sqrt_dim = math.sqrt(self.geometry.head_dim)
        selector_wall_started = time.perf_counter() if self.metrics_enabled else None

        for group_head in range(self.num_query_heads_per_kv):
            q = query[0, group_head, 0].float()

            phase = self._record_phase_start()
            extrema = torch.where(
                q.unsqueeze(0) > 0,
                page_max[:candidate_pages],
                page_min[:candidate_pages],
            )
            page_scores = (extrema.float() * q.unsqueeze(0)).sum(dim=-1)
            chosen_pages = torch.topk(
                page_scores, k=b0_pages, largest=True, sorted=False
            ).indices
            candidate_positions = (
                self.sink_tokens
                + chosen_pages.unsqueeze(1) * self.block_size
                + offsets.unsqueeze(0)
            ).reshape(-1)
            self._record_phase_end("twilight_quest_b0", phase)

            phase = self._record_phase_start()
            selected_packed = packed.index_select(0, candidate_positions)
            estimated_keys = affine_int4_dequantize(
                selected_packed,
                scale.index_select(0, candidate_positions),
                minimum.index_select(0, candidate_positions),
            )
            logits = torch.mv(estimated_keys, q) / sqrt_dim
            self._record_phase_end("twilight_int4_qk", phase)

            phase = self._record_phase_start()
            order = torch.argsort(logits, descending=True)
            ordered_logits = logits.index_select(0, order)
            cumulative = torch.softmax(ordered_logits, dim=0).cumsum(dim=0)
            desired = int(
                torch.searchsorted(
                    cumulative,
                    torch.tensor(self.top_p, device=self.device),
                    right=False,
                ).item()
            ) + 1
            desired = min(desired, int(order.numel()))
            self._record_phase_end("twilight_top_p", phase)
            ranked_tokens.append(candidate_positions.index_select(0, order))
            desired_counts.append(desired)
            candidate_counts.append(int(order.numel()))

        target_total = fixed_pages * self.block_size * self.num_query_heads_per_kv
        allocated_counts = (
            desired_counts
            if self.budget_mode == "dynamic"
            else matched_budget_counts(
                desired_counts, target_total, candidate_counts
            )
        )
        positions: list[torch.Tensor] = []
        sync_started = time.perf_counter() if self.metrics_enabled else None
        for ranked, allocated in zip(ranked_tokens, allocated_counts):
            chosen_cpu = ranked[:allocated].sort().values.cpu()
            positions.append(torch.cat((sink_cpu, chosen_cpu, recent_cpu)))
            if self._capture_budget_trace:
                head_index = len(positions) - 1
                source_pages = torch.div(
                    chosen_cpu - self.sink_tokens,
                    self.block_size,
                    rounding_mode="floor",
                )
                protected_tokens = sink_end + (old_length - recent_start)
                self._budget_trace.append(
                    {
                        "decode_step": (
                            old_length - self._prompt_length + 1
                            if self._prompt_length is not None
                            else None
                        ),
                        "entry": entry,
                        "layer": None,
                        "kv_head": None,
                        "query_head_in_group": head_index,
                        "query_head": None,
                        "budget_mode": self.budget_mode,
                        "b0_tokens": candidate_counts[head_index],
                        "raw_b1_tokens": desired_counts[head_index],
                        "selected_adaptive_tokens": allocated,
                        "protected_tokens": protected_tokens,
                        "selected_history_tokens": protected_tokens + allocated,
                        "eligible_history_tokens": eligible_tokens,
                        "full_history_tokens": old_length,
                        "unique_source_pages": int(
                            torch.unique(source_pages).numel()
                        ),
                    }
                )
            if self._capture_selection_trace:
                self._selection_trace.append(
                    {
                        "decode_step": (
                            old_length - self._prompt_length + 1
                            if self._prompt_length is not None
                            else None
                        ),
                        "entry": entry,
                        "query_head_in_group": len(positions) - 1,
                        "selected_token_indices": tuple(
                            int(value) for value in chosen_cpu.tolist()
                        ),
                    }
                )
        if self.metrics_enabled:
            sync_finished = time.perf_counter()
            self.metrics["twilight_index_sync_and_position_wall_seconds"] += (
                sync_finished - sync_started
            )
            self.metrics["twilight_selector_wall_seconds"] += (
                sync_finished - selector_wall_started
            )
            self.metrics["twilight_raw_top_p_tokens_total"] += sum(desired_counts)
            self.metrics["twilight_allocated_tokens_total"] += sum(allocated_counts)
            self.metrics["twilight_allocated_tokens_min_sum"] += min(allocated_counts)
            self.metrics["twilight_allocated_tokens_max_sum"] += max(allocated_counts)
            self.metrics["twilight_b0_tokens_total"] += sum(candidate_counts)
            self.metrics["twilight_groups"] += 1
        if self.budget_mode == "matched" and sum(allocated_counts) != target_total:
            raise AssertionError("Twilight matched budget drifted from Quest")
        return positions

    def _consume_post_selection_pipeline(
        self,
        *,
        key_states: list[torch.Tensor],
        value_states: list[torch.Tensor],
        queries: list[torch.Tensor],
        entries: list[int],
        old_length: int,
        new_lengths: list[int],
        decode_step: int,
        update_started: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[int, ...]]:
        """Consume chunked post-Selection work after the Top-p barrier.

        Selection has already recorded ``selection_done``.  The GPU union/index
        stream waits on that event, CPU decode/gather waits on each chunk's D2H
        completion, and selected-KV H2D uses a separate stream.  Consequently no
        post-Selection operation can overlap backwards into Selection, while
        later chunks can overlap GPU index work, CPU gather, and PCIe H2D.
        """
        if not self.post_selection_pipeline:
            raise RuntimeError("post-selection pipeline is disabled")
        if self._selected_kv_h2d_stream is None:
            raise AssertionError("selected-KV H2D stream is missing")
        if self._post_selection_host_membership is None:
            raise AssertionError("post-selection host membership is missing")
        group_count = len(entries)
        layer_index = entries[0] // group_count
        pending = self._post_selection_pending.pop(layer_index, None)
        if pending is None or pending["old_length"] != old_length:
            raise AssertionError("prepared post-selection chunks are missing")

        slot = self._layer_flat_cursor
        self._layer_flat_cursor = 1 - self._layer_flat_cursor
        wait_started = time.perf_counter() if self.metrics_enabled else None
        ready = self._layer_flat_ready[slot]
        if ready is not None:
            ready.synchronize()
        if self.metrics_enabled:
            assert wait_started is not None
            self.metrics["pack_ready_wait_wall_seconds"] += (
                time.perf_counter() - wait_started
            )

        host_flat_key = self._host_layer_flat_keys[slot]
        host_flat_value = self._host_layer_flat_values[slot]
        host_rows = self._host_layer_flat_row_indices[slot]
        assert self._gpu_layer_flat_attention_keys is not None
        assert self._gpu_layer_flat_attention_values is not None

        group_positions: list[torch.Tensor] = []
        new_token_offsets: list[int] = []
        attention_offset = 0
        pipeline_enqueue_seconds = 0.0
        cpu_gather_seconds = 0.0
        cpu_decode_seconds = 0.0
        index_wait_seconds = 0.0
        host_origin = float(pending["host_origin"])
        timeline_chunks: list[dict[str, Any]] = []

        # The transfer stream may otherwise race a previous default-stream user
        # of the reusable attention buffer.  This dependency is also the hard
        # proof that selected-KV H2D starts strictly after Selection completion.
        self._selected_kv_h2d_stream.wait_event(pending["selection_done"])
        for chunk in pending["chunks"]:
            chunk_wait_started = time.perf_counter()
            chunk["done"].synchronize()
            chunk_wait_finished = time.perf_counter()
            index_wait_seconds += chunk_wait_finished - chunk_wait_started

            decode_started = time.perf_counter()
            chunk_positions = [
                torch.from_numpy(np.flatnonzero(row.numpy()))
                for row in self._post_selection_host_membership[
                    chunk["group_start"] : chunk["group_stop"], :old_length
                ]
            ]
            decode_finished = time.perf_counter()
            cpu_decode_seconds += decode_finished - decode_started

            chunk_attention_start = attention_offset
            for group_index, selected in zip(
                range(chunk["group_start"], chunk["group_stop"]),
                chunk_positions,
            ):
                entry = entries[group_index]
                history_stop = attention_offset + int(selected.numel())
                torch.add(
                    selected,
                    entry * self.geometry.max_cache_len,
                    out=host_rows[attention_offset:history_stop],
                )
                host_rows[history_stop] = 0
                new_token_offsets.append(history_stop)
                attention_offset = history_stop + 1
                group_positions.append(selected)
                self._cached_twilight_group_positions[entry] = selected

                if self.gpu_union_validate_cpu:
                    per_q_selected = self._cached_twilight_positions[entry]
                    if per_q_selected is None:
                        raise AssertionError(
                            "GPU union validation requires retained per-Q rows"
                        )
                    sink_end = min(self.sink_tokens, old_length)
                    recent_start = max(
                        sink_end, old_length - max(0, self.recent_tokens - 1)
                    )
                    sink_cpu = torch.arange(sink_end, dtype=torch.long)
                    recent_cpu = torch.arange(
                        recent_start, old_length, dtype=torch.long
                    )
                    cpu_positions = [
                        torch.cat((sink_cpu, row, recent_cpu))
                        for row in per_q_selected
                    ]
                    expected = (
                        self._cpu_token_union(cpu_positions, old_length)
                        if self.cpu_bitmap_union
                        else torch.unique(torch.cat(cpu_positions), sorted=True)
                    )
                    if not torch.equal(selected, expected):
                        raise AssertionError(
                            f"post-selection GPU union mismatch for entry {entry}"
                        )

            chunk_attention_stop = attention_offset
            gather_started = time.perf_counter()
            torch.index_select(
                self._host_key_rows,
                0,
                host_rows[chunk_attention_start:chunk_attention_stop],
                out=host_flat_key[chunk_attention_start:chunk_attention_stop],
            )
            torch.index_select(
                self._host_value_rows,
                0,
                host_rows[chunk_attention_start:chunk_attention_stop],
                out=host_flat_value[chunk_attention_start:chunk_attention_stop],
            )
            gather_finished = time.perf_counter()
            cpu_gather_seconds += gather_finished - gather_started

            enqueue_started = time.perf_counter()
            h2d_start = self._event(timing=True) if self.metrics_enabled else None
            h2d_end = self._event(timing=True) if self.metrics_enabled else None
            with torch.cuda.stream(self._selected_kv_h2d_stream):
                if h2d_start is not None:
                    h2d_start.record(self._selected_kv_h2d_stream)
                self._gpu_layer_flat_attention_keys[
                    chunk_attention_start:chunk_attention_stop, 0
                ].copy_(
                    host_flat_key[chunk_attention_start:chunk_attention_stop],
                    non_blocking=True,
                )
                self._gpu_layer_flat_attention_values[
                    chunk_attention_start:chunk_attention_stop, 0
                ].copy_(
                    host_flat_value[chunk_attention_start:chunk_attention_stop],
                    non_blocking=True,
                )
                if h2d_end is not None:
                    h2d_end.record(self._selected_kv_h2d_stream)
            enqueue_finished = time.perf_counter()
            pipeline_enqueue_seconds += enqueue_finished - enqueue_started
            if self.metrics_enabled:
                assert h2d_start is not None and h2d_end is not None
                self._timing_events.append(("h2d", h2d_start, h2d_end))
                chunk_rows = chunk_attention_stop - chunk_attention_start
                self.metrics["h2d_bytes"] += (
                    2
                    * chunk_rows
                    * self.geometry.head_dim
                    * key_states[0].element_size()
                )
                self.metrics["h2d_copy_calls"] += 2
            chunk.update(
                {
                    "cpu_index_wait_start_ms": 1000.0
                    * (chunk_wait_started - host_origin),
                    "cpu_index_wait_end_ms": 1000.0
                    * (chunk_wait_finished - host_origin),
                    "cpu_decode_start_ms": 1000.0
                    * (decode_started - host_origin),
                    "cpu_decode_end_ms": 1000.0
                    * (decode_finished - host_origin),
                    "cpu_gather_start_ms": 1000.0
                    * (gather_started - host_origin),
                    "cpu_gather_end_ms": 1000.0
                    * (gather_finished - host_origin),
                    "selected_kv_h2d_start_event": h2d_start,
                    "selected_kv_h2d_end_event": h2d_end,
                    "attention_row_start": chunk_attention_start,
                    "attention_row_stop": chunk_attention_stop,
                }
            )
            timeline_chunks.append(chunk)

        copied = self._event()
        with torch.cuda.stream(self._selected_kv_h2d_stream):
            copied.record(self._selected_kv_h2d_stream)
        torch.cuda.current_stream(self.device).wait_event(copied)
        self._layer_flat_ready[slot] = copied

        group_history_lengths = tuple(
            int(selected.numel()) for selected in group_positions
        )
        group_total_lengths = tuple(length + 1 for length in group_history_lengths)
        total_history = sum(group_history_lengths)
        total_attention = sum(group_total_lengths)
        if total_attention != attention_offset:
            raise AssertionError("post-selection attention layout length drift")

        cu_seqlens = torch.tensor(
            [0, *torch.tensor(group_total_lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=self.device,
        )
        for offset, key, value in zip(new_token_offsets, key_states, value_states):
            self._gpu_layer_flat_attention_keys[offset : offset + 1, 0].copy_(
                key[0, 0]
            )
            self._gpu_layer_flat_attention_values[offset : offset + 1, 0].copy_(
                value[0, 0]
            )

        if self._capture_selection_trace:
            for entry, query in zip(entries, queries):
                self._selected_positions(
                    query=query, entry=entry, old_length=old_length
                )
            for group_index, (entry, union) in enumerate(
                zip(entries, group_positions)
            ):
                self._group_union_trace.append(
                    {
                        "decode_step": decode_step,
                        "entry": entry,
                        "layer": layer_index,
                        "kv_head": group_index,
                        "selected_history_indices": tuple(
                            int(value) for value in union.tolist()
                        ),
                    }
                )
        else:
            for entry in entries:
                self._layer_prepared_entries.discard(entry)

        if (
            self._capture_resident_attention_trace
            and decode_step in self._resident_attention_trace_steps
        ):
            key_cpu = (
                self._gpu_layer_flat_attention_keys[:total_attention]
                .detach()
                .cpu()
                .contiguous()
            )
            value_cpu = (
                self._gpu_layer_flat_attention_values[:total_attention]
                .detach()
                .cpu()
                .contiguous()
            )
            self._resident_attention_trace.append(
                {
                    "decode_step": decode_step,
                    "layer": layer_index,
                    "valid_lengths": group_total_lengths,
                    "key_sha256": hashlib.sha256(
                        key_cpu.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "value_sha256": hashlib.sha256(
                        value_cpu.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
            )

        for entry, total_length in zip(entries, group_total_lengths):
            self._last_query_head_lengths[entry] = (
                total_length,
            ) * self.num_query_heads_per_kv
        append_started = time.perf_counter() if self.metrics_enabled else None
        if self.batched_new_kv_d2h:
            self._schedule_batched_new_kv_d2h(
                entries=entries,
                old_length=old_length,
                key_states=key_states,
                value_states=value_states,
            )
        else:
            for entry, new_length, key, value in zip(
                entries, new_lengths, key_states, value_states
            ):
                self._schedule_d2h(entry, old_length, new_length, key, value)

        if self.metrics_enabled:
            for chunk in timeline_chunks:
                union_start = chunk.get("union_start_event")
                union_end = chunk.get("union_end_event")
                index_start = chunk.get("index_d2h_start_event")
                index_end = chunk.get("index_d2h_end_event")
                if union_start is not None and union_end is not None:
                    self._timing_events.append(
                        ("post_selection_union", union_start, union_end)
                    )
                if index_start is not None and index_end is not None:
                    self._timing_events.append(
                        ("post_selection_index_d2h", index_start, index_end)
                    )
            self._post_selection_timeline_events.append(
                {
                    "layer": layer_index,
                    "decode_step": decode_step,
                    "origin": pending["origin"],
                    "chunks": timeline_chunks,
                }
            )
            self.metrics["post_selection_index_wait_wall_seconds"] += (
                index_wait_seconds
            )
            self.metrics["twilight_compact_index_decode_wall_seconds"] += (
                cpu_decode_seconds
            )
            self.metrics["host_gather_pack_wall_seconds"] += cpu_gather_seconds
            self.metrics["h2d_enqueue_wall_seconds"] += pipeline_enqueue_seconds
            self.metrics["host_gather_layer_calls"] += 1
            self.metrics["host_gather_primitive_calls"] += 2 * len(timeline_chunks)
            self.metrics["host_gather_index_select_calls"] += 2 * len(
                timeline_chunks
            )
            self.metrics["host_flat_pack_write_calls"] += 2 * group_count
            self.metrics["layer_gqa_group_h2d_calls"] += 1
            self.metrics["twilight_group_union_prepare_wall_seconds"] += (
                index_wait_seconds + cpu_decode_seconds
            )
            self.metrics["twilight_group_union_prepare_calls"] += 1
            self.metrics["append_and_d2h_enqueue_wall_seconds"] += (
                0.0
                if append_started is None
                else time.perf_counter() - append_started
            )
            self.metrics["cache_update_calls"] += group_count
            self.metrics["layer_gqa_group_update_calls"] += 1
            per_q_lengths = tuple(
                length
                for entry in entries
                for length in (self._cached_twilight_per_q_lengths[entry] or ())
            )
            per_q_history_total = sum(per_q_lengths)
            protected = min(self.sink_tokens, old_length) + min(
                self.recent_tokens - 1,
                max(0, old_length - min(self.sink_tokens, old_length)),
            )
            self.metrics["selected_history_tokens_total"] += per_q_history_total
            if per_q_lengths:
                self.metrics["selected_history_tokens_per_query_head"] += (
                    per_q_history_total / len(per_q_lengths)
                ) * group_count
                self.metrics["selected_adaptive_tokens_total"] += sum(
                    max(0, length - protected) for length in per_q_lengths
                )
            self.metrics["group_union_history_tokens_total"] += total_history
            self.metrics["group_union_history_tokens_min_sum"] += min(
                group_history_lengths
            )
            self.metrics["group_union_history_tokens_max_sum"] += max(
                group_history_lengths
            )
            self.metrics["group_union_duplicate_rows_eliminated"] += (
                per_q_history_total - total_history
            )
            self.metrics["group_attention_valid_tokens_total"] += total_attention
            self.metrics["group_attention_logical_qk_tokens_total"] += sum(
                length * self.num_query_heads_per_kv
                for length in group_total_lengths
            )
            self.metrics["attention_valid_tokens_total"] += total_attention
            self.metrics["attention_valid_tokens_min_sum"] += min(
                group_total_lengths
            )
            self.metrics["attention_valid_tokens_max_sum"] += max(
                group_total_lengths
            )
            if update_started is not None:
                self.metrics["cache_update_total_wall_seconds"] += (
                    time.perf_counter() - update_started
                )
            post_selection_started = self._post_selection_host_starts.pop(
                layer_index, host_origin
            )
            self.metrics["post_selection_exposed_wall_seconds"] += (
                time.perf_counter() - post_selection_started
            )

        return (
            self._gpu_layer_flat_attention_keys[:total_attention],
            self._gpu_layer_flat_attention_values[:total_attention],
            cu_seqlens,
            max(group_total_lengths),
            group_total_lengths,
        )

    def _consume_resident_zero_copy(
        self, *, entries, group_positions, group_history_lengths,
        group_total_lengths, key_states, value_states, old_length,
        new_lengths, decode_step, layer_index,
    ):
        from .resident_gpu_mapping import map_membership

        normal_timeline = bool(
            getattr(self, "_normal_overlap_timeline_enabled", False)
        )
        normal_host_origin = getattr(self, "_normal_overlap_host_origin", None)
        normal_gpu_stream = (
            torch.cuda.current_stream(self.device) if normal_timeline else None
        )
        normal_subphases = []

        def phase_start():
            if not normal_timeline:
                return None
            event = self._event(timing=True)
            event.record(normal_gpu_stream)
            return event, time.perf_counter()

        def phase_end(name, started):
            if started is None:
                return
            event = self._event(timing=True)
            event.record(normal_gpu_stream)
            normal_subphases.append({
                "name": name,
                "start_event": started[0],
                "end_event": event,
                "host_start_ms": (started[1] - normal_host_origin) * 1000.0,
                "host_end_ms": (time.perf_counter() - normal_host_origin) * 1000.0,
            })

        normal_cache_host_start = time.perf_counter() if normal_timeline else None
        normal_cache_gpu_start = phase_start()
        previous_key = self._resident_keys[layer_index]
        previous_value = self._resident_values[layer_index]
        previous_bitmap = self._resident_bitmaps[layer_index]
        current_bitmap = self._gpu_group_membership[:, :old_length]
        started = phase_start()
        if self.resident_precompute_gpu_mapping:
            mapped = self._resident_precomputed_maps.pop(layer_index, None)
            if mapped is None:
                # Diagnostic Selection traces intentionally use the full
                # CPU-visible union path and do not precompute the fast map.
                mapped = map_membership(current_bitmap, previous_bitmap)
        else:
            mapped = map_membership(current_bitmap, previous_bitmap)
        phase_end("gpu_bitmap_hit_miss_mapping", started)
        sources, _, _, current_lengths, previous_lengths, miss_lengths, positions = mapped
        target_key = self._gpu_layer_flat_attention_keys[:, 0]
        target_value = self._gpu_layer_flat_attention_values[:, 0]
        assert self._resident_zero_copy_kernel is not None
        started = phase_start()
        self._resident_zero_copy_kernel(
            previous_key, previous_value, sources, positions,
            current_lengths, previous_lengths, target_key, target_value,
            host_capacity=self.geometry.max_cache_len,
            layer=layer_index, max_group_rows=max(group_history_lengths),
        )
        phase_end("fused_resident_hit_mapped_miss_assembly", started)
        started = phase_start()
        attention_offset = 0
        for key, value, length in zip(
            key_states, value_states, group_history_lengths
        ):
            attention_offset += length
            target_key[attention_offset].copy_(key[0, 0, 0])
            target_value[attention_offset].copy_(value[0, 0, 0])
            attention_offset += 1
        phase_end("current_new_token_slot_copy", started)
        total_history = sum(group_history_lengths)
        total_attention = sum(group_total_lengths)
        started = phase_start()
        self._ensure_resident_layer_capacity(layer_index, total_attention)
        resident_key = self._resident_keys[layer_index]
        resident_value = self._resident_values[layer_index]
        resident_key[:total_attention].copy_(target_key[:total_attention])
        resident_value[:total_attention].copy_(target_value[:total_attention])
        phase_end("resident_kv_snapshot_update", started)
        started = phase_start()
        self._resident_positions[layer_index] = tuple(group_positions)
        self._resident_bitmaps[layer_index] = current_bitmap.clone()
        phase_end("resident_position_bitmap_state_update", started)
        trace_row = {
            "decode_step": decode_step,
            "layer": layer_index,
            "history_rows": total_history,
            "hit_rows": None,
            "miss_rows": None,
            "hit_ratio": None,
            "selected_kv_h2d_bytes": 0,
            "resident_logical_bytes": (
                2 * total_history * self.geometry.head_dim
                * key_states[0].element_size()
            ),
        }
        self._resident_reuse_trace.append(trace_row)
        self._resident_zero_copy_pending_counts.append((trace_row, miss_lengths))
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(group_total_lengths).cumsum(0).tolist()],
            dtype=torch.int32, device=self.device,
        )
        phase_end("cache_metadata_and_control", normal_cache_gpu_start)
        if normal_timeline:
            self._normal_overlap_cache_events.append({
                "decode_step": decode_step,
                "layer": layer_index,
                "cache_host_start_ms": (
                    normal_cache_host_start - normal_host_origin
                ) * 1000.0,
                "cache_host_end_ms": (
                    time.perf_counter() - normal_host_origin
                ) * 1000.0,
                "subphases": normal_subphases,
                "chunks": [],
            })
        if (
            self._capture_resident_attention_trace
            and decode_step in self._resident_attention_trace_steps
        ):
            key_cpu = self._gpu_layer_flat_attention_keys[:total_attention].detach().cpu().contiguous()
            value_cpu = self._gpu_layer_flat_attention_values[:total_attention].detach().cpu().contiguous()
            self._resident_attention_trace.append({
                "decode_step": decode_step,
                "layer": layer_index,
                "valid_lengths": group_total_lengths,
                "key_sha256": hashlib.sha256(key_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                "value_sha256": hashlib.sha256(value_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
            })
        for entry, length in zip(entries, group_total_lengths):
            self._last_query_head_lengths[entry] = (
                length,
            ) * self.num_query_heads_per_kv
        if self.batched_new_kv_d2h:
            self._schedule_batched_new_kv_d2h(
                entries=entries, old_length=old_length,
                key_states=key_states, value_states=value_states,
            )
        else:
            for entry, new_length, key, value in zip(
                entries, new_lengths, key_states, value_states
            ):
                self._schedule_d2h(entry, old_length, new_length, key, value)
        return (
            self._gpu_layer_flat_attention_keys[:total_attention],
            self._gpu_layer_flat_attention_values[:total_attention],
            cu_seqlens, max(group_total_lengths), group_total_lengths,
        )

    def update_layer_gqa_group_ragged(
        self,
        *,
        key_states: list[torch.Tensor],
        value_states: list[torch.Tensor],
        queries: list[torch.Tensor],
        entries: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[int, ...]]:
        """Pack one shared union of selected K/V for every GQA query group.

        Quest B0 and Twilight Top-p remain per Query head.  For each KV head,
        the final history is the exact set union of the selected histories of
        its Query heads.  That union is gathered and transferred once, then
        consumed by a varlen attention sequence with ``num_q_heads_per_kv``
        Query heads and one KV head, matching Twilight Appendix B.2.
        """
        if not self.gqa_groupwise_execution:
            raise RuntimeError("Twilight GQA group-wise execution is disabled")
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
                raise ValueError("GQA group decode expects [1, 1, 1, dim] K/V")
            old_length, new_length = self._bookkeeping.preview(entry, 1)
            self._bookkeeping.advance(entry, 1)
            self._initialized[entry] = True
            old_lengths.append(old_length)
            new_lengths.append(new_length)
        self._seen_tokens = self._bookkeeping.seen_tokens
        if len(set(old_lengths)) != 1:
            raise AssertionError("layer KV entries have unequal logical lengths")
        old_length = old_lengths[0]
        decode_step = (
            old_length - self._prompt_length + 1
            if self._prompt_length is not None
            else old_length + 1
        )
        normal_timeline = bool(
            getattr(self, "_normal_overlap_timeline_enabled", False)
        )
        normal_host_origin = getattr(
            self, "_normal_overlap_host_origin", None
        )
        normal_cache_host_start = time.perf_counter() if normal_timeline else None
        normal_cache_chunks: list[dict[str, Any]] = []

        update_started = time.perf_counter() if self.metrics_enabled else None
        phase_started = time.perf_counter() if self.metrics_enabled else None
        if self.batched_new_kv_d2h:
            batch_id = (
                0
                if self.new_kv_d2h_granularity == "token"
                else entries[0] // len(entries)
            )
            self._flush_batched_new_kv(batch_id)
        else:
            for entry in entries:
                host_ready = self._host_write_done[entry]
                if host_ready is not None:
                    host_ready.synchronize()
        if self.metrics_enabled and not self.batched_new_kv_d2h:
            self.metrics["host_ready_wait_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
        self._verify_new_kv_before_read(entries, decode_step)
        if self.post_selection_pipeline:
            return self._consume_post_selection_pipeline(
                key_states=key_states,
                value_states=value_states,
                queries=queries,
                entries=entries,
                old_length=old_length,
                new_lengths=new_lengths,
                decode_step=decode_step,
                update_started=update_started,
            )

        layer_positions: list[list[torch.Tensor]] = []
        union_started = time.perf_counter() if self.metrics_enabled else None
        if self.gpu_compact_gqa_union:
            group_positions = []
            per_q_lengths_parts: list[int] = []
            for entry, query in zip(entries, queries):
                union = self._cached_twilight_group_positions[entry]
                if union is None:
                    raise AssertionError("prepared GPU GQA union is missing")
                group_positions.append(union)
                cached_lengths = self._cached_twilight_per_q_lengths[entry]
                if cached_lengths is not None:
                    per_q_lengths_parts.extend(cached_lengths)
                if self._capture_selection_trace:
                    # Diagnostic-only: prepare_layer_selection retained exact
                    # per-Q rows so the existing trace format remains usable.
                    layer_positions.append(
                        self._selected_positions(
                            query=query, entry=entry, old_length=old_length
                        )
                    )
                else:
                    self._layer_prepared_entries.discard(entry)
            per_q_lengths = tuple(per_q_lengths_parts)
            if self.metrics_enabled:
                self.metrics["twilight_gpu_group_union_handoff_calls"] += 1
        else:
            # Reproduce the unmodified per-Query-head selected sets first.  The
            # CPU union below is the baseline selection-semantic transformation.
            for entry, query in zip(entries, queries):
                positions = self._selected_positions(
                    query=query, entry=entry, old_length=old_length
                )
                layer_positions.append(positions)
            group_positions = [
                self._cpu_token_union(positions, old_length)
                if self.cpu_bitmap_union
                else torch.unique(torch.cat(positions), sorted=True)
                for positions in layer_positions
            ]
            per_q_lengths = tuple(
                int(selected.numel())
                for positions in layer_positions
                for selected in positions
            )
            for entry, union in zip(entries, group_positions):
                self._cached_twilight_group_positions[entry] = union
        if self._capture_selection_trace:
            decode_step = (
                old_length - self._prompt_length + 1
                if self._prompt_length is not None
                else None
            )
            groups_per_layer = len(entries)
            for group_index, (entry, union) in enumerate(
                zip(entries, group_positions)
            ):
                self._group_union_trace.append(
                    {
                        "decode_step": decode_step,
                        "entry": entry,
                        "layer": entries[0] // groups_per_layer,
                        "kv_head": group_index,
                        "selected_history_indices": tuple(
                            int(value) for value in union.tolist()
                        ),
                    }
                )
        if self.metrics_enabled:
            assert union_started is not None
            self.metrics["twilight_group_union_prepare_wall_seconds"] += (
                time.perf_counter() - union_started
            )
            self.metrics["twilight_group_union_prepare_calls"] += 1

        group_history_lengths = tuple(
            int(selected.numel()) for selected in group_positions
        )
        group_total_lengths = tuple(length + 1 for length in group_history_lengths)
        per_q_history_total = sum(per_q_lengths)
        total_history = sum(group_history_lengths)
        total_attention = sum(group_total_lengths)
        self._ensure_audit_attention_capacity(total_attention)
        if per_q_lengths and total_history > per_q_history_total:
            raise AssertionError("GQA union cannot exceed per-Query-head payload")

        layer_index = entries[0] // group_count
        if entries != list(
            range(layer_index * group_count, (layer_index + 1) * group_count)
        ):
            raise AssertionError("GQA entries must be one contiguous decoder layer")
        decode_step = (
            old_length - self._prompt_length + 1
            if self._prompt_length is not None
            else -1
        )
        previous_positions = self._resident_positions.get(layer_index)
        reuse_active = bool(
            self.previous_token_resident_cache
            and previous_positions is not None
            and layer_index in self._resident_keys
        )
        if self.resident_zero_copy and reuse_active:
            return self._consume_resident_zero_copy(
                entries=entries, group_positions=group_positions,
                group_history_lengths=group_history_lengths,
                group_total_lengths=group_total_lengths,
                key_states=key_states, value_states=value_states,
                old_length=old_length, new_lengths=new_lengths,
                decode_step=decode_step, layer_index=layer_index,
            )
        hit_source_parts: list[torch.Tensor] = []
        packed_hit_indices: Optional[torch.Tensor] = None
        hit_destination_parts: list[torch.Tensor] = []
        miss_row_parts: list[torch.Tensor] = []
        miss_destination_parts: list[torch.Tensor] = []
        history_destination_parts: list[torch.Tensor] = []
        previous_offset = 0
        attention_offset = 0
        resident_mapping_started = (
            time.perf_counter()
            if self.metrics_enabled and self.previous_token_resident_cache
            else None
        )
        gpu_map_active = bool(
            reuse_active and self.resident_mapping_backend == "gpu_bitmap"
        )
        gpu_mapping = None
        if gpu_map_active:
            from .resident_gpu_mapping import map_membership
            previous_bitmap = self._resident_bitmaps[layer_index]
            current_bitmap = self._gpu_group_membership[:, :old_length]
            gpu_mapping = map_membership(current_bitmap, previous_bitmap)
            gpu_sources, gpu_miss_positions, gpu_miss_destinations, gpu_lengths, gpu_previous_lengths, gpu_miss_lengths = gpu_mapping
            required_host = gpu_miss_positions.numel()
            if (self._resident_miss_positions_host is None
                    or self._resident_miss_positions_host.numel() < required_host):
                self._resident_miss_positions_host = torch.empty(
                    max(required_host, 8 * self.geometry.max_cache_len),
                    dtype=torch.int32, pin_memory=True,
                )
            host_miss_positions = self._resident_miss_positions_host[:required_host]
            host_miss_positions.copy_(gpu_miss_positions.flatten(), non_blocking=True)
            host_miss_lengths = gpu_miss_lengths.cpu().tolist()
            torch.cuda.current_stream(self.device).synchronize()
            if gpu_lengths.cpu().tolist() != list(group_history_lengths):
                raise AssertionError("GPU resident selected lengths changed")
            if gpu_previous_lengths.cpu().tolist() != [
                int(x.numel()) for x in previous_positions
            ]:
                raise AssertionError("GPU resident previous lengths changed")
            miss_parts = []
            for group_index, miss_length in enumerate(host_miss_lengths):
                if miss_length:
                    start = group_index * old_length
                    positions = host_miss_positions[start:start + miss_length].numpy()
                    miss_parts.append(
                        positions.astype(np.int64) +
                        entries[group_index] * self.geometry.max_cache_len
                    )
            miss_rows = torch.from_numpy(np.concatenate(miss_parts)) if miss_parts else torch.empty(0, dtype=torch.long)
            miss_count = int(miss_rows.numel())
            hit_count = total_history - miss_count
            dummy = torch.empty(1, dtype=torch.int32)
            hit_sources = dummy.expand(hit_count)
            hit_destinations = dummy.expand(hit_count)
            miss_destinations = dummy.expand(miss_count)
            history_destinations = dummy.expand(total_history)
        elif reuse_active and self._native_resident_mapping is not None:
            assert previous_positions is not None
            if len(previous_positions) != group_count:
                raise AssertionError("resident GQA group count changed")
            mapping_outputs = self._native_resident_mapping(
                entries, group_positions, previous_positions
            )
            (
                hit_sources, hit_destinations, miss_rows,
                miss_destinations, history_destinations,
            ) = mapping_outputs[:5]
            packed_hit_indices = (
                mapping_outputs[5] if self.resident_pack_hit_indices else None
            )
        elif reuse_active:
            assert previous_positions is not None
            if len(previous_positions) != group_count:
                raise AssertionError("resident GQA group count changed")
            for entry, current, previous in zip(
                entries, group_positions, previous_positions
            ):
                matches = torch.searchsorted(previous, current)
                valid = matches.lt(previous.numel())
                safe_matches = matches.clamp_max(max(0, previous.numel() - 1))
                hits = valid & previous.index_select(0, safe_matches).eq(current)
                current_rows = torch.arange(current.numel(), dtype=torch.long)
                hit_rows = current_rows[hits]
                missed_rows = current_rows[~hits]
                hit_source_parts.append(previous_offset + matches[hits])
                hit_destination_parts.append(attention_offset + hit_rows)
                miss_row_parts.append(
                    entry * self.geometry.max_cache_len + current[~hits]
                )
                miss_destination_parts.append(attention_offset + missed_rows)
                history_destination_parts.append(attention_offset + current_rows)
                previous_offset += previous.numel()
                attention_offset += current.numel() + 1
        if not gpu_map_active and not (reuse_active and self._native_resident_mapping is not None):
            empty_indices = torch.empty(0, dtype=torch.long)
            hit_sources = (
                torch.cat(hit_source_parts) if hit_source_parts else empty_indices
            )
            hit_destinations = (
                torch.cat(hit_destination_parts)
                if hit_destination_parts
                else empty_indices
            )
            miss_rows = torch.cat(miss_row_parts) if miss_row_parts else empty_indices
            miss_destinations = (
                torch.cat(miss_destination_parts)
                if miss_destination_parts
                else empty_indices
            )
            history_destinations = (
                torch.cat(history_destination_parts)
                if history_destination_parts
                else empty_indices
            )
        if reuse_active and hit_sources.numel() + miss_rows.numel() != total_history:
            raise AssertionError("resident hit/miss partition is incomplete")
        if resident_mapping_started is not None:
            self.metrics["resident_mapping_wall_seconds"] += (
                time.perf_counter() - resident_mapping_started
            )
            self.metrics["resident_mapping_calls"] += 1

        def build_cu_seqlens():
            started = time.perf_counter() if self.metrics_enabled else None
            result = torch.tensor(
                [0, *torch.tensor(group_total_lengths).cumsum(0).tolist()],
                dtype=torch.int32, device=self.device,
            )
            if started is not None:
                self.metrics['gpu_metadata_build_wall_seconds'] += time.perf_counter() - started
            return result

        cu_seqlens = build_cu_seqlens() if self.early_gpu_metadata else None

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
        transfer_rows = (
            int(miss_rows.numel())
            if reuse_active
            else total_attention if self.direct_attention_layout else total_history
        )
        if transfer_rows > host_flat_key.shape[0]:
            raise ValueError('insufficient host pack capacity for direct attention layout')
        new_token_offsets = []
        pipeline_enqueue_seconds = 0.0
        pipeline_chunks = 0
        for entry, selected in zip(entries, group_positions):
            if reuse_active:
                continue
            if not (self.skip_unused_host_views and self.cpu_flat_gather):
                host_key = self._host_tensor("key", entry)[0, 0, :old_length]
                host_value = self._host_tensor("value", entry)[0, 0, :old_length]
            stop = history_offset + int(selected.numel())
            if self.cpu_flat_gather:
                torch.add(selected, entry * self.geometry.max_cache_len,
                          out=self._host_layer_flat_row_indices[slot][history_offset:stop])
            else:
                torch.index_select(
                    host_key, 0, selected, out=host_flat_key[history_offset:stop]
                )
                torch.index_select(
                    host_value, 0, selected, out=host_flat_value[history_offset:stop]
                )
            if self.direct_attention_layout:
                # Initialized source row for each gap; overwritten by the GPU
                # new token on the same stream before Attention can read it.
                self._host_layer_flat_row_indices[slot][stop] = 0
                new_token_offsets.append(stop)
                stop += 1
            history_offset = stop
        if reuse_active:
            self._host_layer_flat_row_indices[slot][:transfer_rows].copy_(miss_rows)
            history_offset = transfer_rows
        if self.cpu_flat_gather:
            rows = self._host_layer_flat_row_indices[slot][:transfer_rows]
            if self.gather_h2d_chunks:
                # Disjoint slices of persistent pinned storage: CPU writes the
                # next slice while the current stream transfers the previous.
                # The existing slot completion event protects reuse next layer.
                chunk_rows = max(1, (transfer_rows + self.gather_h2d_chunks - 1) // self.gather_h2d_chunks)
                for chunk_start in range(0, transfer_rows, chunk_rows):
                    chunk_stop = min(transfer_rows, chunk_start + chunk_rows)
                    normal_gather_start = (
                        time.perf_counter() if normal_timeline else None
                    )
                    if self.cpu_run_gather:
                        self._cpu_run_gather(self._host_key_rows, self._host_value_rows,
                            rows[chunk_start:chunk_stop], host_flat_key[chunk_start:chunk_stop],
                            host_flat_value[chunk_start:chunk_stop])
                    else:
                        torch.index_select(self._host_key_rows, 0, rows[chunk_start:chunk_stop],
                                           out=host_flat_key[chunk_start:chunk_stop])
                        torch.index_select(self._host_value_rows, 0, rows[chunk_start:chunk_stop],
                                           out=host_flat_value[chunk_start:chunk_stop])
                    normal_gather_end = (
                        time.perf_counter() if normal_timeline else None
                    )
                    def copy_chunk(start=chunk_start, stop=chunk_stop):
                        target_key = (
                            self._gpu_layer_flat_history_keys
                            if reuse_active
                            else self._gpu_layer_flat_attention_keys[:, 0]
                        )
                        target_value = (
                            self._gpu_layer_flat_history_values
                            if reuse_active
                            else self._gpu_layer_flat_attention_values[:, 0]
                        )
                        target_key[start:stop].copy_(
                            host_flat_key[start:stop], non_blocking=True
                        )
                        target_value[start:stop].copy_(
                            host_flat_value[start:stop], non_blocking=True
                        )
                    enqueue_started = time.perf_counter() if self.metrics_enabled else None
                    normal_h2d_start = (
                        self._event(timing=True) if normal_timeline else None
                    )
                    if normal_h2d_start is not None:
                        normal_h2d_start.record(
                            torch.cuda.current_stream(self.device)
                        )
                    normal_h2d_enqueue_start = (
                        time.perf_counter() if normal_timeline else None
                    )
                    self._record_copy('h2d', copy_chunk,
                        2 * (chunk_stop-chunk_start) * self.geometry.head_dim * key_states[0].element_size(), copy_calls=2)
                    normal_h2d_enqueue_end = (
                        time.perf_counter() if normal_timeline else None
                    )
                    normal_h2d_end = (
                        self._event(timing=True) if normal_timeline else None
                    )
                    if normal_h2d_end is not None:
                        normal_h2d_end.record(
                            torch.cuda.current_stream(self.device)
                        )
                    if normal_timeline:
                        if (
                            normal_host_origin is None
                            or normal_gather_start is None
                            or normal_gather_end is None
                            or normal_h2d_enqueue_start is None
                            or normal_h2d_enqueue_end is None
                        ):
                            raise AssertionError(
                                "normal-overlap chunk timestamps are missing"
                            )
                        normal_cache_chunks.append(
                            {
                                "chunk_start_row": chunk_start,
                                "chunk_stop_row": chunk_stop,
                                "cpu_gather_start_ms": (
                                    normal_gather_start - normal_host_origin
                                )
                                * 1000.0,
                                "cpu_gather_end_ms": (
                                    normal_gather_end - normal_host_origin
                                )
                                * 1000.0,
                                "h2d_enqueue_start_ms": (
                                    normal_h2d_enqueue_start - normal_host_origin
                                )
                                * 1000.0,
                                "h2d_enqueue_end_ms": (
                                    normal_h2d_enqueue_end - normal_host_origin
                                )
                                * 1000.0,
                                "selected_kv_h2d_start_event": normal_h2d_start,
                                "selected_kv_h2d_end_event": normal_h2d_end,
                            }
                        )
                    if enqueue_started is not None:
                        pipeline_enqueue_seconds += time.perf_counter() - enqueue_started
                    pipeline_chunks += 1
            elif self.cpu_native_gather:
                self._cpu_kv_gather(self._host_key_rows,self._host_value_rows,rows,
                                    host_flat_key[:transfer_rows],host_flat_value[:transfer_rows])
            else:
                torch.index_select(self._host_key_rows, 0, rows,
                                   out=host_flat_key[:transfer_rows])
                torch.index_select(self._host_value_rows, 0, rows,
                                   out=host_flat_value[:transfer_rows])
        if history_offset != transfer_rows:
            raise AssertionError("GQA group history packing length mismatch")
        if self.metrics_enabled:
            self.metrics["host_gather_pack_wall_seconds"] += (
                time.perf_counter() - phase_started - pipeline_enqueue_seconds
            )
            self.metrics["host_gather_layer_calls"] += 1
            self.metrics["host_gather_primitive_calls"] += (1 if self.cpu_run_gather else 2) * pipeline_chunks if self.gather_h2d_chunks else (1 if self.cpu_native_gather else (2 if self.cpu_flat_gather else 2 * group_count))
            self.metrics["host_gather_index_select_calls"] += (0 if self.cpu_run_gather else 2) * pipeline_chunks if self.gather_h2d_chunks else (0 if self.cpu_native_gather else (2 if self.cpu_flat_gather else 2 * group_count))
            self.metrics["host_gather_index_concat_calls"] += 0
            self.metrics["host_gather_index_temporary_allocations"] += 0
            self.metrics["host_flat_pack_write_calls"] += 2 * group_count

        assert self._gpu_layer_flat_history_keys is not None
        assert self._gpu_layer_flat_history_values is not None
        byte_count = (
            2 * transfer_rows * self.geometry.head_dim * key_states[0].element_size()
        )

        def copy_layer_history() -> None:
            target_key = (self._gpu_layer_flat_attention_keys[:, 0] if self.direct_attention_layout
                          else self._gpu_layer_flat_history_keys)
            target_value = (self._gpu_layer_flat_attention_values[:, 0] if self.direct_attention_layout
                            else self._gpu_layer_flat_history_values)
            target_key[:transfer_rows].copy_(
                host_flat_key[:transfer_rows], non_blocking=True
            )
            target_value[:transfer_rows].copy_(
                host_flat_value[:transfer_rows], non_blocking=True
            )

        phase_started = time.perf_counter() if self.metrics_enabled else None
        if not self.gather_h2d_chunks:
            self._record_copy("h2d", copy_layer_history, byte_count, copy_calls=2)
        if self.metrics_enabled:
            self.metrics["h2d_enqueue_wall_seconds"] += (
                pipeline_enqueue_seconds if self.gather_h2d_chunks else time.perf_counter() - phase_started
            )
            self.metrics["layer_gqa_group_h2d_calls"] += 1
            phase_started = time.perf_counter()
        copied = self._event()
        copied.record(torch.cuda.current_stream(self.device))
        self._layer_flat_ready[slot] = copied

        assert self._gpu_layer_flat_attention_keys is not None
        assert self._gpu_layer_flat_attention_values is not None
        gpu_indices: dict[str, torch.Tensor] = {}
        if reuse_active:
            if not gpu_map_active:
                index_parts = (
                    hit_sources, hit_destinations, miss_destinations,
                )
                if self.resident_snapshot_backend != "contiguous":
                    index_parts += (history_destinations,)
                index_bytes = sum(
                    index.numel() * index.element_size()
                    for index in index_parts
                )
                self._resident_index_logical_peak_bytes = max(
                    self._resident_index_logical_peak_bytes, index_bytes
                )

                def copy_resident_indices() -> None:
                    if packed_hit_indices is not None:
                        gpu_packed_hits = packed_hit_indices.to(self.device)
                        split = int(hit_sources.numel())
                        gpu_indices["hit_sources"] = gpu_packed_hits[:split]
                        gpu_indices["hit_destinations"] = gpu_packed_hits[split:]
                    else:
                        gpu_indices["hit_sources"] = hit_sources.to(self.device)
                        gpu_indices["hit_destinations"] = hit_destinations.to(self.device)
                    gpu_indices["miss_destinations"] = miss_destinations.to(self.device)
                    if self.resident_snapshot_backend != "contiguous":
                        gpu_indices["history_destinations"] = (
                            history_destinations.to(self.device)
                        )

                self._record_copy(
                    "resident_index_h2d",
                    copy_resident_indices,
                    index_bytes,
                    copy_calls=(
                        2 if self.resident_pack_hit_indices
                        else 3 if self.resident_snapshot_backend == "contiguous"
                        else 4
                    ),
                )
            previous_key = self._resident_keys[layer_index]
            previous_value = self._resident_values[layer_index]
            hit_count = int(hit_sources.numel())
            miss_count = int(miss_rows.numel())
            if self.resident_hit_copy_backend == "torch":
                self._resident_hit_gather_logical_peak_bytes = max(
                    self._resident_hit_gather_logical_peak_bytes,
                    2
                    * hit_count
                    * self.geometry.head_dim
                    * key_states[0].element_size(),
                )

            def assemble_resident_hits() -> None:
                if hit_count:
                    if gpu_map_active:
                        from .resident_gpu_mapping import copy_hits
                        assert gpu_mapping is not None
                        copy_hits(
                            previous_key, previous_value,
                            self._gpu_layer_flat_attention_keys[:, 0],
                            self._gpu_layer_flat_attention_values[:, 0],
                            gpu_mapping[0], gpu_mapping[3], gpu_mapping[4],
                            max(group_history_lengths),
                        )
                    elif self.resident_hit_copy_backend == "triton":
                        from .resident_hit_copy import copy_resident_hits
                        copy_resident_hits(
                            previous_key, previous_value,
                            self._gpu_layer_flat_attention_keys[:, 0],
                            self._gpu_layer_flat_attention_values[:, 0],
                            gpu_indices["hit_sources"],
                            gpu_indices["hit_destinations"],
                        )
                    else:
                        self._gpu_layer_flat_attention_keys[:, 0].index_copy_(
                            0,
                            gpu_indices["hit_destinations"],
                            previous_key.index_select(0, gpu_indices["hit_sources"]),
                        )
                        self._gpu_layer_flat_attention_values[:, 0].index_copy_(
                            0,
                            gpu_indices["hit_destinations"],
                            previous_value.index_select(0, gpu_indices["hit_sources"]),
                        )

            self._record_copy(
                "resident_hit_copy",
                assemble_resident_hits,
                2
                * hit_count
                * self.geometry.head_dim
                * key_states[0].element_size(),
                copy_calls=(1 if self.resident_hit_copy_backend == "triton"
                            else 2) if hit_count else 0,
            )

            def assemble_resident_misses() -> None:
                if miss_count:
                    if gpu_map_active:
                        from .resident_gpu_mapping import scatter_misses
                        assert gpu_mapping is not None
                        scatter_misses(
                            self._gpu_layer_flat_history_keys,
                            self._gpu_layer_flat_history_values,
                            self._gpu_layer_flat_attention_keys[:, 0],
                            self._gpu_layer_flat_attention_values[:, 0],
                            gpu_mapping[2], gpu_mapping[3], gpu_mapping[5],
                            max(host_miss_lengths),
                        )
                    else:
                        self._gpu_layer_flat_attention_keys[:, 0].index_copy_(
                            0,
                            gpu_indices["miss_destinations"],
                            self._gpu_layer_flat_history_keys[:miss_count],
                        )
                        self._gpu_layer_flat_attention_values[:, 0].index_copy_(
                            0,
                            gpu_indices["miss_destinations"],
                            self._gpu_layer_flat_history_values[:miss_count],
                        )

            self._record_copy(
                "resident_miss_scatter",
                assemble_resident_misses,
                2
                * miss_count
                * self.geometry.head_dim
                * key_states[0].element_size(),
                copy_calls=(1 if gpu_map_active else 2) if miss_count else 0,
            )
            attention_offset = 0
            new_token_offsets = []
            for history_length in group_history_lengths:
                new_token_offsets.append(attention_offset + history_length)
                attention_offset += history_length + 1
        key_segments: list[torch.Tensor] = []
        value_segments: list[torch.Tensor] = []
        history_offset = 0
        for key, value, history_length in zip(
            key_states, value_states, group_history_lengths
        ):
            stop = history_offset + history_length
            key_segments.extend(
                (self._gpu_layer_flat_history_keys[history_offset:stop], key[0, 0])
            )
            value_segments.extend(
                (self._gpu_layer_flat_history_values[history_offset:stop], value[0, 0])
            )
            history_offset = stop
        if self.direct_attention_layout:
            for offset, key, value in zip(new_token_offsets, key_states, value_states):
                self._gpu_layer_flat_attention_keys[offset:offset+1, 0].copy_(key[0, 0])
                self._gpu_layer_flat_attention_values[offset:offset+1, 0].copy_(value[0, 0])
        else:
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

        if self.previous_token_resident_cache:
            resident_rows = (
                total_attention
                if self.resident_snapshot_backend == "contiguous"
                else total_history
            )
            self._ensure_resident_layer_capacity(layer_index, resident_rows)
            resident_key = self._resident_keys[layer_index]
            resident_value = self._resident_values[layer_index]
            gpu_history_destinations = None
            if self.resident_snapshot_backend != "contiguous":
                if reuse_active:
                    gpu_history_destinations = gpu_indices["history_destinations"]
                else:
                    destinations = []
                    attention_offset = 0
                    for history_length in group_history_lengths:
                        destinations.append(
                            attention_offset
                            + torch.arange(history_length, dtype=torch.long)
                        )
                        attention_offset += history_length + 1
                    host_history_destinations = torch.cat(destinations)
                    gpu_history_destinations = host_history_destinations.to(self.device)

            def snapshot_current_union() -> None:
                if self.resident_snapshot_backend == "contiguous":
                    resident_key[:total_attention].copy_(
                        self._gpu_layer_flat_attention_keys[:total_attention, 0]
                    )
                    resident_value[:total_attention].copy_(
                        self._gpu_layer_flat_attention_values[:total_attention, 0]
                    )
                elif self.resident_snapshot_backend == "triton":
                    assert gpu_history_destinations is not None
                    from .resident_hit_copy import snapshot_resident_union
                    snapshot_resident_union(
                        self._gpu_layer_flat_attention_keys[:, 0],
                        self._gpu_layer_flat_attention_values[:, 0],
                        resident_key, resident_value,
                        gpu_history_destinations, total_history,
                    )
                else:
                    assert gpu_history_destinations is not None
                    torch.index_select(
                        self._gpu_layer_flat_attention_keys[:, 0],
                        0,
                        gpu_history_destinations,
                        out=resident_key[:total_history],
                    )
                    torch.index_select(
                        self._gpu_layer_flat_attention_values[:, 0],
                        0,
                        gpu_history_destinations,
                        out=resident_value[:total_history],
                    )

            self._record_copy(
                "resident_snapshot_copy",
                snapshot_current_union,
                2
                * resident_rows
                * self.geometry.head_dim
                * key_states[0].element_size(),
                copy_calls=2,
            )
            resident_position_snapshot_started = (
                time.perf_counter() if self.metrics_enabled else None
            )
            self._resident_positions[layer_index] = (
                tuple(group_positions)
                if self.resident_position_snapshot_backend == "reference"
                else tuple(selected.clone() for selected in group_positions)
            )
            if self.resident_mapping_backend == "gpu_bitmap":
                self._resident_bitmaps[layer_index] = (
                    self._gpu_group_membership[:, :old_length].clone()
                )
                if decode_step == 1 and layer_index == 0:
                    # D1 is excluded from TPOT.  Compile the stable-shape
                    # kernels here so D2 does not pay JIT compilation cost.
                    from .resident_gpu_mapping import (
                        map_membership, copy_hits, scatter_misses,
                    )
                    bitmap = self._resident_bitmaps[layer_index]
                    warm = map_membership(bitmap, bitmap)
                    if self.resident_zero_copy:
                        from .resident_zero_copy import ResidentZeroCopy
                        self._resident_zero_copy_kernel = ResidentZeroCopy(
                            self._host_key_rows, self._host_value_rows
                        )
                        self._resident_zero_copy_kernel(
                            resident_key, resident_value,
                            warm[0], warm[6], warm[3], warm[4],
                            self._gpu_layer_flat_attention_keys[:, 0],
                            self._gpu_layer_flat_attention_values[:, 0],
                            host_capacity=self.geometry.max_cache_len,
                            layer=layer_index,
                            max_group_rows=max(group_history_lengths),
                        )
                    else:
                        copy_hits(
                            resident_key, resident_value,
                            self._gpu_layer_flat_attention_keys[:, 0],
                            self._gpu_layer_flat_attention_values[:, 0],
                            warm[0], warm[3], warm[4],
                            max(group_history_lengths),
                        )
                        scatter_misses(
                            self._gpu_layer_flat_history_keys,
                            self._gpu_layer_flat_history_values,
                            self._gpu_layer_flat_attention_keys[:, 0],
                            self._gpu_layer_flat_attention_values[:, 0],
                            warm[2], warm[3], warm[5], 1,
                        )
            if resident_position_snapshot_started is not None:
                self.metrics["resident_position_snapshot_wall_seconds"] += (
                    time.perf_counter() - resident_position_snapshot_started
                )
                self.metrics["resident_position_snapshot_calls"] += 1
            resident_hit_rows = int(hit_sources.numel()) if reuse_active else 0
            resident_miss_rows = (
                int(miss_rows.numel()) if reuse_active else total_history
            )
            self._resident_reuse_trace.append(
                {
                    "decode_step": decode_step,
                    "layer": layer_index,
                    "history_rows": total_history,
                    "hit_rows": resident_hit_rows,
                    "miss_rows": resident_miss_rows,
                    "hit_ratio": (
                        resident_hit_rows / total_history if total_history else 0.0
                    ),
                    "selected_kv_h2d_bytes": (
                        2
                        * resident_miss_rows
                        * self.geometry.head_dim
                        * key_states[0].element_size()
                    ),
                    "resident_logical_bytes": (
                        2
                        * total_history
                        * self.geometry.head_dim
                        * key_states[0].element_size()
                    ),
                }
            )
        if cu_seqlens is None:
            cu_seqlens = build_cu_seqlens()

        if (
            self._capture_resident_attention_trace
            and decode_step in self._resident_attention_trace_steps
        ):
            key_cpu = (
                self._gpu_layer_flat_attention_keys[:total_attention]
                .detach()
                .cpu()
                .contiguous()
            )
            value_cpu = (
                self._gpu_layer_flat_attention_values[:total_attention]
                .detach()
                .cpu()
                .contiguous()
            )
            self._resident_attention_trace.append(
                {
                    "decode_step": decode_step,
                    "layer": layer_index,
                    "valid_lengths": group_total_lengths,
                    "key_sha256": hashlib.sha256(
                        key_cpu.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                    "value_sha256": hashlib.sha256(
                        value_cpu.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
            )

        for entry, total_length in zip(entries, group_total_lengths):
            self._last_query_head_lengths[entry] = (
                total_length,
            ) * self.num_query_heads_per_kv
        if self.batched_new_kv_d2h:
            self._schedule_batched_new_kv_d2h(
                entries=entries,
                old_length=old_length,
                key_states=key_states,
                value_states=value_states,
            )
        else:
            for entry, new_length, key, value in zip(
                entries, new_lengths, key_states, value_states
            ):
                if self._capture_new_kv_trace:
                    self._new_kv_control_pending[entry] = (
                        decode_step,
                        old_length,
                    )
                self._schedule_d2h(entry, old_length, new_length, key, value)

        if self.metrics_enabled:
            self.metrics["append_and_d2h_enqueue_wall_seconds"] += (
                time.perf_counter() - phase_started
            )
            self.metrics["cache_update_calls"] += group_count
            self.metrics["layer_gqa_group_update_calls"] += 1
            protected = min(self.sink_tokens, old_length) + min(
                self.recent_tokens - 1,
                max(0, old_length - min(self.sink_tokens, old_length)),
            )
            # Preserve the original per-Q visibility metric independently from
            # the physical shared-union payload.
            self.metrics["selected_history_tokens_total"] += per_q_history_total
            self.metrics["selected_history_tokens_per_query_head"] += (
                per_q_history_total / len(per_q_lengths)
            ) * group_count
            self.metrics["selected_adaptive_tokens_total"] += sum(
                max(0, length - protected) for length in per_q_lengths
            )
            self.metrics["group_union_history_tokens_total"] += total_history
            self.metrics["group_union_history_tokens_min_sum"] += min(
                group_history_lengths
            )
            self.metrics["group_union_history_tokens_max_sum"] += max(
                group_history_lengths
            )
            self.metrics["group_union_duplicate_rows_eliminated"] += (
                per_q_history_total - total_history
            )
            self.metrics["group_attention_valid_tokens_total"] += total_attention
            self.metrics["group_attention_logical_qk_tokens_total"] += sum(
                length * self.num_query_heads_per_kv
                for length in group_total_lengths
            )
            self.metrics["attention_valid_tokens_total"] += total_attention
            self.metrics["attention_valid_tokens_min_sum"] += min(
                group_total_lengths
            )
            self.metrics["attention_valid_tokens_max_sum"] += max(
                group_total_lengths
            )
            assert update_started is not None
            self.metrics["cache_update_total_wall_seconds"] += (
                time.perf_counter() - update_started
            )
            post_selection_started = self._post_selection_host_starts.pop(
                layer_index, None
            )
            if post_selection_started is not None:
                self.metrics["post_selection_exposed_wall_seconds"] += (
                    time.perf_counter() - post_selection_started
                )
        if normal_timeline:
            if normal_host_origin is None or normal_cache_host_start is None:
                raise AssertionError("normal-overlap cache origin is missing")
            self._normal_overlap_cache_events.append(
                {
                    "decode_step": decode_step,
                    "layer": layer_index,
                    "cache_host_start_ms": (
                        normal_cache_host_start - normal_host_origin
                    )
                    * 1000.0,
                    "cache_host_end_ms": (
                        time.perf_counter() - normal_host_origin
                    )
                    * 1000.0,
                    "chunks": normal_cache_chunks,
                }
            )
        return (
            self._gpu_layer_flat_attention_keys[:total_attention],
            self._gpu_layer_flat_attention_values[:total_attention],
            cu_seqlens,
            max(group_total_lengths),
            group_total_lengths,
        )

    def synchronize(self) -> None:
        super().synchronize()
        if self.batched_new_kv_d2h:
            self._flush_all_batched_new_kv()
        elif self._capture_new_kv_trace and self._new_kv_control_pending:
            pending_by_step: dict[tuple[int, int], list[int]] = {}
            for entry, (decode_step, position) in self._new_kv_control_pending.items():
                pending_by_step.setdefault((decode_step, position), []).append(entry)
            self._new_kv_control_pending = {}
            for (decode_step, position), pending_entries in sorted(
                pending_by_step.items()
            ):
                self._capture_new_kv_host_rows(
                    tuple(sorted(pending_entries)),
                    decode_step=decode_step,
                    position=position,
                )

    def state_snapshot(self) -> dict[str, Any]:
        snapshot = super().state_snapshot()
        quant_bytes = 0
        for tensors in (
            self._quant_key_packed,
            self._quant_key_scale,
            self._quant_key_minimum,
        ):
            quant_bytes += sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors
                if tensor is not None
            )
        snapshot.update(
            {
                "policy": (
                    "twilight_inspired_int4_top_p_dynamic"
                    if self.budget_mode == "dynamic"
                    else "twilight_inspired_int4_top_p_matched_quest_h2d"
                ),
                "candidate_token_budget": self.candidate_token_budget,
                "top_p": self.top_p,
                "matched_budget_fraction": self.matched_budget_fraction,
                "budget_mode": self.budget_mode,
                "quantized_key_metadata_gpu_bytes": quant_bytes,
                "official_twilight_kernel_used": False,
                "qk_backend": self.qk_backend,
                "cpu_flat_gather": self.cpu_flat_gather,
                "cpu_bitmap_union": self.cpu_bitmap_union,
                "cpu_native_gather": self.cpu_native_gather,
                "direct_attention_layout": self.direct_attention_layout,
                "early_gpu_metadata": self.early_gpu_metadata,
                "gather_h2d_chunks": self.gather_h2d_chunks,
                "reuse_quant_metadata": self.reuse_quant_metadata,
                "fused_final_indices": self.fused_final_indices,
                "fused_quest_score": self.fused_quest_score,
                "skip_unused_host_views": self.skip_unused_host_views,
                "cpu_run_gather": self.cpu_run_gather,
                "previous_token_resident_cache": (
                    self.previous_token_resident_cache
                ),
                "gpu_compact_gqa_union": self.gpu_compact_gqa_union,
                "gpu_union_validate_cpu": self.gpu_union_validate_cpu,
                "post_selection_pipeline": self.post_selection_pipeline,
                "post_selection_groups_per_chunk": (
                    self.post_selection_groups_per_chunk
                    if self.post_selection_pipeline
                    else None
                ),
                "post_selection_pinned_index_bytes": (
                    0
                    if self._post_selection_host_membership is None
                    else self._post_selection_host_membership.numel()
                    * self._post_selection_host_membership.element_size()
                ),
                "post_selection_additional_gpu_buffer_bytes": 0,
                "selection_boundary": "top_p_membership_decision_complete",
                "batched_new_kv_d2h": self.batched_new_kv_d2h,
                "new_kv_d2h_granularity": (
                    self.new_kv_d2h_granularity
                    if self.batched_new_kv_d2h
                    else None
                ),
                "new_kv_gpu_staging_allocated_bytes": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (
                        self._new_kv_gpu_keys,
                        self._new_kv_gpu_values,
                    )
                    if tensor is not None
                ),
                "new_kv_pinned_host_staging_allocated_bytes": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (
                        self._new_kv_host_keys,
                        self._new_kv_host_values,
                    )
                    if tensor is not None
                ),
                "gpu_group_membership_allocated_bytes": (
                    0
                    if self._gpu_group_membership is None
                    else self._gpu_group_membership.numel()
                    * self._gpu_group_membership.element_size()
                ),
                "resident_cache_logical_bytes": sum(
                    2
                    * sum(position.numel() for position in positions)
                    * self.geometry.head_dim
                    * torch.tensor([], dtype=self.dtype).element_size()
                    for positions in self._resident_positions.values()
                ),
                "resident_cache_allocated_bytes": sum(
                    tensor.numel() * tensor.element_size()
                    for tensors in (self._resident_keys, self._resident_values)
                    for tensor in tensors.values()
                ),
                "resident_position_metadata_cpu_bytes": sum(
                    position.numel() * position.element_size()
                    for positions in self._resident_positions.values()
                    for position in positions
                ),
                "resident_added_pinned_cpu_bytes": 0,
                "resident_index_logical_peak_bytes": (
                    self._resident_index_logical_peak_bytes
                ),
                "resident_hit_gather_logical_peak_bytes": (
                    self._resident_hit_gather_logical_peak_bytes
                ),
                "skip_first_two_dense_layers": False,
                "layer_batched_selection": self.layer_batched_selection,
                "layer_batched_twilight_selection": self.layer_batched_selection,
                "twilight_gqa_groupwise_execution": self.gqa_groupwise_execution,
                "detailed_selection_profile": self.detailed_selection_profile,
                "twilight_gqa_group_semantics": (
                    "exact union of the original per-Query-head selected token sets"
                    if self.gqa_groupwise_execution
                    else None
                ),
            }
        )
        return snapshot
