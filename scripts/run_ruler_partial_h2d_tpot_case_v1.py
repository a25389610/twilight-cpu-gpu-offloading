#!/usr/bin/env python3
"""Run one true Partial-H2D TPOT case for the frozen RULER 2K request."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "headinfer"))
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
NUM_ENTRIES = 224
FIXED_TOKEN_ID = 1
SEED = 20260816


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--short-head-count", type=int, required=True)
    parser.add_argument("--full-layer-flat", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--recent-tokens", type=int, default=256)
    parser.add_argument("--quest-budget-fraction", type=float)
    parser.add_argument("--quest-block-size", type=int, default=16)
    parser.add_argument("--quest-selection-interval", type=int, default=1)
    parser.add_argument("--quest-layer-batched-selection", action="store_true")
    parser.add_argument("--quest-layer-flat-ragged", action="store_true")
    parser.add_argument("--sparse-gather-stabilized", action="store_true")
    parser.add_argument("--quest-per-head-varlen-reference", action="store_true")
    parser.add_argument("--capture-quest-selection-trace", action="store_true")
    parser.add_argument("--twilight-top-p", type=float)
    parser.add_argument("--twilight-candidate-token-budget", type=int, default=8192)
    parser.add_argument("--twilight-match-budget-fraction", type=float, default=0.05)
    parser.add_argument(
        "--twilight-budget-mode",
        choices=("matched", "dynamic"),
        default="matched",
    )
    parser.add_argument("--capture-twilight-budget-trace", action="store_true")
    parser.add_argument("--twilight-layer-flat-ragged", action="store_true")
    parser.add_argument("--twilight-gqa-group", action="store_true")
    parser.add_argument("--twilight-cpu-flat-gather", action="store_true")
    parser.add_argument("--twilight-cpu-bitmap-union", action="store_true")
    parser.add_argument("--twilight-cpu-native-gather", action="store_true")
    parser.add_argument("--twilight-direct-attention-layout", action="store_true")
    parser.add_argument("--twilight-layer-projection", action="store_true")
    parser.add_argument("--twilight-layer-rope", action="store_true")
    parser.add_argument("--twilight-early-gpu-metadata", action="store_true")
    parser.add_argument("--twilight-gather-h2d-chunks", type=int, default=0)
    parser.add_argument("--twilight-reuse-quant-metadata", action="store_true")
    parser.add_argument("--twilight-fused-final-indices", action="store_true")
    parser.add_argument("--twilight-fused-quest-score", action="store_true")
    parser.add_argument("--twilight-skip-unused-host-views", action="store_true")
    parser.add_argument("--twilight-cpu-run-gather", action="store_true")
    parser.add_argument(
        "--twilight-previous-token-resident-cache", action="store_true"
    )
    parser.add_argument(
        "--twilight-gpu-compact-gqa-union", action="store_true",
        help="Build the exact GQA union in a GPU bitmap and transfer that compact map.",
    )
    parser.add_argument(
        "--twilight-gpu-union-validate-cpu", action="store_true",
        help="Diagnostic-only exact comparison against the original per-Q CPU union.",
    )
    parser.add_argument(
        "--capture-twilight-resident-attention-trace",
        action="store_true",
        help="Diagnostic-only exact attention K/V hashes at D1, D2, and D32.",
    )
    parser.add_argument(
        "--twilight-resident-profile-each-token",
        action="store_true",
        help="Diagnostic-only per-token cache metrics; TPOT from this run is invalid.",
    )
    parser.add_argument("--host-memory-trace", action="store_true",
                        help="Diagnostic snapshots outside per-token latency timers; perturbs inter-token schedule.")
    parser.add_argument(
        "--twilight-qk-backend", choices=("pytorch", "triton", "triton_prepare"),
        default="pytorch",
        help="triton_prepare fuses K preparation; triton also changes QK reduction (experimental).",
    )
    parser.add_argument(
        "--twilight-detailed-selection-profile",
        action="store_true",
        help="Add fine-grained CUDA Events inside Twilight Selection diagnostics.",
    )
    parser.add_argument(
        "--twilight-per-head-varlen-reference", action="store_true"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-flat-h2d-reference-json", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Time one post-TPOT diagnostic token with Quest phase instrumentation.",
    )
    parser.add_argument(
        "--profile-breakdown-steps",
        type=int,
        default=1,
        help="Number of post-TPOT diagnostic tokens; these never enter TPOT.",
    )
    parser.add_argument(
        "--profile-exclusive-breakdown",
        action="store_true",
        help=(
            "Synchronize diagnostic phase boundaries and report a mutually "
            "exclusive 100%% work breakdown. This intentionally disables "
            "overlap and never enters formal TPOT."
        ),
    )
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "tokens": len(values),
        "total_seconds": sum(values),
        "mean_seconds_per_token": statistics.mean(values),
        "median_seconds_per_token": statistics.median(values),
        "p95_seconds_per_token": ordered[p95_index],
        "min_seconds_per_token": min(values),
        "max_seconds_per_token": max(values),
        "tokens_per_second": len(values) / sum(values),
    }


def prompt_sha256(prompt_ids: torch.Tensor) -> str:
    values = ",".join(str(int(value)) for value in prompt_ids.flatten().tolist())
    return hashlib.sha256(values.encode("ascii")).hexdigest()


def runtime_metadata() -> dict[str, Any]:
    return {
        "sys_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class ExclusiveBreakdownProfiler:
    """Serialize named diagnostic phases so their wall times are exclusive.

    This is deliberately a work decomposition, not a latency measurement.  CUDA
    synchronization at every boundary prevents selector/gather/transfer/
    attention work from being charged to two categories at once.
    """

    def __init__(self, cache: Any, model: Any) -> None:
        self.cache = cache
        self.model = model
        self.metrics: dict[str, float] = {}
        self._originals: dict[str, Any] = {}
        self._hook_handles: list[Any] = []
        self._module_starts: dict[int, float] = {}

    def _add(self, name: str, seconds: float) -> None:
        self.metrics[name] = self.metrics.get(name, 0.0) + seconds

    @staticmethod
    def _sync_all(cache: Any) -> None:
        del cache
        torch.cuda.synchronize()

    def reset(self) -> None:
        self.metrics = {}

    def install(self) -> None:
        cache = self.cache

        original_update = cache.update
        self._originals["update"] = original_update

        def update_wrapper(*args: Any, **kwargs: Any):
            self._sync_all(cache)
            started = time.perf_counter()
            result = original_update(*args, **kwargs)
            self._sync_all(cache)
            self._add("kv_cache_total_wall_seconds", time.perf_counter() - started)
            return result

        cache.update = update_wrapper

        if hasattr(cache, "update_layer_flat_ragged"):
            original_layer_flat_update = cache.update_layer_flat_ragged
            self._originals["update_layer_flat_ragged"] = original_layer_flat_update

            def layer_flat_update_wrapper(*args: Any, **kwargs: Any):
                self._sync_all(cache)
                started = time.perf_counter()
                result = original_layer_flat_update(*args, **kwargs)
                self._sync_all(cache)
                self._add(
                    "kv_cache_total_wall_seconds",
                    time.perf_counter() - started,
                )
                return result

            cache.update_layer_flat_ragged = layer_flat_update_wrapper

        if hasattr(cache, "update_layer_gqa_group_ragged"):
            original_gqa_group_update = cache.update_layer_gqa_group_ragged
            self._originals["update_layer_gqa_group_ragged"] = (
                original_gqa_group_update
            )

            def gqa_group_update_wrapper(*args: Any, **kwargs: Any):
                self._sync_all(cache)
                started = time.perf_counter()
                result = original_gqa_group_update(*args, **kwargs)
                self._sync_all(cache)
                self._add(
                    "kv_cache_total_wall_seconds",
                    time.perf_counter() - started,
                )
                return result

            cache.update_layer_gqa_group_ragged = gqa_group_update_wrapper

        if hasattr(cache, "_selected_positions"):
            original_selected = cache._selected_positions
            self._originals["_selected_positions"] = original_selected

            def selected_wrapper(*args: Any, **kwargs: Any):
                self._sync_all(cache)
                started = time.perf_counter()
                result = original_selected(*args, **kwargs)
                self._sync_all(cache)
                self._add("selection_wall_seconds", time.perf_counter() - started)
                return result

            cache._selected_positions = selected_wrapper

        if getattr(cache, "layer_batched_selection", False) and hasattr(
            cache, "prepare_layer_selection"
        ):
            original_prepare_layer = cache.prepare_layer_selection
            self._originals["prepare_layer_selection"] = original_prepare_layer

            def prepare_layer_wrapper(*args: Any, **kwargs: Any):
                self._sync_all(cache)
                started = time.perf_counter()
                result = original_prepare_layer(*args, **kwargs)
                self._sync_all(cache)
                elapsed = time.perf_counter() - started
                self._add("selection_wall_seconds", elapsed)
                # Sequential Quest performs Selection inside cache.update(), so
                # its time is already part of kv_cache_total.  Layer-batched
                # Selection runs immediately before update(); include the same
                # interval in kv_cache_total to preserve mutually exclusive
                # classification boundaries across both execution paths.
                self._add("kv_cache_total_wall_seconds", elapsed)
                return result

            cache.prepare_layer_selection = prepare_layer_wrapper

        if hasattr(cache, "_record_copy"):
            original_record_copy = cache._record_copy
            self._originals["_record_copy"] = original_record_copy

            def record_copy_wrapper(
                name: str,
                operation: Any,
                byte_count: int,
                *,
                copy_calls: int = 2,
            ) -> None:
                self._sync_all(cache)
                started = time.perf_counter()
                original_record_copy(
                    name, operation, byte_count, copy_calls=copy_calls
                )
                self._sync_all(cache)
                self._add(f"{name}_wall_seconds", time.perf_counter() - started)

            cache._record_copy = record_copy_wrapper

        if hasattr(cache, "_schedule_d2h"):
            original_schedule_d2h = cache._schedule_d2h
            self._originals["_schedule_d2h"] = original_schedule_d2h

            def schedule_d2h_wrapper(*args: Any, **kwargs: Any) -> None:
                self._sync_all(cache)
                started = time.perf_counter()
                original_schedule_d2h(*args, **kwargs)
                self._sync_all(cache)
                self._add("d2h_wall_seconds", time.perf_counter() - started)

            cache._schedule_d2h = schedule_d2h_wrapper

        if hasattr(cache, "_record_transfer"):
            original_record_transfer = cache._record_transfer
            self._originals["_record_transfer"] = original_record_transfer

            def record_transfer_wrapper(*args: Any, **kwargs: Any) -> None:
                name = kwargs.get("name")
                if name is None and args:
                    name = args[0]
                if name not in {"h2d", "d2h"}:
                    raise AssertionError(f"unexpected transfer phase: {name}")
                self._sync_all(cache)
                started = time.perf_counter()
                original_record_transfer(*args, **kwargs)
                self._sync_all(cache)
                self._add(f"{name}_wall_seconds", time.perf_counter() - started)

            cache._record_transfer = record_transfer_wrapper

        def register_module(module: Any, category: str) -> None:
            def pre_hook(current: Any, inputs: Any) -> None:
                del inputs
                torch.cuda.synchronize()
                self._module_starts[id(current)] = time.perf_counter()

            def post_hook(current: Any, inputs: Any, output: Any) -> None:
                del inputs, output
                torch.cuda.synchronize()
                started = self._module_starts.pop(id(current))
                self._add(category, time.perf_counter() - started)

            self._hook_handles.append(module.register_forward_pre_hook(pre_hook))
            self._hook_handles.append(module.register_forward_hook(post_hook))

        register_module(
            self.model.model.embed_tokens, "embedding_wall_seconds"
        )
        for layer in self.model.model.layers:
            for projection in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            ):
                register_module(
                    projection, "attention_projection_wall_seconds"
                )
            register_module(layer.mlp, "mlp_wall_seconds")
            register_module(
                layer.input_layernorm, "normalization_wall_seconds"
            )
            register_module(
                layer.post_attention_layernorm, "normalization_wall_seconds"
            )
        register_module(self.model.model.norm, "normalization_wall_seconds")
        register_module(self.model.lm_head, "lm_head_wall_seconds")

    def uninstall(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._module_starts = {}
        for name, original in self._originals.items():
            setattr(self.cache, name, original)
        self._originals = {}


def main() -> int:
    args = parse_args()
    if not 0 <= args.short_head_count <= NUM_ENTRIES:
        raise ValueError("short-head-count must be in [0, 224]")
    if args.decode_steps < 2:
        raise ValueError("decode-steps must be at least 2")
    if args.quest_selection_interval <= 0:
        raise ValueError("quest-selection-interval must be positive")
    if args.profile_breakdown_steps < 1:
        raise ValueError("profile-breakdown-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.quest_budget_fraction is not None and args.twilight_top_p is not None:
        raise ValueError("Quest and Twilight policies are mutually exclusive")
    if args.full_layer_flat and (
        args.short_head_count != 0
        or args.quest_budget_fraction is not None
        or args.twilight_top_p is not None
    ):
        raise ValueError("Full layer-flat execution requires the Full-only policy")
    if args.quest_layer_batched_selection and (
        args.quest_budget_fraction is None and args.twilight_top_p is None
    ):
        raise ValueError("layer-batched selection requires Quest or Twilight")
    if args.quest_layer_flat_ragged and args.quest_budget_fraction is None:
        raise ValueError("Quest layer-flat ragged execution requires Quest")
    if args.quest_layer_flat_ragged and not args.quest_layer_batched_selection:
        raise ValueError(
            "Quest layer-flat ragged execution requires layer-batched selection"
        )
    if args.sparse_gather_stabilized and not (
        args.quest_layer_flat_ragged or args.twilight_layer_flat_ragged
    ):
        raise ValueError(
            "stabilized sparse gather requires Quest/Twilight layer-flat execution"
        )
    if args.quest_per_head_varlen_reference and args.quest_budget_fraction is None:
        raise ValueError("Quest per-Head varlen reference requires Quest")
    if (
        args.quest_per_head_varlen_reference
        and not args.quest_layer_batched_selection
    ):
        raise ValueError(
            "Quest per-Head varlen reference requires layer-batched selection"
        )
    if args.quest_per_head_varlen_reference and args.quest_layer_flat_ragged:
        raise ValueError(
            "Quest per-Head varlen reference and layer-flat are exclusive"
        )
    if args.capture_quest_selection_trace and (
        args.quest_budget_fraction is None and args.twilight_top_p is None
    ):
        raise ValueError("selection trace requires Quest or Twilight")
    if args.capture_twilight_budget_trace and args.twilight_top_p is None:
        raise ValueError("Twilight budget trace requires Twilight")
    if args.twilight_layer_flat_ragged and args.twilight_top_p is None:
        raise ValueError("layer-flat ragged execution requires Twilight")
    if args.twilight_layer_flat_ragged and not args.quest_layer_batched_selection:
        raise ValueError(
            "layer-flat ragged execution requires layer-batched selection"
        )
    if args.twilight_gqa_group and args.twilight_top_p is None:
        raise ValueError("Twilight GQA group-wise execution requires Twilight")
    if args.twilight_qk_backend != "pytorch" and args.twilight_top_p is None:
        raise ValueError("fused QK backend requires Twilight")
    if args.twilight_gqa_group and not args.quest_layer_batched_selection:
        raise ValueError(
            "Twilight GQA group-wise execution requires layer-batched selection"
        )
    if args.twilight_gqa_group and args.twilight_layer_flat_ragged:
        raise ValueError(
            "Twilight GQA group-wise and per-Q flat execution are exclusive"
        )
    if args.twilight_previous_token_resident_cache and not args.twilight_gqa_group:
        raise ValueError("previous-token resident cache requires Twilight GQA")
    if args.twilight_gpu_compact_gqa_union and not args.twilight_gqa_group:
        raise ValueError("GPU compact union requires Twilight GQA")
    if (
        args.twilight_gpu_union_validate_cpu
        and not args.twilight_gpu_compact_gqa_union
    ):
        raise ValueError("GPU union validation requires GPU compact union")
    if (
        args.capture_twilight_resident_attention_trace
        and not args.twilight_gqa_group
    ):
        raise ValueError("attention trace requires Twilight GQA")
    if (
        args.twilight_resident_profile_each_token
        and not args.twilight_previous_token_resident_cache
    ):
        raise ValueError("resident per-token metrics require the resident prototype")
    if args.twilight_detailed_selection_profile and args.twilight_top_p is None:
        raise ValueError("detailed Twilight Selection profiling requires Twilight")
    if args.twilight_detailed_selection_profile and not (
        args.profile_breakdown or args.profile_exclusive_breakdown
    ):
        raise ValueError("detailed Twilight Selection profiling requires diagnostics")
    if args.twilight_per_head_varlen_reference and args.twilight_top_p is None:
        raise ValueError("per-Head varlen reference requires Twilight")
    if (
        args.twilight_per_head_varlen_reference
        and not args.quest_layer_batched_selection
    ):
        raise ValueError(
            "per-Head varlen reference requires layer-batched selection"
        )
    if args.twilight_per_head_varlen_reference and args.twilight_layer_flat_ragged:
        raise ValueError("per-Head varlen reference and layer-flat are exclusive")

    request = json.loads((args.request_dir / "request.json").read_text())
    path = json.loads((args.request_dir / "selected_path.json").read_text())
    selected = [int(entry) for entry in path["selected_short_heads"]]
    if path["status"] != "complete" or len(selected) != NUM_ENTRIES:
        raise AssertionError("Conditional Greedy path is incomplete")
    short_heads = selected[: args.short_head_count]
    if len(short_heads) != len(set(short_heads)):
        raise AssertionError("short-head prefix contains duplicates")
    prompt_payload = torch.load(
        args.request_dir / "prompt_ids.pt", map_location="cpu", weights_only=True
    )
    prompt_ids_cpu = prompt_payload["prompt_ids"].to(torch.long).unsqueeze(0)
    if int(prompt_ids_cpu.shape[1]) != int(request["prompt_token_count"]):
        raise AssertionError("prompt token count mismatch")
    if prompt_sha256(prompt_ids_cpu) != request["prompt_sha256"]:
        raise AssertionError("prompt hash mismatch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, use_fast=True, token=True, local_files_only=args.local_files_only
    )
    if not 0 <= FIXED_TOKEN_ID < tokenizer.vocab_size:
        raise AssertionError("fixed token ID is outside vocabulary")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        token=True,
        local_files_only=args.local_files_only,
    ).eval().to("cuda")
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise AssertionError("model weights must remain on CUDA")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = True

    import headinfer.mp as mp_module
    from headinfer.offload_ablation_cache import (
        GroupedPinnedSlabOffloadedCache,
        StaticHeadwiseShortWindowOffloadedCache,
    )
    from headinfer.quest_offload_cache import QuestTopKOffloadedCache
    from headinfer.twilight_offload_cache import TwilightMatchedBudgetOffloadedCache

    mp_module.mp_headinfer(model)
    if args.twilight_layer_projection or args.twilight_layer_rope:
        if args.twilight_top_p is None or not args.quest_layer_batched_selection:
            raise ValueError('layer projection requires layer-batched Twilight')
        if model.config.model_type != 'llama':
            raise ValueError('layer projection pilot supports Llama only')
        for layer in model.model.layers:
            attention = layer.self_attn
            projections = (attention.q_proj, attention.k_proj, attention.v_proj)
            if any(p.bias is not None for p in projections):
                raise ValueError('layer projection pilot requires bias-free projections')
            attention._twilight_full_projection_weights = tuple(p.weight.detach() for p in projections)
            attention._twilight_layer_projection = args.twilight_layer_projection
            attention._twilight_layer_rope = args.twilight_layer_rope
    capacity = (
        int(prompt_ids_cpu.shape[1])
        + args.decode_steps
        + args.profile_breakdown_steps
        + 4
    )
    common = {
        "max_cache_len": capacity,
        "transfer_group_size": 8 if args.full_layer_flat else 1,
        "writeback_mode": "delta",
        "activation_wait_mode": "event",
    }
    if args.twilight_top_p is not None:
        cache = TwilightMatchedBudgetOffloadedCache.from_llama_model(
            model,
            max_cache_len=capacity,
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
            block_size=args.quest_block_size,
            candidate_token_budget=args.twilight_candidate_token_budget,
            top_p=args.twilight_top_p,
            matched_budget_fraction=args.twilight_match_budget_fraction,
            budget_mode=args.twilight_budget_mode,
            layer_batched_selection=args.quest_layer_batched_selection,
            layer_flat_ragged_execution=args.twilight_layer_flat_ragged,
            gqa_groupwise_execution=args.twilight_gqa_group,
            detailed_selection_profile=args.twilight_detailed_selection_profile,
            qk_backend=args.twilight_qk_backend,
            cpu_flat_gather=args.twilight_cpu_flat_gather,
            cpu_bitmap_union=args.twilight_cpu_bitmap_union,
            cpu_native_gather=args.twilight_cpu_native_gather,
            direct_attention_layout=args.twilight_direct_attention_layout,
            early_gpu_metadata=args.twilight_early_gpu_metadata,
            gather_h2d_chunks=args.twilight_gather_h2d_chunks,
            reuse_quant_metadata=args.twilight_reuse_quant_metadata,
            fused_final_indices=args.twilight_fused_final_indices,
            fused_quest_score=args.twilight_fused_quest_score,
            skip_unused_host_views=args.twilight_skip_unused_host_views,
            cpu_run_gather=args.twilight_cpu_run_gather,
            previous_token_resident_cache=(
                args.twilight_previous_token_resident_cache
            ),
            gpu_compact_gqa_union=args.twilight_gpu_compact_gqa_union,
            gpu_union_validate_cpu=args.twilight_gpu_union_validate_cpu,
            sparse_gather_stabilized=args.sparse_gather_stabilized,
            per_head_varlen_attention_reference=(
                args.twilight_per_head_varlen_reference
            ),
        )
        if args.capture_quest_selection_trace:
            cache.enable_selection_trace()
        if args.capture_twilight_budget_trace:
            cache.enable_budget_trace()
        if args.capture_twilight_resident_attention_trace:
            cache.enable_resident_attention_trace()
        h2d_policy = (
            "twilight_int4_top_p_dynamic_gqa_group_union"
            if args.twilight_budget_mode == "dynamic" and args.twilight_gqa_group
            else "twilight_int4_top_p_dynamic"
            if args.twilight_budget_mode == "dynamic"
            else "twilight_int4_top_p_matched_quest_h2d"
        )
    elif args.quest_budget_fraction is not None:
        if not 0.0 < args.quest_budget_fraction <= 1.0:
            raise ValueError("quest-budget-fraction must be in (0, 1]")
        cache = QuestTopKOffloadedCache.from_llama_model(
            model,
            max_cache_len=capacity,
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
            block_size=args.quest_block_size,
            budget_fraction=args.quest_budget_fraction,
            selection_interval=args.quest_selection_interval,
            layer_batched_selection=args.quest_layer_batched_selection,
            layer_flat_ragged_execution=args.quest_layer_flat_ragged,
            sparse_gather_stabilized=args.sparse_gather_stabilized,
            per_head_varlen_attention_reference=(
                args.quest_per_head_varlen_reference
            ),
        )
        if args.capture_quest_selection_trace:
            cache.enable_selection_trace()
        h2d_policy = "quest_per_query_head_topk_blocks"
    elif short_heads:
        cache = StaticHeadwiseShortWindowOffloadedCache.from_llama_model(
            model,
            short_head_entries=short_heads,
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
            **common,
        )
        h2d_policy = "partial_sink_recent"
    else:
        cache = GroupedPinnedSlabOffloadedCache.from_llama_model(
            model,
            layer_full_attention_execution=args.full_layer_flat,
            **common,
        )
        h2d_policy = (
            "full_history_layer_flat_gqa"
            if args.full_layer_flat
            else "full_history"
        )

    host_memory_trace = []

    def capture_host_memory(stage):
        if not args.host_memory_trace:
            return
        import resource
        status = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.split(':')[0] in {'VmRSS', 'VmHWM', 'VmSwap'}:
                key, value = line.split(':')
                status[key + '_bytes'] = int(value.split()[0]) * 1024
        usage = resource.getrusage(resource.RUSAGE_SELF)
        stats_fn = getattr(torch.cuda, 'host_memory_stats', None)
        host_memory_trace.append(dict(stage=stage, **status,
                                      minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt,
                                      host_allocator=stats_fn() if stats_fn else None,
                                      cache_state=cache.state_snapshot()))

    capture_host_memory('cache_allocated')
    prompt_ids = prompt_ids_cpu.to("cuda")
    fixed_id = torch.tensor([[FIXED_TOKEN_ID]], dtype=torch.long, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=prompt_ids,
            past_key_values=cache,
            use_cache=True,
            num_logits_to_keep=1,
            return_dict=True,
        )
    cache.synchronize()
    if args.full_layer_flat:
        cache.prepare_layer_full_layout()
        cache.synchronize()
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_started
    logits = {"P4": output.logits[:, -1, :].float().cpu()}
    capture_host_memory('P4')
    if not torch.isfinite(logits["P4"]).all():
        raise AssertionError("non-finite P4 logits")
    if int(cache.get_seq_length()) != int(prompt_ids_cpu.shape[1]):
        raise AssertionError("prefill cache length mismatch")

    latencies: list[float] = []
    resident_per_token_metrics: list[dict[str, Any]] = []
    total_started: float | None = None
    total_decode_seconds: float | None = None
    with torch.inference_mode():
        for decode_index in range(1, args.decode_steps + 1):
            if args.twilight_resident_profile_each_token:
                cache.enable_metrics()
                cache.reset_metrics()
            torch.cuda.synchronize()
            if decode_index == 1:
                total_started = time.perf_counter()
            started = time.perf_counter()
            output = model(
                input_ids=fixed_id,
                past_key_values=cache,
                use_cache=True,
                num_logits_to_keep=1,
                return_dict=True,
            )
            cache.synchronize()
            torch.cuda.synchronize()
            finished = time.perf_counter()
            latencies.append(finished - started)
            if args.twilight_resident_profile_each_token:
                resident_per_token_metrics.append(
                    {"decode_step": decode_index, **cache.resolve_metrics()}
                )
            if decode_index == args.decode_steps:
                if total_started is None:
                    raise AssertionError("decode timer did not start")
                total_decode_seconds = finished - total_started
            if decode_index in {1, 2, 32, 128, args.decode_steps}:
                checkpoint = output.logits[:, -1, :].float().cpu()
                if not torch.isfinite(checkpoint).all():
                    raise AssertionError(f"non-finite D{decode_index} logits")
                logits[f"D{decode_index}"] = checkpoint
                capture_host_memory(f'D{decode_index}')
    if total_decode_seconds is None:
        raise AssertionError("decode timing incomplete")

    attention_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    exclusive_attention_wall_seconds = 0.0
    original_flash_attention = mp_module._flash_attention_forward
    original_flash_attention_varlen = mp_module._flash_attention_varlen_forward
    exclusive_profiler = (
        ExclusiveBreakdownProfiler(cache, model)
        if args.profile_exclusive_breakdown
        else None
    )

    def timed_flash_attention(*profile_args: Any, **profile_kwargs: Any):
        nonlocal exclusive_attention_wall_seconds
        if exclusive_profiler is not None:
            exclusive_profiler._sync_all(cache)
            wall_started = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        result = original_flash_attention(*profile_args, **profile_kwargs)
        end.record(torch.cuda.current_stream())
        attention_events.append((start, end))
        if exclusive_profiler is not None:
            exclusive_profiler._sync_all(cache)
            exclusive_attention_wall_seconds += time.perf_counter() - wall_started
        return result

    def timed_flash_attention_varlen(*profile_args: Any, **profile_kwargs: Any):
        nonlocal exclusive_attention_wall_seconds
        if exclusive_profiler is not None:
            exclusive_profiler._sync_all(cache)
            wall_started = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        result = original_flash_attention_varlen(*profile_args, **profile_kwargs)
        end.record(torch.cuda.current_stream())
        attention_events.append((start, end))
        if exclusive_profiler is not None:
            exclusive_profiler._sync_all(cache)
            exclusive_attention_wall_seconds += time.perf_counter() - wall_started
        return result

    sparse_policy = (
        args.quest_budget_fraction is not None or args.twilight_top_p is not None
    )
    if args.profile_breakdown or args.profile_exclusive_breakdown:
        mp_module._flash_attention_forward = timed_flash_attention
        mp_module._flash_attention_varlen_forward = timed_flash_attention_varlen
    diagnostic_breakdown_tokens: list[dict[str, float]] = []
    exclusive_breakdown_tokens: list[dict[str, float]] = []
    diagnostic_selected_positions_sha256: list[str] = []
    diagnostic_group_positions_sha256: list[str] = []
    diagnostic_logits_sha256: list[str] = []
    try:
        if exclusive_profiler is not None:
            exclusive_profiler.install()
        diagnostic_steps = (
            args.profile_breakdown_steps
            if args.profile_breakdown or args.profile_exclusive_breakdown
            else 0
        )
        for _ in range(diagnostic_steps):
            attention_events.clear()
            exclusive_attention_wall_seconds = 0.0
            if exclusive_profiler is not None:
                exclusive_profiler.reset()
            cache.enable_metrics()
            cache.reset_metrics()
            diagnostic_history_tokens = int(cache.get_seq_length())
            torch.cuda.synchronize()
            diagnostic_started = time.perf_counter()
            with torch.inference_mode():
                output = model(
                    input_ids=fixed_id,
                    past_key_values=cache,
                    use_cache=True,
                    num_logits_to_keep=1,
                    return_dict=True,
                )
            cache.synchronize()
            torch.cuda.synchronize()
            diagnostic_wall_seconds = time.perf_counter() - diagnostic_started
            if args.twilight_top_p is not None:
                diagnostic_selected_positions_sha256.append(
                    cache.selected_positions_sha256()
                )
                diagnostic_group_positions_sha256.append(
                    cache.group_positions_sha256()
                )
            diagnostic_logits = output.logits[:, -1, :].float().cpu().contiguous()
            diagnostic_logits_sha256.append(
                hashlib.sha256(diagnostic_logits.numpy().tobytes()).hexdigest()
            )
            token_metrics = cache.resolve_metrics()
            # System-level KV percentage is defined by actual transferred bytes
            # relative to Full Attention at the same history length.  This
            # intentionally includes any GQA expansion or repeated packing in
            # the sparse implementation; per-Query-Head visibility is only an
            # auxiliary algorithm statistic.
            full_h2d_reference_bytes = (
                2
                * NUM_ENTRIES
                * diagnostic_history_tokens
                * int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
                * next(model.parameters()).element_size()
            )
            token_metrics["diagnostic_history_tokens"] = diagnostic_history_tokens
            token_metrics["full_h2d_reference_bytes"] = full_h2d_reference_bytes
            token_metrics["formula_h2d_fraction_diagnostic_only"] = (
                float(token_metrics.get("h2d_bytes", 0.0))
                / full_h2d_reference_bytes
            )
            if attention_events:
                token_metrics["attention_cuda_seconds"] = sum(
                    start.elapsed_time(end) / 1000.0
                    for start, end in attention_events
                )
                token_metrics["attention_event_intervals"] = len(attention_events)
            token_metrics["diagnostic_token_wall_seconds"] = diagnostic_wall_seconds
            diagnostic_breakdown_tokens.append(token_metrics)
            if exclusive_profiler is not None:
                exclusive = dict(exclusive_profiler.metrics)
                exclusive["attention_wall_seconds"] = (
                    exclusive_attention_wall_seconds
                )
                exclusive["cpu_gather_wall_seconds"] = token_metrics.get(
                    "host_gather_pack_wall_seconds", 0.0
                )
                exclusive["cpu_union_wall_seconds"] = token_metrics.get(
                    "twilight_group_union_prepare_wall_seconds", 0.0
                )
                kv_total = exclusive.get("kv_cache_total_wall_seconds", 0.0)
                named_kv = sum(
                    exclusive.get(name, 0.0)
                    for name in (
                        "selection_wall_seconds",
                        "cpu_gather_wall_seconds",
                        "cpu_union_wall_seconds",
                        "h2d_wall_seconds",
                        "d2h_wall_seconds",
                    )
                )
                exclusive["kv_control_and_pack_wall_seconds"] = max(
                    0.0, kv_total - named_kv
                )
                named_model = sum(
                    exclusive.get(name, 0.0)
                    for name in (
                        "attention_wall_seconds",
                        "attention_projection_wall_seconds",
                        "mlp_wall_seconds",
                        "normalization_wall_seconds",
                        "embedding_wall_seconds",
                        "lm_head_wall_seconds",
                    )
                )
                exclusive[
                    "rope_residual_tensor_and_runtime_wall_seconds"
                ] = max(
                    0.0, diagnostic_wall_seconds - kv_total - named_model
                )
                exclusive["exclusive_accounted_wall_seconds"] = sum(
                    exclusive.get(name, 0.0)
                    for name in (
                        "selection_wall_seconds",
                        "cpu_gather_wall_seconds",
                        "cpu_union_wall_seconds",
                        "h2d_wall_seconds",
                        "d2h_wall_seconds",
                        "kv_control_and_pack_wall_seconds",
                        "attention_wall_seconds",
                        "attention_projection_wall_seconds",
                        "mlp_wall_seconds",
                        "normalization_wall_seconds",
                        "embedding_wall_seconds",
                        "lm_head_wall_seconds",
                        "rope_residual_tensor_and_runtime_wall_seconds",
                    )
                )
                exclusive["diagnostic_token_wall_seconds"] = diagnostic_wall_seconds
                exclusive_breakdown_tokens.append(exclusive)
    finally:
        if exclusive_profiler is not None:
            exclusive_profiler.uninstall()
        mp_module._flash_attention_forward = original_flash_attention
        mp_module._flash_attention_varlen_forward = original_flash_attention_varlen
    transfer_metrics = (
        diagnostic_breakdown_tokens[0] if diagnostic_breakdown_tokens else {}
    )
    physical_metrics = {
        "per_q_logical_visibility_percent": None,
        "group_union_kv_percent": None,
        "gqa_duplication_factor": None,
        "h2d_mib_per_token": None,
        "physical_kv_transfer_percent": None,
        "actual_full_flat_h2d_reference_bytes": None,
    }
    if transfer_metrics:
        history_tokens = int(transfer_metrics["diagnostic_history_tokens"])
        h2d_bytes = float(transfer_metrics.get("h2d_bytes", 0.0))
        physical_metrics["h2d_mib_per_token"] = h2d_bytes / (1024.0**2)
        if sparse_policy:
            selected_total = float(
                transfer_metrics.get("selected_history_tokens_total", 0.0)
            )
            physical_metrics["per_q_logical_visibility_percent"] = (
                100.0 * selected_total / (model.config.num_hidden_layers * model.config.num_attention_heads * history_tokens)
            )
            group_union_total = float(
                transfer_metrics.get("group_union_history_tokens_total", 0.0)
            )
            if group_union_total:
                physical_metrics["group_union_kv_percent"] = (
                    100.0
                    * group_union_total
                    / (
                        model.config.num_hidden_layers
                        * model.config.num_key_value_heads
                        * history_tokens
                    )
                )
                physical_metrics["gqa_duplication_factor"] = (
                    selected_total / group_union_total
                )
        else:
            physical_metrics["per_q_logical_visibility_percent"] = 100.0

        full_flat_bytes = None
        if args.full_layer_flat:
            full_flat_bytes = h2d_bytes
            formula_bytes = float(transfer_metrics["full_h2d_reference_bytes"])
            if full_flat_bytes != formula_bytes:
                raise AssertionError(
                    "Full-flat actual H2D payload differs from the shape assertion"
                )
        elif args.full_flat_h2d_reference_json is not None:
            reference = json.loads(args.full_flat_h2d_reference_json.read_text())
            if reference.get("status") != "ok" or not reference.get(
                "full_layer_flat", False
            ):
                raise ValueError("H2D denominator must be a valid Full-flat artifact")
            reference_metrics = reference[
                "diagnostic_transfer_metrics_one_post_timing_token"
            ]
            if int(reference_metrics["diagnostic_history_tokens"]) != history_tokens:
                raise ValueError(
                    "Full-flat H2D denominator uses a different history length"
                )
            full_flat_bytes = float(reference_metrics["h2d_bytes"])
        if full_flat_bytes is not None:
            physical_metrics["actual_full_flat_h2d_reference_bytes"] = (
                full_flat_bytes
            )
            physical_metrics["physical_kv_transfer_percent"] = (
                100.0 * h2d_bytes / full_flat_bytes
            )

    per_token_path = args.output.with_suffix(".per_token.csv")
    with per_token_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token_index", "latency_seconds"])
        writer.writerows(enumerate(latencies, 1))
    logits_path = args.output.with_suffix(".logits.pt")
    torch.save({"checkpoints": logits}, logits_path)
    result = {
        "status": "ok",
        "classification": (
            "Twilight-inspired Partial-H2D fixed-token TPOT case"
            if args.twilight_top_p is not None
            else "true Quest Top-K Partial-H2D fixed-token TPOT case"
            if args.quest_budget_fraction is not None
            else "true Partial-H2D fixed-token TPOT case"
        ),
        "request_id": request["request_id"],
        "prompt_token_count": int(prompt_ids_cpu.shape[1]),
        "prompt_sha256": request["prompt_sha256"],
        "model": MODEL,
        "model_parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "dtype": "bfloat16",
        "attention": "flash_attention_2",
        "batch_size": 1,
        "decode_policy": "fixed",
        "fixed_token_id": FIXED_TOKEN_ID,
        "decode_steps": args.decode_steps,
        "short_head_count": len(short_heads),
        "full_head_count": NUM_ENTRIES - len(short_heads),
        "short_heads": short_heads,
        "h2d_policy": h2d_policy,
        "full_layer_flat": args.full_layer_flat,
        "sink_tokens": (
            args.sink_tokens
            if short_heads or sparse_policy
            else None
        ),
        "recent_tokens": (
            args.recent_tokens
            if short_heads or sparse_policy
            else None
        ),
        "quest_block_size": (
            args.quest_block_size if sparse_policy else None
        ),
        "quest_budget_fraction": args.quest_budget_fraction,
        "quest_selection_interval": args.quest_selection_interval,
        "quest_layer_batched_selection": args.quest_layer_batched_selection,
        "quest_layer_flat_ragged": (
            args.quest_layer_flat_ragged
            if args.quest_budget_fraction is not None
            else None
        ),
        "sparse_gather_stabilized": (
            args.sparse_gather_stabilized if sparse_policy else None
        ),
        "quest_per_head_varlen_reference": (
            args.quest_per_head_varlen_reference
            if args.quest_budget_fraction is not None
            else None
        ),
        "twilight_top_p": args.twilight_top_p,
        "twilight_qk_backend": args.twilight_qk_backend,
        "twilight_candidate_token_budget": (
            args.twilight_candidate_token_budget
            if args.twilight_top_p is not None
            else None
        ),
        "twilight_match_budget_fraction": (
            args.twilight_match_budget_fraction
            if args.twilight_top_p is not None
            else None
        ),
        "twilight_budget_mode": (
            args.twilight_budget_mode if args.twilight_top_p is not None else None
        ),
        "twilight_layer_flat_ragged": (
            args.twilight_layer_flat_ragged
            if args.twilight_top_p is not None
            else None
        ),
        "twilight_gqa_group": (
            args.twilight_gqa_group
            if args.twilight_top_p is not None
            else None
        ),
        "twilight_previous_token_resident_cache": (
            args.twilight_previous_token_resident_cache
            if args.twilight_top_p is not None
            else None
        ),
        "twilight_gpu_compact_gqa_union": (
            args.twilight_gpu_compact_gqa_union
            if args.twilight_top_p is not None
            else None
        ),
        "twilight_gpu_union_validate_cpu": (
            args.twilight_gpu_union_validate_cpu
            if args.twilight_top_p is not None
            else None
        ),
        "resident_diagnostic_only": bool(
            args.capture_twilight_resident_attention_trace
            or args.twilight_resident_profile_each_token
        ),
        "resident_per_token_metrics": resident_per_token_metrics,
        "resident_reuse_trace": (
            cache.resident_reuse_trace()
            if args.twilight_previous_token_resident_cache
            else None
        ),
        "resident_attention_trace": (
            cache.resident_attention_trace()
            if args.capture_twilight_resident_attention_trace
            else None
        ),
        "twilight_detailed_selection_profile": (
            args.twilight_detailed_selection_profile
            if args.twilight_top_p is not None
            else None
        ),
        "twilight_selected_positions_sha256": (
            cache.selected_positions_sha256()
            if args.twilight_top_p is not None
            else None
        ),
        "twilight_group_positions_sha256": (
            cache.group_positions_sha256()
            if args.twilight_top_p is not None
            else None
        ),
        "diagnostic_selected_positions_sha256": (
            diagnostic_selected_positions_sha256
            if args.twilight_top_p is not None
            else None
        ),
        "diagnostic_group_positions_sha256": (
            diagnostic_group_positions_sha256
            if args.twilight_top_p is not None
            else None
        ),
        "diagnostic_logits_sha256": diagnostic_logits_sha256,
        "twilight_per_head_varlen_reference": (
            args.twilight_per_head_varlen_reference
            if args.twilight_top_p is not None
            else None
        ),
        "prefill_seconds": prefill_seconds,
        "twilight_layer_projection": args.twilight_layer_projection,
        "twilight_layer_rope": args.twilight_layer_rope,
        "host_memory_trace": host_memory_trace,
        "host_memory_trace_enabled": args.host_memory_trace,
        "cpu_threads": torch.get_num_threads(),
        "D1": summarize(latencies[:1]),
        "D2_D128_tpot": summarize(latencies[1:]),
        "D1_D128_all": summarize(latencies),
        "total_decode_time_s": total_decode_seconds,
        "overall_decode_latency_ms_per_token": total_decode_seconds * 1000 / args.decode_steps,
        "overall_decode_throughput_tok_s": args.decode_steps / total_decode_seconds,
        "primary_tpot_definition": (
            "mean synchronized wall time per output token over "
            f"D2-D{args.decode_steps}; D1 excluded as warm-up"
        ),
        "primary_tpot_token_range": f"D2-D{args.decode_steps}",
        "prefill_and_model_load_excluded_from_tpot": True,
        "diagnostic_transfer_metrics_one_post_timing_token": transfer_metrics,
        "physical_kv_transfer_metrics": physical_metrics,
        "diagnostic_breakdown_tokens": diagnostic_breakdown_tokens,
        "exclusive_breakdown_tokens": exclusive_breakdown_tokens,
        "cache_state": cache.state_snapshot(),
        "quest_selection_trace": (
            cache.selection_trace()
            if args.capture_quest_selection_trace
            else None
        ),
        "twilight_budget_trace": (
            cache.budget_trace()
            if args.capture_twilight_budget_trace
            else None
        ),
        "twilight_group_union_trace": (
            cache.group_union_trace()
            if args.capture_quest_selection_trace
            and hasattr(cache, "group_union_trace")
            else None
        ),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "final_gpu_allocated_bytes": torch.cuda.memory_allocated(),
        "final_gpu_reserved_bytes": torch.cuda.memory_reserved(),
        "checkpoint_logits_sha256": {
            name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            for name, value in logits.items()
        },
        "final_logits_sha256": hashlib.sha256(
            logits[f"D{args.decode_steps}"].contiguous().numpy().tobytes()
        ).hexdigest(),
        "per_token_csv": str(per_token_path),
        "logits_file": str(logits_path),
        "runtime": runtime_metadata(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": "ok",
        "short_head_count": len(short_heads),
        "tpot_ms": result["D2_D128_tpot"]["mean_seconds_per_token"] * 1000,
        "h2d_bytes": transfer_metrics.get("h2d_bytes"),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
