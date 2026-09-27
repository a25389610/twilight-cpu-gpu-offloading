"""Fused exact resident K/V hit copy into the attention layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_hits(PREV_K, PREV_V, ATTN_K, ATTN_V, SOURCES, DESTINATIONS,
               N, D: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    live = rows < N
    source = tl.load(SOURCES + rows, live, 0)
    destination = tl.load(DESTINATIONS + rows, live, 0)
    offsets = source[:, None] * D + cols[None, :]
    target = destination[:, None] * D + cols[None, :]
    mask = live[:, None]
    k = tl.load(PREV_K + offsets, mask, 0)
    v = tl.load(PREV_V + offsets, mask, 0)
    tl.store(ATTN_K + target, k, mask)
    tl.store(ATTN_V + target, v, mask)


def copy_resident_hits(previous_key: torch.Tensor, previous_value: torch.Tensor,
                       attention_key: torch.Tensor, attention_value: torch.Tensor,
                       sources: torch.Tensor, destinations: torch.Tensor) -> None:
    count = sources.numel()
    if count == 0:
        return
    dim = previous_key.shape[1]
    if dim != attention_key.shape[1] or not triton.next_power_of_2(dim) == dim:
        raise ValueError("unsupported resident head dimension")
    if any(not t.is_cuda or not t.is_contiguous() for t in
           (previous_key, previous_value, attention_key, attention_value,
            sources, destinations)):
        raise ValueError("resident hit copy needs contiguous CUDA tensors")
    _copy_hits[(triton.cdiv(count, 16),)](
        previous_key, previous_value, attention_key, attention_value,
        sources, destinations, count, dim, 16,
    )


@triton.jit
def _snapshot_union(ATTN_K, ATTN_V, RESIDENT_K, RESIDENT_V, DESTINATIONS,
                    N, D: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    live = rows < N
    source = tl.load(DESTINATIONS + rows, live, 0)
    input_offsets = source[:, None] * D + cols[None, :]
    output_offsets = rows[:, None] * D + cols[None, :]
    mask = live[:, None]
    k = tl.load(ATTN_K + input_offsets, mask, 0)
    v = tl.load(ATTN_V + input_offsets, mask, 0)
    tl.store(RESIDENT_K + output_offsets, k, mask)
    tl.store(RESIDENT_V + output_offsets, v, mask)


def snapshot_resident_union(attention_key: torch.Tensor,
                            attention_value: torch.Tensor,
                            resident_key: torch.Tensor,
                            resident_value: torch.Tensor,
                            destinations: torch.Tensor, count: int) -> None:
    if count == 0:
        return
    dim = resident_key.shape[1]
    if dim != attention_key.shape[1] or triton.next_power_of_2(dim) != dim:
        raise ValueError("unsupported resident head dimension")
    if any(not t.is_cuda or not t.is_contiguous() for t in
           (attention_key, attention_value, resident_key, resident_value,
            destinations)):
        raise ValueError("resident snapshot needs contiguous CUDA tensors")
    _snapshot_union[(triton.cdiv(count, 16),)](
        attention_key, attention_value, resident_key, resident_value,
        destinations, count, dim, 16,
    )
