"""Opt-in GPU bitmap mapping for the previous-token resident KV cache.

The two short scans keep selection membership and hit destinations on device.
Only the compact miss positions need to return to the CPU KV store.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["C_STRIDE", "P_STRIDE", "COUNT_STRIDE", "LENGTH"])
def _block_counts(C, P, CC, PC, MC,
                  C_STRIDE, P_STRIDE, COUNT_STRIDE, LENGTH,
                  B: tl.constexpr):
    g = tl.program_id(0)
    b = tl.program_id(1)
    x = b * B + tl.arange(0, B)
    current = tl.load(C + g * C_STRIDE + x, x < LENGTH, other=0).to(tl.int32)
    previous = tl.load(P + g * P_STRIDE + x, x < P_STRIDE, other=0).to(tl.int32)
    tl.store(CC + g * COUNT_STRIDE + b, tl.sum(current, 0))
    tl.store(PC + g * COUNT_STRIDE + b, tl.sum(previous, 0))
    tl.store(MC + g * COUNT_STRIDE + b, tl.sum(current * (1 - previous), 0))


@triton.jit(do_not_specialize=["C_STRIDE", "P_STRIDE", "COUNT_STRIDE", "LENGTH"])
def _map_blocks(C, P, CC, PC, MC, SRC, MISS_POS, MISS_DST, CURR_POS,
                CL, PL, ML,
                C_STRIDE, P_STRIDE, COUNT_STRIDE, LENGTH,
                B: tl.constexpr, NB: tl.constexpr):
    g = tl.program_id(0)
    b = tl.program_id(1)
    blocks = tl.arange(0, NB)
    current_before = tl.sum(tl.load(CC + g * COUNT_STRIDE + blocks, blocks < b, other=0), 0)
    previous_before = tl.sum(tl.load(PC + g * COUNT_STRIDE + blocks, blocks < b, other=0), 0)
    miss_before = tl.sum(tl.load(MC + g * COUNT_STRIDE + blocks, blocks < b, other=0), 0)
    x = b * B + tl.arange(0, B)
    current = tl.load(C + g * C_STRIDE + x, x < LENGTH, other=0).to(tl.int32)
    previous = tl.load(P + g * P_STRIDE + x, x < P_STRIDE, other=0).to(tl.int32)
    current_row = current_before + tl.cumsum(current) - 1
    previous_row = previous_before + tl.cumsum(previous) - 1
    miss = (current != 0) & (previous == 0) & (x < LENGTH)
    miss_row = miss_before + tl.cumsum(miss.to(tl.int32)) - 1
    tl.store(SRC + g * LENGTH + current_row,
             tl.where(previous != 0, previous_row, -1),
             (current != 0) & (x < LENGTH))
    tl.store(CURR_POS + g * LENGTH + current_row,
             x, (current != 0) & (x < LENGTH))
    tl.store(MISS_POS + g * LENGTH + miss_row, x, miss)
    tl.store(MISS_DST + g * LENGTH + miss_row, current_row, miss)
    if b == COUNT_STRIDE - 1:
        tl.store(CL + g, current_before + tl.sum(current, 0))
        tl.store(PL + g, previous_before + tl.sum(previous, 0))
        tl.store(ML + g, miss_before + tl.sum(miss.to(tl.int32), 0))


@triton.jit(do_not_specialize=["LENGTH"])
def _copy_hits(PK, PV, DK, DV, SRC, CL, PL,
               LENGTH, DIM: tl.constexpr, ROWS: tl.constexpr):
    g = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    groups = tl.arange(0, 8)
    previous_base = tl.sum(tl.load(PL + groups, groups < g, other=0), 0) + g
    current_base = tl.sum(tl.load(CL + groups, groups < g, other=0), 0) + g
    count = tl.load(CL + g)
    source = tl.load(SRC + g * LENGTH + row, row < count, other=-1)
    dims = tl.arange(0, DIM)
    mask = (row < count) & (source >= 0)
    k = tl.load(PK + (previous_base + source[:, None]) * DIM + dims[None, :],
                mask[:, None], other=0)
    v = tl.load(PV + (previous_base + source[:, None]) * DIM + dims[None, :],
                mask[:, None], other=0)
    tl.store(DK + (current_base + row[:, None]) * DIM + dims[None, :],
             k, mask[:, None])
    tl.store(DV + (current_base + row[:, None]) * DIM + dims[None, :],
             v, mask[:, None])


@triton.jit(do_not_specialize=["LENGTH"])
def _scatter_misses(HK, HV, DK, DV, MISS_DST, CL, ML,
                    LENGTH, DIM: tl.constexpr, ROWS: tl.constexpr):
    g = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    groups = tl.arange(0, 8)
    miss_base = tl.sum(tl.load(ML + groups, groups < g, other=0), 0)
    current_base = tl.sum(tl.load(CL + groups, groups < g, other=0), 0) + g
    count = tl.load(ML + g)
    dst = tl.load(MISS_DST + g * LENGTH + row, row < count, other=0)
    dims = tl.arange(0, DIM)
    mask = row < count
    k = tl.load(HK + (miss_base + row[:, None]) * DIM + dims[None, :],
                mask[:, None], other=0)
    v = tl.load(HV + (miss_base + row[:, None]) * DIM + dims[None, :],
                mask[:, None], other=0)
    tl.store(DK + (current_base + dst[:, None]) * DIM + dims[None, :],
             k, mask[:, None])
    tl.store(DV + (current_base + dst[:, None]) * DIM + dims[None, :],
             v, mask[:, None])


def map_membership(current: torch.Tensor, previous: torch.Tensor):
    if current.ndim != 2 or current.shape[0] != 8 or current.dtype != torch.bool:
        raise ValueError("GPU resident mapping expects eight bool GQA rows")
    length = current.shape[1]
    block = 512
    nblocks = triton.cdiv(length, block)
    shape = (8, nblocks)
    counts = [torch.empty(shape, dtype=torch.int32, device=current.device)
              for _ in range(3)]
    sources = torch.empty((8, length), dtype=torch.int32, device=current.device)
    miss_positions = torch.empty_like(sources)
    miss_destinations = torch.empty_like(sources)
    current_positions = torch.empty_like(sources)
    lengths = [torch.empty(8, dtype=torch.int32, device=current.device)
               for _ in range(3)]
    _block_counts[(8, nblocks)](current, previous, *counts,
                                current.stride(0), previous.stride(0),
                                nblocks, length, block)
    _map_blocks[(8, nblocks)](current, previous, *counts,
                              sources, miss_positions, miss_destinations,
                              current_positions,
                              *lengths, current.stride(0), previous.stride(0),
                              nblocks, length, block, triton.next_power_of_2(nblocks))
    return sources, miss_positions, miss_destinations, *lengths, current_positions


def copy_hits(previous_key, previous_value, target_key, target_value,
              sources, current_lengths, previous_lengths, max_rows):
    length = sources.shape[1]
    _copy_hits[(8, triton.cdiv(max_rows, 64))](
        previous_key, previous_value, target_key, target_value,
        sources, current_lengths, previous_lengths,
        length, previous_key.shape[1], 64, num_warps=8)


def scatter_misses(history_key, history_value, target_key, target_value,
                   miss_destinations, current_lengths, miss_lengths, max_rows):
    length = miss_destinations.shape[1]
    _scatter_misses[(8, triton.cdiv(max_rows, 64))](
        history_key, history_value, target_key, target_value,
        miss_destinations, current_lengths, miss_lengths,
        length, history_key.shape[1], 64, num_warps=8)
