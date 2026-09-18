"""Local Quest score fusion. Not the official Twilight kernel.

Preserves the torch 2.7 CUDA FP32 contiguous dim=128 sum order for >=16
output rows: each lane accumulates d,d+32,d+64,d+96, then shfl-down
offsets 1,2,4,8,16. No FMA, TF32, changed budgets, or approximate score.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _score(Q, LO, HI, OUT, P, R: tl.constexpr, BLOCK_P: tl.constexpr):
    row = tl.program_id(0)
    group = row // R
    page = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    lane = tl.arange(0, 32)
    accum = tl.full((BLOCK_P, 32), 0, tl.float32)
    for chunk in tl.static_range(4):
        d = lane + chunk * 32
        q = tl.load(Q + row * 128 + d)
        offset = (group * P + page[:, None]) * 128 + d[None, :]
        lo = tl.load(LO + offset, page[:, None] < P, 0).to(tl.float32)
        hi = tl.load(HI + offset, page[:, None] < P, 0).to(tl.float32)
        product = tl.where(q[None, :] > 0, hi, lo) * q[None, :]
        accum = accum + product
    for shift in tl.static_range(5):
        other_lane = (lane + (1 << shift)) % 32
        other = tl.gather(accum, tl.broadcast_to(other_lane[None, :], (BLOCK_P, 32)), 1)
        accum = accum + other
    score = tl.sum(tl.where(lane[None, :] == 0, accum, 0), 1)
    tl.store(OUT + row * P + page, score, page < P)


def fused_quest_score(q, page_min, page_max):
    g, r, d = q.shape
    p = page_min.shape[1]
    if d != 128 or g*r*p < 16 or page_min.shape != (g,p,d) or page_max.shape != page_min.shape:
        raise ValueError('Quest fusion requires head_dim=128 and >=16 output rows')
    if q.dtype != torch.float32 or page_min.dtype not in (torch.bfloat16,torch.float16,torch.float32) or page_max.dtype != page_min.dtype:
        raise ValueError('unsupported Quest score dtypes')
    if not all(t.is_cuda and t.device == q.device and t.is_contiguous() for t in (q,page_min,page_max)):
        raise ValueError('Quest fusion requires contiguous same-device CUDA tensors')
    out = torch.empty((g,r,p),device=q.device,dtype=torch.float32)
    _score[(g*r,triton.cdiv(p,4))](q,page_min,page_max,out,p,r,4,num_warps=4,enable_fp_fusion=False)
    return out
