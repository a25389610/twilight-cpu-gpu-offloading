"""Local affine INT4 sparse GEMV; inspired by Twilight's fused loading design.

Not the official kernel: preserves adjacent nibble layout, per-token scale/min,
and FP32 arithmetic. Direct QK writes only logits; materialize=True writes one
FP32 K tensor to retain the original PyTorch matmul reduction order.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _qk(Q, K, S, M, POS, OUT, H: tl.constexpr, N,
        R: tl.constexpr, D: tl.constexpr, T: tl.constexpr, MATERIALIZE: tl.constexpr):
    row = tl.program_id(0)
    g = row // R
    n = tl.program_id(1) * T + tl.arange(0, T)
    d = tl.arange(0, D)
    pos = tl.load(POS + row * N + n, n < N, 0)
    packed = tl.load(K + (g * H + pos[:, None]) * (D // 2)
                     + d[None, :] // 2, n[:, None] < N, 0).to(tl.int32)
    code = ((packed >> ((d[None, :] % 2) * 4)) & 15).to(tl.float32)
    scale = tl.load(S + g * H + pos, n < N, 0).to(tl.float32)
    minimum = tl.load(M + g * H + pos, n < N, 0).to(tl.float32)
    key = code * scale[:, None] + minimum[:, None]
    if MATERIALIZE:
        tl.store(OUT + (row * N + n[:, None]) * D + d[None, :], key, n[:, None] < N)
    else:
        query = tl.load(Q + row * D + d).to(tl.float32)
        value = tl.sum(key * query[None, :], 1)
        tl.store(OUT + row * N + n, value, n < N)


def fused_affine_int4_qk(q, packed, scale, minimum, positions, *, materialize=False):
    """Return FP32 QK or prepared K. Positions must be in [0,H), as guaranteed
    by the cache's B0 expansion. Direct QK does not promise bitwise torch parity.
    """
    g, r, d = q.shape
    h = packed.shape[1]
    n = positions.shape[-1]
    if d != 128 or packed.shape != (g, h, d // 2):
        raise ValueError("fused Twilight QK currently requires head_dim=128")
    if scale.shape != (g, h) or minimum.shape != (g, h) or positions.shape != (g, r, n):
        raise ValueError("incompatible affine INT4 metadata/positions")
    if packed.dtype != torch.uint8 or positions.dtype != torch.int64 or q.dtype != torch.float32:
        raise ValueError("expected uint8 packed K, int64 positions, FP32 query")
    if not all(t.is_cuda and t.device == q.device and t.is_contiguous()
               for t in (q, packed, scale, minimum, positions)):
        raise ValueError("fused QK expects contiguous tensors on the same CUDA device")
    shape = (g, r, n, d) if materialize else (g, r, n)
    out = torch.empty(shape, device=q.device, dtype=torch.float32)
    if n:
        _qk[(g * r, triton.cdiv(n, 16))](q, packed, scale, minimum, positions,
                                       out, h, n, r, d, 16, materialize, enable_fp_fusion=False)
    return out
