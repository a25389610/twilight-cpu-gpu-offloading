"""Reference decode attention used only by joint-gate diagnostics.

The production HeadInfer path continues to use FlashAttention.  This helper
forces PyTorch SDPA's math backend for the per-KV-entry, q_len=1 calls emitted
by ``mp_headinfer`` so backward reproducibility can be isolated without
changing ``mp.py``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def sdpa_math_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    *,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Evaluate one causal decode token with the deterministic SDPA math path.

    Inputs follow Transformers' FlashAttention layout ``[batch, seq, heads,
    dim]``.  At decode q_len=1 all supplied keys are causally visible, so
    ``is_causal=False`` is intentional; PyTorch's non-square causal mask would
    otherwise describe a top-left-aligned query rather than the final token.
    """

    del kwargs
    if int(query_length) != 1 or query_states.shape[1] != 1:
        raise AssertionError("reference attention is decode-only (q_len=1)")
    if attention_mask is not None:
        raise AssertionError("reference diagnostic expects no explicit attention mask")
    if float(dropout) != 0.0:
        raise AssertionError("reference diagnostic requires dropout=0")
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        raise AssertionError("reference attention expects rank-4 Q/K/V")
    if key_states.shape != value_states.shape:
        raise AssertionError("reference K/V shapes differ")
    if key_states.shape[2] != 1:
        raise AssertionError("reference path expects one KV head per call")
    if scaling is not None and softmax_scale is not None and scaling != softmax_scale:
        raise AssertionError("conflicting attention scales")
    scale = scaling if scaling is not None else softmax_scale

    query = query_states.transpose(1, 2)
    key = key_states.transpose(1, 2)
    value = value_states.transpose(1, 2)
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
    return output.transpose(1, 2).contiguous()
