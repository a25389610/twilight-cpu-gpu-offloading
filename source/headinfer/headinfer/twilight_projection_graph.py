"""Replay the original grouped Llama QKV and RoPE operations per layer.

Weights remain at their original group views. The graph owns only a static
copy of the current token's hidden state and rotary position embeddings.
"""

from __future__ import annotations

import torch


class ExactLayerProjectionGraph:
    def __init__(self, hidden, position_embeddings, projection_fn):
        cos, sin = position_embeddings
        self.hidden = torch.empty_like(hidden)
        self.cos = torch.empty_like(cos)
        self.sin = torch.empty_like(sin)
        warmup = torch.cuda.Stream(device=hidden.device)
        warmup.wait_stream(torch.cuda.current_stream(hidden.device))
        with torch.cuda.stream(warmup):
            for _ in range(3):
                projection_fn(self.hidden, (self.cos, self.sin))
        torch.cuda.current_stream(hidden.device).wait_stream(warmup)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.states = projection_fn(self.hidden, (self.cos, self.sin))
        self.input_shapes = tuple(tuple(x.shape) for x in (hidden, cos, sin))
        self.input_dtypes = tuple(x.dtype for x in (hidden, cos, sin))

    def __call__(self, hidden, position_embeddings):
        cos, sin = position_embeddings
        inputs = (hidden, cos, sin)
        if tuple(tuple(x.shape) for x in inputs) != self.input_shapes:
            raise ValueError("projection graph input shape changed")
        if tuple(x.dtype for x in inputs) != self.input_dtypes:
            raise ValueError("projection graph input dtype changed")
        self.hidden.copy_(hidden)
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        self.graph.replay()
        return self.states
