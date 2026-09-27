"""CUDA Graph replay of the unchanged PyTorch sort, softmax, scan and Top-p.

The graph is a submission optimization.  It keeps the 24 separate 1-D
cumsums that define the current exact Top-p boundary behavior.
"""

from __future__ import annotations

import torch

from .twilight_exact_scan_graph import _per_head_scan


class ExactTopPGraph:
    def __init__(self, top_p: float) -> None:
        self.top_p = float(top_p)
        self._graphs: dict[tuple[tuple[int, ...], torch.dtype, torch.device], tuple] = {}

    def _execute(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        order = torch.argsort(logits, dim=-1, descending=True)
        ordered_logits = torch.gather(logits, -1, order)
        probabilities = torch.softmax(ordered_logits, dim=-1)
        cumulative = _per_head_scan(probabilities)
        thresholds = torch.full(
            (*cumulative.shape[:-1], 1), self.top_p,
            dtype=cumulative.dtype, device=cumulative.device,
        )
        desired = torch.searchsorted(
            cumulative.contiguous(), thresholds, right=False
        ).squeeze(-1) + 1
        desired.clamp_max_(order.shape[-1])
        return order, desired

    def __call__(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not logits.is_cuda or not logits.is_contiguous():
            raise ValueError("Top-p graph requires contiguous CUDA logits")
        key = (tuple(logits.shape), logits.dtype, logits.device)
        state = self._graphs.get(key)
        if state is None:
            static_input = torch.empty_like(logits)
            warmup = torch.cuda.Stream(device=logits.device)
            warmup.wait_stream(torch.cuda.current_stream(logits.device))
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    self._execute(static_input)
            torch.cuda.current_stream(logits.device).wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                order, desired = self._execute(static_input)
            state = (static_input, order, desired, graph)
            self._graphs[key] = state
        static_input, order, desired, graph = state
        static_input.copy_(logits)
        graph.replay()
        return order, desired
