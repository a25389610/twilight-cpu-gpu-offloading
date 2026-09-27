"""Replay the original per-head FP32 cumsum kernels with fewer host launches.

The scan algorithm and each row's tensor shape remain PyTorch's original
one-dimensional cumsum.  A private static input is required by CUDA Graph;
the caller must consume the returned output before the next replay.
"""

from __future__ import annotations

import torch


def _per_head_scan(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [row.cumsum(dim=0) for row in probabilities.flatten(0, 1)], dim=0
    ).view_as(probabilities)


class ExactPerHeadScanGraph:
    def __init__(self) -> None:
        self._graphs: dict[tuple[tuple[int, ...], torch.dtype, torch.device], tuple] = {}

    def __call__(self, probabilities: torch.Tensor) -> torch.Tensor:
        if not probabilities.is_cuda or not probabilities.is_contiguous():
            raise ValueError("CUDA graph scan requires contiguous CUDA probabilities")
        key = (tuple(probabilities.shape), probabilities.dtype, probabilities.device)
        state = self._graphs.get(key)
        if state is None:
            static_input = torch.empty_like(probabilities)
            warmup = torch.cuda.Stream(device=probabilities.device)
            warmup.wait_stream(torch.cuda.current_stream(probabilities.device))
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    _per_head_scan(static_input)
            torch.cuda.current_stream(probabilities.device).wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = _per_head_scan(static_input)
            state = (static_input, static_output, graph)
            self._graphs[key] = state
        static_input, static_output, graph = state
        static_input.copy_(probabilities)
        graph.replay()
        return static_output
