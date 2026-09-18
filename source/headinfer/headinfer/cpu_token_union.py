"""Exact sorted CPU token union with a reusable bounded membership array."""
import numpy as np
import torch


class CpuTokenUnion:
    def __init__(self, capacity: int):
        self.members = np.zeros(capacity, dtype=np.bool_)

    def __call__(self, positions: list[torch.Tensor], length: int) -> torch.Tensor:
        if not 0 <= length <= self.members.size:
            raise ValueError('invalid union length')
        members = self.members[:length]
        members.fill(False)
        for selected in positions:
            if selected.device.type != 'cpu' or selected.dtype != torch.int64:
                raise ValueError('union expects CPU int64 indices')
            array = selected.numpy()
            if array.size and (array.min() < 0 or array.max() >= length):
                raise ValueError('selected position outside history')
            members[array] = True
        # flatnonzero owns a separate output: later buffer reuse cannot mutate it.
        return torch.from_numpy(np.flatnonzero(members))
