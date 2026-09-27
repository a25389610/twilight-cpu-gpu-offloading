"""Diagnostic control: same mapped-read path, but all history rows are misses.

This wrapper patches only the bitmap passed as previous-token membership.
The underlying runner and cache implementation stay in their default state.
"""

import runpy
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/headinfer"))
import headinfer.resident_gpu_mapping as mapping  # noqa: E402


original_map = mapping.map_membership
zero_bitmap = None


def force_all_miss(current_bitmap, previous_bitmap):
    global zero_bitmap
    if zero_bitmap is None:
        zero_bitmap = torch.zeros(
            (previous_bitmap.shape[0], 40000),
            dtype=torch.bool, device=previous_bitmap.device,
        )
    if previous_bitmap.shape[1] > zero_bitmap.shape[1]:
        raise ValueError("no-hit diagnostic bitmap capacity exceeded")
    return original_map(
        current_bitmap, zero_bitmap[:, :previous_bitmap.shape[1]]
    )


mapping.map_membership = force_all_miss
runner = ROOT / "scripts/run_ruler_partial_h2d_tpot_case_v1.py"
sys.argv = [str(runner), *sys.argv[1:]]
runpy.run_path(str(runner), run_name="__main__")
