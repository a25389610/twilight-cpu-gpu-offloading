"""Opt-in fused CPU resident hit/miss mapping with bounded direct addressing."""

import ctypes
import subprocess
import tempfile
from pathlib import Path

import torch


class NativeResidentMapping:
    def __init__(self, capacity: int, *, compact_hits: bool = False,
                 attention_layout: bool = False, pack_hits: bool = False):
        self.capacity = int(capacity)
        self.compact_hits = bool(compact_hits)
        self.attention_layout = bool(attention_layout)
        self.pack_hits = bool(pack_hits)
        if self.attention_layout and not self.compact_hits:
            raise ValueError("attention-layout mapping requires compact hit indices")
        if self.pack_hits and not self.compact_hits:
            raise ValueError("packed hit indices require compact hit indices")
        self.build_dir = tempfile.TemporaryDirectory(prefix="twilight-resident-map-")
        library = Path(self.build_dir.name) / "resident_mapping.so"
        subprocess.run(
            ["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
             str(Path(__file__).with_suffix(".cpp")), "-o", str(library)],
            check=True,
        )
        self.library = ctypes.CDLL(str(library))
        self.fn = (
            self.library.map_resident_rows_compact_attention_layout
            if self.attention_layout else
            self.library.map_resident_rows_compact_hits
            if compact_hits else self.library.map_resident_rows
        )
        self.fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 2 + [ctypes.c_void_p] * 7
        self.fn.restype = ctypes.c_int

    def __call__(self, entries, current, previous):
        groups = len(entries)
        if groups == 0 or len(current) != groups or len(previous) != groups:
            raise ValueError("resident mapping groups must be aligned")
        tensors = (*current, *previous)
        if any(t.device.type != "cpu" or t.dtype != torch.int64 or
               t.ndim != 1 or not t.is_contiguous() for t in tensors):
            raise ValueError("resident positions must be contiguous CPU int64 vectors")
        total = sum(t.numel() for t in current)
        if self.pack_hits:
            hit_storage = torch.empty(2 * total, dtype=torch.int32)
            outputs = [hit_storage[:total], hit_storage[total:]]
            outputs.extend(torch.empty(total, dtype=torch.int64) for _ in range(3))
        else:
            outputs = [
                torch.empty(total, dtype=(torch.int32 if self.compact_hits and i < 2
                                          else torch.int64))
                for i in range(5)
            ]
        ptrs = ctypes.c_void_p * groups
        lengths = ctypes.c_int64 * groups
        previous_ptrs = ptrs(*(t.data_ptr() for t in previous))
        current_ptrs = ptrs(*(t.data_ptr() for t in current))
        previous_lengths = lengths(*(t.numel() for t in previous))
        current_lengths = lengths(*(t.numel() for t in current))
        entry_values = lengths(*(int(entry) for entry in entries))
        hit_count = ctypes.c_int64()
        miss_count = ctypes.c_int64()
        code = self.fn(
            previous_ptrs, current_ptrs, previous_lengths, current_lengths,
            entry_values, groups, self.capacity,
            *(t.data_ptr() for t in outputs),
            ctypes.byref(hit_count), ctypes.byref(miss_count),
        )
        if code:
            raise ValueError(f"invalid resident positions (native error {code})")
        if hit_count.value + miss_count.value != total:
            raise AssertionError("resident hit/miss partition is incomplete")
        if self.pack_hits:
            # The native pass needs the second output at a known fixed offset.
            # Compact it after the hit count is known so one H2D copies both
            # variable-length hit arrays without padding to ``total``.
            ctypes.memmove(
                hit_storage.data_ptr() + hit_count.value * 4,
                hit_storage.data_ptr() + total * 4,
                hit_count.value * 4,
            )
            packed = hit_storage[:2 * hit_count.value]
            return (
                packed[:hit_count.value], packed[hit_count.value:],
                outputs[2][:miss_count.value], outputs[3][:miss_count.value],
                outputs[4], packed,
            )
        return (
            outputs[0][:hit_count.value], outputs[1][:hit_count.value],
            outputs[2][:miss_count.value], outputs[3][:miss_count.value],
            outputs[4],
        )
