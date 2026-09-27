"""CUDA mapped-host resident K/V assembly, opt-in for bounded VRAM studies."""

import ctypes
import subprocess
import tempfile
from pathlib import Path

import torch


class ResidentZeroCopy:
    def __init__(self, host_key: torch.Tensor, host_value: torch.Tensor):
        if not host_key.is_pinned() or not host_value.is_pinned():
            raise ValueError("zero-copy source KV must be pinned")
        runtime = ctypes.CDLL("libcudart.so.12")
        map_pointer = runtime.cudaHostGetDevicePointer
        map_pointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint
        ]
        map_pointer.restype = ctypes.c_int
        self.host_pointers = []
        for tensor in (host_key, host_value):
            mapped = ctypes.c_void_p()
            code = map_pointer(
                ctypes.byref(mapped), ctypes.c_void_p(tensor.data_ptr()), 0
            )
            if code or not mapped.value:
                raise RuntimeError(f"cudaHostGetDevicePointer failed: {code}")
            self.host_pointers.append(mapped.value)
        self.build_dir = tempfile.TemporaryDirectory(
            prefix="twilight-resident-zero-copy-"
        )
        library = Path(self.build_dir.name) / "resident_zero_copy.so"
        subprocess.run([
            "nvcc", "-shared", "-Xcompiler", "-fPIC", "-O3",
            "-arch=sm_120", str(Path(__file__).with_suffix(".cu")),
            "-o", str(library),
        ], check=True)
        self.library = ctypes.CDLL(str(library))
        self.fn = self.library.launch_resident_zero_copy
        self.fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        self.fn.restype = ctypes.c_int

    def __call__(
        self, previous_key, previous_value, hit_sources, current_positions,
        current_lengths, previous_lengths, attention_key, attention_value,
        *, host_capacity, layer, max_group_rows,
    ):
        tensors = (
            previous_key, previous_value, hit_sources, current_positions,
            current_lengths, previous_lengths, attention_key, attention_value,
        )
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("zero-copy GPU tensors must be contiguous")
        stream = torch.cuda.current_stream(attention_key.device).cuda_stream
        code = self.fn(
            *self.host_pointers,
            *(tensor.data_ptr() for tensor in tensors),
            current_positions.shape[1], host_capacity, layer,
            attention_key.shape[1], max_group_rows, stream,
        )
        if code:
            raise RuntimeError(f"resident zero-copy launch failed: {code}")
