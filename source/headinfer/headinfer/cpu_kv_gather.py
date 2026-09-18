"""Opt-in bitwise CPU KV row-copy, compiled before model timing."""
import ctypes
import subprocess
import tempfile
from pathlib import Path
import torch


class CpuKVGather:
    def __init__(self):
        self.build_dir = tempfile.TemporaryDirectory(prefix='twilight-cpu-kv-')
        library = Path(self.build_dir.name)/'gather.so'
        subprocess.run(['g++','-O3','-fopenmp','-shared','-fPIC',
                        str(Path(__file__).with_suffix('.cpp')),'-o',str(library)], check=True)
        self.library = ctypes.CDLL(str(library))
        self.fn = self.library.gather_kv
        self.fn.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int64]*3 + [ctypes.c_int]
        self.fn.restype = ctypes.c_int

    def __call__(self, key, value, rows, out_key, out_value):
        tensors=(key,value,rows,out_key,out_value)
        if any(t.device.type!='cpu' or not t.is_contiguous() for t in tensors):
            raise ValueError('CPU contiguous tensors required')
        if key.ndim!=2 or key.shape!=value.shape or key.dtype!=value.dtype:
            raise ValueError('invalid source layout')
        if rows.ndim!=1 or rows.dtype!=torch.int64:
            raise ValueError('int64 row indices required')
        if out_key.shape!=(rows.numel(),key.shape[1]) or out_key.shape!=out_value.shape:
            raise ValueError('invalid output layout')
        if out_key.dtype!=key.dtype or out_value.dtype!=key.dtype:
            raise ValueError('dtype mismatch')
        if rows.numel()==0:
            return
        if self.fn(*(t.data_ptr() for t in (key,value,out_key,out_value,rows)),rows.numel(),key.shape[0],
                   key.shape[1]*key.element_size(),torch.get_num_threads()):
            raise ValueError('row index out of bounds')
