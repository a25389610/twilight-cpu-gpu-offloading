"""Exact hybrid run/row copy, threshold 16; compilation outside decode timing."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
from .cpu_kv_gather import CpuKVGather

class CpuKVRunGather(CpuKVGather):
    def __init__(self):
        self.build_dir = tempfile.TemporaryDirectory(prefix='twilight-cpu-run-')
        library = Path(self.build_dir.name) / 'gather.so'
        subprocess.run(['g++','-O3','-fopenmp','-shared','-fPIC',
                        str(Path(__file__).with_suffix('.cpp')),'-o',str(library)],check=True)
        self.library = ctypes.CDLL(str(library))
        self.fn = self.library.gather_kv
        self.fn.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int64]*3 + [ctypes.c_int]
        self.fn.restype = ctypes.c_int
