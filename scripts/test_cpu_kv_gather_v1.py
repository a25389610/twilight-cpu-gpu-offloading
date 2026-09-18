import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/headinfer'))
from headinfer.cpu_kv_gather import CpuKVGather
gather=CpuKVGather()
torch.manual_seed(123)
for n in (0,1,8192):
    k=torch.randn(10000,128,dtype=torch.bfloat16)
    v=torch.randn_like(k)
    rows=torch.randint(10000,(n,))
    a=torch.empty(n,128,dtype=k.dtype); b=torch.empty_like(a)
    gather(k,v,rows,a,b)
    assert torch.equal(a,k[rows]) and torch.equal(b,v[rows])
for bad in (-1,10000):
    try:
        gather(k,v,torch.tensor([bad]),a[:1],b[:1])
    except ValueError:
        pass
    else:
        raise AssertionError('invalid index accepted')
print('PASS CPU KV gather: exact K/V, empty, singleton, duplicates, invalid indices')
