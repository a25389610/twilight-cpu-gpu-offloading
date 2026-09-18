import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/headinfer'))
from headinfer.cpu_kv_run_gather import CpuKVRunGather
gather=CpuKVRunGather()
torch.manual_seed(1234)
cases=0
with torch.inference_mode():
    k=torch.randn(10000,128,dtype=torch.bfloat16)
    v=torch.randn_like(k)
    patterns=[torch.empty(0,dtype=torch.long),torch.tensor([9999]),torch.arange(9999),
        torch.randint(10000,(8192,)),torch.arange(8192).flip(0),
        torch.cat([torch.arange(100,500),torch.tensor([0,0,9999]),torch.arange(1000,7000)]),
        torch.cat([torch.arange(i,i+n) for i,n in ((100,15),(300,16),(600,17),(1000,33))])]
    for threads in (1,2,6):
        torch.set_num_threads(threads)
        for rows in patterns:
            # Pinned buffer offsets and guard rows emulate chunk slices.
            a=torch.full((rows.numel()+2,128),-42.,dtype=k.dtype,pin_memory=True)
            b=torch.full_like(a,-42.)
            gather(k,v,rows,a[1:-1],b[1:-1])
            assert torch.equal(a[1:-1],k.index_select(0,rows)) and torch.equal(b[1:-1],v.index_select(0,rows))
            assert (a[[0,-1]]==-42).all() and (b[[0,-1]]==-42).all()
            cases+=1
    for bad in (-1,10000):
        try:gather(k,v,torch.tensor([bad]),a[:1],b[:1])
        except ValueError:pass
        else:raise AssertionError('invalid index accepted')
print(f'PASS hybrid run gather: {cases} bit-exact patterns/thread cases, guards and invalid indices')
