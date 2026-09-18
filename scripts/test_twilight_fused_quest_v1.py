import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/headinfer'))
from headinfer.twilight_fused_quest import fused_quest_score

torch.manual_seed(20260911)
cases=0
with torch.inference_mode():
    for dtype in (torch.bfloat16,torch.float16,torch.float32):
        for pages in (17,186,465,1000,2025,2048):
            for query_kind in ('bf16','fp32','exponents'):
                q=torch.randn(8,3,128,device='cuda')
                if query_kind=='bf16':q=q.bfloat16().float()
                if query_kind=='exponents':q=q*torch.exp2(torch.randint(-12,12,q.shape,device='cuda').float())
                a=torch.randn(8,pages,128,device='cuda',dtype=dtype)
                b=torch.randn_like(a)
                lo=torch.minimum(a,b);hi=torch.maximum(a,b)
                ref=(torch.where(q.unsqueeze(2)>0,hi.unsqueeze(1),lo.unsqueeze(1)).float()*q.unsqueeze(2)).sum(-1)
                got=fused_quest_score(q,lo,hi)
                assert torch.equal(ref,got),(dtype,pages,query_kind,(ref-got).abs().max().item(),int((ref!=got).sum()))
                k=min(pages,512)
                assert torch.equal(ref.topk(k,dim=-1,sorted=False).indices,got.topk(k,dim=-1,sorted=False).indices)
                cases+=1
    for value in (0.,1.,-1.):
        q=torch.full((8,3,128),value,device='cuda')
        lo=torch.full((8,513,128),-2.,device='cuda',dtype=torch.bfloat16)
        hi=torch.ones_like(lo)
        ref=(torch.where(q.unsqueeze(2)>0,hi.unsqueeze(1),lo.unsqueeze(1)).float()*q.unsqueeze(2)).sum(-1)
        got=fused_quest_score(q,lo,hi)
        assert torch.equal(ref,got)
        assert torch.equal(ref.topk(512,dim=-1,sorted=False).indices,got.topk(512,dim=-1,sorted=False).indices)
        cases+=1
print(f'PASS: {cases} score tensors and Top-k indices bit-exact; torch={torch.__version__}')
