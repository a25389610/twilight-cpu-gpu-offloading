"""Exact integer gather/mask fusion. Sorting remains the original torch.sort."""
import torch
import triton
import triton.language as tl

@triton.jit
def _gather_mask(POS, ORDER, COUNTS, OUT, N, M, sentinel, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    count=tl.load(COUNTS+row)
    live=(col<M)&(col<count)
    rank=tl.load(ORDER+row*N+col,live,0)
    pos=tl.load(POS+row*N+rank,live,0)
    tl.store(OUT+row*M+col,tl.where(live,pos,sentinel),col<M)

def fused_gather_mask(positions,order,counts,width,sentinel):
    if positions.shape!=order.shape or positions.shape[:-1]!=counts.shape:
        raise ValueError('incompatible indices shapes')
    if not all(t.is_cuda and t.device==positions.device and t.dtype==torch.int64 and t.is_contiguous() for t in (positions,order,counts)):
        raise ValueError('expected contiguous CUDA int64 tensors')
    if not 0<width<=positions.shape[-1]:raise ValueError('invalid output width')
    out=torch.empty((*counts.shape,width),device=positions.device,dtype=torch.int64)
    _gather_mask[(counts.numel(),triton.cdiv(width,256))](positions,order,counts,out,positions.shape[-1],width,sentinel,256)
    return out
