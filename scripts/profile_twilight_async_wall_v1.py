"""Mutually exclusive host intervals, independent GPU phase lane; no phase sync."""
import json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
mode=sys.argv[1] if len(sys.argv)>1 else 'profile'
gpu_full=mode in ('gpu_profile','gpu_baseline')
profiling=mode in ('profile','gpu_profile')
OUT=ROOT/('results/twilight_gpu_wall_v1' if gpu_full else 'results/twilight_async_wall_v1')
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'source/headinfer'))
from headinfer.quest_offload_cache import QuestTopKOffloadedCache as Q
from headinfer.twilight_offload_cache import TwilightMatchedBudgetOffloadedCache as T
from headinfer.cpu_token_union import CpuTokenUnion
import headinfer.mp as mp
active=False
stack=[]
segments=[]
gpu=[]
steps=[]
anchor=None
clock=time.perf_counter

def mark(label,now):
    global last
    if now>last: segments.append((stack[-1],last,now))
    last=now

def phase(label,fn,device=False):
    def call(*a,**kw):
        if not active: return fn(*a,**kw)
        mark(label,clock());stack.append(label)
        x=y=None
        if device:
            stream=torch.cuda.current_stream()
            x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True);x.record()
        try: return fn(*a,**kw)
        finally:
            if device: y.record();gpu.append((label,x,y,int(stream.cuda_stream)))
            mark(label,clock());stack.pop()
    return call

enable=Q.enable_metrics
def enabled(self,on=True):
    global anchor,anchor_cpu,uncertainty
    if on and profiling:
        before=clock();anchor=torch.cuda.Event(enable_timing=True)
        anchor.record();anchor.synchronize() # diagnostic entry, outside measured token
        anchor_cpu=clock();uncertainty=anchor_cpu-before
    return enable(self,on)
Q.enable_metrics=enabled

def begin(start):
    global active,last,segments,gpu,stack,origin
    if not profiling: return
    segments=[];gpu=[];stack=['Other_control'];last=origin=start;active=True

def end(total):
    global active
    if not profiling: return
    mark('end',origin+total);active=False
    assert stack==['Other_control']
    totals={}
    for label,s,e in segments: totals[label]=totals.get(label,0)+(e-s)*1000
    assert abs(sum(totals.values())-total*1000)<1e-6
    device=[{'label':label,'start_ms':anchor.elapsed_time(x)+(anchor_cpu-origin)*1000,
             'stream':stream,'duration_ms':x.elapsed_time(y)} for label,x,y,stream in gpu]
    steps.append({'wall_ms':total*1000,'cpu_ms':totals,'gpu':device,
                  'alignment_uncertainty_ms':uncertainty*1000,
                  'segments':[(n,(s-origin)*1000,(e-s)*1000) for n,s,e in segments]})
    (OUT/'intervals.json').write_text(json.dumps(steps))
    events=[]
    for i,step in enumerate(steps):
        for n,s,d in step['segments']:
            events.append(dict(name=n,ph='X',pid=i,tid=1,ts=s*1000,dur=d*1000))
        for r in step['gpu']:
            events.append(dict(name=r['label'],ph='X',pid=i,tid=r['stream'],ts=r['start_ms']*1000,dur=r['duration_ms']*1000))
    (OUT/'timeline.chrome.json').write_text(json.dumps({'traceEvents':events}))

patch=mp.mp_headinfer
def model_patch(model,*a,**kw):
    result=patch(model,*a,**kw)
    def module(m,n,device=False): m.forward=phase(n,m.forward,device or gpu_full)
    module(model.model.embed_tokens,'Embedding')
    for layer in model.model.layers:
        for kind in ('q_proj','k_proj','v_proj','o_proj'): module(getattr(layer.self_attn,kind),'QKVO_launch')
        module(layer.mlp,'MLP_launch',True)
        module(layer.input_layernorm,'Norm_launch');module(layer.post_attention_layernorm,'Norm_launch')
    module(model.model.norm,'Norm_launch');module(model.lm_head,'LM_head_launch')
    if hasattr(model.model,'rotary_emb'):module(model.model.rotary_emb,'RoPE_launch')
    return result
mp.mp_headinfer=model_patch
mp.apply_rotary_pos_emb=phase('RoPE_launch',mp.apply_rotary_pos_emb,gpu_full)
mp._flash_attention_varlen_forward=phase('Attention_launch',mp._flash_attention_varlen_forward,True)
T.prepare_layer_selection=phase('Selection_host_excluding_cpu_wait',T.prepare_layer_selection,True)
T.update_layer_gqa_group_ragged=phase('KV_control_exclusive',T.update_layer_gqa_group_ragged)
T._record_copy=phase('H2D_enqueue',T._record_copy,True)
T._schedule_d2h=phase('New_KV_D2H_enqueue',T._schedule_d2h)
CpuTokenUnion.__call__=phase('CPU_union',CpuTokenUnion.__call__)
orig_cpu=torch.Tensor.cpu
def cpu(t,*a,**kw):
    return phase('Indices_cpu_blocking',orig_cpu,gpu_full)(t,*a,**kw) if active and t.is_cuda else orig_cpu(t,*a,**kw)
torch.Tensor.cpu=cpu
orig_select=torch.index_select
def select(t,*a,**kw):
    return phase('CPU_gather_operator',orig_select)(t,*a,**kw) if active and t.device.type=='cpu' and kw.get('out') is not None else orig_select(t,*a,**kw)
torch.index_select=select
orig_tensor=torch.tensor
def tensor(*a,**kw):
    if active and stack[-1]=='KV_control_exclusive':
        is_gpu='cuda' in str(kw.get('device','cpu'))
        return phase('GPU_metadata_call' if is_gpu else 'CPU_metadata_call',orig_tensor,gpu_full and is_gpu)(*a,**kw)
    return orig_tensor(*a,**kw)
torch.tensor=tensor
torch.cuda.Event.synchronize=phase('Existing_sync_wait',torch.cuda.Event.synchronize)
torch.cuda.Stream.synchronize=phase('Existing_sync_wait',torch.cuda.Stream.synchronize)
torch.cuda.synchronize=phase('Existing_sync_wait',torch.cuda.synchronize)

if gpu_full:
    original_copy=torch.Tensor.copy_
    def copy(dst,src,*a,**kw):
        if active and isinstance(src,torch.Tensor):
            if dst.device.type=='cpu' and src.is_cuda:
                return phase('New_KV_D2H_copy_interval',original_copy,True)(dst,src,*a,**kw)
            if dst.is_cuda and src.is_cuda and stack[-1]=='KV_control_exclusive':
                return phase('New_token_GPU_copy',original_copy,True)(dst,src,*a,**kw)
        return original_copy(dst,src,*a,**kw)
    torch.Tensor.copy_=copy
    original_add=torch.Tensor.__add__
    def add(t,other):
        if active and stack==['Other_control'] and isinstance(other,torch.Tensor) and tuple(t.shape)==(1,1,3072) and t.shape==other.shape:
            return phase('Residual_add',original_add,True)(t,other)
        return original_add(t,other)
    torch.Tensor.__add__=add

prior=json.loads((ROOT/'results/twilight_early_metadata_v1/manifest.json').read_text())
cmd=next(x['command'] for x in prior['cases'] if x['mode']=='early').copy()
cmd[cmd.index('--output')+1]=str(OUT/f'{mode}.json')
(OUT/f'{mode}_command.json').write_text(json.dumps(cmd,indent=2))
runner=Path(cmd[1]);source=runner.read_text()
# Two diagnostic-only callbacks use the runner's exact measured boundaries.
# Original runner on disk and all normal timing code remain unchanged.
needle='            diagnostic_started = time.perf_counter()\n'
assert source.count(needle)==1
source=source.replace(needle,needle+'            _async_begin(diagnostic_started)\n')
needle='            diagnostic_wall_seconds = time.perf_counter() - diagnostic_started\n'
assert source.count(needle)==1
source=source.replace(needle,needle+'            _async_end(diagnostic_wall_seconds)\n')
sys.argv=cmd[1:]
exec(compile(source,str(runner),'exec'),{'__name__':'__main__','__file__':str(runner),
     '_async_begin':begin,'_async_end':end})
