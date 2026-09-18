"""CPU/GPU scopes for a supplied frozen command; production files unchanged."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
command_file=Path(sys.argv[1]).resolve()
kind=sys.argv[2]
assert kind in ('cpu','gpu')
command=json.loads(command_file.read_text())
if '--smoke' in sys.argv[3:]:
    command.append('--help')
out=Path(command[command.index('--output')+1]).parent
path=ROOT/'scripts/profile_twilight_complete_20260914.py'
source=path.read_text()
# Reuse complete scope layer, but supply per-request command and isolated output.
source=source.replace('kind = sys.argv[1]', f'kind = {kind!r}')
needle="sys.argv = [str(path), 'gpu_profile' if kind == 'gpu' else 'profile']"
addition='''
start=source.index("prior=json.loads")
stop=source.index("runner=Path(cmd[1]);source=runner.read_text()",start)
source=source[:start]+"cmd="+repr(FROZEN_COMMAND)+"\\n"+source[stop:]
source=source.replace("OUT=ROOT/'results/twilight_complete_20260914/"+kind+"'", "OUT=Path("+repr(str(FROZEN_OUT))+ ")")
# Split post-cpu() ragged unpack from Selection host interval, without sync.
hook="""
import inspect, textwrap
import headinfer.twilight_offload_cache as twilight_module
def ragged_begin():
    if active:
        mark('ragged',clock())
        stack[-1]='CPU_ragged_unpack_and_cache'
function=textwrap.dedent(inspect.getsource(T.prepare_layer_selection))
needle='    allocated_cpu = transfer_cpu[:, 0].view(groups, query_heads)'
assert function.count(needle)==1
function=function.replace(needle,'    _diagnostic_ragged_begin()'+chr(10)+needle)
scope=dict(twilight_module.__dict__)
scope['_diagnostic_ragged_begin']=ragged_begin
exec(compile(function,'<diagnostic_ragged_scope>','exec'),scope)
T.prepare_layer_selection=scope['prepare_layer_selection']
"""
needle="T.prepare_layer_selection=phase('Selection_host_excluding_cpu_wait',T.prepare_layer_selection,True)"
assert source.count(needle)==1
source=source.replace(needle,hook+'\\n'+needle)
# Legacy union and history GPU cat are absent from the latest optimized path.
extra="""
original_unique=torch.unique
def unique(t,*a,**kw):
    if active and stack[-1]=='KV_control_exclusive' and t.device.type=='cpu':
        return phase('CPU_union',original_unique)(t,*a,**kw)
    return original_unique(t,*a,**kw)
torch.unique=unique
original_cat=torch.cat
def cat(tensors,*a,**kw):
    if active and stack[-1]=='KV_control_exclusive':
        device=tensors[0].is_cuda
        return phase('GPU_history_cat' if device else 'CPU_union_concat',original_cat,device)(tensors,*a,**kw)
    return original_cat(tensors,*a,**kw)
torch.cat=cat
"""
source=source.replace('cmd='+repr(FROZEN_COMMAND),extra+'\\ncmd='+repr(FROZEN_COMMAND))
'''
assert source.count(needle)==1
source=source.replace(needle,addition+'\n'+needle)
sys.argv=[str(path),kind]
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),
    'FROZEN_COMMAND':command,'FROZEN_OUT':out})
