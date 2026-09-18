import hashlib
import json
from pathlib import Path
import statistics as st
import subprocess
import torch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/twilight_gather_improvements_v1'
OUT.mkdir(parents=True,exist_ok=True)
prior=json.loads((ROOT/'results/twilight_fused_quest_v1/manifest.json').read_text())
base=next(c['command'] for c in prior['cases'] if c['mode']=='fused').copy()
base.remove('--twilight-detailed-selection-profile')
files=['source/headinfer/headinfer/twilight_offload_cache.py','scripts/run_ruler_partial_h2d_tpot_case_v1.py',
       'source/headinfer/headinfer/cpu_kv_run_gather.cpp','source/headinfer/headinfer/cpu_kv_run_gather.py']
manifest=dict(cases=[],source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})
flags={'base':[],'views':['--twilight-skip-unused-host-views'],'hybrid':['--twilight-cpu-run-gather']}
summary=[]
for trial,modes in ((1,('base','views','hybrid')),(2,('hybrid','views','base'))):
    for mode in modes:
        out=OUT/f'{trial}_{mode}.json'
        if out.exists():raise RuntimeError('refusing overwrite')
        cmd=base.copy();cmd[cmd.index('--output')+1]=str(out);cmd+=flags[mode]
        with out.with_suffix('.log').open('w') as log:done=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        manifest['cases'].append(dict(trial=trial,mode=mode,command=cmd,returncode=done.returncode))
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if done.returncode:raise RuntimeError(out)
        print(trial,mode,'done',flush=True)
    a=json.loads((OUT/f'{trial}_base.json').read_text())
    x=torch.load(a['logits_file'],weights_only=True,map_location='cpu')['checkpoints']
    for mode in ('base','views','hybrid'):
        b=json.loads((OUT/f'{trial}_{mode}.json').read_text())
        for k in ('prompt_sha256','diagnostic_selected_positions_sha256','diagnostic_logits_sha256'):assert a[k]==b[k],(trial,mode,k)
        y=torch.load(b['logits_file'],weights_only=True,map_location='cpu')['checkpoints']
        assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)
        assert len(a['diagnostic_breakdown_tokens'])==len(b['diagnostic_breakdown_tokens'])==5
        for t,u in zip(a['diagnostic_breakdown_tokens'],b['diagnostic_breakdown_tokens']):
            for k in ('h2d_bytes','d2h_bytes','twilight_b0_tokens_total','group_union_history_tokens_total'):assert t[k]==u[k]
        summary.append(dict(trial=trial,mode=mode,tpot_ms=b['D2_D128_tpot']['mean_seconds_per_token']*1000,
            median_ms=b['D2_D128_tpot']['median_seconds_per_token']*1000,
            metrics_ms={k:st.mean(t[k] for t in b['diagnostic_breakdown_tokens'])*1000 for k in b['diagnostic_breakdown_tokens'][0] if k.endswith('_seconds')}))
    (OUT/'summary.json').write_text(json.dumps(dict(cases=summary,verified_trials=trial),indent=2))
    print(trial,'correctness PASS',flush=True)
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['source_sha256'].items())
print('PASS: 20 diagnostic hash pairs, 16 checkpoints, counts/bytes, source hashes')
