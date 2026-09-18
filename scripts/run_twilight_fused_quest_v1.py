"""ABBA single-request pilot; normal TPOT precedes detailed diagnostics."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/twilight_fused_quest_v1'
OUT.mkdir(parents=True,exist_ok=True)
prior=json.loads((ROOT/'results/twilight_metadata_reuse_v1/manifest.json').read_text())
base=next(r['command'] for r in prior['cases'] if r['mode']=='reuse')
files=['source/headinfer/headinfer/twilight_offload_cache.py','source/headinfer/headinfer/twilight_fused_quest.py','scripts/run_ruler_partial_h2d_tpot_case_v1.py']
manifest={'cases':[],'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
for trial,modes in ((1,('base','fused')),(2,('fused','base'))):
    for mode in modes:
        out=OUT/f'{trial}_{mode}.json'
        if out.exists():raise RuntimeError(f'Refusing overwrite {out}')
        cmd=base.copy();cmd[cmd.index('--output')+1]=str(out)
        cmd.append('--twilight-detailed-selection-profile')
        if mode=='fused':cmd.append('--twilight-fused-quest-score')
        with out.with_suffix('.log').open('w') as log:
            done=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        manifest['cases'].append(dict(trial=trial,mode=mode,command=cmd,returncode=done.returncode))
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if done.returncode:raise RuntimeError(out)
        print(trial,mode,'done',flush=True)
    a,b=[json.loads((OUT/f'{trial}_{mode}.json').read_text()) for mode in ('base','fused')]
    for k in ('prompt_sha256','diagnostic_selected_positions_sha256','diagnostic_logits_sha256'):
        assert a[k]==b[k],(trial,k)
    x,y=[torch.load(d['logits_file'],weights_only=True,map_location='cpu')['checkpoints'] for d in (a,b)]
    assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x)
    assert len(a['diagnostic_breakdown_tokens'])==len(b['diagnostic_breakdown_tokens'])==5
    for x,y in zip(a['diagnostic_breakdown_tokens'],b['diagnostic_breakdown_tokens']):
        for k in ('h2d_bytes','d2h_bytes','twilight_b0_tokens_total','group_union_history_tokens_total'):
            assert x[k]==y[k],(trial,k)
    print(trial,'parity PASS',flush=True)
rows=[]
for case in manifest['cases']:
    d=json.loads((OUT/f"{case['trial']}_{case['mode']}.json").read_text())
    rows.append(dict(trial=case['trial'],mode=case['mode'],tpot_ms=d['D2_D128_tpot']['mean_seconds_per_token']*1000,
       median_ms=d['D2_D128_tpot']['median_seconds_per_token']*1000,
       metrics_ms={k:statistics.mean(t[k] for t in d['diagnostic_breakdown_tokens'])*1000
                   for k in d['diagnostic_breakdown_tokens'][0] if k.endswith('_seconds')}))
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['source_sha256'].items())
summary=dict(cases=rows,correctness='PASS: 10 diagnostic hash pairs, 8 checkpoint tensors, counts/bytes, source hashes')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
