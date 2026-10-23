"""Independently reconstruct primary metrics from exported cycle records."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

root=Path(__file__).parent
p=root/'results_v2'
per=pd.read_csv(p/'per_seed_results_v2.csv')
rec=pd.read_csv(p/'cycle_warnings_v2.csv.gz')
for keys,g in rec.groupby(['seed','scenario','method']):
    seed,scenario,method=keys
    event=g.event_time>=0
    advance=event&(g.warning_time>=0)&(g.warning_time<g.event_time)
    timely=advance&((g.event_time-g.warning_time)<=60)
    early=advance&~timely
    false=(~event)&(g.warning_time>=0)
    denom=timely.sum()+early.sum()+false.sum()
    calculated={'detection_rate':timely.sum()/event.sum(),
        'false_alarm_rate':false.sum()/(~event).sum(),
        'early_warning_rate':early.sum()/event.sum(),
        'precision':timely.sum()/denom if denom else np.nan,
        'mean_lead_s':(g.event_time-g.warning_time)[timely].mean()}
    r=per[(per.seed==seed)&(per.scenario==scenario)&(per.method==method)].iloc[0]
    for metric,value in calculated.items():
        if not np.isclose(value,r[metric],equal_nan=True):
            raise AssertionError((keys,metric,value,r[metric]))
manifest=json.loads((p/'run_manifest.json').read_text())
actual=hashlib.sha256((root/'overflow_study_v2.py').read_bytes()).hexdigest()
if actual!=manifest['code_sha256']:raise AssertionError('Source differs from final run manifest')
print(f'PASS: {len(rec)} cycle records reconstruct {len(per)} pipeline/condition/seed metric rows.')
print('PASS: source SHA-256 matches final run manifest.')
