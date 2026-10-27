"""Reconstruct reported sensitivity metrics from individual warning records."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).parent
p=ROOT/'sensitivity_results'
rec=pd.read_csv(p/'cycle_warnings_sensitivity.csv.gz')
per=pd.read_csv(p/'per_seed_sensitivity.csv')
su=pd.read_csv(p/'summary_sensitivity.csv')
assert len(rec)==270000 and len(per)==540
for keys,g in rec.groupby(['seed','scenario','method']):
    seed,scenario,method=keys
    ev=g.event_time>=0
    advance=ev&(g.warning_time>=0)&(g.warning_time<g.event_time)
    timely=advance&((g.event_time-g.warning_time)<=60)
    early=advance&~timely
    false=(~ev)&(g.warning_time>=0)
    denom=timely.sum()+early.sum()+false.sum()
    lead=(g.event_time-g.warning_time)[timely]
    values={'detection_rate':timely.sum()/ev.sum(),'false_alarm_rate':false.sum()/(~ev).sum(),
            'early_warning_rate':early.sum()/ev.sum(),'precision':timely.sum()/denom if denom else np.nan,
            'mean_lead_s':lead.mean(),'median_lead_s':lead.median(),
            'any_advance_detection':advance.sum()/ev.sum()}
    r=per[(per.seed==seed)&(per.scenario==scenario)&(per.method==method)].iloc[0]
    for name,value in values.items():
        np.testing.assert_allclose(value,r[name],equal_nan=True,rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(timely.to_numpy(),g.timely.to_numpy())
    assert len(g)==500 and len(set(g.cycle_seed))==500
for (condition,method),g in per.groupby(['scenario','method']):
    r=su[(su.scenario==condition)&(su.method==method)].iloc[0]
    for metric in ('detection_rate','false_alarm_rate','early_warning_rate','mean_lead_s'):
        values=g[metric].dropna()
        ci=stats.t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
        np.testing.assert_allclose(values.mean(),r[metric+'_mean'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(ci,r[metric+'_ci95'],rtol=1e-10,atol=1e-10)
paired=[]
for method in per.method.unique():
    a=per[(per.method==method)&(per.scenario=='fresh_nominal')].set_index('seed')
    b=per[(per.method==method)&(per.scenario=='freeboard_error_20pct')].set_index('seed')
    for metric in ('detection_rate','false_alarm_rate','mean_lead_s'):
        x=(b[metric]-a[metric]).dropna()
        paired.append({'method':method,'metric':metric,'difference_error_minus_nominal':x.mean(),
                       'ci95':stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)),'n_seeds':len(x)})
pd.DataFrame(paired).to_csv(p/'calibration_paired_differences.csv',index=False)
m=json.loads((p/'sensitivity_manifest.json').read_text())
assert hashlib.sha256((ROOT/'sensitivity_study.py').read_bytes()).hexdigest()==m['extension_code_sha256']
assert hashlib.sha256((ROOT/'overflow_study_v2.py').read_bytes()).hexdigest()==m['base_code_sha256']
print(f'PASS: {len(rec):,} warning records reconstruct {len(per)} per-seed rows and {len(su)} summary rows.')
print('PASS: 95% confidence intervals, labels, cycle counts, and source hashes verified.')
