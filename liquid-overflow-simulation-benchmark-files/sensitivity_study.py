"""Post hoc sensitivity extension of the frozen v2 synthetic benchmark.

Run after overflow_study_v2.py. This extension does not change its results.
Alternative exponents and geometry error are illustrative, not calibrated.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import overflow_study_v2 as base
from overflow_study_v2 import C, first_crossing

CONDITIONS = {
    'fresh_nominal': (1., 2.),
    'heating_power_0.8': (0.8, 2.),
    'heating_power_1.2': (1.2, 2.),
    'growth_power_1.5': (1., 1.5),
    'growth_power_2.5': (1., 2.5),
    'freeboard_error_20pct': (1., 2.),
}

def sensitivity_cycle(seed: int, heating_power=1., growth_power=2., cfg=C):
    """Causal latent dynamics; no has-event flag or event time drives sensors."""
    scenario = 'nominal'
    rng = np.random.default_rng(int(seed))
    t = np.arange(cfg.duration_s)
    tau = rng.uniform(65., 100.)
    foamability = rng.uniform(0., 1.)
    collapse = rng.uniform(.006, .018)
    rim = rng.uniform(6., 12.) if scenario != 'small_freeboard' else rng.uniform(3., 6.)
    bulk_temperature = 25. + 74. * (1. - np.exp(-(t / tau) ** heating_power))
    activity = np.clip((bulk_temperature - 85.) / 14., 0., 1.)
    growth = (.025 + .18 * foamability) * (1.5 if scenario == 'faster_foam' else 1.)
    height = np.zeros(cfg.duration_s)
    for k in range(1, cfg.duration_s):
        height[k] = min(rim, max(0., height[k-1] + growth * activity[k-1] ** growth_power
                                - collapse * height[k-1]))
    event = int(first_crossing(height, np.asarray(rim)))
    noise = 3. if scenario == 'noise_x3' else 1.
    coupling = 0. if scenario == 'weak_acoustic' else (2. if scenario == 'faster_foam' else 9.)
    gain = rng.uniform(.8, 1.2)
    acoustic = 65. + gain * (150. * activity + coupling * height)
    acoustic += rng.normal(0., 12. * noise, len(t))
    distance = 40. - height + rng.normal(0., .15 * noise, len(t))
    temperature = bulk_temperature + rng.normal(0., .4 * noise, len(t))
    humidity = 40. + 45. * activity + rng.normal(0., 1.2 * noise, len(t))
    # Measurement disturbances only; these do not change latent ground truth.
    if rng.random() < .5:
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(60, cfg.duration_s-20))
            n = int(rng.integers(5, 16)); env = np.hanning(n+2)[1:-1]
            acoustic[start:start+n] += rng.uniform(30., 100.) * env
            distance[start:start+n] -= rng.uniform(.2, 1.) * env
    if rng.random() < .2:
        start = int(rng.integers(30, cfg.duration_s-15)); n = int(rng.integers(3,11))
        for x in (acoustic, distance): x[start:start+n] = x[start-1]
    if scenario == 'tof_freeze_90s' and rng.random() < .8:
        start = int(rng.integers(100, 270))
        distance[start:min(start+90, len(t))] = distance[start-1]
    if scenario == 'distance_bias_1cm': distance -= 1.
    acoustic = np.clip(acoustic, 10., 900.)
    humidity = np.clip(humidity, 0., 100.)
    raw = np.stack([acoustic,distance,temperature,humidity],axis=-1)
    risk = np.stack([np.clip((acoustic-65.)/300.,0.,1.),
                     np.clip((40.-distance)/rim,0.,1.),
                     np.clip((temperature-75.)/25.,0.,1.),
                     np.clip((humidity-40.)/45.,0.,1.)],axis=-1)
    return risk.astype('float32'), raw.astype('float32'), height, rim, event

def generate_sensitivity(seed, n, condition, cfg=C):
    hp, gp = CONDITIONS[condition]
    seeds = np.random.default_rng(seed).integers(0, 2**63-1, n, dtype=np.int64)
    rows = [sensitivity_cycle(int(s), hp, gp, cfg) for s in seeds]
    ds = base.Dataset(np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]),
                      np.stack([r[2] for r in rows]), np.array([r[3] for r in rows]),
                      np.array([r[4] for r in rows]), seeds)
    if condition == 'freeboard_error_20pct':
        # A separate RNG leaves trajectories, labels, and raw measurements fixed.
        factor = np.random.default_rng(seed + 910_000).uniform(.8, 1.2, n)
        estimate = ds.freeboard * factor
        ds.risk[:, :, 1] = np.clip((40. - ds.raw[:, :, 1]) / estimate[:, None], 0, 1)
    return ds


def run(original, out, n_cycles=500):
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((original/'run_manifest.json').read_text())
    source_hash = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    if source_hash != cfg['code_sha256']:
        raise ValueError('Base source does not match the original run manifest')
    manifest = {
        'design': 'Post hoc sensitivity analysis specified before this extension run; not preregistered.',
        'base_code_sha256': source_hash,
        'extension_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_seeds': cfg['seeds'], 'test_seeds': [75000+i for i in range(len(cfg['seeds']))],
        'cycles_per_condition_per_seed': n_cycles, 'conditions': CONDITIONS,
        'heating_equation': 'T(t)=25+74*(1-exp(-(t/tau)**heating_power))',
        'growth_equation': 'h(t)=min(r,max(0,h(t-1)+g*b(t-1)**growth_power-c*h(t-1)))',
        'geometry_error': 'r_est=r*U(0.8,1.2) independently per cycle; only normalized height changes',
        'comparison': 'Matched test cycle seeds across conditions; frozen nominal models and thresholds.',
        'physical_validation': False, 'python': platform.python_version(),
        'versions': cfg['versions'],
    }
    (out/'sensitivity_manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, records, settings = [], [], {}
    original_results = pd.read_csv(original/'per_seed_results_v2.csv')
    started=time.time()
    with threadpool_limits(limits=1):
        for i, seed in enumerate(cfg['seeds']):
            saved=json.loads((original/f'configuration_{seed}.json').read_text())
            ds=base.generate(seed, C.n_cycles)
            if base.digest(ds) != saved['dataset_sha256']['nominal_full']:
                raise ValueError('Original data regeneration hash mismatch')
            tr,va,te=base.split_cycles(ds.event,seed+1000)
            train,val,test=ds.take(tr),ds.take(va),ds.take(te)
            methods, selected=base.fit_methods(train,seed)
            assert selected['fixed_weights']==saved['fixed_weights']
            assert selected['tof_trend_alpha_scale']==saved['tof_trend_alpha_scale']
            for name in base.METHODS[7:]:
                assert selected[name]['selected_parameter']==saved[name]['selected_parameter']
            thresholds={}
            for name, fn in methods.items():
                sw=base.sweep(fn(val.risk),val.event)
                j,ok=base.select_threshold(sw)
                threshold=float(sw['threshold'][j])
                assert threshold==saved[name+'_threshold']['value']
                thresholds[name]=threshold
                # Verify frozen models against every original nominal metric.
                old=original_results[(original_results.seed==seed)&(original_results.scenario=='nominal')&(original_results.method==name)].iloc[0]
                repeat=base.sweep(fn(test.risk),test.event,np.array([threshold]))
                for metric in base.METRICS:
                    np.testing.assert_allclose(repeat[metric][0],old[metric],rtol=1e-10,atol=1e-10,equal_nan=True)
            settings[str(seed)]={'settings':selected,'thresholds':thresholds,'data_sha256':{}}
            test_seed=manifest['test_seeds'][i]
            for condition in CONDITIONS:
                current=generate_sensitivity(test_seed,n_cycles,condition)
                assert not np.intersect1d(ds.cycle_seed,current.cycle_seed).size
                assert (current.event>=0).any() and (current.event<0).any()
                settings[str(seed)]['data_sha256'][condition]=base.digest(current)
                for name,fn in methods.items():
                    scores=fn(current.risk)
                    sw=base.sweep(scores,current.event,np.array([thresholds[name]]))
                    row={'seed':seed,'test_seed':test_seed,'scenario':condition,'method':name,
                         'n_event':int((current.event>=0).sum()),'n_nonevent':int((current.event<0).sum())}
                    row.update({k:float(sw[k][0]) for k in base.METRICS})
                    rows.append(row)
                    records.append(base.cycle_records(current,scores,thresholds[name],seed,condition,name))
            pd.DataFrame(rows).to_csv(out/'per_seed_sensitivity.csv',index=False)
            print(f'Sensitivity seed {seed} complete; elapsed {time.time()-started:.1f} s',flush=True)
    df=pd.DataFrame(rows)
    base.summary(df,['scenario','method']).to_csv(out/'summary_sensitivity.csv',index=False)
    pd.concat(records,ignore_index=True).to_csv(out/'cycle_warnings_sensitivity.csv.gz',index=False,compression='gzip')
    (out/'frozen_settings_and_hashes.json').write_text(json.dumps(settings,indent=2))
    print('All original nominal metrics, selected settings, thresholds, and data hashes reproduced.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--original',type=Path,default=Path('results_v2'))
    p.add_argument('--out',type=Path,default=Path('sensitivity_results'))
    p.add_argument('--cycles',type=int,default=500)
    args=p.parse_args()
    if args.cycles<100:raise ValueError('Use at least 100 cycles per condition')
    run(args.original,args.out,args.cycles)
