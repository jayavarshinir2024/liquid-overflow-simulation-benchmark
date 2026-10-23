"""Reproducible, explicitly synthetic overflow-warning benchmark.

Events are first crossings of a simulated vessel rim. The phenomenological
generator is NOT fitted to measured boiling, acoustics, or household sensors.
No claim of physical validation, patentability, or safety certification is made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Config:
    duration_s: int = 360
    horizon_s: int = 60
    trend_lag_s: int = 10
    persistence: int = 3
    n_cycles: int = 1000
    stress_cycles: int = 300
    far_budget: float = .05
    min_detection: float = .90
    train_frame_stride: int = 3
    seed_start: int = 24000

C = Config()
THRESHOLD_QUANTILES = 257
SCENARIOS = ('nominal', 'noise_x3', 'weak_acoustic', 'tof_freeze_90s',
             'distance_bias_1cm', 'faster_foam', 'small_freeboard')
METHODS = ('Equal four channels', 'Original adaptive heuristic', 'Acoustic only',
           'ToF only', 'Acoustic plus ToF', 'Optimized fixed weights',
           'Tuned ToF plus trend', 'Logistic common features', 'Boosting common features')
METRICS = ('detection_rate', 'false_alarm_rate', 'early_warning_rate', 'precision',
           'mean_lead_s', 'median_lead_s', 'any_advance_detection', 'threshold')

@dataclass
class Dataset:
    risk: np.ndarray
    raw: np.ndarray  # frequency Hz, distance cm, temperature proxy C, humidity %
    height: np.ndarray  # latent pre-measurement surface trajectory, capped at rim
    freeboard: np.ndarray
    event: np.ndarray  # first crossing sample, -1 if no crossing in observation
    cycle_seed: np.ndarray

    def take(self, idx):
        return Dataset(*(getattr(self, k)[idx] for k in self.__dataclass_fields__))

def first_crossing(height, rim):
    hit = np.asarray(height) >= np.asarray(rim)[..., None]
    return np.where(hit.any(axis=-1), hit.argmax(axis=-1), -1)

def make_cycle(seed: int, scenario='nominal', cfg=C):
    """Causal latent dynamics; no has-event flag or event time drives sensors."""
    rng = np.random.default_rng(int(seed))
    t = np.arange(cfg.duration_s)
    tau = rng.uniform(65., 100.)
    foamability = rng.uniform(0., 1.)
    collapse = rng.uniform(.006, .018)
    rim = rng.uniform(6., 12.) if scenario != 'small_freeboard' else rng.uniform(3., 6.)
    bulk_temperature = 25. + 74. * (1. - np.exp(-t / tau))
    activity = np.clip((bulk_temperature - 85.) / 14., 0., 1.)
    growth = (.025 + .18 * foamability) * (1.5 if scenario == 'faster_foam' else 1.)
    height = np.zeros(cfg.duration_s)
    for k in range(1, cfg.duration_s):
        height[k] = min(rim, max(0., height[k-1] + growth * activity[k-1] ** 2
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

def generate(seed, n, scenario='nominal', cfg=C):
    # Per-cycle seeds permit reconstruction of an individual recorded warning.
    seeds=np.random.default_rng(seed).integers(0,2**63-1,n,dtype=np.int64)
    rows=[make_cycle(int(s),scenario,cfg) for s in seeds]
    return Dataset(np.stack([r[0] for r in rows]),np.stack([r[1] for r in rows]),
                   np.stack([r[2] for r in rows]),np.array([r[3] for r in rows]),
                   np.array([r[4] for r in rows]),seeds)

def split_cycles(event, seed, fractions=(.6,.2,.2)):
    rng=np.random.default_rng(seed); parts=[[],[],[]]
    for flag in (False,True):
        idx=np.flatnonzero((event>=0)==flag);rng.shuffle(idx)
        a=int(len(idx)*fractions[0]);b=a+int(len(idx)*fractions[1])
        for dest,chunk in zip(parts,(idx[:a],idx[a:b],idx[b:])):dest.extend(chunk)
    return tuple(np.asarray(x,dtype=int) for x in parts)

def lag_change(x, lag=10):
    return x-np.concatenate([np.repeat(x[:,:1],lag,axis=1),x[:,:-lag]],axis=1)

def features(risk, cfg=C):
    return np.concatenate([risk,np.stack([lag_change(risk[:,:,0],cfg.trend_lag_s),
                                         lag_change(risk[:,:,1],cfg.trend_lag_s)],axis=-1)],axis=-1)

def training_frames(ds,cfg=C):
    x=features(ds.risk,cfg)[:,::cfg.train_frame_stride]
    times=np.arange(cfg.duration_s)[::cfg.train_frame_stride][None,:]
    gap=ds.event[:,None]-times
    valid=(ds.event[:,None]<0)|(gap>0)
    labels=(ds.event[:,None]>=0)&(gap>0)&(gap<=cfg.horizon_s)
    return x[valid],labels[valid].astype(int)

def adaptive(r):
    a,h,t,m=np.moveaxis(r,-1,0);p=np.maximum(a,h)
    q=np.stack([.3*(1+2*a),.4*(1+2*h),.2*(.25+.75*p),.1*(.25+.75*p)],axis=-1)
    return 100.*np.sum(q*r,axis=-1)/q.sum(axis=-1)

def candidate_thresholds(scores,event,cfg=C):
    """Score-scale-aware validation grid, including cycle alarm boundaries.

    Integer score grids can disable saturated learned probabilities while
    resolving linear scores adequately. Quantiles avoid that arbitrary scale
    advantage. The grid is approximate, not an exhaustive optimization.
    """
    scores=np.asarray(scores,dtype=float)
    pm=np.minimum.reduce([scores[:,k:scores.shape[1]-cfg.persistence+1+k]
                          for k in range(cfg.persistence)])
    times=np.arange(cfg.persistence-1,scores.shape[1])[None,:]
    valid=(event[:,None]<0)|(times<event[:,None])
    base=np.quantile(pm[valid],np.linspace(0,1,THRESHOLD_QUANTILES))
    bounds=[]
    for i,e in enumerate(event):
        bounds.append(pm[i,valid[i]].max())
        if e>=0:
            early=times[0] < e-cfg.horizon_s
            timely=(times[0]>=e-cfg.horizon_s)&(times[0]<e)
            if early.any():bounds.append(pm[i,early].max())
            if timely.any():bounds.append(pm[i,timely].max())
    values=np.unique(np.concatenate([base,np.asarray(bounds),[scores.min(),scores.max()]]))
    return np.unique(np.concatenate([values,np.nextafter(values,np.inf)]))

def warnings_at_thresholds(scores, thresholds, cfg=C):
    windows=[scores[:,k:scores.shape[1]-cfg.persistence+1+k] for k in range(cfg.persistence)]
    cm=np.maximum.accumulate(np.minimum.reduce(windows),axis=1)
    # searchsorted uses the monotonic cumulative maximum; warning is sample 3.
    first=np.stack([np.searchsorted(row,thresholds,side='left') for row in cm])
    return np.where(first<cm.shape[1],first+cfg.persistence-1,-1)

def sweep(scores,event,thresholds=None,cfg=C):
    if thresholds is None:thresholds=candidate_thresholds(scores,event,cfg)
    wt=warnings_at_thresholds(scores,thresholds,cfg)
    ev=event[:,None];positive=ev>=0
    advance=positive&(wt>=0)&(wt<ev)
    timely=advance&((ev-wt)<=cfg.horizon_s)
    early=advance&~timely
    negative_alarm=(~positive)&(wt>=0)
    npos=max(int((event>=0).sum()),1);nneg=max(int((event<0).sum()),1)
    lead=np.where(timely,ev-wt,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        ml=np.nanmean(lead,axis=0); med=np.nanmedian(lead,axis=0)
    denom=timely.sum(0)+early.sum(0)+negative_alarm.sum(0)
    return dict(detection_rate=timely.sum(0)/npos,false_alarm_rate=negative_alarm.sum(0)/nneg,
                early_warning_rate=early.sum(0)/npos,
                precision=np.divide(timely.sum(0),denom,out=np.full(len(denom),np.nan),where=denom>0),
                mean_lead_s=ml,median_lead_s=med,any_advance_detection=advance.sum(0)/npos,
                threshold=np.asarray(thresholds), warning_times=wt)

def select_threshold(s,cfg=C):
    feasible=(s['false_alarm_rate']<=cfg.far_budget)&(s['detection_rate']>=cfg.min_detection)
    if feasible.any():
        idx=int(np.argmax(np.where(feasible,np.nan_to_num(s['mean_lead_s'],nan=-1),-np.inf)))
    else:
        allowed=s['false_alarm_rate']<=cfg.far_budget
        # Prefer sensitivity, then fewer premature warnings, then longer lead.
        idx=max(np.flatnonzero(allowed),key=lambda j:(s['detection_rate'][j],-s['early_warning_rate'][j],
                 np.nan_to_num(s['mean_lead_s'][j],nan=-1),-s['threshold'][j]))
    return idx,bool(feasible.any())

def quality(s,j,feasible):
    return (int(feasible),float(s['mean_lead_s'][j]) if feasible else float(s['detection_rate'][j]),
            -float(s['false_alarm_rate'][j]),-float(s['early_warning_rate'][j]))

def tune_rule(candidates, scorer, ds,cfg=C):
    best=None
    for candidate in candidates:
        s=sweep(scorer(ds.risk,candidate),ds.event,cfg=cfg);j,ok=select_threshold(s,cfg)
        key=quality(s,j,ok)
        if best is None or key>best[0]:best=(key,candidate)
    return best[1]

def weighted(r,w):return 100.*np.sum(r*np.asarray(w),axis=-1)
def tof_trend(r,pars):
    alpha,scale=pars
    return 100.*np.clip(alpha*r[:,:,1]+(1-alpha)*np.clip(lag_change(r[:,:,1])/scale,0,1),0,1)

def make_estimator(kind,param,seed):
    if kind=='Logistic common features':
        return make_pipeline(StandardScaler(),LogisticRegression(C=param,max_iter=1000,
                             class_weight='balanced',random_state=seed))
    return HistGradientBoostingClassifier(max_depth=param,max_iter=100,learning_rate=.1,
                                         early_stopping=False,l2_regularization=1.,random_state=seed)

def fit_estimator(kind,param,ds,seed,cfg=C):
    model=make_estimator(kind,param,seed);x,y=training_frames(ds,cfg)
    if kind.startswith('Boosting'):model.fit(x,y,sample_weight=compute_sample_weight('balanced',y))
    else:model.fit(x,y)
    return model

def predict(model,r,cfg=C):
    f=features(r,cfg)
    return 100.*model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(r.shape[:2])

def fit_methods(train,seed,cfg=C):
    # Internal tuning never sees the final validation or test cycles.
    fit_idx,_,tune_idx=split_cycles(train.event,seed+1,(2/3,0.,1/3))
    fit,tune=train.take(fit_idx),train.take(tune_idx)
    candidates=[np.asarray(v)/4 for v in product(range(5),repeat=4) if sum(v)==4]
    weights=tune_rule(candidates,weighted,tune,cfg)
    trpars=tune_rule(list(product((0.,.25,.5,.75,1.),(.05,.1,.2))),tof_trend,tune,cfg)
    methods={METHODS[0]:lambda r:weighted(r,(.25,)*4),METHODS[1]:adaptive,
             METHODS[2]:lambda r:100.*r[:,:,0],METHODS[3]:lambda r:100.*r[:,:,1],
             METHODS[4]:lambda r:weighted(r,(.5,.5,0.,0.)),
             METHODS[5]:lambda r,w=weights:weighted(r,w),
             METHODS[6]:lambda r,p=trpars:tof_trend(r,p)}
    settings={'fixed_weights':weights.tolist(),'tof_trend_alpha_scale':list(trpars),
              'inner_fit_n':len(fit_idx),'inner_tune_n':len(tune_idx)}
    for kind,params in [(METHODS[7],(.1,1.,10.)),(METHODS[8],(2,3,5))]:
        best=None
        for param in params:
            model=fit_estimator(kind,param,fit,seed,cfg)
            s=sweep(predict(model,tune.risk,cfg),tune.event,cfg=cfg)
            j,ok=select_threshold(s,cfg);key=quality(s,j,ok)
            if best is None or key>best[0]:best=(key,param)
        param=best[1];model=fit_estimator(kind,param,train,seed,cfg)
        methods[kind]=lambda r,model=model:predict(model,r,cfg)
        settings[kind]={'selected_parameter':param,'candidates':list(params)}
    return methods,settings

def ci95(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return float(stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan

def summary(df,keys):
    rows=[]
    for vals,g in df.groupby(keys,sort=False):
        if not isinstance(vals,tuple):vals=(vals,)
        r=dict(zip(keys,vals));r['n_seeds']=len(g)
        for col in METRICS:
            r[col+'_mean']=g[col].mean();r[col+'_ci95']=ci95(g[col])
            r[col+'_n_defined']=int(g[col].notna().sum())
        rows.append(r)
    return pd.DataFrame(rows)

def paired(df):
    # Three predeclared comparisons x three metrics; Holm controls this family.
    pairs=[(METHODS[1],METHODS[0]),(METHODS[1],METHODS[5]),(METHODS[8],METHODS[7])]
    rows=[]
    for a,b in pairs:
        for metric in ('detection_rate','false_alarm_rate','mean_lead_s'):
            aa=df[(df.scenario=='nominal')&(df.method==a)].set_index('seed')[metric]
            bb=df[(df.scenario=='nominal')&(df.method==b)].set_index('seed')[metric]
            diff=(aa-bb).dropna().to_numpy()
            if len(diff)<2:p=np.nan
            elif np.allclose(diff,0):p=1.
            else:p=float(stats.wilcoxon(diff,method='approx').pvalue)
            rows.append(dict(method_a=a,method_b=b,metric=metric,mean_difference=np.mean(diff) if len(diff) else np.nan,
                             n_pairs=len(diff),p_raw=p))
    out=pd.DataFrame(rows);order=out.p_raw.dropna().sort_values().index;prev=0.
    for rank,idx in enumerate(order):
        prev=max(prev,min(1.,(len(order)-rank)*out.loc[idx,'p_raw']));out.loc[idx,'p_holm']=prev
    return out

def digest(ds):
    h=hashlib.sha256()
    for key in ds.__dataclass_fields__:
        a=np.ascontiguousarray(getattr(ds,key));h.update(key.encode());h.update(a.tobytes())
    return h.hexdigest()

def save_dataset(ds,path):
    np.savez_compressed(path,**{k:getattr(ds,k) for k in ds.__dataclass_fields__})

def cycle_records(ds,scores,threshold,seed,scenario,method,cfg=C):
    wt=warnings_at_thresholds(scores,np.array([threshold]),cfg)[:,0]
    out=pd.DataFrame(dict(seed=seed,scenario=scenario,method=method,cycle_index=np.arange(len(wt)),
                         cycle_seed=ds.cycle_seed,event_time=ds.event,warning_time=wt,
                         freeboard_cm=ds.freeboard,threshold=threshold))
    out['timely']=(out.event_time>=0)&(out.warning_time>=0)&(out.warning_time<out.event_time)&((out.event_time-out.warning_time)<=cfg.horizon_s)
    return out

def run_seed(seed,out,save_data=False,cfg=C):
    ds=generate(seed,cfg.n_cycles,cfg=cfg)
    tr,va,te=split_cycles(ds.event,seed+1000)
    train,val,test=ds.take(tr),ds.take(va),ds.take(te)
    if min((val.event>=0).sum(),(val.event<0).sum(),(test.event>=0).sum(),(test.event<0).sum())<10:
        raise ValueError('Insufficient event or non-event cycles; change generator before evaluation, not after comparing methods.')
    methods,settings=fit_methods(train,seed,cfg)
    settings['split_indices']={'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()}
    chosen={};rows=[];records=[];curves=[];hashes={'nominal_full':digest(ds)}
    for name,fn in methods.items():
        s=sweep(fn(val.risk),val.event,cfg=cfg);j,ok=select_threshold(s,cfg)
        chosen[name]=float(s['threshold'][j]);settings[name+'_threshold']={'value':chosen[name],
            'validation_constraints_met':ok,'validation_detection':float(s['detection_rate'][j]),
            'validation_far':float(s['false_alarm_rate'][j])}
    if save_data:save_dataset(ds,out/f'dataset_{seed}_nominal.npz')
    for si,scenario in enumerate(SCENARIOS):
        # Separate condition seeds: independent tests, not pairwise cycle perturbations.
        current=test if scenario=='nominal' else generate(seed+100_000*(si+1),cfg.stress_cycles,scenario,cfg)
        hashes[scenario]=digest(current)
        if save_data and scenario!='nominal':save_dataset(current,out/f'dataset_{seed}_{scenario}.npz')
        for name,fn in methods.items():
            scores=fn(current.risk)
            s=sweep(scores,current.event,np.array([chosen[name]]),cfg)
            r=dict(seed=seed,scenario=scenario,method=name,n_event=int((current.event>=0).sum()),
                   n_nonevent=int((current.event<0).sum()),validation_feasible=settings[name+'_threshold']['validation_constraints_met'])
            r.update({k:float(s[k][0]) for k in METRICS});rows.append(r)
            records.append(cycle_records(current,scores,chosen[name],seed,scenario,name,cfg))
            if scenario=='nominal':
                sw=sweep(scores,current.event,cfg=cfg)
                curves.extend(dict(seed=seed,method=name,threshold=float(thr),
                     **{k:float(sw[k][j]) for k in ('detection_rate','false_alarm_rate','mean_lead_s')})
                     for j,thr in enumerate(sw['threshold']))
        if seed==cfg.seed_start and scenario=='nominal':
            pieces=[]
            for idx in [int(np.flatnonzero(current.event>=0)[0]),int(np.flatnonzero(current.event<0)[0])]:
                piece=pd.DataFrame(current.raw[idx],columns=['acoustic_hz','distance_cm','temperature_proxy_c','relative_humidity_pct'])
                piece.insert(0,'time_s',np.arange(cfg.duration_s));piece.insert(0,'cycle_seed',current.cycle_seed[idx])
                piece['height_cm']=current.height[idx];piece['freeboard_cm']=current.freeboard[idx];piece['event_time_s']=current.event[idx]
                pieces.append(piece)
            pd.concat(pieces).to_csv(out/'example_traces.csv',index=False,float_format='%.9g')
    settings['dataset_sha256']=hashes
    (out/f'configuration_{seed}.json').write_text(json.dumps(settings,indent=2))
    return rows,pd.concat(records,ignore_index=True),curves

def figures(df,curve,out):
    nom=df[df.scenario=='nominal'];fig,ax=plt.subplots(figsize=(9,5))
    ax.errorbar(100*nom.false_alarm_rate_mean,100*nom.detection_rate_mean,
                xerr=100*nom.false_alarm_rate_ci95,yerr=100*nom.detection_rate_ci95,fmt='none',ecolor='.7',zorder=0)
    for _,r in nom.iterrows():
        ax.scatter(100*r.false_alarm_rate_mean,100*r.detection_rate_mean,label=r.method,s=30)
    ax.axvline(5,color='.5',ls=':');ax.set(xlabel='Non-event cycles with warnings (%)',ylabel='Events warned within 60 s (%)',
       title='Synthetic test performance at validation-selected thresholds')
    ax.legend(fontsize=7,loc='best');fig.tight_layout();fig.savefig(out/'fig_nominal_operating_points.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name,g in curve.groupby('method',sort=False):
        # Data-dependent thresholds differ across seeds. Interpolate each seed's
        # sweep onto a common false-alarm grid for descriptive averaging only.
        grid=np.linspace(0,1,201); seed_curves=[]
        for seed,sg in g.groupby('seed'):
            sg=sg.groupby('false_alarm_rate')[['detection_rate','mean_lead_s']].mean().sort_index()
            seed_curves.append(pd.DataFrame({'false_alarm_rate':grid,
                **{metric:np.interp(grid,sg.index,sg[metric]) for metric in ('detection_rate','mean_lead_s')}}))
        avg=pd.concat(seed_curves).groupby('false_alarm_rate').mean().reset_index()
        axes[0].plot(100*avg.false_alarm_rate,100*avg.detection_rate,label=name)
        axes[1].plot(100*avg.false_alarm_rate,avg.mean_lead_s,label=name)
    axes[0].set(ylabel='Events warned within 60 s (%)',title='Synthetic threshold sweep')
    axes[1].set(ylabel='Mean lead among timely detections (s)',title='Conditional warning lead time')
    for ax in axes:ax.set(xlabel='Non-event cycles with warnings (%)',xlim=(0,30));ax.axvline(5,color='.5',ls=':')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'fig_tradeoffs_v2.png',dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=10);p.add_argument('--out',default='results_v2')
    p.add_argument('--save-data',action='store_true',help='Save all regenerated arrays; may require hundreds of MB')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.seeds<2:raise ValueError('Use at least two seeds for uncertainty summaries.')
    configuration={'config':asdict(C),'scenarios':SCENARIOS,'methods':METHODS,
       'threshold_search':{'quantiles':THRESHOLD_QUANTILES,'cycle_boundaries':True,
                           'post_event_excluded':True,'grid':'data-dependent; strictly-above boundaries included'},
       'seeds':list(range(C.seed_start,C.seed_start+args.seeds)),
       'protocol_status':'Design fixed in source before this evaluation; not externally preregistered.',
       'physical_validation':False,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                   'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__}}
    (out/'run_manifest.json').write_text(json.dumps(configuration,indent=2))
    rows=[];records=[];curves=[];start=time.time()
    with threadpool_limits(limits=1):
        for seed in configuration['seeds']:
            r,rec,cv=run_seed(seed,out,args.save_data);rows.extend(r);records.append(rec);curves.extend(cv)
            pd.DataFrame(rows).to_csv(out/'per_seed_results_v2.csv',index=False)
            print(f'Seed {seed} complete; elapsed {time.time()-start:.1f} s',flush=True)
    df=pd.DataFrame(rows);su=summary(df,['scenario','method'])
    su.to_csv(out/'summary_v2.csv',index=False)
    su[su.scenario=='nominal'].to_csv(out/'table_main_v2.csv',index=False)
    su[su.scenario!='nominal'].to_csv(out/'table_stress_v2.csv',index=False)
    paired(df).to_csv(out/'paired_tests_holm_v2.csv',index=False)
    pd.concat(records).to_csv(out/'cycle_warnings_v2.csv.gz',index=False,compression='gzip')
    curve=pd.DataFrame(curves);curve.to_csv(out/'threshold_curves_v2.csv.gz',index=False,compression='gzip')
    figures(su,curve,out)
    print(su[su.scenario=='nominal'][['method','detection_rate_mean','false_alarm_rate_mean','early_warning_rate_mean','mean_lead_s_mean']].to_string(index=False),flush=True)

if __name__=='__main__':main()
