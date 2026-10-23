"""Readable manuscript figure from fixed final outputs; no model fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from overflow_study_v2 import METHODS

p=Path(__file__).parent/'results_v2'
c=pd.read_csv(p/'threshold_curves_v2.csv.gz')
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11})
fig,axes=plt.subplots(2,1,figsize=(8,7.5),sharex=True)
short={METHODS[0]:'Equal four channels',METHODS[1]:'Original adaptive rule',
       METHODS[3]:'ToF only',METHODS[5]:'Optimized fixed weights',
       METHODS[7]:'Logistic regression',METHODS[8]:'Gradient boosting'}
for method,label in short.items():
    g=c[c.method==method];items=[];grid=np.linspace(0,1,201)
    for seed,sg in g.groupby('seed'):
        sg=sg.groupby('false_alarm_rate')[['detection_rate','mean_lead_s']].mean().sort_index()
        items.append(pd.DataFrame({'far':grid,**{k:np.interp(grid,sg.index,sg[k]) for k in ['detection_rate','mean_lead_s']}}))
    avg=pd.concat(items).groupby('far').mean().reset_index()
    axes[0].plot(avg.far*100,avg.detection_rate*100,label=label,lw=1.8)
    axes[1].plot(avg.far*100,avg.mean_lead_s,label=label,lw=1.8)
axes[0].set(ylabel='Timely detection (%)',ylim=(0,103),title='Synthetic warning trade-offs')
axes[1].set(ylabel='Conditional mean lead (s)',xlabel='Non-event cycles with warnings (%)')
for ax in axes:
    ax.axvline(5,color='.45',ls=':');ax.set_xlim(0,30);ax.grid(alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=10,frameon=False)
fig.tight_layout(rect=(0,.12,1,1));fig.savefig(p/'fig_manuscript_v2.png',dpi=200);plt.close(fig)
