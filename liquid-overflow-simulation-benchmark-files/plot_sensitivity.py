"""Create the two added manuscript figures from documented data and workflow."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from overflow_study_v2 import METHODS
from sensitivity_study import CONDITIONS

ROOT=Path(__file__).parent
OUT=ROOT/'sensitivity_results'
LABELS=['Fresh nominal','Heating exponent 0.8','Heating exponent 1.2',
        'Growth exponent 1.5','Growth exponent 2.5','Freeboard error ±20%']
SELECTED=[METHODS[1],METHODS[3],METHODS[5],METHODS[8]]
SHORT=['Adaptive','ToF only','Fixed weights','Boosting']


def sensitivity_figure():
    df=pd.read_csv(OUT/'summary_sensitivity.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.9),gridspec_kw={'wspace':.1})
    for k,(ax,metric,title) in enumerate(zip(axes,['detection_rate','false_alarm_rate'],
                                          ['Timely detection (%)','Non-event false alarms (%)'])):
        z=np.array([[100*df[(df.scenario==c)&(df.method==m)][metric+'_mean'].iloc[0]
                     for m in SELECTED] for c in CONDITIONS])
        im=ax.imshow(z,cmap='Blues' if k==0 else 'Oranges',vmin=0,vmax=100 if k==0 else 15,aspect='auto')
        for i in range(6):
            for j in range(4):
                ax.text(j,i,f'{z[i,j]:.1f}',ha='center',va='center',fontsize=11,
                        color='white' if z[i,j]>(65 if k==0 else 10) else 'black')
        ax.set_xticks(range(4),SHORT,rotation=25,ha='right')
        ax.set_yticks(range(6),LABELS if k==0 else ['']*6)
        ax.set_title(title,pad=10,fontweight='bold')
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5,4,1),minor=True)
        ax.set_yticks(np.arange(-.5,6,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.4)
        ax.tick_params(which='minor',length=0)
    fig.subplots_adjust(left=.23,right=.99,bottom=.23,top=.88)
    fig.savefig(OUT/'fig_sensitivity.png',dpi=300)
    fig.savefig(OUT/'fig_sensitivity.svg')
    plt.close(fig)


def workflow_figure():
    fig,ax=plt.subplots(figsize=(8,5.8));ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
    def box(cx,cy,w,h,text):
        ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',
                                   facecolor='#f2f2f2',edgecolor='black',linewidth=1.1))
        ax.text(cx,cy,text,ha='center',va='center',fontsize=10.3,linespacing=1.35)
    def arrow(a,b,label=None):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':'black','lw':1.1})
        if label:
            ax.text((a[0]+b[0])/2+.07,(a[1]+b[1])/2,label,fontsize=8.7,ha='left',va='center')
    box(5,9.15,5.2,1.05,'Generate latent surface and sensor proxies\nDerive first rim-crossing labels')
    box(5,7.4,4.6,.95,'Partition whole nominal cycles\nTraining ~60% | validation ~20% | test ~20%')
    box(2.55,5.25,4.2,1.55,'Training cycles\nInner fitting and settings selection\nRefit learned models on all training cycles')
    box(7.45,5.25,4.2,1.55,'Validation cycles\nSelect each operating threshold\n5% false-alarm budget; 90% detection target')
    box(5,2.9,6.5,1.15,'Apply frozen models and thresholds to test cycles\nOriginal nominal and stress tests; fresh sensitivity tests')
    box(5,1.0,6.5,1.05,'Record first three-sample warning\nReport detection, false alarms, premature warnings and lead')
    arrow((5,8.55),(5,7.93));arrow((4,6.86),(2.55,6.08));arrow((6,6.86),(7.45,6.08))
    arrow((4.72,5.25),(5.28,5.25))
    arrow((2.55,4.42),(3.8,3.52));arrow((7.45,4.42),(6.2,3.52))
    arrow((5,2.25),(5,1.58))
    fig.tight_layout(pad=.3)
    fig.savefig(OUT/'fig_workflow.png',dpi=300)
    fig.savefig(OUT/'fig_workflow.svg')
    plt.close(fig)


if __name__=='__main__':
    sensitivity_figure();workflow_figure()
