#!/usr/bin/env python3
"""Independent ClinVar subset review figure; does not change six-task FigS7."""

import tempfile
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BENCHMARK=Path(__file__).resolve().parents[3]
ORDER=['hEvo-base','hEvo-large','evo2_7b_base','evo2_1b_base','NTv3-650M-pre','NT_2.5b_MS','NTv3-100M-pre','NTv3-8M-pre','NT_500m_MS','GPN-promoter','GPN-Star-M','GPN-Star-P','GPN-Star-V','GPN-MSA','SpeciesLM','AIDO.DNA-300M','Caduceus-PS','Genos-1.2B','GENA-LM-large','Caduceus-PH','DNABERT2','GENERator-3B','GENERator-1.2B','HyenaDNA_tiny_1k','Mistral-DNA-138M','Mistral-DNA-17M','GENA-LM-base','Mistral-DNA-422M','GROVER']
NAMES={'hEvo-base':'hEVO-base','hEvo-large':'hEVO-large','evo2_7b_base':'Evo2-7B','evo2_1b_base':'Evo2-1B','NTv3-650M-pre':'NTv3-650M','NT_2.5b_MS':'NTv1-2.5B-MS','NTv3-100M-pre':'NTv3-100M','NTv3-8M-pre':'NTv3-8M','NT_500m_MS':'NTv2-500M-MS','GPN-promoter':'GPN-Promoter','HyenaDNA_tiny_1k':'HyenaDNA-1K'}
COL={'hEVO':'#B35C1E','Evo 2':'#0E7C86','NT':'#65717F','GPN':'#0B2E83','Other models':'#B8BEC8'}
def family(m):
    if m.startswith('hEvo'):return 'hEVO'
    if m.startswith('evo2'):return 'Evo 2'
    if m.startswith('NT'):return 'NT'
    if m.startswith('GPN'):return 'GPN'
    return 'Other models'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,default=BENCHMARK/'reports/clinvar_gpn_subsets/metrics.csv')
    p.add_argument('--output-dir',type=Path,default=BENCHMARK/'figures/ClinVar_subsets_GPN_rebuilt')
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    rebuilt=pd.read_csv(args.metrics)
    assert len(rebuilt)==8 and not rebuilt.duplicated(['model','dataset']).any()
    assert rebuilt.formal_table_gate.eq('PASS').all()
    tables={}
    for subset,n in [('coding',24),('noncoding',25)]:
        table=pd.read_csv(next((BENCHMARK/'tables').glob(f'S{n}_*.csv')))
        replacement=rebuilt.loc[rebuilt.dataset==subset]
        table=table.loc[~table.model.isin(replacement.model)]
        table=pd.concat([table,replacement],ignore_index=True).set_index('model')
        tables[subset]=table.loc[ORDER].copy()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,4,figsize=(15.5,10),sharey=True)
    fig.subplots_adjust(left=.17,right=.97,top=.90,bottom=.24,wspace=.35)
    rendered=[];text_artists=[]
    for col,(subset,metric) in enumerate([('coding','auprc'),('coding','auroc'),('noncoding','auprc'),('noncoding','auroc')]):
        ax=axes[col];table=tables[subset];rank=table[metric+'_mean'].rank(ascending=False,method='min')
        for i,m in enumerate(ORDER):
            r=table.loc[m];value=float(r[metric+'_mean']);lo=float(r[metric+'_ci_lower']);hi=float(r[metric+'_ci_upper']);color=COL[family(m)]
            assert 0<=lo<=hi<=1 and 0<=value<=1
            if m.startswith('hEvo'):ax.axhspan(i-.40,i+.40,color='#F7EDE2',zorder=0)
            ax.hlines(i,lo,hi,color=color,lw=1.6,zorder=3)
            ax.scatter([value],[i],s=22,color=color,zorder=4)
            # Per-row references retain each model's own evaluated prevalence.
            baseline=float(r['baseline']) if metric=='auprc' else .5
            ax.vlines(baseline,i-.34,i+.34,colors='#848484',linestyles='dashed',lw=.75,zorder=1)
            a=ax.text(1.035,i,f'{value:.3f}',ha='left',va='center',fontsize=8,clip_on=False);text_artists.append(a)
            if rank[m]<=3:
                a=ax.text(-.055,i,str(int(rank[m])),ha='right',va='center',fontsize=8,color=color,fontweight='bold',clip_on=False);text_artists.append(a)
            rendered.append(dict(model=m,display_name=NAMES.get(m,m),subset=subset,metric=metric,value=value,ci_lower=lo,ci_upper=hi,rank=int(rank[m]),baseline=baseline,num_samples=int(r.num_samples),num_positive=int(r.num_positive),num_negative=int(r.num_negative),coverage=float(r.num_samples)/(144285 if subset=='coding' else 119907),origin='rebuilt' if m in set(rebuilt.model) else 'historical'))
        ax.set_xlim(0,1);ax.set_ylim(len(ORDER)-.3,-1.0);ax.set_xticks([0,.5,1]);ax.xaxis.tick_top();ax.set_xlabel(metric.upper(),labelpad=10)
        ax.set_yticks(range(len(ORDER)));ax.tick_params(axis='y',length=0,pad=28)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(axis='x',length=3,color='#999999');ax.grid(axis='x',color='#E8E8E8',lw=.6,zorder=0)
    axes[0].set_yticklabels([NAMES.get(m,m) for m in ORDER])
    fig.text(.17,.94,'a',fontweight='bold',fontsize=14);fig.text(.59,.94,'b',fontweight='bold',fontsize=14)
    fig.text(.355,.195,'ClinVar coding\n(P = 70,787, N = 73,498)',ha='center',va='top',fontsize=10)
    fig.text(.795,.195,'ClinVar noncoding\n(P = 13,793, N = 106,114 before model exclusions)',ha='center',va='top',fontsize=10)
    handles=[Line2D([],[],color=c,marker='o',linestyle='',label=f) for f,c in COL.items()]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.035,.105),ncol=5,frameon=False,columnspacing=2.5)
    notes=[
        'GPN-Star-M/P/V: rebuilt from corrected 2026-09-12 all scores. Coding: 144,285 (100%); noncoding: 118,990 (99.24%), P = 13,763, N = 105,227.',
        'GPN-MSA: rebuilt from the 2026-09-28 masked-marginal rerun. Coding: 144,285 (100%); noncoding: 119,907 (100%). Other models: historical results.',
        'GPN-MSA uses the newer scoring protocol; historical S23 all results use the older official-MSA protocol and are not directly comparable.',
        'Whiskers: 95% CIs (rebuilt rows: 1,000 row bootstrap replicates, seed 42). Dashed row segments: evaluated prevalence (AUPRC) or 0.5 (AUROC).',
        '1, 2, 3: rank among the models displayed in each metric. Ordering and model roster follow the supplied review PDF.'
    ]
    for i,note in enumerate(notes):text_artists.append(fig.text(.035,.102-i*.019,note,fontsize=7.7,va='top'))
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
    for artist in text_artists+axes[0].get_yticklabels()+[a.xaxis.label for a in axes]:
        box=artist.get_window_extent(renderer);bounds.append([float(v) for v in box.bounds])
        if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:raise ValueError(f'Text outside figure: {artist.get_text()}')
    stem='ClinVar_coding_noncoding_GPN_rebuilt'
    for ext in ['pdf','png','svg']:fig.savefig(out/f'{stem}.{ext}',dpi=220,facecolor='white')
    pd.DataFrame(rendered).to_csv(out/'source_data.csv',index=False)
    plt.close(fig)
if __name__=='__main__':main()
