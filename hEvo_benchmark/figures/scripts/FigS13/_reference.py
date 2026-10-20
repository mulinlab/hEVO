"""FigS13: S43 measured resources and S49 available/common task ranks."""
from pathlib import Path
import numpy as np
import pandas as pd
from common.reference_pdf import ReferencePage,load_table,nearest,small
MODELS=['hEvo-large','hEvo-base','evo2_7b_base','evo2_1b_base','NT_2.5b_1kG','NT_2.5b_MS','NT_500m_MS','NTv3-650M-pre','NTv3-100M-pre','NT_500m_1kG','NTv3-8M-pre','GPN-Star-M','GPN-Star-P','GPN-Star-V','GPN-MSA','GPN-promoter','GENA-LM-large','SpeciesLM','Genos-1.2B','DNABERT2','AIDO.DNA-300M','DNABERT','GROVER','Mistral-DNA-422M','GENA-LM-base','GENERator-3B','GENERator-1.2B','Mistral-DNA-17M','Caduceus-PS','Caduceus-PH','HyenaDNA_tiny_1k','Mistral-DNA-138M']
METRICS=['params_m','latency_ms','throughput_bps','vram_mb']


def render(tables,out):
    p=ReferencePage(Path(__file__).parent/'assets/reference.pdf');s=load_table(tables,43).set_index('model').reindex(MODELS)
    # Keep source tables untouched while normalizing the exported family label.
    s['family']=s['family'].replace({'hEvo':'hEVO'})
    r=load_table(tables,49);eligible=r[r.model.isin(MODELS)]
    counts=eligible.groupby('task_unit').model.nunique();common=counts[counts.eq(len(MODELS))].index
    available=eligible.groupby('model').task_rank.mean().reindex(MODELS)
    shared=eligible[eligible.task_unit.isin(common)].groupby('model').task_rank.mean().reindex(MODELS)
    ys=[748.6299-i*12.47244 for i in range(len(MODELS))]
    # The log transforms are measured from the reference's labeled axis ticks.
    transforms=[lambda v:124.9583+np.log10(v)*(158.7402-124.9583)/4,
                lambda v:201.1578+np.log10(v)*(238.1102-201.1578)/3,
                lambda v:279.5740+np.log10(v/1000)*(314.0235-279.5740)/3,
                lambda v:362.5854+np.log10(v/200)*(380.5903-362.5854)]
    for t in p.texts:
        if 351<t['y']<751 and t['x']>160 and t['x']<430 and t['text'].replace('.','').replace(',','').isdigit():
            row=nearest(ys,t['y']+2.4094);j=nearest([181,260,342,420],t['x']);value=float(s.iloc[row][METRICS[j]])
            if j==2:value/=1000
            value_text=f'{value:,.0f}' if j==3 else f'{value:.2f}' if j==1 else f'{value:,.1f}'
            p.set_text(t,value_text,align='right')
    for q in p.paths:
        if q['paint']==b'B*' and small(q) and 350<q['y']<753:
            row=nearest(ys,q['y'])
            if 118<q['x']<400:
                j=nearest([139,218,297,377],q['x']);p.move(q,x=transforms[j](float(s.iloc[row][METRICS[j]])))
            elif q['x']>450:
                value=available.iloc[row] if sum(q['fill'])>2.9 else shared.iloc[row]
                p.move(q,x=464.7960+(value-5)*(500.0143-464.7960)/10)
        elif q['paint']==b'S' and q['x']>450 and 350<q['y']<753 and q['bbox'][3]-q['bbox'][1]<.01 and q['bbox'][2]-q['bbox'][0]<30:
            row=nearest(ys,q['y']);a=464.7960+(available.iloc[row]-5)*3.52183;b=464.7960+(shared.iloc[row]-5)*3.52183
            p.line(q,min(a,b),max(a,b))
    # Retain the original label but make it reflect the current common-task intersection.
    for t in p.texts:
        if t['text']=='Common 35 tasks':p.set_text(t,f'Common {len(common)} tasks',align='left')
    src=Path(out)/'source_data';src.mkdir(parents=True,exist_ok=True)
    s.assign(available_task_mean_rank=available,common_task_mean_rank=shared,n_common_tasks=len(common)).reset_index().to_csv(src/'S13_resources_and_ranks.csv',index=False)
    pd.DataFrame({'task_unit':common}).to_csv(src/'S13_common_tasks.csv',index=False)
    return p.finish(out)
