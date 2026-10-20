"""FigS10: locus decisions and model summaries from S38/S39."""
from pathlib import Path
import numpy as np
import pandas as pd
from common.reference_pdf import ReferencePage,load_table,nearest,small,marker_center

MODELS=['hEvo-base','hEvo-large','evo2_1b_base','evo2_7b_base','NT_500m_MS','NT_2.5b_1kG','NT_2.5b_MS','NT_500m_1kG','NTv3-100M-pre','NTv3-8M-pre','NTv3-650M-pre','GPN-promoter','GPN-Star-P','GPN-MSA','GPN-Star-M','GPN-Star-V','Mistral-DNA-17M','GENA-LM-base','Mistral-DNA-422M','Caduceus-PS','HyenaDNA_tiny_1k','Mistral-DNA-138M','AIDO.DNA-300M','Caduceus-PH','GENA-LM-large','GROVER','SpeciesLM','DNABERT2','GENERator-1.2B','GENERator-3B','Genos-1.2B']
SELECTED=(.701961,.360784,.117647)
ALTERNATIVE=(.85098,.870588,.894118)

def render(tables,out):
    p=ReferencePage(Path(__file__).parent/'assets/reference.pdf')
    loci=[t['text'] for t in p.texts if t['text'].startswith('rs')]
    s38=load_table(tables,38).set_index('model_id').reindex(MODELS)
    s39=load_table(tables,39)
    chosen=s39[s39.model_id.isin(MODELS)&s39.variant_id.isin(loci)]
    matrix=chosen.pivot(index='variant_id',columns='model_id',values='directional_correct').reindex(index=loci,columns=MODELS)
    if matrix.isna().any().any():raise ValueError('S39 is missing a displayed model/locus decision')
    cells=[q for q in p.paths if q['paint']==b'f*' and q['x']>306 and 277<q['y']<720]
    xs=sorted(set(round(q['x'],3) for q in cells));ys=sorted(set(round(q['y'],3) for q in cells),reverse=True)
    if len(cells)!=len(MODELS)*len(loci):raise ValueError('Reference tile geometry changed')
    for q in cells:p.color(q,SELECTED if int(matrix.iloc[nearest(ys,q['y']),nearest(xs,q['x'])]) else ALTERNATIVE)
    pref=s38.accuracy.to_numpy(float)*100
    ch=[]
    for model in MODELS:
        v=chosen[chosen.model_id.eq(model)].set_index('variant_id').reindex(loci)[['selected_logprob','other_logprob']].to_numpy(float)
        means=v.mean(0);overall=v.mean();between=len(v)*np.sum((means-overall)**2)
        within=np.sum((v-means)**2);ch.append(between/(within/(v.size-2)))
    # Recompute every lollipop stem and marker from S38.
    base=212.5984;scale=(269.2913-base)/70
    stems=[q for q in p.paths if q['paint']==b'S' and q['bbox'][2]-q['bbox'][0]<.01 and abs(q['bbox'][1]-base)<.01 and q['x']>306]
    for q in stems:
        j=nearest(xs,q['x']);top=base+pref[j]*scale;oldtop=q['bbox'][3]
        p.transform(q,lambda x,y,b=base,o=oldtop,v=top:(x,b+(y-b)/(o-b)*(v-b)))
    marks=[q for q in p.paths if q['paint']==b'B*' and small(q) and q['x']>306 and 212<q['y']<271]
    for q in marks:
        j=nearest(xs,q['x']);stem=stems[j]
        p.transform(q,lambda x,y,v=base+pref[j]*scale,o=stem['bbox'][3]:(x,y+v-o))
    for t in p.texts:
        if abs(t['y']-202.3937)<.02 and t['x']>306:p.set_text(t,f'{pref[nearest(xs,t["x"]+3)]:.0f}')
        if t['text'] in ('1','2','3') and t['x']>306 and 259<t['y']<271:p.set_text(t,'',align='left')
    rank_style=next(t for t in p.texts if t['text'] in ('1','2','3') and t['x']>306 and 259<t['y']<271)
    rank_offset=rank_style['y']-stems[0]['bbox'][3]
    ranks=pd.Series(pref).rank(ascending=False,method='min').astype(int)
    for j,rank in enumerate(ranks):
        if rank<=3:p.add_text(xs[j]+2.2677,base+pref[j]*scale+rank_offset,rank,size=rank_style['size'],font=rank_style['font'],color=rank_style['fill'])
    # The reference CH grid has two decades per 14.5782 pt.
    for q in p.paths:
        if q['paint']==b'B*' and small(q) and q['x']>306 and 138<q['y']<192:
            j=nearest(xs,q['x']);cy=marker_center(q)[1]
            target=175.3431+np.log10(ch[j])*7.2891
            p.transform(q,lambda x,y,v=target,o=cy:(x,y+v-o))
    out=Path(out);src=out/'source_data';src.mkdir(parents=True,exist_ok=True)
    matrix.rename_axis('variant_id').to_csv(src/'S10_locus_preferences.csv')
    pd.DataFrame({'model':MODELS,'selected_allele_preference_pct':pref,'preference_rank':ranks,'raw_score_CH':ch}).to_csv(src/'S10_model_summaries.csv',index=False)
    return p.finish(out)
