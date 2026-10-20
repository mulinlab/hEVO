"""FigS11: task MCC and within-corpus SW minus NoSW from S41."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common.reference_pdf import ReferencePage,load_table,nearest,small
TASKS=['H2AFZ','H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me2','H3K4me3','H3K9ac','H3K9me3','H4K20me1','enhancers','enhancers_types','promoter_all','promoter_no_tata','promoter_tata','splice_sites_acceptors','splice_sites_donors','splice_sites_all']
CORPORA=['PopReduce_60K','1000G_3K','UKB_500K']
GROUPS=[TASKS[:10],TASKS[10:15],TASKS[15:]]

def render(tables,out):
    p=ReferencePage(Path(__file__).parent/'assets/reference.pdf');d=load_table(tables,41)
    v=np.array([[d.loc[d.training_dataset.eq(c)&d.windowing.eq(w)&d.dataset.eq(t),'metric_mean'].iloc[0] for t in TASKS] for c in CORPORA for w in ['NoSW','SW']])
    span=v.max(0)-v.min(0);z=np.divide(v-v.min(0),span,out=np.zeros_like(v),where=span>0)
    rects=[q for q in p.paths if q['paint']==b'f*']
    xs=sorted(set(round(q['x'],3) for q in rects));ys=sorted(set(round(q['y'],3) for q in rects),reverse=True)
    # Exact vector-fill swatches, interpolated on the within-task color scale.
    ramps=json.loads((Path(__file__).parent/'assets/heatmap_palette.json').read_text())
    for q in rects:
        i,j=nearest(ys,q['y']),nearest(xs,q['x'])
        knots=np.array([r[0] for r in ramps[j]['swatches']]);colors=np.array([r[1] for r in ramps[j]['swatches']])
        p.color(q,[np.interp(z[i,j],knots,colors[:,c]) for c in range(3)])
    for t in p.texts:
        if 610<t['y']<703 and t['x']>155 and t['text'].startswith('0.'):
            i,j=nearest(ys,t['y']+2),nearest(xs,t['x']+7);strong=z[i,j]>.88
            p.set_text(t,f'{v[i,j]:.3f}',bold=z[i,j]==1.0,color=(1,1,1) if strong else (.066667,)*3)
    differences=v[1::2]-v[::2];centers=[518.7402,487.5591,456.3780];axis_left=[144.5669,289.1339,433.7008];width=96.3780
    means=[]
    for g,tasks in enumerate(GROUPS):
        left=axis_left[g];inds=[TASKS.index(t) for t in tasks]
        for row in range(3):
            marks=sorted([q for q in p.paths if q['paint']==b'B*' and small(q) and left-6<q['x']<left+width+3 and abs(q['y']-centers[row])<7],key=lambda q:q['index'])
            if len(marks)!=len(tasks)+1:raise ValueError(f'Unexpected dot geometry: group {g}, corpus {row}')
            for q,value in zip(marks[:-1],differences[row,inds]):p.move(q,x=left+(value+.04)/.14*width)
            mean=float(differences[row,inds].mean());p.move(marks[-1],x=left+(mean+.04)/.14*width)
            text=next(t for t in p.texts if t['text'].startswith('+') and abs(t['y']-(centers[row]-2.4095))<.1 and left<t['x']<left+width+30)
            p.set_text(text,f'{mean:+.3f}',align='left')
            means.append({'corpus':CORPORA[row],'task_group':g,'MCC_difference':mean})
    out=Path(out);src=out/'source_data';src.mkdir(parents=True,exist_ok=True)
    d.to_csv(src/'S11_task_MCC.csv',index=False)
    pd.DataFrame(differences,index=CORPORA,columns=TASKS).rename_axis('training_corpus').to_csv(src/'S11_SW_minus_NoSW.csv')
    pd.DataFrame(means).to_csv(src/'S11_group_differences.csv',index=False)
    return p.finish(out)
