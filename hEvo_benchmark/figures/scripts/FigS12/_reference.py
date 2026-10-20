"""FigS12: four hEvo models, using the local task and summary S tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pypdf.generic import FloatObject
from common.reference_pdf import ReferencePage,load_table,nearest,small,colored
MODELS=['hEvo-large','hEvo-base','hEvo-open-large','hEvo-open-base']
DRAW_ORDER=[3,2,1,0]
NT=['H2AFZ','H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me2','H3K4me3','H3K9ac','H3K9me3','H4K20me1','enhancers','enhancers_types','promoter_all','promoter_no_tata','promoter_tata','splice_sites_acceptors','splice_sites_donors','splice_sites_all']
GB=['demo_coding_vs_intergenomic_seqs','demo_human_or_worm','human_enhancers_cohn','human_enhancers_ensembl','human_ensembl_regulatory','human_nontata_promoters','human_ocr_ensembl']
CLASSES=['ELS','PLS','Insulator','Silencer','URE']


def norm_rank(values):
    values=values.dropna()
    return (1-(values.rank(ascending=False,method='min')-1)/len(values)).reindex(MODELS).to_numpy(float)


def set_vertices(p,q,points):
    inverse=np.linalg.inv(q['ctm']);j=0
    for i in q['indices']:
        a,op=p.ops[i]
        if op in (b'm',b'l'):
            xy=(inverse@np.array([*points[j],1]))[:2]
            p.ops[i]=([FloatObject(v) for v in xy],op);j+=1


def update_interval(p,paths,mean,low,high,tx):
    """Update a mean marker plus the reference's three error-bar strokes."""
    marks=[q for q in paths if q['paint']==b'B*']
    lines=[q for q in paths if q['paint']==b'S']
    for q in marks:p.move(q,x=tx(mean))
    if lines:
        p.line(lines[0],tx(low),tx(high))
        if len(lines)==3:
            p.move(lines[1],x=tx(low));p.move(lines[2],x=tx(high))


def update_runs(p,paths,values,tx):
    """Retain five jitter positions, observed range, and the model mean."""
    marks=sorted([q for q in paths if q['paint']==b'B*'],key=lambda q:q['index'])
    lines=[q for q in paths if q['paint']==b'S']
    if len(marks)!=6:raise ValueError(f'Expected five runs plus one mean; got {len(marks)}')
    for q,value in zip(marks[:5],values):p.move(q,x=tx(value))
    p.move(marks[-1],x=tx(float(np.mean(values))))
    for q in lines:p.line(q,tx(min(values)),tx(max(values)))


def render(tables,out):
    p=ReferencePage(Path(__file__).parent/'assets/reference.pdf')
    tables={n:load_table(tables,n) for n in [17,18,19,20,21,23,26,27,28,29,30,33,34,35,36,37,38]}
    # S21 contains two open-large runs. The reference uses the later run 24.
    sea=tables[21].copy();sea['_run']=sea.run_id.str.extract(r'(\d+)$').astype(int)
    sea=sea.sort_values('_run').drop_duplicates(['model','task'],keep='last')
    tables[21]=sea
    source=[]
    gb=tables[18].groupby('model').filter(lambda d:len(d)==7).groupby('model').mcc_mean.mean()
    nt=tables[17].groupby('model').mcc_mean.mean()
    ure=(tables[19].groupby('model').mcc_mean.mean()+tables[20].groupby('model').per_class_f1.mean())/2
    gwas=(tables[26].groupby('model').auprc_mean.mean()+tables[27].groupby('model').auprc_mean.mean())/2
    ancestry=(tables[33].set_index('model').rank_score+tables[34].set_index('model').classification_mcc+tables[35].loc[tables[35].shot.eq(60)].set_index('model').pearson_r_mean)/3
    ds=sea.groupby('model').auroc.mean()
    radar=[np.array([norm_rank(v) for v in [gb,nt,ure,ds]]).T,
           np.array([norm_rank(v) for v in [tables[23].groupby('model').auprc_mean.mean(),tables[30].groupby('model').auprc_mean.mean(),gwas,tables[28].groupby('model').auprc_mean.mean(),tables[29].groupby('model').auprc_mean.mean()]]).T,
           np.array([norm_rank(v) for v in [tables[37].groupby('model').mcc_mean.mean(),tables[36].groupby('model').mcc_mean.mean(),ancestry,tables[38].groupby('model_id').accuracy.mean()]]).T]
    # a: preserve all strokes/markers, update radar endpoints from source ranks.
    for panel,(cx,count) in enumerate([(113.3858,4),(306.1417,5),(498.8976,4)]):
        lines=[q for q in p.paths if q['paint']==b'S' and q['width']==.85 and abs(q['x']-cx)<55 and 660<q['y']<775]
        marks=[q for q in p.paths if q['paint']==b'B*' and abs(q['x']-cx)<55 and 660<q['y']<775 and colored(q)]
        if len(lines)!=count*4 or len(marks)!=count*4:raise ValueError('Reference radar geometry changed')
        for order,model in enumerate(DRAW_ORDER):
            desired=[];original=[]
            for j in range(count):
                angle=np.pi/2-j*2*np.pi/count
                r=radar[panel][model,j]*44.7874
                desired.append((cx+r*np.cos(angle),720+r*np.sin(angle)))
                q=lines[order*count+j];a,_=p.ops[q['indices'][0]]
                original.append((q['ctm']@np.array([float(a[0]),float(a[1]),1]))[:2])
                source.append({'panel':'a','model':MODELS[model],'metric':[['Genomic','NT','URE','DeepSEA'],['ClinVar','De novo','GWAS causal','Rare variants','MPRA'],['HAR','HAQER','Ancestry','Adaptation']][panel][j],'value':radar[panel][model,j]})
            for j in range(count):
                set_vertices(p,lines[order*count+j],[desired[j],desired[(j+1)%count]])
                q=marks[order*count+j];dx,dy=np.array(desired[j])-original[j]
                p.transform(q,lambda x,y,dx=dx,dy=dy:(x+dx,y+dy))
    # b: all 25 source-backed MCC columns, normalized only for cell color.
    gbv=tables[18].pivot(index='model',columns='dataset',values='mcc_mean').reindex(index=MODELS,columns=GB)
    ntv=tables[17].pivot(index='model',columns='dataset',values='mcc_mean').reindex(index=MODELS,columns=NT)
    values=np.column_stack([gbv.to_numpy(),ntv.to_numpy()]);span=values.max(0)-values.min(0)
    z=np.divide(values-values.min(0),span,out=np.zeros_like(values),where=span>0)
    cells=[q for q in p.paths if q['paint']==b'f*' and 503<q['y']<550 and q['x']>116]
    xs=sorted(set(round(q['x'],3) for q in cells));ys=sorted(set(round(q['y'],3) for q in cells),reverse=True)
    ramps=json.loads((Path(__file__).parent/'assets/heatmap_palette.json').read_text())
    for q in cells:
        row,j=nearest(ys,q['y']),nearest(xs,q['x'])
        knots=np.array([r[0] for r in ramps[j]['swatches']]);colors=np.array([r[1] for r in ramps[j]['swatches']])
        p.color(q,[np.interp(z[row,j],knots,colors[:,c]) for c in range(3)])
    for t in p.texts:
        if 503<t['y']<551 and t['x']>116 and t['text'].startswith('0.'):
            row,j=nearest(ys,t['y']+2),nearest(xs,t['x']+7)
            p.set_text(t,f'{values[row,j]:.3f}',bold=values[row,j]==values[:,j].max(),color=(1,1,1) if z[row,j]>.82 else (.066667,)*3)
    for row,m in enumerate(MODELS):
        for j,task in enumerate(GB+NT):source.append({'panel':'b','model':m,'metric':task,'value':values[row,j]})
    # c: F1 means across the five negative-control datasets, same task order.
    f1=tables[20].groupby(['model','class_label']).per_class_f1.mean().unstack().reindex(index=MODELS,columns=CLASSES).to_numpy()
    xs_c=[45.35433,82.91339,120.4724,158.0315,195.5906]
    ty=lambda value:376.7898+(value-.4)*(442.2047-376.7898)/.6
    lines=[q for q in p.paths if q['paint']==b'S' and colored(q) and q['width']>.6 and 45<q['x']<196 and 370<q['y']<445]
    marks=[q for q in p.paths if q['paint']==b'B*' and colored(q) and 44<q['x']<197 and 370<q['y']<445]
    if len(lines)!=16 or len(marks)!=20:raise ValueError('Reference F1 geometry changed')
    for order,model in enumerate(DRAW_ORDER):
        original=[]
        for j in range(4):
            q=lines[order*4+j];a,_=p.ops[q['indices'][0]];original.append((q['ctm']@np.array([float(a[0]),float(a[1]),1]))[:2])
            if j==3:
                a,_=p.ops[q['indices'][1]];original.append((q['ctm']@np.array([float(a[0]),float(a[1]),1]))[:2])
            set_vertices(p,q,[(xs_c[j],ty(f1[model,j])),(xs_c[j+1],ty(f1[model,j+1]))])
        for j in range(5):
            dx,dy=np.array([xs_c[j],ty(f1[model,j])])-original[j]
            p.transform(marks[order*5+j],lambda x,y,dx=dx,dy=dy:(x+dx,y+dy))
            source.append({'panel':'c','model':MODELS[model],'metric':CLASSES[j],'value':f1[model,j]})
    # d: URE run ranges and DeepSEA task AUROC from the selected latest run.
    ys_d=[432.7087,415.7008,398.6929,381.6850]
    for row,m in enumerate(MODELS):
        vals=tables[19].loc[tables[19].model.eq(m)].sort_values('run').mcc_mean.to_numpy()
        paths=[q for q in p.paths if 310<q['x']<382 and abs(q['y']-ys_d[row])<3.2 and colored(q)]
        update_runs(p,paths,vals,lambda v:311.811+(v-.6)/.4*68.0315)
        source.append({'panel':'d','model':m,'metric':'URE MCC','value':float(vals.mean())})
    deep=sea.pivot(index='model',columns='task',values='auroc').reindex(index=MODELS,columns=['DHS','HM','TF']).to_numpy()
    for q in p.paths:
        if q['paint']==b'f*' and 378<q['y']<438 and q['x']>394:
            row=nearest(ys_d,q['y']);j=nearest([418,477,537],q['x']);x0=q['bbox'][0]
            p.line(q,x0,x0+56.6929*deep[row,j])
    for t in p.texts:
        if 378<t['y']<438 and t['x']>394 and t['text'].startswith('0.'):
            row=nearest(ys_d,t['y']+2);j=nearest([416,475,535],t['x'])
            p.set_text(t,f'{deep[row,j]:.3f}',bold=deep[row,j]==deep[:,j].max())
    for row,m in enumerate(MODELS):
        for j,task in enumerate(['DHS','HM','TF']):source.append({'panel':'d','model':m,'metric':task,'value':deep[row,j]})
    # e: six AUPRC values with S-table confidence bounds.
    ys_e=[273.9685,256.9606,239.9528,222.9449]
    lefts=[116.2205,193.3228,270.4252,347.5276,424.6299,501.7323]
    limits=[(.8,.94),(.64,.85),(.075,.205),(.35,.65),(.65,.8),(.22,.35)]
    for j,n in enumerate([23,30,26,27,28,29]):
        d=tables[n].set_index('model').reindex(MODELS);left=lefts[j];lo,hi=limits[j]
        tx=lambda v,left=left,lo=lo,hi=hi:left+(v-lo)/(hi-lo)*51.3071
        for row,m in enumerate(MODELS):
            r=d.loc[m];paths=[q for q in p.paths if left-8<q['x']<left+59 and abs(q['y']-ys_e[row])<3.2 and colored(q)]
            update_interval(p,paths,r.auprc_mean,r.auprc_ci_lower,r.auprc_ci_upper,tx)
            text=next(t for t in p.texts if t['text'].startswith('0.') and abs(t['y']-(ys_e[row]-1.8425))<.1 and left+51<t['x']<left+65)
            p.set_text(text,f'{r.auprc_mean:.3f}',bold=r.auprc_mean==d.auprc_mean.max(),align='right')
            source.append({'panel':'e','model':m,'metric':f'S{n} AUPRC','value':r.auprc_mean,'low':r.auprc_ci_lower,'high':r.auprc_ci_upper})
    # f: ancestry MCC, 60-shot Pearson r, HAR/HAQER run distributions, preference.
    ys_f=[120.8976,103.8898,86.88189,69.87402]
    lefts_f=[116.2205,207.4961,298.7717,390.0472,481.3228]
    limits_f=[(.94,1),(.88,.98),(.6,.92),(.6,.92),(0,100)]
    fvalues=np.zeros((4,5));fintervals=np.full((4,5,2),np.nan)
    for j in range(5):
        lo,hi=limits_f[j];left=lefts_f[j];tx=lambda v,left=left,lo=lo,hi=hi:left+(v-lo)/(hi-lo)*62.9291
        for row,m in enumerate(MODELS):
            paths=[q for q in p.paths if left-5<q['x']<left+66 and abs(q['y']-ys_f[row])<3.2 and colored(q)]
            if j==0:
                r=tables[34].set_index('model').loc[m];value=r.classification_mcc
                for q in paths:
                    if q['paint']==b'B*':p.move(q,x=tx(value))
            elif j==1:
                r=tables[35].loc[tables[35].model.eq(m)&tables[35].shot.eq(60)].iloc[0];value=r.pearson_r_mean
                lower,upper=value-r.pearson_r_std,value+r.pearson_r_std
                update_interval(p,paths,value,lower,upper,tx);fintervals[row,j]=[lower,upper]
            elif j in [2,3]:
                n=37 if j==2 else 36
                vals=tables[n].loc[tables[n].model.eq(m)].sort_values('run').mcc_mean.to_numpy();value=vals.mean()
                update_runs(p,paths,vals,tx);fintervals[row,j]=[min(vals),max(vals)]
            else:
                r=tables[38].set_index('model_id').loc[m];value=r.accuracy*100;lower=r.accuracy_ci_low*100;upper=r.accuracy_ci_high*100
                update_interval(p,paths,value,lower,upper,tx);fintervals[row,j]=[lower,upper]
            fvalues[row,j]=value
        for row,m in enumerate(MODELS):
            text=next(t for t in p.texts if abs(t['y']-(ys_f[row]-1.8425))<.1 and left+70<t['x']<left+83 and t['text'].replace('.','').isdigit())
            p.set_text(text,f'{fvalues[row,j]:.1f}' if j==4 else f'{fvalues[row,j]:.3f}',bold=fvalues[row,j]==fvalues[:,j].max(),align='right')
            source.append({'panel':'f','model':m,'metric':str(j),'value':fvalues[row,j],'low':fintervals[row,j,0],'high':fintervals[row,j,1]})
    src=Path(out)/'source_data';src.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(source).to_csv(src/'S12_plotted_values.csv',index=False)
    pd.DataFrame(deep,index=MODELS,columns=['DHS','HM','TF']).rename_axis('model').to_csv(src/'S12_DeepSEA_latest_run.csv')
    return p.finish(out)
