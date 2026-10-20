"""Refine S5 and PCA-only S6 using the reviewed S4 typography and family palette.

Original V8 tables and previous PDFs are never modified.
"""

from pathlib import Path
import json, math, shutil, subprocess, sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from reportlab.lib.units import mm
from . import supplementary_v8_base as b

b.VERSION = 'v9_S5_S6_refined'
b.OUT = b.ROOT/'figures'/'FigS5'
b.PNG = b.OUT
b.SRC = b.OUT/'source_data'
b.HEAT = ['#FAFAF8','#EEEEEB','#D3D3D0','#A4A4A1','#5C5C59']
b.ROW_HIGHLIGHT = '#FCF5EE'
OPEN = ['hEvo-open-large','hEvo-open-base']
DEEPSEA_RUN_OVERRIDES = {'hEvo-open-large': 'S21_run_24'}
b.MODELS = [m for m in b.MODELS if m not in OPEN]
MODELS = b.MODELS


def ranks(frame):
    result=frame.rank(ascending=False,method='min')
    # Independently check all exact, unrounded competition ranks.
    if isinstance(frame,pd.Series):
        for m,v in frame.dropna().items():assert result[m]==1+int((frame>v).sum())
    else:
        for k in frame:
            for m,v in frame[k].dropna().items():assert result.loc[m,k]==1+int((frame[k]>v).sum())
    return result


def rank_note(p,x,y,rank,color='#303030'):
    # Keep the small rank fully inside a heatmap row, not on its white gutter.
    if pd.notna(rank) and rank<=3:p.text(x,y+.5,str(int(rank)),4.5,color=color)


def score_bar(p,x,y,w,value,model,rank,hi=1.):
    if pd.isna(value):
        p.text(x+w/2,y+.65,'/',5.8,align='center');return
    assert 0<=value<=hi
    tone=.28 if b.FAMILY[model]=='hEVO' else .23 if b.FAMILY[model]!='Other' else .44
    p.rect(x+.25,y-.88,(w-.5)*value/hi,1.76,b.blend('#FFFFFF',b.COL[b.FAMILY[model]],tone))
    p.text(x+w*.46,y+.65,f'{value:.3f}',5.8,align='center')
    rank_note(p,x+w-1.45,y-.38,rank)


SOURCE_TABLES = {}


def active_deepsea_rows(data):
    """Select the current DeepSEA run for each model without dropping history."""
    data = data.copy()
    aliases = {'NT-2.5B-1kG':'NT_2.5b_1kG','NT-2.5B-MS':'NT_2.5b_MS','NT-500M-1kG':'NT_500m_1kG'}
    data['model'] = data.model.replace(aliases)
    for model, run_id in DEEPSEA_RUN_OVERRIDES.items():
        model_rows = data.model.eq(model)
        if not data.loc[model_rows, 'run_id'].eq(run_id).any():
            raise ValueError(f'S21 is missing the selected {model} run {run_id}')
        data = data.loc[~model_rows | data.run_id.eq(run_id)].copy()
    if data.duplicated(['model', 'task']).any():
        raise ValueError('S21 must have one active run per model and task')
    return data


def reconcile(frame,old_name):
    """Recompute each plotted aggregate from the current local source tables."""
    t = SOURCE_TABLES
    if old_name == 'S5a_class_F1_means':
        expected = (t[20].loc[t[20].model.isin(MODELS)]
                    .groupby(['model','class_label']).per_class_f1.mean().unstack()
                    .reindex(index=MODELS, columns=['ELS','PLS','Insulator','Silencer','URE']))
    elif old_name == 'S5b_five_dataset_MCC':
        expected = (t[19].loc[t[19].model.isin(MODELS)]
                    .pivot(index='model',columns='run',values='mcc_mean').reindex(MODELS))
    elif old_name in {'S5c_AUROC','S5c_macro_AUPRC'}:
        data = active_deepsea_rows(t[21])
        metric = 'auroc' if old_name == 'S5c_AUROC' else 'macro_auprc'
        expected = (data.loc[data.model.isin(MODELS)]
                    .pivot(index='model',columns='task',values=metric)
                    .reindex(index=MODELS,columns=['DHS','HM','TF']))
        expected['Mean'] = expected.mean(axis=1)
        expected = expected.loc[frame.index]
    elif old_name == 'S6_PCA_CH':
        expected = pd.DataFrame(index=MODELS)
        for label, num in [('ENCODE cCRE v4',11),('GENCODE v45',13),('UCSC RefSeq',15)]:
            rows = t[num].loc[t[num].method.eq('PCA') & t[num].model_name.isin(MODELS)]
            expected[label] = rows.groupby('model_name').ch_index.mean().reindex(MODELS)
        expected = expected.loc[frame.index]
    else:
        raise KeyError(f'No local-source reconciliation for {old_name}')
    expected = expected.reindex(index=frame.index, columns=frame.columns)
    np.testing.assert_allclose(frame.to_numpy(dtype=float), expected.to_numpy(dtype=float),
                               rtol=1e-12, atol=1e-12, equal_nan=True)


def build_s5(t,include_deepsea=True):
    classes=['ELS','PLS','Insulator','Silencer','URE']
    class_labels=['Enhancer','Promoter','Insulator','Silencer','URE']
    source_f=t[20].loc[t[20].model.isin(MODELS)].copy()
    source_b=t[19].loc[t[19].model.isin(MODELS)].copy()
    assert source_f.groupby('model').size().eq(25).all()
    assert source_b.groupby('model').size().eq(5).all()
    assert not source_f.duplicated(['model','dataset','class_label']).any()
    assert not source_b.duplicated(['model','run']).any()
    f=source_f.groupby(['model','class_label']).per_class_f1.mean().unstack().reindex(index=MODELS,columns=classes)
    fm=f.mean(axis=1)
    binary=source_b.pivot(index='model',columns='run',values='mcc_mean').reindex(MODELS)
    bm=binary.mean(axis=1)
    scaled=(f-f.min())/(f.max()-f.min())
    assert f.max().sub(f.min()).gt(0).all()
    np.testing.assert_allclose(scaled.min(),0)
    np.testing.assert_allclose(scaled.max(),1)
    pd.testing.assert_frame_equal(ranks(f),ranks(scaled))
    fr,fmr,br=ranks(f),ranks(fm),ranks(bm)
    reconcile(f,'S5a_class_F1_means')
    reconcile(binary,'S5b_five_dataset_MCC')

    p=b.Page('FigureS5_regulatory_function_all_models_A4','Supplementary Figure S5. Regulatory sequence discrimination'+(' and chromatin prediction' if include_deepsea else ''))
    ys=b.rows(MODELS,29.5)
    p.panel('a',9,12,'Regulatory classification')
    p.panel('b',123,12,'URE versus enhancer')
    b.row_labels(p,MODELS,ys)
    x0,cw=47.,10.5
    for j,label in enumerate(class_labels):p.text(x0+(j+.5)*cw,23,label,6.,align='center')
    p.text(109.5,23,'Mean F1',6.,align='center')
    p.text(154,23,'Five benchmark sets',5.9,align='center')
    p.text(195,26,'Mean MCC',5.9,align='center')
    p.line(46.8,27.7,116.8,27.7,.5)
    p.line(123,27.7,200,27.7,.5)
    plot0,plot1=126.,185.
    for model in MODELS:
        y=ys[model]
        for j,key in enumerate(classes):
            x=x0+j*cw
            fill=b.heat(scaled.loc[model,key])
            p.rect(x+.08,y-1.235,cw-.16,2.47,fill)
            lum=.2126*fill.red+.7152*fill.green+.0722*fill.blue
            color='#FFFFFF' if lum<.55 else '#000000'
            p.text(x+cw*.44,y+.64,f'{f.loc[model,key]:.3f}',5.55,color=color,align='center')
            rank_note(p,x+cw-1.5,y-.38,fr.loc[model,key],color)
        score_bar(p,102,ys[model],14.5,fm[model],model,fmr[model])
        values=binary.loc[model].to_numpy()
        xs=plot0+(plot1-plot0)*values
        color=b.COL[b.FAMILY[model]]
        p.line(xs.min(),y,xs.max(),y,.55,color,dash=[2.4,1.5] if model=='hEvo-base' else None)
        for k,x in enumerate(xs):p.marker(x,y+[-.48,-.24,0,.24,.48][k],.37,color,True)
        p.marker(plot0+bm[model]*(plot1-plot0),y,.72,color,open_=model=='hEvo-base')
        p.text(196.9,y+.65,f'{bm[model]:.3f}',5.8,align='right')
        rank_note(p,198.,y-.38,br[model])
    bot=max(ys.values())+2
    p.line(8.7,bot,116.8,bot,.5)
    p.line(plot0,bot,plot1,bot,.5)
    for v in [0,.25,.5,.75,1]:
        x=plot0+(plot1-plot0)*v
        p.line(x,bot,x,bot+1,.4)
        p.text(x,bot+3.8,f'{v:g}',5.7,align='center')
    p.text((plot0+plot1)/2,bot+8.1,'MCC',6.2,align='center')
    b.colorbar(p,47,bot+7.7,34,'Within-class relative F1',ticks=[0,.5,1])

    if not include_deepsea:
        b.family_legend(p,bot+20)
        p.text(143,bot+19.8,'1, 2, 3   Within-column rank',5.7)
        for frame,name in [(f,'S5a_class_F1_means'),(scaled,'S5a_class_F1_minmax'),(fr,'S5a_class_F1_ranks'),(binary,'S5b_five_dataset_MCC')]:
            b.save(frame.rename_axis('model').reset_index(),name)
        b.save(pd.DataFrame({'model':MODELS,'mean_class_F1':fm.values,'F1_rank':fmr.values,
                             'mean_binary_MCC':bm.values,'MCC_rank':br.values}),'S5_aggregate_scores_and_ranks')
        b.save(source_f,'S5a_all_class_dataset_F1')
        b.save(source_b,'S5b_source_rows_with_reported_intervals')
        b.save(pd.DataFrame({'class':classes,'minimum_F1':f.min().values,'maximum_F1':f.max().values}),'S5a_normalization_parameters')
        b.CHECKS['figures']['S5']={'regulatory_models':len(MODELS),'class_values':int(f.size),
            'binary_dataset_values':int(binary.size),'removed_models':OPEN,'content_bottom_mm':bot+22,
            'deepsea_moved_to':'S7','raw_scores_reconciled_to_v2':True,
            'class_color':'per-class min-max; printed scores and aggregate values are raw F1',
            'intervals':'range across five benchmark sets, not confidence intervals'}
        return p.finish()

    deep=t[21].copy()
    alias={'NT-2.5B-1kG':'NT_2.5b_1kG','NT-2.5B-MS':'NT_2.5b_MS','NT-500M-1kG':'NT_500m_1kG'}
    deep['model']=deep.model.replace(alias)
    deep=deep.loc[deep.model.isin(MODELS)].copy()
    dm=[m for m in MODELS if m in set(deep.model)]
    assert len(dm)==19 and deep.groupby('model').size().eq(3).all()
    assert not deep.duplicated(['model','task']).any()
    au=deep.pivot(index='model',columns='task',values='auroc').reindex(index=dm,columns=['DHS','HM','TF'])
    ap=deep.pivot(index='model',columns='task',values='macro_auprc').reindex(index=dm,columns=['DHS','HM','TF'])
    for frame in [au,ap]:frame['Mean']=frame.mean(axis=1)
    ar,pr=ranks(au),ranks(ap)
    reconcile(au,'S5c_AUROC')
    reconcile(ap,'S5c_macro_AUPRC')
    cy=bot+23
    p.panel('c',9,cy,'DeepSEA')
    p.text(80,cy-.4,'AUROC',7.,True,align='center')
    p.text(164,cy-.4,'Macro AUPRC',7.,True,align='center')
    for xx in [47,132]:
        for j,label in enumerate(['DHS','Histone','TF','Mean']):p.text(xx+8+j*16.7,cy+6,label,6.,align='center')
        p.line(xx,cy+8.5,xx+66.8,cy+8.5,.5)
    dys=b.rows(dm,cy+10.5)
    b.row_labels(p,dm,dys)
    for model in dm:
        for frame,rr,x,hi in [(au,ar,47,1.),(ap,pr,132,.4)]:
            for j,key in enumerate(['DHS','HM','TF','Mean']):
                score_bar(p,x+j*16.7,dys[model],16.7,frame.loc[model,key],model,rr.loc[model,key],hi)
    dbot=max(dys.values())+2
    p.line(8.7,dbot,200,dbot,.5)
    p.text(47,dbot+5.3,'AUROC (0-1)',6.)
    p.text(132,dbot+5.3,'Macro AUPRC (0-0.4)',6.)
    b.family_legend(p,dbot+12)
    p.text(143,dbot+11.8,'1, 2, 3   Within-column rank',5.7)

    for frame,name in [(f,'S5a_class_F1_means'),(scaled,'S5a_class_F1_minmax'),(fr,'S5a_class_F1_ranks'),
                       (binary,'S5b_five_dataset_MCC'),(au,'S5c_AUROC'),(ap,'S5c_macro_AUPRC'),
                       (ar,'S5c_AUROC_ranks'),(pr,'S5c_macro_AUPRC_ranks')]:
        b.save(frame.rename_axis('model').reset_index(),name)
    b.save(pd.DataFrame({'model':MODELS,'mean_class_F1':fm.values,'F1_rank':fmr.values,
                         'mean_binary_MCC':bm.values,'MCC_rank':br.values}),'S5_aggregate_scores_and_ranks')
    b.save(source_f,'S5a_all_class_dataset_F1')
    b.save(source_b,'S5b_source_rows_with_reported_intervals')
    b.save(deep,'S5c_DeepSEA_source_rows')
    b.save(pd.DataFrame({'class':classes,'minimum_F1':f.min().values,'maximum_F1':f.max().values}),'S5a_normalization_parameters')
    b.CHECKS['figures']['S5']={'regulatory_models':len(MODELS),'class_values':int(f.size),
        'binary_dataset_values':int(binary.size),'deepsea_models':len(dm),'deepsea_category_values_per_metric':len(deep),
        'removed_models':OPEN,'content_bottom_mm':dbot+14,'raw_scores_reconciled_to_v2':True,
        'class_color':'per-class min-max across displayed models; printed scores and means are raw F1',
        'intervals':'range across five benchmark-set estimates; not a confidence interval',
        'mean_aggregation':'equal weight per regulatory class or per DeepSEA category',
        'top_ranks':'raw unrounded values; competition ranks; descriptive, not significance'}
    return p.finish()


def build_s6(t):
    datasets=[('ENCODE cCRE v4',11,12,5,5000),('GENCODE v45',13,14,3,3000),('UCSC RefSeq',15,16,5,5000)]
    frame=pd.DataFrame(index=MODELS)
    checks=[]
    for label,seednum,sumnum,nclass,nseq in datasets:
        seed=t[seednum].loc[t[seednum].method.eq('PCA') & t[seednum].model_name.isin(MODELS)].copy()
        assert seed.groupby('model_name').size().eq(3).all()
        assert seed.n_rows.eq(nseq).all() and seed.n_classes.eq(nclass).all()
        assert seed.coordinate_space.eq('2D projected coordinates').all()
        agg=seed.groupby('model_name').ch_index.agg(['mean','std','min','max']).reindex(MODELS)
        summary=t[sumnum].loc[t[sumnum].method.eq('PCA')].set_index('model_name').reindex(MODELS)
        np.testing.assert_allclose(agg['mean'],summary.ch_index_mean,rtol=1e-10)
        np.testing.assert_allclose(agg['std'],summary.ch_index_sd,rtol=1e-7,atol=1e-7)
        frame[label]=agg['mean']
        b.save(seed,f'S6_PCA_seed_rows_S{seednum}')
        b.save(agg.rename_axis('model').reset_index(),f'S6_PCA_recomputed_S{seednum}')
        checks.append({'dataset':label,'models':len(MODELS),'seed_values':len(seed),
                       'largest_PCA_seed_SD':float(agg['std'].max()),'mean_and_SD_match_source_summary':True})
    reconcile(frame,'S6_PCA_CH')
    rr=ranks(frame)
    b.save(frame.rename_axis('model').reset_index(),'S6_PCA_CH')
    b.save(rr.rename_axis('model').reset_index(),'S6_PCA_ranks')

    p=b.Page('FigureS6_PCA_representation_separation_all_models_A4','Supplementary Figure S6. Functional class separation in two-dimensional PCA representations')
    ys=b.rows(MODELS,27.5)
    b.row_labels(p,MODELS,ys)
    domains=[(30,210,[40,80,160]),(10,2500,[10,100,1000]),(35,1600,[50,200,1000])]
    for j,(label,_,__,nclass,nseq) in enumerate(datasets):
        x0=46+52*j
        w=34.
        value_right=x0+46.5
        lo,hi,ticks=domains[j]
        fx=lambda value:x0+w*(math.log10(value)-math.log10(lo))/(math.log10(hi)-math.log10(lo))
        p.text(x0+24,13,label,6.5,True,align='center')
        p.text(x0+24,17.2,f'{nclass} classes',5.8,align='center')
        p.text(value_right-3,23,'CH',5.8,align='center')
        p.line(x0,25.6,x0+48.6,25.6,.5)
        for model in MODELS:
            value=frame.loc[model,label]
            assert lo<value<hi
            y=ys[model]
            family=b.FAMILY[model]
            p.marker(fx(value),y,.77 if model in b.FOCUS else .63,b.COL[family],
                     open_=model=='hEvo-base',shape={'Evo 2':'triangle','NT':'square','GPN':'diamond'}.get(family,'circle'))
            p.text(value_right,y+.65,f'{value:,.1f}',5.75,align='right')
            rank_note(p,value_right+.65,y-.38,rr.loc[model,label])
        bot=max(ys.values())+2
        p.line(x0,bot,x0+w,bot,.5)
        for tick in ticks:
            xx=fx(tick)
            p.line(xx,bot,xx,bot+1,.4)
            p.text(xx,bot+3.9,f'{tick:,}',5.7,align='center')
    p.text(123,bot+8.5,'Calinski-Harabasz index (PCA 2D, log scale)',6.3,align='center')
    b.family_legend(p,bot+16)
    p.text(143,bot+15.8,'1, 2, 3   Within-dataset rank',5.7)
    b.CHECKS['figures']['S6']={'models':len(MODELS),'mean_CH_values':int(frame.size),
        'seed_CH_values':sum(x['seed_values'] for x in checks),'projection':'PCA only, two dimensions',
        'axis':'log10 CH, dataset-specific labeled ranges; printed CH is not transformed',
        'native_embedding_CH':False,'source_checks':checks,'removed_models':OPEN,
        'raw_scores_reconciled_to_v2':True,'content_bottom_mm':bot+18}
    return p.finish()


def finish(paths):
    poppler = shutil.which('pdftoppm')
    if not poppler:
        raise FileNotFoundError('pdftoppm is required to render integrated PNG previews')
    b.CHECKS['pdf_checks']={}
    for path in paths:
        with pdfplumber.open(path) as doc:
            assert len(doc.pages)==1
            page=doc.pages[0]
            assert abs(page.width/mm-210)<.01 and abs(page.height/mm-297)<.01
            assert not page.images
            text=page.extract_text()
            assert 'open-base' not in text.lower() and 'open-large' not in text.lower()
            assert '?' not in text and 'n/a' not in text
            if 'S6_' in path.name:assert 't-SNE' not in text and 'tSNE' not in text and 'Projection rank agreement' not in text
            b.CHECKS['pdf_checks'][path.name]={'pages':1,'size_mm':[page.width/mm,page.height/mm],
                'fonts':sorted({c['fontname'] for c in page.chars}),'raster_images':0}
        subprocess.run([poppler,'-r','600','-png','-singlefile',str(path),str(b.PNG/path.stem)],check=True,capture_output=True)
    b.CHECKS['scope']={'original_V8_data_modified':False,'previous_PDFs_preserved':True,
                      'Open_models':'excluded here; retained in original sources for Fig5',
                      'DeepSEA_missing_models':'no results imputed for models absent from S21',
                      'PCA_seed_variation':'numerical/projection variation, not biological replication or model retraining'}
    print(json.dumps({'pdfs':[str(x) for x in paths],'all_source_checks_passed':True},indent=2))


CAPTIONS='''Supplementary Figure S5. Regulatory sequence discrimination and chromatin prediction.
(a) Class-specific F1 values averaged across the same five regulatory benchmark sets for 32 model configurations. Cell numbers are raw F1 values. Shading is independently min-max normalized within each class among the displayed models; equal shades across classes need not denote equal raw F1. The mean F1 column equally weights all five classes and five sets and is not calculated from normalized values. (b) URE versus enhancer classification across the five benchmark sets. Small open circles are set-specific MCC estimates, the connecting line is their observed range, and the larger symbol is the arithmetic mean. These ranges are not confidence intervals. Reported source intervals remain in the exported source table. (c) DeepSEA category-macro AUROC and macro AUPRC for the 19 recorded model configurations remaining after omission of Open models. Miniature bars start at zero and use the labeled metric ranges. Means equally weight DHS, histone modifications and transcription factor binding, not individual labels. The categories have 125, 104 and 690 nominal labels, with 689 valid labels contributing to the TF macro metrics. Recorded sequence lengths differ between models (512 or 1,024 bp), so these are not matched-length comparisons. Unavailable GPN-MSA/Star DeepSEA results are not imputed. Small numerals 1, 2 and 3 denote within-column ranks of unrounded scores; exact ties receive the same competition rank. Ranks do not establish statistical significance. The hEVO-base mean marker is open. All family colors, model names and row spacing follow S4. hEVO-base-o and hEVO-large-o are reserved for separate presentation with Fig. 5. Sources: S19-S21. Original raw and aggregate values are unchanged.

Supplementary Figure S6. Functional class separation in two-dimensional PCA representations.
Calinski-Harabasz (CH) indices for 32 model configurations in ENCODE cCRE v4 (five classes, 5,000 sequences), GENCODE v45 (three classes, 3,000 sequences) and UCSC RefSeq curated (five classes, 5,000 sequences). Points and printed numbers show the mean CH across the three recorded PCA projection seeds, 42, 43 and 44. Horizontal positions use independently labeled logarithmic ranges for each dataset; printed numbers retain the original CH scale. Family grouping and model order match S4. Small numerals 1, 2 and 3 mark the top ranks within each dataset among the displayed models, using unrounded CH values and competition ranks for ties. They are descriptive, not tests of significance. The indices are computed in two-dimensional projected coordinates and must not be interpreted as separation in the original embedding space. PCA seed variation is numerical/projection variation, not independent biological replication or encoder retraining; the very small seed-level dispersion remains in the source export. Only PCA is included in this figure, with no t-SNE panel or projection-rank comparison. Means and standard deviations were independently reconciled with seed-level S11, S13 and S15 and summary S12, S14 and S16. The previously noted discrepancy in a t-SNE standard deviation does not affect this PCA-only figure. hEVO-base-o and hEVO-large-o models are reserved for Fig. 5. Original source tables are unchanged.
'''

if __name__=='__main__':
    t={n:b.load(n) for n in [11,12,13,14,15,16,19,20,21]}
    paths=[build_s5(t),build_s6(t)]
    (b.OUT/'FigureS5_S6_captions.txt').write_text(CAPTIONS,encoding='utf-8')
    finish(paths)
