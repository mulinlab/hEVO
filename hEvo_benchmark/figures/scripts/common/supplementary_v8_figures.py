"""Render the active S3-S6 supplementary benchmark figures from local tables."""

import sys, json, shutil, subprocess
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from pypdf import PdfReader
from . import supplementary_v8_refined as r
b=r.b

b.VERSION='v10_local_FigS3_S6_family_sorted'
BASE_MODELS=list(b.MODELS)
DEEPSEA_ROSTER=[model for _,members in b.GROUPS for model in members]
FAMILY_ORDER=[family for family,_ in b.GROUPS]


def set_order(scores,figure,model_roster=None):
    """Equal weight to component rankings; score magnitudes never mixed."""
    assert not scores.isna().any().any()
    parts=scores.rank(ascending=False,method='average')
    composite=parts.mean(axis=1)
    roster=BASE_MODELS if model_roster is None else list(model_roster)
    lookup={m:i for i,m in enumerate(roster)}
    order=[]
    for family in FAMILY_ORDER:
        members=[m for m in roster if m in scores.index and b.FAMILY[m]==family]
        members=sorted(members,key=lambda m:(float(composite[m]),lookup[m]))
        order.extend(members)
        assert all(composite[a]<=composite[z] for a,z in zip(members,members[1:]))
    assert len(order)==len(scores) and set(order)==set(scores.index)
    b.MODELS=order
    r.MODELS=order
    data=scores.copy()
    for column in parts:data[column+'_rank']=parts[column]
    data['composite_mean_rank']=composite
    data['family']=[b.FAMILY[m] for m in data.index]
    data['display_order']=[order.index(m)+1 for m in data.index]
    b.save(data.reindex(order).rename_axis('model').reset_index(),figure+'_ordering_basis')
    b.CHECKS['ordering'][figure]={'method':'within family, ascending equal-weight mean of component ranks',
        'components':list(scores.columns),'ties':'average ranks per component; stable original order for composite ties',
        'family_order':FAMILY_ORDER,'display_order':order}


def deepsea(t):
    data=r.active_deepsea_rows(t[21])
    data=data.loc[data.model.isin(DEEPSEA_ROSTER)].copy()
    assert not data.duplicated(['model','task']).any()
    au=data.pivot(index='model',columns='task',values='auroc').reindex(columns=['DHS','HM','TF'])
    ap=data.pivot(index='model',columns='task',values='macro_auprc').reindex(columns=['DHS','HM','TF'])
    if not data.groupby('model').size().eq(3).all() or data.duplicated(['model','task']).any():
        raise ValueError('S21 must contain exactly one DHS/HM/TF row per included model')
    if 'GPN-Star-M' not in set(data['model']):
        raise ValueError('S21 is missing the completed GPN-Star-M DeepSEA run')
    for frame in [au,ap]:frame['Mean']=frame.mean(axis=1)
    return data,au,ap


def build_s7(data,au,ap,figure_number='S6'):
    models=b.MODELS
    au,ap=au.reindex(models),ap.reindex(models)
    ar,pr=r.ranks(au),r.ranks(ap)
    r.reconcile(au,'S5c_AUROC')
    r.reconcile(ap,'S5c_macro_AUPRC')
    p=b.Page(f'Figure{figure_number}_DeepSEA_all_models_A4',
             f'Supplementary Figure {figure_number}. DeepSEA chromatin prediction')
    p.text(9,11.6,'DeepSEA',7.,True)
    p.text(80,11.6,'AUROC',7.,True,align='center')
    p.text(164,11.6,'Macro AUPRC',7.,True,align='center')
    for x0 in [47,132]:
        for j,label in enumerate(['DHS','Histone','TF','Mean']):p.text(x0+8+j*16.7,21,label,6.,align='center')
        p.line(x0,23.6,x0+66.8,23.6,.5)
    ys=b.rows(models,25.5)
    b.row_labels(p,models,ys)
    for model in models:
        for frame,rr,x0,hi in [(au,ar,47,1.),(ap,pr,132,.4)]:
            for j,key in enumerate(['DHS','HM','TF','Mean']):
                r.score_bar(p,x0+j*16.7,ys[model],16.7,frame.loc[model,key],model,rr.loc[model,key],hi)
    bot=max(ys.values())+2
    p.line(8.7,bot,200,bot,.5)
    p.text(47,bot+5.3,'AUROC (0-1)',6.)
    p.text(132,bot+5.3,'Macro AUPRC (0-0.4)',6.)
    b.family_legend(p,bot+12)
    p.text(143,bot+11.8,'1, 2, 3   Within-column rank',5.7)
    for frame,name in [(au,f'{figure_number}_AUROC'),(ap,f'{figure_number}_macro_AUPRC'),
                       (ar,f'{figure_number}_AUROC_ranks'),(pr,f'{figure_number}_macro_AUPRC_ranks')]:
        b.save(frame.rename_axis('model').reset_index(),name)
    b.save(data,f'{figure_number}_DeepSEA_source_rows')
    b.CHECKS['figures'][figure_number]={'models':len(models),'categories':3,'category_values_per_metric':len(data),
        'content_bottom_mm':bot+14,'source_panel':'former S5c','raw_scores_reconciled_to_v2':True,
        'aggregation':'equal-weight category mean','included_open_models':r.OPEN,
        'selected_run_overrides':r.DEEPSEA_RUN_OVERRIDES}
    return p.finish()


def configure_output(figure, output_dir):
    output_dir = Path(output_dir).resolve()
    b.OUT = output_dir
    b.PNG = output_dir
    b.SRC = output_dir/'source_data'
    for directory in [b.OUT,b.SRC]: directory.mkdir(parents=True,exist_ok=True)
    b.CHECKS = {'sources':{},'figures':{},'text_bounds':{},'known_limitations':[],
                'ordering':{},'figure':figure,'reference':'Codex V8 compact S4-S7'}


def export_figure(figure, tables_dir=None, output_dir=None):
    """Render one active S3-S6 page from the user's local S11-S21 tables."""
    from pathlib import Path
    from .paths import tables_dir as resolve_tables_dir
    figure = str(figure)
    source_figure = {'FigS3': 'FigS4', 'FigS4': 'FigS6',
                     'FigS5': 'FigS5', 'FigS6': 'FigS7'}.get(figure)
    if source_figure is None:
        raise ValueError(f'Unsupported supplementary benchmark figure: {figure}')
    default = b.ROOT/'figures'/figure
    out = Path(output_dir).resolve() if output_dir is not None else default
    configure_output(figure, out)
    b.TABLES = resolve_tables_dir(tables_dir)
    t={n:b.load(n) for n in [11,12,13,14,15,16,17,18,19,20,21]}
    r.SOURCE_TABLES = t
    b.MODELS = list(BASE_MODELS)
    r.MODELS = list(BASE_MODELS)
    if source_figure == 'FigS4':
        gb_tasks=['human_enhancers_cohn','human_enhancers_ensembl','human_ensembl_regulatory',
                  'human_nontata_promoters','human_ocr_ensembl']
        common=t[18].loc[t[18].dataset.isin(gb_tasks)].groupby('model').mcc_mean.mean()
        nt=t[17].groupby('model').mcc_mean.mean()
        scores=pd.DataFrame({'GB_common5_mean_MCC':common,'NT18_mean_MCC':nt}).reindex(BASE_MODELS)
        if scores.isna().any().any():
            raise ValueError('S3 within-family ordering requires complete shared-human and NT coverage')
        set_order(scores,'S3')
        path=b.figure_s4(t,column_scaled=True,unified_missing=True,top_three=True,show_notes=False)
    elif source_figure == 'FigS5':
        f=t[20].groupby('model').per_class_f1.mean()
        binary=t[19].groupby('model').mcc_mean.mean()
        set_order(pd.DataFrame({'regulatory_mean_F1':f,'URE_mean_MCC':binary}).reindex(BASE_MODELS),'S5')
        path=r.build_s5(t,include_deepsea=False)
    elif source_figure == 'FigS6':
        ch=pd.DataFrame(index=BASE_MODELS)
        for label,num in [('ENCODE',11),('GENCODE',13),('RefSeq',15)]:
            d=t[num]
            ch[label+'_PCA_CH']=d.loc[d.method.eq('PCA')].groupby('model_name').ch_index.mean().reindex(BASE_MODELS)
        set_order(ch,'S4')
        path=r.build_s6(t)
    else:
        data,au,ap=deepsea(t)
        set_order(pd.DataFrame({'category_mean_AUROC':au['Mean'],
                                'category_mean_macro_AUPRC':ap['Mean']}),'S6',
                  model_roster=DEEPSEA_ROSTER)
        path=build_s7(data,au,ap,figure_number='S6')
    if not path.is_file(): raise FileNotFoundError(path)
    page=PdfReader(str(path)).pages[0]
    text=page.extract_text() or ''
    expected={'FigS3':['Genomic Benchmark','NT Benchmark'],
              'FigS4':['ENCODE cCRE v4','Calinski-Harabasz','PCA 2D'],
              'FigS5':['Regulatory classification','URE versus enhancer'],
              'FigS6':['DeepSEA','AUROC','Macro AUPRC']}[figure]
    missing=[label for label in expected if label not in text]
    if missing: raise ValueError(f'{figure} output misses labels: {missing}')
    final_pdf=out/'A4_integrated.pdf'
    if path.resolve()!=final_pdf.resolve(): shutil.copyfile(path,final_pdf)
    pdftoppm=shutil.which('pdftoppm')
    if not pdftoppm: raise FileNotFoundError('pdftoppm is required to render A4 PNGs')
    subprocess.run([pdftoppm,'-f','1','-l','1','-singlefile','-png','-r','600',
                    str(final_pdf),str(out/'A4_integrated')],check=True,capture_output=True)
    pdftocairo=shutil.which('pdftocairo')
    if pdftocairo:
        svg_prefix = out/'A4_integrated'
        subprocess.run([pdftocairo,'-f','1','-l','1','-svg',
                        str(final_pdf),str(svg_prefix)],check=True,capture_output=True)
        generated_svg = svg_prefix if svg_prefix.is_file() else svg_prefix.with_suffix('.svg')
        if generated_svg != out/'A4_integrated.svg':
            generated_svg.replace(out/'A4_integrated.svg')
    else:
        raise FileNotFoundError('pdftocairo is required to render the editable A4 SVG')
    b.CHECKS['output']={'figure':figure,'pdf':final_pdf.name,'source_pdf':path.name,
                        'page_mm':[210,297],'vector_pdf':True,'source_table_values_recomputed':True}
    active_captions = '''Supplementary Figures S3-S6. Model-family color and marker conventions follow the shared figure system.

FigS3. Genomic Benchmark and nucleotide-transformer classification. Panels a and b show local source-table task performance from S18 and S17, respectively.

FigS4. Functional class separation in two-dimensional PCA representations. Three columns show ENCODE cCRE v4, GENCODE v45, and UCSC RefSeq; values are three-seed means from S11-S16.

FigS5. Regulatory classification. Panel a shows class-specific F1; panel b shows URE-versus-enhancer MCC across five benchmark sets, sourced from S19-S20.

FigS6. DeepSEA chromatin prediction. Category-macro AUROC and macro AUPRC with equal-weight category means, sourced from S21. The displayed roster includes 22 model configurations with complete DHS, HM and TF results, including hEVO-base-o and hEVO-large-o; for hEVO-large-o the figure uses one-epoch FP16 rerun S21_run_24, whose manifest/model summary report actual pooler output although the declared configuration says mean-pooling. Historical S21_run_22 (three-epoch BF16, mean-pooling) remains in S21; the two runs are not a matched training comparison. GPN-Star-M is included. Input lengths vary by model (512, 1,000 or 1,024 bp), so this is not a matched-length comparison. GPN-Star-M uses native 1,000-bp MSA input and has 436,931 of 455,024 test windows mapped (96.02%); unmapped windows are excluded from its evaluation.
'''

    source_number = int(source_figure[-1])
    target_number = int(figure[-1])
    if source_number != target_number:
        for item in sorted(out.rglob('*'), reverse=True):
            if item.is_file() and item.suffix.lower() in {'.json', '.txt'}:
                try:
                    content = item.read_text(encoding='utf-8')
                    content = content.replace(f'FigureS{source_number}', f'FigureS{target_number}')
                    content = content.replace(f'FigS{source_number}', f'FigS{target_number}')
                    content = content.replace(f'S{source_number}_', f'S{target_number}_')
                    item.write_text(content, encoding='utf-8')
                except UnicodeDecodeError:
                    pass
            if item.is_file():
                new_name = item.name.replace(f'FigureS{source_number}', f'FigureS{target_number}')
                new_name = new_name.replace(f'S{source_number}_', f'S{target_number}_')
                if new_name != item.name:
                    item.rename(item.with_name(new_name))
    caption_path = out/'source_data'/f'{figure}_caption.txt'
    caption_backup = Path('/tmp/supplementary_before_reference_remap')/'old_captions'/figure
    for stale in sorted((out/'source_data').glob('*caption*.txt')):
        if stale != caption_path:
            caption_backup.mkdir(parents=True,exist_ok=True)
            destination = caption_backup/stale.name
            version = 1
            if destination.exists():
                destination = caption_backup/f'{stale.stem}_old{version}{stale.suffix}'
                while destination.exists():
                    version += 1
                    destination = caption_backup/f'{stale.stem}_old{version}{stale.suffix}'
            shutil.move(str(stale),str(destination))
    caption_path.write_text(active_captions,encoding='utf-8')
    return {'pdf':str(final_pdf),'png':str(out/'A4_integrated.png'),
            'svg':str(out/'A4_integrated.svg'),'source_pdf':str(path)}


if __name__=='__main__':
    for item in ['FigS3', 'FigS4', 'FigS5', 'FigS6']:
        print(export_figure(item))
