"""Compact, source-backed Fig. 2 supplements. V8 inputs remain read-only.

Run with the bundled Python runtime. PDF objects are vector; Arial is embedded.
All aggregates and source exclusions are exported alongside the figures.
"""

from pathlib import Path
import sys, json, math, hashlib, importlib.util, subprocess, os, shutil
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = ROOT/'tables'
VERSION = 'v2'
OUT = ROOT/'figures'/'FigS4'
PNG = OUT
SRC = OUT/'source_data'
for directory in [OUT, PNG, SRC]: directory.mkdir(parents=True, exist_ok=True)
from .font_config import register_reportlab_arial
register_reportlab_arial()
spec = importlib.util.spec_from_file_location('hevo_style', ROOT/'figures/scripts/_backend/figures/scripts/common/hevo_model_style.py')
style = importlib.util.module_from_spec(spec); spec.loader.exec_module(style)
COL = {'hEVO':'#B35C1E','Evo 2':'#0E7C86','NT':'#65717F','GPN':'#0B2E83','Other':'#B8BEC8'}
GROUPS = [
 ('hEVO',['hEvo-large','hEvo-base','hEvo-open-large','hEvo-open-base']),
 ('Evo 2',['evo2_7b_base','evo2_1b_base']),
 ('NT',['NT_2.5b_MS','NT_2.5b_1kG','NT_500m_MS','NT_500m_1kG','NTv3-650M-pre','NTv3-100M-pre','NTv3-8M-pre']),
 ('GPN',['GPN-promoter','GPN-MSA','GPN-Star-V','GPN-Star-M','GPN-Star-P']),
 ('Other',['AIDO.DNA-300M','Caduceus-PH','Caduceus-PS','DNABERT','DNABERT2','GENA-LM-base','GENA-LM-large','GENERator-1.2B','GENERator-3B','Genos-1.2B','GROVER','HyenaDNA_tiny_1k','Mistral-DNA-17M','Mistral-DNA-138M','Mistral-DNA-422M','SpeciesLM'])
]
MODELS = [m for _,ms in GROUPS for m in ms]
FAMILY = {m:f for f,ms in GROUPS for m in ms}
FOCUS = {'hEvo-large','hEvo-base'}
PAGE_W, PAGE_H = 210., 297.
CHECKS = {'sources':{}, 'figures':{}, 'text_bounds':{}, 'known_limitations':[]}

def load(n):
    path = next(TABLES.glob(f'S{n}_*.csv'))
    data = pd.read_csv(path)
    CHECKS['sources'][f'S{n}'] = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'rows':len(data)}
    return data

def save(data, name): data.to_csv(SRC/(name+'.csv'), index=False, encoding='utf-8-sig')

def display(m): return style.canonical_model_name(m)

def rgb(h): return np.array([int(h[i:i+2],16)/255 for i in (1,3,5)])

def blend(a,b,t):
    return Color(*(rgb(a)*(1-t)+rgb(b)*t))

HEAT = ['#FFFFFF','#EFF1F3','#D9DEE4','#AFB8C3','#788594']
ROW_HIGHLIGHT = '#FBF1E7'
def heat(v, lo=0., hi=1.):
    p=np.clip((v-lo)/(hi-lo),0,1)*(len(HEAT)-1)
    i=min(int(p),len(HEAT)-2)
    return blend(HEAT[i],HEAT[i+1],p-i)

class Page:
    def __init__(self,name,title):
        self.name=name; self.path=OUT/(name+'.pdf')
        self.c=canvas.Canvas(str(self.path), pagesize=(PAGE_W*mm,PAGE_H*mm), pageCompression=1)
        self.c.setTitle(title); self.c.setAuthor('hEVO')
        self.records=[]
    def text(self,x,y,s,size=6.,bold=False,color='#000000',align='left',angle=0):
        c=self.c; font='Arial-Bold' if bold else 'Arial'; s=str(s)
        w=pdfmetrics.stringWidth(s,font,size)/mm
        ox=0 if align=='left' else -w/2 if align=='center' else -w
        c.saveState();c.translate(x*mm,(PAGE_H-y)*mm);c.rotate(angle)
        c.setFont(font,size);c.setFillColor(HexColor(color))
        c.drawString(ox*mm,0,s);c.restoreState()
        # Coordinates recorded for text fitting and collision diagnostics.
        asc=pdfmetrics.getAscent(font)*size/1000/mm
        desc=pdfmetrics.getDescent(font)*size/1000/mm
        theta=math.radians(angle)
        corners=[(x+dx*math.cos(theta)-dy*math.sin(theta), y-dx*math.sin(theta)-dy*math.cos(theta))
                 for dx,dy in [(ox,desc),(ox+w,desc),(ox,asc),(ox+w,asc)]]
        xs,ys=zip(*corners)
        self.records.append({'text':s,'size':size,'x0':min(xs),'x1':max(xs),'y0':min(ys),'y1':max(ys),'angle':angle})
    def line(self,x1,y1,x2,y2,width=.35,color='#000000',dash=None):
        c=self.c;c.saveState();c.setStrokeColor(HexColor(color));c.setLineWidth(width)
        if dash:c.setDash(dash)
        c.line(x1*mm,(PAGE_H-y1)*mm,x2*mm,(PAGE_H-y2)*mm);c.restoreState()
    def rect(self,x,y,w,h,fill=None,stroke=None,lw=.35):
        c=self.c;c.saveState();c.setLineWidth(lw)
        if fill:c.setFillColor(HexColor(fill) if isinstance(fill,str) else fill)
        if stroke:c.setStrokeColor(HexColor(stroke))
        c.rect(x*mm,(PAGE_H-y-h)*mm,w*mm,h*mm,fill=int(fill is not None),stroke=int(stroke is not None));c.restoreState()
    def marker(self,x,y,r=.7,color='#000000',open_=False,shape='circle'):
        c=self.c;c.saveState();c.setLineWidth(.55);c.setStrokeColor(HexColor(color));c.setFillColor(HexColor('#FFFFFF' if open_ else color))
        if shape=='circle':c.circle(x*mm,(PAGE_H-y)*mm,r*mm,stroke=1,fill=1)
        else:
            points=([(x-r,y+r*.8),(x+r,y+r*.8),(x,y-r)] if shape=='triangle' else
                    [(x-r,y),(x,y-r),(x+r,y),(x,y+r)] if shape=='diamond' else
                    [(x-r,y-r),(x+r,y-r),(x+r,y+r),(x-r,y+r)])
            p=c.beginPath();p.moveTo(points[0][0]*mm,(PAGE_H-points[0][1])*mm)
            for xx,yy in points[1:]:p.lineTo(xx*mm,(PAGE_H-yy)*mm)
            p.close();c.drawPath(p,fill=1,stroke=1)
        c.restoreState()
    def panel(self,letter,x,y,title=None):
        self.text(x,y,letter,12.,True)
        if title:self.text(x+6,y-.4,title,7.,True)
    def finish(self):
        self.c.showPage();self.c.save()
        bad=[r for r in self.records if r['x0']<7 or r['x1']>203 or r['y0']<5 or r['y1']>289]
        collisions=[]
        for i,a in enumerate(self.records):
            for b in self.records[i+1:]:
                if a['angle'] or b['angle']:continue
                dx=min(a['x1'],b['x1'])-max(a['x0'],b['x0'])
                dy=min(a['y1'],b['y1'])-max(a['y0'],b['y0'])
                if dx>.08 and dy>.1:collisions.append([a['text'],b['text'],dx,dy])
        CHECKS['text_bounds'][self.name]={'text_count':len(self.records),'out_of_safe_page_bounds':bad,'unrotated_text_collisions':collisions}
        assert not bad, bad
        assert not collisions, collisions
        return self.path

def rows(models, top, pitch=2.65, gap=.8):
    ypos={};y=top;prev=None
    for m in models:
        f=FAMILY[m]
        if prev is not None and f!=prev:y+=gap
        ypos[m]=y; y+=pitch;prev=f
    return ypos

def row_labels(p,models,ys,pitch=2.65,x0=9.5,right=200,shading=True):
    prev=None
    for m in models:
        y=ys[m];f=FAMILY[m]
        if m in FOCUS and shading:p.rect(x0-.8,y-pitch*.48,right-x0+.8,pitch*.96,ROW_HIGHLIGHT)
        if prev is not None and f!=prev:p.line(x0-.8,y-pitch*.5-.4,right,y-pitch*.5-.4,.32)
        p.marker(x0,y,.67 if m in FOCUS else .55,COL[f],open_=m=='hEvo-base',shape={'Evo 2':'triangle','NT':'square','GPN':'diamond'}.get(f,'circle'))
        p.text(x0+1.7,y+.66,display(m),6.05,bold=m in FOCUS)
        prev=f

def family_legend(p,y,x=10):
    for family,width in [('hEVO',24),('Evo 2',24),('NT',20),('GPN',22),('Other models',32)]:
        key=family if family in COL else 'Other'
        p.marker(x,y-.55,.65,COL[key],shape={'Evo 2':'triangle','NT':'square','GPN':'diamond'}.get(key,'circle'))
        p.text(x+2,y,family,6.1);x+=width

def colorbar(p,x,y,w,title,lo=0,hi=1,ticks=None):
    p.text(x,y-2,title,6.)
    for i in range(100):p.rect(x+w*i/100,y,w/100+.01,2.0,heat(lo+(hi-lo)*i/99,lo,hi))
    for v in ticks if ticks is not None else [lo,(lo+hi)/2,hi]:
        xx=x+(v-lo)/(hi-lo)*w
        p.line(xx,y+2,xx,y+2.7,.3)
        p.text(xx,y+5.1,f'{v:g}',5.6,align='center')

def cell(p,x,y,w,h,v,lo=0,hi=1,number=None,best=False,color_value=None,rank=None):
    if pd.isna(v):
        p.rect(x,y-h/2,w,h,'#FFFFFF',stroke='#000000',lw=.2)
        p.text(x+w/2,y+.62,'n/a',5.5,align='center');return
    fill=heat(v if color_value is None else color_value,lo,hi)
    p.rect(x+.08,y-h/2+.04,w-.16,h-.08,fill)
    if best:p.rect(x+.12,y-h/2+.08,w-.24,h-.16,stroke='#000000',lw=.5)
    if number is not None:
        lumin=.2126*fill.red+.7152*fill.green+.0722*fill.blue
        p.text(x+w/2,y+.64,number,5.75,bold=best,color='#FFFFFF' if lumin<.48 else '#000000',align='center')
    if rank is not None and pd.notna(rank) and rank<=3:
        lumin=.2126*fill.red+.7152*fill.green+.0722*fill.blue
        p.text(x+w/2,y+.59,str(int(rank)),5.1,color='#FFFFFF' if lumin<.55 else '#303030',align='center')

def data_bar(p,x,y,w,h,v,model,hi=1.,best=False):
    """Compact zero-based bar with an explicit numeric value, on a white field.

    Family is categorical color; length alone encodes performance. No heatmap.
    """
    f=FAMILY[model]
    tone=.42 if f=='hEVO' else .29 if f!='Other' else .57
    fill=blend('#FFFFFF',COL[f],tone)
    p.rect(x+.25,y-h/2,w-.5,h,fill='#FFFFFF')
    p.rect(x+.25,y-h/2,(w-.5)*v/hi,h,fill=fill)
    if best:
        p.line(x+.25,y+h/2,x+.25+(w-.5)*v/hi,y+h/2,.55)
    p.text(x+w/2,y+.64,f'{v:.3f}',5.75,bold=best,align='center')

def load_all():
    tables={n:load(n) for n in [11,12,13,14,15,16,17,18,19,20,21]}
    for n,keys in [(17,['model','dataset']),(18,['model','dataset']),(19,['model','run']),(20,['model','dataset','class_id']),(21,['model','task'])]:
        assert not tables[n].duplicated(keys).any(),n
    assert set(tables[17].model)==set(MODELS)
    assert tables[17].groupby('model').size().eq(18).all()
    assert tables[19].groupby('model').size().eq(5).all()
    assert tables[20].groupby('model').size().eq(25).all()
    assert tables[21].groupby('model').size().eq(3).all()
    return tables

def figure_s4(t, column_scaled=False, unified_missing=False, top_three=False, show_notes=True):
    p=Page('FigureS4_all_model_sequence_classification_A4','Supplementary Figure S4. Task resolved performance of genomic sequence representations')
    gb=t[18].copy(); nt=t[17].copy()
    gb_tasks=['human_enhancers_cohn','human_enhancers_ensembl','human_ensembl_regulatory','human_nontata_promoters','human_ocr_ensembl','demo_coding_vs_intergenomic_seqs','demo_human_or_worm']
    gb_labels=['Enhancer (Cohn)','Enhancer (Ensembl)','Regulatory (Ensembl)','Non-TATA promoter','Open chromatin','Coding / intergenic','Human / worm']
    nt_tasks=['H2AFZ','H3K27ac','H3K27me3','H3K36me3','H3K4me1','H3K4me2','H3K4me3','H3K9ac','H3K9me3','H4K20me1','enhancers','enhancers_types','promoter_all','promoter_no_tata','promoter_tata','splice_sites_all','splice_sites_acceptors','splice_sites_donors']
    nt_labels=nt_tasks[:10]+['Enhancer','Enhancer types','Promoter all','Promoter non-TATA','Promoter TATA','Splice all','Splice acceptor','Splice donor']
    g=gb.pivot(index='model',columns='dataset',values='mcc_mean').reindex(index=MODELS,columns=gb_tasks)
    n=nt.pivot(index='model',columns='dataset',values='mcc_mean').reindex(index=MODELS,columns=nt_tasks)
    gr=g.rank(axis=0,ascending=False,method='min')
    nr=n.rank(axis=0,ascending=False,method='min')
    if top_three:
        save(gr.reset_index(),'S4_GB_task_ranks')
        save(nr.reset_index(),'S4_NT_task_ranks')
    if column_scaled:
        # Transform eligible values only. Raw scores remain the source of all aggregates.
        scaled=[]
        for frame,name in [(g,'GB'),(n,'NT')]:
            low=frame.min(axis=0); high=frame.max(axis=0); span=high-low
            assert span.gt(0).all(), 'Constant columns need an explicit neutral-color rule'
            z=(frame-low)/span
            assert np.array_equal(z.isna().to_numpy(),frame.isna().to_numpy())
            np.testing.assert_allclose(z.min(),0.,atol=1e-12)
            np.testing.assert_allclose(z.max(),1.,atol=1e-12)
            pd.testing.assert_frame_equal(frame.rank(),z.rank())
            save(z.reset_index(),'S4_'+name+'_within_task_minmax')
            save(pd.DataFrame({'task':frame.columns,'n_eligible_models':frame.notna().sum().values,
                               'min_MCC':low.values,'max_MCC':high.values,'range_MCC':span.values}),
                 'S4_'+name+'_normalization_parameters')
            scaled.append(z)
        gs,ns=scaled
    else:
        gs,ns=g,n
    gm7=g.mean(axis=1).where(g.notna().sum(axis=1).eq(7))
    gm5=g[gb_tasks[:5]].mean(axis=1)
    nm=n.mean(axis=1)
    save(g.reset_index(),'S4a_GB_task_MCC'); save(n.reset_index(),'S4b_NT_task_MCC')
    summary=pd.DataFrame({'model':MODELS,'display':[display(m) for m in MODELS],'family':[FAMILY[m] for m in MODELS],
                          'GB_mean_seven_tasks':gm7.values,'GB_mean_five_shared_human_tasks':gm5.values,'NT_mean_18_tasks':nm.values})
    save(summary,'S4_aggregates')
    ys=rows(MODELS,45.8)
    p.panel('a',9,12,'Genomic Benchmark')
    p.panel('b',100,12,'NT Benchmark')
    row_labels(p,MODELS,ys)
    ax=45.;aw=4.75;bx=103.;bw=4.65
    p.text(ax,18.2,'Seven tasks',6.)
    p.line(ax,21,ax+7*aw,21)
    for j,label in enumerate(gb_labels):p.text(ax+(j+.48)*aw,42.0,label,5.65,angle=65)
    for title,first,last in [('Histone marks',0,10),('Regulatory',10,15),('Splice',15,18)]:
        x0=bx+first*bw;x1=bx+last*bw
        p.text((x0+x1)/2,18.2,title,5.9,align='center');p.line(x0,21,x1-.4,21)
    for j,label in enumerate(nt_labels):p.text(bx+(j+.43)*bw,42,label,5.6,angle=65)
    for x,label in [(80.4,'Mean (7)'),(89.2,'Human (5)'),(191.0,'Mean (18)')]:p.text(x+3.8,42,label,5.7,angle=65)
    p.line(44.8,44.0,98.2,44.0,.5);p.line(102.8,44.0,200,44.0,.5)
    for m in MODELS:
        y=ys[m]
        for j,task in enumerate(gb_tasks):
            value=g.loc[m,task]
            if pd.isna(value):
                p.rect(ax+j*aw+.08,y-1.23,aw-.16,2.46,'#FFFFFF',stroke='#000000',lw=.2)
                p.text(ax+(j+.5)*aw,y+.64,'/' if unified_missing else '?' if m=='GPN-MSA' else '/',6.,align='center')
            else:cell(p,ax+j*aw,y,aw,2.55,value,best=not top_three and np.isclose(value,g[task].max()),color_value=gs.loc[m,task],rank=gr.loc[m,task] if top_three else None)
        for x,v in [(80.4,gm7[m]),(89.2,gm5[m])]:
            if column_scaled and pd.notna(v):
                p.rect(x+.2,y-.86,7.2*v,1.72,blend('#FFFFFF',COL[FAMILY[m]],.28))
            p.text(x+3.8,y+.66,f'{v:.3f}' if pd.notna(v) else '/' if unified_missing else 'n/a',5.9,bold=m in FOCUS,align='center')
        for j,task in enumerate(nt_tasks):cell(p,bx+j*bw,y,bw,2.55,n.loc[m,task],best=not top_three and np.isclose(n.loc[m,task],n[task].max()),color_value=ns.loc[m,task],rank=nr.loc[m,task] if top_three else None)
        if column_scaled:p.rect(191.2,y-.86,7.6*nm[m],1.72,blend('#FFFFFF',COL[FAMILY[m]],.28))
        p.text(195,y+.66,f'{nm[m]:.3f}',5.9,bold=m in FOCUS,align='center')
    bot=max(ys.values())+2
    p.line(8.7,bot,200,bot,.5)
    family_legend(p,bot+7)
    if top_three:
        p.text(143,bot+6.8,'1, 2, 3   Within-task rank',5.7)
    else:
        p.rect(143,bot+4.8,2.3,2.3,stroke='#000000',lw=.5);p.text(147,bot+6.8,'Highest MCC per task',5.7)
    colorbar(p,11,bot+14,37,'Within-task relative MCC' if column_scaled else 'MCC',0,1,[0,.5,1])
    if column_scaled and show_notes:
        p.text(59,bot+13.8,'Heatmap: per-task min-max scaling. Summary bars and values: raw MCC (0-1).',5.75)
        p.text(59,bot+17.6,'Human (5): the same five human tasks for every model.',5.75)
        p.text(59,bot+21.4,'/ Not shown or incomplete coverage' if unified_missing else '/ Input not supported     ? Input applicability unresolved     n/a No complete aggregate',5.65)
    elif show_notes:
        p.text(59,bot+14.5,'Human (5): identical five human tasks for every model.',5.8)
        p.text(59,bot+18.4,'/ Input not supported     ? Input applicability unresolved     n/a No complete aggregate',5.65)
    CHECKS['figures']['S4']={'models':len(MODELS),'GB_plotted_values':int(g.notna().sum().sum()),'GB_complete_seven':int(gm7.notna().sum()),'GB_common_five':int(gm5.notna().sum()),'NT_plotted_values':int(n.notna().sum().sum()),'content_bottom_mm':bot+24 if column_scaled else bot+21,'heatmap_encoding':'within-task min-max' if column_scaled else 'raw MCC','summary_columns':'raw arithmetic mean MCC; unchanged by normalization','missing_symbol':'/' if unified_missing else 'type-specific'}
    if top_three:
        CHECKS['figures']['S4']['rank_marks']={'method':'descending raw MCC; exact ties share minimum competition rank',
                                              'GB_mark_count':int(gr.le(3).sum().sum()),'NT_mark_count':int(nr.le(3).sum().sum())}
    CHECKS['figures']['S4']['explanatory_notes_on_figure']=show_notes
    if not show_notes:CHECKS['figures']['S4']['content_bottom_mm']=bot+19.1
    return p.finish()

def figure_s5(t):
    p=Page('FigureS5_regulatory_function_all_models_A4','Supplementary Figure S5. Fine grained regulatory classification and chromatin prediction')
    classes=['ELS','PLS','Insulator','Silencer','URE'];clabels=['Enhancer','Promoter','Insulator','Silencer','URE']
    f=t[20].groupby(['model','class_label']).per_class_f1.mean().unstack().reindex(index=MODELS,columns=classes)
    fmean=f.mean(axis=1)
    binary=t[19].pivot(index='model',columns='run',values='mcc_mean').reindex(MODELS)
    bmean=binary.mean(axis=1)
    ys=rows(MODELS,29.5,pitch=2.65)
    p.panel('a',9,12,'Five regulatory classes')
    p.panel('b',123,12,'URE versus enhancer')
    row_labels(p,MODELS,ys)
    for j,label in enumerate(clabels):p.text(47+10.5*j+5.25,23,label,6.,align='center')
    p.text(108.5,23,'Mean F1',6.,align='center')
    p.text(155,22.6,'Five benchmark sets',5.7,align='center')
    p.text(196,26.2,'Mean',5.9,align='center')
    p.line(46.8,27.7,116.8,27.7,.5);p.line(123,27.7,200,27.7,.5)
    plot0,plot1=126.,188.
    for m in MODELS:
        y=ys[m]
        for j,k in enumerate(classes):data_bar(p,47+j*10.5,y,10.5,1.85,f.loc[m,k],m,best=np.isclose(f.loc[m,k],f[k].max()))
        # Zero-based slim bar is a compact exact-value summary, not a rank proxy.
        p.rect(102,y-.85,13*fmean[m],1.7,blend('#FFFFFF',COL[FAMILY[m]],.23))
        p.text(115.8,y+.64,f'{fmean[m]:.3f}',5.8,bold=m in FOCUS,align='right')
        vals=binary.loc[m].to_numpy();xs=plot0+vals*(plot1-plot0)
        p.line(xs.min(),y,xs.max(),y,.55,COL[FAMILY[m]])
        # Small deterministic vertical offsets keep all five observations visible.
        for j,xx in enumerate(xs):p.marker(xx,y+[-.52,-.26,0,.26,.52][j],.40,COL[FAMILY[m]],True)
        p.marker(plot0+bmean[m]*(plot1-plot0),y,.78,COL[FAMILY[m]],open_=m=='hEvo-base')
        p.text(199.5,y+.64,f'{bmean[m]:.3f}',5.8,bold=m in FOCUS,align='right')
    bot=max(ys.values())+2
    p.line(8.7,bot,116.8,bot,.5);p.line(plot0,bot,plot1,bot,.5)
    for v in [0,.25,.5,.75,1]:
        x=plot0+(plot1-plot0)*v;p.line(x,bot,x,bot+1,.4);p.text(x,bot+3.7,f'{v:g}',5.7,align='center')
    p.text(47,bot+5.8,'F1 (0-1)',6.)
    p.text(47,bot+9.6,'Bar length and values: mean across five benchmark sets',5.6)
    p.text(157,bot+7.5,'MCC',6.3,align='center')
    p.text(126,bot+11.5,'Small circles, datasets; line, range; large circle, mean.',5.5)
    # DeepSEA: retain every source model, including low-performing recorded runs.
    deep=t[21].copy()
    alias={'NT-2.5B-1kG':'NT_2.5b_1kG','NT-2.5B-MS':'NT_2.5b_MS','NT-500M-1kG':'NT_500m_1kG'}
    deep['model']=deep.model.replace(alias)
    models=[m for m in MODELS if m in set(deep.model)]
    au=deep.pivot(index='model',columns='task',values='auroc').reindex(index=models,columns=['DHS','HM','TF'])
    ap=deep.pivot(index='model',columns='task',values='macro_auprc').reindex(index=models,columns=['DHS','HM','TF'])
    cy=bot+23
    p.panel('c',9,cy,'DeepSEA')
    p.text(79,cy-.4,'AUROC',7.,True,align='center');p.text(163,cy-.4,'Macro AUPRC',7.,True,align='center')
    for x0 in [47,132]:
        for j,label in enumerate(['DHS','Histone','TF','Mean']):p.text(x0+8+j*16.7,cy+6,label,6.,align='center')
    p.line(47,cy+8.5,113.8,cy+8.5,.5);p.line(132,cy+8.5,198.8,cy+8.5,.5)
    dys=rows(models,cy+10.5,pitch=2.65,gap=.8)
    row_labels(p,models,dys)
    for frame in [au,ap]:frame['Mean']=frame.mean(axis=1)
    for m in models:
        for frame,x0,hi in [(au,47,1.),(ap,132,.4)]:
            for j,k in enumerate(['DHS','HM','TF','Mean']):
                v=frame.loc[m,k];data_bar(p,x0+j*16.7,dys[m],16.7,1.85,v,m,hi=hi,best=np.isclose(v,frame[k].max()))
    dbot=max(dys.values())+2
    p.line(8.7,dbot,200,dbot,.5)
    p.text(47,dbot+5.4,'AUROC (0-1)',6.);p.text(132,dbot+5.4,'Macro AUPRC (0-0.4)',6.)
    p.text(47,dbot+9.4,'All bars start at zero; values are printed for direct comparison.',5.7)
    family_legend(p,dbot+16)
    p.line(143,dbot+15.7,145.5,dbot+15.7,.55);p.text(147,dbot+15.8,'Highest value per column',5.7)
    p.text(9,dbot+21,'Recorded LoRA runs; model-specific input lengths. Mean gives equal weight to DHS, Histone and TF.',5.65)
    save(f.reset_index(),'S5a_class_F1_means'); save(t[20],'S5a_all_class_dataset_F1')
    save(binary.reset_index(),'S5b_five_dataset_MCC');save(deep,'S5c_DeepSEA_source_rows')
    save(au.reset_index(),'S5c_AUROC');save(ap.reset_index(),'S5c_macro_AUPRC')
    CHECKS['figures']['S5']={'regulatory_models':34,'class_values':170,'binary_dataset_values':170,'deepsea_models':21,'deepsea_values_per_metric':63,'deepsea_valid_labels':{'DHS':125,'Histone':104,'TF':689},'content_bottom_mm':dbot+23}
    return p.finish()

def figure_s6(t):
    p=Page('FigureS6_representation_separation_all_models_A4','Supplementary Figure S6. Functional separation across annotation sources and projection methods')
    datasets=[('ENCODE cCRE v4',11,12,5,5000),('GENCODE v45',13,14,3,3000),('UCSC RefSeq',15,16,5,5000)]
    arrays={method:pd.DataFrame(index=MODELS) for method in ['PCA','tSNE']}
    for label,seednum,sum_num,nclass,nseq in datasets:
        seed=t[seednum];summary=t[sum_num]
        assert seed.groupby(['model_name','method']).size().eq(3).all()
        assert seed.n_rows.eq(nseq).all() and seed.n_classes.eq(nclass).all()
        agg=seed.groupby(['model_name','method']).ch_index.agg(['mean','std'])
        check=summary.set_index(['model_name','method']).sort_index()
        np.testing.assert_allclose(agg.sort_index()['mean'],check.ch_index_mean,rtol=1e-10)
        sd_ok=np.isclose(agg.sort_index()['std'],check.ch_index_sd,atol=1e-7,rtol=1e-7)
        if not sd_ok.all():
            mismatch=agg.sort_index().join(check[['ch_index_mean','ch_index_sd']]).loc[~sd_ok].reset_index()
            save(mismatch,f'S6_summary_SD_discrepancy_S{sum_num}')
            CHECKS.setdefault('source_discrepancies',[]).append({'table':f'S{sum_num}','rows':mismatch.to_dict('records'),'action':'Recompute from seed-level table; original source unchanged; plotted means unaffected.'})
        save(agg.reset_index(),f'S6_recomputed_seed_summary_S{seednum}')
        for method in arrays:arrays[method][label]=agg.xs(method,level='method')['mean'].reindex(MODELS)
    ranks={method:frame.rank(ascending=False,method='average') for method,frame in arrays.items()}
    ys=rows(MODELS,29.5,pitch=2.65)
    p.panel('a',9,12,'Class separation')
    p.marker(152,11,.7,'#65717F');p.text(154,11.65,'PCA',6.)
    p.marker(170,11,.7,'#65717F',True,'triangle');p.text(172,11.65,'t-SNE',6.)
    row_labels(p,MODELS,ys)
    domains=[(30,210,[40,80,160]),(10,2500,[10,100,1000]),(35,1600,[50,200,1000])]
    for j,(label,_,__,nclass,nseq) in enumerate(datasets):
        x0=47+j*53;w=44.;lo,hi,ticks=domains[j]
        fx=lambda value:x0+w*(math.log10(value)-math.log10(lo))/(math.log10(hi)-math.log10(lo))
        p.text(x0+w/2,21,label,6.3,True,align='center')
        p.text(x0+w/2,24.5,f'{nclass} classes',5.7,align='center')
        p.line(x0,27.7,x0+w,27.7,.5)
        for m in MODELS:
            x_pca=fx(arrays['PCA'].loc[m,label]);x_tsne=fx(arrays['tSNE'].loc[m,label])
            y=ys[m];color=COL[FAMILY[m]]
            p.line(x_pca,y,x_tsne,y,.48,color)
            p.marker(x_tsne,y,.72,color,True,'triangle')
            p.marker(x_pca,y,.74 if m in FOCUS else .63,color,m=='hEvo-base')
        axis_y=max(ys.values())+2
        p.line(x0,axis_y,x0+w,axis_y,.5)
        for tick in ticks:
            x=fx(tick);p.line(x,axis_y,x,axis_y+1,.4);p.text(x,axis_y+3.9,f'{tick:,}',5.7,align='center')
    bot=max(ys.values())+2
    p.text(122,bot+8.2,'Calinski-Harabasz index in 2D projections (log scale)',6.3,align='center')
    p.text(122,bot+12.2,'Paired values are means of three projection seeds.',5.7,align='center')
    top=bot+24
    p.panel('b',9,top-3,'Projection rank agreement')
    correlations=[]
    for k,(label,*_) in enumerate(datasets):
        x0=21+65*k;y0=top+8;w=45.;h=43.
        p.text(x0+w/2,top+3.5,label,6.7,True,align='center')
        # Rank 1 is at lower left. No axis inversion selected to favor a model.
        fx=lambda v:x0+(v-1)*w/33
        fy=lambda v:y0+h-(v-1)*h/33
        p.line(x0,y0+h,x0+w,y0+h,.5);p.line(x0,y0,x0,y0+h,.5)
        p.line(x0,y0+h,x0+w,y0,.4,'#000000',dash=[1.6,2.])
        for v in [1,10,20,34]:
            xx=fx(v);yy=fy(v)
            p.line(xx,y0+h,xx,y0+h+1,.35);p.text(xx,y0+h+3.9,v,5.5,align='center')
            p.line(x0-1,yy,x0,yy,.35);p.text(x0-2,yy+.65,v,5.5,align='right')
        p.text(x0+w/2,y0+h+8,'PCA rank',6.1,align='center')
        p.text(x0-7,y0+h/2,'t-SNE rank',6.1,align='center',angle=90)
        for m in sorted(MODELS,key=lambda m:(m in FOCUS,m.startswith('GPN-Star'))):
            f=FAMILY[m];r=.85 if m in FOCUS else .63
            p.marker(fx(ranks['PCA'].loc[m,label]),fy(ranks['tSNE'].loc[m,label]),r,COL[f],open_=m=='hEvo-base',shape={'Evo 2':'triangle','NT':'square','GPN':'diamond'}.get(f,'circle'))
        rho=float(np.corrcoef(ranks['PCA'][label].to_numpy(),ranks['tSNE'][label].to_numpy())[0,1])
        p.text(x0+1,y0+3.1,f'Spearman r = {rho:.2f}',5.9)
        correlations.append({'dataset':label,'models':34,'spearman_r':rho})
    end=top+8+43+15
    family_legend(p,end)
    p.marker(138,end-.6,.85,COL['hEVO']);p.text(140.5,end,display('hEvo-large'),5.9)
    p.marker(163,end-.6,.85,COL['hEVO'],True);p.text(165.5,end,display('hEvo-base'),5.9)
    p.text(137,end+4.4,'Rank 1 indicates the highest CH index.',5.7)
    save(pd.DataFrame(correlations),'S6b_projection_rank_agreement')
    for method,frame in arrays.items():save(frame.rename_axis('model').reset_index(),'S6_'+method+'_CH')
    for method,frame in ranks.items():save(frame.rename_axis('model').reset_index(),'S6_'+method+'_ranks')
    CHECKS['figures']['S6']={'models':34,'mean_CH_values':204,'seed_CH_values':612,'panels':2,'paired_CH_axis':'log10 with dataset-specific domains','correlations':correlations,'native_embedding_CH':False,'content_bottom_mm':end+7}
    return p.finish()

def qa_and_render(paths):
    pdfchecks={}
    pdftoppm=shutil.which(os.environ.get('POPPLER_PDFTOPPM', 'pdftoppm'))
    if pdftoppm is None:
        raise FileNotFoundError('pdftoppm was not found; install Poppler or set POPPLER_PDFTOPPM')
    for path in paths:
        with pdfplumber.open(path) as pdf:
            page=pdf.pages[0]
            assert len(pdf.pages)==1
            assert abs(page.width/mm-210)<.01 and abs(page.height/mm-297)<.01
            chars=page.chars
            assert len(chars)>100
            # Check visible label duplicates are intentional; source rows are independently keyed.
            pdfchecks[path.name]={'pages':1,'width_mm':page.width/mm,'height_mm':page.height/mm,'embedded_font_names':sorted({c['fontname'] for c in chars}),'text_characters':len(chars),'raster_images':len(page.images)}
            assert len(page.images)==0
        subprocess.run([pdftoppm,'-r','180','-png','-singlefile',str(path),str(PNG/path.stem)],check=True,capture_output=True)
    writer=PdfWriter()
    for path in paths:writer.append(str(path))
    combined=OUT/'FigureS4-S6_Fig2_supplements_A4.pdf'
    writer.write(str(combined));writer.close()
    assert len(PdfReader(combined).pages)==3
    CHECKS['pdf_checks']=pdfchecks
    CHECKS['combined_pdf']={'path':str(combined),'pages':3}
    CHECKS['known_limitations']=[
      'GPN-Star has no source rows for coding/intergenic and human/worm tasks; its MSA evaluator uses coordinate-aware hg38 inputs.',
      'DeepSEA has 21 recorded model runs with model-dependent max_length (512 or 1024), not a matched-length architecture comparison. GPN-MSA/Star have no S21 rows.',
      'DeepSEA TF macro metrics use 689 valid labels of 690 nominal labels.',
      'Projection seeds are not biological replicates or independently trained encoders. CH is computed in 2D, not native embedding space.',
      'Heatmaps show point estimates; available source confidence intervals are preserved in original S17-S19 tables, not converted to resample ranges.'
    ]
    print(json.dumps({'files':[str(p) for p in paths]+[str(combined)],'model_counts':[34,34,21,34],'bounds_ok':all(not v['out_of_safe_page_bounds'] for v in CHECKS['text_bounds'].values())},indent=2))

def write_captions():
    captions='''Supplementary Figure S4. Task resolved performance of genomic sequence representations.
(a) Frozen-backbone classification on seven Genomic Benchmarks tasks. (b) Frozen-backbone classification on 18 Nucleotide Transformer tasks. All 34 source model configurations are retained where applicable, in a fixed family-grouped order. Cell color denotes MCC on a shared 0-1 scale; black outlines identify the highest observed MCC in each task, not statistical significance. The seven-task mean requires all seven task values. Human (5) is the arithmetic mean across the same five human genomic tasks for every model. GPN-Star has no source records for transcript-derived coding or mixed-species human/worm tasks; '/' marks those cells. 'n/a' marks an unavailable complete aggregate. Exact estimates and reported confidence limits remain in S17-S18. Warm model-row highlights identify hEVO-large and hEVO-base only and do not change heatmap scaling.

Supplementary Figure S5. Fine grained regulatory classification and chromatin prediction.
(a) Class-specific F1 averaged over five regulatory datasets for all 34 models. ELS and PLS source labels are displayed as enhancer and promoter, consistent with Fig. 2. The mean column gives equal weight to the five classes and five datasets. Black outlines identify the largest observed value within a column, not a significance test. (b) URE versus enhancer MCC for all models across five independently constructed datasets. Small open circles represent dataset-level estimates, thin lines their observed minimum-to-maximum range, and the large symbol their mean. These are not five training seeds, and the range is not a confidence interval. The hEVO-base mean is open to distinguish it from hEVO-large. (c) Category-macro AUROC and macro AUPRC after supervised LoRA adaptation for all 21 models recorded in S21. The final mean equally weights DHS, histone and TF categories, rather than pooling all labels. Categories contain 125, 104 and 690 nominal labels; TF macro metrics use 689 valid labels. Recorded input lengths are model dependent (512 or 1,024 bp). The category rows and model-specific settings are exported without substituting frozen-probe results or imputing unavailable GPN-MSA/Star runs. Source tables S19-S21.

Supplementary Figure S6. Functional separation across annotation sources and projection methods.
(a,b) Calinski-Harabasz indices in two-dimensional PCA and t-SNE coordinates for all 34 models across ENCODE cCRE v4 (five classes, 5,000 sequences), GENCODE v45 (three classes, 3,000 sequences), and UCSC RefSeq curated (five classes, 5,000 sequences). Cell text gives the raw mean CH value across seeds 42, 43 and 44. Cell color encodes rank separately within each dataset and projection method; it is not a common absolute CH scale. Black outlines mark the top rank. The plotted CH indices were independently reconciled to seed-level S11, S13 and S15 and summary S12, S14 and S16. No native or higher-dimensional CH values are inferred. (c) Agreement between model rankings under PCA and t-SNE for each dataset, with Spearman correlation across 34 models. Rank one denotes the highest CH index; the diagonal indicates unchanged rank. Family colors follow the main figures. Projection-seed variation is not independent biological replication or encoder retraining. Full seed-level variation is retained in the source tables.
'''
    captions += '\nSource reconciliation note (not part of figure captions): S14 reports hEVO-large GENCODE t-SNE CH SD as 0.193264, whereas the sample SD recomputed from S13 seeds is approximately 0.019326. The mean, plotted values and rankings agree. Original V8 files are unchanged. Recomputed summaries and the discrepancy are retained in source_data.\n'
    captions=captions.replace('across five independently constructed datasets','across five benchmark sets')
    captions=captions.replace('Black outlines identify the largest observed value within a column, not a significance test.', 'Bars begin at zero and use the manuscript model-family colors. Bold values and a black underline identify the largest observed value within a column, not a significance test.')
    start=captions.index('Supplementary Figure S6.')
    end=captions.index('\nSource reconciliation note')
    captions=captions[:start]+'''Supplementary Figure S6. Functional separation across annotation sources and projection methods.
(a) Paired Calinski-Harabasz indices in two-dimensional PCA (circles) and t-SNE (open triangles) coordinates for all 34 models across ENCODE cCRE v4 (five classes, 5,000 sequences), GENCODE v45 (three classes, 3,000 sequences), and UCSC RefSeq curated (five classes, 5,000 sequences). Each mark is the mean across seeds 42, 43 and 44. Lines connect the two projection methods for the same model and do not denote confidence intervals. Horizontal axes are logarithmic with independently labeled ranges for the three datasets. Model color, row grouping and compact dot layouts follow the main figures; hEVO-base PCA circles are open. The plotted means were independently reconciled with seed-level S11, S13 and S15 and summary S12, S14 and S16. No native or higher-dimensional CH values are inferred, and a higher CH under a different projection is not an improvement in the underlying encoder. (b) Agreement between model rankings under PCA and t-SNE for each dataset, with Spearman correlation across 34 models. Rank one denotes the highest CH index, and the diagonal indicates unchanged rank. Projection-seed variation is not independent biological replication or encoder retraining. Full seed-level variation remains in the source tables.
''' +captions[end:]
    (OUT/'Figure_captions_and_scope.txt').write_text(captions,encoding='utf-8')

if __name__=='__main__':
    t=load_all()
    paths=[figure_s4(t),figure_s5(t),figure_s6(t)]
    write_captions();qa_and_render(paths)
