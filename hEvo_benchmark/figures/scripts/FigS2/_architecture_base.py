"""Source-audited, vector-only standalone A4 hEVO model schematic.

The audit is static: source/configuration consistency, not checkpoint execution.
Training glyphs are illustrative and are never presented as measured outcomes.
"""

import tempfile
from pathlib import Path
import hashlib
import math
import json
import sys
sys.dont_write_bytecode=True
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon, Circle, Path as RPath
from reportlab.graphics import renderPDF, renderSVG
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'figures'/'FigS2'
PDF=OUT
for p in [PDF]: p.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT.parent
# Public schematic metadata, reviewed against S47. Full training configs live
# in the manually gated pretraining-code repository.
configs=[
    {'model_id':'hEvo-base','expected_parameter_count':87094922,'architecture':{
        'vocab_size':10,'hidden_size':768,'num_hidden_layers':12,
        'num_attention_heads':12,'intermediate_size':3072,
        'max_position_embeddings':1024,'use_frequency_embedding':True,
        'frequency_embedding_dim':64}},
    {'model_id':'hEvo-large','expected_parameter_count':319462410,'architecture':{
        'vocab_size':10,'hidden_size':1280,'num_hidden_layers':16,
        'num_attention_heads':20,'intermediate_size':5120,
        'max_position_embeddings':1024,'use_frequency_embedding':False,
        'frequency_embedding_dim':64}},
]

def config_count(c):
    a=c['architecture'];d=a['hidden_size'];m=a['intermediate_size'];v=a['vocab_size']
    embeddings=v*d+a['max_position_embeddings']*d+2*d+2*d
    layer=4*d*d+2*d*m+9*d+m
    pooler=d*d+d
    head=d*d+3*d+v*d+v
    f=a['frequency_embedding_dim']
    optional=(f//2+f//2+(f//2)*f+f+f*d+d+d) if a['use_frequency_embedding'] else 0
    total=embeddings+a['num_hidden_layers']*layer+pooler+head+optional
    assert total==c['expected_parameter_count'],(c['model_id'],total)
    return dict(total=total,mlm_head=head,layers=a['num_hidden_layers'],hidden=d,
                heads=a['num_attention_heads'],ffn=m,optional_frequency_branch=a['use_frequency_embedding'])

counts=[config_count(c) for c in configs]
assert [c['mlm_head'] for c in counts]==[599818,1655050]
sys.path.insert(0, str(ROOT/'figures'/'scripts'))
from common.font_config import register_reportlab_arial
register_reportlab_arial()
W,H=297,210
D=Drawing(W*mm,H*mm)
INK='#17212B';BLACK='#111111';SLATE='#8392A4';BLUE='#6D91B7'
LIGHT='#E5EAF0';PALE='#F4F7FA';TEAL='#478F98';TEAL_LIGHT='#DEEBE9'
ORANGE='#B35C1E';WARM='#F5E7D8';WARM_EDGE='#BD9A77';WARM_BG='#F7F3ED'
TEXT=[]

def text(x,y,s,size=8.0,bold=False,anchor='middle',color=INK):
    if not s:return
    font='Arial-Bold' if bold else 'Arial'
    D.add(String(x*mm,(H-y)*mm,s,fontName=font,fontSize=size,textAnchor=anchor,fillColor=HexColor(color)))
    width=pdfmetrics.stringWidth(s,font,size)/mm
    left=x if anchor=='start' else x-width if anchor=='end' else x-width/2
    TEXT.append(dict(text=s,x0=left,x1=left+width,baseline=y,size=size))

def lines(x,y,labels,size=8.0,bold=False,step=3.7,**kw):
    for i,s in enumerate(labels): text(x,y+step*i,s,size,bold,**kw)

def line(x1,y1,x2,y2,color=INK,width=.65):
    D.add(Line(x1*mm,(H-y1)*mm,x2*mm,(H-y2)*mm,strokeColor=HexColor(color),strokeWidth=width))

def box(x,y,w,h,fill=None,stroke=SLATE,width=.7,r=1):
    D.add(Rect(x*mm,(H-y-h)*mm,w*mm,h*mm,rx=r*mm,ry=r*mm,
               fillColor=HexColor(fill) if fill else None,strokeColor=HexColor(stroke) if stroke else None,strokeWidth=width))

def arrow(x1,y1,x2,y2,color=INK,width=.7,head=1.1):
    line(x1,y1,x2,y2,color,width)
    a=math.atan2(y2-y1,x2-x1)
    pts=[]
    for x,y in [(x2,y2),(x2-head*math.cos(a)+.45*math.sin(a),y2-head*math.sin(a)-.45*math.cos(a)),
                (x2-head*math.cos(a)-.45*math.sin(a),y2-head*math.sin(a)+.45*math.cos(a))]:
        pts.extend([x*mm,(H-y)*mm])
    D.add(Polygon(pts,fillColor=HexColor(color),strokeColor=None))

def matrix(x,y,w=18,h=10,nc=6,nr=3,palette=None):
    palette=palette or [LIGHT,'#BDCBD9','#FFFFFF']
    for i in range(nr):
        for j in range(nc):
            box(x+j*w/nc,y+i*h/nr,w/nc,h/nr,palette[(i*2+j)%len(palette)],SLATE,.35,0)

def sequence(x,y,n=15,w=35,h=2.5,masks=(),letters=None):
    pitch=w/n
    for j in range(n):
        fill=LIGHT if j%3 else '#BAC6D3'
        if j in masks:
            box(x+j*pitch,y,pitch-.22,h,'#FFFFFF',ORANGE,.65,.2)
            line(x+j*pitch+.3,y+h-.3,x+(j+1)*pitch-.52,y+.3,ORANGE,.5)
        else:
            box(x+j*pitch,y,pitch-.22,h,fill,None,r=.15)
        if letters: text(x+(j+.5)*pitch-.1,y+h*.74,letters[j],7.7)

def stack(x,y,w,h,fill='#FFFFFF',stroke=SLATE):
    for shift in [1.8,1.2,.6,0]:box(x+shift,y-shift,w,h,fill,stroke,.65,.9)

def lock(x,y):
    p=RPath(strokeColor=HexColor(SLATE),strokeWidth=.8,fillColor=None)
    p.moveTo((x+.6)*mm,(H-y-2.8)*mm)
    p.lineTo((x+.6)*mm,(H-y-1.3)*mm)
    p.curveTo((x+.6)*mm,(H-y+.5)*mm,(x+3.6)*mm,(H-y+.5)*mm,(x+3.6)*mm,(H-y-1.3)*mm)
    p.lineTo((x+3.6)*mm,(H-y-2.8)*mm)
    D.add(p)
    box(x,y+2.2,4.2,3.4,'#FFFFFF',SLATE,.75,.45)
    D.add(Circle((x+2.1)*mm,(H-y-3.5)*mm,.28*mm,fillColor=HexColor(SLATE),strokeColor=None))
    line(x+2.1,y+3.5,x+2.1,y+4.5,SLATE,.5)

text(7,10,'a',14,True,anchor='start')
text(16,10,'hEVO architecture and two-stage learning',10,True,anchor='start')

# Input pathway: nucleotide tokenization and the implemented sum of embeddings.
box(14,16,44,27,'#F5FAFD','#B2D9E8')
text(36,21.0,'DNA sequence',8.8,True)
sequence(19,25,n=8,w=34,h=4.2,letters=list('ACGTATCG'))
text(36,34.4,'One nucleotide per token',7.7)
text(36,40.0,'S ≤ 1,024 positions',7.7)
arrow(58,29,64,29)
box(64,16,44,27,WARM_BG,WARM_EDGE)
text(86,21,'1-mer tokenizer',8.8,True)
text(86,28,'A  T  G  C  N',8.7)
text(86,34.4,'10 tokens, including specials',7.5)
text(86,40,'No added [CLS] / [SEP]',7.5)
arrow(108,29,117,29)

for y,label,pal in [(17,'Token',[LIGHT,'#BDD0E2']),
                    (27,'Position',[TEAL_LIGHT,'#A9D1CE']),
                    (37,'Type (ID = 0)',['#F4E7CB','#E8C99B'])]:
    matrix(120,y,w=15,h=5.7,nc=5,nr=2,palette=pal)
    text(147.8,y+4,label,7.7)
    line(160,y+2.85,165,y+2.85,width=.55)
line(165,19.85,165,39.85,width=.55)
arrow(165,29,168.7,29,head=.7)
D.add(Circle(171*mm,(H-29)*mm,2.1*mm,fillColor=None,strokeColor=HexColor(INK),strokeWidth=.65))
line(169.7,29,172.3,29)
line(171,27.7,171,30.3)
arrow(173.3,29,182,29)
box(182,22,37,14,PALE,SLATE)
lines(200.5,27.5,['LayerNorm','Dropout'],8.0,step=4.2)
arrow(219,29,240,29)
matrix(244,19,27,17,nc=8,nr=4,palette=[LIGHT,'#C2D2E0',TEAL_LIGHT,'#F4E7CB'])
text(257.5,42,'Input embeddings*',8.5,True)
text(257.5,47,'S × d',8.0)

# Expanded post-normalized bidirectional Transformer block.
text(149,56,'hEVO backbone · bidirectional Transformer',9.5,True)
text(25,63,'Input embeddings',8.0)
matrix(16,69,20,15,nc=6,nr=4)
text(26,89.8,'S × d',7.7)
arrow(37.3,77,44.8,77)
stack(47,66.8,20,18)
text(57,92,'L blocks',8.0)
arrow(69.3,77,77,77)
box(77,61.5,156,34.5,WARM_BG,WARM_EDGE)
text(154,66.0,'Transformer block',8.4,True)
box(82,72,42,16,WARM,WARM_EDGE)
lines(103,76.5,['Multi-head','self-attention','FlashAttention2'],7.8,step=4.1)
arrow(124,80,130,80)
box(130,74,24,12,'#F0F7FB',WARM_EDGE)
lines(142,79,['Add &','LayerNorm'],7.5,step=3.8)
arrow(154,80,160,80)
box(160,72,40,16,WARM,WARM_EDGE)
lines(180,76.6,['Feed-forward network','Linear · GELU · Linear','d → 4d → d'],7.5,step=4.1)
arrow(200,80,206,80)
box(206,74,23,12,'#F0F7FB',WARM_EDGE)
lines(217.5,79,['Add &','LayerNorm'],7.5,step=3.8)
# Correct residual origins: input to attention, and output of its Add & Norm.
line(79.5,80,79.5,68.8)
line(79.5,68.8,142,68.8)
arrow(142,68.8,142,74,head=.85)
line(156.8,80,156.8,92.2)
line(156.8,92.2,217.5,92.2)
arrow(217.5,92.2,217.5,86,head=.85)
line(77,77,79.5,77)
line(79.5,77,79.5,80)
arrow(79.5,80,82,80,head=.7)
arrow(229,80,253.7,80)
matrix(257,73.5,26,15,nc=8,nr=4,palette=[LIGHT,'#BDCBD9','#E4BE93'])
lines(270,63,['Contextual','representation'],8.2,True,step=4.0)
text(270,93.5,'S × d',7.7)

# Configuration values come directly from canonical repository configs.
text(15,105,'Model configurations',8.6,True,anchor='start')
columns=[15,51,66,82,99]
for x,s in zip(columns,['Model','L','d','Heads','FFN']):text(x,110.6,s,7.2,True,anchor='start')
for y,name,c in [(116,'hEVO-base',counts[0]),(121.5,'hEVO-large',counts[1])]:
    for x,s in zip(columns,[name,str(c['layers']),f"{c['hidden']:,}",str(c['heads']),f"{c['ffn']:,}"]):
        text(x,y,s,7.5,anchor='start')
text(15,127,'Stored parameters: 87.095M / 319.462M',7.1,anchor='start')

# Full BertOnlyMLMHead, not a decoder-only linear approximation.
box(114,104,170,25,WARM_BG,WARM_EDGE)
text(199,109,'MLM prediction head · same head in both stages',8.5,True)
head_items=[(120,22,['Linear','d → d']), (148,17,['GELU']),
            (171,26,['LayerNorm']), (203,28,['Decoder','d → 10'])]
for x,w,labels in head_items:
    box(x,113,w,11,WARM,WARM_EDGE)
    lines(x+w/2,117 if len(labels)>1 else 119.5,labels,7.6,step=4)
for x1,x2 in [(142,148),(165,171),(197,203),(231,239)]:arrow(x1,118.5,x2,118.5,head=.9)
for i in range(10):
    box(241+i*3.7,114.5,2.9,4.2,fill=LIGHT if i<5 else '#D6AD85',stroke=SLATE,width=.35,r=.2)
text(259,123.0,'10-token logits',7.6)
# Route the representation into the head without crossing the configuration table.
line(286,80,289,80)
line(289,80,289,99.0)
line(289,99.0,116.8,99.0)
line(116.8,99.0,116.8,118.5)
arrow(116.8,118.5,120,118.5,head=.8)
line(283,80,286,80)

# Two training panels use colors semantically: orange=trainable, slate=frozen.
for x,w,title in [(14,126,'Stage I · Sequence pretraining'),(158,126,'Stage II · Allelic refinement')]:
    box(x,135,w,55,fill=None,stroke=SLATE,width=.65)
    text(x+w/2,141,title,9.1,True)
    line(x+2,143,x+w-2,143,'#B8C1CB',.45)

text(77,148,'Deduplicated haplotypes · 1,024 bp',8.0)
for y,mask in [(153.7,(3,4,5)),(158.1,(9,10,11)),(162.5,(5,6,7))]:
    sequence(20,y,n=15,w=35,h=2.5,masks=mask)
text(37.5,172,'20% span masking',7.7)
arrow(57,159.3,64.0,159.3)
stack(67,153,27,14,WARM,ORANGE)
lines(80.5,158.5,['Embeddings','+ encoder'],7.8,step=4.1)
text(80.5,172,'Train backbone',7.6,color=ORANGE)
arrow(96.5,160,104,160)
box(106,154,26,12,WARM,ORANGE)
lines(119,159.0,['MLM','prediction head'],7.6,step=4)
arrow(119,166,119,180,head=.9)
box(67,180,65,7,fill='#FFFFFF',stroke=SLATE,width=.55)
text(99.5,184.8,'Cross-entropy at masked positions',7.4)

text(221,148,'1000G SNVs · 1,023 bp',8.0)
text(184,153,'Equal sampling across 5 AF strata',7.2)
for j in range(5):box(172.0+j*4.6,155.5,3.5,2.6,fill=['#EFF2F6','#D7DEE8','#BCC7D7','#9EADC2','#7F94AF'][j],stroke=SLATE,width=.35,r=.2)
arrow(183,158.7,183,161.0,head=.6)
sequence(167,162.0,n=15,w=35,h=2.7,masks=(7,))
text(184.5,172,'Centre mask',7.7)
arrow(202.7,163.3,209.2,163.3)
stack(212,153,27,14,PALE,SLATE)
lines(225.5,158.5,['Embeddings','+ encoder'],7.8,step=4.1)
lock(238.5,149.5)
text(225.5,172,'Frozen backbone',7.6,color=SLATE)
arrow(241.5,160,248,160)
box(250,154,26,12,WARM,ORANGE)
lines(263,159.0,['Existing MLM','prediction head'],7.6,step=4)
arrow(263,166,263,180,head=.9)
box(211,180,65,7,fill='#FFFFFF',stroke=SLATE,width=.55)
text(243.5,184.8,'Cross-entropy for observed ALT',7.4)

text(149,168.5,'Transfer',7.5)
arrow(140,172,158,172,head=1)
text(149,177.5,'weights',7.5)
text(37.5,185,'Trainable',7.2,color=ORANGE)
box(20,182.5,3.8,2.8,WARM,ORANGE,.5,r=.3)
text(184.5,184,'Head parameters',7.0)
text(184.5,187.6,'Base 599,818 · Large 1,655,050',6.8)

text(149,196,'Allelic score = log P(ALT | masked context) - log P(REF | masked context)',8.2)
text(14,203,'* Base adds a learned constant offset before LayerNorm. Ancestry and measured allele frequency are not inputs to the backbone.',7.5,anchor='start')

pdf=PDF/'FigureS2a_hEVO_architecture_A4_reviewed.pdf'
svg=OUT/'FigureS2a_hEVO_architecture_A4_reviewed.svg'
renderPDF.drawToFile(D,str(pdf),title='Figure S2a | hEVO architecture and two-stage learning',author='hEVO')
renderSVG.drawToFile(D,str(svg))
r=PdfReader(pdf)
assert len(r.pages)==1 and len(r.pages[0].images)==0
txt=r.pages[0].extract_text()
for term in ['hEVO-large','16','319.462M','599,818','1,655,050','1,023 bp','20% span masking','10-token logits']:
    assert term in txt,term
assert all(t['x0']>=0 and t['x1']<=W and 0<t['baseline']<H for t in TEXT)
collisions=[(u['text'],v['text']) for i,u in enumerate(TEXT) for v in TEXT[i+1:]
            if abs(u['baseline']-v['baseline'])<2.0 and min(u['x1'],v['x1'])>max(u['x0'],v['x0'])+.3]
assert not collisions,collisions
qa={'page_mm':[W,H],'vector_only':True,'model_configuration_checks':counts,'text_bounds':TEXT,
    'potential_text_collisions':collisions,'source_review_only':True,'runtime_tests_run':False,
    'runtime_limit':'torch and transformers are not installed in the local bundled runtime; no dependencies installed',
    'parameter_count_method':'Symbolic sum of configured module dimensions, including stored pooler and optional Base branch'}
sources=[SOURCE/'hEvo_benchmark/tables/S47_architecture_training_scope_and_provenance_of_evaluated_models.csv']
print(pdf)
print(json.dumps({'configuration_checks':counts,'potential_text_collisions':collisions},indent=2))
