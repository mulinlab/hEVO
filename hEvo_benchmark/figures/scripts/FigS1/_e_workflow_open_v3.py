"""Simplify the original open S1e composition, preserving its glyph-first design."""
from pathlib import Path
import math
import json
import sys
sys.dont_write_bytecode = True
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon, Circle
from reportlab.graphics import renderPDF, renderSVG
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'figures'/'FigS1'
PDF=OUT
from common.font_config import register_reportlab_arial
register_reportlab_arial()
W,H=188,48
D=Drawing(W*mm,H*mm)
INK='#161616'
SLATE='#8A95A5'
MID='#AFBAC8'
LIGHT='#E8ECF1'
ORANGE='#B35C1E'
TEXT=[]

def text(x,y,s,size=6.2,bold=False,anchor='middle',color=INK):
    if not s:
        return
    font='Arial-Bold' if bold else 'Arial'
    D.add(String(x*mm,(H-y)*mm,s,fontName=font,fontSize=size,textAnchor=anchor,fillColor=HexColor(color)))
    w=pdfmetrics.stringWidth(s,font,size)/mm
    left=x if anchor=='start' else x-w if anchor=='end' else x-w/2
    TEXT.append(dict(text=s,x0=left,x1=left+w,baseline=y,size=size))

def line(x1,y1,x2,y2,color=INK,width=.55):
    D.add(Line(x1*mm,(H-y1)*mm,x2*mm,(H-y2)*mm,strokeColor=HexColor(color),strokeWidth=width))

def box(x,y,w,h,fill=None,stroke=SLATE,width=.55,r=.55):
    D.add(Rect(x*mm,(H-y-h)*mm,w*mm,h*mm,rx=r*mm,ry=r*mm,
               fillColor=HexColor(fill) if fill else None,
               strokeColor=HexColor(stroke) if stroke else None,strokeWidth=width))

def arrow(x1,y1,x2,y2,color=INK,width=.6,head=.95):
    line(x1,y1,x2,y2,color,width)
    a=math.atan2(y2-y1,x2-x1)
    pts=[]
    for x,y in [(x2,y2),(x2-head*math.cos(a)+.4*math.sin(a),y2-head*math.sin(a)-.4*math.cos(a)),
                (x2-head*math.cos(a)-.4*math.sin(a),y2-head*math.sin(a)+.4*math.cos(a))]:
        pts.extend([x*mm,(H-y)*mm])
    D.add(Polygon(pts,fillColor=HexColor(color),strokeColor=None))

def sequence(x,y,pattern,w=24,h=2.1,outline=False):
    step=w/len(pattern)
    for j,v in enumerate(pattern):
        color=ORANGE if v else MID if j%3==0 else LIGHT
        box(x+j*step,y,step-.18,h,color,stroke=None,r=0)
    if outline:
        box(x-.4,y-.45,w+.62,h+.9,width=.5,r=.3)

def person(x,y,color=SLATE):
    D.add(Circle(x*mm,(H-y)*mm,.65*mm,fillColor=HexColor(color),strokeColor=None))
    box(x-.9,y+.95,1.8,2.15,color,stroke=None,r=.42)

def people(x,y):
    for dx,dy in [(-3.5,.3),(0,0),(3.5,.3)]:
        person(x+dx,y+dy)

def bracket(x1,x2,y,label=None):
    line(x1,y,x2,y,width=.45)
    for x in [x1,x2]:
        line(x,y-.5,x,y+.5,width=.45)
    if label:
        text((x1+x2)/2,y-1.2,label,6.0)

# Same open five-step composition as the first version; no large cards.
text(.9,5.7,'e',14,True,anchor='start')
for x,s in [(25,'Cohort assembly'),(66,'Haplotype reconstruction'),
            (105,'Overlapping windows'),(144,'Global deduplication'),(177,'Stage I input')]:
    text(x,5.7,s,6.7,True)

# Cohort glyphs are schematic and do not encode sample counts.
text(25,11.2,'550,237 samples · 8 cohorts',6.1)
text(25,15.2,'QC + duplicate removal',6.0)
text(15.2,20.0,'UKB',6.1)
text(34.5,20.0,'Other cohorts',6.1)
people(15.2,22.2)
people(34.5,22.2)
arrow(15.2,26.2,15.2,28.5,head=.7)
arrow(34.5,26.2,34.5,28.5,head=.7)
text(15.2,30.4,'UKB ancestry-code',5.0)
text(15.2,32.0,'selection',5.0)
text(34.5,31.2,'Retain eligible',6.0)
line(15.2,32.3,15.2,34.2)
line(34.5,32.3,34.5,34.2)
line(15.2,34.2,34.5,34.2)
arrow(25,34.2,25,37.3,head=.7)
text(25,40.7,'63,809 individuals',6.4,True)

# Phasing and reconstruction, not a model-embedding diagram.
text(66,11.2,'GRCh38 + phased SNVs',6.1)
text(52.8,19.8,'Ref.',5.9,anchor='end')
sequence(54,17.8,[0]*10,w=24)
arrow(66,21.0,66,23.0,head=.7)
text(52.8,26.5,'H1',5.9,anchor='end')
sequence(54,24.5,[0,1,0,0,0,0,1,0,0,0],w=24)
text(52.8,33.2,'H2',5.9,anchor='end')
sequence(54,31.2,[0,0,0,1,0,0,0,0,1,0],w=24)
text(66,40.7,'Preserve cis variants',6.0)

# Windows retain the original proportional length / step relationship.
text(105,11.2,'Autosomes (chr1-22)',6.1)
pat=[0,0,1,0,0,0,1,0,0,1]
bracket(91,115,18.1,'1,024 bp')
for k,y in enumerate([21.4,26.5,31.6]):
    sequence(91+3*k,y,pat[k:k+8],w=24,h=2.0,outline=True)
line(91,34.8,91,37.1,SLATE,.45)
line(94,34.8,94,37.1,SLATE,.45)
bracket(91,94,37.1)
text(105,40.7,'128-bp stride',6.0)

# Exact duplicates merge; alternative sequence identities remain visible.
text(144,11.2,'Exact sequence identity',6.1)
a=[0,0,1,0,0,0]
b=[0,0,0,0,1,0]
sequence(128,19.0,a,w=12,h=1.9)
sequence(128,24.6,a,w=12,h=1.9)
sequence(128,32.0,b,w=12,h=1.9)
line(141,19.95,143,19.95)
line(141,25.55,143,25.55)
line(143,19.95,143,25.55)
arrow(143,22.75,147,22.75,head=.7)
sequence(148,21.8,a,w=11,h=1.9)
arrow(141,32.95,147,32.95,head=.7)
sequence(148,32.0,b,w=11,h=1.9)
text(144,40.7,'4.784B distinct sequences',6.1,True)

# Packed sequence records and one sequence-only output, no training diagram.
text(177,11.2,'Equal-weight sampling',6.0)
for offset in [1.1,.55,0]:
    box(166+offset,18.3-offset,20,14.0,fill='#FFFFFF')
for y,pat in [(20.3,a),(24.3,b),(28.3,[0,1,0,0,0,0])]:
    sequence(168,y,pat,w=16,h=2.0)
text(177,36.0,'2-bit storage',6.0)
arrow(177,37.2,177,39.8,head=.8)
sequence(166,41.3,[0,1,0,0,0,0,1,0],w=21,h=2.3)

# Separate metadata sidecar: a small table glyph, not another large container.
arrow(144,42.0,144,44.0,head=.7)
box(127.2,44.5,5.2,2.65,fill=None,stroke=SLATE,width=.4,r=0)
for x in [128.9,130.6]:
    line(x,44.5,x,47.15,SLATE,.35)
line(127.2,45.4,132.4,45.4,SLATE,.35)
line(127.2,46.3,132.4,46.3,SLATE,.35)
text(134,46.8,'Metadata: counts, ancestry, loci',5.9,anchor='start')

for x1,x2 in [(44.1,48),(80.5,87.5),(121.5,126),(160,164.5)]:
    arrow(x1,26.5,x2,26.5,head=.95)


from common.paths import tables_dir as resolve_tables_dir, figure_dir
from common.panel_runtime import input_files, panel_output_dir
from common.reportlab_common import export_drawing
import pandas as pd

STEM = 'e_cohort_haplotype_workflow'

def validate_sources(tables_dir=None):
    root = resolve_tables_dir(tables_dir)
    source1 = next(root.glob('S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv'))
    source7 = next(root.glob('S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv'))
    n_preselection = sum(1 for _ in source1.open(encoding='utf-8')) - 1
    chrom = pd.read_csv(source7)
    if n_preselection != 550237:
        raise ValueError(f'S1 preselection rows changed: {n_preselection}')
    if len(chrom) != 22 or not chrom.sample_count_in_vcf.eq(63809).all():
        raise ValueError('S7 must retain 22 autosomes and 63,809 final-corpus samples')
    distinct = int(chrom.unique_windows.sum())
    if distinct != 4784082124:
        raise ValueError(f'S7 distinct sequence count changed: {distinct}')
    labels = [row['text'] for row in TEXT]
    for value in ('550,237 samples · 8 cohorts','UKB ancestry-code','selection',
                  '63,809 individuals','4.784B distinct sequences'):
        if value not in labels:
            raise ValueError(f'Workflow label is missing: {value}')
    return [source1, source7]

def build_panel(tables_dir=None):
    validate_sources(tables_dir)
    return D

def export_panel(tables_dir=None, output_dir=None):
    sources = validate_sources(tables_dir)
    out = panel_output_dir('FigS1', output_dir)
    return export_drawing(D, out, STEM, input_files(*sources), width_mm=W, height_mm=H,
                          png_dpi=600)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(export_panel(args.tables_dir, args.output_dir))
