"""Requested title-free, rebalanced S1a-d. Data/definitions unchanged."""

from pathlib import Path
import sys, inspect, json, hashlib, subprocess
sys.dont_write_bytecode=True
ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = ROOT / 'figures' / 'scripts'
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from FigS1 import _build_abcd_v8 as B
import pandas as pd
import numpy as np
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Rect, Image as RLImage
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

HERE=Path(__file__).resolve().parent
OUT=B.OUT
TMP=Path('/tmp/hevo_figs1_reference_v8/layout_v2')
TMP.mkdir(parents=True,exist_ok=True)
B.TMP=TMP

def patch(name, pairs):
    src=inspect.getsource(getattr(B,name))
    for old,new in pairs:
        assert old in src,(name,old)
        src=src.replace(old,new)
    exec(compile(src,'<S1 layout-only v2>','exec'),B.__dict__)

def draw_pca(d,pool):
    domains=(B.pca.nice_domain(pool.PC1.to_numpy()),B.pca.nice_domain(pool.PC2.to_numpy()))
    labels=dict(B.pca.COHORT_LABELS);labels['1kgenome']='1000G'
    shorts=dict(B.pca.COHORT_SHORT_LABELS);shorts['1kgenome']='1000G'
    counts=pool.Group.value_counts()
    B.letter(d,9,284,'a');B.letter(d,85,284,'b')
    B.text(d,23,280.4,'n = 550,237',6.3)
    for i,key in enumerate(B.pca.COHORT_ORDER):
        row,col=divmod(i,2);x=23+col*30.5;y=273.8-row*3.6
        B.dot(d,x,y+.6,B.pca.COHORT_COLORS[key],.62)
        label=shorts[key] if key in ('EAS','AFR','SG10K') else labels[key]
        B.text(d,x+1.9,y,f'{label} ({counts[key]:,})',6.1)
    box=(23,202,77,257)
    p=TMP/'a_all_PC1_PC2.png'
    B.pca.make_all_cohort_layer(pool,p,box[2]-box[0],box[3]-box[1],domains)
    d.add(RLImage(box[0]*mm,box[1]*mm,(box[2]-box[0])*mm,(box[3]-box[1])*mm,str(p)))
    B.pca.draw_axes(d,box,domains,show_x_title=True,show_y_title=True,tick_size=6.1,title_size=6.3)
    for i,key in enumerate(B.pca.COHORT_ORDER):
        row,col=divmod(i,4);x=99+col*26.5;y=247 if row==0 else 202
        width,height=22.5,31.0
        B.text(d,x+width/2,y+height+6,shorts[key],6.1,anchor='middle')
        B.text(d,x+width/2,y+height+2.7,f'({counts[key]:,})',6.0,anchor='middle')
        p=TMP/f'b_{key}.png'
        B.pca.make_highlight_layer(pool,key,p,width,height,domains)
        d.add(RLImage(x*mm,y*mm,width*mm,height*mm,str(p)))
        B.pca.draw_axes(d,(x,y,x+width,y+height),domains,show_x_tick_labels=row==1,show_y_tick_labels=col==0,tick_size=6.1)
    B.text(d,150,195.4,'PC1 (×1,000)',6.3,anchor='middle')
    B.text(d,89.5,240,'PC2 (×1,000)',6.3,anchor='middle',angle=90)

def main():
    # Reuse the preceding source review; this turn changes layout only.
    previous_source_frames=B.SOURCE_FRAMES
    pool,phasing,chrom,hashes=B.audit_sources()
    assert all(B.sha(Path(p))==h for p,h in hashes.items())
    pd.testing.assert_frame_equal(phasing,previous_source_frames['phasing'])
    pd.testing.assert_frame_equal(chrom,previous_source_frames['chrom'])
    patch('draw_phasing',[
        ("    text(d,23,178,'Chromosome 22 phasing',6.7,True)\n",''),
        ("letter(d,9,177,'c')","letter(d,9,185,'c')"),
        ('23,91,125,167','23,91,134,176'),
        ('173.4','180.5'),('174.1','181.2'),('122.5','131.5'),('110.3','119.3'),
    ])
    patch('draw_redundancy',[
        ("    text(d,119,178,'Haplotype sequence deduplication',6.7,True)\n",''),
        ("letter(d,105,177,'d')","letter(d,105,185,'d')"),
        ('119,200,125,167','119,200,134,176'),
        ('173.4','185.6'),('169.4','180.6'),('169.3','180.5'),('110.3','119.3'),
    ])
    B.register_arial_aliases();B.pca.PNG_DPI=600
    d=Drawing(210*mm,297*mm)
    d.add(Rect(0,0,d.width,d.height,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    draw_pca(d,pool);B.draw_phasing(d,phasing);new_coords=B.draw_redundancy(d,chrom)
    dest=OUT/'FigureS1_abcd_A4_compact_v2.pdf'
    renderPDF.drawToFile(d,str(dest))
    svg=dest.with_suffix('.svg');renderSVG.drawToFile(d,str(svg));B.pca.embed_svg_images(svg)
    preview=OUT/'FigureS1_abcd_A4_compact_v2'
    subprocess.run(['pdftoppm','-singlefile','-png','-r','180',str(dest),str(preview)],check=True)
    forbidden=['Preselection cohort','Source-cohort distributions','Chromosome 22 phasing','Haplotype sequence deduplication']
    reader=PdfReader(dest);p=reader.pages[0];t=p.extract_text() or ''
    assert len(reader.pages)==1 and abs(float(p.mediabox.width)/mm-210)<.01 and abs(float(p.mediabox.height)/mm-297)<.01
    assert all(s not in t for s in forbidden)
    assert t.count('1000G')==2 and '550,237' in t and '63,809' in t
    for label in 'abcd': assert label in t.split()
    assert ' e ' not in f' {t} '
    old=B.BASE_RENDER_COORDS
    new=new_coords
    pd.testing.assert_frame_equal(old[['chrom','metric','value_billions']],new[['chrom','metric','value_billions']])
    assert np.allclose(old.y_mm+9,new.y_mm) and np.allclose(old.x_mm,new.x_mm)
    qa=dict(source_data_unchanged=True,phasing_summary_unchanged=True,
        panel_d_counts_and_scale_unchanged=True,removed_subtitles=forbidden,
        panel_a_plot_mm=[54,55],panel_a_previous_plot_mm=[68,66],
        panel_b_facet_mm=[22.5,31],panel_b_previous_facet_mm=[18,27],
        panel_b_area_increase_percent=(22.5*31/(18*27)-1)*100,
        ab_bottom_aligned_mm=202,cd_bottom_aligned_mm=134,source_review='reference V8 Figure S1 review; source rows checked against local tables')
    print(dest)
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
