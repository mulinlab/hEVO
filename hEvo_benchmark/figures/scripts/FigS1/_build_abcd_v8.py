"""Rebuild S1a-d from V8 tables; preserve all source files and provenance limits."""

from pathlib import Path
import os, sys, json, math, hashlib, subprocess
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / 'tables'
OUT = ROOT / 'figures' / 'FigS1'
TMP = Path('/tmp/hevo_figs1_reference_v8')
for p in (OUT, TMP): p.mkdir(parents=True, exist_ok=True)
os.environ['HEVO_FIGURE_CACHE_DIR'] = str(TMP / 'cache')
sys.path.insert(0, str(ROOT / 'figures/scripts'))
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Group, Rect, Line, Circle, String, Image as RLImage
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from pypdf import PdfReader
from FigS1 import _pca as pca
from common.reportlab_common import register_arial_aliases

NAVY = '#0B2E83'
MID = '#5D8FBF'
PALE = '#CCD9EA'
FONTSIZE = 6.3
SOURCE_FRAMES = {}
BASE_RENDER_COORDS = None

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def table(n): return next(TABLES.glob(f'S{n}_*.csv'))

def audit_sources():
    global SOURCE_FRAMES
    paths = [table(n) for n in (1, 2, 3, 4, 7, 8, 9)]
    paths += list((ROOT / 'figures/scripts/FigS1').glob('*.py'))
    hashes = {str(p): sha(p) for p in paths}
    pool = pd.read_csv(table(1), dtype={'IID': str})
    final = pd.read_csv(table(2), dtype={'IID': str})
    phasing = pd.read_csv(table(3))
    chrom = pd.read_csv(table(7))
    rank_abundance = pd.read_csv(table(8))
    sharing = pd.read_csv(table(9))
    assert len(pool) == 550237 and pool.IID.is_unique
    assert len(final) == 63809 and final.IID.is_unique
    assert np.isfinite(pool[[f'PC{i}' for i in range(1, 11)]].to_numpy()).all()
    counts = pool.Group.value_counts().to_dict()
    assert counts == pca.EXPECTED_COUNTS
    matched = final.merge(pool[['IID', 'Group']], on='IID', how='left', suffixes=('_final', '_pool'), validate='one_to_one')
    assert (matched.Group_final == matched.Group_pool).all()
    assert len(phasing) == 9 and phasing.mac_bin.is_unique
    assert phasing.mean_ser_pct.between(0, 100).all() and phasing.sd_ser_pct.ge(0).all()
    assert phasing.mac_min.is_monotonic_increasing
    assert len(chrom) == 22 and set(chrom.chrom) == {f'chr{i}' for i in range(1, 23)}
    assert chrom.sample_count_in_vcf.eq(63809).all()
    assert chrom.haplotype_count_in_vcf.eq(2 * 63809).all()
    assert (chrom.raw_windows >= chrom.n_free_windows).all()
    assert (chrom.n_free_windows >= chrom.unique_windows).all()
    assert np.allclose(chrom.dedup_retention, chrom.unique_windows / chrom.raw_windows, rtol=1e-12)
    global_distinct_from_s8 = int(rank_abundance['rank'].max())
    global_distinct_from_s9 = int(sharing['count'].sum())
    assert global_distinct_from_s8 == global_distinct_from_s9
    lengths = [248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468]
    expected = np.array([(x-1024)//128+1 for x in lengths], dtype=np.int64)*127618
    assert (chrom.raw_windows.to_numpy() == expected).all()
    # Independent reproduction of the faulty original geometry, in mm.
    old_usable_y = 17 + (np.log10(chrom.n_free_windows/1e6) - math.log10(.1)) / (math.log10(4000)-math.log10(.1))*42.5
    old_candidate_y = 63.5 + (np.log10(chrom.raw_windows/1e6)-math.log10(40000)) / (math.log10(300000)-math.log10(40000))*15.5
    geom = chrom[['chrom','raw_windows','n_free_windows','unique_windows']].copy()
    geom['old_candidate_y_mm'] = old_candidate_y
    geom['old_usable_y_mm'] = old_usable_y
    geom['false_visual_reversal'] = old_usable_y > old_candidate_y
    count_frame = pd.DataFrame({'cohort': list(pca.COHORT_ORDER)})
    count_frame['preselection_n'] = count_frame.cohort.map(counts)
    count_frame['final_n'] = count_frame.cohort.map(final.Group.value_counts())
    report = {
        'pool_n':len(pool),'final_n':len(final),'preselection_counts':counts,
        'all_final_IIDs_match_source_pool':True,'PC_coordinates_finite':True,
        'PCA_explained_variance_not_available':True,
        'chromosome_counts_exactly_match_GRCh38_1024bp_stride128_127618_haplotypes':True,
        'chromosome_unique_sum':int(chrom.unique_windows.sum()),
        'global_distinct_sequence_count_from_S8_S9':global_distinct_from_s8,
        'S8_and_S9_global_distinct_counts_agree':global_distinct_from_s8 == global_distinct_from_s9,
        'chromosome_global_difference':int(chrom.unique_windows.sum()-global_distinct_from_s8),
        'candidate_window_total':int(chrom.raw_windows.sum()),
        'n_free_window_total':int(chrom.n_free_windows.sum()),
        'old_false_visual_reversal_chromosomes':geom.loc[geom.false_visual_reversal,'chrom'].tolist(),
        'c_provenance':'S3 summary reconstructed from a legacy PDF; per-window matrix, window counts and switch-test raw output are not available in this V8 snapshot. Supplied mean and sd retained, not independently recomputed.',
        'c_legacy_bounds_not_identical_to_mean_plus_minus_sd':bool(not np.allclose(phasing.legacy_lower_pct, (phasing.mean_ser_pct-phasing.sd_ser_pct).clip(lower=0))),
        'd_method':'Both raw_windows and n_free_windows count haplotype-window occurrences; unique_windows counts sequence identities within each chromosome. Not biological diversity-retention rates.',
        'panel_e':'validated separately and appended by A4_integrated.py',
    }
    print(json.dumps(report, indent=2))
    SOURCE_FRAMES = {'phasing': phasing.copy(), 'chrom': chrom.copy()}
    return pool, phasing, chrom, hashes

def text(d, x, y, value, size=FONTSIZE, bold=False, anchor='start', color='#000000', angle=0):
    s = String(0 if angle else x*mm, 0 if angle else y*mm, value, fontName='Arial-Bold' if bold else 'Arial', fontSize=size, fillColor=HexColor(color), textAnchor=anchor)
    if angle:
        g = Group(s); r = math.radians(angle)
        g.transform = (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), x*mm, y*mm)
        d.add(g)
    else: d.add(s)

def line(d,x1,y1,x2,y2,color='#000000',width=.55):
    d.add(Line(x1*mm,y1*mm,x2*mm,y2*mm,strokeColor=HexColor(color),strokeWidth=width))

def dot(d,x,y,color,r=.65,white=False):
    d.add(Circle(x*mm,y*mm,r*mm,fillColor=HexColor('#FFFFFF' if white else color),strokeColor=HexColor(color),strokeWidth=.6))

def letter(d,x,y,s):text(d,x,y,s,14,True)

def draw_pca(d,pool):
    domains=(pca.nice_domain(pool.PC1.to_numpy()),pca.nice_domain(pool.PC2.to_numpy()))
    labels=dict(pca.COHORT_LABELS); labels['1kgenome']='1000G'
    shorts=dict(pca.COHORT_SHORT_LABELS); shorts['1kgenome']='1000G'
    counts=pool.Group.value_counts()
    letter(d,9,284,'a'); letter(d,105,284,'b')
    text(d,23,284.8,'Preselection cohort',6.7,True)
    text(d,23,280.8,'n = 550,237',6.3)
    text(d,119,284.8,'Source-cohort distributions',6.7,True)
    for i,key in enumerate(pca.COHORT_ORDER):
        row,col=divmod(i,2); x=23+col*39; y=276.6-row*3.8
        dot(d,x,y+.6,pca.COHORT_COLORS[key],.62)
        label=shorts[key] if key in ('EAS','AFR','SG10K') else labels[key]
        text(d,x+1.9,y,f'{label} ({counts[key]:,})',6.1)
    box=(23,195,91,261)
    p=TMP/'a_all_PC1_PC2.png'
    pca.make_all_cohort_layer(pool,p,box[2]-box[0],box[3]-box[1],domains)
    d.add(RLImage(box[0]*mm,box[1]*mm,(box[2]-box[0])*mm,(box[3]-box[1])*mm,str(p)))
    pca.draw_axes(d,box,domains,show_x_title=True,show_y_title=True,tick_size=6.1,title_size=6.3)
    # All eight facets retain exactly the same source PCA coordinates and domains.
    for i,key in enumerate(pca.COHORT_ORDER):
        row,col=divmod(i,4); x=119+col*21; y=234 if row==0 else 195
        width=18.0; height=27.0
        text(d,x+width/2,y+height+6.0,shorts[key],6.1,anchor='middle')
        text(d,x+width/2,y+height+2.7,f'({counts[key]:,})',6.0,anchor='middle')
        p=TMP/f'b_{key}.png'
        pca.make_highlight_layer(pool,key,p,width,height,domains)
        d.add(RLImage(x*mm,y*mm,width*mm,height*mm,str(p)))
        pca.draw_axes(d,(x,y,x+width,y+height),domains,show_x_tick_labels=row==1,show_y_tick_labels=col==0,tick_size=6.1)
    text(d,159.5,188.4,'PC1 (×1,000)',6.3,anchor='middle')
    text(d,109.5,227.5,'PC2 (×1,000)',6.3,anchor='middle',angle=90)

def draw_phasing(d,phasing):
    letter(d,9,177,'c')
    text(d,23,178,'Chromosome 22 phasing',6.7,True)
    text(d,23,173.4,'UK Biobank',6.3)
    x0,x1,y0,y1=23,91,125,167
    sy=lambda v:y0+float(v)/13.5*(y1-y0)
    line(d,x0,y0,x1,y0); line(d,x0,y0,x0,y1)
    for v in range(0,13,2):
        y=sy(v);line(d,x0-.8,y,x0,y);text(d,x0-1.5,y-.7,str(v),6.1,anchor='end')
    text(d,14.8,(y0+y1)/2,'Switch error rate (%)',6.3,anchor='middle',angle=90)
    xs=np.linspace(x0+3,x1-3,len(phasing)); means=phasing.mean_ser_pct.to_numpy(); sds=phasing.sd_ser_pct.to_numpy()
    for i in range(len(xs)-1):line(d,xs[i],sy(means[i]),xs[i+1],sy(means[i+1]),NAVY,.8)
    for x,m,s,label in zip(xs,means,sds,phasing.mac_bin):
        lo,hi=sy(max(0,m-s)),sy(m+s)
        line(d,x,lo,x,hi,NAVY,.6);line(d,x-1,lo,x+1,lo,NAVY,.6);line(d,x-1,hi,x+1,hi,NAVY,.6)
        dot(d,x,sy(m),NAVY,.95)
        text(d,x,122.5,label,5.8,anchor='end',angle=48)
    text(d,(x0+x1)/2,110.3,'Minor allele count',6.3,anchor='middle')
    line(d,59,174.1,64,174.1,NAVY,.8);dot(d,61.5,174.1,NAVY,.65)
    text(d,66,173.4,'Mean ± SD',6.1)

def draw_redundancy(d,chrom):
    letter(d,105,177,'d')
    text(d,119,178,'Haplotype sequence deduplication',6.7,True)
    text(d,200,173.4,'n = 63,809',6.3,anchor='end')
    x0,x1,y0,y1=119,200,125,167
    lo,hi=.03,350.0
    sy=lambda v:y0+(math.log10(v)-math.log10(lo))/(math.log10(hi)-math.log10(lo))*(y1-y0)
    line(d,x0,y0,x1,y0);line(d,x0,y0,x0,y1)
    for v in (.1,1,10,100):
        y=sy(v);line(d,x0-.8,y,x0,y);text(d,x0-1.5,y-.7,f'{v:g}',6.1,anchor='end')
    text(d,110.8,(y0+y1)/2,'Count (billions, log scale)',6.3,anchor='middle',angle=90)
    groupw=(x1-x0)/22; bw=.80; offsets=(-.91,0,.91)
    colors=(PALE,MID,NAVY);cols=('raw_windows','n_free_windows','unique_windows')
    coords=[]
    for i,row in enumerate(chrom.itertuples(index=False)):
        x=x0+groupw*(i+.5)
        ys=[]
        for dx,color,col in zip(offsets,colors,cols):
            value=float(getattr(row,col))/1e9;y=sy(value);ys.append(y)
            d.add(Rect((x+dx-bw/2)*mm,y0*mm,bw*mm,(y-y0)*mm,fillColor=HexColor(color),strokeColor=None))
            coords.append(dict(chrom=row.chrom,metric=col,value_billions=value,x_mm=x+dx,y_mm=y))
            assert abs(10**((y-y0)/(y1-y0)*(math.log10(hi)-math.log10(lo))+math.log10(lo))-value)/value < 1e-12
        assert ys[0]>=ys[1]>=ys[2]
        text(d,x,y0-3.4,str(i+1),5.9,anchor='middle')
    text(d,(x0+x1)/2,110.3,'Chromosome',6.3,anchor='middle')
    legends=[(119,'Candidate windows',PALE),(150,'Retained windows',MID),(178,'Distinct sequences',NAVY)]
    for x,label,color in legends:
        d.add(Rect(x*mm,169.4*mm,2.5*mm,1.4*mm,fillColor=HexColor(color),strokeColor=None))
        text(d,x+3.5,169.3,label,5.8)
    return pd.DataFrame(coords)

def main():
    global BASE_RENDER_COORDS
    pool,phasing,chrom,hashes=audit_sources()
    register_arial_aliases();pca.PNG_DPI=600
    d=Drawing(210*mm,297*mm)
    d.add(Rect(0,0,d.width,d.height,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    draw_pca(d,pool);draw_phasing(d,phasing);BASE_RENDER_COORDS=draw_redundancy(d,chrom)
    dest=OUT/'FigureS1_abcd_A4_reviewed.pdf'
    renderPDF.drawToFile(d,str(dest))
    renderSVG.drawToFile(d,str(OUT/'FigureS1_abcd_A4_reviewed.svg'))
    pca.embed_svg_images(OUT/'FigureS1_abcd_A4_reviewed.svg')
    preview=OUT/'FigureS1_abcd_A4_reviewed'
    subprocess.run(['pdftoppm','-singlefile','-png','-r','180',str(dest),str(preview)],check=True)
    subprocess.run(['pdftoppm','-singlefile','-png','-r','400','-x','100','-y','1800','-W','3200','-H','1350',str(dest),str(TMP/'cd_detail')],check=True)
    page = PdfReader(dest).pages[0]
    t = page.extract_text() or ''
    assert len(PdfReader(dest).pages)==1 and abs(float(page.mediabox.width)/mm-210)<.01 and abs(float(page.mediabox.height)/mm-297)<.01
    assert '1KGP' not in t and t.count('1000G')==2 and '550,237' in t and '63,809' in t
    assert 'Global corpus assembly' not in t
    for label in ('a','b','c','d'): assert label in t.split()
    sizes=sorted({round(float(args[1]),2) for args,op in page.get_contents().operations if op==b'Tf'})
    assert min(sizes)>=5.79
    assert all(sha(Path(p))==h for p,h in hashes.items())
    print(dest)

if __name__=='__main__':main()
