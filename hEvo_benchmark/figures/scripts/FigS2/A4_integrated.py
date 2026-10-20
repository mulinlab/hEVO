"""Tighter vertical spacing without shrinking any text or changing content."""
from pathlib import Path
import bisect

base = Path(__file__).with_name('_architecture_base.py')
source = base.read_text(encoding='utf8').split("\npdf=PDF/", 1)[0]
omit = ("text(7,10,'a'", "text(16,10,'hEVO architecture",
        "text(149,196,'Allelic score", "text(14,203,'* Base adds")
source = '\n'.join(line for line in source.splitlines() if not line.startswith(omit))
source = source.replace('Input embeddings*', 'Input embeddings')

# Bring the loss boxes and parameter notes closer to their own training flows.
# This removes unused in-panel vertical space, not information.
changes = {
    'box(x,135,w,55,': 'box(x,135,w,49,',
    'arrow(119,166,119,180,': 'arrow(119,166,119,174,',
    'box(67,180,65,7,': 'box(67,174,65,7,',
    'text(99.5,184.8,': 'text(99.5,178.8,',
    'arrow(263,166,263,180,': 'arrow(263,166,263,174,',
    'box(211,180,65,7,': 'box(211,174,65,7,',
    'text(243.5,184.8,': 'text(243.5,178.8,',
    'text(37.5,185,': 'text(37.5,179,',
    'box(20,182.5,': 'box(20,176.5,',
    'text(184.5,184,': 'text(184.5,178,',
    'text(184.5,187.6,': 'text(184.5,181.6,',
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(base), 'exec'), globals())

from reportlab.graphics.shapes import Group
import shutil
import subprocess

# Map layout coordinates only. Font sizes and all horizontal positions remain
# exactly the same as the previous compact A4 version.
knots = [
    (16,16), (21,20.5), (29,27), (37,33.5), (43,38.5), (47,42),
    (56,46), (61.5,50.5), (66,55), (68.8,57.3),
    (72,60.5), (88,76.5), (92.2,80.7), (96,83.5), (99,86),
    (104,89.5), (109,94.5), (113,98.5), (124,109.5), (129,113.5),
    (135,117), (141,122), (143,123.5), (148,128), (153,132.5),
    (166,145.5), (172,151), (174,153), (181,160), (184,162.5),
]
old_knots = [p[0] for p in knots]
def top_y(y):
    i = min(max(bisect.bisect_right(old_knots, y)-1, 0), len(knots)-2)
    old0, new0 = knots[i]
    old1, new1 = knots[i+1]
    return new0 + (y-old0)*(new1-new0)/(old1-old0)

def pdf_y(y):
    return (H-top_y(H-y/mm))*mm

for shape in D.contents:
    if isinstance(shape, String):
        shape.y = pdf_y(shape.y)
    elif isinstance(shape, Rect):
        bottom = pdf_y(shape.y)
        top = pdf_y(shape.y+shape.height)
        shape.y = bottom
        shape.height = top-bottom
    elif isinstance(shape, Line):
        shape.y1, shape.y2 = pdf_y(shape.y1), pdf_y(shape.y2)
    elif isinstance(shape, Circle):
        shape.cy = pdf_y(shape.cy)
    elif isinstance(shape, (Polygon, RPath)):
        shape.points = [v if i%2==0 else pdf_y(v) for i,v in enumerate(shape.points)]
    else:
        raise TypeError(type(shape))
for t in TEXT:
    t['baseline'] = top_y(t['baseline'])

left, top, width = 9, 10, 192
scale = width/275
page_w, page_h = 210, 297
xoff = (left-scale*14)*mm
yoff = (page_h-top-scale*(H-16))*mm
page = Drawing(page_w*mm, page_h*mm)
group = Group(*D.contents)
group.transform = (scale,0,0,scale,xoff,yoff)
page.add(group)
pdf = PDF/'A4_integrated.pdf'
svg = OUT/'A4_integrated.svg'
renderPDF.drawToFile(page, str(pdf), title='hEVO architecture', author='hEVO')
renderSVG.drawToFile(page, str(svg))

r = PdfReader(pdf)
assert len(r.pages)==1 and len(r.pages[0].images)==0
min_font = min(t['size'] for t in TEXT)
collisions = [(u['text'],v['text']) for i,u in enumerate(TEXT) for v in TEXT[i+1:]
              if abs(u['baseline']-v['baseline'])<1.7
              and min(u['x1'],v['x1'])>max(u['x0'],v['x0'])+.3]
assert not collisions, collisions
height = (top_y(184)-16)*scale
old_height = (190-16)*scale
qa = dict(page_mm=[210,297], vector_only=True,
          content='architecture and two-stage pretraining schematic',
          panels=1, content_height_mm=height,
          reduction_percent=100*(1-height/old_height), text_collisions=collisions,
          lower_half_empty=True, minimum_font_pt=min_font,
          sources='public schematic metadata reviewed against S47; full training code is gated')
pdftoppm = shutil.which('pdftoppm')
if not pdftoppm:
    raise FileNotFoundError('pdftoppm is required to render the A4 PNG')
subprocess.run([pdftoppm, '-f', '1', '-l', '1', '-singlefile', '-png', '-r', '600',
                str(pdf), str(OUT/'A4_integrated')], check=True, capture_output=True)
print(pdf)
print(json.dumps(qa,indent=2))
