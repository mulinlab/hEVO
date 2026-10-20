"""Edit table-backed data elements while retaining measured PDF typography/layout.

The supplied PDF is a local style asset, never a quantitative data source. Figure
renderers explicitly replace each data mark/value with values from local S tables.
Fonts, static labels, colors, and page-space geometry come from the reference.
"""
from __future__ import annotations
import copy
import math
import subprocess
from pathlib import Path
import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, ByteStringObject, NameObject
from reportlab.pdfbase import pdfmetrics
from .font_config import register_reportlab_arial


def matrix(a):
    a,b,c,d,e,f=map(float,a)
    return np.array([[a,c,e],[b,d,f],[0,0,1.]])


class ReferencePage:
    def __init__(self, path):
        register_reportlab_arial()
        self.reader=PdfReader(path)
        self.page=self.reader.pages[0]
        self.stream=ContentStream(self.page.get_contents(),self.reader)
        self.ops=self.stream.operations
        self.height=float(self.page.mediabox.height)
        self.texts=[];self.paths=[]
        state=dict(ctm=np.eye(3),fill=(0.,0.,0.),stroke=(0.,0.,0.),fill_op=None,
                   stroke_op=None,font=None,font_op=None,size=12.,tm=np.eye(3),tm_op=None,width=1.)
        stack=[];indices=[]
        for i,(a,op) in enumerate(self.ops):
            if op==b'q': stack.append(copy.deepcopy(state))
            elif op==b'Q': state=stack.pop()
            elif op==b'cm':state['ctm']=state['ctm']@matrix(a)
            elif op in (b'rg',b'RG'):
                k='fill' if op==b'rg' else 'stroke';state[k]=tuple(map(float,a));state[k+'_op']=i
            elif op==b'w':state['width']=float(a[0])
            elif op==b'Tf':state.update(font=str(a[0]),size=float(a[1]),font_op=i)
            elif op==b'Tm':state.update(tm=matrix(a),tm_op=i)
            elif op==b'Tj':
                t=str(a[0]) if not isinstance(a[0],bytes) else bytes(a[0]).decode('latin1')
                origin=(state['ctm']@state['tm'])[:2,2]
                self.texts.append(dict(index=i,text=t,x=float(origin[0]),y=float(origin[1]),**copy.deepcopy(state)))
            elif op==b'n':indices=[]
            elif op in (b'm',b'l',b'c',b're',b'h'):indices.append(i)
            elif op in (b'S',b'f',b'f*',b'B',b'B*',b'b',b'b*'):
                points=[]
                for j in indices:
                    aa,oo=self.ops[j]
                    if oo==b're':
                        x,y,w,h=map(float,aa);ps=[(x,y),(x+w,y+h)]
                    elif oo in (b'm',b'l',b'c'):ps=list(zip(map(float,aa[::2]),map(float,aa[1::2])))
                    else:continue
                    points.extend([(state['ctm']@np.array([x,y,1]))[:2] for x,y in ps])
                if points:
                    p=np.array(points);lo=p.min(axis=0);hi=p.max(axis=0)
                    self.paths.append(dict(index=i,indices=indices[:],bbox=tuple([*lo,*hi]),
                        x=float((lo[0]+hi[0])/2),y=float((lo[1]+hi[1])/2),
                        paint=op,vertices=sum(self.ops[j][1] in (b'm',b'l') for j in indices),**copy.deepcopy(state)))
                indices=[]

    def xy(self,p):return p['x'],self.height-p['y']

    def set_text(self,t,value,align='center',bold=None,color=None):
        value=str(value);old=t['text']
        font='Arial-Bold' if 'Bold' in str(self.page['/Resources']['/Font'][t['font']].get_object().get('/BaseFont')) else 'Arial'
        oldw=pdfmetrics.stringWidth(old,font,t['size'])
        newfont=font if bold is None else 'Arial-Bold' if bold else 'Arial'
        neww=pdfmetrics.stringWidth(value,newfont,t['size'])
        if align in ('center','right'):
            delta=(oldw-neww)*(0.5 if align=='center' else 1.)
            a,op=self.ops[t['tm_op']];a=list(a);a[4]=FloatObject(float(a[4])+delta);self.ops[t['tm_op']]=(a,op)
        self.ops[t['index']]=([ByteStringObject(value.encode('latin1'))],b'Tj')
        if bold is not None:
            desired='Arial-BoldMT' if bold else 'ArialMT'
            key=next(k for k,v in self.page['/Resources']['/Font'].items() if str(v.get_object().get('/BaseFont','')).endswith(desired))
            a,op=self.ops[t['font_op']];self.ops[t['font_op']]=([NameObject(key),a[1]],op)
        if color is not None:self.color(t,color)

    def color(self,p,color,stroke=False):
        i=p['stroke_op' if stroke else 'fill_op'];a,op=self.ops[i]
        self.ops[i]=([FloatObject(v) for v in color],op)

    def transform(self,p,fn):
        ctm=p['ctm'];inverse=np.linalg.inv(ctm)
        for i in p['indices']:
            a,op=self.ops[i];a=list(map(float,a))
            if op in (b'm',b'l',b'c'):
                for j in range(0,len(a),2):
                    xy=(ctm@np.array([a[j],a[j+1],1]))[:2];xx,yy=fn(*xy)
                    xy=(inverse@np.array([xx,yy,1]))[:2];a[j:j+2]=xy.tolist()
            elif op==b're':
                x,y,w,h=a;v=(ctm@np.array([x,y,1]))[:2];v2=(ctm@np.array([x+w,y+h,1]))[:2]
                q=(inverse@np.array([*fn(*v),1]))[:2];q2=(inverse@np.array([*fn(*v2),1]))[:2]
                a=[*q,*(q2-q)]
            else:continue
            self.ops[i]=([FloatObject(v) for v in a],op)

    def move(self,p,x=None,y=None):
        dx=0 if x is None else x-p['x'];dy=0 if y is None else y-p['y']
        self.transform(p,lambda xx,yy:(xx+dx,yy+dy))

    def line(self,p,x0,x1):
        lo,_,hi,_=p['bbox'];w=hi-lo
        if w<1e-6:self.move(p,x=(x0+x1)/2)
        else:self.transform(p,lambda xx,yy:(x0+(xx-lo)/w*(x1-x0),yy))

    def remove(self,p):
        self.ops[p['index']]=([],b'n')

    def add_text(self,x,y,value,size=4.15,font='/F3+0',color=(.19,.19,.19)):
        f=lambda v:FloatObject(float(v))
        self.ops.extend([([],b'q'),([f(v) for v in color],b'rg'),([],b'BT'),
                         ([f(1),f(0),f(0),f(1),f(x),f(y)],b'Tm'),
                         ([NameObject(font),f(size)],b'Tf'),
                         ([ByteStringObject(str(value).encode('latin1'))],b'Tj'),([],b'ET'),([],b'Q')])

    def finish(self,out):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        self.stream.operations=self.ops;self.page.replace_contents(self.stream)
        writer=PdfWriter();writer.add_page(self.page)
        writer.add_metadata({'/Title':out.name,'/Author':'hEVO','/Subject':'Data read from local supplementary tables; reference PDF geometry'})
        path=out/'A4_integrated.pdf'
        with path.open('wb') as f:writer.write(f)
        subprocess.run(['pdftoppm','-f','1','-l','1','-singlefile','-png','-r','600',str(path),str(out/'A4_integrated')],check=True,capture_output=True)
        subprocess.run(['pdftocairo','-f','1','-l','1','-svg',str(path),str(out/'A4_integrated.svg')],check=True,capture_output=True)
        return path


def load_table(root,n):
    import pandas as pd
    return pd.read_csv(next(Path(root).glob(f'S{n}_*.csv')))


def nearest(values,v):return int(np.argmin(np.abs(np.asarray(values)-v)))


def small(p):
    x0,y0,x1,y1=p['bbox'];return x1-x0<8 and y1-y0<8


def colored(p):return max(p['stroke'])-min(p['stroke'])>.1 or max(p['fill'])-min(p['fill'])>.1

def marker_center(p):
    """Recover the geometric center of the reference's stars/triangles/circles."""
    x0,y0,x1,y1=p['bbox']
    # Each marker is stored in one path; its original vertex count distinguishes it.
    count=p.get('vertices',0)
    if count==3:return (x0+x1)/2,y0+(y1-y0)/2.2
    if count==10:return (x0+x1)/2,y1-(y1-y0)/(1+math.cos(math.pi/5))
    return (x0+x1)/2,(y0+y1)/2



