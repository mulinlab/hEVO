#!/usr/bin/env python3
from __future__ import annotations

# [Shared QA] Compare PNG dimensions, pixel differences, and tolerated error to ensure annotation or layout edits did not unintentionally change the figure.
# [Note] This QA utility is not part of the formal Fig. 1–4 plotting workflow and does not modify source data.
import argparse
from pathlib import Path
from PIL import Image, ImageChops, ImageStat

def main():
    p=argparse.ArgumentParser()
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    args=p.parse_args()
    ref=Image.open(args.reference).convert('RGB')
    cand=Image.open(args.candidate).convert('RGB')
    same_size=ref.size==cand.size
    if not same_size:
        cand_for_diff=cand.resize(ref.size)
    else:
        cand_for_diff=cand
    diff=ImageChops.difference(ref,cand_for_diff)
    stat=ImageStat.Stat(diff)
    mae=sum(stat.mean)/3.0
    rms=(sum(v*v for v in stat.rms)/3.0)**0.5
    extrema=diff.getextrema()
    max_diff=max(ch[1] for ch in extrema)
    print(f'reference={args.reference}')
    print(f'candidate={args.candidate}')
    print(f'reference_size={ref.size[0]}x{ref.size[1]}')
    print(f'candidate_size={cand.size[0]}x{cand.size[1]}')
    print(f'same_size={same_size}')
    print(f'mae={mae:.4f}')
    print(f'rms={rms:.4f}')
    print(f'max_diff={max_diff}')
if __name__ == '__main__':
    main()
