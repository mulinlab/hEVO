"""Reproduce FigS12 using local supplementary tables and reference PDF styling."""
from pathlib import Path
import sys
import argparse
HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[1]))
from FigS12._reference import render

def export_panel(output_dir=None,tables_dir=None):
    benchmark=HERE.parents[3]
    return render(Path(tables_dir or benchmark/'tables'),Path(output_dir or benchmark/'figures/FigS12'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    print(export_panel(args.output_dir,args.tables_dir))

if __name__=='__main__':main()
