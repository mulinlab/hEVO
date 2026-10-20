from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path: sys.path.insert(0,str(HERE.parents[1]))
from common.data_io import require_columns
from common.legacy_loader import load_module
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir
from common.paths import tables_dir, figure_dir
from common.reportlab_common import export_drawing, mm_drawing, pad_drawing, register_arial_aliases
BACKEND=load_module("Fig1/plot_figure1_a4_integrated.py")

WIDTH_MM, HEIGHT_MM = 90.0, 51.0
STANDALONE_RIGHT_PAD_MM = 4.0
SOURCE_NAME = "S6_local_haplotype_diversity_after_sequential_cohort_integration.csv"
SCENARIO_MAP = {"S1_UKB":"UKB subset","S2_UK10K":"+ UK10K","S3_EAS2K":"+ EAS2K","S4_SG10K":"+ SG10K","S5_AFR1K":"+ AFR1K","S6_TOPMed":"+ TOPMed","S7_HGDP":"+ HGDP","S8_1KGP":"+ 1KGP"}

def _load(tables_dir=None):
    source=panel_tables_dir("Fig1",tables_dir)/SOURCE_NAME
    raw=pd.read_csv(source)
    require_columns(raw, ["scenario","mean_cohort_balanced_unique_haplotypes_per_window","relative_to_ukb_reference"], "Fig1c")
    table=raw.loc[raw["scenario"].isin(SCENARIO_MAP)].copy()
    if len(table)!=8: raise ValueError(f"Fig1c: expected 8 cumulative rows, found {len(table)}")
    table["scenario"]=table["scenario"].map(SCENARIO_MAP)
    table["recovered_absolute_per_100kb"]=pd.to_numeric(table["mean_cohort_balanced_unique_haplotypes_per_window"],errors="raise")
    table=table.reset_index(drop=True)
    return table,source

def build_panel(tables_dir=None):
    table,_=_load(tables_dir); register_arial_aliases()
    drawing=mm_drawing(WIDTH_MM,HEIGHT_MM)
    BACKEND.draw_panel_c(drawing,table,0.0,0.0,WIDTH_MM,HEIGHT_MM)
    return drawing

def export_panel(output_dir=None,tables_dir=None):
    table,source=_load(tables_dir)
    drawing=pad_drawing(build_panel(tables_dir),right_mm=STANDALONE_RIGHT_PAD_MM)
    return export_drawing(drawing,panel_output_dir("Fig1",output_dir),"c_haplotype_diversity",input_files(source),width_mm=WIDTH_MM+STANDALONE_RIGHT_PAD_MM,height_mm=HEIGHT_MM)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--tables-dir","--data-dir",dest="tables_dir",type=Path,default=None); p.add_argument("--output-dir",type=Path,default=figure_dir("Fig1")); a=p.parse_args()
    export_panel(a.output_dir,a.tables_dir)
if __name__=="__main__": main()
