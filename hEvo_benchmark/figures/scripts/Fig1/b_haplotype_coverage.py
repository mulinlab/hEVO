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
from common.reportlab_common import export_drawing, mm_drawing, register_arial_aliases
BACKEND=load_module("Fig1/plot_figure1_a4_integrated.py")

WIDTH_MM, HEIGHT_MM = 90.0, 42.0
SOURCE_NAME = "S5_coverage_of_uk_biobank_reference_haplotypes_by_the_selected_subset.csv"
COHORT_SOURCE_NAME = "S2_principal_component_coordinates_of_the_final_63_809_participant_cohort.csv"

def _load(tables_dir=None):
    root=panel_tables_dir("Fig1", tables_dir)
    source=root/SOURCE_NAME
    raw=pd.read_csv(source)
    require_columns(raw, ["target","reference_haplotype_frequency_threshold","threshold_weighted_coverage"], "Fig1b")
    raw=raw.loc[raw["target"].eq("Selected UKB subset")].copy()
    if raw.empty: raise ValueError("Fig1b: Selected UKB subset is missing from S3")
    raw["minimum_haplotype_frequency_pct"]=pd.to_numeric(raw["reference_haplotype_frequency_threshold"], errors="raise")*100.0
    # Keep source coverage as a fraction; the backend converts it to percent.
    raw["weighted_coverage"]=pd.to_numeric(raw["threshold_weighted_coverage"], errors="raise")
    raw=raw.sort_values("minimum_haplotype_frequency_pct").reset_index(drop=True)
    if not raw["minimum_haplotype_frequency_pct"].is_monotonic_increasing: raise ValueError("Fig1b: threshold is not sorted")
    cohort_source=root/COHORT_SOURCE_NAME
    cohort=pd.read_csv(cohort_source)
    require_columns(cohort, ["IID", "Group"], "Fig1b final cohort")
    cohort=cohort.loc[cohort["Group"].eq("UKB")]
    if cohort.empty or cohort["IID"].isna().any() or cohort["IID"].duplicated().any():
        raise ValueError("Fig1b: final UKB sample IDs must be nonempty and unique")
    selected_ukb_n=len(cohort)
    return raw, source, cohort_source, selected_ukb_n

def build_panel(tables_dir=None):
    table,_,_,selected_ukb_n=_load(tables_dir); register_arial_aliases()
    drawing=mm_drawing(WIDTH_MM,HEIGHT_MM)
    BACKEND.draw_panel_b(drawing, table, 0.0,0.0,WIDTH_MM,HEIGHT_MM,selected_ukb_n)
    return drawing

def export_panel(output_dir=None, tables_dir=None):
    _,source,cohort_source,_=_load(tables_dir)
    return export_drawing(build_panel(tables_dir), panel_output_dir("Fig1",output_dir), "b_haplotype_coverage", input_files(source,cohort_source), width_mm=WIDTH_MM,height_mm=HEIGHT_MM)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--tables-dir","--data-dir",dest="tables_dir",type=Path,default=None); p.add_argument("--output-dir",type=Path,default=figure_dir("Fig1")); a=p.parse_args()
    export_panel(a.output_dir,a.tables_dir)
if __name__=="__main__": main()
