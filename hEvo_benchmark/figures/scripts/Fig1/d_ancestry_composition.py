"""Fig.1d：S2 队列祖源组成堆叠图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.data_io import require_columns  # noqa: E402
from common.legacy_loader import load_module  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, mm_drawing, pad_drawing, register_arial_aliases  # noqa: E402

BACKEND = load_module("Fig1/plot_figure1_a4_integrated.py")
WIDTH_MM, HEIGHT_MM = 78.0, 52.0
STANDALONE_BOTTOM_PAD_MM = 8.0


def build_panel(tables_dir: Path | None = None):
    """读取并验证 S2 计数和，随后绘制 panel d。"""

    source = panel_tables_dir("Fig1", tables_dir) / "S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv"
    table = pd.read_csv(source)
    columns = ["Cohort", "Total_samples", "EUR_count", "AFR_count", "EAS_count", "SAS_count", "Others_count"]
    require_columns(table, columns, "Fig1d")
    counts = ["EUR_count", "AFR_count", "EAS_count", "SAS_count", "Others_count"]
    for ancestry in ("EUR", "AFR", "EAS", "SAS", "Others"):
        table[f"{ancestry}_percent"] = 100.0 * table[f"{ancestry}_count"] / table["Total_samples"]
    if not (table[counts].sum(axis=1) == table["Total_samples"]).all():
        raise ValueError("Fig1d: ancestry counts do not sum to cohort totals")
    register_arial_aliases()
    drawing = mm_drawing(WIDTH_MM, HEIGHT_MM)
    BACKEND.draw_panel_d(drawing, table, 0.0, 0.0, WIDTH_MM, HEIGHT_MM)
    return drawing


def export_panel(output_dir: Path | None = None):
    source = panel_tables_dir("Fig1") / "S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv"
    drawing = pad_drawing(build_panel(), bottom_mm=STANDALONE_BOTTOM_PAD_MM)
    return export_drawing(drawing, panel_output_dir("Fig1", output_dir), "d_ancestry_composition", input_files(source), width_mm=WIDTH_MM, height_mm=HEIGHT_MM + STANDALONE_BOTTOM_PAD_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig1"))
    args = parser.parse_args()
    drawing = pad_drawing(build_panel(args.tables_dir), bottom_mm=STANDALONE_BOTTOM_PAD_MM)
    export_drawing(drawing, args.output_dir, "d_ancestry_composition", input_files(args.tables_dir / "S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv"), width_mm=WIDTH_MM, height_mm=HEIGHT_MM + STANDALONE_BOTTOM_PAD_MM)


if __name__ == "__main__":
    main()
