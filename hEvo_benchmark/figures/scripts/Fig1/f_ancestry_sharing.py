"""Fig.1f：31 个精确祖源交集的 sharing/UpSet 式 panel。"""

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
from common.reportlab_common import export_drawing, mm_drawing, register_arial_aliases  # noqa: E402

BACKEND = load_module("Fig1/plot_figure1_a4_integrated.py")
WIDTH_MM, HEIGHT_MM = 46.0, 52.0


def build_panel(tables_dir: Path | None = None):
    """验证交集 rank 为 1..31，再绘制祖源 sharing matrix。"""

    source = panel_tables_dir("Fig1", tables_dir) / "S9_haplotype_sequence_sharing_across_five_ancestry_groups.csv"
    table = pd.read_csv(source)
    # In the current S8 table, row order represents the intersection rank; the field name is no longer the legacy name.
    # `intersection_rank`; the plotting function reads count and the five ancestry indicator columns.
    require_columns(table, ["members", "count", "AFR", "EUR", "EAS", "SAS", "Others"], "Fig1f")
    if len(table) != 31:
        raise ValueError("Fig1f: expected 31 ordered intersections")
    register_arial_aliases()
    drawing = mm_drawing(WIDTH_MM, HEIGHT_MM)
    BACKEND.draw_panel_f(drawing, table, 0.0, 0.0, WIDTH_MM, HEIGHT_MM)
    return drawing


def export_panel(output_dir: Path | None = None):
    source = panel_tables_dir("Fig1") / "S9_haplotype_sequence_sharing_across_five_ancestry_groups.csv"
    return export_drawing(build_panel(), panel_output_dir("Fig1", output_dir), "f_ancestry_sharing", input_files(source), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig1"))
    args = parser.parse_args()
    drawing = build_panel(args.tables_dir)
    export_drawing(drawing, args.output_dir, "f_ancestry_sharing", input_files(args.tables_dir / "S9_haplotype_sequence_sharing_across_five_ancestry_groups.csv"), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


if __name__ == "__main__":
    main()
