"""Fig.3c：ClinVar all 18 个模型代表的 AUPRC/95% CI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reportlab.lib.units import mm

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, mm_drawing, register_arial_aliases  # noqa: E402
from Fig3._backend import BACKEND, load_sources  # noqa: E402

WIDTH_MM, HEIGHT_MM = 64.6666666667, 60.0


def build_panel(tables_dir: Path | None = None):
    clinvar, _, _ = load_sources(tables_dir)
    rows = clinvar[clinvar["task"].eq("all")]
    meta = BACKEND.core.CLINVAR_META["all"]
    d = mm_drawing(210.0, 297.0)
    BACKEND.draw_numbered_bar_panel(d, rows, x=5.0, bottom=130.0, width=WIDTH_MM, top=185.5, letter="c", n_samples=int(rows.iloc[0]["num_samples"]), baseline=float(rows.iloc[0]["baseline"]), limits=meta["scale"], ticks=meta["ticks"], score_col="auprc_mean", low_col="auprc_ci_lower", high_col="auprc_ci_upper")
    from common.reportlab_common import crop_drawing
    return crop_drawing(d, 5.0, 130.0, WIDTH_MM, HEIGHT_MM)


def export_panel(output_dir: Path | None = None):
    root = panel_tables_dir("Fig3")
    sources = [root / "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv"]
    return export_drawing(build_panel(), panel_output_dir("Fig3", output_dir), "c_clinvar", input_files(*sources), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig3"))
    args = parser.parse_args()
    sources = [args.tables_dir / "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv"]
    export_drawing(build_panel(args.tables_dir), args.output_dir, "c_clinvar", input_files(*sources), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


if __name__ == "__main__":
    main()
