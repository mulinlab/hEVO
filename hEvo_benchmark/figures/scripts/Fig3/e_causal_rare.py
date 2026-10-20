"""Fig.3e：稀有 causal variants benchmark。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing  # noqa: E402
from Fig3.static_helpers import build_static  # noqa: E402

WIDTH_MM, HEIGHT_MM = 64.6666666667, 55.5


def build_panel(tables_dir: Path | None = None, *, panel_letter: str = "e"):
    """用 S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv 的 rare_causal task 绘制 e。"""
    return build_static(tables_dir, "S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv", "rare_causal", panel_letter)


def export_panel(output_dir: Path | None = None):
    source = panel_tables_dir("Fig3") / "S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv"
    return export_drawing(build_panel(), panel_output_dir("Fig3", output_dir), "e_causal_rare", input_files(source), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig3"))
    args = parser.parse_args()
    export_drawing(build_panel(args.tables_dir), args.output_dir, "e_causal_rare", input_files(args.tables_dir / "S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv"), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


if __name__ == "__main__":
    main()
