"""Fig.3g：DDD de novo LoF/splice 与 ClinVar benign 对照。"""

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


def build_panel(tables_dir: Path | None = None, *, panel_letter: str = "g"):
    """用 S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv 的 ddd_clinvar task 绘制 g。"""
    return build_static(tables_dir, "S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv", "ddd_clinvar", panel_letter)


def export_panel(output_dir: Path | None = None):
    source = panel_tables_dir("Fig3") / "S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv"
    return export_drawing(build_panel(), panel_output_dir("Fig3", output_dir), "g_ddd", input_files(source), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig3"))
    args = parser.parse_args()
    export_drawing(build_panel(args.tables_dir), args.output_dir, "g_ddd", input_files(args.tables_dir / "S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv"), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


if __name__ == "__main__":
    main()
