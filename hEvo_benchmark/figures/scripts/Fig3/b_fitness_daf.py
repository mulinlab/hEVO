"""Fig.3b：fitness effect spectrum 与 derived allele frequency placement map。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import crop_drawing, export_drawing, mm_drawing, pad_drawing, register_arial_aliases  # noqa: E402
from Fig3._backend import BACKEND, placement_table  # noqa: E402

WIDTH_MM, HEIGHT_MM = 98.0, 72.0
STANDALONE_BOTTOM_PAD_MM = 9.0


def build_panel(tables_dir: Path | None = None):
    """绘制 v2 panel a 内容并裁成独立的 Fig.3b。"""

    root = panel_tables_dir("Fig3", tables_dir)
    register_arial_aliases()
    full = mm_drawing(210.0, 297.0)
    BACKEND.draw_architecture_panel(full, placement_table(root, name="S22_schematic_coordinates_for_variant_effect_and_allele_frequency_benchmarks.csv"), panel="a", letter="b", title="Fitness-effect landscape (schematic)", x=107.0, width=98.0)
    return crop_drawing(full, 107.0, 225.5, WIDTH_MM, HEIGHT_MM)


def export_panel(output_dir: Path | None = None):
    source = panel_tables_dir("Fig3") / "S22_schematic_coordinates_for_variant_effect_and_allele_frequency_benchmarks.csv"
    drawing = pad_drawing(build_panel(), bottom_mm=STANDALONE_BOTTOM_PAD_MM)
    return export_drawing(drawing, panel_output_dir("Fig3", output_dir), "b_fitness_daf", input_files(source), width_mm=WIDTH_MM, height_mm=HEIGHT_MM + STANDALONE_BOTTOM_PAD_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig3"))
    args = parser.parse_args()
    drawing = pad_drawing(build_panel(args.tables_dir), bottom_mm=STANDALONE_BOTTOM_PAD_MM)
    export_drawing(drawing, args.output_dir, "b_fitness_daf", input_files(args.tables_dir / "S22_schematic_coordinates_for_variant_effect_and_allele_frequency_benchmarks.csv"), width_mm=WIDTH_MM, height_mm=HEIGHT_MM + STANDALONE_BOTTOM_PAD_MM)


if __name__ == "__main__":
    main()
