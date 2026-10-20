"""Export the ancestry composition as Fig.1c, matching the A4 panel labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from Fig1 import d_ancestry_composition as source_panel
from Fig1.A4_integrated import relabel_panel, revise_panel_text
from common.panel_runtime import input_files
from common.paths import figure_dir, tables_dir
from common.reportlab_common import export_drawing, pad_drawing


def build_panel(root=None):
    panel = revise_panel_text(
        source_panel.build_panel(root),
        {"EAS2K": "Internal Chinese", "AFR1K": "CAAPA"},
    )
    return relabel_panel(panel, "d", "c")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig1"))
    args = parser.parse_args()
    panel = pad_drawing(build_panel(args.tables_dir), bottom_mm=source_panel.STANDALONE_BOTTOM_PAD_MM)
    export_drawing(
        panel, args.output_dir, "c_ancestry_composition",
        input_files(args.tables_dir / "S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv"),
        width_mm=source_panel.WIDTH_MM,
        height_mm=source_panel.HEIGHT_MM + source_panel.STANDALONE_BOTTOM_PAD_MM,
    )


if __name__ == "__main__":
    main()
