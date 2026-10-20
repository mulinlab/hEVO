"""Fig.3j：LB/B → later upgrade 的 temporal ClinVar enrichment。"""

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
from Fig3.temporal_helpers import build_temporal  # noqa: E402

WIDTH_MM, HEIGHT_MM = 64.6666666667, 55.0


def build_panel(tables_dir: Path | None = None):
    return build_temporal(tables_dir, "S32_retrospective_enrichment_of_clinvar_reclassification_events_by_cutoff_year.csv", "i", panel_letter="j")


def export_panel(output_dir: Path | None = None):
    source = panel_tables_dir("Fig3") / "S32_retrospective_enrichment_of_clinvar_reclassification_events_by_cutoff_year.csv"
    return export_drawing(build_panel(), panel_output_dir("Fig3", output_dir), "i_clinvar_upgrade", input_files(source), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig3"))
    args = parser.parse_args()
    export_drawing(build_panel(args.tables_dir), args.output_dir, "i_clinvar_upgrade", input_files(args.tables_dir / "S32_retrospective_enrichment_of_clinvar_reclassification_events_by_cutoff_year.csv"), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


if __name__ == "__main__":
    main()
