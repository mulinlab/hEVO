"""Fig.4g focus view for the same summit-based example region as Fig.4f."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.paths import figure_dir
from Fig4._har_haqer_genome_browser import DATA_ROOT


def build_panel(tables_dir: Path | None = None, *, figure=None, slot=None, integrated: bool = False, letter: str | None = None):
    from Fig4._har_haqer_genome_browser import build_panel as _build

    return _build(tables_dir, panel_id="h", figure=figure, slot=slot, mode="comparison", integrated=integrated, letter=letter, view_span="focus")


def export_panel(output_dir: Path | None = None):
    from Fig4._har_haqer_genome_browser import export_panel as _export

    return _export(DATA_ROOT, output_dir or figure_dir("Fig4"), "h", "Fig4g_example_focus", mode="comparison", view_span="focus", letter="g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig4"))
    args = parser.parse_args()
    from Fig4._har_haqer_genome_browser import export_panel as _export

    _export(args.tables_dir, args.output_dir, "h", "Fig4g_example_focus", mode="comparison", view_span="focus", letter="g")


if __name__ == "__main__":
    main()
