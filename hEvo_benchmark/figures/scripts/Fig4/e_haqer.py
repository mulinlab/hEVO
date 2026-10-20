"""Fig.4e: HAQER robustness across five negative-set definitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.matplotlib_common import export_figure  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from Fig4._har_haqer import DATA_FILES, HEIGHT_MM, WIDTH_MM, build_panel as _build_panel  # noqa: E402


def build_panel(tables_dir: Path | None = None, *, figure=None, slot=None, integrated: bool = False):
    return _build_panel(tables_dir, figure=figure, slot=slot, integrated=integrated, dataset="HAQER", letter="e")


def export_panel(output_dir: Path | None = None, tables_dir_override: Path | None = None):
    root = panel_tables_dir("Fig4", tables_dir_override)
    fig = build_panel(root)
    result = export_figure(
        fig,
        panel_output_dir("Fig4", output_dir),
        "e_haqer",
        input_files(root / DATA_FILES["HAQER"]),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
    )
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig4"))
    args = parser.parse_args()
    export_panel(args.output_dir, args.tables_dir)


if __name__ == "__main__":
    main()
