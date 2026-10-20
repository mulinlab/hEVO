"""Fig.4g+h: independent 180 x 145.5 mm formal HAR/HAQER preview."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.paths import figure_dir
from Fig4._har_haqer_genome_browser import DATA_ROOT, build_combined


def build_panel(tables_dir: Path | None = None):
    return build_combined(tables_dir)


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    from common.matplotlib_common import export_figure
    from common.panel_runtime import input_files

    root = tables_dir or DATA_ROOT
    fig = build_combined(root)
    files = [root / "S40_genomic_context_tracks_for_human_accelerated_and_quickly_evolved_regions.csv", root / "har_haqer_track_source_manifest.csv"]
    result = export_figure(fig, output_dir or figure_dir("Fig4"), "g_h_integrated", input_files(*files), 180.0, 145.5)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig4"))
    args = parser.parse_args()
    from common.matplotlib_common import export_figure
    from common.panel_runtime import input_files
    import matplotlib.pyplot as plt

    fig = build_combined(args.tables_dir)
    files = [args.tables_dir / "S40_genomic_context_tracks_for_human_accelerated_and_quickly_evolved_regions.csv", args.tables_dir / "har_haqer_track_source_manifest.csv"]
    export_figure(fig, args.output_dir, "g_h_integrated", input_files(*files), 180.0, 145.5)
    plt.close(fig)


if __name__ == "__main__":
    main()
