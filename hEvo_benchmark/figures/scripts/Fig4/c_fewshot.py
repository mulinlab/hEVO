"""Fig.4c：few-shot ADMIXTURE K=5 轨迹和 60-shot endpoint。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.matplotlib_common import configure_matplotlib, export_figure  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import figure_dir  # noqa: E402
from Fig4._backend import BACKEND, load_data  # noqa: E402

WIDTH_MM, HEIGHT_MM = 118.0, 60.0


def build_panel(tables_dir: Path | None = None, *, figure=None, slot=None, integrated: bool = False):
    """构建 c 的学习曲线+终点排名 compound panel。"""

    configure_matplotlib()
    _, _, few = load_data(tables_dir)
    endpoint = few.loc[few["shot"].eq(60)].sort_values("pearson_r_mean", ascending=False).reset_index(drop=True)
    end_main = BACKEND.representative_rows(endpoint, "pearson_r_mean")
    if figure is not None and slot is not None:
        return BACKEND.draw_combined_fewshot(figure, slot, few, end_main, letter="c", family_colours=True, compact=integrated)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    gs = fig.add_gridspec(1, 1, left=0.075, right=0.96, bottom=0.13, top=0.95)
    axes = BACKEND.draw_combined_fewshot(fig, gs[0, 0], few, end_main, letter="c", family_colours=True, compact=False)
    BACKEND.add_panel_header(fig, axes, "Few-shot ADMIXTURE K=5 proportion regression")
    return fig


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    root = panel_tables_dir("Fig4", tables_dir)
    fig = build_panel(root)
    result = export_figure(fig, panel_output_dir("Fig4", output_dir), "c_fewshot", input_files(root / "S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv", root / "S34_supervised_classification_of_1000_genomes_ancestry_groups.csv", root / "S35_few_shot_regression_of_ancestry_proportions.csv"), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig4"))
    args = parser.parse_args()
    export_panel(args.output_dir, args.tables_dir)


if __name__ == "__main__":
    main()
