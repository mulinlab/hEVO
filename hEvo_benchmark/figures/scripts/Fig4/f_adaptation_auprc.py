"""Supplementary directional preference panel for selected alleles across loci.

The historical filename is retained for the standalone supplemental output.
The active A4 Fig.4h renderer is ``h_adaptive_allele_preference.py``, which
combines this directional view with a raw-score Calinski-Harabasz view.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.matplotlib_common import configure_matplotlib, export_figure
from common.paths import figure_dir, tables_dir as default_tables_dir
from common.model_names import canonical_model_name


HEVO_COLORS = {"hEvo-base": "#1F77B4", "hEvo-large": "#D62728"}
COMPETITOR_COLOR = "#B7BEC8"
BLOCKED_COLOR = "#C9CED6"


def _short_label(row: pd.Series) -> str:
    return canonical_model_name(str(row["model_id"]))


def _sort_models(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.copy()
    ordered["_blocked_last"] = ordered["status"].eq("BLOCKED")
    return ordered.sort_values(
        ["_blocked_last", "accuracy", "model_id"],
        ascending=[True, False, True],
        na_position="last",
    ).drop(columns="_blocked_last").reset_index(drop=True)


def _style_axis(ax):
    ax.grid(axis="x", color="#D9DEE7", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.tick_params(axis="x", length=2, width=0.45, pad=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.45)
    ax.spines["bottom"].set_linewidth(0.45)


def build_panel(
    tables_dir: Path | None = None,
    *,
    figure=None,
    slot=None,
    integrated: bool = False,
    letter: str | None = None,
):
    configure_matplotlib()
    root = Path(tables_dir) if tables_dir is not None else default_tables_dir()
    data = pd.read_csv(
        root / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv"
    )
    data = _sort_models(data)
    data["label"] = data.apply(_short_label, axis=1)
    data["y"] = range(len(data))

    if figure is None:
        figure = plt.figure(figsize=(95 / 25.4, 64 / 25.4), facecolor="white")
    ax = figure.add_subplot(slot) if slot is not None else figure.add_subplot(111)

    valid = data["status"].eq("PASS")
    valid_data = data.loc[valid]
    colors = [HEVO_COLORS.get(model, COMPETITOR_COLOR) for model in valid_data["model_id"]]
    ax.errorbar(
        valid_data["accuracy"],
        valid_data["y"],
        xerr=[
            valid_data["accuracy"] - valid_data["accuracy_ci_low"],
            valid_data["accuracy_ci_high"] - valid_data["accuracy"],
        ],
        fmt="none",
        ecolor="#263238",
        elinewidth=0.45,
        capsize=1.2,
        zorder=3,
    )
    ax.scatter(
        valid_data["accuracy"],
        valid_data["y"],
        s=10,
        c=colors,
        edgecolors="#263238",
        linewidths=0.25,
        zorder=4,
    )

    blocked = data["status"].eq("BLOCKED")
    if blocked.any():
        ax.scatter(
            [0.03] * int(blocked.sum()),
            data.loc[blocked, "y"],
            marker="x",
            s=15,
            c=BLOCKED_COLOR,
            linewidths=0.8,
            zorder=4,
        )

    ax.axvline(0.5, color="#9AA1AA", linestyle="--", linewidth=0.5, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1, len(data))
    ax.set_yticks(data["y"])
    ax.set_yticklabels(data["label"], fontsize=4.0 if integrated else 4.2)
    ax.invert_yaxis()
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=5)
    ax.set_xlabel("Fraction of loci favoring selected allele", fontsize=5.2, labelpad=2)
    ax.set_title(
        "Directional preference",
        loc="left",
        fontsize=6.1,
        pad=3,
        fontweight="bold",
    )
    _style_axis(ax)

    ax.text(
        -0.18 if integrated else -0.10,
        1.05,
        letter or "h",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        1.0,
        -0.105,
        f"× no valid directional score; n=30 loci; {int(valid.sum())} scored, {int(blocked.sum())} blocked",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=3.8,
        color="#59636E",
    )
    return ax


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    root = Path(tables_dir) if tables_dir is not None else default_tables_dir()
    fig = plt.figure(figsize=(95 / 25.4, 64 / 25.4), facecolor="white")
    build_panel(root, figure=fig, integrated=False, letter="h")
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.18, top=0.90)
    result = export_figure(
        fig,
        output_dir or figure_dir("Fig4"),
        "h_adaptive_allele_preference_accuracy",
        [root / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv"],
        width_mm=95,
        height_mm=64,
    )
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    export_panel(args.output_dir, args.tables_dir)


if __name__ == "__main__":
    main()
