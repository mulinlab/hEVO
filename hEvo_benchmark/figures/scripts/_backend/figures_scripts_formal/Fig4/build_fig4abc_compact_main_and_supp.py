#!/usr/bin/env python
"""Build compact Figure 4a-c and source-backed supplementary evidence.

The main A4 output intentionally occupies only the top band.  Panel c combines
the few-shot trajectory and the 60-shot endpoint comparison, replacing the
legacy separate panel d.  The QA metadata has a user-requested PASS label, but
the ancestry-classification method and result lineage still need review.
"""

from __future__ import annotations

# [Fig. 4 entry point] Use Matplotlib to draw the ancestry unsupervised, classification, and few-shot panels and supplementary views.
# [Input] S26/S27/S28; load/prepare functions filter and order models and calculate means, intervals, and family mappings.
# [Style] Use the vendored Matplotlib, black text, hEVO orange, and neutral gray; rcParams and color constants are defined here.
# [Output] main/build_* writes PDF/PNG/SVG, source data, captions/summaries, and QA; it does not use ReportLab A4 translated_group.

import json
import sys
from pathlib import Path

# The desktop runtime is isolated with a minimal scientific stack.  Keep the
# figure-only Matplotlib wheel inside the project so the renderer is portable.
VENDOR = Path(__file__).resolve().parents[6] / ".vendor" / "matplotlib"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from figures.scripts.common.hevo_model_style import (  # noqa: E402
    MODEL_FAMILY_STYLE,
    canonical_model_name,
    model_family,
    model_registry_record,
)


TABLES = ROOT / "tables"
OUT = ROOT / "figures" / "formal" / "Fig4" / "fig4abc_compact_v6"
SOURCE = OUT / "source_data"

S26 = TABLES / "S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv"
S27 = TABLES / "S34_supervised_classification_of_1000_genomes_ancestry_groups.csv"
S28 = TABLES / "S35_few_shot_regression_of_ancestry_proportions.csv"

HEVO_BASE = "hEvo-base"
HEVO_LARGE = "hEvo-large"
FOCAL = {HEVO_BASE, HEVO_LARGE}

BLACK = "#000000"
WHITE = "#FFFFFF"
HEVO = MODEL_FAMILY_STYLE["hEVO"]["color"]
HEVO_WASH = "#FBF2EA"
NEUTRAL = "#69727D"
NEUTRAL_LIGHT = "#D7DCE2"
NEUTRAL_PALE = "#EEF1F4"
FAMILY_ORDER = ["hEVO", "Evo 2", "NT", "GPN", "Other"]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "axes.labelsize": 6.2,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.2,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "text.color": BLACK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def display(raw: str) -> str:
    return canonical_model_name(raw)


def compact_display(raw: str) -> str:
    """Use the manuscript-wide canonical label in both main and supplement."""
    return display(raw)


def algorithm_identity(raw: str) -> str:
    """Group parameter scales while preserving biologically distinct variants."""
    fam = model_family(raw)
    if fam == "hEVO":
        return "hEVO"
    if fam == "GPN" or str(raw).startswith("Caduceus-"):
        return display(raw)
    record = model_registry_record(raw)
    return record["algorithm"] or display(raw)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unsup = pd.read_csv(S26)
    cls = pd.read_csv(S27)
    few = pd.read_csv(S28)
    for frame in (unsup, cls, few):
        frame["model_display"] = frame["model"].map(display)
        frame["family"] = frame["model"].map(model_family)
        frame["algorithm_identity"] = frame["model"].map(algorithm_identity)
    return unsup, cls, few


def representative_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Select one best parameterization per algorithm plus both focal hEVO sizes."""
    keep: list[int] = []
    for raw in (HEVO_LARGE, HEVO_BASE):
        match = frame.index[frame["model"] == raw]
        if len(match):
            keep.append(int(match[0]))
    other = frame.loc[frame["family"] != "hEVO"]
    for _, group in other.groupby("algorithm_identity", sort=False):
        keep.append(int(group[metric].idxmax()))
    result = frame.loc[sorted(set(keep))].copy()
    result = result.sort_values([metric, "model_display"], ascending=[False, True])
    result["plot_rank"] = np.arange(1, len(result) + 1)
    result["plot_label"] = result["model"].map(compact_display)
    return result.reset_index(drop=True)


def trim_spines(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.1, width=0.55, pad=1.6)
    ax.grid(False)


def add_panel_letter(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    letter: str,
    dx: float = -0.055,
    dy: float = 0.016,
) -> None:
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, letter, ha="left", va="bottom", fontsize=10.5, fontweight="bold")


def add_panel_header(fig: mpl.figure.Figure, axes: tuple[mpl.axes.Axes, ...], text: str) -> None:
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    top = max(ax.get_position().y1 for ax in axes)
    fig.text(
        (left + right) / 2,
        top + 0.008,
        text,
        ha="center",
        va="bottom",
        fontsize=5.7,
        fontweight="bold",
        color=BLACK,
    )


def shade_hevo_rows(ax: mpl.axes.Axes, rows: pd.DataFrame) -> None:
    for y, raw in enumerate(rows["model"]):
        if raw in FOCAL:
            ax.axhspan(y - 0.47, y + 0.47, color=HEVO_WASH, zorder=0, lw=0)


def y_model_axis(ax: mpl.axes.Axes, rows: pd.DataFrame, *, show: bool = True, font_size: float = 5.0) -> None:
    y = np.arange(len(rows))
    ax.set_yticks(y)
    if show:
        labels = rows["plot_label"] if "plot_label" in rows.columns else rows["model_display"]
        ax.set_yticklabels(labels, fontsize=font_size)
        for tick, raw in zip(ax.get_yticklabels(), rows["model"]):
            tick.set_color(BLACK)
            tick.set_fontweight("bold" if raw in FOCAL else "normal")
    else:
        # With a shared y axis, assigning empty tick labels here would also
        # erase the labels on the left metric column.
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_ylim(len(rows) - 0.5, -0.5)


def dot_style(raw: str, family_colours: bool = False) -> tuple[str, str, str, float, float]:
    if raw == HEVO_LARGE:
        return HEVO, HEVO, "o", 20.0, 0.9
    if raw == HEVO_BASE:
        return WHITE, HEVO, "o", 20.0, 0.9
    fam = model_family(raw)
    color = MODEL_FAMILY_STYLE[fam]["color"] if family_colours else NEUTRAL
    marker = MODEL_FAMILY_STYLE[fam]["marker"] if family_colours else "o"
    return WHITE, color, marker, 14.0, 0.70


def scatter_rows(ax: mpl.axes.Axes, rows: pd.DataFrame, metric: str, *, family_colours: bool = False) -> None:
    for y, row in rows.iterrows():
        fill, edge, marker, size, width = dot_style(row["model"], family_colours)
        ax.scatter(row[metric], y, s=size, marker=marker, facecolor=fill, edgecolor=edge, linewidth=width, zorder=4)


def draw_unsupervised(
    fig: mpl.figure.Figure,
    slot,
    rows: pd.DataFrame,
    *,
    letter: str | None,
    family_colours: bool = False,
    font_size: float = 5.0,
) -> tuple[mpl.axes.Axes, mpl.axes.Axes]:
    gs = slot.subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    y_model_axis(ax1, rows, font_size=font_size)
    y_model_axis(ax2, rows, show=False)
    shade_hevo_rows(ax1, rows)
    shade_hevo_rows(ax2, rows)

    for y, row in rows.iterrows():
        ax1.plot([0, row["tsne_silhouette"]], [y, y], color=HEVO if row["model"] in FOCAL else NEUTRAL_LIGHT, lw=0.55, zorder=1)
        ax2.plot([0, row["tsne_ch_index"]], [y, y], color=HEVO if row["model"] in FOCAL else NEUTRAL_LIGHT, lw=0.55, zorder=1)
    scatter_rows(ax1, rows, "tsne_silhouette", family_colours=family_colours)
    scatter_rows(ax2, rows, "tsne_ch_index", family_colours=family_colours)

    ax1.axvline(0, color=BLACK, lw=0.55, zorder=2)
    ax1.set_xlim(min(-0.045, float(rows["tsne_silhouette"].min()) - 0.015), 0.43)
    ax1.set_xticks([0.0, 0.2, 0.4])
    ax1.set_xlabel("t-SNE silhouette", labelpad=2.0)
    # Reserve a full marker radius beyond the leading CH value.  The legacy
    # 1,800 limit visibly clipped the hEvo-large circle at the right boundary.
    ax2.set_xlim(0, max(1900, float(rows["tsne_ch_index"].max()) * 1.09))
    ax2.set_xticks([0, 900, 1800])
    ax2.set_xlabel("CH index", labelpad=2.0)
    for ax in (ax1, ax2):
        trim_spines(ax)
    if letter:
        add_panel_letter(fig, ax1, letter, dx=-0.070)
    return ax1, ax2


def draw_mcc(
    fig: mpl.figure.Figure,
    slot,
    rows: pd.DataFrame,
    *,
    letter: str | None,
    family_colours: bool = False,
    font_size: float = 5.0,
) -> mpl.axes.Axes:
    ax = fig.add_subplot(slot)
    y_model_axis(ax, rows, font_size=font_size)
    shade_hevo_rows(ax, rows)
    scatter_rows(ax, rows, "classification_mcc", family_colours=family_colours)
    xmin = max(0.90, float(rows["classification_mcc"].min()) - 0.006)
    xmax = min(1.0, float(rows["classification_mcc"].max()) + 0.006)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("MCC", labelpad=2.0)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    trim_spines(ax)
    if letter:
        add_panel_letter(fig, ax, letter, dx=-0.060)
    return ax


def draw_fewshot_curve(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    selected_models: list[str],
    *,
    family_colours: bool = False,
) -> None:
    selected = data.loc[data["model"].isin(selected_models)].copy()
    for raw, group in selected.groupby("model", sort=False):
        group = group.sort_values("shot")
        is_focal = raw in FOCAL
        if is_focal:
            color = HEVO
            lw = 1.45 if raw == HEVO_LARGE else 1.15
            ls = "-" if raw == HEVO_LARGE else "--"
            marker = "o"
            z = 5
        else:
            fam = model_family(raw)
            color = MODEL_FAMILY_STYLE[fam]["color"] if family_colours else NEUTRAL_LIGHT
            lw, ls, marker, z = 0.60, "-", None, 1
        ax.plot(group["shot"], group["pearson_r_mean"], color=color, lw=lw, ls=ls, marker=marker, ms=2.0, zorder=z)
        if is_focal:
            low = group["pearson_r_mean"] - group["pearson_r_std"]
            high = group["pearson_r_mean"] + group["pearson_r_std"]
            ax.fill_between(group["shot"], low, high, color=HEVO, alpha=0.10, lw=0, zorder=2)
    ymin = max(0.58, float(selected["pearson_r_mean"].min()) - 0.018)
    ymax = min(0.97, float(selected["pearson_r_mean"].max()) + 0.012)
    ax.set_xlim(9, 61)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.set_xlabel("Training samples per population", labelpad=2.0)
    ax.set_ylabel("Mean Pearson r", labelpad=2.2)
    trim_spines(ax)


def draw_endpoint(
    ax: mpl.axes.Axes,
    endpoint: pd.DataFrame,
    *,
    family_colours: bool = False,
    font_size: float = 4.8,
) -> None:
    y_model_axis(ax, endpoint, font_size=font_size)
    shade_hevo_rows(ax, endpoint)
    scatter_rows(ax, endpoint, "pearson_r_mean", family_colours=family_colours)
    xmin = max(0.75, float(endpoint["pearson_r_mean"].min()) - 0.010)
    xmax = min(0.97, float(endpoint["pearson_r_mean"].max()) + 0.008)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Pearson r at 60/population", labelpad=2.0)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    trim_spines(ax)


def draw_combined_fewshot(
    fig: mpl.figure.Figure,
    slot,
    few: pd.DataFrame,
    endpoint: pd.DataFrame,
    *,
    letter: str | None,
    family_colours: bool = False,
    compact: bool = True,
) -> tuple[mpl.axes.Axes, mpl.axes.Axes]:
    # The compact gap is the minimum that accommodates the longest canonical
    # model label without allowing it to intrude into the learning curves.
    gs = slot.subgridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.68 if compact else 0.46)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    draw_fewshot_curve(ax_curve, few, endpoint["model"].tolist(), family_colours=family_colours)
    draw_endpoint(ax_rank, endpoint, family_colours=family_colours, font_size=4.25 if compact else 5.0)
    legend = [
        Line2D([0], [0], color=HEVO, lw=1.4, marker="o", ms=2.5, label=display("hEvo-large")),
        Line2D([0], [0], color=HEVO, lw=1.1, ls="--", marker="o", mfc=WHITE, ms=2.5, label=display("hEvo-base")),
    ]
    if not family_colours:
        legend.append(Line2D([0], [0], color=NEUTRAL_LIGHT, lw=0.7, label="other models"))
    ax_curve.legend(handles=legend, loc="lower right", frameon=False, fontsize=4.8, handlelength=1.7, borderaxespad=0.2, labelspacing=0.25)
    if letter:
        # A compound panel needs its letter anchored to its own left edge;
        # the generic negative offset visually attached c to panel b.
        add_panel_letter(
            fig,
            ax_curve,
            letter,
            dx=-0.018 if compact else -0.040,
            dy=0.016 if compact else -0.010,
        )
    return ax_curve, ax_rank


def save_bundle(fig: mpl.figure.Figure, stem: str, *, dpi: int = 600) -> None:
    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", facecolor=WHITE)
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, facecolor=WHITE)


def add_family_key(fig: mpl.figure.Figure, *, y: float, fontsize: float = 4.7) -> None:
    handles = []
    for family in FAMILY_ORDER:
        style = MODEL_FAMILY_STYLE[family]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color=style["color"],
                markeredgecolor=style["color"],
                markerfacecolor=WHITE,
                lw=0.7,
                ms=3.5,
                label=family,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=fontsize,
        handletextpad=0.30,
        columnspacing=0.85,
        bbox_to_anchor=(0.55, y),
    )


def build_main(unsup_main: pd.DataFrame, cls_main: pd.DataFrame, few: pd.DataFrame, end_main: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4), facecolor=WHITE)
    gs = fig.add_gridspec(
        1,
        3,
        left=0.105,
        right=0.965,
        bottom=0.800,
        top=0.960,
        width_ratios=[1.30, 0.72, 1.78],
        wspace=0.48,
    )
    a_axes = draw_unsupervised(fig, gs[0, 0], unsup_main, letter="a", family_colours=True, font_size=4.7)
    b_ax = draw_mcc(fig, gs[0, 1], cls_main, letter="b", family_colours=True, font_size=4.65)
    c_axes = draw_combined_fewshot(fig, gs[0, 2], few, end_main, letter="c", family_colours=True, compact=True)
    add_panel_header(fig, a_axes, "Unsupervised separation of five 1KGP super-populations")
    add_panel_header(fig, (b_ax,), "Five-way 1KGP super-population classification")
    add_panel_header(fig, c_axes, "Few-shot ADMIXTURE K=5 proportion regression")
    add_family_key(fig, y=0.752)
    save_bundle(fig, "Figure4abc_A4_toprow_compact_v6_METHOD_QC_DRAFT")
    plt.close(fig)


def build_standalones(unsup_main: pd.DataFrame, cls_main: pd.DataFrame, few: pd.DataFrame, end_main: pd.DataFrame) -> None:
    specs = [
        ("Fig4a_unsupervised_compact_v6_METHOD_QC_DRAFT", (88, 60), "a"),
        ("Fig4b_MCC_compact_v6_METHOD_QC_DRAFT", (62, 60), "b"),
        ("Fig4c_fewshot_endpoint_combined_v6_METHOD_QC_DRAFT", (118, 60), "c"),
    ]
    for stem, size_mm, which in specs:
        fig = plt.figure(figsize=(size_mm[0] / 25.4, size_mm[1] / 25.4), facecolor=WHITE)
        gs = fig.add_gridspec(1, 1, left=0.27 if which != "c" else 0.075, right=0.96, bottom=0.13, top=0.95)
        if which == "a":
            axes = draw_unsupervised(fig, gs[0, 0], unsup_main, letter="a", family_colours=True, font_size=4.8)
            add_panel_header(fig, axes, "Unsupervised separation of five 1KGP super-populations")
        elif which == "b":
            ax = draw_mcc(fig, gs[0, 0], cls_main, letter="b", family_colours=True, font_size=4.8)
            add_panel_header(fig, (ax,), "Five-way 1KGP super-population classification")
        else:
            axes = draw_combined_fewshot(fig, gs[0, 0], few, end_main, letter="c", family_colours=True, compact=False)
            add_panel_header(fig, axes, "Few-shot ADMIXTURE K=5 proportion regression")
        save_bundle(fig, stem)
        plt.close(fig)


def build_supplement_all_models(unsup: pd.DataFrame, cls: pd.DataFrame) -> None:
    unsup = unsup.sort_values("rank_score", ascending=False).reset_index(drop=True)
    cls = cls.sort_values("classification_mcc", ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4), facecolor=WHITE)
    gs = fig.add_gridspec(2, 1, left=0.19, right=0.95, bottom=0.075, top=0.97, hspace=0.20)
    a_axes = draw_unsupervised(fig, gs[0, 0], unsup, letter="a", family_colours=True, font_size=5.4)
    b_ax = draw_mcc(fig, gs[1, 0], cls, letter="b", family_colours=True, font_size=5.4)
    add_panel_header(fig, a_axes, "Unsupervised separation of five 1KGP super-populations")
    add_panel_header(fig, (b_ax,), "Five-way 1KGP super-population classification")
    handles = []
    for fam in FAMILY_ORDER:
        style = MODEL_FAMILY_STYLE[fam]
        handles.append(Line2D([0], [0], marker=style["marker"], color="none", markeredgecolor=style["color"], markerfacecolor=WHITE, ms=4, label=fam))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.0, bbox_to_anchor=(0.57, 0.025))
    save_bundle(fig, "FigureS4_ancestry_all_models_unsupervised_and_MCC_A4_v6_METHOD_QC_DRAFT")
    plt.close(fig)


def build_supplement_fewshot(few: pd.DataFrame) -> None:
    endpoint = few.loc[few["shot"] == 60].sort_values("pearson_r_mean", ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4), facecolor=WHITE)
    gs = fig.add_gridspec(2, 1, left=0.19, right=0.95, bottom=0.075, top=0.97, hspace=0.28, height_ratios=[0.82, 1.18])
    ax_curve = fig.add_subplot(gs[0, 0])
    draw_fewshot_curve(ax_curve, few, endpoint["model"].tolist(), family_colours=True)
    add_panel_letter(fig, ax_curve, "a", dx=-0.065)
    add_panel_header(fig, (ax_curve,), "Few-shot ADMIXTURE K=5 proportion regression")
    ax_rank = fig.add_subplot(gs[1, 0])
    draw_endpoint(ax_rank, endpoint, family_colours=True, font_size=5.4)
    add_panel_letter(fig, ax_rank, "b", dx=-0.065)
    add_panel_header(fig, (ax_rank,), "Performance at 60 training samples per population")
    handles = []
    for fam in FAMILY_ORDER:
        style = MODEL_FAMILY_STYLE[fam]
        handles.append(Line2D([0], [0], marker=style["marker"], color=style["color"], markerfacecolor=WHITE, ms=4, lw=0.8, label=fam))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.0, bbox_to_anchor=(0.57, 0.025))
    save_bundle(fig, "FigureS5_ancestry_fewshot_all_models_A4_v6_METHOD_QC_DRAFT")
    plt.close(fig)


def write_source_and_contract(
    unsup: pd.DataFrame,
    cls: pd.DataFrame,
    few: pd.DataFrame,
    unsup_main: pd.DataFrame,
    cls_main: pd.DataFrame,
    end_main: pd.DataFrame,
) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    unsup.to_csv(SOURCE / "Figure4a_all_models_source.csv", index=False)
    cls.to_csv(SOURCE / "Figure4b_all_models_source.csv", index=False)
    few.to_csv(SOURCE / "Figure4c_all_models_fewshot_source.csv", index=False)
    unsup_main.to_csv(SOURCE / "Figure4a_main_representatives.csv", index=False)
    cls_main.to_csv(SOURCE / "Figure4b_main_representatives.csv", index=False)
    end_main.to_csv(SOURCE / "Figure4c_main_representatives_at_60.csv", index=False)

    contract = """# Figure 4a-c chart and evidence contract

## Main-panel logic

- **a** reports t-SNE silhouette and Calinski-Harabasz (CH) index on their native scales. The legacy visually normalized CH bar is removed. Rows contain the best parameterization per algorithm, plus both hEvo-base and hEvo-large; all 33 checkpoints are in Figure S4a.
- **b** reports multiclass Matthews correlation coefficient (MCC) only. Accuracy is omitted because it is numerically redundant with MCC in this balanced five-class task; the all-checkpoint MCC view is Figure S4b.
- **c** combines the few-shot learning trajectory with the 60-sample endpoint ranking. This replaces the legacy separate panel d. All checkpoints and all endpoint values are in Figure S5.

## Supported claims

- hEvo-large is first in the source tables for the Figure 4a composite ordering, Figure 4b MCC, and the 60-sample Figure 4c endpoint.
- The two native Figure 4a metrics must be interpreted separately; no universal weighting between silhouette and CH is claimed.
- hEvo-base and hEvo-large are both retained, so scale-dependent performance is visible rather than hidden.

## Naming convention

- Population resource labels use **1000 Genomes Project (1KGP)** on first mention and **1KGP** thereafter.
- Official Nucleotide Transformer checkpoints preserve the paper/model-card tag **1000G**, for example **NTv1-1000G-500M**.
- Project-defined hEVO corpus ablations use **1KGP**, for example **hEVO-1KGP-86M**.
- The ambiguous bare label **1KG** is not used in display text.
"""
    (OUT / "FIGURE4_CHART_CONTRACT.md").write_text(contract, encoding="utf-8")


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    unsup, cls, few = load_data()
    unsup_main = representative_rows(unsup, "rank_score")
    cls_main = representative_rows(cls, "classification_mcc")
    endpoint = few.loc[few["shot"] == 60].copy()
    end_main = representative_rows(endpoint, "pearson_r_mean")
    build_main(unsup_main, cls_main, few, end_main)
    build_standalones(unsup_main, cls_main, few, end_main)
    build_supplement_all_models(unsup, cls)
    build_supplement_fewshot(few)
    write_source_and_contract(unsup, cls, few, unsup_main, cls_main, end_main)


if __name__ == "__main__":
    main()
