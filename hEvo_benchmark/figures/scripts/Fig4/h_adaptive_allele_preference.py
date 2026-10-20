"""Fig.4h selected-allele preference and raw-score CH index.

The left panel reports the directional accuracy and exact binomial confidence
interval already summarized in S38.  The right panel computes a model-native
Calinski-Harabasz index from the two raw likelihood vectors in S39: the 30
selected-allele scores and the 30 other-allele scores are treated as two
one-dimensional groups.  Models without two finite raw score vectors remain
explicitly NA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from sklearn.metrics import calinski_harabasz_score

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.matplotlib_common import configure_matplotlib, export_figure
from common.public_paths import publicize_paths
from common.paths import figure_dir, tables_dir as default_tables_dir
from Fig4._backend import BACKEND
from Fig4.f_adaptation_auprc import _sort_models


WIDTH_MM, HEIGHT_MM = 190.0, 70.0
EXPECTED_MODELS = 34
EXPECTED_LOCI = 30
EXPECTED_PASS_MODELS = 33
EXPECTED_BLOCKED_MODELS = 1
BLACK = "#000000"
WHITE = "#FFFFFF"
NEUTRAL = "#69727D"
NEUTRAL_LIGHT = "#C7CED6"
HEVO_WASH = "#FBF2EA"


def _family_style(model_id: str) -> dict[str, str]:
    family = BACKEND.model_family(model_id)
    return BACKEND.MODEL_FAMILY_STYLE.get(
        family,
        {"color": NEUTRAL, "marker": "o"},
    )


def prepare_data(tables_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(tables_dir) if tables_dir is not None else default_tables_dir()
    summary = pd.read_csv(
        root / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv"
    )
    scores = pd.read_csv(
        root / "S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv"
    )
    required_summary = {
        "model_id",
        "display_model",
        "model_type",
        "status",
        "n_loci",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
    }
    required_scores = {
        "model_id",
        "benchmark_status",
        "selected_logprob",
        "other_logprob",
    }
    if missing := required_summary.difference(summary.columns):
        raise ValueError(f"S38 is missing columns: {sorted(missing)}")
    if missing := required_scores.difference(scores.columns):
        raise ValueError(f"S39 is missing columns: {sorted(missing)}")
    if len(summary) != EXPECTED_MODELS:
        raise ValueError(f"Expected {EXPECTED_MODELS} S38 models, got {len(summary)}")
    if int(summary["status"].eq("PASS").sum()) != EXPECTED_PASS_MODELS:
        raise ValueError(f"S38 PASS model count is not {EXPECTED_PASS_MODELS}")
    if int(summary["status"].eq("BLOCKED").sum()) != EXPECTED_BLOCKED_MODELS:
        raise ValueError(f"S38 BLOCKED model count is not {EXPECTED_BLOCKED_MODELS}")

    ordered = _sort_models(summary).copy()
    # The source table keeps its original display metadata; figure outputs
    # carry the canonical manuscript spelling in their display-only fields.
    ordered["display_model"] = ordered["model_id"].map(BACKEND.canonical_model_name)
    ordered["plot_label"] = ordered["model_id"].map(BACKEND.canonical_model_name)
    ordered["y"] = np.arange(len(ordered), dtype=float)

    scores = scores.copy()
    scores["selected_logprob"] = pd.to_numeric(scores["selected_logprob"], errors="coerce")
    scores["other_logprob"] = pd.to_numeric(scores["other_logprob"], errors="coerce")
    rows: list[dict[str, object]] = []
    for row in ordered.itertuples(index=False):
        subset = scores.loc[scores["model_id"].eq(row.model_id)]
        pass_subset = subset.loc[subset["benchmark_status"].eq("PASS")]
        finite = pass_subset[["selected_logprob", "other_logprob"]].notna().all(axis=1)
        ch_value = np.nan
        status = "NA"
        if row.status == "PASS" and len(pass_subset) == EXPECTED_LOCI and int(finite.sum()) == EXPECTED_LOCI:
            selected = pass_subset["selected_logprob"].to_numpy(dtype=float)
            other = pass_subset["other_logprob"].to_numpy(dtype=float)
            values = np.concatenate([selected, other]).reshape(-1, 1)
            labels = np.concatenate(
                [np.ones(EXPECTED_LOCI, dtype=int), np.zeros(EXPECTED_LOCI, dtype=int)]
            )
            ch_value = float(calinski_harabasz_score(values, labels))
            status = "PASS"
        rows.append(
            {
                "model_id": row.model_id,
                "display_model": row.display_model,
                "plot_label": row.plot_label,
                "family": BACKEND.model_family(row.model_id),
                "model_type": row.model_type,
                "summary_status": row.status,
                # Preserve the source table's missing count for blocked models.
                # Falling back to EXPECTED_LOCI falsely implied DNABERT had data.
                "n_loci": int(row.n_loci) if pd.notna(row.n_loci) else np.nan,
                "accuracy": float(row.accuracy),
                "accuracy_ci_low": float(row.accuracy_ci_low),
                "accuracy_ci_high": float(row.accuracy_ci_high),
                "finite_selected_other_pairs": int(finite.sum()),
                "ch_index": ch_value,
                "ch_status": status,
                "score_definition": "Calinski-Harabasz on 30 selected and 30 other raw logprob values",
            }
        )
    ch = pd.DataFrame(rows)
    ch["n_loci"] = ch["n_loci"].astype("Int64")
    return ordered, ch


def write_source_data(ch: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ch.to_csv(output_path, index=False)


def _style_axis(ax: mpl.axes.Axes) -> None:
    ax.tick_params(axis="y", length=0, pad=1.2)
    ax.tick_params(axis="x", length=2, width=0.5, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def _add_row_washes(axes: tuple[mpl.axes.Axes, mpl.axes.Axes], ordered: pd.DataFrame) -> None:
    for y, model_id in zip(ordered["y"], ordered["model_id"]):
        if model_id in {"hEvo-base", "hEvo-large"}:
            for ax in axes:
                ax.axhspan(y - 0.46, y + 0.46, color=HEVO_WASH, lw=0, zorder=0)


def _format_ch(value: object, status: str) -> str:
    if status != "PASS" or pd.isna(value):
        return "NA"
    return f"{float(value):.3g}"


def build_panel(
    tables_dir: Path | None = None,
    *,
    figure=None,
    slot=None,
    integrated: bool = False,
    letter: str | None = None,
):
    configure_matplotlib()
    ordered, ch = prepare_data(tables_dir)

    if figure is None:
        figure = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor=WHITE)
        grid = figure.add_gridspec(
            1,
            2,
            left=0.35,
            right=0.91,
            bottom=0.22,
            top=0.86,
            width_ratios=[0.86, 1.08],
            wspace=0.27,
        )
    else:
        if integrated and slot is not None:
            # A4 needs a dedicated gutter for the 30 visible model labels.  Using the
            # full outer slot would place the left axis against the page edge.
            grid = figure.add_gridspec(
                1,
                2,
                left=0.150,
                right=0.940,
                bottom=0.085,
                top=0.255,
                width_ratios=[0.86, 1.08],
                wspace=0.27,
            )
        elif slot is None:
            grid = figure.add_gridspec(
                1,
                2,
                left=0.35,
                right=0.91,
                bottom=0.22,
                top=0.86,
                width_ratios=[0.86, 1.08],
                wspace=0.27,
            )
        else:
            grid = slot.subgridspec(
                1,
                2,
                width_ratios=[0.86, 1.08],
                wspace=0.27,
            )

    # BLOCKED rows have no plotted result and are intentionally omitted from
    # the visible panel. Keep them in the source table and calculation summary.
    visible_ordered = ordered.loc[
        ~ordered["model_id"].astype(str).str.lower().str.startswith("hevo-open-")
    ]
    plot_ordered = visible_ordered.loc[visible_ordered["status"].eq("PASS")].copy().reset_index(drop=True)
    plot_ordered["y"] = np.arange(len(plot_ordered), dtype=float)

    left_ax = figure.add_subplot(grid[0, 0])
    right_ax = figure.add_subplot(grid[0, 1], sharey=left_ax)
    _add_row_washes((left_ax, right_ax), plot_ordered)
    left_position = left_ax.get_position()
    right_position = right_ax.get_position()

    valid_rows = plot_ordered
    left_ax.errorbar(
        valid_rows["accuracy"].to_numpy(float) * 100.0,
        valid_rows["y"].to_numpy(float),
        xerr=[
            (valid_rows["accuracy"] - valid_rows["accuracy_ci_low"]).to_numpy(float) * 100.0,
            (valid_rows["accuracy_ci_high"] - valid_rows["accuracy"]).to_numpy(float) * 100.0,
        ],
        fmt="none",
        ecolor=NEUTRAL,
        elinewidth=0.55,
        capsize=1.5,
        zorder=2,
    )
    left_ax.scatter(
        valid_rows["accuracy"].to_numpy(float) * 100.0,
        valid_rows["y"].to_numpy(float),
        s=17,
        c=[_family_style(model)["color"] if model in {"hEvo-base", "hEvo-large"} else NEUTRAL_LIGHT for model in valid_rows["model_id"]],
        edgecolors=NEUTRAL,
        linewidths=0.45,
        zorder=3,
    )
    left_ax.axvline(50.0, color="#AAB2BC", linestyle="--", linewidth=0.55, zorder=1)
    left_ax.set_xlim(0, 100)
    left_ax.set_xticks([0, 25, 50, 75, 100])
    left_ax.set_xlabel("Loci favoring selected allele (%)", fontsize=5.8, labelpad=2)
    left_ax.set_yticks(plot_ordered["y"])
    left_ax.tick_params(axis="y", labelleft=False, left=False)
    left_ax.set_ylim(len(plot_ordered) - 0.5, -0.5)
    # Axes share the same y transform; draw the hEvo highlight rows as a
    # continuous band across both plots and the inter-panel gutter. This is
    # intentionally added after the final y limits are set, so the
    # data-to-figure transform has the correct 30-row geometry.
    for y, model_id in zip(plot_ordered["y"], plot_ordered["model_id"]):
        if model_id not in {"hEvo-base", "hEvo-large"}:
            continue
        y0 = figure.transFigure.inverted().transform(left_ax.transData.transform((0, y - 0.46)))[1]
        y1 = figure.transFigure.inverted().transform(left_ax.transData.transform((0, y + 0.46)))[1]
        # Figure-level artists are drawn above Axes, so only fill the
        # non-Axes gutter here; the two Axes interiors are already washed by
        # ``axhspan`` above and the external label/value columns must remain
        # available for their text artists.
        wash_ranges = ((left_position.x1, right_position.x0),)
        for wash_left, wash_right in wash_ranges:
            if wash_right <= wash_left:
                continue
            figure.add_artist(
                Rectangle(
                    (wash_left, y0),
                    wash_right - wash_left,
                    y1 - y0,
                    transform=figure.transFigure,
                    facecolor=HEVO_WASH,
                    edgecolor="none",
                    zorder=1.0,
                )
                )
    # The reference panel uses a compact family dot followed by a left-aligned
    # model name. Draw these after the figure-level washes so highlighted rows
    # cannot cover their labels.
    for row in plot_ordered.itertuples(index=False):
        style = _family_style(row.model_id)
        left_ax.scatter(
            -0.275,
            row.y,
            transform=left_ax.get_yaxis_transform(),
            marker="o",
            s=10.5,
            facecolor=style["color"],
            edgecolor=BLACK,
            linewidth=0.35,
            clip_on=False,
            zorder=4,
        )
        left_ax.text(
            -0.235,
            row.y,
            row.plot_label,
            transform=left_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=4.1,
            fontweight="bold" if row.model_id in {"hEvo-base", "hEvo-large"} else "normal",
            color=BLACK,
            clip_on=False,
            zorder=4,
        )
    left_ax.set_title("Selected-allele preference", fontsize=6.2, fontweight="bold", pad=4)
    _style_axis(left_ax)
    left_ax.spines["left"].set_visible(False)

    ch_lookup = ch.set_index("model_id")
    for row in plot_ordered.itertuples(index=False):
        value = ch_lookup.loc[row.model_id, "ch_index"]
        status = ch_lookup.loc[row.model_id, "ch_status"]
        if status == "PASS":
            style = _family_style(row.model_id)
            marker = "o" if row.model_type == "masked" else "^"
            right_ax.scatter(
                float(value),
                row.y,
                marker=marker,
                s=24 if marker == "o" else 27,
                facecolor=style["color"],
                edgecolor=BLACK,
                linewidth=0.45,
                zorder=3,
            )
        right_ax.text(
            1.015,
            row.y,
            _format_ch(value, status),
            transform=right_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=4.45,
            color=BLACK,
            clip_on=False,
        )
    right_ax.set_xscale("log")
    right_ax.set_xlim(1e-6, 200)
    right_ax.set_xticks([1e-6, 1e-4, 1e-2, 1, 100])
    right_ax.set_xticklabels(["0.000001", "0.0001", "0.01", "1", "100"], fontsize=4.7)
    right_ax.xaxis.set_minor_locator(NullLocator())
    right_ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    right_ax.set_xlabel("Calinski-Harabasz index (log scale)", fontsize=5.8, labelpad=2)
    right_ax.set_title("Raw-score Calinski-Harabasz index", fontsize=6.2, fontweight="bold", pad=4)
    right_ax.tick_params(axis="y", labelleft=False, left=False)
    right_ax.grid(axis="x", color="#E3E7EB", linewidth=0.35, zorder=0)
    _style_axis(right_ax)
    right_ax.spines["left"].set_visible(False)

    left_position = left_ax.get_position()
    # Match the formal f/g letter: same x gutter, fontsize and top alignment.
    letter_x = 0.055 if integrated else 0.025
    letter_y = left_position.y1 + 0.018 if integrated else 0.935
    figure.text(
        letter_x,
        letter_y,
        letter or "h",
        fontsize=12.5 if integrated else 5.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=_family_style("hEvo-base")["color"], markeredgecolor=BLACK, markersize=3.2, label="hEVO"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=_family_style("NT_500m_MS")["color"], markeredgecolor=BLACK, markersize=3.2, label="NT"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=_family_style("evo2_1b_base")["color"], markeredgecolor=BLACK, markersize=3.2, label="EVO 2"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=_family_style("GPN-promoter")["color"], markeredgecolor=BLACK, markersize=3.2, label="GPN"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=NEUTRAL_LIGHT, markeredgecolor=BLACK, markersize=3.2, label="Other"),
            Line2D([0], [0], marker="o", color=BLACK, markerfacecolor=NEUTRAL_LIGHT, markeredgecolor=BLACK, markersize=3.5, label="Masked-LM score"),
            Line2D([0], [0], marker="^", color=BLACK, markerfacecolor=NEUTRAL_LIGHT, markeredgecolor=BLACK, markersize=3.8, label="Full-sequence score"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, 0.030 if integrated else 0.025),
        ncol=7,
        frameon=False,
        fontsize=4.45,
        handletextpad=0.25,
        columnspacing=0.75,
    )
    figure._hevo_fig4_h_source_data = ch
    figure._hevo_fig4_h_ordered_data = ordered
    return left_ax, right_ax


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    root = Path(tables_dir) if tables_dir is not None else default_tables_dir()
    out = Path(output_dir) if output_dir is not None else figure_dir("Fig4")
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor=WHITE)
    build_panel(root, figure=fig, integrated=False, letter="h")
    source_path = out / "h_adaptive_allele_preference_ch_source_data.csv"
    write_source_data(fig._hevo_fig4_h_source_data, source_path)
    fig.subplots_adjust(left=0.30, right=0.88, bottom=0.20, top=0.87)
    result = export_figure(
        fig,
        out,
        "h_adaptive_allele_preference",
        [
            root / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv",
            root / "S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv",
        ],
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
    )
    plt.close(fig)
    result["source_data"] = str(source_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(export_panel(args.output_dir, args.tables_dir))


if __name__ == "__main__":
    main()
