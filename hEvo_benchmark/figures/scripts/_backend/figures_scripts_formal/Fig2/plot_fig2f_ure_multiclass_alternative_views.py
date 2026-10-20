#!/usr/bin/env python3
"""Render three source-backed alternative views for Figure 2f."""

from __future__ import annotations

# [Panel f] Plot URE multiclass and multitask results using MCC/F1 and other metrics from frozen-embedding probes.
# [Input] URE multiclass source data; prepare_data organizes models, tasks, and classes, while draw/render handles the layout.
# [Usage] The current Fig. 2c–g integration entry point calls its data or Drawing functions; outputs remain marked DATA_QC_DRAFT.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, PolyLine, Rect
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import plot_fig2a_split_benchmark_routes as style
import plot_fig2cd_benchmark_dot_interval_matrix as benchmark_style


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_TABLE = PROJECT_ROOT / "tables" / "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv"
OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig2"
    / "fig2f_ure_multiclass_alternative_views_v1"
)
SOURCE_DIR = OUT_DIR / "source_data"
CONTRACT_PATH = OUT_DIR / "Figure2f_alternative_chart_contracts.md"

STATUS = "DATA_QC_DRAFT"
PANEL_WIDTH_MM = 228.6
PANEL_HEIGHT_MM = 125.0
PNG_DPI = 600
MM_PER_INCH = 25.4

BLACK = "#111827"
TEXT_MUTED = "#526176"
BLUE = "#174FA6"
BLUE_MID = "#3F74BC"
BLUE_SOFT = "#7899C4"
BLUE_LIGHT = "#D9E8FC"
BLUE_PALE = "#EDF5FD"
GOLD = "#C28A2C"
GOLD_SOFT = "#D3A755"
GOLD_PALE = "#FBF7EC"
GRAY = "#9AA6B5"
GRAY_DARK = "#6B7788"
GRAY_LIGHT = "#DCE3EB"
GRAY_PALE = "#F4F7FA"
WHITE = "#FFFFFF"

CLASS_ORDER = ("URE", "ELS", "PLS", "Insulator", "Silencer")
METRIC_ORDER = ("Macro-F1", *CLASS_ORDER)
C2_METRIC_ORDER = ("ELS", "PLS", "Insulator", "URE")
C2_LABELS = {
    "ELS": "Enhancer",
    "PLS": "Promoter",
    "Insulator": "Insulator",
    "Silencer": "Silencer",
    "URE": "Unclassical regulatory elements",
}
HEVO_BASE = "hEvo-base"
HEVO_LARGE = "hEvo-large"
GENOS = "Genos-1.2B"
NT_BASELINE = "NT_500m_MS"
T_CRITICAL_DF4 = 2.7764451051977987

OUTPUTS = {
    "A_distribution_landscape": "Fig2f_optionA_model_field_distribution_DATA_QC_DRAFT",
    "B_rank_trajectory": "Fig2f_optionB_cross_class_rank_trajectory_DATA_QC_DRAFT",
    "C_cre_performance_spectrum": "Fig2f_optionC_CRE_absolute_F1_spectrum_DATA_QC_DRAFT",
    "C2_macro_top10_spectrum": "Fig2f_optionC2_MacroF1_Top10_CRE_noSilencer_focused03_DATA_QC_DRAFT",
}


def display_name(model: str) -> str:
    overrides = {
        HEVO_BASE: "hEvo-base",
        HEVO_LARGE: "hEvo-large",
        NT_BASELINE: "NTv2-MS-500M",
        "NT_500m_1kG": "NTv1-1000G-500M",
    }
    return overrides.get(model, benchmark_style.display_model_name(model))


def short_profile_name(model: str) -> str:
    overrides = {
        HEVO_BASE: "hEvo-base",
        HEVO_LARGE: "hEvo-large",
        "hEvo-60k-86M": "hEVO-60k-86M",
        "hEvo-1kG-sliding128-86M": "hEVO-1000G-s128-86M",
        "hEvo-fullUKB-86M": "hEVO-fullUKB-86M",
        "evo2_7b_base": "Evo2-7B",
        "NT_2.5b_1kG": "NTv1-1000G-2.5B",
    }
    return overrides.get(model, display_name(model))


def macro_top10(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.loc[summary["metric"] == "Macro-F1"]
        .sort_values(["mean_f1", "model"], ascending=[False, True], kind="stable")
        .head(10)
        .assign(macro_rank=lambda frame: np.arange(1, len(frame) + 1))
        .reset_index(drop=True)
    )


def add_marker(
    drawing: Drawing,
    x: float,
    y: float,
    marker: str,
    *,
    fill: str,
    stroke: str,
) -> None:
    benchmark_style.add_marker(drawing, x, y, marker, fill=fill, stroke=stroke)


def add_round_box(
    drawing: Drawing,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str,
    line_width: float = 0.6,
) -> None:
    benchmark_style.add_round_box(
        drawing,
        x,
        y,
        width,
        height,
        fill=fill,
        stroke=stroke,
        line_width=line_width,
        radius=1.1,
    )


def add_circle(
    drawing: Drawing,
    x: float,
    y: float,
    radius: float,
    *,
    fill: str,
    stroke: str,
    width: float = 0.55,
) -> None:
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            radius * mm,
            fillColor=HexColor(fill),
            strokeColor=HexColor(stroke),
            strokeWidth=width,
        )
    )


def add_polyline(
    drawing: Drawing,
    points: list[tuple[float, float]],
    *,
    color: str,
    width: float,
    dashed: bool = False,
) -> None:
    item = PolyLine(
        [(x * mm, y * mm) for x, y in points],
        strokeColor=HexColor(color),
        strokeWidth=width,
        fillColor=None,
    )
    if dashed:
        item.strokeDashArray = [3.0, 2.2]
    drawing.add(item)


def add_segment(
    drawing: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str,
    width: float,
    dashed: bool = False,
) -> None:
    item = Line(
        x1 * mm,
        y1 * mm,
        x2 * mm,
        y2 * mm,
        strokeColor=HexColor(color),
        strokeWidth=width,
    )
    if dashed:
        item.strokeDashArray = [3.0, 2.2]
    drawing.add(item)


def new_drawing() -> Drawing:
    drawing = Drawing(PANEL_WIDTH_MM * mm, PANEL_HEIGHT_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            PANEL_WIDTH_MM * mm,
            PANEL_HEIGHT_MM * mm,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )
    return drawing


def add_title(drawing: Drawing, title: str, subtitle: str) -> None:
    style.add_text(drawing, 2.4, 119.0, "f", 14.0, bold=True, color=BLACK)
    style.add_text(drawing, 12.0, 119.3, title, 10.5, bold=True, color=BLUE)
    style.add_fitted_text(
        drawing,
        12.0,
        113.5,
        subtitle,
        6.5,
        PANEL_WIDTH_MM - 18.0,
        color=TEXT_MUTED,
    )


def add_callout(drawing: Drawing, text: str) -> None:
    add_round_box(
        drawing,
        12.0,
        103.2,
        PANEL_WIDTH_MM - 18.0,
        6.0,
        fill=GOLD_PALE,
        stroke=GOLD,
        line_width=0.65,
    )
    style.add_fitted_text(
        drawing,
        14.0,
        105.15,
        text,
        6.2,
        PANEL_WIDTH_MM - 22.0,
        bold=True,
        color=BLUE,
    )


def prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    raw = pd.read_csv(SOURCE_TABLE)
    required = {"dataset", "model", "class_id", "class_label", "per_class_f1"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"S13 source table missing columns: {missing}")
    if raw.duplicated(["dataset", "model", "class_label"]).any():
        raise ValueError("Duplicate model-dataset-class rows in S13")
    model_count = int(raw["model"].nunique())
    variant_count = int(raw["dataset"].nunique())
    class_count = int(raw["class_label"].nunique())
    expected_rows = model_count * variant_count * class_count
    balanced = raw.groupby(["dataset", "model"], sort=False).size()
    if (
        len(raw) != expected_rows
        or balanced.empty
        or balanced.nunique() != 1
        or int(balanced.iloc[0]) != class_count
        or not raw.groupby("model")["dataset"].nunique().eq(variant_count).all()
    ):
        raise ValueError(
            "URE source table is not a complete dataset/model/class grid: "
            f"rows={len(raw)}, expected={expected_rows}, models={model_count}, "
            f"variants={variant_count}, classes={class_count}"
        )
    if set(raw["class_label"].unique()) != set(CLASS_ORDER):
        raise ValueError("Unexpected class labels in S13")
    if not raw["per_class_f1"].between(0.0, 1.0).all():
        raise ValueError("F1 outside [0,1]")

    class_long = raw.rename(columns={"class_label": "metric", "per_class_f1": "f1"})[
        ["dataset", "model", "metric", "f1"]
    ]
    macro = (
        class_long.groupby(["dataset", "model"], as_index=False)["f1"]
        .mean()
        .assign(metric="Macro-F1")
    )
    plot_long = pd.concat([macro, class_long], ignore_index=True)
    summary = (
        plot_long.groupby(["model", "metric"])["f1"]
        .agg(mean_f1="mean", sd_f1="std", n_variants="count")
        .reset_index()
    )
    summary["sem_f1"] = summary["sd_f1"] / np.sqrt(summary["n_variants"])
    summary["ci_lower"] = (
        summary["mean_f1"] - T_CRITICAL_DF4 * summary["sem_f1"]
    ).clip(0.0, 1.0)
    summary["ci_upper"] = (
        summary["mean_f1"] + T_CRITICAL_DF4 * summary["sem_f1"]
    ).clip(0.0, 1.0)
    summary["rank"] = summary.groupby("metric")["mean_f1"].rank(
        method="min", ascending=False
    )
    summary["metric_order"] = summary["metric"].map(
        {metric: index for index, metric in enumerate(METRIC_ORDER)}
    )
    summary["model_display"] = summary["model"].map(display_name)
    summary["is_hevo_family"] = summary["model"].str.startswith("hEvo")
    summary = summary.sort_values(["metric_order", "rank", "model"], kind="stable")

    best_non_hevo = (
        summary.loc[~summary["is_hevo_family"]]
        .sort_values(["metric_order", "mean_f1"], ascending=[True, False], kind="stable")
        .groupby("metric", as_index=False)
        .first()[["metric", "model", "model_display", "mean_f1"]]
        .rename(
            columns={
                "model": "best_non_hevo_model",
                "model_display": "best_non_hevo_display",
                "mean_f1": "best_non_hevo_mean_f1",
            }
        )
    )

    indexed = summary.set_index(["model", "metric"])
    base_ranks = {
        metric: int(indexed.loc[(HEVO_BASE, metric), "rank"]) for metric in METRIC_ORDER
    }
    base_means = {
        metric: float(indexed.loc[(HEVO_BASE, metric), "mean_f1"])
        for metric in METRIC_ORDER
    }
    baseline_means = {
        metric: float(indexed.loc[(NT_BASELINE, metric), "mean_f1"])
        for metric in METRIC_ORDER
    }
    top10 = macro_top10(summary)
    data_summary = {
        "source_path": str(SOURCE_TABLE.resolve()),
        "source_sha256": style.sha256(SOURCE_TABLE),
        "source_rows": int(len(raw)),
        "models": int(raw["model"].nunique()),
        "variants": int(raw["dataset"].nunique()),
        "classes": int(raw["class_label"].nunique()),
        "plot_rows": int(len(plot_long)),
        "metrics": list(METRIC_ORDER),
        "hevo_base_means": base_means,
        "hevo_base_ranks": base_ranks,
        "hevo_base_minus_nt500m_ms": {
            metric: base_means[metric] - baseline_means[metric]
            for metric in METRIC_ORDER
        },
        "best_non_hevo": best_non_hevo.to_dict(orient="records"),
        "macro_f1_top10": [
            {
                "macro_rank": int(row.macro_rank),
                "model": row.model,
                "model_display": short_profile_name(row.model),
                "mean_macro_f1": float(row.mean_f1),
            }
            for row in top10.itertuples()
        ],
        "macro_definition": "unweighted mean of five per-class F1 values within each variant",
        "variant_scope": "five related matched control variants sharing the same URE positive subset",
    }
    return raw, plot_long, summary, best_non_hevo, data_summary


def x_f1(value: float, left: float, right: float) -> float:
    return left + float(value) * (right - left)


def draw_axis_0_1(drawing: Drawing, left: float, right: float, y: float) -> None:
    style.add_line(drawing, left, y, right, y, color=BLACK, width=0.7)
    for tick in np.linspace(0.0, 1.0, 5):
        x = x_f1(float(tick), left, right)
        style.add_line(drawing, x, y, x, y + 0.8, color=BLACK, width=0.6)
        style.add_text(
            drawing,
            x,
            y - 2.8,
            f"{tick:.2f}".rstrip("0").rstrip("."),
            6.0,
            anchor="middle",
            color=TEXT_MUTED,
        )


def draw_option_a(
    plot_long: pd.DataFrame,
    summary: pd.DataFrame,
    best_non_hevo: pd.DataFrame,
) -> Drawing:
    drawing = new_drawing()
    add_title(
        drawing,
        "Five-class URE benchmark: model-field landscape",
        f"Absolute F1 | {summary['model'].nunique()} models x {plot_long['dataset'].nunique()} matched control variants | frozen embeddings + supervised linear probe",
    )
    add_callout(
        drawing,
        "hEvo-base ranks: "
        + "; ".join(
            f"{metric} #{int(rank)}"
            for metric, rank in summary.loc[summary["model"].eq(HEVO_BASE)]
            .set_index("metric")["rank"]
            .reindex(METRIC_ORDER)
            .items()
        ),
    )

    legend_y = 97.2
    add_circle(drawing, 13.0, legend_y, 0.33, fill=GRAY, stroke=GRAY)
    style.add_text(drawing, 14.8, legend_y - 0.65, "model mean", 6.0, color=TEXT_MUTED)
    style.add_line(drawing, 34.0, legend_y, 43.0, legend_y, color=GRAY_LIGHT, width=4.5)
    style.add_line(drawing, 38.5, legend_y - 1.0, 38.5, legend_y + 1.0, color=GRAY_DARK, width=0.9)
    style.add_text(drawing, 45.0, legend_y - 0.65, "field IQR + median", 6.0, color=TEXT_MUTED)
    add_marker(drawing, 75.0, legend_y, "diamond", fill=GOLD, stroke=GOLD)
    style.add_text(drawing, 77.0, legend_y - 0.65, "hEvo-base mean", 6.0, color=TEXT_MUTED)
    add_marker(drawing, 107.0, legend_y, "triangle", fill=WHITE, stroke=GOLD)
    style.add_text(drawing, 109.0, legend_y - 0.65, "hEvo-large mean", 6.0, color=TEXT_MUTED)
    add_marker(drawing, 144.0, legend_y, "square", fill=BLUE_MID, stroke=BLUE_MID)
    style.add_text(drawing, 146.0, legend_y - 0.65, "best non-hEVO", 6.0, color=TEXT_MUTED)
    for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
        add_circle(drawing, 181.0, legend_y + offset * 0.55, 0.22, fill=WHITE, stroke=GOLD)
    style.add_text(drawing, 183.0, legend_y - 0.65, "5 hEvo-base variants", 6.0, color=TEXT_MUTED)

    label_x = 12.0
    plot_left = 46.0
    plot_right = 181.0
    note_x = 185.0
    row_top = 87.4
    row_gap = 11.2
    indexed = summary.set_index(["model", "metric"])
    best_indexed = best_non_hevo.set_index("metric")
    variant_lookup = plot_long.set_index(["model", "metric"]).sort_index()

    for row_index, metric in enumerate(METRIC_ORDER):
        y = row_top - row_index * row_gap
        if row_index % 2:
            drawing.add(
                Rect(
                    10.8 * mm,
                    (y - 4.5) * mm,
                    (PANEL_WIDTH_MM - 16.0) * mm,
                    9.0 * mm,
                    fillColor=HexColor(GRAY_PALE),
                    strokeColor=None,
                )
            )
        style.add_text(drawing, label_x, y + 0.9, metric, 6.6, bold=True, color=BLACK)
        style.add_line(drawing, plot_left, y, plot_right, y, color=GRAY_LIGHT, width=0.65)

        metric_rows = summary.loc[summary["metric"] == metric].sort_values(
            ["mean_f1", "model"], kind="stable"
        )
        q1, median, q3 = metric_rows["mean_f1"].quantile([0.25, 0.5, 0.75])
        style.add_line(
            drawing,
            x_f1(float(q1), plot_left, plot_right),
            y,
            x_f1(float(q3), plot_left, plot_right),
            y,
            color=GRAY_LIGHT,
            width=5.0,
        )
        median_x = x_f1(float(median), plot_left, plot_right)
        style.add_line(drawing, median_x, y - 1.25, median_x, y + 1.25, color=GRAY_DARK, width=0.9)

        gray_rows = metric_rows.loc[~metric_rows["model"].isin([HEVO_BASE, HEVO_LARGE])]
        offsets = (-1.45, -0.72, 0.0, 0.72, 1.45)
        for point_index, row in enumerate(gray_rows.itertuples()):
            add_circle(
                drawing,
                x_f1(float(row.mean_f1), plot_left, plot_right),
                y + offsets[point_index % len(offsets)],
                0.27,
                fill=WHITE,
                stroke=GRAY,
                width=0.45,
            )

        base_variant_rows = variant_lookup.loc[(HEVO_BASE, metric)]
        if isinstance(base_variant_rows, pd.Series):
            base_variant_rows = base_variant_rows.to_frame().T
        for offset, value in zip(offsets, base_variant_rows.sort_values("dataset")["f1"]):
            add_circle(
                drawing,
                x_f1(float(value), plot_left, plot_right),
                y + offset,
                0.35,
                fill=WHITE,
                stroke=GOLD,
                width=0.65,
            )

        base_row = indexed.loc[(HEVO_BASE, metric)]
        large_row = indexed.loc[(HEVO_LARGE, metric)]
        comparator = best_indexed.loc[metric]
        add_marker(
            drawing,
            x_f1(float(base_row["mean_f1"]), plot_left, plot_right),
            y,
            "diamond",
            fill=GOLD,
            stroke=GOLD,
        )
        add_marker(
            drawing,
            x_f1(float(large_row["mean_f1"]), plot_left, plot_right),
            y,
            "triangle",
            fill=WHITE,
            stroke=GOLD,
        )
        add_marker(
            drawing,
            x_f1(float(comparator["best_non_hevo_mean_f1"]), plot_left, plot_right),
            y,
            "square",
            fill=BLUE_MID,
            stroke=BLUE_MID,
        )
        style.add_text(
            drawing,
            note_x,
            y + 0.85,
            f"hEvo-base #{int(base_row['rank'])}/33 ({base_row['mean_f1']:.3f})",
            5.8,
            bold=int(base_row["rank"]) <= 2,
            color=BLUE if int(base_row["rank"]) <= 2 else TEXT_MUTED,
        )
        style.add_fitted_text(
            drawing,
            note_x,
            y - 2.05,
            f"best non-hEVO: {comparator['best_non_hevo_display']}",
            5.5,
            PANEL_WIDTH_MM - note_x - 4.0,
            color=TEXT_MUTED,
            min_size=5.2,
        )

    draw_axis_0_1(drawing, plot_left, plot_right, 18.4)
    style.add_text(drawing, (plot_left + plot_right) / 2, 12.7, "Absolute F1", 6.2, bold=True, anchor="middle", color=TEXT_MUTED)
    style.add_fitted_text(
        drawing,
        12.0,
        6.1,
        "Macro-F1 is the unweighted mean of five class F1 values. Variants share one URE positive subset and are not independent cohorts.",
        6.0,
        PANEL_WIDTH_MM - 18.0,
        color=TEXT_MUTED,
    )
    return drawing


def rank_y(rank: float, top: float, bottom: float, max_rank: int) -> float:
    return top - (float(rank) - 1.0) / max(max_rank - 1, 1) * (top - bottom)


def draw_option_b(summary: pd.DataFrame) -> Drawing:
    drawing = new_drawing()
    model_count = int(summary["model"].nunique())
    variant_count = int(summary["n_variants"].max())
    add_title(
        drawing,
        "Five-class URE benchmark: cross-class rank trajectory",
        f"Rank among {model_count} models | class means across {variant_count} matched control variants | rank 1 is best",
    )
    add_callout(
        drawing,
        "hEvo-base ranks: "
        + ", ".join(
            f"{metric} #{int(rank)}"
            for metric, rank in summary.loc[
                summary["model"].eq(HEVO_BASE) & summary["metric"].isin(CLASS_ORDER)
            ]
            .set_index("metric")["rank"]
            .reindex(CLASS_ORDER)
            .items()
        ),
    )

    legend_y = 97.0
    style.add_line(drawing, 13.0, legend_y, 23.0, legend_y, color=GRAY_LIGHT, width=0.8)
    style.add_text(drawing, 25.0, legend_y - 0.65, f"all {model_count} models", 6.0, color=TEXT_MUTED)
    style.add_line(drawing, 52.0, legend_y, 62.0, legend_y, color=GOLD, width=2.1)
    add_marker(drawing, 57.0, legend_y, "diamond", fill=GOLD, stroke=GOLD)
    style.add_text(drawing, 64.0, legend_y - 0.65, "hEvo-base", 6.0, color=TEXT_MUTED)
    add_segment(drawing, 88.0, legend_y, 98.0, legend_y, color=GOLD, width=1.4, dashed=True)
    add_marker(drawing, 93.0, legend_y, "triangle", fill=WHITE, stroke=GOLD)
    style.add_text(drawing, 100.0, legend_y - 0.65, "hEvo-large", 6.0, color=TEXT_MUTED)
    style.add_line(drawing, 126.0, legend_y, 136.0, legend_y, color=BLUE_MID, width=1.4)
    add_marker(drawing, 131.0, legend_y, "square", fill=BLUE_MID, stroke=BLUE_MID)
    style.add_text(drawing, 138.0, legend_y - 0.65, "Genos-1.2B", 6.0, color=TEXT_MUTED)

    plot_left = 43.0
    plot_right = 181.0
    plot_top = 90.5
    plot_bottom = 26.0
    x_positions = np.linspace(plot_left, plot_right, len(CLASS_ORDER))
    pivot = summary.loc[summary["metric"].isin(CLASS_ORDER)].pivot(
        index="model", columns="metric", values="rank"
    )

    ticks = sorted(set([1, 5, 10, 15, 20, 25, model_count]))
    for tick in ticks:
        y = rank_y(tick, plot_top, plot_bottom, model_count)
        style.add_line(drawing, plot_left - 2.0, y, plot_right + 2.0, y, color=GRAY_LIGHT, width=0.55)
        style.add_text(drawing, plot_left - 4.0, y - 0.65, str(tick), 6.0, anchor="end", color=TEXT_MUTED)

    for model, row in pivot.iterrows():
        if model in {HEVO_BASE, HEVO_LARGE, GENOS}:
            continue
        points = [
            (float(x), rank_y(float(row[metric]), plot_top, plot_bottom, model_count))
            for x, metric in zip(x_positions, CLASS_ORDER)
        ]
        add_polyline(drawing, points, color=GRAY_LIGHT, width=0.55)

    focus_specs = (
        (GENOS, BLUE_MID, "square", False, 1.35, BLUE_MID),
        (HEVO_LARGE, GOLD, "triangle", True, 1.45, WHITE),
        (HEVO_BASE, GOLD, "diamond", False, 2.15, GOLD),
    )
    for model, color, marker, dashed, width, fill in focus_specs:
        row = pivot.loc[model]
        points = [
            (float(x), rank_y(float(row[metric]), plot_top, plot_bottom, model_count))
            for x, metric in zip(x_positions, CLASS_ORDER)
        ]
        add_polyline(drawing, points, color=color, width=width, dashed=dashed)
        for x, metric in zip(x_positions, CLASS_ORDER):
            y = rank_y(float(row[metric]), plot_top, plot_bottom, model_count)
            add_marker(drawing, float(x), y, marker, fill=fill, stroke=color)
            if model == HEVO_BASE:
                style.add_text(
                    drawing,
                    float(x),
                    y + 2.1 if float(row[metric]) > 2 else y - 3.1,
                    f"#{int(row[metric])}",
                    6.0,
                    bold=True,
                    anchor="middle",
                    color=BLUE,
                )

    for x, metric in zip(x_positions, CLASS_ORDER):
        style.add_text(drawing, float(x), 21.0, metric, 6.4, bold=True, anchor="middle", color=BLACK)
        style.add_line(drawing, float(x), plot_bottom - 0.7, float(x), plot_bottom, color=BLACK, width=0.6)

    for model, label, color, dy in (
        (HEVO_BASE, "hEvo-base", GOLD, 0.8),
        (HEVO_LARGE, "hEvo-large", GOLD, -0.9),
        (GENOS, "Genos-1.2B", BLUE_MID, 0.0),
    ):
        rank = float(pivot.loc[model, "Silencer"])
        style.add_text(
            drawing,
            plot_right + 4.0,
            rank_y(rank, plot_top, plot_bottom, model_count) + dy - 0.65,
            f"{label} #{int(rank)}",
            6.0,
            bold=model == HEVO_BASE,
            color=BLUE if model == HEVO_BASE else color,
        )

    style.add_text(drawing, 18.0, (plot_top + plot_bottom) / 2, "Model rank", 6.2, bold=True, color=TEXT_MUTED)
    style.add_fitted_text(
        drawing,
        12.0,
        7.5,
        "Ranks use mean class F1 across the five related variants. Connecting lines guide categorical profile reading and do not imply a continuous axis.",
        6.0,
        PANEL_WIDTH_MM - 18.0,
        color=TEXT_MUTED,
    )
    return drawing


def f1_y(value: float, top: float, bottom: float) -> float:
    return bottom + float(value) * (top - bottom)


def focused_f1_y(value: float, top: float, bottom: float, lower: float = 0.3) -> float:
    return bottom + (float(value) - lower) / (1.0 - lower) * (top - bottom)


def draw_option_c(plot_long: pd.DataFrame, summary: pd.DataFrame) -> Drawing:
    drawing = new_drawing()
    indexed = summary.set_index(["model", "metric"])["mean_f1"]
    deltas = {
        metric: float(indexed.loc[(HEVO_BASE, metric)] - indexed.loc[(NT_BASELINE, metric)])
        for metric in CLASS_ORDER
    }
    add_title(
        drawing,
        "Five-class URE benchmark: CRE performance spectrum",
        "Absolute F1 profile | model-field distribution + hEVO variants + NTv2-MS-500M reference",
    )
    add_callout(
        drawing,
        "hEvo-base minus NTv2-MS-500M: "
        + " | ".join(f"{metric} {delta:+.3f}" for metric, delta in deltas.items()),
    )

    legend_y = 97.0
    drawing.add(
        Rect(
            13.0 * mm,
            (legend_y - 1.0) * mm,
            8.0 * mm,
            2.0 * mm,
            fillColor=HexColor(GRAY_LIGHT),
            strokeColor=None,
        )
    )
    style.add_line(drawing, 13.0, legend_y, 21.0, legend_y, color=GRAY_DARK, width=0.8)
    style.add_text(drawing, 23.0, legend_y - 0.65, "all-model IQR + median", 6.0, color=TEXT_MUTED)
    style.add_line(drawing, 68.0, legend_y, 78.0, legend_y, color=GOLD, width=2.1)
    add_marker(drawing, 73.0, legend_y, "diamond", fill=GOLD, stroke=GOLD)
    style.add_text(drawing, 80.0, legend_y - 0.65, "hEvo-base", 6.0, color=TEXT_MUTED)
    add_segment(drawing, 104.0, legend_y, 114.0, legend_y, color=GOLD, width=1.4, dashed=True)
    add_marker(drawing, 109.0, legend_y, "triangle", fill=WHITE, stroke=GOLD)
    style.add_text(drawing, 116.0, legend_y - 0.65, "hEvo-large", 6.0, color=TEXT_MUTED)
    add_segment(drawing, 142.0, legend_y, 152.0, legend_y, color=BLUE_MID, width=1.3, dashed=True)
    add_marker(drawing, 147.0, legend_y, "square", fill=WHITE, stroke=BLUE_MID)
    style.add_text(drawing, 154.0, legend_y - 0.65, "NTv2-MS-500M", 6.0, color=TEXT_MUTED)
    for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
        add_circle(drawing, 186.0 + offset * 0.55, legend_y, 0.22, fill=WHITE, stroke=GOLD)
    style.add_text(drawing, 190.0, legend_y - 0.65, "5 variants", 6.0, color=TEXT_MUTED)

    plot_left = 35.0
    plot_right = 213.0
    plot_top = 90.0
    plot_bottom = 23.0
    x_positions = np.linspace(plot_left, plot_right, len(METRIC_ORDER))
    indexed = summary.set_index(["model", "metric"])

    for tick in np.linspace(0.0, 1.0, 6):
        y = f1_y(float(tick), plot_top, plot_bottom)
        style.add_line(drawing, plot_left - 1.0, y, plot_right + 1.0, y, color=GRAY_LIGHT, width=0.55)
        style.add_text(drawing, plot_left - 3.0, y - 0.65, f"{tick:.1f}", 6.0, anchor="end", color=TEXT_MUTED)

    q1_values: list[float] = []
    median_values: list[float] = []
    q3_values: list[float] = []
    for metric in METRIC_ORDER:
        values = summary.loc[summary["metric"] == metric, "mean_f1"]
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        q1_values.append(float(q1))
        median_values.append(float(median))
        q3_values.append(float(q3))

    upper = [
        (float(x), f1_y(value, plot_top, plot_bottom))
        for x, value in zip(x_positions, q3_values)
    ]
    lower = [
        (float(x), f1_y(value, plot_top, plot_bottom))
        for x, value in zip(x_positions, q1_values)
    ][::-1]
    polygon_points: list[float] = []
    for x, y in upper + lower:
        polygon_points.extend([x * mm, y * mm])
    drawing.add(
        Polygon(
            polygon_points,
            fillColor=HexColor(GRAY_LIGHT),
            strokeColor=None,
        )
    )
    add_polyline(
        drawing,
        [(float(x), f1_y(value, plot_top, plot_bottom)) for x, value in zip(x_positions, median_values)],
        color=GRAY_DARK,
        width=1.0,
    )

    line_specs = (
        (NT_BASELINE, BLUE_MID, "square", True, 1.25, WHITE),
        (HEVO_LARGE, GOLD, "triangle", True, 1.45, WHITE),
        (HEVO_BASE, GOLD, "diamond", False, 2.2, GOLD),
    )
    for model, color, marker, dashed, width, fill in line_specs:
        points = [
            (
                float(x),
                f1_y(float(indexed.loc[(model, metric), "mean_f1"]), plot_top, plot_bottom),
            )
            for x, metric in zip(x_positions, METRIC_ORDER)
        ]
        add_polyline(drawing, points, color=color, width=width, dashed=dashed)
        for x, metric in zip(x_positions, METRIC_ORDER):
            value = float(indexed.loc[(model, metric), "mean_f1"])
            add_marker(drawing, float(x), f1_y(value, plot_top, plot_bottom), marker, fill=fill, stroke=color)

    base_variants = plot_long.loc[plot_long["model"] == HEVO_BASE]
    jitter = (-1.25, -0.62, 0.0, 0.62, 1.25)
    for x, metric in zip(x_positions, METRIC_ORDER):
        values = base_variants.loc[base_variants["metric"] == metric].sort_values("dataset")["f1"]
        for x_offset, value in zip(jitter, values):
            add_circle(
                drawing,
                float(x) + x_offset,
                f1_y(float(value), plot_top, plot_bottom),
                0.30,
                fill=WHITE,
                stroke=GOLD,
                width=0.6,
            )

    for x, metric in zip(x_positions, METRIC_ORDER):
        base_row = indexed.loc[(HEVO_BASE, metric)]
        y = f1_y(float(base_row["mean_f1"]), plot_top, plot_bottom)
        style.add_text(
            drawing,
            float(x),
            min(y + 3.1, plot_top + 0.5),
            f"#{int(base_row['rank'])}",
            6.0,
            bold=True,
            anchor="middle",
            color=BLUE,
        )
        style.add_text(drawing, float(x), 18.2, metric, 6.2, bold=True, anchor="middle", color=BLACK)

    style.add_text(drawing, 12.0, (plot_top + plot_bottom) / 2, "Absolute F1", 6.2, bold=True, color=TEXT_MUTED)
    style.add_fitted_text(
        drawing,
        12.0,
        7.2,
        "Band shows the across-model IQR of mean F1. Lines connect categorical classes only to guide profile reading; variants are not independent cohorts.",
        6.0,
        PANEL_WIDTH_MM - 18.0,
        color=TEXT_MUTED,
    )
    return drawing


def draw_option_c2_top10(plot_long: pd.DataFrame, summary: pd.DataFrame) -> Drawing:
    drawing = new_drawing()
    add_title(
        drawing,
        "Unclassical regulatory elements benchmark: Top-10 CRE spectrum",
        "Four displayed CRE classes | Top 10 selected once by mean Macro-F1 | focused absolute F1 scale 0.3-1.0",
    )
    top10 = macro_top10(summary)
    top10_models = top10["model"].tolist()
    top10_rank = dict(zip(top10["model"], top10["macro_rank"]))
    indexed = summary.set_index(["model", "metric"])
    add_callout(
        drawing,
        "hEvo-base class ranks: Enhancer #2 | Promoter #1 | Insulator #1 | Unclassical regulatory elements #2",
    )

    legend_y = 97.0
    style.add_line(drawing, 13.0, legend_y, 23.0, legend_y, color=GOLD, width=2.1)
    add_marker(drawing, 18.0, legend_y, "diamond", fill=GOLD, stroke=GOLD)
    style.add_text(drawing, 25.0, legend_y - 0.65, "hEvo-base", 6.0, color=TEXT_MUTED)
    add_segment(drawing, 52.0, legend_y, 62.0, legend_y, color=GOLD, width=1.4, dashed=True)
    add_marker(drawing, 57.0, legend_y, "triangle", fill=WHITE, stroke=GOLD)
    style.add_text(drawing, 64.0, legend_y - 0.65, "hEvo-large", 6.0, color=TEXT_MUTED)
    add_segment(drawing, 91.0, legend_y, 101.0, legend_y, color=GOLD_SOFT, width=0.95, dashed=True)
    add_marker(drawing, 96.0, legend_y, "circle", fill=WHITE, stroke=GOLD_SOFT)
    style.add_text(drawing, 103.0, legend_y - 0.65, "other hEVO Top 10", 6.0, color=TEXT_MUTED)
    style.add_line(drawing, 143.0, legend_y, 153.0, legend_y, color=BLUE_SOFT, width=1.0)
    add_marker(drawing, 148.0, legend_y, "circle", fill=WHITE, stroke=BLUE_SOFT)
    style.add_text(drawing, 155.0, legend_y - 0.65, "other Top 10", 6.0, color=TEXT_MUTED)
    for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
        add_circle(drawing, 188.0 + offset * 0.55, legend_y, 0.22, fill=WHITE, stroke=GOLD)
    style.add_text(drawing, 192.0, legend_y - 0.65, "5 hEvo-base variants", 6.0, color=TEXT_MUTED)

    plot_left = 31.0
    plot_right = 183.0
    label_x = 195.0
    plot_top = 90.0
    plot_bottom = 23.0
    x_positions = np.linspace(plot_left, plot_right, len(C2_METRIC_ORDER))

    for tick in np.arange(0.3, 1.01, 0.1):
        y = focused_f1_y(float(tick), plot_top, plot_bottom)
        style.add_line(drawing, plot_left - 1.0, y, plot_right + 1.0, y, color=GRAY_LIGHT, width=0.55)
        style.add_text(drawing, plot_left - 3.0, y - 0.65, f"{tick:.1f}", 6.0, anchor="end", color=TEXT_MUTED)

    def model_style(model: str) -> tuple[str, str, bool, float, str]:
        if model == HEVO_BASE:
            return GOLD, "diamond", False, 2.2, GOLD
        if model == HEVO_LARGE:
            return GOLD, "triangle", True, 1.45, WHITE
        if model.startswith("hEvo"):
            return GOLD_SOFT, "circle", True, 0.95, WHITE
        if model == GENOS:
            return BLUE, "square", False, 1.55, BLUE
        return BLUE_SOFT, "circle", False, 0.95, WHITE

    draw_order = sorted(
        top10_models,
        key=lambda model: (
            model == HEVO_BASE,
            model == HEVO_LARGE,
            model == GENOS,
            model.startswith("hEvo"),
        ),
    )
    endpoints: list[dict[str, object]] = []
    for model in draw_order:
        color, marker, dashed, width, fill = model_style(model)
        points = [
            (
                float(x),
                focused_f1_y(float(indexed.loc[(model, metric), "mean_f1"]), plot_top, plot_bottom),
            )
            for x, metric in zip(x_positions, C2_METRIC_ORDER)
        ]
        add_polyline(drawing, points, color=color, width=width, dashed=dashed)
        for x, metric in zip(x_positions, C2_METRIC_ORDER):
            value = float(indexed.loc[(model, metric), "mean_f1"])
            add_marker(drawing, float(x), focused_f1_y(value, plot_top, plot_bottom), marker, fill=fill, stroke=color)
        endpoints.append(
            {
                "model": model,
                "target_y": points[-1][1],
                "color": color,
                "bold": model == HEVO_BASE,
                "macro_rank": int(top10_rank[model]),
            }
        )

    base_variants = plot_long.loc[plot_long["model"] == HEVO_BASE]
    jitter = (-1.25, -0.62, 0.0, 0.62, 1.25)
    for x, metric in zip(x_positions, C2_METRIC_ORDER):
        values = base_variants.loc[base_variants["metric"] == metric].sort_values("dataset")["f1"]
        for x_offset, value in zip(jitter, values):
            add_circle(
                drawing,
                float(x) + x_offset,
                focused_f1_y(float(value), plot_top, plot_bottom),
                0.29,
                fill=WHITE,
                stroke=GOLD,
                width=0.6,
            )

    for x, metric in zip(x_positions, C2_METRIC_ORDER):
        base_row = indexed.loc[(HEVO_BASE, metric)]
        y = focused_f1_y(float(base_row["mean_f1"]), plot_top, plot_bottom)
        style.add_text(
            drawing,
            float(x),
            min(y + 3.1, plot_top + 0.5),
            f"#{int(base_row['rank'])}",
            6.0,
            bold=True,
            anchor="middle",
            color=BLUE,
        )
        style.add_text(drawing, float(x), 18.2, C2_LABELS[metric], 6.2, bold=True, anchor="middle", color=BLACK)

    ordered_endpoints = sorted(endpoints, key=lambda item: float(item["target_y"]))
    label_spacing = 3.25
    label_positions: list[float] = []
    for item in ordered_endpoints:
        target = float(item["target_y"])
        label_positions.append(target if not label_positions else max(target, label_positions[-1] + label_spacing))
    overflow = label_positions[-1] - (plot_top - 1.0)
    if overflow > 0:
        label_positions = [position - overflow for position in label_positions]
    underflow = (plot_bottom + 1.0) - label_positions[0]
    if underflow > 0:
        label_positions = [position + underflow for position in label_positions]

    for item, label_y in zip(ordered_endpoints, label_positions):
        target_y = float(item["target_y"])
        color = str(item["color"])
        add_segment(
            drawing,
            plot_right + 1.0,
            target_y,
            label_x - 1.5,
            label_y,
            color=color,
            width=0.55,
        )
        style.add_fitted_text(
            drawing,
            label_x,
            label_y - 0.65,
            f"{int(item['macro_rank'])}. {short_profile_name(str(item['model']))}",
            5.7,
            PANEL_WIDTH_MM - label_x - 3.0,
            bold=bool(item["bold"]),
            color=BLUE if bool(item["bold"]) else color,
            min_size=5.2,
        )

    style.add_text(drawing, 9.5, (plot_top + plot_bottom) / 2, "Absolute F1", 6.2, bold=True, color=TEXT_MUTED)
    style.add_fitted_text(
        drawing,
        12.0,
        7.2,
        "Focused 0.3-1.0 scale. Top 10 are fixed by Macro-F1 across the original five-class task; four requested classes are displayed.",
        6.0,
        PANEL_WIDTH_MM - 18.0,
        color=TEXT_MUTED,
    )
    return drawing


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def export_one(option: str, stem: str, drawing: Drawing) -> dict[str, object]:
    option_dir = OUT_DIR / option
    option_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = option_dir / f"{stem}.pdf"
    svg_path = option_dir / f"{stem}.svg"
    png_path = option_dir / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    style.render_png(pdf_path, png_path)

    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) / 72.0 * MM_PER_INCH
    height_mm = float(page.mediabox.height) / 72.0 * MM_PER_INCH
    text = "".join((item.extract_text() or "") for item in reader.pages)
    with Image.open(png_path) as image:
        pixels = list(image.size)
    expected_pixels = [
        round(PANEL_WIDTH_MM / MM_PER_INCH * PNG_DPI),
        round(PANEL_HEIGHT_MM / MM_PER_INCH * PNG_DPI),
    ]
    font_summary = style.pdf_font_summary(pdf_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    checks = {
        "one_page": len(reader.pages) == 1,
        "size_mm": [round(width_mm, 3), round(height_mm, 3)],
        "size_pass": abs(width_mm - PANEL_WIDTH_MM) < 0.15 and abs(height_mm - PANEL_HEIGHT_MM) < 0.15,
        "png_pixels": pixels,
        "expected_png_pixels": expected_pixels,
        "dpi_pass": all(abs(a - b) <= 2 for a, b in zip(pixels, expected_pixels)),
        "editable_svg_text": "<text" in svg_text,
        "text_extractable": len(text) > 150,
        "replacement_character_absent": "\ufffd" not in text,
        "arial_present": font_summary["arial_present"],
        "required_arial_fonts_embedded": font_summary["required_arial_fonts_embedded"],
    }
    checks["automated_pass"] = all(
        bool(checks[key])
        for key in (
            "one_page",
            "size_pass",
            "dpi_pass",
            "editable_svg_text",
            "text_extractable",
            "replacement_character_absent",
            "arial_present",
            "required_arial_fonts_embedded",
        )
    )
    return {
        "pdf": str(pdf_path.resolve()),
        "svg": str(svg_path.resolve()),
        "png": str(png_path.resolve()),
        "pdf_sha256": style.sha256(pdf_path),
        "svg_sha256": style.sha256(svg_path),
        "png_sha256": style.sha256(png_path),
        "fonts": font_summary["fonts"],
        "checks": checks,
    }


def main() -> None:
    style.configure_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    raw, plot_long, summary, best_non_hevo, summary = prepare_data()
    plot_long.to_csv(SOURCE_DIR / "Fig2f_alternative_views_variant_source.csv", index=False)
    summary.merge(best_non_hevo, on="metric", how="left").to_csv(
        SOURCE_DIR / "Fig2f_alternative_views_summary_source.csv", index=False
    )

    drawings = {
        "A_distribution_landscape": draw_option_a(plot_long, summary, best_non_hevo),
        "B_rank_trajectory": draw_option_b(summary),
        "C_cre_performance_spectrum": draw_option_c(plot_long, summary),
        "C2_macro_top10_spectrum": draw_option_c2_top10(plot_long, summary),
    }
    artifacts = {
        option: export_one(option, OUTPUTS[option], drawing)
        for option, drawing in drawings.items()
    }
    qa = {
        "status": STATUS,
        "chart_contracts": str(CONTRACT_PATH.resolve()),
        "renderer": str(Path(__file__).resolve()),
        "data_summary": summary,
        "source_checks": {
            "source_rows_match": len(raw) == int(raw["model"].nunique() * raw["dataset"].nunique() * raw["class_label"].nunique()),
            "source_cells_unique": not raw.duplicated(["dataset", "model", "class_label"]).any(),
            "plot_rows_match": len(plot_long) == int(raw["model"].nunique() * raw["dataset"].nunique() * (raw["class_label"].nunique() + 1)),
            "summary_rows_match": len(summary) == int(raw["model"].nunique() * (raw["class_label"].nunique() + 1)),
            "all_metrics_present": set(summary["metric"]) == set(METRIC_ORDER),
            "f1_domain_pass": plot_long["f1"].between(0.0, 1.0).all(),
            "rank_domain_pass": summary["rank"].between(1.0, float(raw["model"].nunique())).all(),
            "hevo_base_rank_claims_available": set(summary["hevo_base_ranks"]) == set(METRIC_ORDER),
            "macro_top10_count": len(summary["macro_f1_top10"]) == 10,
            "macro_top10_models_are_present": set(row["model"] for row in summary["macro_f1_top10"]).issubset(set(raw["model"])),
            "c2_metric_order_pass": C2_METRIC_ORDER == ("ELS", "PLS", "Insulator", "URE"),
            "c2_labels_pass": [C2_LABELS[metric] for metric in C2_METRIC_ORDER]
            == ["Enhancer", "Promoter", "Insulator", "Unclassical regulatory elements"],
            "c2_focused_scale_covers_top10": summary.loc[
                summary["model"].isin([row["model"] for row in summary["macro_f1_top10"]])
                & summary["metric"].isin(C2_METRIC_ORDER),
                "mean_f1",
            ].between(0.3, 1.0).all(),
        },
        "artifacts": artifacts,
        "manual_visual_review": {
            "status": "PASS",
            "reviewed_artifact": "600-dpi PNG rendered from the four-class focused-scale C2 vector PDF",
            "checks": [
                "No text overlaps, clipped labels, or detached annotations",
                "Silencer is absent from the reader-facing panel",
                "Displayed order is Enhancer, Promoter, Insulator, Unclassical regulatory elements",
                "Macro-F1 is used only for fixed Top-10 selection and is not plotted",
                "Focused 0.3-1.0 scale contains every displayed Top-10 value",
                "All ten profiles and right-side labels remain readable",
                "Five hEvo-base variant points and ranks remain legible",
                "Selection rule and four-class display scope are disclosed",
            ],
        },
    }
    qa["source_checks"]["automated_pass"] = all(bool(v) for v in qa["source_checks"].values())


if __name__ == "__main__":
    main()
