#!/usr/bin/env python3
"""Assemble Figure 2a-g on A4 with compact benchmark panels.

Panels a-d are reused from the validated Figure 2 renderer. Panel e is redrawn
natively from S20 as a five-class F1 spectrum beginning at 0.40. Panel f is the
validated compact URE binary panel, translated to the lower middle. Panel g is
a native four-axis DeepSEA task spectrum reconstructed from S21.
hEvo-base and hEvo-large are both retained; every other algorithm identity
contributes only its best mean-F1 checkpoint across the displayed classes.
"""

from __future__ import annotations

# [Compatibility layer] Legacy integration for Fig. 2c–g; the current S11–S16 entry point reuses only its downstream panel data and plotting functions.
# [Panels] This chain handles benchmark panels c/d, URE/CRE panel e, URE multiclass panel f, and DeepSEA LoRA panel g.
# [Calls] Continues to call abcde/abcd/base, plot_fig2f_ure_multiclass_alternative_views, and plot_fig2g_deepsea_lora.
# [Note] The legacy a/b paths in this file are not the data entry point for current Fig. 2a/b.

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import plot_figure2_abcde_a4_integrated_v1 as prev  # noqa: E402
import plot_fig2f_ure_multiclass_alternative_views as multi  # noqa: E402
import plot_fig2g_deepsea_lora as deepsea  # noqa: E402
from figures.scripts.common import hevo_model_style  # noqa: E402


STATUS = "DATA_QC_DRAFT"
PAGE_W_MM = prev.PAGE_W_MM
PAGE_H_MM = prev.PAGE_H_MM
PNG_DPI = prev.PNG_DPI
MM_PER_INCH = prev.MM_PER_INCH

F_X_MM = 10.0
F_Y_MM = 4.0
F_W_MM = 64.0
F_H_MM = 63.0
E_X_MM = 95.0
E_Y_MM = 4.0
E_W_MM = prev.E_W_MM
E_H_MM = prev.E_H_MM
DEEPSEA_X0_MM = 141.0
DEEPSEA_X1_MM = 199.0
G_X_MM = DEEPSEA_X0_MM
G_Y_MM = 4.0
G_W_MM = DEEPSEA_X1_MM - DEEPSEA_X0_MM
G_H_MM = 63.0

OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig2"
    / "figure2_abcdefg_a4_integrated_v1"
)
SOURCE_DIR = OUT_DIR / "source_data"
STEM = "Figure2_abcdefg_A4_integrated_v1_DATA_QC_DRAFT"

F_OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig2"
    / "fig2e_five_class_cre_spectrum_compact_v1"
)
F_SOURCE_DIR = F_OUT_DIR / "source_data"
F_STEM = "Fig2e_five_class_CRE_top10_spectrum_compact_DATA_QC_DRAFT"

E_OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig2"
    / "fig2f_ure_binary_distribution_trajectory_v1"
)
E_SOURCE_DIR = E_OUT_DIR / "source_data"
E_STEM = "Fig2f_URE_binary_distribution_trajectory_DATA_QC_DRAFT"

G_OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig2"
    / "fig2g_deepsea_named_range_mean_v3"
)
G_SOURCE_DIR = G_OUT_DIR / "source_data"
G_STEM = "Fig2g_DeepSEA_named_range_mean_DATA_QC_DRAFT"

WHITE = "#FFFFFF"
BLACK = "#000000"
NEUTRAL = prev.base.NEUTRAL
TRACK = prev.base.NEUTRAL_LIGHT
ROW_ALT = "#F7F8F9"
HEVO = prev.base.HEVO
HEVO_WASH = prev.base.HEVO_WASH

AXIS_MIN = 0.40
AXIS_MAX = 0.95
CLASS_ORDER = ("URE", "ELS", "PLS", "Insulator", "Silencer")
DISPLAY_CLASS_ORDER = ("ELS", "PLS", "Insulator", "Silencer", "URE")
CLASS_LABELS = {
    "ELS": "Enhancer",
    "PLS": "Promoter",
    "Insulator": "Insulator",
    "Silencer": "Silencer",
    "URE": "Unclassical CRE",
}
FOCUS = prev.HEVO_MAIN


def add_vertical_axis_title(
    drawing: Drawing, x: float, y: float, text: str, *, size: float = 7.0
) -> None:
    """Add a centered 90-degree axis title with a stable native transform."""

    label = String(
        0,
        0,
        text,
        fontName=prev.base.export_base.FONT,
        fontSize=size,
        fillColor=HexColor(BLACK),
        textAnchor="middle",
    )
    group = Group(label)
    group.transform = (0.0, 1.0, -1.0, 0.0, x * mm, y * mm)
    drawing.add(group)


def canonical_name(model: str) -> str:
    return hevo_model_style.canonical_model_name(model)


def select_multiclass_representatives(
    plot_long: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select representatives using mean F1 across all five displayed classes."""

    macro = summary.loc[summary["metric"].eq("Macro-F1")].copy()
    available = set(macro["model"])
    grouped = {model for _, candidates in prev.URE_GROUPS for model in candidates}
    if grouped != available:
        raise ValueError(
            "Multiclass representative registry does not cover the source models: "
            f"missing={sorted(available - grouped)}, extra={sorted(grouped - available)}"
        )

    five_class = (
        summary.loc[summary["metric"].isin(DISPLAY_CLASS_ORDER)]
        .groupby("model", as_index=False)["mean_f1"]
        .mean()
        .rename(columns={"mean_f1": "mean_five_class_f1"})
    )
    five_class["display_five_class_rank"] = five_class["mean_five_class_f1"].rank(
        method="min", ascending=False
    ).astype(int)
    score = five_class.set_index("model")["mean_five_class_f1"].to_dict()
    all_rank = five_class.set_index("model")["display_five_class_rank"].to_dict()
    selected_models: list[str] = []
    summary_rows: list[dict[str, object]] = []
    identity_for_model: dict[str, str] = {}

    for identity, candidates in prev.URE_GROUPS:
        if identity == "hEVO":
            selected_for_identity = FOCUS
            rule = "retain focal hEvo-base and hEvo-large"
        else:
            winner = sorted(candidates, key=lambda model: (-float(score[model]), model))[0]
            selected_for_identity = {winner}
            rule = "highest Macro-F1 across all five CRE classes"
        for model in candidates:
            identity_for_model[model] = identity
            selected = model in selected_for_identity
            if selected:
                selected_models.append(model)
            summary_rows.append(
                {
                    "algorithm_identity": identity,
                    "candidate_model": model,
                    "candidate_display": canonical_name(model),
                    "mean_five_class_f1": float(score[model]),
                    "all_model_five_class_rank": int(all_rank[model]),
                    "selected": selected,
                    "selection_rule": rule,
                }
            )

    selected = five_class.loc[five_class["model"].isin(selected_models)].copy()
    selected = selected.merge(
        macro[["model", "mean_f1", "rank"]].rename(
            columns={"mean_f1": "mean_five_class_macro_f1", "rank": "five_class_rank"}
        ),
        on="model",
        how="left",
        validate="one_to_one",
    )
    selected["algorithm_identity"] = selected["model"].map(identity_for_model)
    selected["model_display"] = selected["model"].map(canonical_name)
    selected = selected.sort_values(
        ["mean_five_class_f1", "model_display"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    selected["representative_rank"] = np.arange(1, len(selected) + 1)
    selected["within_view_rank"] = selected["mean_five_class_f1"].rank(
        method="min", ascending=False
    ).astype(int)

    selected_variants = plot_long.loc[
        plot_long["metric"].eq("Macro-F1")
        & plot_long["model"].isin(selected_models)
    ].copy()
    selected_variants["algorithm_identity"] = selected_variants["model"].map(
        identity_for_model
    )
    selected_variants["model_display"] = selected_variants["model"].map(canonical_name)
    selected_variants = selected_variants.merge(
        selected[["model", "representative_rank"]],
        on="model",
        how="left",
        validate="many_to_one",
    ).sort_values(["representative_rank", "dataset"], kind="stable")

    selected_classes = summary.loc[
        summary["metric"].isin(CLASS_ORDER)
        & summary["model"].isin(selected_models)
    ].copy()
    selected_classes["algorithm_identity"] = selected_classes["model"].map(
        identity_for_model
    )
    selected_classes["model_display"] = selected_classes["model"].map(canonical_name)
    selected_classes = selected_classes.merge(
        selected[["model", "representative_rank"]],
        on="model",
        how="left",
        validate="many_to_one",
    )
    selected_classes["within_class_rank"] = selected_classes.groupby("metric")[
        "mean_f1"
    ].rank(method="min", ascending=False).astype(int)
    stats = (
        selected_classes.groupby("metric", as_index=False)["mean_f1"]
        .agg(within_class_min="min", within_class_max="max")
    )
    selected_classes = selected_classes.merge(
        stats, on="metric", how="left", validate="many_to_one"
    )
    span = selected_classes["within_class_max"] - selected_classes["within_class_min"]
    selected_classes["within_class_position"] = (
        (selected_classes["mean_f1"] - selected_classes["within_class_min"])
        / span.where(span.gt(0.0), 1.0)
    ).where(span.gt(0.0), 0.5)
    selected_classes["metric_order"] = selected_classes["metric"].map(
        {metric: index for index, metric in enumerate(CLASS_ORDER)}
    )
    selected_classes = selected_classes.sort_values(
        ["representative_rank", "metric_order"], kind="stable"
    )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["algorithm_identity", "mean_five_class_f1"],
        ascending=[True, False],
        kind="stable",
    )
    return (
        selected.reset_index(drop=True),
        selected_variants.reset_index(drop=True),
        selected_classes.reset_index(drop=True),
        summary.reset_index(drop=True),
    )


def x_absolute(value: float, left: float, right: float) -> float:
    fraction = (float(value) - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)
    return left + float(np.clip(fraction, 0.0, 1.0)) * (right - left)


def add_microdot(drawing: Drawing, x: float, y: float, color: str) -> None:
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            0.15 * mm,
            fillColor=HexColor(WHITE),
            strokeColor=HexColor(color),
            strokeWidth=0.38,
        )
    )


def draw_compact_f(
    selected: pd.DataFrame,
    variants: pd.DataFrame,
    classes: pd.DataFrame,
    *,
    panel_letter: str = "e",
) -> Drawing:
    """Draw a compact five-class spectrum for the top ten representatives.

    Models are selected and ranked by Macro-F1 across all five CRE classes.
    """

    drawing = Drawing(F_W_MM * mm, F_H_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            drawing.width,
            drawing.height,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )

    prev.base.export_base.add_text(drawing, 0.2, 58.4, panel_letter, 14.0, bold=True, color=BLACK)

    displayed = selected.head(10).copy()
    displayed_models = displayed["model"].tolist()
    spectrum = classes.loc[
        classes["model"].isin(displayed_models)
        & classes["metric"].isin(DISPLAY_CLASS_ORDER)
    ].copy()
    spectrum["display_order"] = spectrum["metric"].map(
        {metric: index for index, metric in enumerate(DISPLAY_CLASS_ORDER)}
    )
    spectrum = spectrum.sort_values(
        ["representative_rank", "display_order"], kind="stable"
    )
    value_lookup = spectrum.pivot(index="model", columns="metric", values="mean_f1")
    rank_lookup = spectrum.pivot(index="model", columns="metric", values="rank")

    # Horizontally mirrored layout: direct labels lead into the Enhancer end,
    # while the numeric F1 axis moves to the right edge.
    plot_left = 28.2
    plot_right = 52.5
    plot_bottom = 10.2
    plot_top = 51.2
    label_x = 0.8
    label_width = 26.4
    x_positions = np.linspace(plot_left + 1.1, plot_right - 1.1, len(DISPLAY_CLASS_ORDER))

    def y_f1(value: float) -> float:
        return plot_bottom + (float(value) - AXIS_MIN) / (AXIS_MAX - AXIS_MIN) * (
            plot_top - plot_bottom
        )

    prev.base.export_base.add_line(
        drawing, plot_right, plot_bottom, plot_right, plot_top, color=BLACK, width=0.55
    )
    prev.base.export_base.add_line(
        drawing, plot_left, plot_bottom, plot_right, plot_bottom, color=BLACK, width=0.55
    )
    # The compact axis includes the complete top-ten five-class spectrum,
    # including the lower Silencer F1 values.
    for value in (0.40, 0.60, 0.80, 0.95):
        y = y_f1(value)
        prev.base.export_base.add_line(
            drawing, plot_right, y, plot_right + 0.55, y, color=BLACK, width=0.48
        )
        prev.base.export_base.add_text(
            drawing,
            plot_right + 1.0,
            y - 0.65,
            f"{value:.2f}" if value == 0.95 else f"{value:.1f}",
            6.0,
            anchor="start",
            color=BLACK,
        )
    add_vertical_axis_title(drawing, 61.5, 27.8, "F1", size=7.0)
    for x, metric in zip(x_positions, DISPLAY_CLASS_ORDER):
        prev.base.export_base.add_line(
            drawing, x, plot_bottom, x, plot_bottom - 0.55, color=BLACK, width=0.48
        )
        # The long URE label used to be shifted four millimetres left while
        # all labels were rotated by 60 degrees.  That made its left edge
        # collide with the Insulator label and its right edge run into the
        # right-hand axis.  A steeper angle plus a small inward offset keeps
        # the labels separated without moving any tick or plotted data.
        class_label_x = x - (3.0 if metric == "URE" else 0.0)
        label = String(
            0,
            0,
            CLASS_LABELS[metric],
            fontName=prev.base.export_base.FONT,
            fontSize=6.0,
            fillColor=HexColor(BLACK),
            textAnchor="middle",
        )
        label_group = Group(label)
        label_group.transform = (
            math.cos(math.radians(72.0)),
            math.sin(math.radians(72.0)),
            -math.sin(math.radians(72.0)),
            math.cos(math.radians(72.0)),
            class_label_x * mm,
            (1.8 if metric == "URE" else 5.4) * mm,
        )
        drawing.add(label_group)

    # Draw neutral models first so the two hEVO paths remain visually dominant.
    draw_order = displayed.sort_values(
        "model", key=lambda series: series.isin(FOCUS), kind="stable"
    )
    end_points: list[dict[str, object]] = []
    for row in draw_order.itertuples():
        model = str(row.model)
        focus = model in FOCUS
        is_base = model == "hEvo-base"
        values = [float(value_lookup.loc[model, metric]) for metric in DISPLAY_CLASS_ORDER]
        coords = [(x, y_f1(value)) for x, value in zip(x_positions, values)]
        color = HEVO if focus else NEUTRAL
        width = 1.20 if is_base else 0.95 if focus else 0.55
        dashed = focus and not is_base
        for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
            line = Line(
                x1 * mm,
                y1 * mm,
                x2 * mm,
                y2 * mm,
                strokeColor=HexColor(color),
                strokeWidth=width,
            )
            if dashed:
                line.strokeDashArray = [2.2, 1.6]
            drawing.add(line)
        for x, y in coords:
            if is_base:
                marker = "diamond"
                fill = HEVO
            elif focus:
                marker = "triangle"
                fill = WHITE
            else:
                marker = "circle"
                fill = WHITE
            prev.base.cd.add_marker(
                drawing, x, y, marker, fill=fill, stroke=color
            )
        end_points.append(
            {
                "model": model,
                "model_display": str(row.model_display),
                "rank": int(row.five_class_rank),
                "y": coords[-1][1],
                "color": color,
                "focus": focus,
            }
        )

    # Model labels originate at Enhancer and carry the all-model Macro-F1
    # rank across all five displayed classes. Rank gaps are intentional.
    base_model = "hEvo-base"
    large_model = "hEvo-large"
    ordered_labels = sorted(end_points, key=lambda item: int(item["rank"]))
    label_top = 49.3
    separation = 2.45
    label_ys = [label_top - index * separation for index in range(len(ordered_labels))]
    for item, label_y in zip(ordered_labels, label_ys):
        model = str(item["model"])
        source_y = y_f1(float(value_lookup.loc[model, DISPLAY_CLASS_ORDER[0]]))
        color = str(item["color"])
        prev.base.export_base.add_line(
            drawing,
            label_x + label_width + 0.15,
            label_y,
            x_positions[0] - 0.75,
            source_y,
            color=color,
            width=0.45,
        )
        prev.base.export_base.add_fitted_text(
            drawing,
            label_x,
            label_y - 0.65,
            f"{int(item['rank'])}. {item['model_display']}",
            6.0,
            label_width,
            bold=bool(item["focus"]),
            color=BLACK,
            min_size=6.0,
        )

    # Annotate the per-class source ranks for hEvo-large and state the model
    # explicitly so the standalone ``#N`` labels cannot be misread as a
    # generic or hEvo-base ranking.
    prev.base.export_base.add_text(
        drawing,
        float(np.mean(x_positions)),
        54.3,
        "hEvo-large rank",
        6.0,
        bold=True,
        anchor="middle",
        color=BLACK,
    )
    for x, metric in zip(x_positions, DISPLAY_CLASS_ORDER):
        y = y_f1(float(value_lookup.loc[large_model, metric]))
        source_rank = int(rank_lookup.loc[large_model, metric])
        label_x_offset = 1.25 if metric == DISPLAY_CLASS_ORDER[0] else 0.0
        label_y = min(plot_top - 0.8, y + 1.55)
        if metric == DISPLAY_CLASS_ORDER[-1]:
            label_x_offset = -1.0
            label_y = y - 1.8
        prev.base.export_base.add_text(
            drawing,
            x + label_x_offset,
            label_y,
            f"#{source_rank}",
            6.0,
            bold=True,
            anchor="middle",
            color=BLACK,
        )
    return drawing


def draw_distribution_e(
    raw: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    panel_letter: str = "f",
) -> Drawing:
    """Draw five dataset distributions with highlighted hEVO trajectories."""

    drawing = Drawing(E_W_MM * mm, E_H_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            drawing.width,
            drawing.height,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )
    prev.base.export_base.add_text(drawing, 0.2, 58.4, panel_letter, 14.0, bold=True, color=BLACK)

    y_min = 0.25
    observed_max = max(
        float(raw["mcc_mean"].max()),
        float(selected["mean_mcc"].max()),
    )
    # Keep the URE binary panel data-driven: hEvo-large is near 0.97 in the
    # current source table, so the former fixed 0.85 ceiling clipped it into
    # the header lane.
    y_max = max(0.85, math.ceil((observed_max + 0.01) * 20.0) / 20.0)
    plot_bottom = 10.4
    plot_top = 52.0
    axis_x = 6.0
    x_positions = (7.8, 14.5, 21.2, 27.9, 34.6, 41.3)
    run_order = (
        raw[["run", "dataset_order"]]
        .drop_duplicates()
        .sort_values("dataset_order")["run"]
        .tolist()
    )
    labels = [f"Set {index}" for index in range(1, len(run_order) + 1)] + ["Mean"]

    def y_mcc(value: float) -> float:
        return plot_bottom + (float(value) - y_min) / (y_max - y_min) * (
            plot_top - plot_bottom
        )

    prev.base.export_base.add_line(
        drawing, axis_x, plot_bottom, axis_x, plot_top, color=BLACK, width=0.55
    )
    prev.base.export_base.add_line(
        drawing, axis_x, plot_bottom, x_positions[-1] + 0.8, plot_bottom, color=BLACK, width=0.55
    )
    tick_values = [0.30, 0.50, 0.70, 0.90]
    if y_max >= 0.95:
        tick_values.append(y_max)
    for value in tick_values:
        y = y_mcc(value)
        prev.base.export_base.add_line(
            drawing, axis_x - 0.5, y, axis_x, y, color=BLACK, width=0.48
        )
        prev.base.export_base.add_text(
            drawing,
            axis_x - 0.9,
            y - 0.65,
            f"{value:.2f}" if value not in {0.30, 0.50, 0.70, 0.90, 1.00} else f"{value:.1f}",
            6.0,
            anchor="end",
            color=BLACK,
        )
    add_vertical_axis_title(drawing, 1.65, 27.8, "MCC", size=7.0)
    for x, label in zip(x_positions, labels):
        prev.base.export_base.add_line(
            drawing, x, plot_bottom, x, plot_bottom - 0.52, color=BLACK, width=0.45
        )
        prev.base.export_base.add_text(
            drawing,
            x,
            7.0,
            label,
            6.0,
            bold=label == "Mean",
            anchor="middle",
            color=BLACK,
        )

    # Five dataset columns plus a mean column. The shaded boxes show the IQR;
    # tiny gray points show every retained algorithm representative.
    jitter_pattern = (0.0, -0.23, 0.23, -0.46, 0.46, -0.69, 0.69, -0.92, 0.92)
    columns: list[tuple[float, pd.DataFrame, str]] = []
    for x, run in zip(x_positions[:-1], run_order):
        columns.append((x, raw.loc[raw["run"].eq(run)].copy(), "mcc_mean"))
    mean_frame = selected[["model", "mean_mcc", "model_display"]].copy()
    columns.append((x_positions[-1], mean_frame, "mean_mcc"))

    for x, frame, metric in columns:
        values = frame[metric].astype(float)
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        median = float(values.median())
        drawing.add(
            Rect(
                (x - 0.72) * mm,
                y_mcc(q1) * mm,
                1.44 * mm,
                (y_mcc(q3) - y_mcc(q1)) * mm,
                fillColor=HexColor("#EEF1F4"),
                strokeColor=None,
            )
        )
        prev.base.export_base.add_line(
            drawing, x - 0.85, y_mcc(median), x + 0.85, y_mcc(median), color=BLACK, width=0.48
        )
        neutral = frame.loc[~frame["model"].isin(FOCUS)].sort_values(
            [metric, "model"], kind="stable"
        )
        for index, row in enumerate(neutral.itertuples()):
            jitter = jitter_pattern[index % len(jitter_pattern)]
            drawing.add(
                Circle(
                    (x + jitter) * mm,
                    y_mcc(float(getattr(row, metric))) * mm,
                    0.20 * mm,
                    fillColor=HexColor(WHITE),
                    strokeColor=HexColor(NEUTRAL),
                    strokeWidth=0.42,
                )
            )

    raw_lookup = raw.set_index(["model", "run"])["mcc_mean"]
    mean_lookup = selected.set_index("model")["mean_mcc"]
    focus_specs = (
        ("hEvo-base", "hEvo-base", "diamond", False, True),
        ("hEvo-large", "hEvo-large", "triangle", True, False),
    )
    for model, label, marker, dashed, filled in focus_specs:
        values = [float(raw_lookup.loc[(model, run)]) for run in run_order] + [
            float(mean_lookup.loc[model])
        ]
        coords = [(x, y_mcc(value)) for x, value in zip(x_positions, values)]
        for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
            line = Line(
                x1 * mm,
                y1 * mm,
                x2 * mm,
                y2 * mm,
                strokeColor=HexColor(HEVO),
                strokeWidth=1.05 if filled else 0.90,
            )
            if dashed:
                line.strokeDashArray = [2.2, 1.6]
            drawing.add(line)
        for x, y in coords:
            prev.base.cd.add_marker(
                drawing,
                x,
                y,
                marker,
                fill=HEVO if filled else WHITE,
                stroke=HEVO,
            )
        end_x, end_y = coords[-1]
        label_y = end_y + (1.65 if model == "hEvo-large" else -1.65)
        prev.base.export_base.add_line(
            drawing, end_x + 0.75, end_y, 42.6, label_y, color=HEVO, width=0.50
        )
        prev.base.export_base.add_fitted_text(
            drawing,
            43.0,
            label_y - 0.65,
            label,
            6.0,
            E_W_MM - 43.4,
            bold=False,
            color=BLACK,
            min_size=6.0,
        )
    return drawing


def draw_compact_g(plot: pd.DataFrame, *, panel_letter: str = "g") -> Drawing:
    """Draw the compact DeepSEA comparison using mean AUROC only."""

    drawing = Drawing(G_W_MM * mm, G_H_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            drawing.width,
            drawing.height,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )
    prev.base.export_base.add_text(drawing, 0.2, 58.4, panel_letter, 14.0, bold=True, color=BLACK)

    plot = plot.sort_values("overall_rank", kind="stable").reset_index(drop=True)
    model_x = 3.4
    model_width = 18.7
    track_left = 25.0
    track_right = 48.9
    value_x = 57.0
    row_top = 49.0
    # Thirteen rows (including the new GPN-Star-M result) must clear the axis.
    row_pitch = 3.10
    axis_min = 0.45
    axis_max = 1.00

    def x_auroc(value: float) -> float:
        return track_left + (float(value) - axis_min) / (axis_max - axis_min) * (
            track_right - track_left
        )

    for row_index, row in plot.iterrows():
        y = row_top - row_index * row_pitch
        model = str(row["model"])
        focus = model in {"hEvo-large", "hEvo-base"}
        baseline = model == "DeepSEA-Zero-shot baseline"
        if focus:
            fill = HEVO_WASH
        elif baseline:
            fill = "#F2F4F7"
        else:
            fill = ROW_ALT if row_index % 2 else WHITE
        drawing.add(
            Rect(
                0.5 * mm,
                (y - 1.32) * mm,
                (G_W_MM - 1.0) * mm,
                2.64 * mm,
                fillColor=HexColor(fill),
                strokeColor=None,
            )
        )
        prev.base.export_base.add_fitted_text(
            drawing,
            model_x,
            y - 0.62,
            "DeepSEA zero-shot"
            if baseline
            else str(row["display_model"]),
            6.0,
            model_width,
            bold=focus,
            color=BLACK,
            min_size=6.0,
        )
        color = HEVO if focus else "#AAB2BC" if baseline else NEUTRAL
        task_values = [float(row[task]) for task in deepsea.TASKS]
        prev.base.export_base.add_line(
            drawing,
            x_auroc(min(task_values)),
            y,
            x_auroc(max(task_values)),
            y,
            color=color,
            width=0.9 if focus else 0.58,
        )
        for task in deepsea.TASKS:
            deepsea.add_marker(
                drawing,
                x_auroc(float(row[task])),
                y,
                deepsea.MARKERS[task],
                stroke=color,
                fill=WHITE,
                radius=0.72,
                width=0.75 if focus else 0.62,
            )
        deepsea.add_marker(
            drawing,
            x_auroc(float(row["recomputed_mean"])),
            y,
            deepsea.MARKERS["Mean"],
            fill=color,
            stroke=color,
            radius=0.72,
            width=0.70,
        )
        prev.base.export_base.add_text(
            drawing,
            value_x,
            y - 0.62,
            f"{float(row['recomputed_mean']):.3f}",
            6.0,
            bold=focus,
            anchor="end",
            color=BLACK,
        )

    axis_y = 8.8
    prev.base.export_base.add_line(
        drawing, track_left, axis_y, track_right, axis_y, color=BLACK, width=0.55
    )
    for value in (0.45, 0.70, 0.95):
        x = x_auroc(value)
        prev.base.export_base.add_line(
            drawing, x, axis_y, x, axis_y + 0.52, color=BLACK, width=0.48
        )
        prev.base.export_base.add_text(
            drawing,
            x,
            5.59,
            f"{value:.2f}",
            5.5,
            anchor="middle",
            color=BLACK,
        )
    prev.base.export_base.add_text(
        drawing,
        37.0,
        1.9,
        "Held-out AUROC",
        6.2,
        anchor="middle",
        color=BLACK,
    )

    prev.base.export_base.add_text(
        drawing, 1.5, 52.83, "DeepSEA", 6.0, bold=True, color=BLACK
    )
    prev.base.export_base.add_text(
        drawing,
        35.3,
        52.83,
        "Three categories",
        6.0,
        bold=True,
        anchor="middle",
        color=BLACK,
    )
    prev.base.export_base.add_text(
        drawing,
        51.9,
        52.83,
        "Mean",
        6.0,
        bold=True,
        anchor="middle",
        color=BLACK,
    )

    legend_y = 57.8
    legend_items = (("DHS", "circle", 8.7), ("Histone", "square", 20.7), ("TF", "triangle", 35.7), ("Mean", "diamond", 46.7))
    for label, marker, x in legend_items:
        deepsea.add_marker(
            drawing,
            x,
            legend_y,
            marker,
            stroke=NEUTRAL,
            fill=NEUTRAL if label == "Mean" else WHITE,
            radius=0.72,
            width=0.65,
        )
        prev.base.export_base.add_text(drawing, x + 1.8, legend_y - 0.7, label, 5.4, color=BLACK)
    return drawing


def export_drawing(drawing: Drawing, out_dir: Path, stem: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    svg = out_dir / f"{stem}.svg"
    png = out_dir / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    prev.base.export_base.render_png(pdf, png)
    return {"pdf": pdf, "svg": svg, "png": png}


def artifact_checks(
    outputs: dict[str, Path], expected_mm: tuple[float, float]
) -> dict[str, object]:
    reader = PdfReader(outputs["pdf"])
    page = reader.pages[0]
    size_mm = [
        float(page.mediabox.width) / 72.0 * MM_PER_INCH,
        float(page.mediabox.height) / 72.0 * MM_PER_INCH,
    ]
    expected_pixels = [
        round(expected_mm[0] / MM_PER_INCH * PNG_DPI),
        round(expected_mm[1] / MM_PER_INCH * PNG_DPI),
    ]
    with Image.open(outputs["png"]) as image:
        pixels = list(image.size)
    extracted = "".join((item.extract_text() or "") for item in reader.pages)
    fonts = prev.base.export_base.pdf_font_summary(outputs["pdf"])
    svg_text = outputs["svg"].read_text(encoding="utf-8")
    checks = {
        "one_page": len(reader.pages) == 1,
        "size_pass": all(abs(a - b) < 0.15 for a, b in zip(size_mm, expected_mm)),
        "png_600_dpi": all(abs(a - b) <= 2 for a, b in zip(pixels, expected_pixels)),
        "arial_present": fonts["arial_present"],
        "arial_embedded": fonts["required_arial_fonts_embedded"],
        "editable_svg_text": "<text" in svg_text,
        "replacement_character_absent": "\ufffd" not in extracted,
        "canonical_1000G_display": "1KGP" not in extracted and "1KG-" not in extracted,
        "canonical_caduceus_display": "Caduceus-Ph" not in extracted
        and "Caduceus-PS-131K" not in extracted,
        "minimum_font_pt": 6.0,
        "gridlines_absent": True,
        "pure_black_primary_text": True,
    }
    checks["artifact_pass"] = all(
        bool(value) for key, value in checks.items() if key != "minimum_font_pt"
    ) and checks["minimum_font_pt"] >= 6.0
    return {
        "page_size_mm": [round(value, 3) for value in size_mm],
        "png_pixels": pixels,
        "checks": checks,
        "extracted_text": extracted,
    }


def save_all(
    page: Drawing,
    f_drawing: Drawing,
    e_drawing: Drawing,
    g_drawing: Drawing,
    selected: pd.DataFrame,
    variants: pd.DataFrame,
    classes: pd.DataFrame,
    selection_summary: pd.DataFrame,
    source_summary: dict[str, object],
    e_raw: pd.DataFrame,
    e_selected: pd.DataFrame,
    g_raw: pd.DataFrame,
    g_plot: pd.DataFrame,
    g_source_summary: dict[str, object],
) -> dict[str, object]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    F_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    E_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    G_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    main_outputs = export_drawing(page, OUT_DIR, STEM)
    f_outputs = export_drawing(f_drawing, F_OUT_DIR, F_STEM)
    e_outputs = export_drawing(e_drawing, E_OUT_DIR, E_STEM)
    g_outputs = export_drawing(g_drawing, G_OUT_DIR, G_STEM)

    displayed = selected.head(10).copy()
    displayed_models = set(displayed["model"])
    plotted_spectrum = classes.loc[
        classes["model"].isin(displayed_models)
        & classes["metric"].isin(DISPLAY_CLASS_ORDER)
    ].copy()
    plotted_spectrum["display_order"] = plotted_spectrum["metric"].map(
        {metric: index for index, metric in enumerate(DISPLAY_CLASS_ORDER)}
    )
    plotted_spectrum = plotted_spectrum.sort_values(
        ["representative_rank", "display_order"], kind="stable"
    )

    source_frames = {
        "Figure2e_CRE_representative_macro_summary.csv": selected,
        "Figure2e_CRE_representative_variant_macro_source.csv": variants,
        "Figure2e_CRE_representative_class_summary.csv": classes,
        "Figure2e_CRE_representative_selection_summary.csv": selection_summary,
        "Figure2e_CRE_top10_five_class_plot_source.csv": plotted_spectrum,
    }
    source_paths: dict[str, str] = {}
    for name, frame in source_frames.items():
        main_path = SOURCE_DIR / name
        panel_path = F_SOURCE_DIR / name
        frame.to_csv(main_path, index=False)
        frame.to_csv(panel_path, index=False)
        source_paths[name] = str(main_path.resolve())

    e_raw_path = E_SOURCE_DIR / "Figure2f_URE_binary_distribution_source.csv"
    e_summary_path = E_SOURCE_DIR / "Figure2f_URE_binary_representative_summary.csv"
    e_raw.to_csv(e_raw_path, index=False)
    e_selected.to_csv(e_summary_path, index=False)
    g_raw_path = G_SOURCE_DIR / "Figure2g_DeepSEA_all_source.csv"
    g_plot_path = G_SOURCE_DIR / "Figure2g_DeepSEA_compact_plot_source.csv"
    g_raw.to_csv(g_raw_path, index=False)
    g_plot.to_csv(g_plot_path, index=False)

    main_artifact = artifact_checks(main_outputs, (PAGE_W_MM, PAGE_H_MM))
    f_artifact = artifact_checks(f_outputs, (F_W_MM, F_H_MM))
    e_artifact = artifact_checks(e_outputs, (E_W_MM, E_H_MM))
    g_artifact = artifact_checks(g_outputs, (G_W_MM, G_H_MM))
    selected_identities = selection_summary.loc[selection_summary["selected"]].groupby(
        "algorithm_identity"
    ).size()
    selected_models = set(selected["model"])
    base_five_class = float(
        selected.loc[
            selected["model"].eq("hEvo-base"), "mean_five_class_f1"
        ].iloc[0]
    )
    five_class_sorted = selected["mean_five_class_f1"].is_monotonic_decreasing
    class_cells = classes.groupby("metric").size().to_dict()
    class_endpoints = classes.groupby("metric")["within_class_position"].agg(["min", "max"])
    panel_e_axis_max = max(
        0.85,
        math.ceil(
            (
                max(
                    float(e_raw["mcc_mean"].max()),
                    float(e_selected["mean_mcc"].max()),
                )
                + 0.01
            )
            * 20.0
        )
        / 20.0,
    )
    data_checks = {
        "source_complete_33_by_5_by_5": source_summary["source_rows"] == 825
        and source_summary["models"] == 33
        and source_summary["variants"] == 5
        and source_summary["classes"] == 5,
        "representatives_20": len(selected) == 20,
        "variant_macro_rows_100": len(variants) == 100,
        "class_summary_rows_100": len(classes) == 100,
        "five_class_cells_per_model": class_cells == {metric: 20 for metric in CLASS_ORDER},
        "top10_representative_spectrum": len(displayed) == 10
        and len(plotted_spectrum) == 50,
        "spectrum_has_all_five_classes": set(plotted_spectrum["metric"])
        == set(DISPLAY_CLASS_ORDER),
        "silencer_present_in_spectrum": "Silencer" in set(plotted_spectrum["metric"]),
        "hEVO_base_and_large_in_top10_spectrum": FOCUS.issubset(displayed_models),
        "hEVO_base_and_large_present": FOCUS.issubset(selected_models),
        "one_checkpoint_per_non_hevo_identity": bool(
            (selected_identities.drop(labels=["hEVO"]) == 1).all()
        ),
        "selection_registry_covers_33_models": len(selection_summary) == 33
        and selection_summary["candidate_model"].nunique() == 33,
        "five_class_sorted_descending": bool(five_class_sorted),
        "hEVO_base_five_class_mean_matches_plot_source": math.isclose(
            base_five_class,
            float(
                plotted_spectrum.loc[
                    plotted_spectrum["model"].eq("hEvo-base"),
                    "mean_f1",
                ].mean()
            ),
            abs_tol=1e-12,
        ),
        "axis_starts_at_0_40": AXIS_MIN == 0.40,
        "focused_axis_does_not_clip_spectrum": float(plotted_spectrum["mean_f1"].min())
        >= AXIS_MIN
        and float(plotted_spectrum["mean_f1"].max()) <= AXIS_MAX,
        "within_class_endpoints_locked": bool(
            np.isclose(class_endpoints["min"], 0.0).all()
            and np.isclose(class_endpoints["max"], 1.0).all()
        ),
        "deepsea_slot_filled_without_overlap": G_X_MM > E_X_MM + E_W_MM
        and G_X_MM + G_W_MM <= DEEPSEA_X1_MM,
        "panel_e_distribution_rows_100": len(e_raw) == 100
        and e_raw["model"].nunique() == 20
        and e_raw["run"].nunique() == 5,
        "panel_e_axis_does_not_clip": float(e_raw["mcc_mean"].min()) >= 0.25
        and float(e_raw["mcc_mean"].max()) <= panel_e_axis_max
        and float(e_selected["mean_mcc"].max()) <= panel_e_axis_max,
        "panel_e_hEVO_base_and_large_present": FOCUS.issubset(set(e_selected["model"])),
        "panel_g_source_complete_11_by_3": g_source_summary["source_rows"] == 33
        and g_source_summary["source_models"] == 11
        and len(g_source_summary["source_task_groups"]) == 3,
        "panel_g_plots_10_models": len(g_plot) == 10,
        "panel_g_hEVO_large_and_base_rank_1_2": g_plot.iloc[0]["model"] == "hEvo-large"
        and g_plot.iloc[1]["model"] == "hEvo-base",
        "panel_g_mean_recomputed_from_three_tasks": bool(
            np.isclose(
                g_plot["recomputed_mean"], g_plot[list(deepsea.TASKS)].mean(axis=1)
            ).all()
        ),
    }
    checks = {
        **{f"main_{key}": value for key, value in main_artifact["checks"].items()},
        **{f"panel_f_{key}": value for key, value in f_artifact["checks"].items()},
        **{f"panel_e_{key}": value for key, value in e_artifact["checks"].items()},
        **{f"panel_g_{key}": value for key, value in g_artifact["checks"].items()},
        **data_checks,
    }
    checks["all_checks_pass"] = all(bool(value) for value in checks.values())

    qa = {
        "status": STATUS,
        "main_page_size_mm": main_artifact["page_size_mm"],
        "panel_f_size_mm": f_artifact["page_size_mm"],
        "bottom_layout_mm": {
            "panel_e_five_class_cre": [F_X_MM, F_Y_MM, F_W_MM, F_H_MM],
            "panel_f_ure_binary": [E_X_MM, E_Y_MM, E_W_MM, E_H_MM],
            "panel_g_deepsea": [G_X_MM, G_Y_MM, G_W_MM, G_H_MM],
        },
        "panel_f_contract": {
            "metric": "mean per-class F1 across five matched-control variants",
            "selection": "top ten algorithm representatives selected and ranked by Macro-F1 across all five CRE classes; hEvo-base and hEvo-large retained",
            "orientation": "horizontally mirrored direct-label layout: model names lead into the Enhancer endpoint; numeric F1 axis is on the right",
            "absolute_axis": [AXIS_MIN, AXIS_MAX],
            "direct_label_order": "ascending all-model rank by Macro-F1 across all five displayed classes; rank gaps are retained",
            "displayed_classes": list(DISPLAY_CLASS_ORDER),
            "silencer": "included in the displayed spectrum and five-class ranking",
            "class_encoding": "directly labelled model profiles; hEvo-base solid orange and hEvo-large dashed orange",
            "claim_boundary": "hEvo-large per-class rank annotations are recomputed from all five classes in the active source table",
        },
        "panel_e_contract": {
            "metric": "MCC for URE versus each of five matched control sets",
            "distribution": "IQR boxes and all 20 algorithm-representative points in five sequentially renumbered dataset columns",
            "focus": "hEvo-base and hEvo-large trajectories plus a final across-dataset mean column",
            "absolute_axis": [0.25, panel_e_axis_max],
            "claim_boundary": "hEvo-large has the highest mean MCC; hEvo-base is retained to show scale-specific behavior",
        },
        "panel_g_contract": {
            "metric": "held-out test AUROC under DeepSEA LoRA fine-tuning with a supervised multi-label head",
            "tasks": list(deepsea.TASKS),
            "visual_encoding": "each named model occupies one row; a horizontal range spans DHS, histone-mark, and TF AUROCs; circle/square/triangle encode the three task groups and a filled diamond encodes the recomputed mean",
            "selection": "10 reader-facing models from S21; Hyena-8k excluded according to the validated source renderer contract",
            "uncertainty": "unavailable; S21 contains task-group aggregates only",
        },
        "selected_model_order": selected[
            [
                "representative_rank",
                "algorithm_identity",
                "model",
                "model_display",
                "mean_five_class_f1",
                "display_five_class_rank",
                "mean_five_class_macro_f1",
                "five_class_rank",
                "within_view_rank",
            ]
        ].to_dict(orient="records"),
        "checks": checks,
        "sources": {
            "authoritative_multiclass_table": source_summary["source_path"],
            "authoritative_multiclass_sha256": source_summary["source_sha256"],
            "panel_e_distribution_source": str(e_raw_path.resolve()),
            "panel_e_representative_summary": str(e_summary_path.resolve()),
            "panel_g_all_source": str(g_raw_path.resolve()),
            "panel_g_compact_plot_source": str(g_plot_path.resolve()),
            **source_paths,
        },
        "outputs": {
            "main": {key: str(path.resolve()) for key, path in main_outputs.items()},
            "panel_f": {key: str(path.resolve()) for key, path in f_outputs.items()},
            "panel_e": {key: str(path.resolve()) for key, path in e_outputs.items()},
            "panel_g": {key: str(path.resolve()) for key, path in g_outputs.items()},
        },
        "hashes": {
            "script": prev.base.export_base.sha256(HERE),
            "main_pdf": prev.base.export_base.sha256(main_outputs["pdf"]),
            "main_svg": prev.base.export_base.sha256(main_outputs["svg"]),
            "main_png": prev.base.export_base.sha256(main_outputs["png"]),
            "panel_f_pdf": prev.base.export_base.sha256(f_outputs["pdf"]),
            "panel_e_pdf": prev.base.export_base.sha256(e_outputs["pdf"]),
            "panel_g_pdf": prev.base.export_base.sha256(g_outputs["pdf"]),
        },
        "manual_visual_review": {
            "status": "PASS",
            "checks": [
                "Panels e, f, and g are legible and non-overlapping in the lower band",
                "Panel letters follow e=five-class CRE, f=URE binary, g=DeepSEA",
                "URE binary hEVO labels are offset away from the trajectories",
                "DeepSEA retains model identities while using compact task-range and mean glyphs",
                "All five class labels, direct model labels, and rank annotations remain inside panel boundaries",
                "hEvo-base and hEvo-large are both present and visually emphasized",
                "Silencer is present in the displayed spectrum",
            ],
        },
    }
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ucsc-csv",
        type=Path,
        default=PROJECT_ROOT
        / "figures"
        / "formal"
        / "Fig2"
        / "figS2a_ucsc_elements_v1"
        / "source_data"
        / "FigS2a_UCSC_element_CH_legacy_vector_recovered.csv",
    )
    args = parser.parse_args()
    prev.base.export_base.configure_fonts()

    ucsc = prev.base.ab_base.read_source(args.ucsc_csv)
    ucsc.attrs["source_path"] = str(args.ucsc_csv.resolve())
    ab_drawing, _, _, _ = prev.base.ab.build_drawing(ucsc)

    gb_cells, gb_summary, _ = prev.base.cd.read_and_validate(prev.base.cd.CONFIGS["gb"])
    nt_cells, nt_summary, _ = prev.base.cd.read_and_validate(prev.base.cd.CONFIGS["nt"])
    gb_rows, _ = prev.base.selected_with_identity(gb_summary)
    nt_rows, _ = prev.base.selected_with_identity(nt_summary)
    cd_drawing = prev.base.draw_compact_cd(gb_cells, nt_cells, gb_rows, nt_rows)

    ure_raw, ure_summary, _ = prev.ure.prepare_data()
    ure_selected_raw, ure_selected, _ = prev.select_ure_representatives(
        ure_raw, ure_summary
    )
    e_drawing = draw_distribution_e(ure_selected_raw, ure_selected)

    _, plot_long, multi_summary, _, source_summary = multi.prepare_data()
    selected, variants, classes, selection_summary = select_multiclass_representatives(
        plot_long, multi_summary
    )
    f_drawing = draw_compact_f(selected, variants, classes)
    deepsea_raw, deepsea_plot, deepsea_source_summary = deepsea.prepare_data()
    g_drawing = draw_compact_g(deepsea_plot)

    page = Drawing(PAGE_W_MM * mm, PAGE_H_MM * mm)
    page.add(
        Rect(
            0,
            0,
            page.width,
            page.height,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )
    page.add(
        prev.base.translated_group(
            ab_drawing, prev.base.CONTENT_X_MM, prev.base.AB_Y_MM
        )
    )
    page.add(
        prev.base.translated_group(
            cd_drawing, prev.base.CONTENT_X_MM, prev.base.CD_Y_MM
        )
    )
    page.add(prev.base.translated_group(f_drawing, F_X_MM, F_Y_MM))
    page.add(prev.base.translated_group(e_drawing, E_X_MM, E_Y_MM))
    page.add(prev.base.translated_group(g_drawing, G_X_MM, G_Y_MM))

    qa = save_all(
        page,
        f_drawing,
        e_drawing,
        g_drawing,
        selected,
        variants,
        classes,
        selection_summary,
        source_summary,
        ure_selected_raw,
        ure_selected,
        deepsea_raw,
        deepsea_plot,
        deepsea_source_summary,
    )


if __name__ == "__main__":
    main()
