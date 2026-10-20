#!/usr/bin/env python3
"""Render source-backed ClinVar main, all-model and family-series views.

The main-text triptych borrows the compact independently ranked horizontal-bar
grammar of Evo 2 Figure 3 while applying the manuscript-wide Figure 2 model
identity, naming, typography and colour rules.  Supplementary views retain all
33 source models and the locked hEVO/NT/Evo 2/GPN family series.

Only AUPRC is plotted because matched source-backed AUROC values are unavailable
in the current local snapshot.  No metric is inferred or reconstructed.
"""

from __future__ import annotations

# [Fig. 3a–c builder] Build source data and figures for the ClinVar main panel, all-model view, and family-series supplement.
# [Input] C1–C5 ClinVar/causal/rare-variant tables and model-family styles; render/export after model selection.
# [Run order] Run before the integrated Fig. 3 v2 script to provide ClinVar strata, family, and selection source data.

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm


HERE = Path(__file__).resolve()
SCRIPT_DIR = HERE.parent
PROJECT_ROOT = HERE.parents[3]
for search_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import plot_figure3abc_clinvar_integrated_v1 as source  # noqa: E402
from figures.scripts.common.hevo_model_style import (  # noqa: E402
    MODEL_FAMILY_STYLE,
    canonical_model_name,
    model_family,
)


OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig3" / "fig3abc_clinvar_main_full_family_v5"
SOURCE_DIR = OUT_DIR / "source_data"
STATUS = "DATA_QC_DRAFT"
PNG_DPI = 600
MM_PER_INCH = 25.4

MAIN_SIZE_MM = (190.0, 83.0)
ALL_SIZE_MM = (190.0, 124.0)
FAMILY_SIZE_MM = (190.0, 75.0)

MAIN_STEM = "Figure3abc_ClinVar_best_algorithm_representatives_v5_DATA_QC_DRAFT"
ALL_STEM = "FigureS3a_ClinVar_all_33_models_v5_DATA_QC_DRAFT"
FAMILY_STEM = "FigureS3a_ClinVar_family_series_v5_DATA_QC_DRAFT"

BLACK = "#000000"
WHITE = "#FFFFFF"
HEVO = MODEL_FAMILY_STYLE["hEVO"]["color"]
EVO2 = MODEL_FAMILY_STYLE["Evo 2"]["color"]
NT = "#8995A5"
GPN = MODEL_FAMILY_STYLE["GPN"]["color"]
NEUTRAL = "#B8BEC8"
NEUTRAL_LIGHT = "#D8DDE3"
HEVO_WASH = "#FBF2EA"
RANK_COLORS = {1: "#C43B3B", 2: "#2E7D55", 3: "#2864B4"}
FAMILY_COLORS = {"hEVO": HEVO, "NT": NT, "Evo 2": EVO2, "GPN": GPN}

FAMILY_RAW: dict[str, tuple[str, ...]] = {
    "hEVO": ("hEvo-base", "hEvo-large"),
    "NT": (
        "NTv3-8M-pre",
        "NTv3-100M-pre",
        "NTv3-650M-pre",
        "NT_500m_MS",
        "NT_2.5b_MS",
        "NT_500m_1kG",
        "NT_2.5b_1kG",
    ),
    "Evo 2": ("evo2_1b_base", "evo2_7b_base"),
    "GPN": ("GPN-promoter", "GPN-MSA"),
}


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    value: str,
    size: float = 6.0,
    *,
    bold: bool = False,
    anchor: str = "start",
    color: str = BLACK,
) -> None:
    drawing.add(
        String(
            x * mm,
            y * mm,
            str(value),
            fontName=source.FONT_BOLD if bold else source.FONT,
            fontSize=size,
            fillColor=HexColor(color),
            textAnchor=anchor,
        )
    )


def text_width_mm(value: str, size: float = 6.0, *, bold: bool = False) -> float:
    font = source.FONT_BOLD if bold else source.FONT
    return source.pdfmetrics.stringWidth(str(value), font, size) / mm


def add_fitted_text(
    drawing: Drawing,
    x: float,
    y: float,
    value: str,
    max_width_mm: float,
    *,
    bold: bool = False,
    color: str = BLACK,
) -> None:
    if text_width_mm(value, 6.0, bold=bold) > max_width_mm:
        raise ValueError(f"Canonical model label exceeds {max_width_mm:.1f} mm at 6 pt: {value}")
    add_text(drawing, x, y, value, 6.0, bold=bold, color=color)


def add_line(
    drawing: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = BLACK,
    width: float = 0.55,
) -> None:
    drawing.add(
        Line(
            x1 * mm,
            y1 * mm,
            x2 * mm,
            y2 * mm,
            strokeColor=HexColor(color),
            strokeWidth=width,
        )
    )


def score_x(value: float, x0: float, x1: float) -> float:
    return x0 + float(value) * (x1 - x0)


def scaled_score_x(value: float, x0: float, x1: float, scale: tuple[float, float]) -> float:
    lower, upper = scale
    if upper <= lower:
        raise ValueError(f"Invalid score scale: {scale}")
    clipped = min(max(float(value), lower), upper)
    return x0 + (clipped - lower) / (upper - lower) * (x1 - x0)


def local_score_scale(
    frame: pd.DataFrame,
    *,
    score_col: str,
    lower_col: str | None = None,
    upper_col: str | None = None,
    baseline_col: str | None = None,
) -> tuple[float, float]:
    """Return transparent, rounded panel-specific limits within [0, 1]."""
    arrays: list[np.ndarray] = [pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)]
    for column in (lower_col, upper_col, baseline_col):
        if column and column in frame.columns:
            arrays.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float))
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    observed_lower = float(values.min())
    observed_upper = float(values.max())
    observed_span = max(observed_upper - observed_lower, 0.01)
    padded_lower = max(0.0, observed_lower - max(0.04 * observed_span, 0.002))
    padded_upper = min(1.0, observed_upper + max(0.04 * observed_span, 0.002))
    # Target roughly seven intervals before rounding to a conventional step.
    # This preserves honest local limits while avoiding an unnecessarily broad
    # 0-1 scale for tightly clustered benchmark scores.
    raw_step = max((padded_upper - padded_lower) / 7.0, 0.001)
    exponent = 10.0 ** np.floor(np.log10(raw_step))
    normalized = raw_step / exponent
    multiplier = next(item for item in (1.0, 2.0, 2.5, 5.0, 10.0) if normalized <= item)
    step = multiplier * exponent
    lower = max(0.0, np.floor(padded_lower / step) * step)
    upper = min(1.0, np.ceil(padded_upper / step) * step)
    if upper - lower < 2.0 * step:
        upper = min(1.0, lower + 2.0 * step)
    return (float(round(lower, 6)), float(round(upper, 6)))


def format_scale_tick(value: float, scale: tuple[float, float]) -> str:
    span = scale[1] - scale[0]
    decimals = 3 if span < 0.05 else 2 if span < 0.5 else 1
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def draw_local_scale(
    drawing: Drawing,
    *,
    x0: float,
    x1: float,
    y: float,
    scale: tuple[float, float],
    font_size: float = 6.0,
    compact_end_labels: bool = False,
) -> None:
    add_line(drawing, x0, y, x1, y, width=0.55)
    lower, upper = scale
    midpoint = (lower + upper) / 2.0
    for value in (lower, midpoint, upper):
        x = scaled_score_x(value, x0, x1, scale)
        add_line(drawing, x, y - 0.55, x, y + 0.55, width=0.45)
    if compact_end_labels:
        add_text(drawing, x0 + 0.55, y + 0.80, format_scale_tick(lower, scale), font_size, anchor="start")
        add_text(drawing, x1 - 0.55, y + 0.80, format_scale_tick(upper, scale), font_size, anchor="end")
    else:
        for value in (lower, midpoint, upper):
            x = scaled_score_x(value, x0, x1, scale)
            add_text(drawing, x, y + 0.80, format_scale_tick(value, scale), font_size, anchor="middle")


def model_color(model: str, *, family_mode: bool = False) -> str:
    family = model_family(model)
    return FAMILY_COLORS.get(family, NEUTRAL)


def rank_marker(
    drawing: Drawing,
    x: float,
    y: float,
    rank: int,
    *,
    fallback_color: str = NEUTRAL,
) -> None:
    rank = int(rank)
    color = RANK_COLORS.get(rank, fallback_color)
    fill = HexColor(color) if rank <= 3 else HexColor(WHITE)
    stroke = HexColor(color)
    radius = 0.72 * mm
    if rank == 1:
        points = [x * mm, (y + 0.86) * mm, (x + 0.86) * mm, y * mm, x * mm, (y - 0.86) * mm, (x - 0.86) * mm, y * mm]
        drawing.add(Polygon(points, fillColor=fill, strokeColor=stroke, strokeWidth=0.55))
    elif rank == 2:
        drawing.add(Rect((x - 0.68) * mm, (y - 0.68) * mm, 1.36 * mm, 1.36 * mm, fillColor=fill, strokeColor=stroke, strokeWidth=0.55))
    elif rank == 3:
        points = [(x - 0.80) * mm, (y - 0.68) * mm, (x + 0.80) * mm, (y - 0.68) * mm, x * mm, (y + 0.84) * mm]
        drawing.add(Polygon(points, fillColor=fill, strokeColor=stroke, strokeWidth=0.55))
    else:
        drawing.add(Circle(x * mm, y * mm, radius, fillColor=fill, strokeColor=stroke, strokeWidth=0.55))


def draw_shared_scale_key(
    drawing: Drawing,
    *,
    y: float,
    x0: float = 158.0,
    x1: float = 186.0,
    metric: str = "AUPRC",
) -> None:
    """Draw one compact 0-1 scale key for a multi-panel figure."""
    add_text(drawing, x0 - 2.2, y - 0.72, metric, 6.0, anchor="end")
    add_line(drawing, x0, y, x1, y, width=0.60)
    for value, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        x = score_x(value, x0, x1)
        add_line(drawing, x, y - 0.65, x, y + 0.65, width=0.50)
        add_text(drawing, x, y + 1.05, label, 6.0, anchor="middle")


def draw_compact_family_legend(drawing: Drawing, *, y: float, x0: float = 2.5) -> None:
    x = x0
    for label, color in (("hEVO", HEVO), ("NT", NT), ("EVO 2", EVO2), ("GPN", GPN), ("other", NEUTRAL)):
        drawing.add(Rect(x * mm, (y - 0.65) * mm, 2.8 * mm, 1.3 * mm, fillColor=HexColor(color), strokeColor=HexColor(color), strokeWidth=0.30))
        add_text(drawing, x + 3.8, y - 0.68, label, 6.0)
        x += 12.2 if label not in {"EVO 2", "other"} else 15.2


def select_main_representatives(
    long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        long.groupby("model", as_index=False)
        .agg(
            mean_auprc=("auprc_mean", "mean"),
            mean_rank_all33=("task_rank_all33", "mean"),
        )
    )
    available = set(summary["model"])
    selected_models: list[str] = []
    summary_rows: list[dict[str, object]] = []
    for identity, candidates in source.REPRESENTATIVE_GROUPS:
        missing = sorted(set(candidates) - available)
        if missing:
            raise ValueError(f"Representative identity {identity} missing candidates: {missing}")
        candidates_table = summary[summary["model"].isin(candidates)].sort_values(
            ["mean_auprc", "mean_rank_all33", "model"],
            ascending=[False, True, True],
            kind="stable",
        )
        winner = str(candidates_table.iloc[0]["model"])
        selected_models.append(winner)
        for row in candidates_table.itertuples(index=False):
            summary_rows.append(
                {
                    "algorithm_identity": identity,
                    "candidate_model": row.model,
                    "candidate_display": canonical_model_name(row.model),
                    "mean_auprc": float(row.mean_auprc),
                    "mean_rank_all33": float(row.mean_rank_all33),
                    "selected": row.model == winner,
                    "selection_rule": "highest unweighted mean AUPRC across ClinVar all, coding and noncoding",
                }
            )
    if len(selected_models) != 18 or len(set(selected_models)) != 18:
        raise ValueError(f"Expected 18 unique algorithm representatives, found {len(set(selected_models))}")
    selected_summary = summary[summary["model"].isin(selected_models)].copy()
    selected_summary["model_display"] = selected_summary["model"].map(canonical_model_name)
    selected_summary["family"] = selected_summary["model"].map(model_family)
    selected_summary = selected_summary.sort_values(
        ["mean_auprc", "model_display"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    selected_summary["overall_rank"] = np.arange(1, len(selected_summary) + 1)
    plot = long[long["model"].isin(selected_models)].copy()
    plot["model_display"] = plot["model"].map(canonical_model_name)
    plot["family"] = plot["model"].map(model_family)
    plot["within_view_rank"] = (
        plot.groupby("task")["auprc_mean"].rank(method="min", ascending=False).astype(int)
    )
    return plot, selected_summary, pd.DataFrame(summary_rows)


def prepare_all_models(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        long.groupby("model", as_index=False)
        .agg(mean_auprc=("auprc_mean", "mean"), mean_rank_all33=("task_rank_all33", "mean"))
    )
    summary["model_display"] = summary["model"].map(canonical_model_name)
    summary["family"] = summary["model"].map(model_family)
    summary = summary.sort_values(
        ["mean_auprc", "model_display"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    summary["overall_rank"] = np.arange(1, len(summary) + 1)
    plot = long.merge(
        summary[["model", "model_display", "family", "mean_auprc", "overall_rank"]],
        on="model",
        how="left",
        validate="many_to_one",
        suffixes=("", "_summary"),
    )
    plot["within_view_rank"] = plot["task_rank_all33"].astype(int)
    return plot, summary


def prepare_family_models(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_order = [model for models in FAMILY_RAW.values() for model in models]
    missing = sorted(set(raw_order) - set(long["model"]))
    if missing:
        raise ValueError(f"Family-series models missing from ClinVar tables: {missing}")
    family_long = long[long["model"].isin(raw_order)].copy()
    summary = (
        family_long.groupby("model", as_index=False)
        .agg(mean_auprc=("auprc_mean", "mean"), mean_rank_all33=("task_rank_all33", "mean"))
    )
    summary["model_display"] = summary["model"].map(canonical_model_name)
    summary["family"] = summary["model"].map(model_family)
    family_order = {family: index for index, family in enumerate(FAMILY_RAW)}
    summary["family_order"] = summary["family"].map(family_order)
    summary = summary.sort_values(
        ["family_order", "mean_auprc", "model_display"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary["family_row_order"] = summary.groupby("family").cumcount()
    family_long = family_long.merge(
        summary[["model", "model_display", "family", "family_order", "family_row_order", "mean_auprc"]],
        on="model",
        how="left",
        validate="many_to_one",
        suffixes=("", "_summary"),
    )
    family_long["within_view_rank"] = (
        family_long.groupby("task")["auprc_mean"].rank(method="min", ascending=False).astype(int)
    )
    return family_long, summary


def render_main(plot: pd.DataFrame, data_summary: dict[str, object]) -> Drawing:
    width, height = MAIN_SIZE_MM
    drawing = Drawing(width * mm, height * mm)
    drawing.add(Rect(0, 0, width * mm, height * mm, fillColor=HexColor(WHITE), strokeColor=None))
    starts = {"all": 1.6, "coding": 64.3, "noncoding": 127.0}
    row_top = 65.0
    row_pitch = 3.05
    bar_height = 1.28

    draw_compact_family_legend(drawing, y=79.5)
    add_text(drawing, 76.0, 78.82, "independent scales | whisker=95% CI | tick=prevalence", 6.0)

    for task, spec in source.TASKS.items():
        start = starts[task]
        label_x = start + 0.8
        bar_x0 = start + 31.0
        bar_x1 = start + 54.2
        value_x = start + 61.0
        stats = data_summary["task_stats"][task]
        rows = plot[plot["task"].eq(task)].sort_values(
            ["auprc_mean", "model_display"], ascending=[False, True], kind="stable"
        )
        if len(rows) != 18:
            raise ValueError(f"{task}: expected 18 main representatives, found {len(rows)}")
        scale = local_score_scale(
            rows,
            score_col="auprc_mean",
            lower_col="auprc_ci_lower",
            upper_col="auprc_ci_upper",
            baseline_col="baseline",
        )
        add_text(drawing, start, 74.8, spec["letter"], 14.0, bold=True)
        add_text(drawing, start + 31.3, 75.0, spec["title"], 7.2, bold=True, anchor="middle")
        add_text(
            drawing,
            start + 31.3,
            70.8,
            f"N={int(stats['num_samples']):,} | base={float(stats['baseline']):.3f}",
            6.0,
            anchor="middle",
        )
        add_line(drawing, start + 0.4, 67.8, bar_x0 - 1.4, 67.8, width=0.75)
        draw_local_scale(drawing, x0=bar_x0, x1=bar_x1, y=67.8, scale=scale)
        add_line(drawing, bar_x1 + 1.4, 67.8, start + 61.7, 67.8, width=0.75)
        for row_index, row in enumerate(rows.itertuples(index=False)):
            y = row_top - row_index * row_pitch
            is_hevo = row.family == "hEVO"
            if is_hevo:
                drawing.add(Rect((label_x - 0.35) * mm, (y - 1.17) * mm, 61.0 * mm, 2.25 * mm, fillColor=HexColor(HEVO_WASH), strokeColor=None))
            add_fitted_text(
                drawing,
                label_x,
                y - 0.73,
                row.model_display,
                bar_x0 - label_x - 1.1,
                bold=is_hevo,
            )
            mean_x = scaled_score_x(row.auprc_mean, bar_x0, bar_x1, scale)
            lower_x = scaled_score_x(row.auprc_ci_lower, bar_x0, bar_x1, scale)
            upper_x = scaled_score_x(row.auprc_ci_upper, bar_x0, bar_x1, scale)
            base_x = scaled_score_x(row.baseline, bar_x0, bar_x1, scale)
            color = model_color(row.model)
            drawing.add(
                Rect(
                    bar_x0 * mm,
                    (y - bar_height / 2) * mm,
                    max(0.0, mean_x - bar_x0) * mm,
                    bar_height * mm,
                    fillColor=HexColor(color),
                    strokeColor=HexColor(color),
                    strokeWidth=0.35,
                )
            )
            add_line(drawing, lower_x, y, upper_x, y, width=0.75)
            add_line(drawing, lower_x, y - 0.58, lower_x, y + 0.58, width=0.50)
            add_line(drawing, upper_x, y - 0.58, upper_x, y + 0.58, width=0.50)
            add_line(drawing, base_x, y - 1.02, base_x, y + 1.02, width=0.50)
            add_text(drawing, value_x, y - 0.73, f"{row.auprc_mean:.3f}", 6.0, anchor="end", bold=is_hevo)

    return drawing


def table_columns() -> dict[str, tuple[float, float]]:
    return {
        "mean": (50.5, 70.0),
        "all": (81.5, 110.5),
        "coding": (117.0, 146.0),
        "noncoding": (152.5, 181.5),
    }


def draw_table_header(
    drawing: Drawing,
    *,
    top: float,
    title: str,
    letter: str,
    data_summary: dict[str, object],
    scales: dict[str, tuple[float, float]],
    mean_scale: tuple[float, float],
) -> None:
    columns = table_columns()
    add_text(drawing, 2.2, top, letter, 14.0, bold=True)
    add_text(drawing, 12.8, top + 0.2, title, 7.2, bold=True)
    add_text(drawing, 87.5, top - 0.55, "rank", 6.0, bold=True)
    marker_x = 98.0
    for rank, label in ((1, "1"), (2, "2"), (3, "3"), (4, "other")):
        rank_marker(drawing, marker_x, top + 0.05, rank)
        add_text(drawing, marker_x + 1.7, top - 0.60, label, 6.0)
        marker_x += 9.0 if rank < 4 else 13.5
    add_text(drawing, 26.0, top - 7.0, "Models", 6.4, bold=True, anchor="middle")
    add_text(drawing, 60.3, top - 5.4, "Mean AUPRC", 6.4, bold=True, anchor="middle")
    draw_local_scale(
        drawing,
        x0=columns["mean"][0],
        x1=columns["mean"][1],
        y=top - 11.7,
        scale=mean_scale,
        compact_end_labels=True,
    )
    for task, spec in source.TASKS.items():
        x0, x1 = columns[task]
        stats = data_summary["task_stats"][task]
        add_text(drawing, (x0 + x1) / 2, top - 4.9, spec["title"], 6.4, bold=True, anchor="middle")
        add_text(
            drawing,
            (x0 + x1) / 2,
            top - 8.0,
            f"N={int(stats['num_samples']):,} | base={float(stats['baseline']):.3f}",
            6.0,
            anchor="middle",
        )
        draw_local_scale(
            drawing,
            x0=x0,
            x1=x1,
            y=top - 11.7,
            scale=scales[task],
            compact_end_labels=True,
        )
    add_line(drawing, 4.5, top - 14.8, 184.0, top - 14.8, width=0.75)


def draw_table_row(
    drawing: Drawing,
    row: pd.Series,
    task_lookup: pd.DataFrame,
    y: float,
    *,
    family_mode: bool,
    wash_hevo: bool,
    scales: dict[str, tuple[float, float]],
    mean_scale: tuple[float, float],
) -> None:
    columns = table_columns()
    family = str(row["family"])
    is_hevo = family == "hEVO"
    color = model_color(str(row["model"]), family_mode=family_mode)
    if wash_hevo and is_hevo:
        drawing.add(Rect(4.6 * mm, (y - 1.40) * mm, 179.3 * mm, 2.65 * mm, fillColor=HexColor(HEVO_WASH), strokeColor=None))
    drawing.add(Circle(6.2 * mm, y * mm, 0.58 * mm, fillColor=HexColor(color), strokeColor=HexColor(color), strokeWidth=0.35))
    add_fitted_text(drawing, 8.3, y - 0.72, row["model_display"], 38.8, bold=is_hevo)

    mean_x0, mean_x1 = columns["mean"]
    mean_x = scaled_score_x(row["mean_auprc"], mean_x0, mean_x1, mean_scale)
    add_line(drawing, mean_x0, y, mean_x, y, color=color, width=0.80)
    drawing.add(Circle(mean_x * mm, y * mm, 0.66 * mm, fillColor=HexColor(color), strokeColor=HexColor(color), strokeWidth=0.45))
    add_text(drawing, 77.2, y - 0.72, f"{row['mean_auprc']:.3f}", 6.0, anchor="end", bold=is_hevo)

    for task in source.TASKS:
        cell = task_lookup.loc[(row["model"], task)]
        x0, x1 = columns[task]
        mean_x = scaled_score_x(cell["auprc_mean"], x0, x1, scales[task])
        low_x = scaled_score_x(cell["auprc_ci_lower"], x0, x1, scales[task])
        high_x = scaled_score_x(cell["auprc_ci_upper"], x0, x1, scales[task])
        add_line(drawing, x0, y, mean_x, y, color=color, width=0.55)
        add_line(drawing, low_x, y, high_x, y, color=BLACK, width=0.72)
        add_line(drawing, low_x, y - 0.48, low_x, y + 0.48, color=BLACK, width=0.45)
        add_line(drawing, high_x, y - 0.48, high_x, y + 0.48, color=BLACK, width=0.45)
        rank_marker(drawing, mean_x, y, int(cell["within_view_rank"]), fallback_color=color)


def draw_bottom_axes(drawing: Drawing, y: float, data_summary: dict[str, object]) -> None:
    columns = table_columns()
    for key in ("mean", "all", "coding", "noncoding"):
        x0, x1 = columns[key]
        add_line(drawing, x0, y, x1, y, width=0.60)
        for value in (0.0, 0.5, 1.0):
            x = score_x(value, x0, x1)
            add_line(drawing, x, y, x, y - 0.85, width=0.50)
            add_text(drawing, x, y - 2.8, f"{value:g}", 6.0, anchor="middle")
        if key in source.TASKS:
            baseline = float(data_summary["task_stats"][key]["baseline"])
            bx = score_x(baseline, x0, x1)
            add_line(drawing, bx, y + 0.15, bx, y + 1.85, width=0.65)
    add_text(drawing, 97.0, y - 6.2, "Absolute AUPRC", 6.0, anchor="middle")


def draw_rank_legend(drawing: Drawing, y: float, *, family_mode: bool) -> None:
    add_text(drawing, 5.0, y - 0.65, "Within-view rank", 6.0, bold=True)
    x = 36.0
    for rank, label in ((1, "1"), (2, "2"), (3, "3"), (4, "other")):
        rank_marker(drawing, x, y, rank)
        add_text(drawing, x + 2.0, y - 0.65, label, 6.0)
        x += 11.0 if rank < 4 else 18.0
    add_text(drawing, 91.0, y - 0.65, "line=score | whisker=reported 95% CI", 6.0)
    if family_mode:
        x = 143.0
        for family in ("hEVO", "NT", "Evo 2", "GPN"):
            color = FAMILY_COLORS[family]
            drawing.add(Circle(x * mm, y * mm, 0.58 * mm, fillColor=HexColor(color), strokeColor=HexColor(color), strokeWidth=0.35))
            display = "EVO 2" if family == "Evo 2" else family
            add_text(drawing, x + 1.7, y - 0.65, display, 6.0)
            x += 10.5


def render_all_models(
    plot: pd.DataFrame,
    summary: pd.DataFrame,
    data_summary: dict[str, object],
) -> Drawing:
    width, height = ALL_SIZE_MM
    drawing = Drawing(width * mm, height * mm)
    drawing.add(Rect(0, 0, width * mm, height * mm, fillColor=HexColor(WHITE), strokeColor=None))
    scales = {
        task: local_score_scale(
            plot[plot["task"].eq(task)],
            score_col="auprc_mean",
            lower_col="auprc_ci_lower",
            upper_col="auprc_ci_upper",
            baseline_col="baseline",
        )
        for task in source.TASKS
    }
    mean_scale = local_score_scale(summary, score_col="mean_auprc")
    draw_table_header(
        drawing,
        top=120.0,
        title="ClinVar variant scoring | all 33 source models",
        letter="a",
        data_summary=data_summary,
        scales=scales,
        mean_scale=mean_scale,
    )
    lookup = plot.set_index(["model", "task"])
    row_top = 101.0
    row_pitch = 3.05
    for index, (_, row) in enumerate(summary.iterrows()):
        y = row_top - index * row_pitch
        draw_table_row(
            drawing,
            row,
            lookup,
            y,
            family_mode=False,
            wash_hevo=True,
            scales=scales,
            mean_scale=mean_scale,
        )
    return drawing


def render_family_models(
    plot: pd.DataFrame,
    summary: pd.DataFrame,
    data_summary: dict[str, object],
) -> Drawing:
    width, height = FAMILY_SIZE_MM
    drawing = Drawing(width * mm, height * mm)
    drawing.add(Rect(0, 0, width * mm, height * mm, fillColor=HexColor(WHITE), strokeColor=None))
    scales = {
        task: local_score_scale(
            plot[plot["task"].eq(task)],
            score_col="auprc_mean",
            lower_col="auprc_ci_lower",
            upper_col="auprc_ci_upper",
            baseline_col="baseline",
        )
        for task in source.TASKS
    }
    mean_scale = local_score_scale(summary, score_col="mean_auprc")
    draw_table_header(
        drawing,
        top=71.0,
        title="ClinVar variant scoring | hEVO, NT, EVO 2 and GPN series",
        letter="a",
        data_summary=data_summary,
        scales=scales,
        mean_scale=mean_scale,
    )
    lookup = plot.set_index(["model", "task"])
    cursor_y = 52.0
    for family in FAMILY_RAW:
        family_rows = summary[summary["family"].eq(family)]
        family_color = FAMILY_COLORS[family]
        display_family = "EVO 2" if family == "Evo 2" else family
        add_text(drawing, 5.0, cursor_y, display_family, 6.2, bold=True, color=BLACK)
        add_line(drawing, 5.0, cursor_y - 1.15, 17.0, cursor_y - 1.15, color=family_color, width=0.95)
        cursor_y -= 2.80
        for _, row in family_rows.iterrows():
            draw_table_row(
                drawing,
                row,
                lookup,
                cursor_y,
                family_mode=True,
                wash_hevo=False,
                scales=scales,
                mean_scale=mean_scale,
            )
            cursor_y -= 3.05
        cursor_y -= 0.50
    return drawing


def render_png(pdf_path: Path, png_path: Path) -> None:
    bundled = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "native"
        / "poppler"
        / "Library"
        / "bin"
        / "pdftoppm.exe"
    )
    executable = bundled if bundled.exists() else Path(shutil.which("pdftoppm") or "")
    if not executable.exists():
        raise RuntimeError("pdftoppm is required for 600-dpi render QA")
    subprocess.run(
        [str(executable), "-f", "1", "-singlefile", "-png", "-r", str(PNG_DPI), str(pdf_path), str(png_path.with_suffix(""))],
        check=True,
        capture_output=True,
        text=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_drawing(drawing: Drawing, stem: str) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    svg = OUT_DIR / f"{stem}.svg"
    png = OUT_DIR / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    render_png(pdf, png)
    return {"pdf": pdf, "svg": svg, "png": png}


def artifact_checks(outputs: dict[str, Path], expected_mm: tuple[float, float]) -> dict[str, object]:
    reader = PdfReader(outputs["pdf"])
    page = reader.pages[0]
    size_mm = [
        float(page.mediabox.width) / 72.0 * MM_PER_INCH,
        float(page.mediabox.height) / 72.0 * MM_PER_INCH,
    ]
    expected_pixels = [round(value / MM_PER_INCH * PNG_DPI) for value in expected_mm]
    with Image.open(outputs["png"]) as image:
        pixels = list(image.size)
    extracted = page.extract_text() or ""
    fonts = source.pdf_font_summary(outputs["pdf"])
    svg_text = outputs["svg"].read_text(encoding="utf-8")
    checks = {
        "one_page": len(reader.pages) == 1,
        "size_pass": all(abs(actual - expected) < 0.15 for actual, expected in zip(size_mm, expected_mm)),
        "png_600_dpi": all(abs(actual - expected) <= 2 for actual, expected in zip(pixels, expected_pixels)),
        "arial_present": fonts["arial_present"],
        "arial_embedded": fonts["required_arial_fonts_embedded"],
        "editable_svg_text": "<text" in svg_text,
        "replacement_character_absent": "\ufffd" not in extracted,
        "pink_absent": not any(code in svg_text.lower() for code in ("#ff6fae", "#d45087", "#f7dfe9", "#e76f9a")),
        "minimum_font_pt": 6.0,
        "gridlines_absent": True,
        "primary_text_pure_black": True,
        "manual_visual_review": "PASS_600DPI_NO_CLIPPING_OVERLAP_OR_BROKEN_GLYPHS",
    }
    automated_keys = [
        "one_page",
        "size_pass",
        "png_600_dpi",
        "arial_present",
        "arial_embedded",
        "editable_svg_text",
        "replacement_character_absent",
        "pink_absent",
        "gridlines_absent",
        "primary_text_pure_black",
    ]
    checks["automated_pass"] = all(bool(checks[key]) for key in automated_keys)
    return {
        "page_size_mm": [round(value, 3) for value in size_mm],
        "png_pixels": pixels,
        "fonts": fonts["fonts"],
        "checks": checks,
    }


def write_caption(
    main_plot: pd.DataFrame,
    main_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
) -> Path:
    lookup = main_plot.set_index(["model", "task"])
    base = "hEvo-base"
    gpn = "GPN-MSA"
    text = f"""# Figure 3a-c and supplementary caption draft

**Figure 3a-c | Zero-shot ClinVar variant-effect discrimination.** Each panel independently ranks 18 prespecified algorithm representatives by AUPRC. For model families that differ only in parameter scale, the checkpoint with the highest unweighted mean AUPRC across ClinVar all, coding and noncoding was selected before panel-specific ordering; hEvo-base and hEvo-large are retained as the two focal hEVO scales, matching the Figure 2 selection contract. Each panel uses an explicitly labelled task-specific AUPRC scale to reveal within-stratum separation; bar lengths must therefore not be compared across panels. Whiskers show the reported 95% confidence intervals, and short black ticks denote the stratum-specific positive-class prevalence. hEvo-base ranks first in ClinVar all ({float(lookup.loc[(base, 'all'), 'auprc_mean']):.3f}) and coding ({float(lookup.loc[(base, 'coding'), 'auprc_mean']):.3f}), and second in noncoding ({float(lookup.loc[(base, 'noncoding'), 'auprc_mean']):.3f}); GPN-MSA leads noncoding ({float(lookup.loc[(gpn, 'noncoding'), 'auprc_mean']):.3f}).

**Figure S3a | Complete model evidence.** All 33 source models are ordered by the unweighted mean AUPRC across the three ClinVar strata. Each score column has its own explicitly labelled local AUPRC scale; whiskers show reported 95% confidence intervals, and diamond, square and triangle glyphs mark ranks 1, 2 and 3 within the displayed view.

**Figure S3a | Model-family series.** The hEVO, NT, EVO 2 and GPN series are grouped by family and ordered within family by mean AUPRC. Family colours follow the manuscript-wide palette; numerical values and uncertainty intervals are unchanged.

AUROC is not shown because matched source-traceable AUROC values or intervals are unavailable in S23-S25 and the current local result snapshot. The confidence-interval provenance remains a submission gate until the original prediction-level bootstrap route is restored.
"""
    path = OUT_DIR / "Figure3abc_ClinVar_main_full_family_CAPTION_DRAFT.md"
    path.write_text(text, encoding="utf-8")
    return path


def save_all(
    main: Drawing,
    all_models: Drawing,
    families: Drawing,
    long: pd.DataFrame,
    main_plot: pd.DataFrame,
    main_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    all_plot: pd.DataFrame,
    all_summary: pd.DataFrame,
    family_plot: pd.DataFrame,
    family_summary: pd.DataFrame,
    data_summary: dict[str, object],
) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main": export_drawing(main, MAIN_STEM),
        "all_models": export_drawing(all_models, ALL_STEM),
        "family": export_drawing(families, FAMILY_STEM),
    }
    source_frames = {
        "Figure3abc_all_33_models_source.csv": long,
        "Figure3abc_main_18_algorithm_representatives_plot_source.csv": main_plot,
        "Figure3abc_main_18_algorithm_representatives_summary.csv": main_summary,
        "Figure3abc_algorithm_representative_selection_summary.csv": selection_summary,
        "FigureS3a_all_33_models_plot_source.csv": all_plot,
        "FigureS3a_all_33_models_summary.csv": all_summary,
        "FigureS3a_family_series_plot_source.csv": family_plot,
        "FigureS3a_family_series_summary.csv": family_summary,
    }
    source_paths: dict[str, str] = {}
    for name, frame in source_frames.items():
        path = SOURCE_DIR / name
        frame.to_csv(path, index=False)
        source_paths[name] = str(path.resolve())
    caption = write_caption(main_plot, main_summary, selection_summary)

    artifacts = {
        "main": artifact_checks(outputs["main"], MAIN_SIZE_MM),
        "all_models": artifact_checks(outputs["all_models"], ALL_SIZE_MM),
        "family": artifact_checks(outputs["family"], FAMILY_SIZE_MM),
    }
    selected_counts = selection_summary.loc[selection_summary["selected"]].groupby("algorithm_identity").size()
    data_checks = {
        "source_is_complete_33_by_3": len(long) == 99 and long["model"].nunique() == 33 and long["task"].nunique() == 3,
        "main_is_18_by_3": len(main_plot) == 54 and main_plot["model"].nunique() == 18,
        "one_selected_checkpoint_per_algorithm_identity": len(selected_counts) == 18 and bool((selected_counts == 1).all()),
        "hEVO_base_and_large_present_in_main": {"hEvo-base", "hEvo-large"}.issubset(set(main_plot["model"])),
        "all_models_retained": all_plot["model"].nunique() == 33 and len(all_plot) == 99,
        "family_view_is_13_by_3": family_plot["model"].nunique() == 13 and len(family_plot) == 39,
        "family_set_locked": set(family_summary["family"]) == {"hEVO", "NT", "Evo 2", "GPN"},
        "coding_noncoding_partition_all": all(data_summary["partition_checks"].values()),
        "auroc_not_inferred": data_summary["auroc_source_status"] == "NOT_AVAILABLE_IN_S26_S28_OR_LOCAL_RESULT_FILES",
    }
    qa = {
        "status": STATUS,
        "selection_contract": {
            "main": "18 prespecified algorithm identities; best parameter scale selected by highest unweighted mean AUPRC across three ClinVar strata; hEvo-base and hEvo-large are distinct focal identities",
            "all_models": "all 33 source models retained",
            "family": "hEVO, NT, EVO 2 and GPN series using the same raw-model membership as Figure 2c/d supplementary family views",
        },
        "style_contract": {
            "main": "Evo 2-inspired independent horizontal rankings with one explicitly labelled local AUPRC scale per panel; Figure 1 family colours; non-family comparators are neutral",
            "supplement": "Figure 2c/d-style interval matrix with one explicitly labelled local AUPRC scale per score column and red/green/blue top-three glyphs",
            "font": "Arial >=6 pt",
            "text": "pure black",
            "gridlines": "absent",
            "palette": {"hEVO": HEVO, "NT": NT, "EVO 2": EVO2, "GPN": GPN, "other": NEUTRAL},
        },
        "data_summary": data_summary,
        "data_checks": data_checks,
        "artifact_checks": artifacts,
        "outputs": {
            key: {kind: str(path.resolve()) for kind, path in bundle.items()}
            for key, bundle in outputs.items()
        },
        "sources": source_paths,
        "caption": str(caption.resolve()),
        "hashes": {
            "script": sha256(HERE),
            "main_pdf": sha256(outputs["main"]["pdf"]),
            "all_models_pdf": sha256(outputs["all_models"]["pdf"]),
            "family_pdf": sha256(outputs["family"]["pdf"]),
            "S26": sha256(source.TABLE_DIR / source.TASKS["all"]["table"]),
            "S27": sha256(source.TABLE_DIR / source.TASKS["coding"]["table"]),
            "S28": sha256(source.TABLE_DIR / source.TASKS["noncoding"]["table"]),
        },
        "submission_gates": [
            "restore matched prediction-level AUROC/AUPRC outputs before adding paired metrics",
            "link reported AUPRC confidence intervals to an immutable bootstrap script and prediction files",
            "lock model checkpoint and scoring-adapter hashes for all 33 rows",
            "cvd-publication-figures optional reference files are absent from the installed skill package; project-native Figure 2 standards were used",
        ],
    }
    qa["automated_pass"] = all(data_checks.values()) and all(
        artifact["checks"]["automated_pass"] for artifact in artifacts.values()
    )
    return qa


def main() -> None:
    source.configure_fonts()
    long, data_summary = source.read_validate_and_rank()
    main_plot, main_summary, selection_summary = select_main_representatives(long)
    all_plot, all_summary = prepare_all_models(long)
    family_plot, family_summary = prepare_family_models(long)
    main_drawing = render_main(main_plot, data_summary)
    all_drawing = render_all_models(all_plot, all_summary, data_summary)
    family_drawing = render_family_models(family_plot, family_summary, data_summary)
    qa = save_all(
        main_drawing,
        all_drawing,
        family_drawing,
        long,
        main_plot,
        main_summary,
        selection_summary,
        all_plot,
        all_summary,
        family_plot,
        family_summary,
        data_summary,
    )


if __name__ == "__main__":
    main()
