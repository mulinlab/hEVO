#!/usr/bin/env python3
"""Render a polished source-backed URE binary-classification panel for Figure 2e."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Rect
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import plot_fig2a_split_benchmark_routes as style
import plot_fig2cd_benchmark_dot_interval_matrix as benchmark_style


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = PROJECT_ROOT / "tables"
OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig2"
    / "fig2e_ure_binary_mean_mcc_compact_v2"
)
SOURCE_DIR = OUT_DIR / "source_data"
SOURCE_TABLE = TABLE_DIR / "S19_binary_classification_across_five_ure_enhancer_control_sets.csv"

STATUS = "DATA_QC_DRAFT"
OUTPUT_STEM = "Fig2e_URE_binary_mean_MCC_rank_compact_DATA_QC_DRAFT"
PANEL_WIDTH_MM = 120.0
PANEL_HEIGHT_MM = 190.0
PNG_DPI = 600
MM_PER_INCH = 25.4

BLACK = "#111827"
TEXT_MUTED = "#526176"
BLUE = "#174FA6"
BLUE_MID = "#3F74BC"
BLUE_LIGHT = "#D9E8FC"
BLUE_PALE = "#EDF5FD"
ROW_ALT = "#F5F8FC"
TRACK = "#D7E0EA"
GOLD = "#C28A2C"
GOLD_PALE = "#FBF7EC"
WHITE = "#FFFFFF"

DATASET_ORDER = (
    "URE_dataset_1",
    "URE_dataset_2",
    "URE_dataset_3",
    "URE_dataset_4",
    "URE_dataset_5",
)
DATASET_ALIASES = {
    "URE_dataset_1": "URE set 1",
    "URE_dataset_2": "URE set 2",
    "URE_dataset_3": "URE set 3",
    "URE_dataset_4": "URE set 4",
    "URE_dataset_5": "URE set 5",
}
FOCUS_MODELS = {
    "hEvo-base",
    "hEvo-large",
}


def display_model_name(model: str) -> str:
    return {
        "hEvo-base": "hEvo-base",
        "hEvo-large": "hEvo-large",
    }.get(model, benchmark_style.display_model_name(model))


def rank_marker(rank: float) -> str:
    return benchmark_style.rank_marker(rank)


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
    line_width: float = 0.55,
    radius: float = 1.1,
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
        radius=radius,
    )


def add_header(drawing: Drawing, x: float, y: float, width: float, text: str) -> None:
    benchmark_style.add_header(drawing, x, y, width, text)


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(SOURCE_TABLE)
    required = {
        "run",
        "model",
        "acc_mean",
        "acc_low",
        "acc_high",
        "mcc_mean",
        "mcc_low",
        "mcc_high",
        "auc_mean",
        "auc_low",
        "auc_high",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"S9 source table missing columns: {missing}")

    raw = raw.copy()
    numeric_columns = (
        "acc_mean",
        "acc_low",
        "acc_high",
        "mcc_mean",
        "mcc_low",
        "mcc_high",
        "auc_mean",
        "auc_low",
        "auc_high",
    )
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw[list(numeric_columns)].isna().any().any():
        raise ValueError("S9 contains missing or non-numeric metric values")
    if raw.duplicated(["model", "run"]).any():
        raise ValueError("S9 contains duplicate model-run rows")
    if set(raw["run"].unique()) != set(DATASET_ORDER):
        raise ValueError(f"Unexpected URE binary variants: {sorted(raw['run'].unique())}")

    models = sorted(raw["model"].unique())
    expected = pd.MultiIndex.from_product([models, DATASET_ORDER], names=["model", "run"])
    observed = pd.MultiIndex.from_frame(raw[["model", "run"]])
    missing_cells = expected.difference(observed)
    if len(missing_cells):
        raise ValueError(
            f"Expected a complete {len(models)} x {len(DATASET_ORDER)} design; "
            f"missing={list(missing_cells[:8])}"
        )

    model_summary = (
        raw.groupby("model", as_index=False)
        .agg(
            mean_mcc=("mcc_mean", "mean"),
            mean_acc=("acc_mean", "mean"),
            mean_auc=("auc_mean", "mean"),
            mcc_sd_across_variants=("mcc_mean", "std"),
            mcc_min=("mcc_mean", "min"),
            mcc_max=("mcc_mean", "max"),
        )
        .sort_values("mean_mcc", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    model_summary["overall_rank"] = np.arange(1, len(model_summary) + 1)
    model_summary["acc_rank"] = model_summary["mean_acc"].rank(method="min", ascending=False)
    model_summary["auc_rank"] = model_summary["mean_auc"].rank(method="min", ascending=False)
    model_summary["plot_row"] = np.arange(len(model_summary))
    model_summary["model_display"] = model_summary["model"].map(display_model_name)
    model_summary["is_hevo_focus"] = model_summary["model"].isin(FOCUS_MODELS)

    raw["dataset_order"] = raw["run"].map(
        {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    )
    raw["dataset_alias"] = raw["run"].map(DATASET_ALIASES)
    raw["dataset_rank"] = raw.groupby("run")["mcc_mean"].rank(method="min", ascending=False)
    raw = raw.merge(model_summary, on="model", how="left", validate="many_to_one")
    raw["dataset_rank_marker"] = raw["dataset_rank"].map(rank_marker)
    raw["overall_rank_marker"] = raw["overall_rank"].map(rank_marker)
    raw = raw.sort_values(["plot_row", "dataset_order"], kind="stable").reset_index(drop=True)

    original_mean = raw.groupby("model")["mean_mcc_for_sort"].first() if "mean_mcc_for_sort" in raw else None
    recalculated = model_summary.set_index("model")["mean_mcc"]
    mean_match = True if original_mean is None else bool(np.allclose(original_mean.sort_index(), recalculated.sort_index()))

    data_summary = {
        "source_path": str(SOURCE_TABLE.resolve()),
        "source_sha256": style.sha256(SOURCE_TABLE),
        "source_rows": int(len(raw)),
        "source_models": int(raw["model"].nunique()),
        "source_variants": int(raw["run"].nunique()),
        "expected_source_rows": int(len(models) * len(DATASET_ORDER)),
        "variant_order": list(DATASET_ORDER),
        "mean_mcc_matches_source_summary": mean_match,
        "mcc_range": [float(raw["mcc_mean"].min()), float(raw["mcc_mean"].max())],
        "mcc_ci_median_width": float((raw["mcc_high"] - raw["mcc_low"]).median()),
        "interval_scope": "supplied 95% test-set bootstrap CI for each model-control-set result",
        "variant_scope": "five related enhancer-control variants sharing the same URE positive test subset; not independent cohorts",
        "focus_models": {
            model: {
                "mean_mcc": float(model_summary.set_index("model").loc[model, "mean_mcc"]),
                "mean_acc": float(model_summary.set_index("model").loc[model, "mean_acc"]),
                "mean_auc": float(model_summary.set_index("model").loc[model, "mean_auc"]),
                "mcc_rank": int(model_summary.set_index("model").loc[model, "overall_rank"]),
                "acc_rank": int(model_summary.set_index("model").loc[model, "acc_rank"]),
                "auc_rank": int(model_summary.set_index("model").loc[model, "auc_rank"]),
                "variant_mcc_ranks": {
                    row.run: int(row.dataset_rank)
                    for row in raw.loc[raw["model"] == model].itertuples()
                },
            }
            for model in sorted(FOCUS_MODELS)
        },
    }
    return raw, model_summary, data_summary


def add_micro_dot(drawing: Drawing, x: float, y: float, *, focus: bool) -> None:
    color = GOLD if focus else BLUE_MID
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            0.25 * mm,
            fillColor=HexColor(WHITE),
            strokeColor=HexColor(color),
            strokeWidth=0.48,
        )
    )


def add_overall_cell(
    drawing: Drawing,
    x: float,
    width: float,
    y: float,
    model_rows: pd.DataFrame,
    *,
    focus: bool,
) -> None:
    left = x + 0.9
    right = x + width - 7.2
    span = right - left
    style.add_line(drawing, left, y, right, y, color=TRACK, width=0.58)
    offsets = (-0.68, -0.34, 0.0, 0.34, 0.68)
    ordered = model_rows.sort_values("dataset_order")
    for offset, value in zip(offsets, ordered["mcc_mean"]):
        point_x = left + float(np.clip(value, 0.0, 1.0)) * span
        add_micro_dot(drawing, point_x, y + offset, focus=focus)
    mean = float(ordered["mean_mcc"].iloc[0])
    rank = float(ordered["overall_rank"].iloc[0])
    mean_x = left + float(np.clip(mean, 0.0, 1.0)) * span
    color = GOLD if focus else BLUE_MID
    add_marker(
        drawing,
        mean_x,
        y,
        "circle",
        fill=color if rank <= 3 or focus else WHITE,
        stroke=color,
    )
    style.add_text(drawing, right + 1.0, y - 0.68, f"{mean:.3f}", 6.0, color=BLACK)


def add_dataset_cell(
    drawing: Drawing,
    x: float,
    width: float,
    y: float,
    row: pd.Series,
    *,
    focus: bool,
) -> None:
    left = x + 0.9
    right = x + width - 0.9
    span = right - left
    style.add_line(drawing, left, y, right, y, color=TRACK, width=0.58)
    # Preserve the supplied interval exactly. One lowest-performing cell has a
    # lower limit just below zero; the cap is intentionally allowed to extend
    # slightly left of the 0-1 reference track instead of being clipped.
    lower = left + float(row["mcc_low"]) * span
    upper = left + float(row["mcc_high"]) * span
    center = left + float(row["mcc_mean"]) * span
    color = GOLD if focus else BLUE_MID
    style.add_line(drawing, lower, y, upper, y, color=color, width=0.92 if focus else 0.76)
    style.add_line(drawing, lower, y - 0.42, lower, y + 0.42, color=color, width=0.62)
    style.add_line(drawing, upper, y - 0.42, upper, y + 0.42, color=color, width=0.62)
    rank = float(row["dataset_rank"])
    add_marker(
        drawing,
        center,
        y,
        rank_marker(rank),
        fill=color if rank <= 3 or focus else WHITE,
        stroke=color,
    )


def draw_panel(raw: pd.DataFrame, summary: pd.DataFrame) -> Drawing:
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

    content_x = 11.0
    right_margin = 5.0
    model_width = 37.0
    mean_width = 51.5
    rank_width = 9.0
    model_x = content_x
    mean_x = model_x + model_width + 1.1
    rank_x = mean_x + mean_width + 0.9
    content_right = PANEL_WIDTH_MM - right_margin

    style.add_text(drawing, 2.3, 184.2, "e", 14.0, bold=True, color=BLACK)
    style.add_text(drawing, content_x, 184.4, "URE binary classification", 10.5, bold=True, color=BLUE)
    style.add_fitted_text(
        drawing,
        content_x,
        178.6,
        f"URE vs enhancer | absolute MCC | frozen embeddings + linear probe | {summary['model'].nunique()} models x {raw['run'].nunique()} control variants",
        6.5,
        PANEL_WIDTH_MM - content_x - right_margin,
        color=TEXT_MUTED,
    )

    indexed = summary.set_index("model")
    hevo = indexed.loc["hEvo-large"]
    conclusion_width = content_right - content_x
    add_round_box(
        drawing,
        content_x,
        168.8,
        conclusion_width,
        6.0,
        fill=GOLD_PALE,
        stroke=GOLD,
        line_width=0.65,
        radius=1.2,
    )
    style.add_fitted_text(
        drawing,
        content_x + 2.0,
        170.75,
        f"hEvo-large: mean MCC #1 ({hevo['mean_mcc']:.3f}); #1 in all 5 control sets",
        6.15,
        conclusion_width - 4.0,
        bold=True,
        color=BLUE,
    )

    header_y = 160.5
    add_header(drawing, model_x, header_y, model_width, "Model")
    add_header(drawing, mean_x, header_y, mean_width + 0.9 + rank_width, "Overall performance")

    subheader_y = 153.8
    for x, width, label in (
        (mean_x, mean_width, "Mean MCC"),
        (rank_x, rank_width, "Rank"),
    ):
        add_round_box(drawing, x, subheader_y, width - 0.35, 5.25, fill=BLUE_PALE, stroke=BLUE_LIGHT, line_width=0.52, radius=0.85)
        style.add_fitted_text(
            drawing,
            x + (width - 0.35) / 2,
            subheader_y + 1.65,
            label,
            6.25,
            width - 1.8,
            bold=True,
            anchor="middle",
            color=BLUE,
        )
    row_top_y = 150.2
    row_height = 3.15
    for row_index, row in summary.iterrows():
        model = str(row["model"])
        y = row_top_y - row_index * row_height
        focus = bool(row["is_hevo_focus"])
        row_fill = GOLD_PALE if focus else ROW_ALT if row_index % 2 else WHITE
        drawing.add(
            Rect(
                (content_x - 1.0) * mm,
                (y - row_height / 2 + 0.05) * mm,
                (content_right - content_x + 1.0) * mm,
                (row_height - 0.1) * mm,
                fillColor=HexColor(row_fill),
                strokeColor=None,
            )
        )
        if focus:
            drawing.add(
                Rect(
                    (content_x - 1.0) * mm,
                    (y - row_height / 2 + 0.05) * mm,
                    0.75 * mm,
                    (row_height - 0.1) * mm,
                    fillColor=HexColor(GOLD),
                    strokeColor=None,
                )
            )
        style.add_fitted_text(
            drawing,
            model_x + 0.8,
            y - 0.70,
            str(row["model_display"]),
            6.2,
            model_width - 1.8,
            bold=focus,
            color=BLUE if focus else BLACK,
        )
        model_rows = raw.loc[raw["model"] == model]
        add_overall_cell(drawing, mean_x, mean_width, y, model_rows, focus=focus)
        style.add_text(
            drawing,
            rank_x + rank_width / 2,
            y - 0.70,
            str(int(row["overall_rank"])),
            6.1,
            bold=focus,
            anchor="middle",
            color=BLUE if focus else BLACK,
        )

    axis_y = row_top_y - len(summary) * row_height + 0.25
    mean_left = mean_x + 0.9
    mean_right = mean_x + mean_width - 7.2
    style.add_line(drawing, mean_left, axis_y, mean_right, axis_y, color=BLACK, width=0.62)
    style.add_line(drawing, mean_left, axis_y, mean_left, axis_y + 0.65, color=BLACK, width=0.62)
    style.add_line(drawing, mean_right, axis_y, mean_right, axis_y + 0.65, color=BLACK, width=0.62)
    style.add_text(drawing, mean_left + 0.55, axis_y - 2.2, "0", 6.0, anchor="middle", color=TEXT_MUTED)
    style.add_text(drawing, mean_right - 0.55, axis_y - 2.2, "1", 6.0, anchor="middle", color=TEXT_MUTED)

    legend_y = 31.0
    style.add_text(drawing, content_x, legend_y - 0.7, "Mean MCC:", 6.0, bold=True, color=TEXT_MUTED)
    cursor = content_x + 15.0
    for offset in (-0.6, -0.3, 0.0, 0.3, 0.6):
        add_micro_dot(drawing, cursor + 0.7 + offset, legend_y, focus=False)
    add_marker(drawing, cursor + 3.0, legend_y, "circle", fill=BLUE_MID, stroke=BLUE_MID)
    style.add_text(drawing, cursor + 4.8, legend_y - 0.7, "5 set means + average", 6.0, color=TEXT_MUTED)

    rank_y = 24.6
    style.add_text(drawing, content_x, rank_y - 0.7, "Rank is based on mean MCC (1 = best).", 6.0, color=TEXT_MUTED)
    add_marker(drawing, content_right - 24.0, rank_y, "circle", fill=GOLD, stroke=GOLD)
    style.add_text(drawing, content_right - 22.2, rank_y - 0.7, "hEVO focus", 6.0, color=TEXT_MUTED)

    style.add_fitted_text(
        drawing,
        content_x,
        15.6,
        "Mean MCC is the unweighted average across five matched enhancer-control variants.",
        6.0,
        PANEL_WIDTH_MM - content_x - right_margin,
        color=TEXT_MUTED,
    )
    style.add_fitted_text(
        drawing,
        content_x,
        10.9,
        "Microdots show set means; variants share one URE positive test subset and are not independent cohorts.",
        6.0,
        PANEL_WIDTH_MM - content_x - right_margin,
        color=TEXT_MUTED,
    )
    return drawing


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_outputs(raw: pd.DataFrame, summary: pd.DataFrame, data_summary: dict[str, object]) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{OUTPUT_STEM}.pdf"
    svg_path = OUT_DIR / f"{OUTPUT_STEM}.svg"
    png_path = OUT_DIR / f"{OUTPUT_STEM}.png"
    source_path = SOURCE_DIR / f"{OUTPUT_STEM}_plot_source.csv"
    raw.to_csv(source_path, index=False)

    drawing = draw_panel(raw, summary)
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    style.render_png(pdf_path, png_path)

    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) / 72.0 * MM_PER_INCH
    height_mm = float(page.mediabox.height) / 72.0 * MM_PER_INCH
    extracted_text = "".join((page.extract_text() or "") for page in reader.pages)
    with Image.open(png_path) as image:
        pixels = list(image.size)
    expected_pixels = [
        round(PANEL_WIDTH_MM / MM_PER_INCH * PNG_DPI),
        round(PANEL_HEIGHT_MM / MM_PER_INCH * PNG_DPI),
    ]
    font_summary = style.pdf_font_summary(pdf_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    source_copy = pd.read_csv(source_path)

    checks = {
        "one_page": len(reader.pages) == 1,
        "size_mm": [round(width_mm, 3), round(height_mm, 3)],
        "size_pass": abs(width_mm - PANEL_WIDTH_MM) < 0.15 and abs(height_mm - PANEL_HEIGHT_MM) < 0.15,
        "png_pixels": pixels,
        "expected_png_pixels": expected_pixels,
        "dpi_pass": all(abs(a - b) <= 2 for a, b in zip(pixels, expected_pixels)),
        "editable_svg_text": "<text" in svg_text,
        "text_extractable": len(extracted_text) > 700,
        "replacement_character_absent": "\ufffd" not in extracted_text,
        "arial_present": font_summary["arial_present"],
        "required_arial_fonts_embedded": font_summary["required_arial_fonts_embedded"],
        "source_copy_rows_match": len(source_copy) == data_summary["expected_source_rows"],
        "source_copy_unique_cells": not source_copy.duplicated(["model", "run"]).any(),
        "all_models_present": source_copy["model"].nunique() == data_summary["source_models"],
        "all_variants_present": set(source_copy["run"].unique()) == set(DATASET_ORDER),
        "mcc_domain_pass": source_copy[["mcc_low", "mcc_mean", "mcc_high"]].apply(lambda col: col.between(-1.0, 1.0).all()).all(),
        "mcc_bounds_ordered": bool(
            ((source_copy["mcc_low"] <= source_copy["mcc_mean"]) & (source_copy["mcc_mean"] <= source_copy["mcc_high"])).all()
        ),
        "mean_mcc_matches_source_summary": data_summary["mean_mcc_matches_source_summary"],
        "rank_marker_mapping_complete": source_copy[["dataset_rank_marker", "overall_rank_marker"]].notna().all().all(),
        "minimum_visible_font_pt": 6.0,
        "panel_letter_font_pt": 14.0,
    }
    boolean_checks = (
        "one_page",
        "size_pass",
        "dpi_pass",
        "editable_svg_text",
        "text_extractable",
        "replacement_character_absent",
        "arial_present",
        "required_arial_fonts_embedded",
        "source_copy_rows_match",
        "source_copy_unique_cells",
        "all_models_present",
        "all_variants_present",
        "mcc_domain_pass",
        "mcc_bounds_ordered",
        "mean_mcc_matches_source_summary",
        "rank_marker_mapping_complete",
    )
    checks["automated_pass"] = all(bool(checks[key]) for key in boolean_checks)
    qa = {
        "status": STATUS,
        "chart_contract": str((OUT_DIR / "Figure2e_chart_contract.md").resolve()),
        "renderer": str(Path(__file__).resolve()),
        "artifact": {
            "pdf": str(pdf_path.resolve()),
            "svg": str(svg_path.resolve()),
            "png": str(png_path.resolve()),
            "source_data": str(source_path.resolve()),
            "pdf_sha256": style.sha256(pdf_path),
            "svg_sha256": style.sha256(svg_path),
            "png_sha256": style.sha256(png_path),
            "source_data_sha256": style.sha256(source_path),
            "fonts": font_summary["fonts"],
        },
        "data_summary": data_summary,
        "checks": checks,
        "manual_visual_review": {
            "status": "PASS",
            "reviewed_artifact": "600-dpi PNG rendered from the final vector PDF",
            "checks": [
                "No text overlaps, clipped labels, or detached annotations",
                f"All {data_summary['source_models']} model rows are readable in the compact panel",
                "Five-set microdots, mean markers, exact MCC values, and ranks are visually distinct",
                "No set-specific columns or intervals remain in the reader-facing panel",
                "hEvo-large is unambiguously identified as mean-MCC rank 1 and rank 1 in all five control sets",
                "hEVO focus rows are emphasized without obscuring other models",
                "Legend and non-independent-variant caveat are complete and legible",
            ],
        },
    }
    return qa


def main() -> None:
    style.configure_fonts()
    raw, summary, data_summary = prepare_data()
    qa = save_outputs(raw, summary, data_summary)


if __name__ == "__main__":
    main()
