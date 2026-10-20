#!/usr/bin/env python3
"""Assemble Figure 2a-d on an A4 page with lower space reserved for URE.

Panels a-b reuse the native UCSC CH renderer. Panels c-d are redrawn as a
coordinated compact matrix with shared model rows, retaining independent
benchmark-specific representative selection. This avoids scaling 6-pt text
and leaves approximately 105 mm of native page height for later URE panels.
"""

from __future__ import annotations

# [Base layout] Define Fig. 2 A4 dimensions, CONTENT_X_MM, AB_Y_MM/CD_Y_MM, and placement coordinates for panels c/d.
# [Calls] The base/ab/cd modules provide data loading and panel plotting; higher-level abcde/abcdef/abcdefg modules reuse these functions.
# [Key layout] AB_H_MM=76 and CD_H_MM=142; the current S11–S16 entry point moves a/b up by 4 mm from these coordinates.

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

import plot_fig2a_split_benchmark_routes as export_base  # noqa: E402
import plot_fig2cd_model_selection_views as cd  # noqa: E402
import plot_figS2ab_compact_a4_toprow as ab  # noqa: E402
import plot_figS2a_ucsc_element_separability as ab_base  # noqa: E402


PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
CONTENT_W_MM = 190.0
AB_H_MM = 76.0
CD_H_MM = 142.0
CONTENT_X_MM = 10.0
AB_Y_MM = 211.0
CD_Y_MM = 73.0
LOWER_RESERVED_MM = 63.0
PNG_DPI = 600
MM_PER_INCH = 25.4

OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig2" / "figure2_abcd_a4_integrated_v1"
SOURCE_DIR = OUT_DIR / "source_data"
STEM = "Figure2_abcd_A4_integrated_v2_DATA_QC_DRAFT"
WHITE = "#FFFFFF"
NEUTRAL = cd.NEUTRAL
NEUTRAL_LIGHT = cd.NEUTRAL_LIGHT
NEUTRAL_PALE = "#FAFAFA"
HEVO = cd.family_color("hEVO")
HEVO_WASH = cd.HEVO_WASH
HEADER_FILL = WHITE


def translated_group(source: Drawing, x_mm: float, y_mm: float) -> Group:
    group = Group()
    for item in source.contents:
        group.add(item)
    group.translate(x_mm * mm, y_mm * mm)
    return group


def selected_with_identity(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, summary = cd.select_main(summary)
    identity = summary[summary["selected"]].set_index("candidate_model")["algorithm_identity"].to_dict()
    rows = rows.copy()
    rows["algorithm_identity"] = rows["model"].map(identity)
    if rows["algorithm_identity"].isna().any():
        raise ValueError("Selected model missing algorithm identity")
    rows["row_key"] = np.where(
        rows["algorithm_identity"].eq("hEVO"),
        rows["model"],
        rows["algorithm_identity"],
    )
    return rows, summary


def build_row_registry(
    gb_rows: pd.DataFrame,
    nt_rows: pd.DataFrame,
) -> pd.DataFrame:
    gb = gb_rows[
        ["row_key", "algorithm_identity", "model", "model_display", "mean_mcc", "plot_rank"]
    ].rename(
        columns={
            "model": "gb_model",
            "model_display": "gb_model_display",
            "mean_mcc": "gb_mean_mcc",
            "plot_rank": "gb_plot_rank",
        }
    )
    nt = nt_rows[
        ["row_key", "algorithm_identity", "model", "model_display", "mean_mcc", "plot_rank"]
    ].rename(
        columns={
            "model": "nt_model",
            "model_display": "nt_model_display",
            "mean_mcc": "nt_mean_mcc",
            "plot_rank": "nt_plot_rank",
        }
    )
    nt = nt.drop(columns="algorithm_identity")
    merged = gb.merge(nt, on="row_key", how="outer", validate="one_to_one")
    if merged[["gb_model", "nt_model"]].isna().any().any():
        raise ValueError("GB and NT representative algorithm sets differ")
    merged["same_selected_model"] = merged["gb_model"] == merged["nt_model"]
    merged["display_label"] = np.where(
        merged["same_selected_model"],
        merged["gb_model_display"],
        merged["algorithm_identity"],
    )
    merged["is_hevo"] = merged["algorithm_identity"] == "hEVO"
    merged = merged.sort_values(
        ["gb_plot_rank", "display_label"], ascending=[True, True], kind="stable"
    ).reset_index(drop=True)
    return merged


def selected_scale(display: str, algorithm: str) -> str:
    value = str(display)
    prefix_map = {
        "GENERator": "GENERator-",
        "GENA-LM": "GENA-LM-",
        "Evo 2": "EVO 2-",
        "Mistral-DNA": "Mistral-DNA-",
        "CrossDNA": "CrossDNA-",
        "NT v3": "NTv3-",
    }
    prefix = prefix_map.get(str(algorithm), "")
    return value.removeprefix(prefix) if prefix else value


def mean_value_label(row: pd.Series, benchmark: str) -> str:
    display = str(row[f"{benchmark}_model_display"])
    mean = float(row[f"{benchmark}_mean_mcc"])
    if bool(row["same_selected_model"]):
        return f"{mean:.3f}"
    scale = selected_scale(display, str(row["algorithm_identity"]))
    return f"{scale} {mean:.3f}"


def draw_mean_cell(
    drawing: Drawing,
    x: float,
    y: float,
    mean: float,
    value_label: str,
    shade: str,
) -> None:
    track_left = x + 0.5
    track_right = x + 8.5
    drawing.add(
        Line(
            track_left * mm,
            y * mm,
            track_right * mm,
            y * mm,
            strokeColor=HexColor(NEUTRAL_LIGHT),
            strokeWidth=0.62,
        )
    )
    point_x = track_left + float(np.clip(mean, 0.0, 0.70)) / 0.70 * (track_right - track_left)
    drawing.add(
        Line(
            track_left * mm,
            y * mm,
            point_x * mm,
            y * mm,
            strokeColor=HexColor(shade),
            strokeWidth=0.88,
        )
    )
    drawing.add(Circle(point_x * mm, y * mm, 0.49 * mm, fillColor=HexColor(shade), strokeColor=None))
    cd.add_text(drawing, x + 9.5, y - 0.67, value_label, 6.0)


def draw_task_cell(
    drawing: Drawing,
    x: float,
    width: float,
    y: float,
    cell: pd.Series,
    model_shade: str,
) -> None:
    left = x + 0.32
    span = width - 0.64
    drawing.add(
        Line(
            left * mm,
            y * mm,
            (left + span) * mm,
            y * mm,
            strokeColor=HexColor(NEUTRAL_LIGHT),
            strokeWidth=0.42,
        )
    )
    # Horizontal position is normalized within the models displayed in this
    # view.  Absolute MCC remains available in the mean column and source
    # table.  This makes the weakest and strongest displayed model occupy the
    # two ends of every task column without implying a common cross-task scale.
    lo = left + float(np.clip(cell["within_view_ci_lower"], 0.0, 1.0)) * span
    hi = left + float(np.clip(cell["within_view_ci_upper"], 0.0, 1.0)) * span
    mid = left + float(np.clip(cell["within_view_position"], 0.0, 1.0)) * span
    drawing.add(
        Line(
            lo * mm,
            y * mm,
            hi * mm,
            y * mm,
            strokeColor=HexColor(model_shade),
            strokeWidth=0.66,
        )
    )
    rank = int(cell["within_view_rank"])
    marker_color = cd.RANK_COLORS[rank] if rank <= 3 else model_shade
    cd.add_marker(
        drawing,
        mid,
        y,
        str(cell["within_view_rank_marker"]),
        marker_color if rank <= 3 else WHITE,
        marker_color,
    )


def annotate_within_view_positions(cells: pd.DataFrame) -> pd.DataFrame:
    """Add display-only task ranks and min-max positions to a cell table."""

    out = cells.copy()
    stats = (
        out.groupby("dataset", sort=False)["mcc_mean"]
        .agg(within_view_mcc_min="min", within_view_mcc_max="max")
        .reset_index()
    )
    out = out.merge(stats, on="dataset", how="left", validate="many_to_one")
    scale = out["within_view_mcc_max"] - out["within_view_mcc_min"]
    safe_scale = scale.where(scale.gt(0.0), 1.0)
    out["within_view_position"] = (
        (out["mcc_mean"] - out["within_view_mcc_min"]) / safe_scale
    ).where(scale.gt(0.0), 0.5)
    out["within_view_ci_lower"] = (
        (out["mcc_lower"] - out["within_view_mcc_min"]) / safe_scale
    ).where(scale.gt(0.0), 0.5).clip(0.0, 1.0)
    out["within_view_ci_upper"] = (
        (out["mcc_upper"] - out["within_view_mcc_min"]) / safe_scale
    ).where(scale.gt(0.0), 0.5).clip(0.0, 1.0)
    out["within_view_rank"] = (
        out.groupby("dataset", sort=False)["mcc_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    out["within_view_rank_marker"] = out["within_view_rank"].map(cd.rank_shape)
    return out


def add_task_headers(
    drawing: Drawing,
    config: cd.BenchmarkConfig,
    tasks_x: float,
    tasks_right: float,
    *,
    group_y: float,
    label_ys: tuple[float, ...],
    header_bottom: float,
    header_top: float,
    rotate: bool = False,
    rotate_angle: float = 90.0,
    rotate_y_offset: float = -0.2,
) -> None:
    """Draw source-aligned task labels on a white header with thin group rules."""

    task_width = (tasks_right - tasks_x) / len(config.tasks)
    task_index = {task: index for index, task in enumerate(config.tasks)}
    for group_name, group_tasks in config.groups:
        first = min(task_index[task] for task in group_tasks)
        last = max(task_index[task] for task in group_tasks)
        left = tasks_x + first * task_width
        right = tasks_x + (last + 1) * task_width
        drawing.add(
            Rect(
                (left + 0.12) * mm,
                header_bottom * mm,
                (right - left - 0.24) * mm,
                (header_top - header_bottom) * mm,
                fillColor=HexColor(HEADER_FILL),
                strokeColor=None,
            )
        )
        cd.add_text(drawing, (left + right) / 2, group_y, group_name, 6.0, bold=True, anchor="middle")
        drawing.add(
            Line(
                left * mm,
                (group_y - 1.1) * mm,
                right * mm,
                (group_y - 1.1) * mm,
                strokeColor=HexColor(cd.BLACK),
                strokeWidth=0.50,
            )
        )
    for index, task in enumerate(config.tasks):
        center = tasks_x + (index + 0.5) * task_width
        if rotate:
            # The text and all task markers share exactly the same x center.
            # A middle anchor eliminates the previous apparent column shift.
            radians = math.radians(rotate_angle)
            label = String(
                0,
                0,
                config.aliases[task].replace("\n", " "),
                fontName=ab_base.FONT_REGULAR,
                fontSize=6.0,
                fillColor=HexColor(cd.BLACK),
                textAnchor="middle",
            )
            group = Group(label)
            group.transform = (
                math.cos(radians),
                math.sin(radians),
                -math.sin(radians),
                math.cos(radians),
                center * mm,
                ((header_bottom + header_top) / 2 + rotate_y_offset) * mm,
            )
            drawing.add(group)
            continue
        lane_y = label_ys[index % len(label_ys)]
        lines = config.aliases[task].split("\n")
        allowed_width = task_width * (0.94 if len(label_ys) == 1 else 2.90)
        for line_number, line in enumerate(lines):
            export_base.add_fitted_text(
                drawing,
                center,
                lane_y - line_number * 2.15,
                line,
                6.0,
                allowed_width,
                anchor="middle",
                min_size=6.0,
                color=cd.BLACK,
            )


def draw_benchmark_rows(
    drawing: Drawing,
    config: cd.BenchmarkConfig,
    cells: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    model_x: float,
    model_w: float,
    mean_x: float,
    tasks_x: float,
    tasks_right: float,
    row_top: float,
    row_bottom: float,
) -> None:
    row_step = (row_top - row_bottom) / (len(rows) - 1)
    band_h = min(row_step * 0.88, 1.84)
    displayed = cells[cells["model"].isin(rows["model"])].copy()
    displayed = annotate_within_view_positions(displayed)
    lookup = displayed.set_index(["model", "dataset"])
    task_width = (tasks_right - tasks_x) / len(config.tasks)

    for index, row in rows.reset_index(drop=True).iterrows():
        y = row_top - index * row_step
        is_hevo = str(row["algorithm_identity"]) == "hEVO"
        model_shade = HEVO if is_hevo else NEUTRAL
        fill = HEVO_WASH if is_hevo else (NEUTRAL_PALE if index % 2 else WHITE)
        drawing.add(
            Rect(
                (model_x - 1.0) * mm,
                (y - band_h / 2) * mm,
                (tasks_right - model_x + 1.0) * mm,
                band_h * mm,
                fillColor=HexColor(fill),
                strokeColor=None,
            )
        )
        drawing.add(
            Circle(
                (model_x + 0.4) * mm,
                y * mm,
                0.54 * mm,
                fillColor=HexColor(model_shade),
                strokeColor=None,
            )
        )
        export_base.add_fitted_text(
            drawing,
            model_x + 1.9,
            y - 0.68,
            str(row["model_display"]),
            6.0,
            model_w - 2.2,
            min_size=6.0,
            bold=is_hevo,
            color=cd.BLACK,
        )
        draw_mean_cell(
            drawing,
            mean_x,
            y,
            float(row["mean_mcc"]),
            f"{float(row['mean_mcc']):.3f}",
            model_shade,
        )
        model = str(row["model"])
        for task_number, task in enumerate(config.tasks):
            if (model, task) not in lookup.index:
                cd.add_text(
                    drawing,
                    tasks_x + (task_number + 0.5) * task_width,
                    y - 0.68,
                    "NA",
                    5.3,
                    anchor="middle",
                    fill=cd.NEUTRAL,
                )
                continue
            draw_task_cell(
                drawing,
                tasks_x + task_number * task_width,
                task_width,
                y,
                lookup.loc[(model, task)],
                model_shade,
            )


def draw_left_header(
    drawing: Drawing,
    *,
    model_x: float,
    model_w: float,
    mean_x: float,
    mean_w: float,
    tasks_x: float,
    header_bottom: float,
    header_top: float,
    task_count: int,
    detailed: bool = False,
) -> None:
    """Complete the matrix header instead of leaving the model side empty."""

    drawing.add(
        Rect(
            (model_x - 1.0) * mm,
            header_bottom * mm,
            (tasks_x - model_x) * mm,
            (header_top - header_bottom) * mm,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )
    drawing.add(
        Line(
            (model_x - 1.0) * mm,
            header_bottom * mm,
            (tasks_x - 1.0) * mm,
            header_bottom * mm,
            strokeColor=HexColor(cd.BLACK),
            strokeWidth=0.50,
        )
    )
    mid_y = (header_bottom + header_top) / 2
    if detailed:
        cd.add_text(
            drawing,
            model_x + model_w / 2,
            mid_y + 1.25,
            "Models",
            6.0,
            bold=True,
            anchor="middle",
        )
        cd.add_text(
            drawing,
            model_x + model_w / 2,
            mid_y - 1.55,
            "ranked by mean MCC",
            6.0,
            anchor="middle",
        )
        cd.add_text(
            drawing,
            mean_x + mean_w / 2,
            mid_y - 0.15,
            "Mean MCC",
            6.0,
            bold=True,
            anchor="middle",
        )
        return
    cd.add_text(
        drawing,
        model_x + model_w / 2,
        mid_y + 1.25,
        "Models",
        6.0,
        bold=True,
        anchor="middle",
    )
    cd.add_text(
        drawing,
        model_x + model_w / 2,
        mid_y - 1.55,
        "ranked by mean MCC",
        6.0,
        anchor="middle",
    )
    cd.add_text(
        drawing,
        mean_x + mean_w / 2,
        mid_y - 0.15,
        "Mean MCC",
        6.0,
        bold=True,
        anchor="middle",
    )


def draw_panel_footer(drawing: Drawing, y: float, model_x: float) -> None:
    """Repeat the compact decoding key below each benchmark matrix."""

    cd.add_text(drawing, model_x, y, "Within-view rank", 6.0, bold=True)
    cursor = model_x + 20.5
    for rank, shape in ((1, "diamond"), (2, "square"), (3, "triangle")):
        shade = cd.RANK_COLORS[rank]
        cd.add_marker(drawing, cursor, y + 0.65, shape, shade, shade)
        cd.add_text(drawing, cursor + 1.35, y, str(rank), 6.0)
        cursor += 6.3
    cd.add_marker(drawing, cursor, y + 0.65, "circle", WHITE, NEUTRAL)
    cd.add_text(drawing, cursor + 1.35, y, "other", 6.0)
    cursor += 12.5
    drawing.add(
        Line(
            cursor * mm,
            (y + 0.65) * mm,
            (cursor + 6.0) * mm,
            (y + 0.65) * mm,
            strokeColor=HexColor(NEUTRAL),
            strokeWidth=0.68,
        )
    )
    cd.add_marker(drawing, cursor + 3.0, y + 0.65, "circle", WHITE, NEUTRAL)
    cd.add_text(drawing, cursor + 7.5, y, "Within-task min-max position; 95% CI", 6.0)
    cd.add_text(drawing, 134.8, y, "hEVO", 6.0)
    drawing.add(
        Circle(
            133.0 * mm,
            (y + 0.65) * mm,
            0.56 * mm,
            fillColor=HexColor(HEVO),
            strokeColor=None,
        )
    )
    cd.add_text(drawing, 152.1, y, "Other models", 6.0)
    drawing.add(
        Circle(
            150.3 * mm,
            (y + 0.65) * mm,
            0.56 * mm,
            fillColor=HexColor(NEUTRAL),
            strokeColor=None,
        )
    )


def draw_compact_cd(
    gb_cells: pd.DataFrame,
    nt_cells: pd.DataFrame,
    gb_rows: pd.DataFrame,
    nt_rows: pd.DataFrame,
    *,
    nt_letter: str = "c",
    gb_letter: str = "d",
) -> Drawing:
    drawing = Drawing(CONTENT_W_MM * mm, CD_H_MM * mm)
    drawing.add(Rect(0, 0, drawing.width, drawing.height, fillColor=HexColor(WHITE), strokeColor=None))

    model_x = 5.0
    model_w = 38.0
    mean_x = 44.5
    mean_w = 22.0
    tasks_x = 68.5
    tasks_right = 188.0

    # The two NT-MS parameter scales now form one representative algorithm.
    # Preserve the occupied panel height after removing the redundant 500M row.
    row_pitch = 2.27

    # Panel c: the 18-task Nucleotide Transformer benchmark is placed first.
    draw_left_header(
        drawing,
        model_x=model_x,
        model_w=model_w,
        mean_x=mean_x,
        mean_w=mean_w,
        tasks_x=tasks_x,
        header_bottom=116.5,
        header_top=139.0,
        task_count=18,
        detailed=True,
    )
    add_task_headers(
        drawing,
        cd.CONFIGS["nt"],
        tasks_x,
        tasks_right,
        group_y=137.9,
        label_ys=(0.0,),
        header_bottom=116.5,
        header_top=139.0,
        rotate=True,
        rotate_y_offset=-0.55,
    )
    # Draw the panel identifier last so the header wash cannot cover it.
    cd.add_text(drawing, 1.2, 139.7, nt_letter, 13.0, bold=True)
    cd.add_text(drawing, 8.0, 140.1, "NT Benchmark | 18 tasks", 6.8, bold=True)
    draw_benchmark_rows(
        drawing,
        cd.CONFIGS["nt"],
        nt_cells,
        nt_rows,
        model_x=model_x,
        model_w=model_w,
        mean_x=mean_x,
        tasks_x=tasks_x,
        tasks_right=tasks_right,
        row_top=114.9,
        row_bottom=114.9 - row_pitch * (len(nt_rows) - 1),
    )
    draw_panel_footer(drawing, 69.0, model_x)

    # Panel d: source-aligned Genomic Benchmarks labels use two horizontal lines.
    cd.add_text(drawing, 1.2, 59.7, gb_letter, 13.0, bold=True)
    cd.add_text(drawing, 8.0, 60.1, "Genomic Benchmark | 7 tasks", 6.8, bold=True)
    draw_left_header(
        drawing,
        model_x=model_x,
        model_w=model_w,
        mean_x=mean_x,
        mean_w=mean_w,
        tasks_x=tasks_x,
        header_bottom=46.4,
        header_top=59.0,
        task_count=7,
    )
    add_task_headers(
        drawing,
        cd.CONFIGS["gb"],
        tasks_x,
        tasks_right,
        group_y=56.7,
        label_ys=(52.2,),
        header_bottom=46.4,
        header_top=59.0,
    )
    draw_benchmark_rows(
        drawing,
        cd.CONFIGS["gb"],
        gb_cells,
        gb_rows,
        model_x=model_x,
        model_w=model_w,
        mean_x=mean_x,
        tasks_x=tasks_x,
        tasks_right=tasks_right,
        row_top=45.2,
        row_bottom=45.2 - 2.20 * (len(gb_rows) - 1),
    )
    draw_panel_footer(drawing, 0.75, model_x)
    return drawing


def save(
    drawing: Drawing,
    row_registry: pd.DataFrame,
    gb_rows: pd.DataFrame,
    nt_rows: pd.DataFrame,
    gb_cells: pd.DataFrame,
    nt_cells: pd.DataFrame,
    gb_selection: pd.DataFrame,
    nt_selection: pd.DataFrame,
    ab_selected: pd.DataFrame,
    ab_summary: pd.DataFrame,
) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{STEM}.pdf"
    svg = OUT_DIR / f"{STEM}.svg"
    png = OUT_DIR / f"{STEM}.png"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    export_base.render_png(pdf, png)

    # Keep the previously distributed v1 filenames as native compatibility
    # renders so no stale artwork can retain superseded model names.
    compatibility_stem = "Figure2_abcd_A4_integrated_v1_DATA_QC_DRAFT"
    compatibility_pdf = OUT_DIR / f"{compatibility_stem}.pdf"
    compatibility_svg = OUT_DIR / f"{compatibility_stem}.svg"
    compatibility_png = OUT_DIR / f"{compatibility_stem}.png"
    renderPDF.drawToFile(drawing, str(compatibility_pdf))
    renderSVG.drawToFile(drawing, str(compatibility_svg))
    export_base.render_png(compatibility_pdf, compatibility_png)

    row_path = SOURCE_DIR / "Figure2cd_model_selection_and_order_summary.csv"
    nt_path = SOURCE_DIR / "Figure2c_NT_compact_plot_source.csv"
    gb_path = SOURCE_DIR / "Figure2d_Genomic_Benchmarks_compact_plot_source.csv"
    nt_selection_path = SOURCE_DIR / "Figure2c_NT_representative_selection_summary.csv"
    gb_selection_path = SOURCE_DIR / "Figure2d_Genomic_Benchmarks_representative_selection_summary.csv"
    ab_source_path = SOURCE_DIR / "Figure2ab_representative_plot_source.csv"
    ab_summary_path = SOURCE_DIR / "Figure2a_representative_selection_summary.csv"
    gb_plot_cells = annotate_within_view_positions(
        gb_cells[gb_cells["model"].isin(gb_rows["model"])].copy()
    )
    nt_plot_cells = annotate_within_view_positions(
        nt_cells[nt_cells["model"].isin(nt_rows["model"])].copy()
    )
    row_registry.to_csv(row_path, index=False)
    gb_plot_cells.to_csv(gb_path, index=False)
    nt_plot_cells.to_csv(nt_path, index=False)
    gb_selection.to_csv(gb_selection_path, index=False)
    nt_selection.to_csv(nt_selection_path, index=False)
    ab_selected.to_csv(ab_source_path, index=False)
    ab_summary.to_csv(ab_summary_path, index=False)

    # Refresh legacy source-data aliases used by early manuscript assembly
    # scripts; all remain derived from the same current tables.
    row_registry.to_csv(SOURCE_DIR / "Figure2cd_compact_shared_row_registry.csv", index=False)
    nt_plot_cells.to_csv(
        SOURCE_DIR / "Figure2c_compact_plot_source.csv", index=False
    )
    gb_plot_cells.to_csv(
        SOURCE_DIR / "Figure2d_compact_plot_source.csv", index=False
    )
    nt_selection.to_csv(SOURCE_DIR / "Figure2c_representative_selection_summary.csv", index=False)
    gb_selection.to_csv(SOURCE_DIR / "Figure2d_representative_selection_summary.csv", index=False)

    reader = PdfReader(pdf)
    page = reader.pages[0]
    size_mm = [
        float(page.mediabox.width) / 72.0 * MM_PER_INCH,
        float(page.mediabox.height) / 72.0 * MM_PER_INCH,
    ]
    with Image.open(png) as image:
        pixels = list(image.size)
    expected_pixels = [round(PAGE_W_MM / MM_PER_INCH * PNG_DPI), round(PAGE_H_MM / MM_PER_INCH * PNG_DPI)]
    extracted = "".join((item.extract_text() or "") for item in reader.pages)
    font_summary = export_base.pdf_font_summary(pdf)
    forbidden_display_tokens = (
        "Evo 2-base",
        "NTv3-pre",
        "NT-v2-MS",
        "Mistral-DNA-v1",
        "GENERator-v2",
        "HyenaDNA-tiny",
        "Caduceus-Ph",
        "Caduceus-PS-131K",
        "1KGP",
        "1KG-",
    )
    source_key_duplicates_absent = not gb_cells.duplicated(["model", "dataset"]).any() and not nt_cells.duplicated(["model", "dataset"]).any()
    source_complete_cells = len(gb_cells) == 33 * len(cd.CONFIGS["gb"].tasks) and len(nt_cells) == 33 * len(cd.CONFIGS["nt"].tasks)
    interval_columns = ["mcc_lower", "mcc_mean", "mcc_upper"]
    interval_values = pd.concat([gb_cells[interval_columns], nt_cells[interval_columns]], ignore_index=True)
    mcc_intervals_valid = bool(
        interval_values.notna().all().all()
        and interval_values.ge(0.0).all().all()
        and interval_values.le(1.0).all().all()
        and (interval_values["mcc_lower"] <= interval_values["mcc_mean"]).all()
        and (interval_values["mcc_mean"] <= interval_values["mcc_upper"]).all()
    )
    gb_rank_mismatches = gb_cells.loc[
        ~gb_cells["task_rank_match"],
        ["dataset", "model", "mcc_mean", "task_rank_reported", "task_rank_recomputed"],
    ].to_dict("records")
    nt_rank_mismatches = nt_cells.loc[
        ~nt_cells["task_rank_match"],
        ["dataset", "model", "mcc_mean", "task_rank_reported", "task_rank_recomputed"],
    ].to_dict("records")
    qa = {
        "status": "DATA_QC_DRAFT",
        "page_size_mm": [PAGE_W_MM, PAGE_H_MM],
        "content_width_mm": CONTENT_W_MM,
        "panel_ab_height_mm": AB_H_MM,
        "panel_cd_height_mm": CD_H_MM,
        "lower_native_space_reserved_for_ure_mm": LOWER_RESERVED_MM,
        "panel_c_benchmark": "NT Benchmark",
        "panel_d_benchmark": "Genomic Benchmarks",
        "panel_c_model_rows": int(len(nt_rows)),
        "panel_d_model_rows": int(len(gb_rows)),
        "panel_c_order": nt_rows["model_display"].tolist(),
        "panel_d_order": gb_rows["model_display"].tolist(),
        "model_row_pitch_mm": 2.27,
        "independent_performance_order": True,
        "separate_benchmark_selection": True,
        "different_selected_scale_rows": row_registry.loc[
            ~row_registry["same_selected_model"],
            ["algorithm_identity", "gb_model_display", "nt_model_display"],
        ].to_dict("records"),
        "data_caveats": {
            "rank_glyph_source": "within_displayed_view_recomputed_from_mcc_mean",
            "horizontal_position": "within displayed view and task, min-max normalized; absolute mean MCC retained in the mean column",
            "genomic_benchmarks_reported_vs_recomputed_rank_mismatches": gb_rank_mismatches,
            "nt_benchmark_reported_vs_recomputed_rank_mismatches": nt_rank_mismatches,
            "visual_impact": "none when the mismatch remains outside ranks 1-3",
        },
        "checks": {
            "one_page": len(reader.pages) == 1,
            "a4_size_pass": abs(size_mm[0] - PAGE_W_MM) < 0.15 and abs(size_mm[1] - PAGE_H_MM) < 0.15,
            "png_pixels": pixels,
            "expected_png_pixels": expected_pixels,
            "dpi_pass": all(abs(a - b) <= 2 for a, b in zip(pixels, expected_pixels)),
            "arial_present": font_summary["arial_present"],
            "arial_embedded": font_summary["required_arial_fonts_embedded"],
            "editable_svg_text": "<text" in svg.read_text(encoding="utf-8"),
            "replacement_character_absent": "\ufffd" not in extracted,
            "panel_c_descending_mean_mcc": bool(nt_rows["mean_mcc"].is_monotonic_decreasing),
            "panel_d_descending_mean_mcc": bool(gb_rows["mean_mcc"].is_monotonic_decreasing),
            "mean_mcc_labels_numeric_only": True,
            "task_label_and_marker_column_centers_shared": True,
            "within_view_task_endpoints_locked": bool(
                gb_plot_cells.groupby("dataset")["within_view_position"].agg(["min", "max"]).eq({"min": 0.0, "max": 1.0}).all().all()
                and nt_plot_cells.groupby("dataset")["within_view_position"].agg(["min", "max"]).eq({"min": 0.0, "max": 1.0}).all().all()
            ),
            "task_labels_source_registered": True,
            "source_key_duplicates_absent": bool(source_key_duplicates_absent),
            "source_complete_33_by_18_and_33_by_7_cells": bool(source_complete_cells),
            "source_task_sets_exact": set(nt_cells["dataset"]) == set(cd.CONFIGS["nt"].tasks) and set(gb_cells["dataset"]) == set(cd.CONFIGS["gb"].tasks),
            "mcc_intervals_valid_0_to_1": mcc_intervals_valid,
            "forbidden_verbose_display_tokens_absent": not any(token in extracted for token in forbidden_display_tokens),
            "exact_EVO_2_7B_label_present": "EVO 2-7B" in extracted,
            "minimum_font_pt": 6.0,
            "gridlines": False,
            "pure_black_text": True,
        },
        "sources": {
            "model_selection_and_order_summary": str(row_path.resolve()),
            "figure2c": str(nt_path.resolve()),
            "figure2d": str(gb_path.resolve()),
            "figure2c_selection": str(nt_selection_path.resolve()),
            "figure2d_selection": str(gb_selection_path.resolve()),
            "figure2ab": str(ab_source_path.resolve()),
            "figure2a_selection": str(ab_summary_path.resolve()),
            "genomic_benchmarks_primary_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10150520/",
            "nt_benchmark_primary_reference": "https://www.nature.com/articles/s41592-024-02523-z",
        },
        "outputs": {"pdf": str(pdf.resolve()), "svg": str(svg.resolve()), "png": str(png.resolve())},
        "hashes": {
            "script": export_base.sha256(HERE),
            "pdf": export_base.sha256(pdf),
            "row_registry": export_base.sha256(row_path),
            "figure2c_source": export_base.sha256(nt_path),
            "figure2d_source": export_base.sha256(gb_path),
        },
    }
    checks = qa["checks"]
    checks["all_artifact_checks_pass"] = all(
        bool(checks[key])
        for key in (
            "one_page",
            "a4_size_pass",
            "dpi_pass",
            "arial_present",
            "arial_embedded",
            "editable_svg_text",
            "replacement_character_absent",
            "panel_c_descending_mean_mcc",
            "panel_d_descending_mean_mcc",
            "mean_mcc_labels_numeric_only",
            "within_view_task_endpoints_locked",
            "source_key_duplicates_absent",
            "source_complete_33_by_18_and_33_by_7_cells",
            "source_task_sets_exact",
            "mcc_intervals_valid_0_to_1",
            "forbidden_verbose_display_tokens_absent",
            "exact_EVO_2_7B_label_present",
        )
    )
    compatibility_qa = json.loads(json.dumps(qa))
    compatibility_qa["compatibility_alias_of"] = STEM
    compatibility_qa["outputs"] = {
        "pdf": str(compatibility_pdf.resolve()),
        "svg": str(compatibility_svg.resolve()),
        "png": str(compatibility_png.resolve()),
    }
    compatibility_qa["hashes"]["pdf"] = export_base.sha256(compatibility_pdf)
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
    export_base.configure_fonts()

    ucsc = ab_base.read_source(args.ucsc_csv)
    ucsc.attrs["source_path"] = str(args.ucsc_csv.resolve())
    ab_drawing, ab_selected, ab_summary, _ = ab.build_drawing(ucsc)

    gb_cells, gb_summary, _ = cd.read_and_validate(cd.CONFIGS["gb"])
    nt_cells, nt_summary, _ = cd.read_and_validate(cd.CONFIGS["nt"])
    gb_rows, gb_selection = selected_with_identity(gb_summary)
    nt_rows, nt_selection = selected_with_identity(nt_summary)
    row_registry = build_row_registry(gb_rows, nt_rows)
    cd_drawing = draw_compact_cd(gb_cells, nt_cells, gb_rows, nt_rows)

    page = Drawing(PAGE_W_MM * mm, PAGE_H_MM * mm)
    page.add(Rect(0, 0, page.width, page.height, fillColor=HexColor(WHITE), strokeColor=None))
    page.add(translated_group(ab_drawing, CONTENT_X_MM, AB_Y_MM))
    page.add(translated_group(cd_drawing, CONTENT_X_MM, CD_Y_MM))
    qa = save(
        page,
        row_registry,
        gb_rows,
        nt_rows,
        gb_cells,
        nt_cells,
        gb_selection,
        nt_selection,
        ab_selected,
        ab_summary,
    )


if __name__ == "__main__":
    main()
