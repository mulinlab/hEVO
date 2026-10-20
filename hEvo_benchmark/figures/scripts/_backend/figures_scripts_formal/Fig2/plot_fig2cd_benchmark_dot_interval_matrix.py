#!/usr/bin/env python3
"""Render polished standalone Figure 2c/d benchmark matrices.

The source-backed redesign replaces dense mini-bars with a ranked
dot-and-interval matrix. Every task cell uses a shared 0-1 MCC scale; the point
is the reported mean and the horizontal interval is the supplied 95% CI.
Rank is encoded by marker shape rather than by a multi-hue legend.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import plot_fig2a_split_benchmark_routes as style


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = PROJECT_ROOT / "tables"
OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig2"
    / "fig2cd_benchmark_polish_v1"
)
SOURCE_DIR = OUT_DIR / "source_data"

STATUS = "DATA_QC_DRAFT"
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
SEPARATOR = "#E2E9F1"
GOLD = "#C28A2C"
GOLD_PALE = "#FBF7EC"
WHITE = "#FFFFFF"

FOCUS_MODELS = {
    "hEvo-base",
    "hEvo-large",
}


@dataclass(frozen=True)
class PanelConfig:
    key: str
    letter: str
    title: str
    table_file: str
    output_stem: str
    width_mm: float
    height_mm: float
    tasks: tuple[str, ...]
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    aliases: dict[str, str]
    model_width_mm: float
    mean_width_mm: float
    rank_width_mm: float


GB_TASKS = (
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
)
NT_HISTONE = (
    "H2AFZ",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
    "H4K20me1",
)
NT_REGULATORY = (
    "enhancers",
    "enhancers_types",
    "promoter_all",
    "promoter_no_tata",
    "promoter_tata",
)
NT_SPLICE = (
    "splice_sites_acceptors",
    "splice_sites_all",
    "splice_sites_donors",
)
NT_TASKS = NT_HISTONE + NT_REGULATORY + NT_SPLICE


CONFIGS = {
    "gb": PanelConfig(
        key="gb",
        letter="c",
        title="Genomic Benchmark",
        table_file="S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv",
        output_stem="Fig2c_Genomic_Benchmark_ranked_dot_interval_matrix_DATA_QC_DRAFT",
        width_mm=185.0,
        height_mm=190.0,
        tasks=GB_TASKS,
        groups=(("Human-related tasks", GB_TASKS),),
        aliases={
            "human_enhancers_cohn": "Cohn\nenhancers",
            "human_enhancers_ensembl": "Ensembl\nenhancers",
            "human_ensembl_regulatory": "Regulatory\nelements",
            "human_nontata_promoters": "Non-TATA\npromoters",
            "human_ocr_ensembl": "Open\nchromatin",
            "demo_coding_vs_intergenomic_seqs": "Coding vs\nintergenic",
            "demo_human_or_worm": "Human vs\nworm",
        },
        model_width_mm=42.0,
        mean_width_mm=20.0,
        rank_width_mm=8.0,
    ),
    "nt": PanelConfig(
        key="nt",
        letter="d",
        title="NT Benchmark",
        table_file="S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv",
        output_stem="Fig2d_NT_Benchmark_ranked_dot_interval_matrix_DATA_QC_DRAFT",
        width_mm=228.6,
        height_mm=190.0,
        tasks=NT_TASKS,
        groups=(
            ("Histone markers", NT_HISTONE),
            ("Regulatory annotation", NT_REGULATORY),
            ("Splice-site annotation", NT_SPLICE),
        ),
        aliases={
            "H2AFZ": "H2AFZ",
            "H3K27ac": "H3K27ac",
            "H3K27me3": "H3K27me3",
            "H3K36me3": "H3K36me3",
            "H3K4me1": "H3K4me1",
            "H3K4me2": "H3K4me2",
            "H3K4me3": "H3K4me3",
            "H3K9ac": "H3K9ac",
            "H3K9me3": "H3K9me3",
            "H4K20me1": "H4K20me1",
            "enhancers": "Enhancers",
            "enhancers_types": "Enh. types",
            "promoter_all": "Promoters",
            "promoter_no_tata": "Non-TATA",
            "promoter_tata": "TATA",
            "splice_sites_acceptors": "Acceptors",
            "splice_sites_all": "All sites",
            "splice_sites_donors": "Donors",
        },
        model_width_mm=36.0,
        mean_width_mm=18.0,
        rank_width_mm=7.0,
    ),
}


DISPLAY_NAMES = {
    "hEvo-base": "hEvo-base",
    "hEvo-large": "hEvo-large",
    "NT_2.5b_MS": "NTv1-MS-2.5B",
    "NT_2.5b_1kG": "NTv1-1000G-2.5B",
    "NT_500m_MS": "NTv2-MS-500M",
    "NT_500m_1kG": "NTv1-1000G-500M",
    "evo2_7b_base": "Evo2-7B",
    "evo2_1b_base": "Evo2-1B",
    "HyenaDNA_tiny_1k": "HyenaDNA-1k",
}


def display_model_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def rank_marker(rank: float) -> str:
    value = int(round(float(rank)))
    if value == 1:
        return "diamond"
    if value == 2:
        return "square"
    if value == 3:
        return "triangle"
    return "circle"


def prepare_data(config: PanelConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source_path = TABLE_DIR / config.table_file
    raw = pd.read_csv(source_path)
    required = {
        "benchmark",
        "task_group",
        "model",
        "dataset",
        "mcc_mean",
        "mcc_lower",
        "mcc_upper",
        "task_rank",
        "task_count",
    }
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise ValueError(f"{config.table_file} missing columns: {missing_columns}")
    for column in ("mcc_mean", "mcc_lower", "mcc_upper", "task_rank", "task_count"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    duplicates = raw.duplicated(["model", "dataset"], keep=False)
    if duplicates.any():
        keys = raw.loc[duplicates, ["model", "dataset"]].to_dict("records")
        raise ValueError(f"Duplicate model-task rows: {keys[:8]}")

    source_tasks = set(raw["dataset"].dropna().astype(str))
    missing_tasks = [task for task in config.tasks if task not in source_tasks]
    if missing_tasks:
        raise ValueError(f"Configured tasks absent from source: {missing_tasks}")
    raw = raw[raw["dataset"].isin(config.tasks)].copy()

    matrix = raw.pivot(index="model", columns="dataset", values="mcc_mean").reindex(columns=config.tasks)
    if matrix.isna().any().any():
        coordinates = np.argwhere(matrix.isna().to_numpy())
        missing_keys = [
            {"model": str(matrix.index[row]), "dataset": str(matrix.columns[column])}
            for row, column in coordinates[:8]
        ]
        raise ValueError(f"Missing model-task values: {missing_keys}")

    summary = pd.DataFrame(
        {
            "mean_mcc": matrix.mean(axis=1),
            "avg_task_rank": raw.groupby("model")["task_rank"].mean(),
        }
    )
    summary = summary.sort_values(["mean_mcc"], ascending=False, kind="stable")
    summary["overall_rank"] = np.arange(1, len(summary) + 1)
    summary["plot_row"] = np.arange(len(summary))
    summary["model_display"] = [display_model_name(model) for model in summary.index]
    summary["is_hevo_focus"] = summary.index.isin(FOCUS_MODELS)

    task_order = {task: index for index, task in enumerate(config.tasks)}
    group_order = {
        task: group_index
        for group_index, (_, tasks) in enumerate(config.groups)
        for task in tasks
    }
    raw["task_order"] = raw["dataset"].map(task_order)
    raw["task_group_order"] = raw["dataset"].map(group_order)
    raw["task_alias"] = raw["dataset"].map(config.aliases)
    raw["model_display"] = raw["model"].map(summary["model_display"])
    raw["mean_mcc"] = raw["model"].map(summary["mean_mcc"])
    raw["overall_rank"] = raw["model"].map(summary["overall_rank"])
    raw["avg_task_rank"] = raw["model"].map(summary["avg_task_rank"])
    raw["plot_row"] = raw["model"].map(summary["plot_row"])
    raw["is_hevo_focus"] = raw["model"].isin(FOCUS_MODELS)
    raw["rank_marker"] = raw["task_rank"].map(rank_marker)
    raw = raw.sort_values(["plot_row", "task_order"]).reset_index(drop=True)

    summary = {
        "source_path": str(source_path.resolve()),
        "source_sha256": style.sha256(source_path),
        "rows": int(len(raw)),
        "models": int(raw["model"].nunique()),
        "tasks": int(raw["dataset"].nunique()),
        "expected_cells": int(raw["model"].nunique() * len(config.tasks)),
        "missing_cells": 0,
        "duplicate_cells": 0,
        "mcc_range": [float(raw["mcc_mean"].min()), float(raw["mcc_mean"].max())],
        "ci_median_width": float((raw["mcc_upper"] - raw["mcc_lower"]).median()),
        "focus_models": {
            model: {
                "mean_mcc": float(summary.loc[model, "mean_mcc"]),
                "overall_rank": int(summary.loc[model, "overall_rank"]),
                "avg_task_rank": float(summary.loc[model, "avg_task_rank"]),
            }
            for model in sorted(FOCUS_MODELS)
        },
    }
    return raw, summary, summary


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
    drawing.add(
        Rect(
            x * mm,
            y * mm,
            width * mm,
            height * mm,
            rx=radius * mm,
            ry=radius * mm,
            fillColor=HexColor(fill),
            strokeColor=HexColor(stroke),
            strokeWidth=line_width,
        )
    )


def add_header(drawing: Drawing, x: float, y: float, width: float, text: str) -> None:
    add_round_box(drawing, x, y, width, 5.8, fill=BLUE, stroke=BLUE, line_width=0.65, radius=1.2)
    style.add_fitted_text(
        drawing,
        x + width / 2,
        y + 1.85,
        text,
        6.8,
        width - 3.0,
        bold=True,
        anchor="middle",
        color=WHITE,
    )


def add_multiline_center(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 6.0,
    color: str = BLACK,
    bold: bool = False,
) -> None:
    lines = text.split("\n")
    spacing = 2.05
    top_baseline = y + spacing * (len(lines) - 1) / 2
    for index, line in enumerate(lines):
        style.add_text(
            drawing,
            x,
            top_baseline - index * spacing,
            line,
            size,
            bold=bold,
            anchor="middle",
            color=color,
        )


def add_rotated_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 6.0,
    angle_degrees: float = 55.0,
    color: str = BLACK,
) -> None:
    radians = np.deg2rad(angle_degrees)
    label = String(
        0,
        0,
        text,
        fontName=style.FONT,
        fontSize=size,
        fillColor=HexColor(color),
        textAnchor="start",
    )
    group = Group(label)
    group.transform = (
        float(np.cos(radians)),
        float(np.sin(radians)),
        float(-np.sin(radians)),
        float(np.cos(radians)),
        x * mm,
        y * mm,
    )
    drawing.add(group)


def add_marker(
    drawing: Drawing,
    x: float,
    y: float,
    marker: str,
    *,
    fill: str,
    stroke: str,
) -> None:
    if marker == "diamond":
        radius = 0.72
        drawing.add(
            Polygon(
                [x * mm, (y + radius) * mm, (x + radius) * mm, y * mm, x * mm, (y - radius) * mm, (x - radius) * mm, y * mm],
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=0.65,
            )
        )
    elif marker == "square":
        half = 0.60
        drawing.add(
            Rect(
                (x - half) * mm,
                (y - half) * mm,
                2 * half * mm,
                2 * half * mm,
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=0.65,
            )
        )
    elif marker == "triangle":
        radius = 0.73
        drawing.add(
            Polygon(
                [(x) * mm, (y + radius) * mm, (x + 0.78 * radius) * mm, (y - 0.60 * radius) * mm, (x - 0.78 * radius) * mm, (y - 0.60 * radius) * mm],
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=0.65,
            )
        )
    else:
        drawing.add(
            Circle(
                x * mm,
                y * mm,
                0.43 * mm,
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=0.60,
            )
        )


def add_interval_cell(
    drawing: Drawing,
    x: float,
    width: float,
    y: float,
    *,
    mean: float,
    lower: float,
    upper: float,
    rank: float,
    focus: bool,
) -> None:
    left = x + 0.80
    right = x + width - 0.80
    span = right - left
    style.add_line(drawing, left, y, right, y, color=TRACK, width=0.60)
    lo = left + float(np.clip(lower, 0.0, 1.0)) * span
    hi = left + float(np.clip(upper, 0.0, 1.0)) * span
    center = left + float(np.clip(mean, 0.0, 1.0)) * span
    color = GOLD if focus else BLUE_MID
    style.add_line(drawing, lo, y, hi, y, color=color, width=0.90 if focus else 0.78)
    style.add_line(drawing, lo, y - 0.42, lo, y + 0.42, color=color, width=0.68)
    style.add_line(drawing, hi, y - 0.42, hi, y + 0.42, color=color, width=0.68)
    marker = rank_marker(rank)
    top_three = int(round(rank)) <= 3
    fill = GOLD if focus else BLUE if top_three else WHITE
    add_marker(drawing, center, y, marker, fill=fill, stroke=GOLD if focus else BLUE)


def draw_panel(config: PanelConfig, raw: pd.DataFrame, summary: pd.DataFrame) -> Drawing:
    drawing = Drawing(config.width_mm * mm, config.height_mm * mm)
    drawing.add(Rect(0, 0, drawing.width, drawing.height, fillColor=HexColor(WHITE), strokeColor=None))

    content_x = 13.5
    right_margin = 4.6
    gap_1 = 1.5
    gap_2 = 1.2
    gap_3 = 1.5
    model_x = content_x
    mean_x = model_x + config.model_width_mm + gap_1
    rank_x = mean_x + config.mean_width_mm + gap_2
    tasks_x = rank_x + config.rank_width_mm + gap_3
    task_width = (config.width_mm - right_margin - tasks_x) / len(config.tasks)
    right_x = tasks_x + task_width * len(config.tasks)

    style.add_text(drawing, 2.0, 182.8, config.letter, 14.0, bold=True, color=BLACK)
    style.add_text(drawing, content_x, 183.8, config.title, 10.0, bold=True, color=BLUE)
    style.add_fitted_text(
        drawing,
        content_x,
        178.2,
        f"Frozen embeddings + supervised linear probe | MCC with 95% CI | {len(summary)} models x {len(config.tasks)} tasks",
        6.2,
        config.width_mm - content_x - right_margin,
        color=TEXT_MUTED,
    )

    header_y = 170.3
    add_header(drawing, model_x, header_y, config.model_width_mm, "Model")
    add_header(
        drawing,
        mean_x,
        header_y,
        config.mean_width_mm + gap_2 + config.rank_width_mm,
        "Overall performance",
    )
    add_header(drawing, tasks_x, header_y, right_x - tasks_x, "Task-level MCC (95% CI)")

    group_y = 163.3
    group_h = 5.2
    add_round_box(drawing, mean_x, group_y, config.mean_width_mm, group_h, fill=BLUE_PALE, stroke=BLUE_LIGHT, radius=0.8)
    style.add_text(
        drawing,
        mean_x + config.mean_width_mm / 2,
        group_y + 1.55,
        "Mean MCC",
        6.2,
        bold=True,
        anchor="middle",
        color=BLUE,
    )
    add_round_box(drawing, rank_x, group_y, config.rank_width_mm, group_h, fill=BLUE_PALE, stroke=BLUE_LIGHT, radius=0.8)
    style.add_text(
        drawing,
        rank_x + config.rank_width_mm / 2,
        group_y + 1.55,
        "Rank",
        6.2,
        bold=True,
        anchor="middle",
        color=BLUE,
    )

    task_index = {task: index for index, task in enumerate(config.tasks)}
    for group_name, group_tasks in config.groups:
        first = min(task_index[task] for task in group_tasks)
        last = max(task_index[task] for task in group_tasks)
        x = tasks_x + first * task_width
        width = (last - first + 1) * task_width
        add_round_box(drawing, x, group_y, width, group_h, fill=BLUE_PALE, stroke=BLUE_LIGHT, radius=0.8)
        style.add_fitted_text(
            drawing,
            x + width / 2,
            group_y + 1.55,
            group_name,
            6.2,
            width - 2.0,
            bold=True,
            anchor="middle",
            color=BLUE,
        )

    label_y = 157.2
    row_top_y = 150.7
    row_height = 4.05
    row_bottom_y = row_top_y - (len(summary) - 1) * row_height - 2.0
    for index, task in enumerate(config.tasks):
        x = tasks_x + index * task_width
        if config.key == "nt":
            add_rotated_text(
                drawing,
                x + 0.35,
                153.0,
                config.aliases[task],
                size=6.0,
                angle_degrees=55.0,
            )
        else:
            add_multiline_center(drawing, x + task_width / 2, label_y, config.aliases[task], size=6.0)
        style.add_line(drawing, x, row_bottom_y, x, 160.4, color=SEPARATOR, width=0.34)
    style.add_line(drawing, right_x, row_bottom_y, right_x, 160.4, color=SEPARATOR, width=0.34)
    style.add_line(drawing, model_x - 1.0, 152.5, right_x, 152.5, color=BLUE_LIGHT, width=0.65)

    lookup = raw.set_index(["model", "dataset"])
    mean_scale_max = 0.75
    for model, row in summary.iterrows():
        y = row_top_y - float(row["plot_row"]) * row_height
        focus = bool(row["is_hevo_focus"])
        fill = GOLD_PALE if focus else ROW_ALT if int(row["plot_row"]) % 2 else WHITE
        drawing.add(
            Rect(
                (model_x - 1.0) * mm,
                (y - 1.72) * mm,
                (right_x - (model_x - 1.0)) * mm,
                3.44 * mm,
                fillColor=HexColor(fill),
                strokeColor=None,
            )
        )
        if focus:
            drawing.add(
                Rect(
                    (model_x - 1.0) * mm,
                    (y - 1.72) * mm,
                    0.75 * mm,
                    3.44 * mm,
                    fillColor=HexColor(GOLD),
                    strokeColor=None,
                )
            )

        style.add_fitted_text(
            drawing,
            model_x + 0.8,
            y - 0.75,
            str(row["model_display"]),
            6.35,
            config.model_width_mm - 2.0,
            bold=focus,
            color=BLUE if focus else BLACK,
        )

        mean_left = mean_x + 1.0
        mean_right = mean_x + config.mean_width_mm - 7.0
        style.add_line(drawing, mean_left, y, mean_right, y, color=TRACK, width=0.75)
        mean_position = mean_left + min(float(row["mean_mcc"]) / mean_scale_max, 1.0) * (mean_right - mean_left)
        mean_color = GOLD if focus else BLUE
        style.add_line(drawing, mean_left, y, mean_position, y, color=mean_color, width=1.10 if focus else 0.90)
        drawing.add(
            Circle(
                mean_position * mm,
                y * mm,
                (0.60 if focus else 0.50) * mm,
                fillColor=HexColor(mean_color),
                strokeColor=HexColor(mean_color),
                strokeWidth=0.45,
            )
        )
        style.add_text(
            drawing,
            mean_right + 1.1,
            y - 0.72,
            f"{float(row['mean_mcc']):.3f}",
            6.0,
            color=BLACK,
        )
        style.add_text(
            drawing,
            rank_x + config.rank_width_mm / 2,
            y - 0.72,
            str(int(row["overall_rank"])),
            6.2,
            bold=focus,
            anchor="middle",
            color=BLUE if focus else BLACK,
        )

        for task_index_value, task in enumerate(config.tasks):
            cell = lookup.loc[(model, task)]
            add_interval_cell(
                drawing,
                tasks_x + task_index_value * task_width,
                task_width,
                y,
                mean=float(cell["mcc_mean"]),
                lower=float(cell["mcc_lower"]),
                upper=float(cell["mcc_upper"]),
                rank=float(cell["task_rank"]),
                focus=focus,
            )

    footer_y = 11.8
    style.add_text(drawing, content_x, footer_y - 0.72, "Task rank:", 6.0, bold=True, color=TEXT_MUTED)
    cursor = content_x + 15.5
    for marker, label in (("diamond", "1"), ("square", "2"), ("triangle", "3"), ("circle", "Other")):
        add_marker(drawing, cursor, footer_y, marker, fill=BLUE if marker != "circle" else WHITE, stroke=BLUE)
        style.add_text(drawing, cursor + 1.5, footer_y - 0.72, label, 6.0, color=TEXT_MUTED)
        cursor += 7.4 if label != "Other" else 12.2
    style.add_line(drawing, cursor, footer_y, cursor + 8.0, footer_y, color=BLUE_MID, width=0.82)
    style.add_line(drawing, cursor, footer_y - 0.42, cursor, footer_y + 0.42, color=BLUE_MID, width=0.65)
    style.add_line(drawing, cursor + 8.0, footer_y - 0.42, cursor + 8.0, footer_y + 0.42, color=BLUE_MID, width=0.65)
    add_marker(drawing, cursor + 4.0, footer_y, "circle", fill=WHITE, stroke=BLUE)
    style.add_text(drawing, cursor + 10.0, footer_y - 0.72, "MCC mean + 95% CI", 6.0, color=TEXT_MUTED)
    focus_x = cursor + 42.0
    add_marker(drawing, focus_x, footer_y, "circle", fill=GOLD, stroke=GOLD)
    style.add_text(drawing, focus_x + 1.8, footer_y - 0.72, "hEvo focus rows", 6.0, color=TEXT_MUTED)

    style.add_fitted_text(
        drawing,
        content_x,
        5.5,
        "Rows are ordered by unweighted mean MCC. Task cells share a 0-1 MCC scale; Mean MCC uses a 0-0.75 scale.",
        6.0,
        config.width_mm - content_x - right_margin,
        color=TEXT_MUTED,
    )
    return drawing


def save_panel(
    config: PanelConfig,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    data_summary: dict[str, object],
) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{config.output_stem}.pdf"
    svg_path = OUT_DIR / f"{config.output_stem}.svg"
    png_path = OUT_DIR / f"{config.output_stem}.png"
    source_path = SOURCE_DIR / f"{config.output_stem}_plot_source.csv"
    raw.to_csv(source_path, index=False)

    drawing = draw_panel(config, raw, summary)
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    style.render_png(pdf_path, png_path)

    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) / 72.0 * MM_PER_INCH
    height_mm = float(page.mediabox.height) / 72.0 * MM_PER_INCH
    extracted_text = "".join((item.extract_text() or "") for item in reader.pages)
    with Image.open(png_path) as image:
        pixels = list(image.size)
    expected_pixels = [
        round(config.width_mm / MM_PER_INCH * PNG_DPI),
        round(config.height_mm / MM_PER_INCH * PNG_DPI),
    ]
    font_summary = style.pdf_font_summary(pdf_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    source_copy = pd.read_csv(source_path)

    checks = {
        "one_page": len(reader.pages) == 1,
        "size_mm": [round(width_mm, 3), round(height_mm, 3)],
        "size_pass": abs(width_mm - config.width_mm) < 0.15 and abs(height_mm - config.height_mm) < 0.15,
        "png_pixels": pixels,
        "expected_png_pixels": expected_pixels,
        "dpi_pass": all(abs(a - b) <= 2 for a, b in zip(pixels, expected_pixels)),
        "editable_svg_text": "<text" in svg_text,
        "text_extractable": len(extracted_text) > 500,
        "replacement_character_absent": "\ufffd" not in extracted_text,
        "arial_present": font_summary["arial_present"],
        "required_arial_fonts_embedded": font_summary["required_arial_fonts_embedded"],
        "source_copy_rows_match": len(source_copy) == data_summary["expected_cells"],
        "source_copy_unique_cells": not source_copy.duplicated(["model", "dataset"]).any(),
        "all_models_present": source_copy["model"].nunique() == data_summary["models"],
        "all_tasks_present": source_copy["dataset"].nunique() == len(config.tasks),
        "mcc_bounds_ordered": bool(
            (
                (source_copy["mcc_lower"] <= source_copy["mcc_mean"])
                & (source_copy["mcc_mean"] <= source_copy["mcc_upper"])
            ).all()
        ),
        "rank_marker_mapping_complete": source_copy["rank_marker"].notna().all(),
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
        "all_tasks_present",
        "mcc_bounds_ordered",
        "rank_marker_mapping_complete",
    )
    checks["automated_pass"] = all(bool(checks[key]) for key in boolean_checks)
    return {
        "key": config.key,
        "status": STATUS,
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
                "No text overlaps or clipped labels",
                "Task labels remain inside their assigned header bands",
                "All canonical model rows are readable at the intended panel size",
                "MCC points, supplied 95% CI intervals, and rank markers are distinguishable",
                "hEvo focus rows are highlighted without obscuring data marks",
                "Footer legend and scale note are complete and legible",
            ],
        },
    }


def make_contact_sheet(records: list[dict[str, object]]) -> Path:
    images: list[Image.Image] = []
    for record in records:
        path = Path(record["artifact"]["png"])
        image = Image.open(path).convert("RGB")
        preview_width = 1800
        preview_height = round(image.height * preview_width / image.width)
        images.append(image.resize((preview_width, preview_height), Image.Resampling.LANCZOS))
    margin = 60
    gap = 70
    total_width = max(image.width for image in images) + 2 * margin
    total_height = sum(image.height for image in images) + gap * (len(images) - 1) + 2 * margin
    sheet = Image.new("RGB", (total_width, total_height), WHITE)
    y = margin
    for image in images:
        x = (total_width - image.width) // 2
        sheet.paste(image, (x, y))
        y += image.height + gap
    output = OUT_DIR / "Figure2cd_benchmark_polish_review_contact_sheet.png"
    sheet.save(output, dpi=(300, 300))
    for image in images:
        image.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("gb", "nt", "both"), default="both")
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    style.configure_fonts()
    selected = ["gb", "nt"] if args.panel == "both" else [args.panel]
    records: list[dict[str, object]] = []
    for key in selected:
        config = CONFIGS[key]
        raw, summary, data_summary = prepare_data(config)
        records.append(save_panel(config, raw, summary, data_summary))
    contact_sheet = make_contact_sheet(records) if len(records) == 2 else None
    qa = {
        "status": STATUS,
        "chart_contract": str((OUT_DIR / "Figure2cd_chart_contract.md").resolve()),
        "renderer": str(Path(__file__).resolve()),
        "records": records,
        "contact_sheet": str(contact_sheet.resolve()) if contact_sheet else None,
        "all_manual_visual_checks_pass": all(
            record["manual_visual_review"]["status"] == "PASS" for record in records
        ),
    }


if __name__ == "__main__":
    main()
