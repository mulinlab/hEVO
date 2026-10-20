#!/usr/bin/env python3
"""Render the revised Figure 3 main page and ClinVar stratum supplement.

Main layout:
  a-b  conceptual variant-architecture maps
  c-e  ClinVar all, CausalDB disease, rare causal variants
  f-h  UKB rare variants, DDD de novo, MPRA
  i-k  temporal ClinVar reclassification

ClinVar coding and noncoding are moved to a standalone supplementary pair.
All quantitative benchmark panels preserve the locked source rows and values.
"""

from __future__ import annotations

# === Comment index ===
# [Role] Current A4 integration entry point for Fig. 3, producing panels a–k and the ClinVar supplement; v2 adds revised layout and data summaries to the v1 core.
# [Input] S26–S39, C1–C8, and archived placement/sampling summaries; the core provides base data and plotting implementations.
# [Layout] Build architecture, benchmark, and temporal/variant panels, then arrange them using MAIN_W_MM × MAIN_H_MM millimeter coordinates.
# [Style] This file controls DATASET_COLORS, DISPLAY_Y, family colors, and Arial/Liberation Sans; dataset colors are separate from model-family colors.
# [Export] main() writes the main and supplementary figures, PDF/SVG/PNG files, source data, captions/summaries, and QA JSON.

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Path as RLPath, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics

import plot_figure3_abcdefghijk_a4_integrated_v1 as core


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig3" / "figure3_abcdefghijk_a4_integrated_v11"
SOURCE_DIR = OUT_DIR / "source_data"
ARCH_SOURCE_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig3"
    / "fig3_variant_architecture_maps_v1"
    / "source_data"
)
ARCH_PLACEMENT_SOURCE = ARCH_SOURCE_DIR / "variant_map_placements.csv"
ARCH_summary_SOURCE = ARCH_SOURCE_DIR / "variant_benchmark_sampling_summary.csv"

MAIN_STEM = "Figure3_abcdefghijk_A4_integrated_v11_DATA_QC_DRAFT"
SUP_STEM = "FigureS3_ClinVar_coding_noncoding_v5_DATA_QC_DRAFT"
MAIN_W_MM = 210.0
MAIN_H_MM = 297.0
SUP_W_MM = 183.0
SUP_H_MM = 80.0
PNG_DPI = 600

BLACK = core.BLACK
WHITE = core.WHITE
HEVO = core.HEVO
NT = core.NT
EVO2 = core.EVO2
GPN = core.GPN
GPN_STAR = "#5B4E9B"
core.FAMILY_COLORS["GPN-Star"] = GPN_STAR
NEUTRAL = core.NEUTRAL
HEVO_WASH = core.HEVO_WASH
PALE_NEUTRAL = "#F3F5F7"
PALE_ORANGE = "#F7EEE8"
PALE_TEAL = "#EAF3F3"
PALE_BLUE = "#EEF2F7"

# Independent benchmark palette: severe clinical/developmental effects progress
# from red through ochre to green for molecular-function assays. These hues do
# not encode model family and remain separate from the bar palette.
DATASET_COLORS = {
    "ClinVar": "#D44A37",
    "CAUSALdb GWAS": "#6F9B45",
    "CAUSALdb rare": "#C49A00",
    "MPRA": "#16856B",
    "UKB rare": "#E07B26",
    "DDD de novo": "#A62A2A",
}

DATASET_LABELS = {
    "ClinVar": "ClinVar variants: pathogenic vs. benign",
    "CAUSALdb GWAS": "GWAS fine-mapped variants: candidate causal vs. low-effect controls",
    "CAUSALdb rare": "Rare GWAS fine-mapped variants: candidate causal vs. rare GWAS controls",
    "MPRA": "Regulatory variants (MPRA): with vs. without detected effects",
    "UKB rare": "UKB rare variants: trait-associated vs. rare GWAS controls",
    "DDD de novo": "DDD de novo LoF/splice variants: vs. benign ClinVar controls",
}

# Vertical placements are intentionally separated to avoid overplotting.  They
# remain conceptual coordinates and are exported as display metadata.
DISPLAY_Y = {
    "a": {
        "DDD de novo": {"positive": -0.92, "negative": 0.08},
        "ClinVar": {"positive": -0.72, "negative": 0.04},
        # UKB likely-causal rare variants are expected to be more deleterious
        # than the broader rare fine-mapped set, but less severe than the
        # ClinVar pathogenic and DDD LoF/splice contrasts. The coordinate is
        # schematic and preserves that ordinal relationship only.
        "UKB rare": {"positive": -0.42, "negative": -0.02},
        "CAUSALdb rare": {"positive": -0.30, "negative": 0.03},
        "CAUSALdb GWAS": {"positive": -0.16, "negative": 0.07},
        # MPRA establishes molecular allelic function, not organismal fitness.
        # Slightly above the limited-impact band for visual separation only;
        # this is not evidence for adaptive organismal fitness.
        "MPRA": {"positive": 0.09, "negative": -0.06},
    },
    "b": {
        "DDD de novo": {"positive": 0.94, "negative": 0.10},
        "ClinVar": {"positive": 0.82, "negative": 0.12},
        "UKB rare": {"positive": 0.70, "negative": 0.14},
        "CAUSALdb rare": {"positive": 0.58, "negative": 0.16},
        "CAUSALdb GWAS": {"positive": 0.42, "negative": 0.18},
        "MPRA": {"positive": 0.28, "negative": 0.20},
    },
}

DATASET_IDS = {
    "ClinVar": 1,
    "CAUSALdb GWAS": 3,
    "CAUSALdb rare": 4,
    "UKB rare": 5,
    "DDD de novo": 2,
    "MPRA": 6,
}

# The map coordinates are schematic placements transcribed from the supplied
# Figure 3 layout. They describe category placement only; observed values and
# ranges remain in S22 and are not implied by these coordinates.
REFERENCE_ARCH_COORDS = {
    "a": {  # Fitness-effect landscape (source panel a)
        "ClinVar": {"positive": (0.21, 0.14), "negative": (0.24, 0.55)},
        "DDD de novo": {"positive": (0.06, 0.04), "negative": (0.16, 0.54)},
        "CAUSALdb GWAS": {"positive": (0.63, 0.42), "negative": (0.81, 0.49)},
        "CAUSALdb rare": {"positive": (0.39, 0.35), "negative": (0.36, 0.50)},
        "UKB rare": {"positive": (0.38, 0.29), "negative": (0.36, 0.45)},
        "MPRA": {"positive": (0.73, 0.545), "negative": (0.76, 0.44)},
    },
    "b": {  # Phenotypic effect and allele-frequency map (source panel b)
        "ClinVar": {"positive": (0.21, 0.82), "negative": (0.24, 0.12)},
        "DDD de novo": {"positive": (0.06, 0.94), "negative": (0.16, 0.10)},
        "CAUSALdb GWAS": {"positive": (0.63, 0.42), "negative": (0.81, 0.13)},
        "CAUSALdb rare": {"positive": (0.39, 0.58), "negative": (0.36, 0.20)},
        "UKB rare": {"positive": (0.38, 0.70), "negative": (0.36, 0.04)},
        "MPRA": {"positive": (0.73, 0.40), "negative": (0.76, 0.20)},
    },
}

PANEL_HEADERS = {
    "c": ("ClinVar", ("ClinVar variants", "pathogenic vs. benign")),
    "d": ("CAUSALdb GWAS", ("GWAS fine-mapped variants", "candidate causal vs. low-effect controls")),
    "e": ("CAUSALdb rare", ("Rare GWAS fine-mapped variants", "candidate causal vs. rare GWAS controls")),
    "f": ("UKB rare", ("UKB rare variants", "trait-associated vs. rare GWAS controls")),
    "g": ("DDD de novo", ("DDD de novo LoF/splice variants", "vs. benign ClinVar controls")),
    "h": ("MPRA", ("Regulatory variants (MPRA)", "with vs. without detected effects")),
}

# The a-b map legend uses the same two-line wording as panels c-h so readers
# can move between the conceptual atlas and quantitative benchmarks without
# translating aliases.
DATASET_LEGEND_LINES = {dataset: lines for _, (dataset, lines) in PANEL_HEADERS.items()}

def blend(color: str, fraction_white: float) -> str:
    color = color.lstrip("#")
    rgb = [int(color[index : index + 2], 16) for index in (0, 2, 4)]
    mixed = [round(value + (255 - value) * fraction_white) for value in rgb]
    return "#" + "".join(f"{value:02X}" for value in mixed)


def add_rotated_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 6.4,
) -> None:
    label = String(
        0,
        0,
        text,
        fontName="Arial",
        fontSize=size,
        textAnchor="middle",
        fillColor=HexColor(BLACK),
    )
    group = Group(label)
    group.transform = (0, 1, -1, 0, x * mm, y * mm)
    drawing.add(group)


def x_log(value: float, x0: float, x1: float) -> float:
    low, high = math.log10(1e-6), math.log10(5e-1)
    clipped = min(max(float(value), 1e-6), 5e-1)
    return x0 + (math.log10(clipped) - low) / (high - low) * (x1 - x0)


def y_arch(value: float, y0: float, y1: float, panel: str) -> float:
    if panel == "a":
        return y0 + (float(value) + 1.0) / 2.0 * (y1 - y0)
    return y0 + float(value) * (y1 - y0)


def read_architecture_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    placements = pd.read_csv(ARCH_PLACEMENT_SOURCE)
    summary = pd.read_csv(ARCH_summary_SOURCE)
    required = {
        "id",
        "dataset",
        "panel",
        "class",
        "allele_frequency",
        "range_low",
        "range_high",
        "range_status",
    }
    missing = sorted(required - set(placements.columns))
    if missing:
        raise ValueError(f"architecture placement table missing {missing}")
    if len(placements) != 24:
        raise ValueError("architecture placement table must contain 6 x 2 x 2 rows")

    display = placements.copy()
    display["source_panel"] = display["panel"]
    display["display_panel_letter"] = display["panel"].map({"a": "b", "b": "a"})
    display["source_allele_frequency"] = pd.to_numeric(display["allele_frequency"], errors="raise")
    display["display_allele_frequency"] = display["source_allele_frequency"]
    schematic_coordinates = [
        REFERENCE_ARCH_COORDS[str(row.panel)][str(row.dataset)][str(row["class"])]
        for _, row in display.iterrows()
    ]
    display["display_x_fraction"] = [coordinate[0] for coordinate in schematic_coordinates]
    display["display_y_fraction"] = [coordinate[1] for coordinate in schematic_coordinates]
    display["display_vertical_coordinate"] = [
        DISPLAY_Y[str(row.panel)][str(row.dataset)][str(row["class"])]
        for _, row in display.iterrows()
    ]
    ddd_negative = display["dataset"].eq("DDD de novo") & display["class"].eq("negative")
    # The ClinVar benign DDD controls have no retained AF.  Placing them at 1%
    # falsely implied an observed median; keep the marker near the de novo edge
    # and state explicitly that the position is schematic.
    display.loc[ddd_negative, "display_allele_frequency"] = 6e-6
    mpra_negative = display["dataset"].eq("MPRA") & display["class"].eq("negative")
    # MPRA AF was not retained. Co-locate the control with the benchmark's
    # positive ascertainment centre rather than imply a measured 8% median.
    display.loc[mpra_negative, "display_allele_frequency"] = 0.07
    display["display_interval_style"] = "solid"
    display["display_dataset_id"] = display["dataset"].map(DATASET_IDS)
    display["display_color"] = display["dataset"].map(DATASET_COLORS)
    display["display_dataset_label"] = display["dataset"].map(DATASET_LABELS)
    display["display_note"] = ""
    display.loc[ddd_negative, "display_note"] = (
        "Negative-control AF was not retained; marker moved from the prior 1% placeholder "
        "to a schematic near-de-novo position."
    )
    display.loc[mpra_negative, "display_note"] = (
        "Control AF was not retained; marker is co-located schematically with the "
        "GWAS/eQTL-ascertained positive set."
    )
    ukb_positive = display["dataset"].eq("UKB rare") & display["class"].eq("positive")
    display.loc[ukb_positive, "display_note"] = (
        "Displayed below the rare fine-mapped positive set to encode greater expected "
        "deleteriousness, while remaining above ClinVar and DDD; the coordinate is schematic."
    )
    return display, summary


def draw_architecture_background(
    drawing: Drawing,
    panel: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> None:
    span = y1 - y0
    if panel == "a":
        drawing.add(
            Rect(
                x0 * mm,
                (y0 + 0.43 * span) * mm,
                (x1 - x0) * mm,
                0.14 * span * mm,
                fillColor=HexColor(PALE_NEUTRAL),
                strokeColor=None,
            )
        )
        deleterious = RLPath()
        deleterious.moveTo(x0 * mm, y0 * mm)
        deleterious.lineTo(x_log(0.04, x0, x1) * mm, y0 * mm)
        deleterious.curveTo(
            x_log(0.01, x0, x1) * mm,
            (y0 + 0.18 * span) * mm,
            x_log(7e-4, x0, x1) * mm,
            (y0 + 0.34 * span) * mm,
            x0 * mm,
            (y0 + 0.43 * span) * mm,
        )
        deleterious.closePath()
        deleterious.fillColor = HexColor(PALE_ORANGE)
        deleterious.strokeColor = None
        drawing.add(deleterious)

        adaptive = RLPath()
        adaptive.moveTo(x_log(8e-4, x0, x1) * mm, (y0 + 0.58 * span) * mm)
        adaptive.curveTo(
            x_log(0.01, x0, x1) * mm,
            (y0 + 0.74 * span) * mm,
            x_log(0.12, x0, x1) * mm,
            (y0 + 0.82 * span) * mm,
            x1 * mm,
            (y0 + 0.88 * span) * mm,
        )
        adaptive.lineTo(x1 * mm, y1 * mm)
        adaptive.lineTo(x_log(0.02, x0, x1) * mm, y1 * mm)
        adaptive.closePath()
        adaptive.fillColor = HexColor(PALE_TEAL)
        adaptive.strokeColor = None
        drawing.add(adaptive)

        core.add_text(drawing, x0 + 11.0, y0 + 2.3, "Purifying selection", 6.1)
        core.add_text(drawing, x0 + 2.0, y0 + 0.585 * span, "Fitness effects not established", 6.0)
        core.add_text(
            drawing,
            x1 - 1.5,
            y1 - 3.2,
            "Positive selection / adaptation",
            6.1,
            anchor="end",
        )
    else:
        rare_high = RLPath()
        rare_high.moveTo(x0 * mm, (y0 + 0.55 * span) * mm)
        rare_high.curveTo(
            x_log(4e-4, x0, x1) * mm,
            (y0 + 0.77 * span) * mm,
            x_log(5e-3, x0, x1) * mm,
            (y0 + 0.84 * span) * mm,
            x_log(1e-2, x0, x1) * mm,
            y1 * mm,
        )
        rare_high.lineTo(x0 * mm, y1 * mm)
        rare_high.closePath()
        rare_high.fillColor = HexColor(PALE_ORANGE)
        rare_high.strokeColor = None
        drawing.add(rare_high)

        low_frequency = RLPath()
        low_frequency.moveTo(x_log(2e-4, x0, x1) * mm, (y0 + 0.20 * span) * mm)
        low_frequency.curveTo(
            x_log(2e-3, x0, x1) * mm,
            (y0 + 0.52 * span) * mm,
            x_log(5e-2, x0, x1) * mm,
            (y0 + 0.58 * span) * mm,
            x_log(0.12, x0, x1) * mm,
            (y0 + 0.30 * span) * mm,
        )
        low_frequency.lineTo(x_log(0.12, x0, x1) * mm, (y0 + 0.12 * span) * mm)
        low_frequency.curveTo(
            x_log(1e-2, x0, x1) * mm,
            (y0 + 0.21 * span) * mm,
            x_log(1e-3, x0, x1) * mm,
            (y0 + 0.08 * span) * mm,
            x_log(2e-4, x0, x1) * mm,
            (y0 + 0.20 * span) * mm,
        )
        low_frequency.closePath()
        low_frequency.fillColor = HexColor(PALE_BLUE)
        low_frequency.strokeColor = None
        drawing.add(low_frequency)

        common_x = x_log(0.035, x0, x1)
        drawing.add(
            Rect(
                common_x * mm,
                (y0 + 0.04 * span) * mm,
                (x1 - common_x) * mm,
                0.22 * span * mm,
                fillColor=HexColor(PALE_TEAL),
                strokeColor=None,
            )
        )
        core.add_text(drawing, x_log(2e-5, x0, x1), y1 - 3.2, "Rare, high-penetrance disease", 6.1)
        core.add_text(
            drawing,
            x_log(0.020, x0, x1),
            y0 + 0.63 * span,
            "Trait-associated effects",
            6.0,
            anchor="middle",
        )
        core.add_text(
            drawing,
            x1 - 1.2,
            y0 + 21.0,
            "GWAS/eQTL regulatory effects",
            6.1,
            anchor="end",
        )


def draw_architecture_axes(
    drawing: Drawing,
    panel: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> None:
    core.add_line(drawing, x0, y0, x1, y0, width=0.70)
    core.add_line(drawing, x0, y0, x0, y1, width=0.70)
    for fraction, label in (
        (0.06, "ultra-rare"),
        (0.33, "rare"),
        (0.60, "low-frequency"),
        (0.87, "common"),
    ):
        px = x0 + fraction * (x1 - x0)
        core.add_line(drawing, px, y0, px, y0 - 0.9, width=0.48)
        core.add_text(drawing, px, y0 - 3.2, label, 6.0, anchor="middle")
    core.add_text(drawing, (x0 + x1) / 2, y0 - 7.0, "Allele-frequency category", 6.6, anchor="middle")

    if panel == "a":
        ticks = ((0.06, "deleterious"), (0.94, "adaptive"))
        axis_label = "Fitness effect (schematic)"
    else:
        ticks = ((0.08, "small"), (0.38, "moderate"), (0.70, "large"), (0.94, "very large"))
        axis_label = "Effect magnitude (schematic)"
    for fraction, label in ticks:
        py = y0 + fraction * (y1 - y0)
        core.add_line(drawing, x0 - 0.9, py, x0, py, width=0.48)
        core.add_text(drawing, x0 - 1.5, py - 0.65, label, 6.0, anchor="end")
    add_rotated_text(drawing, x0 - 11.5, (y0 + y1) / 2, axis_label, size=6.6)


def draw_arch_marker(
    drawing: Drawing,
    x: float,
    y: float,
    color: str,
    label: str,
    *,
    positive: bool,
) -> None:
    radius = 1.55 if positive else 1.22
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            radius * mm,
            fillColor=HexColor(color if positive else WHITE),
            strokeColor=HexColor(BLACK if positive else color),
            strokeWidth=0.60 if positive else 0.48,
        )
    )
    core.add_text(
        drawing,
        x,
        y - 0.72,
        label,
        6.0 if positive else 5.7,
        bold=True,
        anchor="middle",
        color=WHITE if positive else BLACK,
    )


def draw_architecture_data(
    drawing: Drawing,
    placements: pd.DataFrame,
    panel: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> None:
    subset = placements[placements["panel"].eq(panel)]
    for dataset in DATASET_COLORS:
        rows = subset[subset["dataset"].eq(dataset)]
        positive = rows[rows["class"].eq("positive")].iloc[0]
        negative = rows[rows["class"].eq("negative")].iloc[0]
        color = DATASET_COLORS[dataset]
        xp = x0 + float(positive["display_x_fraction"]) * (x1 - x0)
        xn = x0 + float(negative["display_x_fraction"]) * (x1 - x0)
        yp = y0 + float(positive["display_y_fraction"]) * (y1 - y0)
        yn = y0 + float(negative["display_y_fraction"]) * (y1 - y0)

        # Pairing lines remain visible but subordinate to the benchmark atlas.
        core.add_line(drawing, xn, yn, xp, yp, color=blend(color, 0.68), width=0.34)
        draw_arch_marker(drawing, xn, yn, color, "N", positive=False)
        draw_arch_marker(
            drawing,
            xp,
            yp,
            color,
            str(DATASET_IDS[dataset]),
            positive=True,
        )


def draw_architecture_panel(
    drawing: Drawing,
    placements: pd.DataFrame,
    *,
    panel: str,
    letter: str,
    title: str,
    x: float,
    width: float,
) -> None:
    core.add_text(drawing, x, 291.3, letter, 13.0, bold=True)
    core.add_text(drawing, x + width / 2, 291.0, title, 7.2, bold=True, anchor="middle")
    # Match the outer padding used in Figures 1-2: the rotated y-axis title is
    # inset from both the page edge and the a-b gutter, while the plot top is
    # raised to remove excess whitespace below the panel title.
    x0 = x + 15.0
    x1 = x + width - 2.5
    y0 = 225.5
    y1 = 287.2
    draw_architecture_background(drawing, panel, x0, x1, y0, y1)
    draw_architecture_axes(drawing, panel, x0, x1, y0, y1)
    draw_architecture_data(drawing, placements, panel, x0, x1, y0, y1)


def draw_dataset_legend(drawing: Drawing) -> None:
    items = sorted(DATASET_COLORS, key=lambda dataset: DATASET_IDS[dataset])
    columns = (
        [dataset for dataset in items if DATASET_IDS[dataset] % 2 == 1],
        [dataset for dataset in items if DATASET_IDS[dataset] % 2 == 0],
    )
    badge_radius = 1.15
    text_offset = 2.7
    column_gap = 14.0

    # The reference page uses a wide two-column optical spread. These are the
    # badge-left anchors in mm; text remains inside the A4 page and its width
    # does not affect source data or panel geometry.
    column_starts = (40.17 - badge_radius, 140.97 - badge_radius)
    y_positions = (213.4, 207.2, 201.0)
    for column_index, column in enumerate(columns):
        x = column_starts[column_index] + badge_radius
        for row, dataset in enumerate(column):
            y = y_positions[row]
            color = DATASET_COLORS[dataset]
            line_1, line_2 = DATASET_LEGEND_LINES[dataset]
            badge_y = y - 1.05
            drawing.add(
                Circle(
                    x * mm,
                    badge_y * mm,
                    badge_radius * mm,
                    fillColor=HexColor(color),
                    strokeColor=HexColor(BLACK),
                    strokeWidth=0.48,
                )
            )
            core.add_text(
                drawing,
                x,
                badge_y - 0.64,
                str(DATASET_IDS[dataset]),
                6.0,
                bold=True,
                anchor="middle",
                color=WHITE,
            )
            if dataset == "DDD de novo":
                italic_font = "Arial-Italic" if "Arial-Italic" in pdfmetrics.getRegisteredFontNames() else "Arial"
                runs = (
                    ("DDD ", "Arial-Bold", 0.0),
                    ("d", italic_font, 0.33),
                    ("e", italic_font, 0.0),
                    (" ", italic_font, 0.0),
                    ("n", italic_font, 0.44),
                    ("o", italic_font, 0.44),
                    ("v", italic_font, 0.44),
                    ("o", italic_font, 0.0),
                    (" LoF/splice variants", "Arial-Bold", 0.0),
                )
                cursor_pt = (x + text_offset) * mm
                for text, font, tracking_pt in runs:
                    drawing.add(String(cursor_pt, y * mm, text, fontName=font, fontSize=6.0, fillColor=HexColor(BLACK)))
                    cursor_pt += pdfmetrics.stringWidth(text, font, 6.0) + tracking_pt
            else:
                core.add_text(drawing, x + text_offset, y, line_1, 6.0, bold=True)
            core.add_text(drawing, x + text_offset, y - 2.2, line_2, 5.8)


def draw_model_legend(
    drawing: Drawing,
    y: float,
    width_mm: float,
    *,
    include_note: bool = True,
) -> None:
    items = (("hEVO", HEVO), ("NT", NT), ("EVO 2", EVO2), ("GPN", GPN), ("GPN-Star", GPN_STAR), ("Other", NEUTRAL))
    item_widths = (20.0, 16.0, 21.0, 18.0, 25.0, 21.0)
    x = (width_mm - sum(item_widths)) / 2
    for (label, color), item_width in zip(items, item_widths):
        drawing.add(
            Rect(
                x * mm,
                (y - 0.95) * mm,
                3.6 * mm,
                1.5 * mm,
                fillColor=HexColor(color),
                strokeColor=None,
            )
        )
        core.add_text(drawing, x + 4.6, y - 0.63, label, 6.0)
        x += item_width
    if include_note:
        core.add_text(
            drawing,
            width_mm - 5.0,
            y - 0.63,
            "local AUPRC scales | whisker: 95% CI | black tick: prevalence",
            6.0,
            anchor="end",
        )


def draw_rich_ddd_line(drawing: Drawing, center_x: float, y: float, size: float) -> None:
    parts = (
        ("DDD ", "Arial-Bold"),
        ("de novo", "Arial-Italic" if "Arial-Italic" in pdfmetrics.getRegisteredFontNames() else "Arial"),
        (" LoF/splice variants", "Arial-Bold"),
    )
    widths = [pdfmetrics.stringWidth(text, font, size) / 72.0 * 25.4 for text, font in parts]
    cursor = center_x - sum(widths) / 2.0
    for (text, font), width in zip(parts, widths):
        drawing.add(
            String(
                cursor * mm,
                y * mm,
                text,
                fontName=font,
                fontSize=size,
                fillColor=HexColor(BLACK),
            )
        )
        cursor += width


def draw_dataset_badge(drawing: Drawing, dataset: str, x: float, y: float) -> None:
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            1.10 * mm,
            fillColor=HexColor(DATASET_COLORS[dataset]),
            strokeColor=HexColor(BLACK),
            strokeWidth=0.46,
        )
    )
    core.add_text(
        drawing,
        x,
        y - 0.59,
        str(DATASET_IDS[dataset]),
        5.5,
        bold=True,
        anchor="middle",
        color=WHITE,
    )


def draw_numbered_panel_header(
    drawing: Drawing,
    *,
    x: float,
    width: float,
    top: float,
    letter: str,
    header_key: str | None = None,
) -> None:
    header_key = header_key or letter
    dataset, lines = PANEL_HEADERS[header_key]
    size = 6.5
    line_widths = []
    for line in lines:
        if header_key == "g" and line.startswith("DDD de novo"):
            parts = (
                ("DDD ", "Arial-Bold"),
                ("de novo", "Arial-Italic" if "Arial-Italic" in pdfmetrics.getRegisteredFontNames() else "Arial"),
                (" LoF/splice variants", "Arial-Bold"),
            )
            line_widths.append(
                sum(pdfmetrics.stringWidth(text, font, size) for text, font in parts)
                / 72.0
                * 25.4
            )
        else:
            line_widths.append(
                pdfmetrics.stringWidth(line, "Arial-Bold", size) / 72.0 * 25.4
            )
    text_width = max(line_widths)
    group_width = 2.2 + 1.3 + text_width
    group_left = x + width / 2.0 - group_width / 2.0
    badge_x = group_left + 1.1
    text_center = group_left + 3.5 + text_width / 2.0
    draw_dataset_badge(drawing, dataset, badge_x, top - 1.55)
    core.add_text(drawing, x, top - 1.0, letter, 13.0, bold=True)
    for idx, line in enumerate(lines):
        y = top - 0.9 - idx * 2.4
        if header_key == "g" and idx == 0:
            draw_rich_ddd_line(drawing, text_center, y, size)
        else:
            core.add_text(drawing, text_center, y, line, size, bold=True, anchor="middle")


def draw_numbered_bar_panel(
    drawing: Drawing,
    rows: pd.DataFrame,
    *,
    x: float,
    bottom: float,
    width: float,
    top: float,
    letter: str,
    n_samples: int,
    baseline: float,
    limits: tuple[float, float],
    ticks: tuple[float, ...],
    score_col: str,
    low_col: str,
    high_col: str,
    header_key: str | None = None,
) -> None:
    # Reserve a consistent two-line header in every benchmark panel.
    core.draw_bar_panel(
        drawing,
        rows,
        x=x,
        bottom=bottom,
        width=width,
        top=top - 2.5,
        letter="",
        title="",
        n_samples=n_samples,
        baseline=baseline,
        limits=limits,
        ticks=ticks,
        score_col=score_col,
        low_col=low_col,
        high_col=high_col,
    )
    draw_numbered_panel_header(
        drawing,
        x=x,
        width=width,
        top=top,
        letter=letter,
        header_key=header_key,
    )


def prepare_main_source(
    clinvar: pd.DataFrame,
    static: pd.DataFrame,
    temporal: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    clin = clinvar[clinvar["task"].eq("all")].copy()
    clin.insert(0, "panel_letter", "c")
    clin.insert(1, "panel_group", "ClinVar")
    clin.insert(2, "dataset_map_id", 1)
    clin.insert(3, "panel_title", DATASET_LABELS["ClinVar"])
    parts.append(clin)
    static_letters = {
        "causal": "d",
        "rare_causal": "e",
        "rare_ukb": "f",
        "ddd_clinvar": "g",
        "mpra": "h",
    }
    static_datasets = {
        "causal": "CAUSALdb GWAS",
        "rare_causal": "CAUSALdb rare",
        "rare_ukb": "UKB rare",
        "ddd_clinvar": "DDD de novo",
        "mpra": "MPRA",
    }
    for task, letter in static_letters.items():
        part = static[static["task"].eq(task)].copy()
        part.insert(0, "panel_letter", letter)
        part.insert(1, "panel_group", "Static variant benchmark")
        dataset = static_datasets[task]
        part.insert(2, "dataset_map_id", DATASET_IDS[dataset])
        part.insert(3, "panel_title", DATASET_LABELS[dataset])
        parts.append(part)
    temporal_letters = {
        "blb_stable_vs_upgraded": "i",
        "plp_stable_vs_downgraded": "j",
        "vus_resolved_path_vs_benign": "k",
    }
    for task, letter in temporal_letters.items():
        part = temporal[temporal["task_name"].eq(task)].copy()
        part.insert(0, "panel_letter", letter)
        part.insert(1, "panel_group", "Temporal ClinVar")
        part.insert(2, "dataset_map_id", 1)
        part.insert(3, "panel_title", "Temporal ClinVar reclassification")
        parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False)


def render_main(
    clinvar: pd.DataFrame,
    static: pd.DataFrame,
    temporal: pd.DataFrame,
    checks: dict,
    placements: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    drawing = Drawing(MAIN_W_MM * mm, MAIN_H_MM * mm)
    drawing.add(Rect(0, 0, MAIN_W_MM * mm, MAIN_H_MM * mm, fillColor=HexColor(WHITE), strokeColor=None))

    # a-b: architecture maps.
    draw_architecture_panel(
        drawing,
        placements,
        panel="b",
        letter="a",
        title="Phenotypic effects and allele frequency",
        x=5.0,
        width=98.0,
    )
    draw_architecture_panel(
        drawing,
        placements,
        panel="a",
        letter="b",
        title="Fitness-effect landscape (schematic)",
        x=107.0,
        width=98.0,
    )
    draw_dataset_legend(drawing)
    draw_model_legend(drawing, 194.2, MAIN_W_MM, include_note=False)

    margin = 5.0
    gap = 3.0
    panel_width = (MAIN_W_MM - 2 * margin - 2 * gap) / 3.0
    xs = [margin + index * (panel_width + gap) for index in range(3)]

    # c-e.
    first_row = (
        ("c", "clinvar", "all"),
        ("d", "static", "causal"),
        ("e", "static", "rare_causal"),
    )
    for x, (letter, source_kind, task) in zip(xs, first_row):
        if source_kind == "clinvar":
            rows = clinvar[clinvar["task"].eq(task)]
            meta = core.CLINVAR_META[task]
            draw_numbered_bar_panel(
                drawing,
                rows,
                x=x,
                bottom=130.0,
                width=panel_width,
                top=185.5,
                letter=letter,
                n_samples=int(rows.iloc[0]["num_samples"]),
                baseline=float(rows.iloc[0]["baseline"]),
                limits=meta["scale"],
                ticks=meta["ticks"],
                score_col="auprc_mean",
                low_col="auprc_ci_lower",
                high_col="auprc_ci_upper",
            )
        else:
            rows = static[static["task"].eq(task)]
            meta = core.STATIC_META[task]
            stats = checks["static_task_stats"][task]
            draw_numbered_bar_panel(
                drawing,
                rows,
                x=x,
                bottom=130.0,
                width=panel_width,
                top=185.5,
                letter=letter,
                n_samples=int(stats["num_samples"]),
                baseline=float(stats["baseline"]),
                limits=meta["scale"],
                ticks=meta["ticks"],
                score_col="score",
                low_col="ci_lower",
                high_col="ci_upper",
            )

    # f-h.
    second_row = (
        ("f", "rare_ukb"),
        ("g", "ddd_clinvar"),
        ("h", "mpra"),
    )
    for x, (letter, task) in zip(xs, second_row):
        rows = static[static["task"].eq(task)]
        meta = core.STATIC_META[task]
        stats = checks["static_task_stats"][task]
        draw_numbered_bar_panel(
            drawing,
            rows,
            x=x,
            bottom=66.5,
            width=panel_width,
            top=122.0,
            letter=letter,
            n_samples=int(stats["num_samples"]),
            baseline=float(stats["baseline"]),
            limits=meta["scale"],
            ticks=meta["ticks"],
            score_col="score",
            low_col="ci_lower",
            high_col="ci_upper",
        )

    # i-k: the former d-f Candidate A row.
    temporal_letters = ("i", "j", "k")
    temporal_titles = (
        "LB/B to later upgrade",
        "P/LP to later downgrade",
        "VUS to P/LP",
    )
    for task, letter, title in zip(core.TEMPORAL_META, temporal_letters, temporal_titles):
        core.TEMPORAL_META[task]["letter"] = letter
        core.TEMPORAL_META[task]["title"] = title
    core.draw_temporal_legend(drawing, 60.2)
    for x, task in zip(xs, core.TEMPORAL_META):
        core.draw_temporal_panel(
            drawing,
            temporal,
            task=task,
            x=x,
            width=panel_width,
            top=59.5,
            bottom=4.5,
            inner_left=4.25,
            inner_right=4.25,
        )

    return render_artifact(drawing, MAIN_STEM)


def render_supplement(clinvar: pd.DataFrame) -> tuple[Path, Path, Path]:
    drawing = Drawing(SUP_W_MM * mm, SUP_H_MM * mm)
    drawing.add(Rect(0, 0, SUP_W_MM * mm, SUP_H_MM * mm, fillColor=HexColor(WHITE), strokeColor=None))
    draw_model_legend(drawing, 77.0, SUP_W_MM, include_note=False)
    margin = 4.0
    gap = 5.0
    panel_width = (SUP_W_MM - 2 * margin - gap) / 2.0
    for index, task in enumerate(("coding", "noncoding")):
        rows = clinvar[clinvar["task"].eq(task)]
        meta = core.CLINVAR_META[task]
        core.draw_bar_panel(
            drawing,
            rows,
            x=margin + index * (panel_width + gap),
            bottom=5.0,
            width=panel_width,
            top=73.0,
            letter=chr(ord("a") + index),
            title=f"ClinVar {task}",
            n_samples=int(rows.iloc[0]["num_samples"]),
            baseline=float(rows.iloc[0]["baseline"]),
            limits=meta["scale"],
            ticks=meta["ticks"],
            score_col="auprc_mean",
            low_col="auprc_ci_lower",
            high_col="auprc_ci_upper",
        )
    return render_artifact(drawing, SUP_STEM)


def render_artifact(drawing: Drawing, stem: str) -> tuple[Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    svg = OUT_DIR / f"{stem}.svg"
    png = OUT_DIR / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf), title=stem)
    renderSVG.drawToFile(drawing, str(svg))
    poppler = (
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
    subprocess.run(
        [str(poppler), "-png", "-singlefile", "-r", str(PNG_DPI), str(pdf), str(png.with_suffix(""))],
        check=True,
    )
    return pdf, svg, png


def artifact_qa(
    pdf: Path,
    svg: Path,
    png: Path,
    expected_width_mm: float,
    expected_height_mm: float,
) -> dict:
    reader = PdfReader(str(pdf))
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) / 72.0 * 25.4
    height_mm = float(page.mediabox.height) / 72.0 * 25.4
    with Image.open(png) as image:
        pixels = image.size
        dpi = image.info.get("dpi", (None, None))
    svg_text = svg.read_text(encoding="utf-8").lower()
    return {
        "pdf": str(pdf),
        "svg": str(svg),
        "png": str(png),
        "width_mm": width_mm,
        "height_mm": height_mm,
        "expected_width_mm": expected_width_mm,
        "expected_height_mm": expected_height_mm,
        "png_pixels": pixels,
        "png_dpi": dpi,
        **core.inspect_pdf_fonts(pdf),
        "svg_text_preserved": "<text" in svg_text,
        "forbidden_pink_absent": not any(
            token in svg_text for token in ("#ff6fae", "#d45087", "#f7dfe9", "#e78ac3")
        ),
        "hashes": {
            "pdf": core.sha256(pdf),
            "svg": core.sha256(svg),
            "png": core.sha256(png),
        },
    }


def write_supporting_files(
    clinvar: pd.DataFrame,
    static: pd.DataFrame,
    temporal: pd.DataFrame,
    checks: dict,
    placements: pd.DataFrame,
    summary: pd.DataFrame,
    main_paths: tuple[Path, Path, Path],
    sup_paths: tuple[Path, Path, Path],
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for source in (
        core.CLINVAR_SOURCE,
        core.STATIC_SOURCE,
        core.TEMPORAL_SOURCE,
        ARCH_PLACEMENT_SOURCE,
        ARCH_summary_SOURCE,
    ):
        shutil.copy2(source, SOURCE_DIR / source.name)
    placements.to_csv(SOURCE_DIR / "variant_map_placements_v10_display.csv", index=False)
    summary.to_csv(SOURCE_DIR / "variant_benchmark_sampling_summary_v10.csv", index=False)
    prepare_main_source(clinvar, static, temporal).to_csv(
        SOURCE_DIR / "Figure3_abcdefghijk_v10_main_plot_source.csv", index=False
    )
    clinvar[clinvar["task"].isin(["coding", "noncoding"])].to_csv(
        SOURCE_DIR / "FigureS4_ClinVar_coding_noncoding_plot_source.csv", index=False
    )

    qa = {
        "status": "DATA_QC_DRAFT",
        "submission_status": "NOT_SUBMISSION_READY",
        "main": artifact_qa(*main_paths, MAIN_W_MM, MAIN_H_MM),
        "supplement": artifact_qa(*sup_paths, SUP_W_MM, SUP_H_MM),
        "data_checks": checks,
        "architecture_display_changes": {
            "interval_line_style": "solid for every dataset",
            "ddd_negative_af_source_value": 0.01,
            "ddd_negative_af_display_value": 0.000006,
            "ddd_negative_af_reason": (
                "ClinVar benign control AF was not retained. The previous 1% location could be misread "
                "as an observed median; the revised near-de-novo location is explicitly schematic."
            ),
            "mpra_negative_af_source_value": 0.08,
            "mpra_negative_af_display_value": 0.07,
            "mpra_negative_af_reason": (
                "AF was not retained; display is co-located schematically with the "
                "GWAS/eQTL-ascertained positive set."
            ),
            "all_panel_b_negatives_within_limited_fitness_band": True,
            "mpra_positive_panel_b_display_coordinate": 0.18,
            "mpra_positive_panel_b_reason": (
                "Moved just above the limited-impact band for visual separation; "
                "this does not imply adaptive organismal fitness."
            ),
            "ukb_positive_panel_b_display_coordinate": -0.42,
            "ukb_positive_panel_b_reason": (
                "Placed below the rare fine-mapped positive set (dataset 3) to encode greater "
                "expected deleteriousness, while remaining above ClinVar and DDD; this is an "
                "ordinal schematic, not a measured fitness effect."
            ),
            "dataset_color_semantics": (
                "red (severe clinical/developmental) through ochre to green "
                "(molecular function), independent of model-family colors"
            ),
            "display_dataset_id_order": {
                "ClinVar": 1,
                "GWAS fine-mapped all": 2,
                "GWAS fine-mapped rare": 3,
                "UKB WGS/WES rare": 4,
                "DDD de novo": 5,
                "GWAS/eQTL fine-mapped MPRA": 6
            },
            "source_id_preserved_separately": True,
            "source_panel_preserved_separately": True,
            "display_panel_order": {
                "a": "phenotypic consequence and effective allele frequency",
                "b": "fitness effect spectrum and derived allele frequency"
            },
            "placement_semantics": "ascertainment summary, not per-variant effect estimates",
        },
        "layout": {
            "a_b": "variant architecture maps",
            "c_e": "ClinVar all, CausalDB disease, rare causal",
            "f_h": "UKB rare, DDD de novo, MPRA",
            "i_k": "temporal ClinVar",
            "supplement": "ClinVar coding and noncoding",
            "minimum_text_pt": 6.0,
            "temporal_axis_outer_bounds_mm": [9.25, 200.75],
            "architecture_plot_y_bounds_mm": [225.5, 287.2],
            "architecture_axis_label_x_mm": [8.5, 110.5],
            "benchmark_row_y_bounds_mm": {
                "c_e": [130.0, 185.5],
                "f_h": [66.5, 122.0],
                "i_k": [4.5, 59.5]
            },
            "dataset_legend_line1_baselines_mm": [213.4, 207.2, 201.0],
            "model_legend_centerline_mm": 191.8,
        },
        "known_submission_gates": [
            "Architecture-map vertical coordinates and several AF spans are conceptual or ascertainment-based.",
            "UKB positive AF/effect metadata and upstream provenance remain unresolved.",
            "UKB canonical input contains 678 unique negatives sampled without replacement; legacy replacement artifact excluded.",
            "DDD controls are not gene-, consequence-, mutability- or AF-matched.",
            "Temporal source reports n_boot=20 and lacks event-level rows.",
        ],
    }

    main_caption = """# Figure 3 caption draft

**Figure 3 | Human variant benchmarks span the allele-frequency and effect spectrum.** **a,** Conceptual placement of six positive-versus-control contrasts across effective allele frequency and phenotypic consequence, from molecular allelic activity to clinical effect. Numbered colors progress from severe clinical/developmental variation (red) to molecular regulatory function (green) and are independent of model-family colors. **b,** Conceptual placement across derived allele frequency and expected organismal fitness effect. All negative/control markers are confined to the limited-expected-fitness-impact band; positive marker heights reflect benchmark ascertainment rather than measured selection coefficients. The UKB likely-causal rare-variant marker is placed below the broader rare fine-mapped set to encode greater expected deleteriousness, but remains above the ClinVar pathogenic and DDD LoF/splice contrasts. The MPRA positive marker is offset just above the limited-impact band. These vertical offsets are ordinal and schematic rather than measured selection coefficients. In a and b, solid horizontal lines show observed or ascertainment-supported AF spans. Marker coordinates do not represent per-variant penetrance, odds ratios or selection coefficients. CausalDB all-frequency negative AF is the observed selected median (12.73%). MPRA AF was not retained, so its control is co-located schematically with the GWAS/eQTL-ascertained positive set. DDD control AF was not retained and is shown schematically near the de novo edge. **c-h,** AUPRC for (1) ClinVar pathogenic versus benign variants, (2) GWAS fine-mapped causal versus non-causal variants across all frequencies, (3) the corresponding rare-variant contrast, (4) UKB WGS/WES rare variants classified as likely-causal versus controls, (5) DDD *de novo* variants contrasting LoF/splice consequences with controls and (6) GWAS/eQTL fine-mapped MPRA functional versus non-functional variants. Models are ranked independently within each task; whiskers denote reported 95% confidence intervals and black ticks denote positive-class prevalence. Each benchmark uses its own labelled AUPRC scale. **i-k,** enrichment of later ClinVar reclassification events among the top 1% of model-ranked variants across historical cutoff years: LB/B variants later upgraded, P/LP variants later downgraded, and VUS variants resolved as P/LP rather than LB/B. Enrichment is precision at the top 1% divided by cutoff-specific event prevalence; error bars are the reported bootstrap 95% confidence intervals from five seeds and `n_boot = 20`. Lower strips show event prevalence. The temporal analysis is retrospective and has unequal follow-up horizons.
"""
    (OUT_DIR / "Figure3_abcdefghijk_v11_CAPTION_DRAFT.md").write_text(main_caption, encoding="utf-8")

    sup_caption = """# Figure S3 caption draft

**Figure S3 | ClinVar variant-effect discrimination by genomic consequence class.** AUPRC for coding (a) and noncoding (b) pathogenic/likely pathogenic versus benign/likely benign ClinVar variants. Models are ranked independently within each panel; horizontal whiskers denote reported 95% confidence intervals and black ticks denote positive-class prevalence. Each panel uses its own labelled AUPRC scale. hEvo-base ranks first in the coding task and second in the noncoding task, for which GPN-MSA is the leading displayed representative.
"""
    (OUT_DIR / "FigureS3_ClinVar_coding_noncoding_CAPTION_DRAFT.md").write_text(
        sup_caption, encoding="utf-8"
    )


def main() -> None:
    core.configure_fonts()
    clinvar, static, temporal, checks = core.read_sources()
    placements, summary = read_architecture_sources()
    main_paths = render_main(clinvar, static, temporal, checks, placements)
    sup_paths = render_supplement(clinvar)
    write_supporting_files(
        clinvar,
        static,
        temporal,
        checks,
        placements,
        summary,
        main_paths,
        sup_paths,
    )
    print(f"Rendered main: {main_paths[0]}")
    print(f"Rendered supplement: {sup_paths[0]}")


if __name__ == "__main__":
    main()
