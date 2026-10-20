#!/usr/bin/env python3
"""Render the first slot-native A4 candidate for hEVO Figure 1a-h.

The renderer reads the registered Figure 1 source tables and draws panels a-f
directly at their final page dimensions. Panels g and h are inserted from their
native 178 mm renderers without rasterization or scaling. The dense PCA point
layer is the only rasterized analytical layer; all labels, axes and other marks
remain editable vector objects.

This artifact is intentionally marked DATA/QC DRAFT because Fig. 1b and 1c use
digitized values and the available Fig. 1a table lacks final-corpus ancestry
labels and explained-variance metadata.
"""

from __future__ import annotations

# === Comment index ===
# [Role] Current A4 entry point for Fig. 1; combine panels a–h into a single-page PDF/SVG/PNG.
# [Input] Tables S1, S2, and S4–S13, plus source data for Fig. 1c, model architecture, and radar panels.
# [Calls] Active Fig. 1 wrappers reuse the PCA, composition, and radar primitives defined here.
# the formal A4 entry inserts the hand-drawn Fig1-g asset separately.
# [Style] Define Arial/Liberation Sans, black/blue/cyan/orange figure styles, and 600 dpi output here.
# [Layout] Build each panel as a ReportLab Drawing, then place it on the page using A4 millimeter coordinates; main() also handles QA and source data.
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Group,
    Line,
    Rect,
    String,
)
from reportlab.graphics.shapes import (
    Image as RlImage,
)
from reportlab.graphics.shapes import (
    Path as RlPath,
)
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_fig1h_revised_family_profiles as radar_builder
import plot_fig1h_family_rank_radars_compact as radars
from figures.scripts.common.hevo_palette import (
    COHORT_COLORS,
    COHORT_LABELS,
    COHORT_ORDER,
)
from figures.scripts.common.font_config import register_reportlab_arial, resolve_arial_paths

OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig1" / "figure1_a4_integrated_v2"
OUT_BASE = OUT_DIR / "Figure1_a_h_A4_integrated_v2_DATA_QC_DRAFT"
OUT_BASE_3D_EXPLICIT = OUT_DIR / "Figure1_a_PC1_PC2_PC3_A4_integrated_v3_DATA_QC_DRAFT"
OUT_BASE_2D = OUT_DIR / "Figure1_a_PC1_PC2_A4_integrated_v3_DATA_QC_DRAFT"
SOURCE_DIR = OUT_DIR / "source_data"
DENSE_DIR = OUT_DIR / "dense_layers"

PCA_PATH = PROJECT_ROOT / "tables" / "S2_principal_component_coordinates_of_the_final_63_809_participant_cohort.csv"
COMPOSITION_PATH = PROJECT_ROOT / "tables" / "S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv"
C_PATH = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig1"
    / "fig1c_a4_v1"
    / "source_data"
    / "Fig1c_current_summary_transcribed.tsv"
)
B_PATH = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig1"
    / "cvd_standard_v2"
    / "source_data"
    / "Fig1b_empirical_curve.csv"
)
RANK_PATH = PROJECT_ROOT / "tables" / "S8_rank_abundance_profile_of_distinct_haplotype_sequences.csv"
SHARING_PATH = PROJECT_ROOT / "tables" / "S9_haplotype_sequence_sharing_across_five_ancestry_groups.csv"
RADAR_PATH = PROJECT_ROOT / "tables" / "S10_best_model_family_ranks_across_functional_variant_and_evolutionary_tasks.csv"
RADAR_REVISED_PATH = SOURCE_DIR / "Fig1h_revised_family_best_profiles.csv"

PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
PNG_DPI = 600
STATUS = "DATA_QC_DRAFT"
SHOW_PANEL_TITLES = False

BLACK = "#000000"
WHITE = "#FFFFFF"
BLUE = "#0B2E83"
BLUE_MID = "#2F6FD6"
BLUE_LIGHT = "#A8C8E0"
TEAL = "#0E7C86"
ORANGE = "#D97706"
PURPLE = "#7655A6"
COPPER = "#B35C1E"
GREY = "#8995A5"
PALE = "#F3F5F8"
PALE_BLUE = "#EEF3FA"
PALE_COPPER = "#F6E8D8"
INACTIVE = "#E3E7ED"

ANCESTRY_COLORS = {
    "EUR": BLUE,
    "AFR": ORANGE,
    "EAS": TEAL,
    "SAS": PURPLE,
    "Others": GREY,
}

FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_ITALIC = "Arial-Italic"
FONT_PATHS = resolve_arial_paths()


def configure_fonts() -> None:
    register_reportlab_arial()
    for name, path in FONT_PATHS.items():
        if not path.exists():
            raise RuntimeError(f"Required Arial font is missing: {path}")
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))
    radars.configure_fonts()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_text(
    d: Drawing,
    x: float,
    y: float,
    value: str,
    size: float = 6.0,
    *,
    bold: bool = False,
    italic: bool = False,
    anchor: str = "start",
) -> None:
    font = FONT_BOLD if bold else FONT_ITALIC if italic else FONT_REGULAR
    d.add(
        String(
            x * mm,
            y * mm,
            value,
            fontName=font,
            fontSize=size,
            fillColor=HexColor(BLACK),
            textAnchor=anchor,
        )
    )


def add_rotated_text(
    d: Drawing,
    x: float,
    y: float,
    value: str,
    size: float = 6.0,
    *,
    angle: float = 90.0,
    bold: bool = False,
    anchor: str = "middle",
) -> None:
    group = Group()
    group.add(
        String(
            0,
            0,
            value,
            fontName=FONT_BOLD if bold else FONT_REGULAR,
            fontSize=size,
            fillColor=HexColor(BLACK),
            textAnchor=anchor,
        )
    )
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    group.transform = (cosine, sine, -sine, cosine, x * mm, y * mm)
    d.add(group)


def add_line(
    d: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = BLACK,
    width: float = 0.55,
) -> None:
    d.add(
        Line(
            x1 * mm,
            y1 * mm,
            x2 * mm,
            y2 * mm,
            strokeColor=HexColor(color),
            strokeWidth=width,
        )
    )


def panel_header(
    d: Drawing,
    x: float,
    y: float,
    w: float,
    h: float,
    letter: str,
    title: str,
    subtitle: str | None = None,
) -> None:
    add_text(d, x + 0.3, y + h - 5.3, letter, 14.0, bold=True)
    if SHOW_PANEL_TITLES:
        add_text(d, x + w / 2, y + h - 4.8, title, 7.3, bold=True, anchor="middle")
        if subtitle:
            add_text(d, x + w / 2, y + h - 9.0, subtitle, 6.0, anchor="middle")


def nice_domain(values: np.ndarray, pad: float = 0.045) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    span = high - low
    return low - span * pad, high + span * pad


def project_pca_3d(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    raw = frame[["PC1", "PC2", "PC3"]].to_numpy(float)
    low = raw.min(axis=0)
    high = raw.max(axis=0)
    pad = (high - low) * 0.03
    normalized = (raw - (low - pad)) / ((high + pad) - (low - pad))
    azimuth = math.radians(38.0)
    elevation = math.radians(25.0)

    def project(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xx, yy, zz = values[:, 0], values[:, 1], values[:, 2]
        uu = xx * math.cos(azimuth) - yy * math.sin(azimuth)
        vv = (
            xx * math.sin(azimuth) * math.sin(elevation)
            + yy * math.cos(azimuth) * math.sin(elevation)
            + zz * math.cos(elevation)
        )
        depth = (
            xx * math.sin(azimuth) * math.cos(elevation)
            + yy * math.cos(azimuth) * math.cos(elevation)
            - zz * math.sin(elevation)
        )
        return np.column_stack([uu, vv]), depth

    corners = np.array(
        [[xx, yy, zz] for xx in (0.0, 1.0) for yy in (0.0, 1.0) for zz in (0.0, 1.0)]
    )
    projected, depth = project(normalized)
    projected_corners, _ = project(corners)
    minimum = projected_corners.min(axis=0)
    maximum = projected_corners.max(axis=0)

    def scale(values: np.ndarray) -> np.ndarray:
        result = (values - minimum) / (maximum - minimum)
        result[:, 0] = 0.04 + 0.92 * result[:, 0]
        result[:, 1] = 0.04 + 0.92 * result[:, 1]
        return result

    projected = scale(projected)
    corner_scaled = scale(projected_corners)
    corner_map = {
        tuple(int(value) for value in corner): tuple(corner_scaled[index])
        for index, corner in enumerate(corners)
    }
    axes = {
        "origin": corner_map[(0, 0, 0)],
        "PC1": corner_map[(1, 0, 0)],
        "PC2": corner_map[(0, 1, 0)],
        "PC3": corner_map[(0, 0, 1)],
    }
    return projected, depth, axes


def make_pca_3d_layer(
    frame: pd.DataFrame, path: Path, width_mm: float, height_mm: float
) -> dict[str, tuple[float, float]]:
    projected, depth, axes = project_pca_3d(frame)
    dpi = 600
    width_px = max(100, round(width_mm / 25.4 * dpi))
    height_px = max(100, round(height_mm / 25.4 * dpi))
    image = Image.new("RGBA", (width_px, height_px), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    order = np.argsort(depth)
    groups = frame["Group"].to_numpy(str)
    for index in order:
        cohort = groups[index]
        hex_color = COHORT_COLORS[cohort].lstrip("#")
        rgb = tuple(int(hex_color[offset : offset + 2], 16) for offset in (0, 2, 4))
        px = round(projected[index, 0] * (width_px - 1))
        py = round((1.0 - projected[index, 1]) * (height_px - 1))
        radius = 2 if cohort in {"UKB", "TopMed"} else 3
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius), fill=(*rgb, 92)
        )
    image.save(path, format="PNG", dpi=(dpi, dpi), optimize=True)
    return axes


def make_pca_layer(
    frame: pd.DataFrame,
    path: Path,
    width_mm: float,
    height_mm: float,
    domains: tuple[tuple[float, float], tuple[float, float]],
    *,
    group_order: tuple[str, ...] = COHORT_ORDER,
    group_colors: dict[str, str] = COHORT_COLORS,
) -> None:
    dpi = 600
    width_px = max(100, round(width_mm / 25.4 * dpi))
    height_px = max(100, round(height_mm / 25.4 * dpi))
    image = Image.new("RGBA", (width_px, height_px), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    (xmin, xmax), (ymin, ymax) = domains
    counts = frame["Group"].value_counts().to_dict()
    for cohort in sorted(group_order, key=lambda key: counts.get(key, 0), reverse=True):
        part = frame.loc[frame["Group"] == cohort]
        xs = np.rint(
            (part["PC1"].to_numpy(float) - xmin) / (xmax - xmin) * (width_px - 1)
        ).astype(int)
        ys = np.rint(
            (1.0 - (part["PC2"].to_numpy(float) - ymin) / (ymax - ymin))
            * (height_px - 1)
        ).astype(int)
        hex_color = group_colors[cohort].lstrip("#")
        rgb = tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))
        radius = 2 if len(part) > 10000 else 3
        for px, py in zip(xs, ys):
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius), fill=(*rgb, 92)
            )
    image.save(path, format="PNG", dpi=(dpi, dpi), optimize=True)


def draw_panel_a(
    d: Drawing, frame: pd.DataFrame, x: float, y: float, w: float, h: float
) -> None:
    panel_header(d, x, y, w, h, "a", "Genetic structure across source cohorts")
    counts = frame["Group"].value_counts().to_dict()
    legend_y = y + h - 7.0
    for index, cohort in enumerate(COHORT_ORDER):
        row, column = divmod(index, 2)
        lx = x + 9.0 + column * 38.0
        ly = legend_y - row * 4.0
        d.add(
            Circle(
                lx * mm,
                (ly + 0.65) * mm,
                0.75 * mm,
                fillColor=HexColor(COHORT_COLORS[cohort]),
                strokeColor=None,
            )
        )
        label = f"{COHORT_LABELS[cohort]} ({counts[cohort]:,})"
        add_text(d, lx + 2.0, ly, label, 6.0)

    left, right = x + 3.0, x + w - 1.5
    bottom, top = y + 2.5, y + h - 23.0
    layer_path = DENSE_DIR / "Fig1a_compact_source_pool_PC1_PC2_PC3_600dpi.png"
    axes = make_pca_3d_layer(frame, layer_path, right - left, top - bottom)
    d.add(
        RlImage(
            left * mm,
            bottom * mm,
            (right - left) * mm,
            (top - bottom) * mm,
            str(layer_path),
        )
    )

    def map_axis(point: tuple[float, float]) -> tuple[float, float]:
        return left + point[0] * (right - left), bottom + point[1] * (top - bottom)

    origin = map_axis(axes["origin"])
    axis_scales = {"PC1": 1.025, "PC2": 1.025}
    offsets = {"PC1": (1.8, -1.0), "PC2": (-1.8, -1.0), "PC3": (0.0, 1.15)}
    for axis in ("PC1", "PC2", "PC3"):
        endpoint = map_axis(axes[axis])
        if axis == "PC3":
            # Put the PC3 endpoint above the point-cloud envelope and reserve a
            # clean label zone immediately below the cohort legend.
            scale = (top - 0.2 - origin[1]) / (endpoint[1] - origin[1])
        else:
            scale = axis_scales[axis]
        line_endpoint = (
            origin[0] + scale * (endpoint[0] - origin[0]),
            origin[1] + scale * (endpoint[1] - origin[1]),
        )
        add_line(
            d, origin[0], origin[1], line_endpoint[0], line_endpoint[1], width=0.55
        )
        dx, dy = offsets[axis]
        add_text(
            d,
            line_endpoint[0] + dx,
            line_endpoint[1] + dy,
            axis,
            6.3,
            bold=True,
            anchor="middle",
        )


def draw_panel_a_2d(
    d: Drawing,
    frame: pd.DataFrame,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    domains: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    """Draw the 2D PC1-PC2 panel for either source or final-cohort labels."""
    panel_header(d, x, y, w, h, "a", "Genetic structure across source cohorts")
    counts = frame["Group"].value_counts().to_dict()
    final_labels = ("EUR", "AFR", "EAS", "SAS", "Others")
    is_final_cohort = set(counts).issubset(set(final_labels))
    legend_order = final_labels if is_final_cohort else COHORT_ORDER
    legend_labels = (
        {"EUR": "EUR", "AFR": "AFR", "EAS": "EAS", "SAS": "SAS", "Others": "Others"}
        if is_final_cohort
        else COHORT_LABELS
    )
    legend_colors = ANCESTRY_COLORS if is_final_cohort else COHORT_COLORS
    # Fig.1 uses the final S2 IID roster with coordinates taken from the full
    # S1 PCA, so its points share FigS1's coordinate system and axis domains.
    display_counts = counts
    add_text(
        d,
        x + w - 1.5,
        y + h - 2.0,
        f"Final corpus N = {len(frame):,}",
        6.0,
        anchor="end",
    )
    legend_y = y + h - 7.0
    for index, cohort in enumerate(legend_order):
        row, column = divmod(index, 2)
        lx = x + 9.0 + column * (38.0 if not is_final_cohort else 31.0)
        ly = legend_y - row * 4.0
        d.add(
            Circle(
                lx * mm,
                (ly + 0.65) * mm,
                0.75 * mm,
                fillColor=HexColor(legend_colors[cohort]),
                strokeColor=None,
            )
        )
        add_text(d, lx + 2.0, ly, f"{legend_labels[cohort]} ({display_counts[cohort]:,})", 6.0)

    left, right = x + 10.0, x + w - 2.0
    bottom, top = y + 6.0, y + h - 23.0
    if domains is None:
        domains = (
            nice_domain(frame["PC1"].to_numpy(float), pad=0.02),
            nice_domain(frame["PC2"].to_numpy(float), pad=0.02),
        )
    xdomain, ydomain = domains
    layer_path = DENSE_DIR / "Fig1a_final_cohort_S1_axes_PC1_PC2_600dpi.png"
    make_pca_layer(
        frame,
        layer_path,
        right - left,
        top - bottom,
        (xdomain, ydomain),
        group_order=legend_order,
        group_colors=legend_colors,
    )
    d.add(
        RlImage(
            left * mm,
            bottom * mm,
            (right - left) * mm,
            (top - bottom) * mm,
            str(layer_path),
        )
    )
    add_line(d, left, bottom, right, bottom, width=0.6)
    add_line(d, left, bottom, left, top, width=0.6)
    # Short endpoint ticks provide scale boundaries without adding gridlines.
    for xx in (left, right):
        add_line(d, xx, bottom, xx, bottom - 1.0, width=0.4)
    for yy in (bottom, top):
        add_line(d, left, yy, left - 1.0, yy, width=0.4)
    add_text(d, (left + right) / 2, y + 2.0, "PC1", 6.3, anchor="middle", bold=True)
    add_rotated_text(d, x + 3.0, (bottom + top) / 2, "PC2", 6.3, angle=90.0, bold=True)


def draw_panel_b(
    d: Drawing,
    curve: pd.DataFrame,
    x: float,
    y: float,
    w: float,
    h: float,
    selected_ukb_n: int,
) -> None:
    if selected_ukb_n <= 0:
        raise ValueError("Fig1b selected UKB sample count must be positive")
    sample_label = f"N = {selected_ukb_n:,}"
    panel_header(d, x, y, w, h, "b", "UKB reference haplotype coverage", sample_label)
    add_text(d, x + w - 3.0, y + h - 4.8, sample_label, 6.0, anchor="end")
    left, right = x + 13.5, x + w - 3.0
    bottom, top = y + 9.0, y + h - 9.0
    add_line(d, left, bottom, right, bottom, width=0.65)
    add_line(d, left, bottom, left, top, width=0.65)
    xmin, xmax = math.log10(0.001), math.log10(2.0)
    ymin, ymax = 75.0, 100.8
    xmap = lambda value: (
        left + (math.log10(value) - xmin) / (xmax - xmin) * (right - left)
    )
    ymap = lambda value: bottom + (value - ymin) / (ymax - ymin) * (top - bottom)
    path = RlPath()
    points = []
    for row in curve.itertuples(index=False):
        frequency = min(2.0, max(0.001, float(row.minimum_haplotype_frequency_pct)))
        coverage = 100.0 * float(row.weighted_coverage)
        points.append((xmap(frequency), ymap(coverage)))
    path.moveTo(points[0][0] * mm, points[0][1] * mm)
    for px, py in points[1:]:
        path.lineTo(px * mm, py * mm)
    path.fillColor = None
    path.strokeColor = HexColor(BLUE)
    path.strokeWidth = 1.05
    d.add(path)
    for tick, label in ((0.001, "0.001%"), (0.01, "0.01%"), (0.1, "0.1%"), (1.0, "1%")):
        xx = xmap(tick)
        add_line(d, xx, bottom, xx, bottom - 1.0, width=0.45)
        add_text(d, xx, bottom - 3.5, label, 6.0, anchor="middle")
    for tick in (80, 90, 100):
        yy = ymap(tick)
        add_line(d, left, yy, left - 1.0, yy, width=0.45)
        add_text(d, left - 1.7, yy - 0.7, str(tick), 6.0, anchor="end")
    for frequency, coverage, label, dx, dy in (
        (0.001, 77.58, "77.6%", 3.5, 2.4),
        (0.01, 95.29, "95.3%", 3.3, -2.4),
        (0.1, 99.99, ">99.99%", 3.5, -2.5),
    ):
        px, py = xmap(frequency), ymap(coverage)
        d.add(
            Circle(
                px * mm,
                py * mm,
                0.85 * mm,
                fillColor=HexColor(WHITE),
                strokeColor=HexColor(BLUE),
                strokeWidth=0.75,
            )
        )
        add_line(d, px + 0.8, py, px + dx - 0.5, py + dy - 0.2, width=0.4)
        add_text(d, px + dx, py + dy - 0.6, label, 6.0)
    add_rotated_text(d, x + 3.5, (bottom + top) / 2, "Coverage (%)", 6.3, angle=90.0)
    add_text(
        d,
        (left + right) / 2,
        y + 2.0,
        "Minimum reference haplotype frequency (%)",
        6.3,
        anchor="middle",
    )


def draw_panel_c(
    d: Drawing, table: pd.DataFrame, x: float, y: float, w: float, h: float
) -> None:
    panel_header(d, x, y, w, h, "c", "Cohort-balanced haplotype diversity")
    add_text(d, x + w - 3.0, y + h - 4.8, "Fold vs UKB subset", 6.0, anchor="end")
    left, right = x + 31.0, x + w - 3.0
    bottom, top = y + 7.5, y + h - 7.0
    values = table["recovered_absolute_per_100kb"].to_numpy(float)
    fold_values = values / values[0]
    maximum = 1320.0
    xmap = lambda value: left + value / maximum * (right - left)
    row_step = (top - bottom) / len(table)
    point_coords = []
    scenario_cohort = {
        "UKB subset": "UKB",
        "+ UK10K": "UK10K",
        "+ EAS2K": "EAS",
        "+ SG10K": "SG10K",
        "+ AFR1K": "AFR",
        "+ TOPMed": "TopMed",
        "+ HGDP": "HGDP",
        "+ 1KGP": "1kgenome",
    }
    scenario_label = {
        "UKB subset": "UK Biobank subset",
        "+ UK10K": "+ UK10K",
        "+ EAS2K": "+ East Asian cohort",
        "+ SG10K": "+ Singapore 10K",
        "+ AFR1K": "+ African cohort",
        "+ TOPMed": "+ TOPMed",
        "+ HGDP": "+ HGDP",
        "+ 1KGP": "+ 1000 Genomes",
    }
    for index, row in table.reset_index(drop=True).iterrows():
        yy = top - (index + 0.5) * row_step
        value = float(row["recovered_absolute_per_100kb"])
        final = index == len(table) - 1
        fill = "#739BC8" if final else BLUE_LIGHT
        stroke = BLUE if final else BLUE_MID
        d.add(
            Rect(
                left * mm,
                (yy - 0.95) * mm,
                (xmap(value) - left) * mm,
                1.9 * mm,
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=0.35,
            )
        )
        scenario = str(row["scenario"])
        icon_x = left - 28.5
        d.add(
            Circle(
                icon_x * mm,
                yy * mm,
                0.72 * mm,
                fillColor=HexColor(COHORT_COLORS[scenario_cohort[scenario]]),
                strokeColor=None,
            )
        )
        add_text(d, left - 1.6, yy - 0.65, scenario_label[scenario], 6.0, anchor="end")
        point_coords.append((xmap(value), yy))
        d.add(
            Circle(
                xmap(value) * mm,
                yy * mm,
                0.95 * mm,
                fillColor=HexColor(BLUE if final else BLUE_MID),
                strokeColor=HexColor(WHITE),
                strokeWidth=0.45,
            )
        )
        label = f"{fold_values[index]:.2f}x"
        label_x = xmap(value) + 1.6 if final else min(xmap(value) + 1.6, right - 7.5)
        add_text(d, label_x, yy - 0.65, label, 6.0)
    line_path = RlPath()
    line_path.moveTo(point_coords[0][0] * mm, point_coords[0][1] * mm)
    for px, py in point_coords[1:]:
        line_path.lineTo(px * mm, py * mm)
    line_path.strokeColor = HexColor("#6EA8CF")
    line_path.strokeWidth = 0.9
    line_path.fillColor = None
    d.add(line_path)
    add_line(d, left, bottom, right, bottom, width=0.6)
    for tick in (0, 600, 1200):
        xx = xmap(tick)
        add_line(d, xx, bottom, xx, bottom - 0.9, width=0.4)
        add_text(d, xx, bottom - 3.2, f"{tick:,}", 6.0, anchor="middle")
    add_text(
        d,
        (left + right) / 2,
        y + 1.2,
        "Unique haplotypes per 100 kb",
        6.3,
        anchor="middle",
    )


def draw_panel_d(
    d: Drawing, table: pd.DataFrame, x: float, y: float, w: float, h: float
) -> None:
    panel_header(d, x, y, w, h, "d", "Ancestry composition")
    all_row = table.loc[table["Cohort"] == "All"].iloc[0]
    legend_specs = (
        ("EUR", 9.0, 0.0),
        ("AFR", 31.5, 0.0),
        ("EAS", 53.5, 0.0),
        ("SAS", 15.5, -4.0),
        ("Others", 39.5, -4.0),
    )
    legend_y = y + h - 7.0
    for ancestry, offset, dy in legend_specs:
        lx, ly = x + offset, legend_y + dy
        d.add(
            Rect(
                lx * mm,
                ly * mm,
                2.3 * mm,
                1.8 * mm,
                fillColor=HexColor(ANCESTRY_COLORS[ancestry]),
                strokeColor=None,
            )
        )
        count = int(all_row[f"{ancestry}_count"])
        label = "Other/admixed" if ancestry == "Others" else ancestry
        add_text(d, lx + 3.0, ly - 0.15, f"{label} ({count:,})", 6.0)
    order = [
        "UKB",
        "UK10K",
        "EAS",
        "SG10K",
        "AFR",
        "TOPMed",
        "HGDP",
        "1k genome",
        "All",
    ]
    full_labels = (
        "UKB subset",
        "UK10K",
        "EAS2K",
        "SG10K",
        "AFR1K",
        "TOPMed",
        "HGDP",
        "1000G",
        "hEVO corpus",
    )
    cohort_icons = (
        "UKB",
        "UK10K",
        "EAS",
        "SG10K",
        "AFR",
        "TopMed",
        "HGDP",
        "1kgenome",
        None,
    )
    data = table.set_index("Cohort").loc[order]
    left, right = x + 10.0, x + w - 2.5
    bottom, top = y + 13.5, y + h - 14.0
    bar_step = (right - left) / len(data)
    bar_w = bar_step * 0.68
    for index, (_, row) in enumerate(data.iterrows()):
        center = left + (index + 0.5) * bar_step
        cursor = bottom
        for ancestry in ("EUR", "AFR", "EAS", "SAS", "Others"):
            height = (top - bottom) * float(row[f"{ancestry}_percent"]) / 100.0
            d.add(
                Rect(
                    (center - bar_w / 2) * mm,
                    cursor * mm,
                    bar_w * mm,
                    height * mm,
                    fillColor=HexColor(ANCESTRY_COLORS[ancestry]),
                    strokeColor=None,
                )
            )
            cursor += height
        if row.name == "All":
            d.add(
                Rect(
                    (center - bar_w / 2) * mm,
                    bottom * mm,
                    bar_w * mm,
                    (top - bottom) * mm,
                    fillColor=None,
                    strokeColor=HexColor(BLACK),
                    strokeWidth=0.7,
                )
            )
        icon_key = cohort_icons[index]
        icon_color = COHORT_COLORS[icon_key] if icon_key is not None else GREY
        d.add(
            Circle(
                center * mm,
                (bottom - 1.6) * mm,
                0.72 * mm,
                fillColor=HexColor(icon_color),
                strokeColor=HexColor(BLACK) if row.name == "All" else None,
                strokeWidth=0.25 if row.name == "All" else 0,
            )
        )
        add_rotated_text(
            d,
            center + 0.7,
            bottom - 4.0,
            full_labels[index],
            6.0,
            angle=60.0,
            bold=row.name == "All",
            anchor="end",
        )
    add_line(d, left, bottom, right, bottom, width=0.55)
    add_line(d, left, bottom, left, top, width=0.55)
    for tick in (0, 50, 100):
        yy = bottom + tick / 100 * (top - bottom)
        add_line(d, left, yy, left - 0.9, yy, width=0.4)
        add_text(d, left - 1.5, yy - 0.65, str(tick), 6.0, anchor="end")
    add_rotated_text(d, x + 2.8, (bottom + top) / 2, "Proportion (%)", 6.3, angle=90.0)


def draw_panel_e(
    d: Drawing, table: pd.DataFrame, x: float, y: float, w: float, h: float
) -> None:
    add_text(d, x + 0.3, y + h - 5.3, "e", 14.0, bold=True)
    left, right = x + 11.5, x + w - 2.5
    bottom, top = y + 8.5, y + h - 9.0
    xvals = table["rank"].to_numpy(float) / 1e6
    yvals = table["count"].to_numpy(float)
    rank_max_millions = float(table["rank"].max()) / 1e6
    xmax = max(10.0, math.ceil(rank_max_millions / 10.0) * 10.0)
    ymin, ymax = 0.0, math.log10(max(yvals)) + 0.05
    xmap = lambda value: left + value / xmax * (right - left)
    ymap = lambda value: (
        bottom + (math.log10(max(1.0, value)) - ymin) / (ymax - ymin) * (top - bottom)
    )
    singleton = table.loc[table["count"] == 1, "rank"]
    if not singleton.empty:
        start = float(singleton.min()) / 1e6
        d.add(
            Rect(
                xmap(start) * mm,
                bottom * mm,
                (right - xmap(start)) * mm,
                (top - bottom) * mm,
                fillColor=HexColor("#E8EBF0"),
                strokeColor=None,
            )
        )
    path = RlPath()
    path.moveTo(xmap(xvals[0]) * mm, ymap(yvals[0]) * mm)
    for xv, yv in zip(xvals[1:], yvals[1:]):
        path.lineTo(xmap(xv) * mm, ymap(yv) * mm)
    path.strokeColor = HexColor(BLUE)
    path.strokeWidth = 1.15
    path.fillColor = None
    d.add(path)
    add_line(d, left, bottom, right, bottom, width=0.6)
    add_line(d, left, bottom, left, top, width=0.6)
    for tick in (0, xmax / 2.0, xmax):
        xx = xmap(tick)
        add_line(d, xx, bottom, xx, bottom - 0.9, width=0.4)
        add_text(d, xx, bottom - 3.2, f"{tick:g}", 6.0, anchor="middle")
    for tick, label in ((1, "1"), (100, "100"), (10000, "10k")):
        yy = ymap(tick)
        add_line(d, left, yy, left - 0.9, yy, width=0.4)
        add_text(d, left - 1.5, yy - 0.65, label, 6.0, anchor="end")
    add_text(
        d,
        (left + right) / 2,
        y + 2.0,
        "Reported sequence rank (millions)",
        6.0,
        anchor="middle",
    )
    add_rotated_text(
        d, x + 3.3, (bottom + top) / 2, "Occurrences per sequence", 6.0, angle=90.0
    )
    # S8 is a sparse rank curve.  Its sampled abundance values are not
    # additive; maximum rank is the full histogram's global distinct count.
    global_distinct_sequences = int(table["rank"].max())
    if global_distinct_sequences >= 1e9:
        total_label = f"{global_distinct_sequences / 1e9:.3f}B distinct sequences"
    else:
        total_label = f"{global_distinct_sequences / 1e6:.1f}M distinct sequences"
    add_text(d, right, y + h - 4.8, total_label, 6.0, anchor="end")
    if not singleton.empty:
        total_rank = float(table["rank"].max())
        tail_percent = 100.0 * (total_rank - float(singleton.min()) + 1.0) / total_rank
        # Keep the tail annotation and its leader inside the current panel.  The
        # x-axis is now expressed in millions (0–100), so the old 455-million
        # annotation anchor placed the leader outside the panel/page.
        tx, ty = right - 1.0, ymap(7.5)
        add_text(d, tx, ty + 1.7, "count = 1 tail", 6.0, anchor="end")
        add_text(d, tx, ty - 1.2, f"{tail_percent:.1f}% of rank span", 6.0, anchor="end")
        add_line(d, tx - 16.0, ty - 0.3, xmap(start), ymap(1.1), width=0.4)


def pattern_color(row: pd.Series) -> str:
    active = [
        ancestry
        for ancestry in ("EUR", "AFR", "EAS", "SAS", "Others")
        if int(row[ancestry]) == 1
    ]
    return ANCESTRY_COLORS[active[0]] if len(active) == 1 else GREY


def draw_panel_f(
    d: Drawing, table: pd.DataFrame, x: float, y: float, w: float, h: float
) -> None:
    # Match the reference Fig.1f scale: 0–40% makes the 37% leading share
    # occupy the chart area without changing any source percentages.
    share_y_max = 40.0
    total = float(table["count"].sum())
    if total <= 0:
        raise ValueError("Fig1f count total must be positive")
    data = table.head(12).copy()
    top_share = 100.0 * float(data["count"].sum()) / total
    subtitle = f"Top {len(data)} = {top_share:.1f}% of reported total"
    panel_header(
        d,
        x,
        y,
        w,
        h,
        "f",
        "Exact ancestry-set sharing",
        subtitle,
    )
    add_text(d, x + w / 2, y + h - 4.8, subtitle, 6.0, anchor="middle")
    data["share"] = 100.0 * data["count"] / total
    left, right = x + 10.0, x + w - 2.0
    bar_bottom, bar_top = y + 26.0, y + h - 12.0
    matrix_bottom, matrix_top = y + 5.0, y + 23.8
    step = (right - left) / len(data)
    bar_w = step * 0.66
    for index, (_, row) in enumerate(data.iterrows()):
        center = left + (index + 0.5) * step
        height = float(row["share"]) / share_y_max * (bar_top - bar_bottom)
        d.add(
            Rect(
                (center - bar_w / 2) * mm,
                bar_bottom * mm,
                bar_w * mm,
                height * mm,
                fillColor=HexColor(pattern_color(row)),
                strokeColor=None,
            )
        )
        # Rotated Arial glyphs extend to the visual left of their baseline.
        # A small x compensation centers the visible label on the bar axis.
        add_rotated_text(
            d,
            center + 0.72,
            bar_bottom + height + 1.0,
            f"{float(row['share']):.1f}%",
            6.0,
            angle=90.0,
            anchor="start",
        )
    add_line(d, left, bar_bottom, right, bar_bottom, width=0.5)
    row_names = ("EUR", "AFR", "EAS", "SAS", "Others")
    row_step = (matrix_top - matrix_bottom) / len(row_names)
    for r_index, ancestry in enumerate(row_names):
        yy = matrix_top - (r_index + 0.5) * row_step
        add_text(
            d,
            left - 1.2,
            yy - 0.65,
            "Others" if ancestry == "Others" else ancestry,
            6.0,
            anchor="end",
        )
        for index, (_, row) in enumerate(data.iterrows()):
            center = left + (index + 0.5) * step
            active = int(row[ancestry]) == 1
            d.add(
                Circle(
                    center * mm,
                    yy * mm,
                    (0.82 if active else 0.42) * mm,
                    fillColor=HexColor(
                        ANCESTRY_COLORS[ancestry] if active else INACTIVE
                    ),
                    strokeColor=HexColor(BLACK) if active else None,
                    strokeWidth=0.25 if active else 0,
                )
            )
        active_y = []
    for index, (_, row) in enumerate(data.iterrows()):
        center = left + (index + 0.5) * step
        active_y = [
            matrix_top - (r_index + 0.5) * row_step
            for r_index, ancestry in enumerate(row_names)
            if int(row[ancestry]) == 1
        ]
        if len(active_y) > 1:
            add_line(d, center, min(active_y), center, max(active_y), width=0.4)
        add_text(d, center, matrix_bottom - 2.8, str(index + 1), 6.0, anchor="middle")
    add_line(d, left, matrix_bottom, right, matrix_bottom, width=0.5)


def place_native(d: Drawing, sub: Drawing, x: float, y: float) -> None:
    group = Group(*sub.contents)
    group.transform = (1, 0, 0, 1, x * mm, y * mm)
    d.add(group)


def validate_inputs(
    pca: pd.DataFrame,
    composition: pd.DataFrame,
    c_table: pd.DataFrame,
    curve: pd.DataFrame,
    rank: pd.DataFrame,
    sharing: pd.DataFrame,
) -> dict[str, object]:
    if len(pca) != 63809 or pca["IID"].nunique() != len(pca):
        raise ValueError("S2 PCA row or IID count differs from the current cohort snapshot")
    if set(pca["Group"]) != set(COHORT_ORDER):
        raise ValueError("Unexpected PCA cohort set")
    all_row = composition.loc[composition["Cohort"] == "All"].iloc[0]
    if int(all_row["Total_samples"]) != 63809:
        raise ValueError("S4 final corpus sample total must be 63,809")
    cohort_name_map = {
        "AFR": "AFR",
        "HGDP": "HGDP",
        "1k genome": "1kgenome",
        "EAS": "EAS",
        "UK10K": "UK10K",
        "SG10K": "SG10K",
        "TOPMed": "TopMed",
        "UKB": "UKB",
    }
    observed_group_counts = pca["Group"].value_counts().to_dict()
    non_all = composition.loc[composition["Cohort"] != "All"].set_index("Cohort")
    if set(non_all.index) != set(cohort_name_map):
        raise ValueError("S4 cohort rows do not match the final S2 cohort roster")
    for cohort, group in cohort_name_map.items():
        if int(non_all.loc[cohort, "Total_samples"]) != int(observed_group_counts[group]):
            raise ValueError(f"S4 {cohort} count differs from the final S2 IID count")
    if int(non_all["Total_samples"].sum()) != 63809:
        raise ValueError("S4 non-All cohort counts must sum to 63,809")
    expected_percent = 100.0 * non_all["Total_samples"] / 63809.0
    if not np.allclose(
        non_all["Cohort_percent_of_all_samples"].to_numpy(float),
        expected_percent.to_numpy(float),
        rtol=0.0,
        atol=5e-9,
    ):
        raise ValueError("S4 cohort percentages are not based on N=63,809")
    component_columns = [
        "EUR_count",
        "AFR_count",
        "EAS_count",
        "SAS_count",
        "Others_count",
    ]
    if (
        not composition[component_columns]
        .sum(axis=1)
        .eq(composition["Total_samples"])
        .all()
    ):
        raise ValueError("S2 ancestry counts do not sum to cohort totals")
    if (
        len(c_table) != 8
        or not c_table["relative_to_ukb_reference"].is_monotonic_increasing
    ):
        raise ValueError("Fig1c cumulative series is invalid")
    if (
        curve.empty
        or not curve["minimum_haplotype_frequency_pct"].is_monotonic_increasing
    ):
        raise ValueError("Fig1b curve must be non-empty and frequency-sorted")
    if (
        not rank["rank"].is_monotonic_increasing
        or not rank["count"].is_monotonic_decreasing
    ):
        raise ValueError("S4 rank-abundance ordering is invalid")
    if len(sharing) != 31 or sharing["intersection_rank"].tolist() != list(
        range(1, 32)
    ):
        raise ValueError("S5 must contain the 31 ranked exact intersections")
    return {
        "S2_rows": len(pca),
        "S2_unique_IID": int(pca["IID"].nunique()),
        "S3_total": int(all_row["Total_samples"]),
        "Fig1c_rows": len(c_table),
        "Fig1b_curve_rows": len(curve),
        "S8_rows": len(rank),
        "S9_rows": len(sharing),
    }


def render_figure(
    pca: pd.DataFrame,
    composition: pd.DataFrame,
    c_table: pd.DataFrame,
    curve: pd.DataFrame,
    rank: pd.DataFrame,
    sharing: pd.DataFrame,
    radar_table: pd.DataFrame,
    *,
    pca_mode: str = "3d",
) -> Drawing:
    d = Drawing(PAGE_W_MM * mm, PAGE_H_MM * mm)
    d.add(Rect(0, 0, d.width, d.height, fillColor=HexColor(WHITE), strokeColor=None))

    # Slot-native responsive mosaic. The top and bottom edges of a and b+c are
    # explicitly aligned at 192/284 mm.
    if pca_mode == "3d":
        draw_panel_a(d, pca, 16.0, 192.0, 84.0, 92.0)
    elif pca_mode == "2d":
        draw_panel_a_2d(d, pca, 16.0, 192.0, 84.0, 92.0)
    else:
        raise ValueError(f"Unsupported PCA mode: {pca_mode}")
    selected_ukb_n = int((pca["Group"] == "UKB").sum())
    draw_panel_b(d, curve, 104.0, 242.0, 90.0, 42.0, selected_ukb_n)
    draw_panel_c(d, c_table, 104.0, 192.0, 90.0, 46.0)
    draw_panel_d(d, composition, 16.0, 136.0, 78.0, 52.0)
    draw_panel_e(d, rank, 98.0, 136.0, 46.0, 52.0)
    draw_panel_f(d, sharing, 148.0, 136.0, 46.0, 52.0)

    place_native(d, radars.render_revised(radar_table), 16.0, 10.0)
    return d


def render_png(pdf_path: Path, output_base: Path) -> Path:
    executable = (
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
    if not executable.exists():
        found = shutil.which("pdftoppm")
        if not found:
            raise RuntimeError("pdftoppm is required")
        executable = Path(found)
    subprocess.run(
        [
            str(executable),
            "-f",
            "1",
            "-singlefile",
            "-png",
            "-r",
            str(PNG_DPI),
            str(pdf_path),
            str(output_base),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_base.with_suffix(".png")


def pdf_summary(path: Path) -> dict[str, object]:
    reader = PdfReader(path)
    page = reader.pages[0]
    box = page.mediabox
    fonts = []
    for reference in page.get("/Resources", {}).get("/Font", {}).values():
        font = reference.get_object()
        fonts.append(str(font.get("/BaseFont", "")))
    return {
        "pages": len(reader.pages),
        "width_mm": round(float(box.width) / 72.0 * 25.4, 3),
        "height_mm": round(float(box.height) / 72.0 * 25.4, 3),
        "fonts": sorted(set(fonts)),
        "arial_present": any("Arial" in value for value in fonts),
        "image_xobjects": sum(
            1
            for obj in page.get("/Resources", {}).get("/XObject", {}).values()
            if obj.get_object().get("/Subtype") == "/Image"
        ),
    }


def main() -> None:
    configure_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)

    pca = pd.read_csv(PCA_PATH)
    composition = pd.read_csv(COMPOSITION_PATH)
    c_table = pd.read_csv(C_PATH, sep="\t")
    c_display = c_table[["scenario", "recovered_absolute_per_100kb"]].copy()
    c_display["fold_vs_ukb_subset"] = c_display["recovered_absolute_per_100kb"] / float(
        c_display["recovered_absolute_per_100kb"].iloc[0]
    )
    c_display_path = SOURCE_DIR / "Fig1c_derived_UKB_subset_baseline.csv"
    c_display.to_csv(c_display_path, index=False)
    curve = pd.read_csv(B_PATH)
    rank = pd.read_csv(RANK_PATH)
    sharing = pd.read_csv(SHARING_PATH)
    radar_builder.main()
    radar_table = pd.read_csv(RADAR_REVISED_PATH)
    radar_minima = radars.revised_radial_minima(radar_table)
    architecture_claim_qa = {
        "status": "NOT_USED_IN_ACTIVE_FIG1",
        "note": "The active Fig1 A4 uses the hand-drawn Fig1-g.pdf asset; architecture claims are not an input.",
    }
    input_qa = validate_inputs(pca, composition, c_table, curve, rank, sharing)

    drawing = render_figure(
        pca, composition, c_table, curve, rank, sharing, radar_table, pca_mode="3d"
    )
    pdf_path = OUT_BASE.with_suffix(".pdf")
    svg_path = OUT_BASE.with_suffix(".svg")
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    png_path = render_png(pdf_path, OUT_BASE)

    # Preserve the historical integrated filename and also expose an explicit
    # PC1-PC2-PC3 filename for side-by-side comparison with the 2D alternative.
    pdf_3d_explicit = OUT_BASE_3D_EXPLICIT.with_suffix(".pdf")
    svg_3d_explicit = OUT_BASE_3D_EXPLICIT.with_suffix(".svg")
    png_3d_explicit = OUT_BASE_3D_EXPLICIT.with_suffix(".png")
    for source, target in (
        (pdf_path, pdf_3d_explicit),
        (svg_path, svg_3d_explicit),
        (png_path, png_3d_explicit),
    ):
        shutil.copy2(source, target)

    drawing_2d = render_figure(
        pca, composition, c_table, curve, rank, sharing, radar_table, pca_mode="2d"
    )
    pdf_2d = OUT_BASE_2D.with_suffix(".pdf")
    svg_2d = OUT_BASE_2D.with_suffix(".svg")
    renderPDF.drawToFile(drawing_2d, str(pdf_2d))
    renderSVG.drawToFile(drawing_2d, str(svg_2d))
    png_2d = render_png(pdf_2d, OUT_BASE_2D)

    image = Image.open(png_path)
    svg_text = (
        svg_path.read_text(encoding="utf-8") + svg_2d.read_text(encoding="utf-8")
    ).lower()
    prohibited_pink_hex = ("#d45087", "#f7dfe9", "#e7a4c1", "#fbeaf1")
    qa = {
        "status": STATUS,
        "layout": {
            "page_mm": [PAGE_W_MM, PAGE_H_MM],
            "side_margins_mm": 16.0,
            "module_gutter_mm": 4.0,
            "slots_mm": {
                "a": [84.0, 92.0],
                "b": [90.0, 42.0],
                "c": [90.0, 46.0],
                "d": [78.0, 52.0],
                "e": [46.0, 52.0],
                "f": [46.0, 52.0],
                "g": [178.0, 82.0],
                "h": [178.0, 38.0],
            },
        },
        "visual_contract": {
            "font": "Arial",
            "minimum_font_pt": 6.0,
            "panel_letter_pt": 14.0,
            "text_color": BLACK,
            "chart_gridlines": False,
            "ancestry_palette": ANCESTRY_COLORS,
            "model_family_palette": {
                family: str(style["color"])
                for family, style in radars.REVISED_STYLE.items()
            },
            "stage_I": BLUE,
            "stage_II": COPPER,
            "prohibited_pink_hex_absent": not any(
                code in svg_text for code in prohibited_pink_hex
            ),
        },
        "input_validation": input_qa,
        "architecture_claim_validation": architecture_claim_qa,
        "radar_display_scaling": {
            "transform": "linear rescaling of family-best rank percentile from the displayed category-specific minimum to 1.00",
            "category_minima": radar_minima,
            "selection_rule": "minimum is the 0.05 step below the smallest finite category value after 0.02 headroom; no clipping, re-ranking or imputation",
            "interpretation_boundary": "radial distances are comparable within a category only because the three displayed minima differ",
        },
        "pdf": {
            "pc1_pc2_pc3": pdf_summary(pdf_path),
            "pc1_pc2": pdf_summary(pdf_2d),
        },
        "dense_layer_policy": "Both Fig1a alternatives use a 600-dpi PCA point layer; all axes, labels and other panels remain vector.",
        "png": {
            "dpi": PNG_DPI,
            "pixels": list(image.size),
            "mode": image.mode,
        },
        "outputs": [
            str(pdf_path),
            str(svg_path),
            str(png_path),
            str(pdf_3d_explicit),
            str(svg_3d_explicit),
            str(png_3d_explicit),
            str(pdf_2d),
            str(svg_2d),
            str(png_2d),
        ],
        "output_sha256": {
            "pc1_pc2_pc3_pdf": sha256(pdf_path),
            "pc1_pc2_pc3_svg": sha256(svg_path),
            "pc1_pc2_pc3_png": sha256(png_path),
            "pc1_pc2_pdf": sha256(pdf_2d),
            "pc1_pc2_svg": sha256(svg_2d),
            "pc1_pc2_png": sha256(png_2d),
        },
        "submission_gates": [
            "Fig1a uses the linked 63,809-row final-cohort PCA table.",
            "Fig1b requires the authoritative curve, selected/reference identity reconciliation, comparator and uncertainty.",
            "Fig1c fold labels are derived relative to the digitized UKB-subset absolute value; authoritative absolute estimates and window-bootstrap confidence intervals remain required.",
            "Fig1d requires row-level final-corpus manifest linkage and sample-flow reconciliation.",
            "Fig1g is source-code verified, but the production data manifest, final checkpoint hashes and run-level freeze_summary.json remain required.",
            "Fig1h DeepSEA has no Evo 2 or GPN entry in S14 and is shown as NA; all four displayed families use the same best complete-coverage member per axis rule.",
        ],
    }


if __name__ == "__main__":
    main()
