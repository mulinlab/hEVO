#!/usr/bin/env python3
"""Shared vector/raster-layer implementation for FigS1a and FigS1b.

The active input is the updated 550,237-row S1 PCA table.  The helper keeps the
historical panel geometry while exposing package-local paths for the new flat
FigS1 scripts.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Image as RlImage, Line, Rect, String
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

from common.legacy_loader import COMMON_ROOT, load_module  # noqa: E402
from common.font_config import resolve_arial_paths  # noqa: E402
from common.public_paths import recorded_path  # noqa: E402
from common.paths import cache_dir  # noqa: E402

if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from hevo_palette import (  # noqa: E402
    COHORT_COLORS,
    COHORT_LABELS,
    COHORT_ORDER,
    COHORT_SHORT_LABELS,
)


main_figure = load_module("Fig1/plot_figure1_a4_integrated.py")

PCA_PATH = PROJECT_ROOT / "tables" / "S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "FigS1"
SOURCE_DIR = PROJECT_ROOT / "tables"
LAYER_DIR = cache_dir("figs1_pca")

MODULE_W_MM = 178.0
MODULE_H_MM = 96.0
PNG_DPI = 600
STATUS = "DATA_QC_DRAFT"
PANEL_A_W_MM = 84.0
PANEL_GUTTER_MM = 4.0
PANEL_B_X_MM = PANEL_A_W_MM + PANEL_GUTTER_MM
PANEL_LABEL_SIZE = 12.5
PLOT_BOTTOM_MM = 11.0
PLOT_TOP_MM = 79.0

BLACK = "#000000"
WHITE = "#FFFFFF"
BACKGROUND_POINT = "#C7CDD6"
FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_PATHS = resolve_arial_paths()

EXPECTED_COUNTS = {
    "UKB": 490541,
    "UK10K": 3781,
    "EAS": 3188,
    "SG10K": 4810,
    "AFR": 655,
    "TopMed": 43131,
    "HGDP": 929,
    "1kgenome": 3202,
}


def register_fonts() -> None:
    for name, path in FONT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Required Arial font is absent: {path}")
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    value: str,
    size: float,
    *,
    bold: bool = False,
    anchor: str = "start",
) -> None:
    drawing.add(
        String(
            x * mm,
            y * mm,
            value,
            fontName=FONT_BOLD if bold else FONT_REGULAR,
            fontSize=size,
            fillColor=HexColor(BLACK),
            textAnchor=anchor,
        )
    )


def add_rotated_text(
    drawing: Drawing,
    x: float,
    y: float,
    value: str,
    size: float,
    *,
    angle: float = 90.0,
    anchor: str = "middle",
) -> None:
    group = Group()
    group.add(
        String(
            0,
            0,
            value,
            fontName=FONT_REGULAR,
            fontSize=size,
            fillColor=HexColor(BLACK),
            textAnchor=anchor,
        )
    )
    radians = math.radians(angle)
    group.transform = (
        math.cos(radians),
        math.sin(radians),
        -math.sin(radians),
        math.cos(radians),
        x * mm,
        y * mm,
    )
    drawing.add(group)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nice_domain(values: np.ndarray, padding: float = 0.03) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    return low - padding * span, high + padding * span


def nice_ticks(low: float, high: float, target: int = 4) -> list[float]:
    span = high - low
    rough = span / max(target, 1)
    exponent = math.floor(math.log10(rough))
    fraction = rough / 10**exponent
    step_fraction = next(value for value in (1, 2, 2.5, 5, 10) if fraction <= value)
    step = step_fraction * 10**exponent
    start = math.ceil(low / step) * step
    stop = math.floor(high / step) * step
    count = max(0, int(round((stop - start) / step)))
    return [start + index * step for index in range(count + 1)]


def hex_rgb(value: str) -> tuple[int, int, int]:
    clean = value.lstrip("#")
    return tuple(int(clean[index:index + 2], 16) for index in (0, 2, 4))


def coordinate_maps(
    box: tuple[float, float, float, float],
    domains: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    left, bottom, right, top = box
    (xmin, xmax), (ymin, ymax) = domains
    return (
        lambda value: left + (value - xmin) / (xmax - xmin) * (right - left),
        lambda value: bottom + (value - ymin) / (ymax - ymin) * (top - bottom),
    )


def draw_axes(
    drawing: Drawing,
    box: tuple[float, float, float, float],
    domains: tuple[tuple[float, float], tuple[float, float]],
    *,
    show_x_tick_labels: bool = True,
    show_y_tick_labels: bool = True,
    show_x_title: bool = False,
    show_y_title: bool = False,
    tick_size: float = 6.0,
    title_size: float = 7.0,
) -> None:
    left, bottom, right, top = box
    xmap, ymap = coordinate_maps(box, domains)
    add_line(drawing, left, bottom, right, bottom, width=0.55)
    add_line(drawing, left, bottom, left, top, width=0.55)
    x_ticks_scaled = nice_ticks(domains[0][0] * 1000, domains[0][1] * 1000, 3)
    y_ticks_scaled = nice_ticks(domains[1][0] * 1000, domains[1][1] * 1000, 3)
    for tick in x_ticks_scaled:
        xpos = xmap(tick / 1000)
        add_line(drawing, xpos, bottom, xpos, bottom - 0.8, width=0.4)
        if show_x_tick_labels:
            add_text(drawing, xpos, bottom - 3.0, f"{tick:g}", tick_size, anchor="middle")
    for tick in y_ticks_scaled:
        ypos = ymap(tick / 1000)
        add_line(drawing, left - 0.8, ypos, left, ypos, width=0.4)
        if show_y_tick_labels:
            add_text(drawing, left - 1.3, ypos - 0.65, f"{tick:g}", tick_size, anchor="end")
    if show_x_title:
        add_text(drawing, (left + right) / 2, bottom - 6.6, "PC1 (×1,000)", title_size, anchor="middle")
    if show_y_title:
        add_rotated_text(drawing, left - 8.2, (bottom + top) / 2, "PC2 (×1,000)", title_size)


def make_all_cohort_layer(
    frame: pd.DataFrame,
    path: Path,
    width_mm: float,
    height_mm: float,
    domains: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    width_px = max(100, round(width_mm / 25.4 * PNG_DPI))
    height_px = max(100, round(height_mm / 25.4 * PNG_DPI))
    image = Image.new("RGBA", (width_px, height_px), (255, 255, 255, 0))
    counts = frame["Group"].value_counts().to_dict()
    (xmin, xmax), (ymin, ymax) = domains
    for cohort in sorted(COHORT_ORDER, key=lambda key: counts[key], reverse=True):
        part = frame.loc[frame["Group"] == cohort]
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        xs = np.rint((part["PC1"].to_numpy(float) - xmin) / (xmax - xmin) * (width_px - 1)).astype(int)
        ys = np.rint((1 - (part["PC2"].to_numpy(float) - ymin) / (ymax - ymin)) * (height_px - 1)).astype(int)
        rgb = hex_rgb(COHORT_COLORS[cohort])
        radius = 2 if len(part) > 10000 else 3
        for px, py in zip(xs, ys):
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(*rgb, 92))
        image = Image.alpha_composite(image, overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(PNG_DPI, PNG_DPI), optimize=True)


def make_highlight_layer(
    frame: pd.DataFrame,
    cohort: str,
    path: Path,
    width_mm: float,
    height_mm: float,
    domains: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    width_px = max(80, round(width_mm / 25.4 * PNG_DPI))
    height_px = max(80, round(height_mm / 25.4 * PNG_DPI))
    image = Image.new("RGBA", (width_px, height_px), (255, 255, 255, 0))
    (xmin, xmax), (ymin, ymax) = domains

    def pixels(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xs = np.rint((part["PC1"].to_numpy(float) - xmin) / (xmax - xmin) * (width_px - 1)).astype(int)
        ys = np.rint((1 - (part["PC2"].to_numpy(float) - ymin) / (ymax - ymin)) * (height_px - 1)).astype(int)
        return xs, ys

    background = Image.new("RGBA", image.size, (255, 255, 255, 0))
    background_draw = ImageDraw.Draw(background, "RGBA")
    back_x, back_y = pixels(frame)
    back_rgb = hex_rgb(BACKGROUND_POINT)
    for px, py in zip(back_x, back_y):
        background_draw.point((px, py), fill=(*back_rgb, 46))
    image = Image.alpha_composite(image, background)

    foreground = Image.new("RGBA", image.size, (255, 255, 255, 0))
    foreground_draw = ImageDraw.Draw(foreground, "RGBA")
    part = frame.loc[frame["Group"] == cohort]
    fore_x, fore_y = pixels(part)
    fore_rgb = hex_rgb(COHORT_COLORS[cohort])
    radius = 1 if len(part) > 10000 else 2
    alpha = 115 if len(part) > 10000 else 155
    for px, py in zip(fore_x, fore_y):
        foreground_draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(*fore_rgb, alpha))
    image = Image.alpha_composite(image, foreground)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(PNG_DPI, PNG_DPI), optimize=True)


def draw_cohort_legend(drawing: Drawing, frame: pd.DataFrame, x: float, y: float) -> None:
    counts = frame["Group"].value_counts().to_dict()
    for index, cohort in enumerate(COHORT_ORDER):
        row, column = divmod(index, 2)
        xpos = x + column * 38.0
        ypos = y - row * 3.8
        drawing.add(
            Circle(
                xpos * mm,
                (ypos + 0.65) * mm,
                0.72 * mm,
                fillColor=HexColor(COHORT_COLORS[cohort]),
                strokeColor=None,
            )
        )
        add_text(drawing, xpos + 2.0, ypos, f"{COHORT_LABELS[cohort]} ({counts[cohort]:,})", 6.0)


def draw_left_2d(
    drawing: Drawing,
    frame: pd.DataFrame,
    domains: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    add_text(drawing, 1.5, 89.5, "a", PANEL_LABEL_SIZE, bold=True)
    draw_cohort_legend(drawing, frame, 10.0, 92.0)
    # Square field aligned exactly to the top and bottom of the Fig. S1b grid.
    box = (14.0, PLOT_BOTTOM_MM, 82.0, PLOT_TOP_MM)
    layer = LAYER_DIR / "FigS1a_all_cohorts_PC1_PC2_600dpi.png"
    make_all_cohort_layer(frame, layer, box[2] - box[0], box[3] - box[1], domains)
    drawing.add(RlImage(box[0] * mm, box[1] * mm, (box[2] - box[0]) * mm, (box[3] - box[1]) * mm, str(layer)))
    draw_axes(drawing, box, domains, show_x_title=True, show_y_title=True, tick_size=6.0, title_size=7.0)


def draw_left_3d(drawing: Drawing, frame: pd.DataFrame) -> None:
    add_text(drawing, 1.5, 89.5, "a", PANEL_LABEL_SIZE, bold=True)
    draw_cohort_legend(drawing, frame, 10.0, 92.0)
    left, bottom, right, top = 14.0, PLOT_BOTTOM_MM, 82.0, PLOT_TOP_MM
    layer = LAYER_DIR / "FigS1a_all_cohorts_PC1_PC2_PC3_600dpi.png"
    axes = main_figure.make_pca_3d_layer(frame, layer, right - left, top - bottom)
    drawing.add(RlImage(left * mm, bottom * mm, (right - left) * mm, (top - bottom) * mm, str(layer)))

    def map_axis(point: tuple[float, float]) -> tuple[float, float]:
        return left + point[0] * (right - left), bottom + point[1] * (top - bottom)

    origin = map_axis(axes["origin"])
    label_offsets = {"PC1": (1.7, -0.9), "PC2": (-1.7, -0.9), "PC3": (0.0, 1.2)}
    for axis in ("PC1", "PC2", "PC3"):
        endpoint = map_axis(axes[axis])
        if axis == "PC3":
            scale = (top - 2.5 - origin[1]) / (endpoint[1] - origin[1])
            endpoint = (
                origin[0] + scale * (endpoint[0] - origin[0]),
                origin[1] + scale * (endpoint[1] - origin[1]),
            )
        add_line(drawing, origin[0], origin[1], endpoint[0], endpoint[1], width=0.55)
        dx, dy = label_offsets[axis]
        add_text(drawing, endpoint[0] + dx, endpoint[1] + dy, axis, 6.3, bold=True, anchor="middle")


def draw_highlight_facets(
    drawing: Drawing,
    frame: pd.DataFrame,
    domains: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    panel_x, panel_w = PANEL_B_X_MM, 90.0
    add_text(drawing, panel_x - 1.5, 89.5, "b", PANEL_LABEL_SIZE, bold=True)
    inner_left = panel_x + 8.0
    inner_right = panel_x + panel_w - 1.0
    gap = 2.5
    facet_w = (inner_right - inner_left - 3 * gap) / 4
    facet_h = 28.0
    row_bottoms = (51.0, PLOT_BOTTOM_MM)
    counts = frame["Group"].value_counts().to_dict()

    for index, cohort in enumerate(COHORT_ORDER):
        row, column = divmod(index, 4)
        left = inner_left + column * (facet_w + gap)
        bottom = row_bottoms[row]
        box = (left, bottom, left + facet_w, bottom + facet_h)
        add_text(
            drawing,
            left + facet_w / 2,
            bottom + facet_h + 2.0,
            f"{COHORT_SHORT_LABELS[cohort]} ({counts[cohort]:,})",
            6.0,
            anchor="middle",
        )
        layer = LAYER_DIR / f"FigS1b_{cohort}_highlight_PC1_PC2_600dpi.png"
        make_highlight_layer(frame, cohort, layer, facet_w, facet_h, domains)
        drawing.add(RlImage(left * mm, bottom * mm, facet_w * mm, facet_h * mm, str(layer)))
        draw_axes(
            drawing,
            box,
            domains,
            show_x_tick_labels=row == 1,
            show_y_tick_labels=column == 0,
            tick_size=6.0,
        )

    add_text(drawing, (inner_left + inner_right) / 2, PLOT_BOTTOM_MM - 6.6, "PC1 (×1,000)", 7.0, anchor="middle")
    add_rotated_text(drawing, panel_x + 0.7, 45.0, "PC2 (×1,000)", 7.0)


def render_module(frame: pd.DataFrame, left_mode: str) -> Drawing:
    drawing = Drawing(MODULE_W_MM * mm, MODULE_H_MM * mm)
    drawing.add(Rect(0, 0, drawing.width, drawing.height, fillColor=HexColor(WHITE), strokeColor=None))
    domains = (
        nice_domain(frame["PC1"].to_numpy(float)),
        nice_domain(frame["PC2"].to_numpy(float)),
    )
    if left_mode == "2d":
        draw_left_2d(drawing, frame, domains)
    elif left_mode == "3d":
        draw_left_3d(drawing, frame)
    else:
        raise ValueError(f"Unsupported left panel mode: {left_mode}")
    draw_highlight_facets(drawing, frame, domains)
    return drawing


def embed_svg_images(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    marker = 'xlink:href="'
    cursor = 0
    rebuilt: list[str] = []
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            rebuilt.append(text[cursor:])
            break
        rebuilt.append(text[cursor:start + len(marker)])
        end = text.find('"', start + len(marker))
        href = text[start + len(marker):end]
        candidate = Path(href)
        if candidate.exists() and candidate.suffix.lower() == ".png":
            payload = base64.b64encode(candidate.read_bytes()).decode("ascii")
            rebuilt.append(f"data:image/png;base64,{payload}")
        else:
            rebuilt.append(href)
        cursor = end
    svg_path.write_text("".join(rebuilt), encoding="utf-8")


def render_png(pdf_path: Path, output_base: Path) -> Path:
    resolved = shutil.which("pdftoppm")
    if resolved is None:
        raise FileNotFoundError("pdftoppm is required for visual QA")
    poppler = Path(resolved)
    subprocess.run(
        [str(poppler), "-png", "-r", str(PNG_DPI), "-singlefile", str(pdf_path), str(output_base)],
        check=True,
    )
    return output_base.with_suffix(".png")


def save_variant(frame: pd.DataFrame, left_mode: str, stem: str) -> list[Path]:
    base = OUTPUT_DIR / stem
    drawing = render_module(frame, left_mode)
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    embed_svg_images(svg_path)
    png_path = render_png(pdf_path, base)
    return [pdf_path, svg_path, png_path]


def pdf_summary(path: Path) -> dict[str, object]:
    reader = PdfReader(path)
    page = reader.pages[0]
    font_names: set[str] = set()
    image_xobjects = 0
    resources = page.get("/Resources") or {}
    resources = resources.get_object() if hasattr(resources, "get_object") else resources
    fonts = resources.get("/Font") or {}
    fonts = fonts.get_object() if hasattr(fonts, "get_object") else fonts
    for reference in fonts.values():
        font_names.add(str(reference.get_object().get("/BaseFont", "")))
    def count_images(resource_dict: object) -> int:
        resource_dict = resource_dict.get_object() if hasattr(resource_dict, "get_object") else resource_dict
        if not resource_dict:
            return 0
        xobjects = resource_dict.get("/XObject") or {}
        xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
        count = 0
        for reference in xobjects.values():
            item = reference.get_object()
            subtype = str(item.get("/Subtype"))
            if subtype == "/Image":
                count += 1
            elif subtype == "/Form":
                count += count_images(item.get("/Resources") or {})
        return count

    image_xobjects = count_images(resources)
    page_images = len(list(page.images))
    extracted_text = page.extract_text() or ""
    required_labels = (
        "UK Biobank (491,027)",
        "1000 Genomes (3,202)",
        "PC1 (×1,000)",
        "PC2 (×1,000)",
        "EAS cohort (2,727)",
        "1000G (3,202)",
    )
    return {
        "pages": len(reader.pages),
        "width_mm": round(float(page.mediabox.width) / 72 * 25.4, 2),
        "height_mm": round(float(page.mediabox.height) / 72 * 25.4, 2),
        "fonts": sorted(font_names),
        "arial_present": any("Arial" in name for name in font_names),
        "image_xobjects_recursive": image_xobjects,
        "embedded_dense_layers": page_images,
        "text_glyph_summary": {
            "replacement_character_present": "\ufffd" in extracted_text,
            "black_square_present": "□" in extracted_text,
            "missing_required_labels": [label for label in required_labels if label not in extracted_text],
        },
    }


def artifact_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": recorded_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix.lower() == ".pdf":
        record.update(pdf_summary(path))
    elif path.suffix.lower() == ".png":
        with Image.open(path) as image:
            record["pixels"] = list(image.size)
            record["dpi"] = [round(value, 1) for value in image.info.get("dpi", (0, 0))]
    return record


def load_source(tables_dir: Path | None = None) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    source = (tables_dir or SOURCE_DIR) / "S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv"
    frame = pd.read_csv(source, dtype={"IID": str}, low_memory=False)
    return frame, source, validate(frame)


def validate(frame: pd.DataFrame) -> dict[str, object]:
    required = {"IID", "Group", *[f"PC{index}" for index in range(1, 11)]}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"S1 is missing required columns: {missing_columns}")
    pc_columns = [f"PC{index}" for index in range(1, 11)]
    if frame.empty or not frame["IID"].is_unique:
        raise ValueError("S1 IID uniqueness contract failed")
    counts = {key: int(value) for key, value in frame["Group"].value_counts().to_dict().items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"S1 cohort counts differ from the updated snapshot: {counts}")
    numeric = frame[pc_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("S1 PC coordinates contain missing or non-finite values")
    return {
        "grain": "one row per S1 IID",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "unique_IID": int(frame["IID"].nunique()),
        "duplicate_IID_rows": int(frame["IID"].duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "nonfinite_PC_cells": int((~np.isfinite(numeric.to_numpy(float))).sum()),
        "cohort_counts": counts,
        "unexpected_cohorts": sorted(set(frame["Group"]) - set(COHORT_ORDER)),
        "coordinate_ranges": {
            pc: {
                "min": float(numeric[pc].min()),
                "max": float(numeric[pc].max()),
                "q001": float(numeric[pc].quantile(0.001)),
                "q999": float(numeric[pc].quantile(0.999)),
            }
            for pc in ("PC1", "PC2", "PC3")
        },
    }


def crop_module(frame: pd.DataFrame, panel: str) -> Drawing:
    combined = render_module(frame, "2d")
    if panel == "a":
        width_mm, shift_mm = PANEL_A_W_MM, 0.0
    elif panel == "b":
        width_mm, shift_mm = 96.0, -(PANEL_B_X_MM - 6.0)
    else:
        raise ValueError(f"Unknown PCA panel: {panel}")
    drawing = Drawing(width_mm * mm, MODULE_H_MM * mm)
    drawing.add(Rect(0, 0, drawing.width, drawing.height, fillColor=HexColor(WHITE), strokeColor=None))
    group = Group()
    for item in combined.contents:
        group.add(item)
    group.transform = (1, 0, 0, 1, shift_mm * mm, 0)
    drawing.add(group)
    return drawing
