#!/usr/bin/env python3
"""Render compact, source-backed Fig. 1h family-best rank radars.

The 178 x 38 mm strip is designed for flexible A4 Figure 1 assembly. Radial
distance encodes normalized within-task family-best rank; it is not an effect
size or raw benchmark metric. No concentric gridlines are drawn.
"""

from __future__ import annotations

# [Role] Draw the model-family rank and radar views for Fig. 1h and provide a Drawing for the integrated entry point.
# [Input] S6 or revised Fig. 1h family-profile source data; organize records by family and task before drawing radar axes and polygons.
# [Style] MODEL_FAMILY_STYLE supplies family colors; this file controls the radar grid, labels, legend, and output size.
# [Export] Standalone runs save PDF/SVG/PNG, source data, and QA; the integrated Fig. 1 script reuses the plotting functions.
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.scripts.common.hevo_model_style import MODEL_FAMILY_STYLE
from figures.scripts.common.font_config import register_reportlab_arial, resolve_arial_paths

DEFAULT_INPUT = PROJECT_ROOT / "tables" / "S10_best_model_family_ranks_across_functional_variant_and_evolutionary_tasks.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "figures" / "formal" / "Fig1" / "fig1h_radar_compact_a4_v1"
)

WIDTH_MM = 182.0
HEIGHT_MM = 42.0
PNG_DPI = 600
MM_PER_INCH = 25.4

BLACK = "#000000"
WHITE = "#FFFFFF"
HEVO_COLOR = MODEL_FAMILY_STYLE["hEVO"]["color"]
EVO2_TEAL = "#0E7C86"
NT_GREY = "#8995A5"
GPN_COLOR = MODEL_FAMILY_STYLE["GPN"]["color"]
AXIS_GREY = "#CDD2DA"

FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_PATHS = resolve_arial_paths()

FAMILY_ORDER = ("hEvo family", "Evo2 family", "NT family")
FAMILY_STYLE = {
    "hEvo family": {
        "label": "hEVO",
        "color": HEVO_COLOR,
        "marker": "circle",
        "width": 1.35,
    },
    "Evo2 family": {
        "label": "Evo 2",
        "color": EVO2_TEAL,
        "marker": "triangle",
        "width": 0.9,
    },
    "NT family": {"label": "NT", "color": NT_GREY, "marker": "square", "width": 0.9},
}

CATEGORY_SPECS = (
    (
        "Genomic element classification",
        "Genomic elements",
        ("Genomic Benchmark", "NT Benchmark", "URE Benchmark"),
        {"Genomic Benchmark": "Genomic", "NT Benchmark": "NT", "URE Benchmark": "URE"},
    ),
    (
        "Variant scoring",
        "Variant scoring",
        (
            "ClinVar (coding)",
            "ClinVar (noncoding)",
            "Causal Variants",
            "Rare Causal Variants",
            "Rare Variants",
            "MPRA",
        ),
        {
            "ClinVar (coding)": "ClinVar-C",
            "ClinVar (noncoding)": "ClinVar-NC",
            "Causal Variants": "Causal",
            "Rare Causal Variants": "Rare causal",
            "Rare Variants": "Rare",
            "MPRA": "MPRA",
        },
    ),
    (
        "Evolutionary and ancestry analysis",
        "Evolution & ancestry",
        ("HARs", "HAQERs", "Ancestry analysis", "Adaptation variants"),
        {
            "HARs": "HAR",
            "HAQERs": "HAQER",
            "Ancestry analysis": "Ancestry",
            "Adaptation variants": "Adaptation",
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_fonts() -> None:
    register_reportlab_arial()
    for name, path in FONT_PATHS.items():
        if not path.exists():
            raise RuntimeError(f"Required Arial font is missing: {path}")
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    size: float,
    *,
    anchor: str = "start",
    bold: bool = False,
) -> None:
    drawing.add(
        String(
            x * mm,
            y * mm,
            text,
            fontName=FONT_BOLD if bold else FONT_REGULAR,
            fontSize=size,
            fillColor=HexColor(BLACK),
            textAnchor=anchor,
        )
    )


def validate_and_derive(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "category",
        "task",
        "family",
        "model",
        "metric",
        "rank",
        "n_models",
        "metric_label",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if table[list(required)].isna().any().any():
        raise ValueError("Required radar fields contain missing values")
    if table.duplicated(["category", "task", "family"]).any():
        raise ValueError("Duplicate category-task-family rows")
    if not table["family"].isin(FAMILY_ORDER).all():
        raise ValueError("Unexpected model family in S6")
    if not ((table["rank"] >= 1) & (table["rank"] <= table["n_models"])).all():
        raise ValueError("Rank must lie within 1..n_models")

    expected_categories = {spec[0] for spec in CATEGORY_SPECS}
    if set(table["category"]) != expected_categories:
        raise ValueError("Category set differs from the registered three-panel design")
    for category, _, tasks, _ in CATEGORY_SPECS:
        subset = table.loc[table["category"] == category]
        if set(subset["task"]) != set(tasks):
            raise ValueError(f"Task set differs for {category}")
        if not subset.groupby("task")["family"].nunique().eq(len(FAMILY_ORDER)).all():
            raise ValueError(f"Every task must include all three families: {category}")

    derived = table.copy()
    derived["normalized_family_best_rank"] = (
        derived["n_models"] - derived["rank"] + 1.0
    ) / derived["n_models"]
    if not derived["normalized_family_best_rank"].between(0.0, 1.0).all():
        raise ValueError("Normalized rank must lie within [0, 1]")

    qa = {
        "rows": len(derived),
        "unique_category_task_family": True,
        "all_ranks_in_bounds": True,
        "families": list(FAMILY_ORDER),
        "category_task_counts": {
            category: len(tasks) for category, _, tasks, _ in CATEGORY_SPECS
        },
        "encoding": "(n_models - rank + 1) / n_models; outer radius = better rank",
        "interpretation_boundary": "Rank profile only; not raw metric magnitude or effect size.",
    }
    return derived, qa


def marker(
    drawing: Drawing, x: float, y: float, style: dict[str, object], radius: float = 0.65
) -> None:
    color = HexColor(str(style["color"]))
    shape = style["marker"]
    if shape == "circle":
        drawing.add(
            Circle(
                x * mm,
                y * mm,
                radius * mm,
                fillColor=color,
                strokeColor=color,
                strokeWidth=0.2,
            )
        )
    elif shape == "square":
        drawing.add(
            Rect(
                (x - radius) * mm,
                (y - radius) * mm,
                2 * radius * mm,
                2 * radius * mm,
                fillColor=color,
                strokeColor=color,
                strokeWidth=0.2,
            )
        )
    elif shape == "triangle":
        points = [
            x * mm,
            (y + radius) * mm,
            (x - radius) * mm,
            (y - radius) * mm,
            (x + radius) * mm,
            (y - radius) * mm,
        ]
        drawing.add(
            Polygon(points, fillColor=color, strokeColor=color, strokeWidth=0.2)
        )
    else:
        points = [
            x * mm,
            (y + radius) * mm,
            (x - radius) * mm,
            y * mm,
            x * mm,
            (y - radius) * mm,
            (x + radius) * mm,
            y * mm,
        ]
        drawing.add(
            Polygon(points, fillColor=color, strokeColor=color, strokeWidth=0.2)
        )


def polygon_points(
    cx: float, cy: float, radii: list[float], radius: float
) -> list[float]:
    count = len(radii)
    result: list[float] = []
    for index, value in enumerate(radii):
        angle = math.pi / 2 - 2 * math.pi * index / count
        result.extend(
            [
                (cx + radius * value * math.cos(angle)) * mm,
                (cy + radius * value * math.sin(angle)) * mm,
            ]
        )
    return result


def draw_radar(
    drawing: Drawing,
    table: pd.DataFrame,
    spec: tuple[str, str, tuple[str, ...], dict[str, str]],
    *,
    cx: float,
    cy: float,
    radius: float,
) -> None:
    category, title, tasks, labels = spec
    count = len(tasks)
    outer = polygon_points(cx, cy, [1.0] * count, radius)
    drawing.add(
        Polygon(
            outer, fillColor=None, strokeColor=HexColor(AXIS_GREY), strokeWidth=0.55
        )
    )
    for index, task in enumerate(tasks):
        angle = math.pi / 2 - 2 * math.pi * index / count
        vx = cx + radius * math.cos(angle)
        vy = cy + radius * math.sin(angle)
        drawing.add(
            Line(
                cx * mm,
                cy * mm,
                vx * mm,
                vy * mm,
                strokeColor=HexColor(AXIS_GREY),
                strokeWidth=0.35,
            )
        )
        lx = cx + (radius + 2.1) * math.cos(angle)
        ly = cy + (radius + 2.1) * math.sin(angle) - 0.65
        cosine = math.cos(angle)
        anchor = "middle" if abs(cosine) < 0.25 else "start" if cosine > 0 else "end"
        add_text(drawing, lx, ly, labels[task], 6.0, anchor=anchor)

    subset = table.loc[table["category"] == category]
    # Lower-priority families are drawn first; hEVO remains visible on top.
    for family in reversed(FAMILY_ORDER):
        style = FAMILY_STYLE[family]
        family_rows = subset.loc[subset["family"] == family].set_index("task")
        values = [
            float(family_rows.loc[task, "normalized_family_best_rank"])
            for task in tasks
        ]
        points = polygon_points(cx, cy, values, radius)
        drawing.add(
            Polygon(
                points,
                fillColor=None,
                strokeColor=HexColor(str(style["color"])),
                strokeWidth=float(style["width"]),
            )
        )
        for index, value in enumerate(values):
            angle = math.pi / 2 - 2 * math.pi * index / count
            marker(
                drawing,
                cx + radius * value * math.cos(angle),
                cy + radius * value * math.sin(angle),
                style,
                radius=0.7 if family == "hEvo family" else 0.58,
            )
    add_text(drawing, cx, cy + radius + 4.4, title, 6.6, anchor="middle", bold=True)


def render(table: pd.DataFrame) -> Drawing:
    drawing = Drawing(WIDTH_MM * mm, HEIGHT_MM * mm)
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
    add_text(drawing, 1.0, 31.0, "h", 14.0, bold=True)
    add_text(drawing, 10.0, 32.4, "Family-best rank profiles", 7.2, bold=True)
    add_text(drawing, 10.0, 29.0, "outer = better within-task rank", 6.0)

    legend_x = 113.0
    for family in FAMILY_ORDER:
        style = FAMILY_STYLE[family]
        drawing.add(
            Line(
                legend_x * mm,
                32.5 * mm,
                (legend_x + 5.0) * mm,
                32.5 * mm,
                strokeColor=HexColor(str(style["color"])),
                strokeWidth=float(style["width"]),
            )
        )
        marker(drawing, legend_x + 2.5, 32.5, style, radius=0.58)
        add_text(drawing, legend_x + 6.5, 31.75, str(style["label"]), 6.0)
        legend_x += 20.5

    centers = ((52.0, 13.8, 9.2), (100.0, 13.8, 9.2), (149.0, 13.8, 9.2))
    for spec, (cx, cy, radius) in zip(CATEGORY_SPECS, centers):
        draw_radar(drawing, table, spec, cx=cx, cy=cy, radius=radius)
    return drawing


REVISED_CATEGORY_SPECS = (
    ("Genomic elements", ("Genomic", "NT", "URE", "DeepSEA")),
    ("Variant scoring", ("ClinVar", "De novo", "GWAS causal", "Rare variants", "MPRA")),
    ("Evolution & ancestry", ("HAR", "HAQER", "Ancestry", "Adaptation")),
)

REVISED_STYLE = {
    "hEVO family": FAMILY_STYLE["hEvo family"],
    "Evo 2 family": FAMILY_STYLE["Evo2 family"],
    "NT family": FAMILY_STYLE["NT family"],
    "GPN family": {
        "label": "GPN",
        "color": GPN_COLOR,
        "marker": "diamond",
        "width": 0.9,
    },
}


def revised_radial_minima(table: pd.DataFrame) -> dict[str, float]:
    """Choose category ranges that include every current family-best rank."""
    minima: dict[str, float] = {}
    for title, axes in REVISED_CATEGORY_SPECS:
        values = pd.to_numeric(
            table.loc[table["axis"].isin(axes), "display_radius"], errors="coerce"
        ).dropna()
        if values.empty:
            raise ValueError(f"No finite radar values for {title}")
        minima[title] = max(0.0, math.floor(float(values.min()) * 20.0) / 20.0)
    return minima


def draw_revised_radar(
    drawing: Drawing,
    table: pd.DataFrame,
    title: str,
    axes: tuple[str, ...],
    *,
    cx: float,
    cy: float,
    radius: float,
    radial_min: float,
) -> None:
    count = len(axes)
    outer = polygon_points(cx, cy, [1.0] * count, radius)
    drawing.add(
        Polygon(
            outer, fillColor=None, strokeColor=HexColor(AXIS_GREY), strokeWidth=0.55
        )
    )
    axis_labels = (
        table.drop_duplicates("axis").set_index("axis")["axis_label"].to_dict()
    )
    for index, axis in enumerate(axes):
        angle = math.pi / 2 - 2 * math.pi * index / count
        vx = cx + radius * math.cos(angle)
        vy = cy + radius * math.sin(angle)
        drawing.add(
            Line(
                cx * mm,
                cy * mm,
                vx * mm,
                vy * mm,
                strokeColor=HexColor(AXIS_GREY),
                strokeWidth=0.35,
            )
        )
        lx = cx + (radius + 2.2) * math.cos(angle)
        ly = cy + (radius + 2.2) * math.sin(angle) - 0.65
        cosine = math.cos(angle)
        anchor = "middle" if abs(cosine) < 0.25 else "start" if cosine > 0 else "end"
        add_text(drawing, lx, ly, str(axis_labels[axis]), 6.0, anchor=anchor)

    na_drawn: set[str] = set()
    for family in reversed(("hEVO family", "Evo 2 family", "NT family", "GPN family")):
        style = REVISED_STYLE[family]
        family_rows = table.loc[
            (table["family"] == family) & table["axis"].isin(axes)
        ].set_index("axis")
        values = [
            float(family_rows.loc[axis, "display_radius"])
            if axis in family_rows.index
            and pd.notna(family_rows.loc[axis, "display_radius"])
            else math.nan
            for axis in axes
        ]
        coordinates: list[tuple[float, float] | None] = []
        for index, value in enumerate(values):
            angle = math.pi / 2 - 2 * math.pi * index / count
            if math.isnan(value):
                coordinates.append(None)
                axis = axes[index]
                if axis not in na_drawn:
                    na_x = cx + radius * 0.38 * math.cos(angle)
                    na_y = cy + radius * 0.38 * math.sin(angle) - 0.6
                    add_text(drawing, na_x, na_y, "NA", 6.0, anchor="middle")
                    na_drawn.add(axis)
            else:
                if value < radial_min:
                    raise ValueError(
                        f"Radar value {value} falls below displayed minimum {radial_min}"
                    )
                displayed = (value - radial_min) / (1.0 - radial_min)
                coordinates.append(
                    (
                        cx + radius * displayed * math.cos(angle),
                        cy + radius * displayed * math.sin(angle),
                    )
                )
        for index in range(count):
            first = coordinates[index]
            second = coordinates[(index + 1) % count]
            if first is not None and second is not None:
                drawing.add(
                    Line(
                        first[0] * mm,
                        first[1] * mm,
                        second[0] * mm,
                        second[1] * mm,
                        strokeColor=HexColor(str(style["color"])),
                        strokeWidth=float(style["width"]),
                    )
                )
        for point in coordinates:
            if point is not None:
                marker(
                    drawing,
                    point[0],
                    point[1],
                    style,
                    radius=0.48 if family == "hEVO family" else 0.42,
                )
    add_text(
        drawing,
        cx,
        cy + radius + 4.4,
        f"{title} [{radial_min:.2f}-1.00]",
        6.6,
        anchor="middle",
        bold=True,
    )


def render_revised(table: pd.DataFrame) -> Drawing:
    required = {"category", "axis", "axis_label", "family", "display_radius"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing revised radar fields: {missing}")
    drawing = Drawing(WIDTH_MM * mm, HEIGHT_MM * mm)
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
    add_text(drawing, 1.0, 35.0, "h", 14.0, bold=True)
    add_text(
        drawing,
        181.0,
        0,
        "outer = better family-best rank; category-specific ranges and task counts are data-derived",
        6.0,
        anchor="end",
    )

    legend_x = 64.0
    legend_labels = {
        "hEVO family": "hEVO family",
        "Evo 2 family": "Evo 2 family",
        "NT family": "NT family",
        "GPN family": "GPN family",
    }
    for family in ("hEVO family", "Evo 2 family", "NT family", "GPN family"):
        style = REVISED_STYLE[family]
        drawing.add(
            Line(
                legend_x * mm,
                38.5 * mm,
                (legend_x + 5.0) * mm,
                38.5 * mm,
                strokeColor=HexColor(str(style["color"])),
                strokeWidth=float(style["width"]),
            )
        )
        marker(drawing, legend_x + 2.5, 38.5, style, radius=0.58)
        add_text(drawing, legend_x + 6.3, 37.75, legend_labels[family], 6.0)
        legend_x += 29.5

    # Lift the radar row to reserve a clean bottom-right footnote strip.
    centers = ((38.0, 17.0, 12.0), (91.0, 17.0, 12.0), (144.0, 17.0, 12.0))
    radial_minima = revised_radial_minima(table)
    for (title, axes), (cx, cy, radius) in zip(REVISED_CATEGORY_SPECS, centers):
        draw_revised_radar(
            drawing,
            table,
            title,
            axes,
            cx=cx,
            cy=cy,
            radius=radius,
            radial_min=radial_minima[title],
        )
    return drawing


def render_png_from_pdf(pdf_path: Path, png_path: Path) -> None:
    candidates: list[Path] = []
    on_path = shutil.which("pdftoppm")
    if on_path and on_path.lower().endswith(".exe"):
        candidates.append(Path(on_path))
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
    if bundled.exists():
        candidates.insert(0, bundled)
    if not candidates:
        raise RuntimeError("pdftoppm is required for 600-dpi preview rendering")
    subprocess.run(
        [
            str(candidates[0]),
            "-f",
            "1",
            "-singlefile",
            "-png",
            "-r",
            str(PNG_DPI),
            str(pdf_path),
            str(png_path.with_suffix("")),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def pdf_summary(path: Path) -> dict[str, object]:
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError("Expected one-page PDF")
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) / 72.0 * MM_PER_INCH
    height_mm = float(page.mediabox.height) / 72.0 * MM_PER_INCH
    resources = (page.get("/Resources") or {}).get_object()
    fonts = (resources.get("/Font") or {}).get_object()
    embedded = {}
    for reference in fonts.values():
        font = reference.get_object()
        name = str(font.get("/BaseFont", ""))
        descriptor = font.get("/FontDescriptor")
        if not descriptor and font.get("/DescendantFonts"):
            descriptor = font["/DescendantFonts"][0].get_object().get("/FontDescriptor")
        descriptor = (
            descriptor.get_object() if hasattr(descriptor, "get_object") else descriptor
        )
        embedded[name] = bool(
            descriptor
            and any(
                key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
            )
        )
    arial = [name for name in embedded if "Arial" in name]
    return {
        "size_mm": [round(width_mm, 3), round(height_mm, 3)],
        "size_pass": abs(width_mm - WIDTH_MM) < 0.15
        and abs(height_mm - HEIGHT_MM) < 0.15,
        "fonts": embedded,
        "arial_embedded": bool(arial) and all(embedded[name] for name in arial),
        "native_vector_pdf": bool(fonts) and len(path.read_bytes()) > 5000,
    }


def main() -> None:
    args = parse_args()
    configure_fonts()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    table, data_qa = validate_and_derive(raw)
    derived_path = source_dir / "Fig1h_family_best_normalized_rank_source_data.csv"
    table.to_csv(derived_path, index=False)

    output_base = args.out_dir / "Fig1h_family_best_rank_profiles"
    pdf_path = output_base.with_suffix(".pdf")
    svg_path = output_base.with_suffix(".svg")
    png_path = output_base.with_suffix(".png")
    drawing = render(table)
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    render_png_from_pdf(pdf_path, png_path)

    with Image.open(png_path) as image:
        pixels = list(image.size)
    expected_pixels = [
        round(WIDTH_MM / MM_PER_INCH * PNG_DPI),
        round(HEIGHT_MM / MM_PER_INCH * PNG_DPI),
    ]
    pdf = pdf_summary(pdf_path)
    qa = {
        "status": "DATA_VERIFIED_METHOD_QC_DRAFT",
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
        "derived_source_data": {
            "path": str(derived_path.resolve()),
            "sha256": sha256(derived_path),
        },
        "data_validation": data_qa,
        "visual_contract": {
            "size_mm": [WIDTH_MM, HEIGHT_MM],
            "font": "Arial",
            "minimum_text_pt": 6.0,
            "text_color": BLACK,
            "concentric_gridlines": False,
            "model_family_colors": {
                family: FAMILY_STYLE[family]["color"] for family in FAMILY_ORDER
            },
            "outer_margin_mm": 3.5,
            "inter_radar_gutter_mm": 8.0,
        },
        "pdf": pdf,
        "png": {
            "pixels": pixels,
            "expected_pixels_at_600dpi": expected_pixels,
            "dpi_pass": all(abs(a - b) <= 2 for a, b in zip(pixels, expected_pixels)),
        },
        "outputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (pdf_path, svg_path, png_path)
        ],
    }
    qa["artifact_pass"] = bool(
        pdf["size_pass"]
        and pdf["arial_embedded"]
        and pdf["native_vector_pdf"]
        and qa["png"]["dpi_pass"]
    )


if __name__ == "__main__":
    main()
