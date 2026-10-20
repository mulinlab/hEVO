#!/usr/bin/env python3
"""Render the UCSC-element CH benchmark as four comparable model-family blocks.

The four blocks use a locked horizontal scale so that readers can compare both
within-family model variants and between-family performance.  PCA is encoded by
an open point and t-SNE by a filled point; colour is reserved for model family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect
from reportlab.lib.units import mm


HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import plot_figS2a_ucsc_element_separability as base  # noqa: E402


WIDTH_MM = 180.0
HEIGHT_MM = 76.0
AXIS_MIN = 0.0
AXIS_MAX = 5000.0
FAMILIES = ("hEVO", "NT", "Evo 2", "GPN")
FAMILY_POSITIONS = {
    "hEVO": {"column": "left", "tier": "top"},
    "NT": {"column": "right", "tier": "top"},
    "Evo 2": {"column": "left", "tier": "bottom"},
    "GPN": {"column": "right", "tier": "bottom"},
}


def x_map(value: float, x0: float, x1: float) -> float:
    clipped = min(max(float(value), AXIS_MIN), AXIS_MAX)
    return x0 + (clipped - AXIS_MIN) / (AXIS_MAX - AXIS_MIN) * (x1 - x0)


def family_rows(summary: pd.DataFrame, family: str) -> tuple[pd.DataFrame, list[str]]:
    subset = summary[summary["family"] == family].copy()
    order = (
        subset[subset["method"] == "t-SNE"]
        .sort_values(["ch_index_median", "model_name"], ascending=[False, True], kind="stable")
        ["model_name"]
        .tolist()
    )
    return subset, order


def draw_method_key(drawing: Drawing) -> None:
    y = 67.0
    base.draw_open_point(drawing, 72.0, y, base.BLACK)
    base.add_text(drawing, 74.2, y - 0.7, "PCA", 6.1)
    base.draw_filled_point(drawing, 88.0, y, base.BLACK)
    base.add_text(drawing, 90.2, y - 0.7, "t-SNE", 6.1)


def draw_axis(drawing: Drawing, x0: float, x1: float, y: float) -> None:
    drawing.add(
        Line(
            x0 * mm,
            y * mm,
            x1 * mm,
            y * mm,
            strokeColor=base.color(base.BLACK),
            strokeWidth=0.7,
        )
    )
    for value in range(0, 5001, 1000):
        x = x_map(value, x0, x1)
        drawing.add(
            Line(
                x * mm,
                y * mm,
                x * mm,
                (y - 1.2) * mm,
                strokeColor=base.color(base.BLACK),
                strokeWidth=0.6,
            )
        )
        base.add_text(drawing, x, y - 3.8, f"{value:,}", 6.0, anchor="middle")


def draw_family_block(
    drawing: Drawing,
    summary: pd.DataFrame,
    family: str,
    *,
    label_x: float,
    plot_x0: float,
    plot_x1: float,
    header_y: float,
    row_ys: list[float],
    axis_y: float,
    show_axis: bool = True,
) -> list[str]:
    subset, order = family_rows(summary, family)
    family_color = base.MODEL_FAMILY_STYLE[family]["color"]
    center_x = (label_x + plot_x1) / 2

    base.add_text(drawing, center_x, header_y, family, 7.4, anchor="middle", bold=True)
    drawing.add(
        Line(
            (center_x - 9.0) * mm,
            (header_y - 1.8) * mm,
            (center_x + 9.0) * mm,
            (header_y - 1.8) * mm,
            strokeColor=base.color(family_color),
            strokeWidth=1.25,
        )
    )

    if len(order) > len(row_ys):
        raise ValueError(f"Insufficient row positions for {family}: {len(order)} models")

    for y, model in zip(row_ys, order):
        base.add_text(drawing, label_x, y - 0.72, model, 6.25)
        model_rows = subset[subset["model_name"] == model].set_index("method")
        pca = float(model_rows.loc["PCA", "ch_index_median"])
        tsne = float(model_rows.loc["t-SNE", "ch_index_median"])
        x_pca = x_map(pca, plot_x0, plot_x1)
        x_tsne = x_map(tsne, plot_x0, plot_x1)
        drawing.add(
            Line(
                x_pca * mm,
                y * mm,
                x_tsne * mm,
                y * mm,
                strokeColor=base.color(base.CONNECTOR_GREY),
                strokeWidth=0.65,
            )
        )
        base.draw_open_point(drawing, x_pca, y, family_color)
        base.draw_filled_point(drawing, x_tsne, y, family_color)

    if show_axis:
        draw_axis(drawing, plot_x0, plot_x1, axis_y)
    return order


def build_drawing(table: pd.DataFrame) -> tuple[Drawing, pd.DataFrame, dict[str, list[str]]]:
    base.configure_fonts()
    summary = base.summarize(table)
    summary = summary[summary["family"].isin(FAMILIES)].copy()
    present = set(summary["family"].unique())
    missing = set(FAMILIES) - present
    if missing:
        raise ValueError(f"Missing requested model families: {sorted(missing)}")
    if float(summary["ch_index_q3"].max()) > AXIS_MAX:
        raise ValueError("Observed CH index exceeds the locked 0-5,000 comparison scale")

    drawing = Drawing(WIDTH_MM * mm, HEIGHT_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            WIDTH_MM * mm,
            HEIGHT_MM * mm,
            fillColor=base.color(base.WHITE),
            strokeColor=None,
        )
    )

    base.add_text(drawing, 2.0, 70.0, "b", 14.0, bold=True)
    base.add_text(
        drawing,
        176.0,
        71.0,
        "UCSC genomic elements | five 512-bp classes",
        6.2,
        anchor="end",
    )
    draw_method_key(drawing)

    orders: dict[str, list[str]] = {}
    orders["hEVO"] = draw_family_block(
        drawing,
        summary,
        "hEVO",
        label_x=6.5,
        plot_x0=35.5,
        plot_x1=86.0,
        header_y=61.8,
        row_ys=[56.2, 52.2, 48.2, 44.2, 40.2, 36.2],
        axis_y=0.0,
        show_axis=False,
    )
    orders["NT"] = draw_family_block(
        drawing,
        summary,
        "NT",
        label_x=94.0,
        plot_x0=123.5,
        plot_x1=176.0,
        header_y=61.8,
        row_ys=[56.7, 53.1, 49.5, 45.9, 42.3, 38.7, 35.1],
        axis_y=0.0,
        show_axis=False,
    )
    orders["Evo 2"] = draw_family_block(
        drawing,
        summary,
        "Evo 2",
        label_x=6.5,
        plot_x0=35.5,
        plot_x1=86.0,
        header_y=25.8,
        row_ys=[19.3, 15.0],
        axis_y=8.8,
    )
    orders["GPN"] = draw_family_block(
        drawing,
        summary,
        "GPN",
        label_x=94.0,
        plot_x0=123.5,
        plot_x1=176.0,
        header_y=25.8,
        row_ys=[19.3, 15.0],
        axis_y=8.8,
    )
    base.add_text(
        drawing,
        105.75,
        0.7,
        "Calinski-Harabasz index",
        7.1,
        anchor="middle",
    )
    return drawing, summary, orders


def render(table: pd.DataFrame, output_dir: Path, stem: str) -> dict[str, object]:
    drawing, summary, orders = build_drawing(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    base.render_png_from_pdf(pdf, png)

    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    normalized = source_dir / "FigS2a_UCSC_family_series_plot_source.csv"
    family_order = {family: i for i, family in enumerate(FAMILIES)}
    model_order = {
        model: rank
        for family in FAMILIES
        for rank, model in enumerate(orders[family], start=1)
    }
    source = summary.copy()
    source["family_order"] = source["family"].map(family_order)
    source["within_family_rank"] = source["model_name"].map(model_order)
    source.sort_values(
        ["family_order", "within_family_rank", "method"], kind="stable"
    ).to_csv(normalized, index=False)

    with Image.open(png) as image:
        png_size = list(image.size)
    provenance_values = set(
        table.get("provenance_status", pd.Series(dtype=str)).dropna().astype(str)
    )
    status = (
        "LEGACY_VECTOR_RECOVERED_QC_DRAFT"
        if "LEGACY_VECTOR_RECOVERED_QC_DRAFT" in provenance_values
        else "SOURCE_BACKED"
    )
    qa = {
        "status": status,
        "panel": "Figure S2a family-series view",
        "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
        "families": {family: len(orders[family]) for family in FAMILIES},
        "models": int(sum(len(value) for value in orders.values())),
        "methods": ["PCA", "t-SNE"],
        "metric": "Calinski-Harabasz index",
        "axis": {"minimum": AXIS_MIN, "maximum": AXIS_MAX, "shared": True},
        "sort": "descending t-SNE CH within each family",
        "class_labels": list(base.ELEMENT_LABELS),
        "font": "Arial",
        "minimum_font_pt": 6.0,
        "gridlines": False,
        "text_color": base.BLACK,
        "pdf_inspection": base.inspect_pdf(pdf),
        "png_pixels": png_size,
        "exports": {"pdf": str(pdf), "svg": str(svg), "png": str(png)},
        "hashes": {
            "input": base.sha256(Path(table.attrs["source_path"])),
            "script": base.sha256(HERE),
            "normalized_source": base.sha256(normalized),
        },
    }
    return qa


def parse_args() -> argparse.Namespace:
    default_out = base.PROJECT_ROOT / "figures" / "formal" / "Fig2" / "figS2a_ucsc_elements_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument(
        "--stem",
        default="FigS2a_UCSC_family_series_PCA_tSNE_CH_LEGACY_VECTOR_RECOVERED_QC_DRAFT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = base.read_source(args.input_csv)
    table.attrs["source_path"] = str(args.input_csv.resolve())
    qa = render(table, args.output_dir, args.stem)


if __name__ == "__main__":
    main()
