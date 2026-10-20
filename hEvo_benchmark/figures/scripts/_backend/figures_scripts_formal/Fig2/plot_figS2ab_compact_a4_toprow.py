#!/usr/bin/env python3
"""Render compact Fig. S2a-b as an A4-width top-row assembly.

Panel a shows representative algorithms; panel b shows four model-family
series.  The 190 x 88 mm native vector layout occupies less than one third of
an A4 page height and leaves the remainder for subsequent supplementary panels.
"""

from __future__ import annotations

# [Legacy/fallback] Compact top-row plotter for Fig. S2a/b, retained to reproduce earlier Fig. 2a/b or supplementary figures.
# [Note] This is not the current S11–S16 entry point for Fig. 2a/b; use plot_fig2ab_s11_s16.py.
# [Style] Retains compact font sizes, colors, markers, and panel Drawing for the legacy top row.

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect
from reportlab.lib.units import mm


HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import plot_figS2a_best_algorithm_representatives as representative  # noqa: E402
import plot_figS2a_family_series as family_series  # noqa: E402
import plot_figS2a_ucsc_element_separability as base  # noqa: E402


WIDTH_MM = 190.0
HEIGHT_MM = 76.0
AXIS_MAX = 5000.0
PANEL_DIVIDER_X = 94.0


def map_x(value: float, x0: float, x1: float) -> float:
    clipped = min(max(float(value), 0.0), AXIS_MAX)
    return x0 + clipped / AXIS_MAX * (x1 - x0)


def draw_axis(
    drawing: Drawing,
    x0: float,
    x1: float,
    y: float,
    *,
    ticks: tuple[int, ...],
    font_size: float = 6.0,
) -> None:
    drawing.add(
        Line(
            x0 * mm,
            y * mm,
            x1 * mm,
            y * mm,
            strokeColor=base.color(base.BLACK),
            strokeWidth=0.65,
        )
    )
    for value in ticks:
        x = map_x(value, x0, x1)
        drawing.add(
            Line(
                x * mm,
                y * mm,
                x * mm,
                (y - 1.0) * mm,
                strokeColor=base.color(base.BLACK),
                strokeWidth=0.55,
            )
        )
        label = f"{value:,}" if value not in (2500,) else "2,500"
        base.add_text(drawing, x, y - 3.2, label, font_size, anchor="middle")


def draw_method_key(drawing: Drawing, x: float, y: float) -> None:
    base.draw_open_point(drawing, x, y, base.BLACK)
    base.add_text(drawing, x + 2.1, y - 0.68, "PCA", 6.0)
    base.draw_filled_point(drawing, x + 14.0, y, base.BLACK)
    base.add_text(drawing, x + 16.1, y - 0.68, "t-SNE", 6.0)


def draw_panel_a_focus_legend(drawing: Drawing) -> None:
    y = 65.8
    x = 4.5
    for family_color, label, advance in (
        (base.MODEL_FAMILY_STYLE["hEVO"]["color"], "hEVO", 19.0),
        (base.MID_GREY, "Other models", 28.0),
    ):
        base.draw_filled_point(drawing, x, y, family_color)
        base.add_text(drawing, x + 2.0, y - 0.68, label, 6.0)
        x += advance


def draw_panel_a(
    drawing: Drawing, selected: pd.DataFrame, order: list[str]
) -> None:
    base.add_text(drawing, 1.2, 70.0, "a", 13.0, bold=True)
    draw_method_key(drawing, 36.5, 70.1)
    draw_panel_a_focus_legend(drawing)

    label_x = 4.5
    plot_x0 = 36.0
    plot_x1 = 92.0
    row_top = 61.2
    # Lock the same 2.15-mm model-row pitch used in Figure 2c-d.
    row_bottom = row_top - 2.15 * (len(order) - 1)
    step = (row_top - row_bottom) / (len(order) - 1)

    for index, group in enumerate(order):
        y = row_top - index * step
        rows = selected[selected["algorithm_group"] == group].set_index("method")
        model = str(rows.iloc[0]["selected_model"])
        label = str(rows.iloc[0]["display_label"])
        family = base.model_family(model)
        # Panel a encodes the focal comparison only: hEVO versus all other
        # algorithms. Family-specific colors remain exclusive to panel b.
        family_color = (
            base.MODEL_FAMILY_STYLE["hEVO"]["color"]
            if family == "hEVO"
            else base.MID_GREY
        )
        if family == "hEVO":
            drawing.add(
                Rect(
                    2.3 * mm,
                    (y - step * 0.43) * mm,
                    90.0 * mm,
                    step * 0.86 * mm,
                    fillColor=base.color(base.HEVO_WASH),
                    strokeColor=None,
                )
            )
        base.add_text(
            drawing,
            label_x,
            y - 0.68,
            label,
            6.0,
            bold=family == "hEVO",
        )
        pca = float(rows.loc["PCA", "ch_index_median"])
        tsne = float(rows.loc["t-SNE", "ch_index_median"])
        x_pca = map_x(pca, plot_x0, plot_x1)
        x_tsne = map_x(tsne, plot_x0, plot_x1)
        drawing.add(
            Line(
                x_pca * mm,
                y * mm,
                x_tsne * mm,
                y * mm,
                strokeColor=base.color(base.CONNECTOR_GREY),
                strokeWidth=0.6,
            )
        )
        base.draw_open_point(drawing, x_pca, y, family_color)
        base.draw_filled_point(drawing, x_tsne, y, family_color)

    draw_axis(
        drawing,
        plot_x0,
        plot_x1,
        19.0,
        ticks=(0, 1000, 2000, 3000, 4000, 5000),
    )
    base.add_text(
        drawing,
        (plot_x0 + plot_x1) / 2,
        11.8,
        "Calinski-Harabasz index",
        6.6,
        anchor="middle",
    )


SHORT_LABELS = {
    "hEvo-large": "large-319.462M",
    "hEvo-base": "base-87.095M",
    "hEVO-1000G-s128-86M": "1000G-s128-86M",
    "hEVO-full-UKB-86M": "full-UKB-86M",
    "hEVO-60K-86M": "60K-86M",
    "hEVO-1000G-86M": "1000G-86M",
    "NTv3-100M": "v3-100M",
    "NTv3-8M": "v3-8M",
    "NTv3-650M": "v3-650M",
    "NTv2-MS-500M": "v2-MS-500M",
    "NTv1-1000G-2.5B": "v1-1000G-2.5B",
    "NTv1-1000G-500M": "v1-1000G-500M",
    "NTv1-MS-2.5B": "v1-MS-2.5B",
    "EVO 2-7B": "7B",
    "EVO 2-1B": "1B",
    "GPN-Promoter": "Promoter",
    "GPN-MSA": "MSA",
}


def draw_compact_family(
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
    subset, order = family_series.family_rows(summary, family)
    family_color = base.MODEL_FAMILY_STYLE[family]["color"]
    center_x = (label_x + plot_x1) / 2
    family_label = "EVO 2" if family == "Evo 2" else family
    base.add_text(drawing, center_x, header_y, family_label, 6.6, anchor="middle", bold=True)
    drawing.add(
        Line(
            (center_x - 6.5) * mm,
            (header_y - 1.45) * mm,
            (center_x + 6.5) * mm,
            (header_y - 1.45) * mm,
            strokeColor=base.color(family_color),
            strokeWidth=1.0,
        )
    )
    if len(order) > len(row_ys):
        raise ValueError(f"Insufficient rows for compact {family} block")
    for y, model in zip(row_ys, order):
        base.add_text(drawing, label_x, y - 0.66, SHORT_LABELS[model], 6.0)
        rows = subset[subset["model_name"] == model].set_index("method")
        pca = float(rows.loc["PCA", "ch_index_median"])
        tsne = float(rows.loc["t-SNE", "ch_index_median"])
        x_pca = map_x(pca, plot_x0, plot_x1)
        x_tsne = map_x(tsne, plot_x0, plot_x1)
        drawing.add(
            Line(
                x_pca * mm,
                y * mm,
                x_tsne * mm,
                y * mm,
                strokeColor=base.color(base.CONNECTOR_GREY),
                strokeWidth=0.55,
            )
        )
        base.draw_open_point(drawing, x_pca, y, family_color)
        base.draw_filled_point(drawing, x_tsne, y, family_color)
    if show_axis:
        draw_axis(
            drawing,
            plot_x0,
            plot_x1,
            axis_y,
            ticks=(0, 2500, 5000),
            font_size=6.0,
        )
    return order


def draw_panel_b(drawing: Drawing, summary: pd.DataFrame) -> dict[str, list[str]]:
    base.add_text(drawing, 95.2, 70.0, "b", 13.0, bold=True)
    draw_method_key(drawing, 119.0, 70.1)
    orders: dict[str, list[str]] = {}
    orders["hEVO"] = draw_compact_family(
        drawing,
        summary,
        "hEVO",
        label_x=97.5,
        plot_x0=120.2,
        plot_x1=141.5,
        header_y=64.0,
        row_ys=[58.5, 54.5, 50.5, 46.5, 42.5, 38.5],
        axis_y=0.0,
        show_axis=False,
    )
    orders["NT"] = draw_compact_family(
        drawing,
        summary,
        "NT",
        label_x=144.5,
        plot_x0=166.5,
        plot_x1=186.2,
        header_y=64.0,
        row_ys=[59.0, 55.4, 51.8, 48.2, 44.6, 41.0, 37.4],
        axis_y=0.0,
        show_axis=False,
    )
    orders["Evo 2"] = draw_compact_family(
        drawing,
        summary,
        "Evo 2",
        label_x=97.5,
        plot_x0=120.2,
        plot_x1=141.5,
        header_y=32.5,
        row_ys=[27.0, 23.0],
        axis_y=19.0,
    )
    orders["GPN"] = draw_compact_family(
        drawing,
        summary,
        "GPN",
        label_x=144.5,
        plot_x0=166.5,
        plot_x1=186.2,
        header_y=32.5,
        row_ys=[27.0, 23.0],
        axis_y=19.0,
    )
    base.add_text(
        drawing,
        154.35,
        11.8,
        "Calinski-Harabasz index",
        6.6,
        anchor="middle",
    )
    return orders


def build_drawing(
    table: pd.DataFrame,
) -> tuple[Drawing, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    base.configure_fonts()
    summary = base.summarize(table)
    selected, summary = representative.select_representatives(summary)
    order = (
        selected[["algorithm_group", "representative_rank"]]
        .drop_duplicates()
        .sort_values("representative_rank", kind="stable")["algorithm_group"]
        .tolist()
    )
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
    base.add_text(
        drawing,
        188.5,
        72.2,
        "UCSC genomic elements | five 512-bp classes",
        6.0,
        anchor="end",
    )
    draw_panel_a(drawing, selected, order)
    family_orders = draw_panel_b(drawing, summary)
    return drawing, selected, summary, family_orders


def render(table: pd.DataFrame, output_dir: Path, stem: str) -> dict[str, object]:
    drawing, selected, summary, family_orders = build_drawing(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    base.render_png_from_pdf(pdf, png)

    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_a = source_dir / "FigS2ab_panel_a_representatives_plot_source.csv"
    source_b = source_dir / "FigS2ab_panel_b_family_series_plot_source.csv"
    selection_summary = source_dir / "FigS2ab_panel_a_selection_summary.csv"
    selected.sort_values(["representative_rank", "method"], kind="stable").to_csv(
        source_a, index=False
    )
    family_rank = {
        model: index
        for family in family_series.FAMILIES
        for index, model in enumerate(family_orders[family], start=1)
    }
    panel_b_source = base.summarize(table)
    panel_b_source = panel_b_source[
        panel_b_source["family"].isin(family_series.FAMILIES)
    ].copy()
    panel_b_source["within_family_rank"] = panel_b_source["model_name"].map(family_rank)
    panel_b_source.sort_values(
        ["family", "within_family_rank", "method"], kind="stable"
    ).to_csv(source_b, index=False)
    summary.to_csv(selection_summary, index=False)

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
        "panel": "Figure S2a-b compact A4 top row",
        "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
        "a4_height_fraction": round(HEIGHT_MM / 297.0, 3),
        "panel_a_representatives": int(selected["algorithm_group"].nunique()),
        "panel_a_removed": ["NTv1-1000G", "NTv1-MS-2.5B", "Caduceus-PS"],
        "panel_a_hEVO_rows": ["hEvo-large", "hEvo-base"],
        "panel_b_families": {key: len(value) for key, value in family_orders.items()},
        "shared_axis": [0.0, AXIS_MAX],
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
            "panel_a_source": base.sha256(source_a),
            "panel_b_source": base.sha256(source_b),
            "selection_summary": base.sha256(selection_summary),
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
        default="FigS2ab_UCSC_CH_compact_A4_toprow_LEGACY_VECTOR_RECOVERED_QC_DRAFT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = base.read_source(args.input_csv)
    table.attrs["source_path"] = str(args.input_csv.resolve())
    qa = render(table, args.output_dir, args.stem)


if __name__ == "__main__":
    main()
