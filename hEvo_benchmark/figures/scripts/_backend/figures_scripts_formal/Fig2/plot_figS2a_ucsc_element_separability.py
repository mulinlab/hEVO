#!/usr/bin/env python3
"""Render Figure S2a: PCA/t-SNE Calinski-Harabasz comparison.

The renderer intentionally refuses to infer values from the legacy PDF. It
requires a registered long-format table with ``model_name``, ``method`` and
``ch_index``. Repeated seeds are summarized by median and interquartile range.
"""

from __future__ import annotations

# [Legacy/fallback] Data and plotting implementation for the former UCSC element separability panel, retained as a compatibility module.
# [Input] Legacy UCSC element source data. [Output] Standalone panel Drawing, figure files, and QA.
# [Note] Current Fig. 2a/b use S11–S16; their input tables are not defined here.

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from PIL import Image
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.scripts.common.hevo_model_style import (  # noqa: E402
    BLACK,
    MODEL_FAMILY_STYLE,
    canonical_model_name,
    model_family,
)
from figures.scripts.common.font_config import register_reportlab_arial, resolve_arial_paths  # noqa: E402

WIDTH_MM = 180.0
HEIGHT_MM = 136.0
PNG_DPI = 600
WHITE = "#FFFFFF"
MID_GREY = "#8995A5"
CONNECTOR_GREY = "#C9CED5"
HEVO_WASH = "#FBF2EA"

FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_PATHS = resolve_arial_paths()

ELEMENT_LABELS = (
    "UCSC CDS",
    "Promoter low GC",
    "Promoter high GC",
    "SINE",
    "Intergenic",
)

REQUIRED = ("model_name", "method", "ch_index")
METHOD_MAP = {
    "pca": "PCA",
    "tsne": "t-SNE",
    "t-sne": "t-SNE",
    "t_sne": "t-SNE",
}


def configure_fonts() -> None:
    register_reportlab_arial()
    for name, path in FONT_PATHS.items():
        if not path.exists():
            raise RuntimeError(f"Required Arial font is missing: {path}")
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))


def color(value: str):
    return HexColor(value)


def add_text(
    drawing: Drawing,
    x_mm: float,
    y_mm: float,
    text: str,
    size: float,
    *,
    anchor: str = "start",
    bold: bool = False,
    fill: str = BLACK,
) -> None:
    drawing.add(
        String(
            x_mm * mm,
            y_mm * mm,
            text,
            fontName=FONT_BOLD if bold else FONT_REGULAR,
            fontSize=size,
            fillColor=color(fill),
            textAnchor=anchor,
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Registered Fig. S2a metric table is missing: {path}. "
            "Recover the upstream CSV or rerun the analysis; do not trace the legacy PDF."
        )
    table = pd.read_csv(path)
    missing = [column for column in REQUIRED if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = table.copy()
    result["model_name_raw"] = result["model_name"].astype(str).str.strip()
    result["model_name"] = result["model_name_raw"].map(canonical_model_name)
    method_key = result["method"].astype(str).str.strip().str.lower()
    result["method"] = method_key.map(METHOD_MAP)
    if result["method"].isna().any():
        bad = sorted(table.loc[result["method"].isna(), "method"].astype(str).unique())
        raise ValueError(f"Unrecognized reduction methods: {bad}")
    result["ch_index"] = pd.to_numeric(result["ch_index"], errors="coerce")
    if result["ch_index"].isna().any():
        raise ValueError("Non-numeric or missing ch_index values are not allowed")
    if (result["ch_index"] < 0).any():
        raise ValueError("Calinski-Harabasz index must be non-negative")

    counts = result.groupby(["model_name", "method"]).size().unstack(fill_value=0)
    if not {"PCA", "t-SNE"}.issubset(counts.columns):
        raise ValueError("The source table must contain both PCA and t-SNE rows")
    incomplete = counts.index[(counts["PCA"] == 0) | (counts["t-SNE"] == 0)].tolist()
    if incomplete:
        raise ValueError(f"Models missing PCA or t-SNE results: {incomplete}")
    result["family"] = result["model_name"].map(model_family)
    return result


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, family, method), group in table.groupby(
        ["model_name", "family", "method"], sort=False
    ):
        values = group["ch_index"].to_numpy(float)
        rows.append(
            {
                "model_name": model,
                "family": family,
                "method": method,
                "ch_index_median": float(np.median(values)),
                "ch_index_q1": float(np.quantile(values, 0.25)),
                "ch_index_q3": float(np.quantile(values, 0.75)),
                "n_runs": int(len(group)),
            }
        )
    summary = pd.DataFrame(rows)
    tsne = summary[summary["method"] == "t-SNE"][["model_name", "ch_index_median"]].copy()
    tsne["tsne_rank"] = tsne["ch_index_median"].rank(method="min", ascending=False)
    return summary.merge(tsne[["model_name", "tsne_rank"]], on="model_name", how="left")


def nice_axis_max(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
    for nice in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if fraction <= nice:
            return nice * (10**exponent)
    return 10 ** (exponent + 1)


def draw_open_point(drawing: Drawing, x: float, y: float, stroke: str) -> None:
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            1.05 * mm,
            fillColor=color(WHITE),
            strokeColor=color(stroke),
            strokeWidth=0.75,
        )
    )


def draw_filled_point(drawing: Drawing, x: float, y: float, fill: str) -> None:
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            1.05 * mm,
            fillColor=color(fill),
            strokeColor=color(fill),
            strokeWidth=0.65,
        )
    )


def draw_legend(drawing: Drawing) -> None:
    y = 126.7
    x = 61.5
    draw_open_point(drawing, x, y, BLACK)
    add_text(drawing, x + 2.2, y - 0.7, "PCA", 6.0)
    x += 13.0
    draw_filled_point(drawing, x, y, BLACK)
    add_text(drawing, x + 2.2, y - 0.7, "t-SNE", 6.0)
    x += 18.0
    for family in ("hEVO", "Evo 2", "NT", "GPN"):
        family_color = MODEL_FAMILY_STYLE[family]["color"]
        draw_filled_point(drawing, x, y, family_color)
        add_text(drawing, x + 2.2, y - 0.7, family, 6.0)
        x += {"hEVO": 15.0, "Evo 2": 15.0, "NT": 11.0, "GPN": 13.0}[family]


def build_drawing(table: pd.DataFrame) -> tuple[Drawing, pd.DataFrame, list[str], float]:
    configure_fonts()
    summary = summarize(table)
    order = (
        summary[["model_name", "tsne_rank"]]
        .drop_duplicates()
        .sort_values(["tsne_rank", "model_name"], kind="stable")["model_name"]
        .tolist()
    )
    if len(order) > 38:
        raise ValueError(
            f"The 180 x 136 mm panel supports at most 38 readable model rows; got {len(order)}"
        )

    drawing = Drawing(WIDTH_MM * mm, HEIGHT_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            WIDTH_MM * mm,
            HEIGHT_MM * mm,
            fillColor=color(WHITE),
            strokeColor=None,
        )
    )

    label_x = 7.0
    rank_x = 56.0
    plot_x0 = 61.5
    plot_x1 = 176.0
    row_top = 119.2
    row_bottom = 13.5
    row_step = (row_top - row_bottom) / max(len(order) - 1, 1)

    observed_max = float(summary["ch_index_q3"].max()) * 1.05
    axis_max = nice_axis_max(observed_max)
    value_to_x = lambda value: plot_x0 + float(value) / axis_max * (plot_x1 - plot_x0)

    add_text(drawing, 2.0, 130.0, "a", 14.0, bold=True)
    add_text(
        drawing,
        176.0,
        131.0,
        "UCSC genomic elements | five 512-bp classes",
        6.2,
        anchor="end",
    )
    draw_legend(drawing)
    add_text(drawing, label_x, 122.5, "Model", 6.2, bold=True)
    add_text(drawing, rank_x, 122.5, "Rank", 6.2, anchor="end", bold=True)

    for index, model in enumerate(order):
        y = row_top - index * row_step
        if model_family(model) == "hEVO":
            drawing.add(
                Rect(
                    4.5 * mm,
                    (y - row_step * 0.43) * mm,
                    171.5 * mm,
                    row_step * 0.86 * mm,
                    fillColor=color(HEVO_WASH),
                    strokeColor=None,
                )
            )
        add_text(drawing, label_x, y - 0.75, model, 6.25)
        add_text(drawing, rank_x, y - 0.75, str(index + 1), 6.0, anchor="end", fill=MID_GREY)

        model_rows = summary[summary["model_name"] == model].set_index("method")
        pca = float(model_rows.loc["PCA", "ch_index_median"])
        tsne = float(model_rows.loc["t-SNE", "ch_index_median"])
        x_pca = value_to_x(pca)
        x_tsne = value_to_x(tsne)
        family_color = MODEL_FAMILY_STYLE[model_family(model)]["color"]
        drawing.add(
            Line(
                x_pca * mm,
                y * mm,
                x_tsne * mm,
                y * mm,
                strokeColor=color(CONNECTOR_GREY),
                strokeWidth=0.65,
            )
        )
        for method, x, filled in (("PCA", x_pca, False), ("t-SNE", x_tsne, True)):
            row = model_rows.loc[method]
            if int(row["n_runs"]) > 1:
                q1 = value_to_x(float(row["ch_index_q1"]))
                q3 = value_to_x(float(row["ch_index_q3"]))
                drawing.add(
                    Line(
                        q1 * mm,
                        y * mm,
                        q3 * mm,
                        y * mm,
                        strokeColor=color(family_color),
                        strokeWidth=1.15,
                    )
                )
            if filled:
                draw_filled_point(drawing, x, y, family_color)
            else:
                draw_open_point(drawing, x, y, family_color)

    axis_y = 9.7
    drawing.add(
        Line(
            plot_x0 * mm,
            axis_y * mm,
            plot_x1 * mm,
            axis_y * mm,
            strokeColor=color(BLACK),
            strokeWidth=0.75,
        )
    )
    for value in np.linspace(0, axis_max, 6):
        x = value_to_x(value)
        drawing.add(
            Line(
                x * mm,
                axis_y * mm,
                x * mm,
                (axis_y - 1.4) * mm,
                strokeColor=color(BLACK),
                strokeWidth=0.65,
            )
        )
        label = f"{int(round(value)):,}"
        add_text(drawing, x, axis_y - 4.1, label, 6.0, anchor="middle")
    add_text(
        drawing,
        (plot_x0 + plot_x1) / 2,
        1.7,
        "Calinski-Harabasz index",
        7.2,
        anchor="middle",
    )
    return drawing, summary, order, axis_max


def inspect_pdf(pdf_path: Path) -> dict[str, object]:
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    fonts = []
    resources = page.get("/Resources", {})
    font_dict = resources.get("/Font", {}) if resources else {}
    for ref in font_dict.values():
        fonts.append(str(ref.get_object().get("/BaseFont", "")))
    return {
        "pages": len(reader.pages),
        "width_mm": round(float(page.mediabox.width) / 72 * 25.4, 3),
        "height_mm": round(float(page.mediabox.height) / 72 * 25.4, 3),
        "fonts": sorted(set(fonts)),
        "arial_present": any("Arial" in item for item in fonts),
    }


def render_png_from_pdf(pdf: Path, png: Path) -> None:
    dependencies = Path(sys.executable).resolve().parents[1]
    candidates = [dependencies / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"]
    discovered = shutil.which("pdftoppm")
    if discovered:
        candidates.append(Path(discovered))
    executable = next((item for item in candidates if item.exists()), None)
    if executable is None:
        raise RuntimeError("pdftoppm is required to render the 600-dpi PNG")
    subprocess.run(
        [
            str(executable),
            "-f", "1",
            "-l", "1",
            "-singlefile",
            "-png",
            "-r", str(PNG_DPI),
            str(pdf),
            str(png.with_suffix("")),
        ],
        check=True,
    )


def render(table: pd.DataFrame, output_dir: Path, stem: str) -> dict[str, object]:
    drawing, summary, order, axis_max = build_drawing(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    render_png_from_pdf(pdf, png)

    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    normalized = source_dir / "FigS2a_UCSC_element_CH_plot_source.csv"
    summary.sort_values(["tsne_rank", "model_name", "method"]).to_csv(normalized, index=False)

    pdf_info = inspect_pdf(pdf)
    with Image.open(png) as image:
        png_size = list(image.size)
    provenance_values = set(table.get("provenance_status", pd.Series(dtype=str)).dropna().astype(str))
    if "LEGACY_VECTOR_RECOVERED_QC_DRAFT" in provenance_values:
        status = "LEGACY_VECTOR_RECOVERED_QC_DRAFT"
    elif table.groupby(["model_name", "method"]).size().max() > 1:
        status = "SOURCE_BACKED_REPEATED_SEEDS"
    else:
        status = "SINGLE_RUN_SOURCE_BACKED"
    qa = {
        "status": status,
        "panel": "Figure S2a",
        "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
        "models": len(order),
        "methods": sorted(table["method"].unique().tolist()),
        "metric": "Calinski-Harabasz index",
        "axis": {"minimum": 0.0, "maximum": axis_max},
        "class_labels": list(ELEMENT_LABELS),
        "font": "Arial",
        "minimum_font_pt": 6.0,
        "gridlines": False,
        "text_color": BLACK,
        "pdf_inspection": pdf_info,
        "png_pixels": png_size,
        "exports": {"pdf": str(pdf), "svg": str(svg), "png": str(png)},
        "hashes": {
            "input": sha256(Path(table.attrs["source_path"])),
            "script": sha256(HERE),
            "normalized_source": sha256(normalized),
        },
    }
    return qa


def parse_args() -> argparse.Namespace:
    default_out = PROJECT_ROOT / "figures" / "formal" / "Fig2" / "figS2a_ucsc_elements_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument(
        "--stem",
        default="FigS2a_UCSC_element_PCA_tSNE_CH_SOURCE_BACKED",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = read_source(args.input_csv)
    table.attrs["source_path"] = str(args.input_csv.resolve())
    qa = render(table, args.output_dir, args.stem)


if __name__ == "__main__":
    main()
