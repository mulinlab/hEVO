#!/usr/bin/env python3
"""Render Fig. S1d from the final 63,809-sample chr1--22 deduplication table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

HERE = Path(__file__).resolve()
SCRIPTS_ROOT = HERE.parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from common.font_config import (
    register_reportlab_arial,
    resolve_arial_paths,
)
from common.public_paths import publicize_paths

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "figures" / "FigS1"
SOURCE_DIR = PROJECT_ROOT / "tables"
INPUT_PATH = SOURCE_DIR / "S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv"
STEM = "d_chromosome_redundancy"

WIDTH_MM = 96.0
HEIGHT_MM = 96.0
PNG_DPI = 600
PANEL_LABEL_SIZE = 12.5

BLACK = "#000000"
WHITE = "#FFFFFF"
NAVY = "#0B2E83"
MID_BLUE = "#5D8FBF"
PALE_BLUE = "#DCE6F2"

FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_PATHS = resolve_arial_paths()

CHROMS = [str(index) for index in range(1, 23)]
GRCH38_LENGTHS = {
    "1": 248956422,
    "2": 242193529,
    "3": 198295559,
    "4": 190214555,
    "5": 181538259,
    "6": 170805979,
    "7": 159345973,
    "8": 145138636,
    "9": 138394717,
    "10": 133797422,
    "11": 135086622,
    "12": 133275309,
    "13": 114364328,
    "14": 107043718,
    "15": 101991189,
    "16": 90338345,
    "17": 83257441,
    "18": 80373285,
    "19": 58617616,
    "20": 64444167,
    "21": 46709983,
    "22": 50818468,
    "X": 156040895,
}


def register_fonts() -> None:
    register_reportlab_arial()


def add_text(
    drawing: Drawing,
    x_mm: float,
    y_mm: float,
    text: str,
    size_pt: float,
    *,
    bold: bool = False,
    anchor: str = "start",
    angle: float = 0,
) -> None:
    item = String(
        0 if angle else x_mm * mm,
        0 if angle else y_mm * mm,
        text,
        fontName=FONT_BOLD if bold else FONT_REGULAR,
        fontSize=size_pt,
        fillColor=HexColor(BLACK),
        textAnchor=anchor,
    )
    if angle:
        group = Group()
        group.add(item)
        radians = math.radians(angle)
        group.transform = (
            math.cos(radians),
            math.sin(radians),
            -math.sin(radians),
            math.cos(radians),
            x_mm * mm,
            y_mm * mm,
        )
        drawing.add(group)
    else:
        drawing.add(item)


def normalise_chrom(value: object) -> str:
    text = str(value).strip().replace("chr", "").replace("CHR", "")
    return text.upper() if text.lower() == "x" else str(int(float(text)))


def load_source(path: Path) -> tuple[pd.DataFrame, str]:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=sep)
    lower = {str(column).strip().lower(): column for column in frame.columns}
    if {"chrom", "raw_windows", "n_free_windows", "unique_windows"}.issubset(lower):
        result = pd.DataFrame(
            {
                "chrom": frame[lower["chrom"]].map(normalise_chrom),
                "candidate_million": pd.to_numeric(
                    frame[lower["raw_windows"]], errors="raise"
                )
                / 1e6,
                "usable_million": pd.to_numeric(
                    frame[lower["n_free_windows"]], errors="raise"
                )
                / 1e6,
                "distinct_million": pd.to_numeric(
                    frame[lower["unique_windows"]], errors="raise"
                )
                / 1e6,
            }
        )
        if "sample_count_in_vcf" in lower:
            result["sample_count_in_vcf"] = pd.to_numeric(
                frame[lower["sample_count_in_vcf"]], errors="raise"
            )
        if "haplotype_count_in_vcf" in lower:
            result["haplotype_count_in_vcf"] = pd.to_numeric(
                frame[lower["haplotype_count_in_vcf"]], errors="raise"
            )
        return result, "SOURCE_VERIFIED_LOCAL"

    required = {"chrom", "candidate_million", "usable_million", "distinct_million"}
    if not required.issubset(lower):
        raise ValueError(
            f"Unrecognized chromosome-redundancy schema: {frame.columns.tolist()}"
        )
    result = frame.rename(columns={lower[key]: key for key in required})
    result["chrom"] = result["chrom"].map(normalise_chrom)
    return result[list(required)], "DATA_QC_DRAFT"


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if set(result["chrom"]) != set(CHROMS) or result["chrom"].duplicated().any():
        raise ValueError("Exactly one row is required for chromosomes 1-22")
    if "sample_count_in_vcf" in result:
        sample_counts = set(result["sample_count_in_vcf"].astype(int))
        if sample_counts != {63809}:
            raise ValueError(
                f"S7 must report sample_count_in_vcf=63809, got {sample_counts}"
            )
    if "haplotype_count_in_vcf" in result:
        haplotype_counts = set(result["haplotype_count_in_vcf"].astype(int))
        if haplotype_counts != {127618}:
            raise ValueError(
                "S7 must report haplotype_count_in_vcf=127618, "
                f"got {haplotype_counts}"
            )
    for column in ("candidate_million", "usable_million", "distinct_million"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if (result[column] <= 0).any():
            raise ValueError(f"{column} must be positive")
    if (result["usable_million"] > result["candidate_million"] * 1.005).any():
        raise ValueError("Usable observations cannot exceed candidate observations")
    if (result["distinct_million"] > result["usable_million"]).any():
        raise ValueError("Distinct sequence count cannot exceed usable observations")
    order = {chrom: index for index, chrom in enumerate(CHROMS)}
    result["_order"] = result["chrom"].map(order)
    result = result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    result["usable_fraction_pct"] = (
        result["usable_million"] / result["candidate_million"] * 100
    )
    result["distinct_per_usable_pct"] = (
        result["distinct_million"] / result["usable_million"] * 100
    )

    implied_samples: list[float] = []
    for row in result.itertuples(index=False):
        loci = (GRCH38_LENGTHS[row.chrom] - 1024) // 128 + 1
        implied_samples.append(row.candidate_million * 1e6 / loci / 2)
    result["implied_sample_count"] = implied_samples
    return result


def render(frame: pd.DataFrame, status: str) -> Drawing:
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
    add_text(drawing, 2.8, 85.5, "d", PANEL_LABEL_SIZE, bold=True)
    add_text(drawing, 93.0, 81.4, "N = 63,809; chr1-22", 6.0, anchor="end")

    left, right = 14.5, 93.0
    bottom, top = 17.0, 79.0
    lower_min, upper_max = 0.03, 350.0

    def sy(value: float) -> float:
        return bottom + (math.log10(value) - math.log10(lower_min)) / (
            math.log10(upper_max) - math.log10(lower_min)
        ) * (top - bottom)

    drawing.add(
        Line(
            left * mm,
            bottom * mm,
            right * mm,
            bottom * mm,
            strokeColor=HexColor(BLACK),
            strokeWidth=0.65,
        )
    )
    drawing.add(
        Line(
            left * mm,
            bottom * mm,
            left * mm,
            top * mm,
            strokeColor=HexColor(BLACK),
            strokeWidth=0.65,
        )
    )
    ticks = (0.1, 1, 10, 100)
    for tick in ticks:
        y = sy(tick)
        drawing.add(
            Line(
                (left - 1.2) * mm,
                y * mm,
                left * mm,
                y * mm,
                strokeColor=HexColor(BLACK),
                strokeWidth=0.55,
            )
        )
        add_text(drawing, left - 2.0, y - 0.75, f"{tick:g}", 6.0, anchor="end")
    add_text(
        drawing,
        5.6,
        (bottom + top) / 2 - 1.5,
        "Count (billions, log scale)",
        7.0,
        anchor="middle",
        angle=90,
    )
    group_w = (right - left) / len(frame)
    bar_w = 0.78
    offsets = (-0.92, 0.0, 0.92)
    colors = (PALE_BLUE, MID_BLUE, NAVY)
    columns = ("candidate_million", "usable_million", "distinct_million")
    for index, row in enumerate(frame.itertuples(index=False)):
        center = left + group_w * (index + 0.5)
        for offset, color, column in zip(offsets, colors, columns):
            value = float(getattr(row, column)) / 1000.0
            x = center + offset
            y_top = sy(value)
            y_base = bottom
            drawing.add(
                Rect(
                    (x - bar_w / 2) * mm,
                    y_base * mm,
                    bar_w * mm,
                    max(0.1, y_top - y_base) * mm,
                    fillColor=HexColor(color),
                    strokeColor=None,
                )
            )
        add_text(drawing, center, bottom - 4.2, row.chrom, 6.0, anchor="middle")
    add_text(drawing, (left + right) / 2, 7.6, "Chromosome", 7.0, anchor="middle")

    # Draw the legend last on a small white band so the tallest bars cannot
    # obscure the labels in the dense 22-chromosome panel.
    legend_y = 87.1
    drawing.add(
        Rect(
            15.5 * mm,
            83.1 * mm,
            79.0 * mm,
            8.6 * mm,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )
    legend = [
        (18.0, PALE_BLUE, "candidate windows"),
        (44.0, MID_BLUE, "usable windows"),
        (67.5, NAVY, "distinct sequences"),
    ]
    for x, color, label in legend:
        drawing.add(
            Rect(
                x * mm,
                (legend_y - 1.1) * mm,
                3.5 * mm,
                2.2 * mm,
                fillColor=HexColor(color),
                strokeColor=None,
            )
        )
        add_text(drawing, x + 4.5, legend_y - 0.75, label, 6.0)

    return drawing


def render_png(pdf_path: Path, base: Path) -> Path:
    resolved = shutil.which("pdftoppm")
    if resolved is None:
        raise FileNotFoundError("pdftoppm is required")
    poppler = Path(resolved)
    subprocess.run(
        [
            str(poppler),
            "-png",
            "-r",
            str(PNG_DPI),
            "-singlefile",
            str(pdf_path),
            str(base),
        ],
        check=True,
    )
    return base.with_suffix(".png")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_qa(
    pdf_path: Path,
    svg_path: Path,
    png_path: Path,
    frame: pd.DataFrame,
    status: str,
    source: Path,
) -> dict[str, object]:
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    text = page.extract_text() or ""
    resources = page.get("/Resources", {})
    font_names = [
        str(ref.get_object().get("/BaseFont", ""))
        for ref in (resources.get("/Font", {}) if resources else {}).values()
    ]
    xobjects = resources.get("/XObject", {}) if resources else {}
    image_count = sum(
        1 for ref in xobjects.values() if ref.get_object().get("/Subtype") == "/Image"
    )
    from PIL import Image

    with Image.open(png_path) as image:
        png_size = list(image.size)
    expected_px = [round(WIDTH_MM / 25.4 * PNG_DPI), round(HEIGHT_MM / 25.4 * PNG_DPI)]
    svg_text = svg_path.read_text(encoding="utf-8").lower()
    implied_median = float(frame["implied_sample_count"].median())
    checks = {
        "single_page": len(reader.pages) == 1,
        "width_96_mm": abs(float(page.mediabox.width) * 25.4 / 72 - WIDTH_MM) < 0.05,
        "height_96_mm": abs(float(page.mediabox.height) * 25.4 / 72 - HEIGHT_MM) < 0.05,
        "arial_compatible_embedded": any(
            "Arial" in name or "LiberationSans" in name for name in font_names
        ),
        "vector_pdf_no_images": image_count == 0,
        "png_600_dpi_dimensions": all(
            abs(a - b) <= 1 for a, b in zip(png_size, expected_px)
        ),
        "no_replacement_character": "\ufffd" not in text,
        "required_labels_present": all(
            label in text
            for label in ("Count (billions, log scale)", "Chromosome", "distinct sequences")
        ),
        "pure_black_text": "fill: rgb(0%,0%,0%)" in svg_text,
        "no_gridlines": "stroke-dasharray" not in svg_text,
        "no_pink_palette": all(
            token not in svg_text
            for token in ("#e91e63", "#ff69b4", "#f48fb1", "rgb(100%,0%,")
        ),
    }
    return {
        "status": status,
        "submission_ready": False,
        "panel": "d",
        "backend": "ReportLab",
        "data_status": status,
        "data_source": str(source),
        "expected_production_source": "S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv from the final 63,809-sample chr1-22 run",
        "checks": checks,
        "all_artifact_checks_pass": all(checks.values()),
        "inferred_legacy_sample_count_median": round(implied_median, 1),
        "active_sample_count": 63809,
        "active_haplotype_count": 127618,
        "chrX_included": False,
        "axis_contract": {
            "type": "continuous_log10_y_axis",
            "range_billions": [0.03, 350.0],
        },
        "distinct_usable_percent_range": [
            round(float(frame["distinct_per_usable_pct"].min()), 4),
            round(float(frame["distinct_per_usable_pct"].max()), 4),
        ],
        "sha256": {
            "pdf": sha256(pdf_path),
            "svg": sha256(svg_path),
            "png": sha256(png_path),
        },
        "scientific_gates": [
            "Active S7 source reports 63,809 persons and 127,618 haplotypes.",
            "This panel is autosome-only (chr1-chr22); chrX is intentionally excluded.",
            "Do not interpret distinct/usable as a biological retention rate without callability and denominator normalization.",
            "Reconcile chromosome-level statistics with the underlying window-level source before promotion.",
        ],
    }


def _resolve_source(
    tables_dir: Path | None = None, input_path: Path | None = None
) -> Path:
    if input_path is not None:
        return input_path
    return (tables_dir or SOURCE_DIR) / "S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv"


def build_panel(
    tables_dir: Path | None = None, input_path: Path | None = None
) -> Drawing:
    source = _resolve_source(tables_dir, input_path)
    frame, status = load_source(source)
    return render(validate(frame), status)


def export_panel(
    tables_dir: Path | None = None,
    output_dir: Path | None = None,
    input_path: Path | None = None,
) -> dict[str, str]:
    register_fonts()
    source = _resolve_source(tables_dir, input_path)
    out_dir = output_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, status = load_source(source)
    frame = validate(frame)

    drawing = render(frame, status)
    base = out_dir / STEM
    pdf_path, svg_path = base.with_suffix(".pdf"), base.with_suffix(".svg")
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    png_path = render_png(pdf_path, base)

    qa = artifact_qa(pdf_path, svg_path, png_path, frame, status, source)
    qa["rendered_source_data"] = None
    return {
        "pdf": str(pdf_path),
        "svg": str(svg_path),
        "png": str(png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=None
    )
    parser.add_argument(
        "--output-dir", "--out-dir", dest="output_dir", type=Path, default=None
    )
    args = parser.parse_args()
    print(
        json.dumps(export_panel(args.tables_dir, args.output_dir, args.input), indent=2)
    )


if __name__ == "__main__":
    main()
