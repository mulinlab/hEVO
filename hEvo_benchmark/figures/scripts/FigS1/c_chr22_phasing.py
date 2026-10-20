#!/usr/bin/env python3
"""Render Fig. S1c: UKB WGS chromosome-22 phasing switch error by MAC.

The renderer accepts either:
1) a long/summary table with ``mac_bin``, ``mean_ser_pct`` and ``sd_ser_pct``;
2) the legacy window-by-MAC matrix, in which case mean, sample SD and the
   number of non-missing 1-Mb windows are recomputed.

The V1 snapshot does not contain the legacy matrix. The formal S3 input is
therefore a clearly labelled, vector-digitized reconstruction of the legacy
PDF; structural source QC passes, while raw-VCF reconciliation remains an
explicit provenance limitation.
"""

from __future__ import annotations

import argparse
import csv
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
from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Group,
    Line,
    PolyLine,
    Rect,
    String,
)
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).resolve()
SCRIPTS_ROOT = HERE.parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from common.font_config import register_reportlab_arial, resolve_arial_paths  # noqa: E402
from common.public_paths import publicize_paths  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "figures" / "FigS1"
SOURCE_DIR = PROJECT_ROOT / "tables"
TRANSCRIBED_PATH = SOURCE_DIR / "S3_chromosome_22_phasing_switch_error_by_minor_allele_count.csv"
STEM = "c_chr22_phasing"

WIDTH_MM = 84.0
HEIGHT_MM = 96.0
PNG_DPI = 600
PANEL_LABEL_SIZE = 12.5

BLACK = "#000000"
WHITE = "#FFFFFF"
NAVY = "#0B2E83"
PALE_NAVY = "#DCE6F2"

FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_PATHS = resolve_arial_paths()

MAC_ORDER = [
    "2-5",
    "6-10",
    "11-50",
    "51-100",
    "101-1,000",
    "1,001-2,000",
    "2,001-5,000",
    "5,001-10,000",
    ">10,000",
]


def register_fonts() -> None:
    register_reportlab_arial()


def _add_text(
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


def _normalise_bin_label(label: object) -> str:
    text = str(label).strip()
    replacements = {
        "–": "-",
        "—": "-",
        "1001-2000": "1,001-2,000",
        "2001-5000": "2,001-5,000",
        "5001-10000": "5,001-10,000",
        "101-1000": "101-1,000",
        "1k-2k": "1,001-2,000",
        "2k-5k": "2,001-5,000",
        "5k-10k": "5,001-10,000",
        ">10k": ">10,000",
    }
    return replacements.get(text, text)


def _load_summary(path: Path) -> tuple[pd.DataFrame, str]:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=sep)
    lower_names = {str(col).strip().lower(): col for col in frame.columns}

    required = {"mac_bin", "mean_ser_pct", "sd_ser_pct"}
    if required.issubset(lower_names):
        renamed = frame.rename(
            columns={lower_names[key]: key for key in required}
        ).copy()
        if "n_windows" in lower_names:
            renamed = renamed.rename(columns={lower_names["n_windows"]: "n_windows"})
        else:
            renamed["n_windows"] = pd.NA
        renamed["mac_bin"] = renamed["mac_bin"].map(_normalise_bin_label)
        if (
            set(renamed["mac_bin"]) != set(MAC_ORDER)
            or renamed["mac_bin"].duplicated().any()
        ):
            raise ValueError("FigS1c S3 requires nine unique MAC bins")
        for column in ("mean_ser_pct", "sd_ser_pct"):
            values = pd.to_numeric(renamed[column], errors="coerce")
            if not values.notna().all() or (values < 0).any():
                raise ValueError(f"FigS1c S3 has invalid {column}")
        if "legacy_lower_pct" in lower_names and "legacy_upper_pct" in lower_names:
            lower = pd.to_numeric(
                frame[lower_names["legacy_lower_pct"]], errors="coerce"
            )
            upper = pd.to_numeric(
                frame[lower_names["legacy_upper_pct"]], errors="coerce"
            )
            mean = pd.to_numeric(frame[lower_names["mean_ser_pct"]], errors="coerce")
            if (lower.notna() & ((lower < 0) | (lower > mean))).any():
                raise ValueError("FigS1c S3 lower bounds are invalid")
            if (upper.notna() & (upper < mean)).any():
                raise ValueError("FigS1c S3 upper bounds are invalid")
        status = (
            "SOURCE_VERIFIED_FORMAL_S3"
            if path.name.startswith("S3_")
            else "SOURCE_VERIFIED_LOCAL"
        )
        return renamed[["mac_bin", "mean_ser_pct", "sd_ser_pct", "n_windows"]], status

    numeric = frame.copy()
    candidate_columns: list[str] = []
    for column in numeric.columns:
        label = _normalise_bin_label(column)
        if label in MAC_ORDER:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
            candidate_columns.append(column)
    if not candidate_columns:
        raise ValueError(
            "Input is neither a summary table nor a window-by-MAC matrix with recognizable MAC-bin columns"
        )

    rows: list[dict[str, object]] = []
    observed_max = max(
        float(numeric[column].max(skipna=True)) for column in candidate_columns
    )
    multiplier = 100.0 if observed_max <= 1.0 else 1.0
    for column in candidate_columns:
        values = numeric[column].dropna().astype(float) * multiplier
        rows.append(
            {
                "mac_bin": _normalise_bin_label(column),
                "mean_ser_pct": float(values.mean()),
                "sd_ser_pct": float(values.std(ddof=1)),
                "n_windows": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows), "SOURCE_VERIFIED_LOCAL"


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"mac_bin", "mean_ser_pct", "sd_ser_pct", "n_windows"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    result["mac_bin"] = result["mac_bin"].map(_normalise_bin_label)
    if set(result["mac_bin"]) != set(MAC_ORDER):
        raise ValueError(
            f"MAC bins differ from the expected nine-bin contract: {result['mac_bin'].tolist()}"
        )
    if result["mac_bin"].duplicated().any():
        raise ValueError("MAC bins must be unique")
    result["mean_ser_pct"] = pd.to_numeric(result["mean_ser_pct"], errors="raise")
    result["sd_ser_pct"] = pd.to_numeric(result["sd_ser_pct"], errors="raise")
    if (result["mean_ser_pct"] < 0).any() or (result["mean_ser_pct"] > 100).any():
        raise ValueError("SER means must be percentages in [0, 100]")
    if (result["sd_ser_pct"] < 0).any():
        raise ValueError("SER SD values must be non-negative")
    result["mac_bin"] = pd.Categorical(
        result["mac_bin"], categories=MAC_ORDER, ordered=True
    )
    result = result.sort_values("mac_bin").reset_index(drop=True)
    result["mac_bin"] = result["mac_bin"].astype("object")
    result["lower_pct"] = (result["mean_ser_pct"] - result["sd_ser_pct"]).clip(lower=0)
    result["upper_pct"] = result["mean_ser_pct"] + result["sd_ser_pct"]
    return result


def render(frame: pd.DataFrame) -> Drawing:
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

    # Horizontal MAC-bin layout: higher-MAC bins should read as a descending
    # switch-error trend from left to right.
    _add_text(drawing, 2.8, 85.5, "c", PANEL_LABEL_SIZE, bold=True)

    left, right = 18.0, 81.0
    bottom, top = 23.0, 79.0
    # Leave a small visual floor below the lowest positive estimate so the
    # last marker cannot collide with the horizontal axis.
    y_min, y_max = -0.6, 13.5
    x_padding = 5.0
    axis_y = bottom

    def sx(index: int) -> float:
        # Keep the first/last marker and vertical error-bar cap clear of the
        # y-axis and right canvas edge.
        if len(frame) == 1:
            return (left + right) / 2
        return (
            left + x_padding + index * (right - left - 2 * x_padding) / (len(frame) - 1)
        )

    def sy(value: float) -> float:
        return bottom + (float(value) - y_min) / (y_max - y_min) * (top - bottom)

    # Legend is a data-definition key rather than a panel subtitle.
    legend_y = 87.3
    drawing.add(
        Line(
            17.0 * mm,
            legend_y * mm,
            24.0 * mm,
            legend_y * mm,
            strokeColor=HexColor(PALE_NAVY),
            strokeWidth=3.2,
        )
    )
    drawing.add(
        Line(
            17.0 * mm,
            legend_y * mm,
            24.0 * mm,
            legend_y * mm,
            strokeColor=HexColor(NAVY),
            strokeWidth=0.65,
        )
    )
    drawing.add(
        Circle(
            20.5 * mm,
            legend_y * mm,
            1.05 * mm,
            fillColor=HexColor(NAVY),
            strokeColor=HexColor(WHITE),
            strokeWidth=0.35,
        )
    )
    _add_text(drawing, 26.0, legend_y - 0.7, "mean +/- 1 SD across 1-Mb windows", 5.8)

    # Axes.  No internal gridlines by design.
    drawing.add(
        Line(
            left * mm,
            axis_y * mm,
            right * mm,
            axis_y * mm,
            strokeColor=HexColor(BLACK),
            strokeWidth=0.65,
        )
    )
    drawing.add(
        Line(
            left * mm,
            axis_y * mm,
            left * mm,
            top * mm,
            strokeColor=HexColor(BLACK),
            strokeWidth=0.65,
        )
    )

    for tick in (0, 2, 4, 6, 8, 10, 12):
        y = sy(tick)
        drawing.add(
            Line(
                (left - 1.3) * mm,
                y * mm,
                left * mm,
                y * mm,
                strokeColor=HexColor(BLACK),
                strokeWidth=0.55,
            )
        )
        _add_text(drawing, left - 2.0, y - 0.75, str(tick), 6.0, anchor="end")
    _add_text(
        drawing,
        7.0,
        (axis_y + top) / 2 - 1.5,
        "Switch error rate (%)",
        7.0,
        anchor="middle",
        angle=90,
    )
    _add_text(
        drawing, (left + right) / 2, 6.0, "Minor allele count", 7.0, anchor="middle"
    )

    mean_points: list[float] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        x = sx(index)
        lower_y = sy(max(y_min, row.lower_pct))
        upper_y = sy(min(row.upper_pct, y_max))
        mean_y = sy(row.mean_ser_pct)
        mean_points.extend([x * mm, mean_y * mm])

        _add_text(
            drawing, x, axis_y - 2.4, str(row.mac_bin), 5.3, anchor="end", angle=45
        )
        drawing.add(
            Line(
                x * mm,
                lower_y * mm,
                x * mm,
                upper_y * mm,
                strokeColor=HexColor(PALE_NAVY),
                strokeWidth=3.2,
            )
        )
        if row.lower_pct > 0:
            drawing.add(
                Line(
                    (x - 1.35) * mm,
                    lower_y * mm,
                    (x + 1.35) * mm,
                    lower_y * mm,
                    strokeColor=HexColor(NAVY),
                    strokeWidth=0.65,
                )
            )
        drawing.add(
            Line(
                (x - 1.35) * mm,
                upper_y * mm,
                (x + 1.35) * mm,
                upper_y * mm,
                strokeColor=HexColor(NAVY),
                strokeWidth=0.65,
            )
        )

    drawing.add(
        PolyLine(
            mean_points, strokeColor=HexColor(NAVY), strokeWidth=0.8, fillColor=None
        )
    )
    for index, row in enumerate(frame.itertuples(index=False)):
        # Mask the baseline/error bar directly under the marker so the final
        # low-error bins remain visibly separated from the x-axis.
        drawing.add(
            Circle(
                sx(index) * mm,
                sy(row.mean_ser_pct) * mm,
                1.75 * mm,
                fillColor=HexColor(WHITE),
                strokeColor=HexColor(WHITE),
                strokeWidth=0,
            )
        )
        drawing.add(
            Circle(
                sx(index) * mm,
                sy(row.mean_ser_pct) * mm,
                1.35 * mm,
                fillColor=HexColor(NAVY),
                strokeColor=HexColor(WHITE),
                strokeWidth=0.45,
            )
        )

    # _add_text(drawing, right, 81.8, "lower is better", 5.8, anchor="end")
    return drawing


def render_png(pdf_path: Path, base: Path) -> Path:
    resolved = shutil.which("pdftoppm")
    if resolved is None:
        raise FileNotFoundError("pdftoppm is required for the 600-dpi PNG")
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


def pdf_qa(
    pdf_path: Path, svg_path: Path, png_path: Path, status: str, source_path: Path
) -> dict[str, object]:
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    text = page.extract_text() or ""
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {}) if resources else {}
    image_count = 0
    for reference in xobjects.values():
        obj = reference.get_object()
        if obj.get("/Subtype") == "/Image":
            image_count += 1
    font_names: list[str] = []
    for reference in (resources.get("/Font", {}) if resources else {}).values():
        font = reference.get_object()
        name = str(font.get("/BaseFont", ""))
        font_names.append(name)

    from PIL import Image

    with Image.open(png_path) as image:
        png_size = list(image.size)

    svg_text = svg_path.read_text(encoding="utf-8")
    expected_px = [round(WIDTH_MM / 25.4 * PNG_DPI), round(HEIGHT_MM / 25.4 * PNG_DPI)]
    checks = {
        "single_page": len(reader.pages) == 1,
        "width_84_mm": abs(float(page.mediabox.width) * 25.4 / 72 - WIDTH_MM) < 0.05,
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
            for label in (
                "Switch error rate (%)",
                "Minor allele count",
                "2-5",
                ">10,000",
            )
        ),
        "no_pink_palette": all(
            token not in svg_text.lower()
            for token in ("#e91e63", "#ff69b4", "#f48fb1", "#fbdfeb")
        ),
        "pure_black_text_contract": "fill: rgb(0%,0%,0%)" in svg_text.lower(),
        "no_gridlines": "stroke-dasharray" not in svg_text.lower(),
    }
    return {
        "status": status,
        "submission_ready": status == "SOURCE_VERIFIED_LOCAL" and all(checks.values()),
        "data_source": str(source_path),
        "expected_legacy_raw_source": "not_packaged_legacy_PDF_digitized_source_only",
        "checks": checks,
        "all_artifact_checks_pass": all(checks.values()),
        "pdf_fonts": font_names,
        "pdf_image_xobjects": image_count,
        "png_pixels": png_size,
        "sha256": {
            "pdf": sha256(pdf_path),
            "svg": sha256(svg_path),
            "png": sha256(png_path),
        },
        "scientific_gates": [
            "Reconcile displayed means and SDs against the original window-by-MAC TSV.",
            "Report the validation truth set, number of validation offspring and informative switch intervals.",
            "Define whether MAC was calculated in the full phasing cohort or the validation subset.",
            "Confirm empty-window handling and whether the window means were weighted by informative switch intervals.",
            "State why MAC=1 is absent and report n_windows per MAC bin.",
        ],
    }


def write_rendered_source(frame: pd.DataFrame, status: str, source_path: Path) -> Path:
    return source_path


def _resolve_source(
    tables_dir: Path | None = None, input_path: Path | None = None
) -> Path:
    source = input_path or (
        (tables_dir or SOURCE_DIR) / "S3_chromosome_22_phasing_switch_error_by_minor_allele_count.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(f"FigS1c input not found: {source}")
    return source


def build_panel(
    tables_dir: Path | None = None, input_path: Path | None = None
) -> Drawing:
    source = _resolve_source(tables_dir, input_path)
    frame, _ = _load_summary(source)
    return render(validate(frame))


def export_panel(
    tables_dir: Path | None = None,
    output_dir: Path | None = None,
    input_path: Path | None = None,
) -> dict[str, object]:
    source = _resolve_source(tables_dir, input_path)
    frame, status = _load_summary(source)
    frame = validate(frame)
    register_fonts()
    out = output_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    drawing = render(frame)
    base = out / STEM
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    png_path = render_png(pdf_path, base)
    rendered_source = source
    qa = pdf_qa(pdf_path, svg_path, png_path, status, source)
    qa.update(
        {
            "panel": "c",
            "backend": "ReportLab",
            "data_status": status,
            "rendered_source_data": str(rendered_source),
            "scientific_gates": [
                "Raw window-by-MAC reconciliation remains outside the packaged S3 evidence.",
                "Validation truth-set and informative switch-interval denominators are not supplied by the legacy PDF digitization.",
                "MAC and empty-window handling are retained as provenance limitations.",
            ],
            "axis_orientation": {
                "x": "Minor allele count (MAC bin)",
                "y": "Switch error rate (%)",
                "trend_direction": "descending with increasing MAC",
            },
            "trend_contract": {
                "means_nonincreasing": bool(
                    frame["mean_ser_pct"].is_monotonic_decreasing
                ),
                "first_mean_ser_pct": float(frame["mean_ser_pct"].iloc[0]),
                "last_mean_ser_pct": float(frame["mean_ser_pct"].iloc[-1]),
            },
        }
    )
    return {
        "drawing": drawing,
        "pdf": pdf_path,
        "svg": svg_path,
        "png": png_path,
        "qa": qa,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=SOURCE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Summary table or legacy window-by-MAC matrix",
    )
    args = parser.parse_args()
    result = export_panel(args.tables_dir, args.output_dir, args.input)
    print(
        json.dumps(
            {key: str(value) for key, value in result.items() if key != "drawing"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
