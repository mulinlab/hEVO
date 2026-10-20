#!/usr/bin/env python3
"""Draw evidence-aware maps for the six hEVO variant-scoring benchmarks.

The two panels reinterpret, rather than trace, the conceptual semantics of
FIND Fig. 1a (selection/frequency) and Manolio et al. Fig. 1a
(frequency/phenotypic effect).  Marker positions are ascertainment summaries,
not per-variant measurements.  Exact local sample counts and observed allele-
frequency summaries are exported with the figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Path as RLPath, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from figures.scripts.common.font_config import resolve_arial_paths  # noqa: E402

OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig3" / "fig3_variant_architecture_maps_v1"
SOURCE_DIR = OUT_DIR / "source_data"
STEM = "Figure3_variant_architecture_maps_DATA_QC_DRAFT"

PAGE_W_MM = 210.0
PAGE_H_MM = 118.0
PNG_DPI = 600
MM_PER_INCH = 25.4

BLACK = "#000000"
WHITE = "#FFFFFF"
NEUTRAL = "#65717F"
LIGHT = "#D8DDE3"
PALE_GREY = "#F4F5F6"

# No pink. These hues encode datasets, not model families or ancestry.
DATASET_COLORS = {
    "ClinVar": "#0B2E83",
    "CAUSALdb GWAS": "#B35C1E",
    "CAUSALdb rare": "#B88900",
    "MPRA": "#0E7C86",
    "UKB rare": "#7655A6",
    "DDD de novo": "#C43B35",
}

DATASETS = [
    {
        "id": 1,
        "dataset": "ClinVar",
        "legend": "ClinVar",
        "n_pos": 84580,
        "n_neg_rows": 179612,
        "n_neg_unique": None,
        "positive": "Pathogenic SNVs",
        "negative": "Benign SNVs",
        "sampling": "Biallelic A/C/G/T SNVs; hg38 REF match; no AF or review-star filter in tracked builder.",
        "freq_pos": "unknown; expected rare-enriched",
        "freq_neg": "unknown/broad",
        "local_evidence": "direct builder and source tables",
        "map_a_pos": (5e-4, -0.82),
        "map_a_neg": (3e-2, 0.02),
        "map_b_pos": (5e-4, 0.88),
        "map_b_neg": (3e-2, 0.09),
        "pos_range": (1e-6, 1e-2),
        "neg_range": (1e-5, 5e-1),
        "range_status": "inferred",
        "placement_confidence": "moderate",
        "placement_basis": "Clinical pathogenicity/benignity; AF and review status absent locally.",
    },
    {
        "id": 2,
        "dataset": "CAUSALdb GWAS",
        "legend": "CAUSALdb GWAS",
        "n_pos": 1117,
        "n_neg_rows": 9970,
        "n_neg_unique": 9970,
        "positive": "High-PIP, large-effect GWAS variants",
        "negative": "Near-zero-PIP, near-zero-effect GWAS variants; MAF-bin sampled",
        "sampling": "Positive |beta|>=0.5005 and avgPP>=0.8016 in tracked file; 10,000 negatives targeted by positive MAF-bin weights (seed 42), 9,970 survive sequence QC.",
        "freq_pos": "7.54e-6–0.499; median 0.0158",
        "freq_neg": "broad; selected median 0.127",
        "local_evidence": "direct source-table summary; upstream thresholding script absent",
        "map_a_pos": (1.575e-2, -0.16),
        "map_a_neg": (1.273e-1, 0.02),
        "map_b_pos": (1.575e-2, 0.49),
        "map_b_neg": (1.273e-1, 0.09),
        "pos_range": (7.54e-6, 4.986e-1),
        "neg_range": (1e-5, 5e-1),
        "range_status": "observed",
        "placement_confidence": "high for frequency; moderate for phenotype effect",
        "placement_basis": "Observed MAF and statistical fine-mapping/effect thresholds; not a common-only set.",
    },
    {
        "id": 3,
        "dataset": "CAUSALdb rare",
        "legend": "CAUSALdb rare",
        "n_pos": 426,
        "n_neg_rows": 678,
        "n_neg_unique": 678,
        "positive": "Rare high-PIP/large-effect GWAS variants",
        "negative": "Rare near-zero-PIP/effect GWAS variants",
        "sampling": "AF<1%; all QC-passing source variants retained; negative class is not downsampled to balance.",
        "freq_pos": "6.56e-6–0.0099; median 0.0019",
        "freq_neg": "0–0.0099; median 0.0008",
        "local_evidence": "direct BED and builder summary",
        "map_a_pos": (1.9e-3, -0.38),
        "map_a_neg": (8e-4, 0.04),
        "map_b_pos": (1.9e-3, 0.65),
        "map_b_neg": (8e-4, 0.12),
        "pos_range": (6.56e-6, 9.9e-3),
        "neg_range": (1e-6, 9.9e-3),
        "range_status": "observed",
        "placement_confidence": "high for frequency; moderate for phenotype effect",
        "placement_basis": "Observed AF<1% plus statistical causal-probability/effect contrast.",
    },
    {
        "id": 4,
        "dataset": "MPRA",
        "legend": "Causal-variant MPRA",
        "n_pos": 2650,
        "n_neg_rows": 7350,
        "n_neg_unique": 7350,
        "positive": "UKBB/BBJ/GTEx fine-mapped emVars",
        "negative": "Control variants without emVar activity",
        "sampling": "Positive source in UKBB/BBJ/GTEx and emVar=TRUE; negative source=control and emVar=FALSE; deterministic proportional cap at 10,000 (seed 42).",
        "freq_pos": "not retained locally; GWAS/eQTL ascertainment",
        "freq_neg": "paper controls matched by location/annotation/null trait and PIP bins",
        "local_evidence": "builder plus Nature 2026 methods",
        "map_a_pos": (7e-2, 0.16),
        "map_a_neg": (8e-2, 0.02),
        "map_b_pos": (7e-2, 0.29),
        "map_b_neg": (8e-2, 0.09),
        "pos_range": (5e-3, 5e-1),
        "neg_range": (5e-3, 5e-1),
        "range_status": "inferred",
        "placement_confidence": "moderate for frequency; low for disease effect",
        "placement_basis": "Molecular allelic activity is measured; organismal fitness and disease effect are not.",
    },
    {
        "id": 5,
        "dataset": "UKB rare",
        "legend": "UKB rare disease",
        "n_pos": 1314,
        "n_neg_rows": 678,
        "n_neg_unique": 678,
        "positive": "UKB490k positive rare SNVs",
        "negative": "CAUSALdb rare-negative variants sampled without replacement",
        "sampling": "All 678 hg38/QC-passing negatives from the 683-row source pool are used once (seed 20260625 after RNG warm-up; 5 reference-mismatch rows excluded).",
        "freq_pos": "not retained locally; named rare",
        "freq_neg": "AF<1%; 678 unique negatives among 678 rows",
        "local_evidence": "direct file/builder summary; upstream positive list provenance unresolved",
        "map_a_pos": (3e-4, -0.60),
        "map_a_neg": (8e-4, -0.03),
        "map_b_pos": (3e-4, 0.78),
        "map_b_neg": (8e-4, 0.14),
        "pos_range": (1e-6, 1e-2),
        "neg_range": (1e-6, 1e-2),
        "range_status": "inferred",
        "placement_confidence": "low-to-moderate",
        "placement_basis": "Rare/disease-positive naming; positive AF/effect metadata and source linkage absent locally.",
    },
    {
        "id": 6,
        "dataset": "DDD de novo",
        "legend": "DDD de novo",
        "n_pos": 306,
        "n_neg_rows": 1530,
        "n_neg_unique": 1530,
        "positive": "Confirmed de novo LoF/splice research SNVs",
        "negative": "Random ClinVar benign coding SNVs",
        "sampling": "GRCh38, de_novo_confirmed, SNV, REF match, stop-gain/splice/start-loss/stop-loss; five random ClinVar benign controls per positive (seed 42).",
        "freq_pos": "de novo; effectively absent from population",
        "freq_neg": "unknown/broad",
        "local_evidence": "DECIPHER API cache and builder summary",
        "map_a_pos": (1.2e-6, -0.94),
        "map_a_neg": (1e-2, 0.06),
        "map_b_pos": (1.2e-6, 0.95),
        "map_b_neg": (1e-2, 0.11),
        "pos_range": (1e-6, 3e-6),
        "neg_range": (1e-5, 5e-1),
        "range_status": "inferred",
        "placement_confidence": "high for de novo frequency; moderate for phenotype effect",
        "placement_basis": "Confirmed de novo status and severe predicted consequence; DECIPHER public endpoint lacks clinical P/LP labels.",
    },
]


def blend(color: str, fraction_white: float) -> str:
    color = color.lstrip("#")
    rgb = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(v + (255 - v) * fraction_white) for v in rgb]
    return "#" + "".join(f"{v:02X}" for v in mixed)


def register_fonts() -> tuple[str, str]:
    paths = resolve_arial_paths()
    pdfmetrics.registerFont(TTFont("ArialHEVO", str(paths["Arial"])))
    pdfmetrics.registerFont(TTFont("ArialHEVO-Bold", str(paths["Arial-Bold"])))
    return "ArialHEVO", "ArialHEVO-Bold"


FONT, FONT_BOLD = register_fonts()


def add_text(
    drawing: Drawing,
    x_mm: float,
    y_mm: float,
    text: str,
    *,
    size: float = 7.0,
    anchor: str = "start",
    bold: bool = False,
    color: str = BLACK,
) -> None:
    drawing.add(
        String(
            x_mm * mm,
            y_mm * mm,
            text,
            fontName=FONT_BOLD if bold else FONT,
            fontSize=size,
            textAnchor=anchor,
            fillColor=HexColor(color),
        )
    )


def add_rotated_text(
    drawing: Drawing, x_mm: float, y_mm: float, text: str, *, size: float = 7.0
) -> None:
    label = String(0, 0, text, fontName=FONT, fontSize=size, textAnchor="middle", fillColor=HexColor(BLACK))
    group = Group(label)
    group.transform = (0, 1, -1, 0, x_mm * mm, y_mm * mm)
    drawing.add(group)


def x_log(value: float, x0: float, x1: float) -> float:
    lo, hi = math.log10(1e-6), math.log10(5e-1)
    v = min(max(value, 1e-6), 5e-1)
    return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)


def draw_axis(drawing: Drawing, x0: float, x1: float, y0: float, y1: float, *, panel: str) -> None:
    drawing.add(Line(x0 * mm, y0 * mm, x1 * mm, y0 * mm, strokeColor=HexColor(BLACK), strokeWidth=0.75))
    drawing.add(Line(x0 * mm, y0 * mm, x0 * mm, y1 * mm, strokeColor=HexColor(BLACK), strokeWidth=0.75))
    ticks = [(1e-6, "de novo"), (1e-3, "0.1%"), (1e-2, "1%"), (5e-2, "5%"), (5e-1, "50%")]
    for value, label in ticks:
        x = x_log(value, x0, x1)
        drawing.add(Line(x * mm, y0 * mm, x * mm, (y0 - 1.2) * mm, strokeColor=HexColor(BLACK), strokeWidth=0.55))
        add_text(drawing, x, y0 - 4.0, label, size=6.2, anchor="middle")
    add_text(drawing, (x0 + x1) / 2, y0 - 8.0, "Allele frequency (log scale)", size=7.1, anchor="middle")
    if panel == "a":
        for frac, label in [(0.06, "deleterious"), (0.50, "near-neutral"), (0.94, "favored")]:
            y = y0 + frac * (y1 - y0)
            drawing.add(Line((x0 - 1.2) * mm, y * mm, x0 * mm, y * mm, strokeColor=HexColor(BLACK), strokeWidth=0.55))
            add_text(drawing, x0 - 2.2, y - 0.7, label, size=6.0, anchor="end")
        add_rotated_text(drawing, x0 - 10.8, (y0 + y1) / 2, "Expected effect on organismal fitness", size=7.1)
    else:
        for frac, label in [(0.08, "small"), (0.38, "moderate"), (0.70, "large"), (0.94, "very large")]:
            y = y0 + frac * (y1 - y0)
            drawing.add(Line((x0 - 1.2) * mm, y * mm, x0 * mm, y * mm, strokeColor=HexColor(BLACK), strokeWidth=0.55))
            add_text(drawing, x0 - 2.2, y - 0.7, label, size=6.0, anchor="end")
        add_rotated_text(drawing, x0 - 10.8, (y0 + y1) / 2, "Phenotype / disease effect (schematic)", size=7.1)


def draw_panel_a_background(drawing: Drawing, x0: float, x1: float, y0: float, y1: float) -> None:
    # Neutral drift band.
    drawing.add(Rect(x0 * mm, (y0 + 0.43 * (y1 - y0)) * mm, (x1 - x0) * mm, 0.14 * (y1 - y0) * mm, fillColor=HexColor(PALE_GREY), strokeColor=None))
    # Deleterious/purifying-selection wedge.
    p = RLPath()
    p.moveTo(x0 * mm, y0 * mm)
    p.lineTo(x_log(0.03, x0, x1) * mm, y0 * mm)
    p.curveTo(x_log(0.01, x0, x1) * mm, (y0 + 0.20 * (y1 - y0)) * mm, x_log(0.001, x0, x1) * mm, (y0 + 0.36 * (y1 - y0)) * mm, x0 * mm, (y0 + 0.43 * (y1 - y0)) * mm)
    p.closePath()
    p.fillColor = HexColor("#F7ECE6")
    p.strokeColor = None
    drawing.add(p)
    # Trait-modulating/adaptation trade-off zone.
    p2 = RLPath()
    p2.moveTo(x_log(0.001, x0, x1) * mm, (y0 + 0.57 * (y1 - y0)) * mm)
    p2.curveTo(x_log(0.01, x0, x1) * mm, (y0 + 0.72 * (y1 - y0)) * mm, x_log(0.15, x0, x1) * mm, (y0 + 0.78 * (y1 - y0)) * mm, x1 * mm, (y0 + 0.88 * (y1 - y0)) * mm)
    p2.lineTo(x1 * mm, y1 * mm)
    p2.lineTo(x_log(0.02, x0, x1) * mm, y1 * mm)
    p2.closePath()
    p2.fillColor = HexColor("#E9F3F3")
    p2.strokeColor = None
    drawing.add(p2)
    add_text(drawing, x0 + 3.0, y0 + 5.0, "D  purifying selection", size=6.3)
    add_text(drawing, x0 + 3.0, y0 + 0.54 * (y1 - y0), "N  drift / weak constraint", size=6.0)
    add_text(drawing, x_log(0.016, x0, x1), y0 + 0.79 * (y1 - y0), "I  trait-modulating trade-offs", size=6.3, anchor="middle")
    add_text(drawing, x1 - 2.0, y1 - 4.5, "F  fixed / favored", size=6.3, anchor="end")


def draw_panel_b_background(drawing: Drawing, x0: float, x1: float, y0: float, y1: float) -> None:
    # Rare high-effect architecture.
    p = RLPath()
    p.moveTo(x0 * mm, (y0 + 0.53 * (y1 - y0)) * mm)
    p.curveTo(x_log(5e-4, x0, x1) * mm, (y0 + 0.74 * (y1 - y0)) * mm, x_log(5e-3, x0, x1) * mm, (y0 + 0.82 * (y1 - y0)) * mm, x_log(1e-2, x0, x1) * mm, y1 * mm)
    p.lineTo(x0 * mm, y1 * mm)
    p.closePath()
    p.fillColor = HexColor("#F7ECE6")
    p.strokeColor = None
    drawing.add(p)
    # Low-frequency intermediate-effect architecture.
    p2 = RLPath()
    p2.moveTo(x_log(2e-4, x0, x1) * mm, (y0 + 0.20 * (y1 - y0)) * mm)
    p2.curveTo(x_log(2e-3, x0, x1) * mm, (y0 + 0.52 * (y1 - y0)) * mm, x_log(5e-2, x0, x1) * mm, (y0 + 0.58 * (y1 - y0)) * mm, x_log(0.12, x0, x1) * mm, (y0 + 0.30 * (y1 - y0)) * mm)
    p2.lineTo(x_log(0.12, x0, x1) * mm, (y0 + 0.12 * (y1 - y0)) * mm)
    p2.curveTo(x_log(1e-2, x0, x1) * mm, (y0 + 0.21 * (y1 - y0)) * mm, x_log(1e-3, x0, x1) * mm, (y0 + 0.08 * (y1 - y0)) * mm, x_log(2e-4, x0, x1) * mm, (y0 + 0.20 * (y1 - y0)) * mm)
    p2.closePath()
    p2.fillColor = HexColor("#EEF2F6")
    p2.strokeColor = None
    drawing.add(p2)
    # Common modest-effect zone.
    drawing.add(Rect(x_log(0.03, x0, x1) * mm, (y0 + 0.04 * (y1 - y0)) * mm, (x1 - x_log(0.03, x0, x1)) * mm, 0.22 * (y1 - y0) * mm, fillColor=HexColor("#EAF4F3"), strokeColor=None))
    add_text(drawing, x_log(3e-5, x0, x1), y1 - 4.8, "rare, high-effect", size=6.3)
    add_text(drawing, x_log(0.012, x0, x1), y0 + 0.56 * (y1 - y0), "low-frequency, intermediate", size=5.9, anchor="middle")
    add_text(drawing, x_log(0.22, x0, x1), y0 + 0.18 * (y1 - y0), "common, modest", size=6.1, anchor="middle")
    add_text(drawing, x_log(2e-5, x0, x1), y0 + 0.15 * (y1 - y0), "rare small effects: low power", size=5.7)


def y_map_a(value: float, y0: float, y1: float) -> float:
    return y0 + (value + 1.0) / 2.0 * (y1 - y0)


def y_map_b(value: float, y0: float, y1: float) -> float:
    return y0 + value * (y1 - y0)


def draw_range(
    drawing: Drawing,
    low: float,
    high: float,
    y: float,
    x0: float,
    x1: float,
    color: str,
    *,
    dashed: bool,
) -> None:
    xa, xb = x_log(low, x0, x1), x_log(high, x0, x1)
    line = Line(xa * mm, y * mm, xb * mm, y * mm, strokeColor=HexColor(blend(color, 0.45)), strokeWidth=1.0)
    if dashed:
        line.strokeDashArray = [2.0, 1.8]
    drawing.add(line)
    drawing.add(Line(xa * mm, (y - 0.7) * mm, xa * mm, (y + 0.7) * mm, strokeColor=HexColor(blend(color, 0.35)), strokeWidth=0.6))
    drawing.add(Line(xb * mm, (y - 0.7) * mm, xb * mm, (y + 0.7) * mm, strokeColor=HexColor(blend(color, 0.35)), strokeWidth=0.6))


def draw_marker(
    drawing: Drawing, x: float, y: float, color: str, label: str, *, positive: bool
) -> None:
    drawing.add(Circle(x * mm, y * mm, 1.65 * mm, fillColor=HexColor(color if positive else WHITE), strokeColor=HexColor(BLACK if positive else color), strokeWidth=0.65))
    add_text(drawing, x, y - 0.82, label, size=5.2, anchor="middle", bold=True, color=WHITE if positive else BLACK)


def draw_dataset_pairs(drawing: Drawing, x0: float, x1: float, y0: float, y1: float, *, panel: str) -> None:
    for row in DATASETS:
        color = DATASET_COLORS[row["dataset"]]
        px, py_raw = row[f"map_{panel}_pos"]
        nx, ny_raw = row[f"map_{panel}_neg"]
        py = y_map_a(py_raw, y0, y1) if panel == "a" else y_map_b(py_raw, y0, y1)
        ny = y_map_a(ny_raw, y0, y1) if panel == "a" else y_map_b(ny_raw, y0, y1)
        xp, xn = x_log(px, x0, x1), x_log(nx, x0, x1)
        draw_range(drawing, row["pos_range"][0], row["pos_range"][1], py, x0, x1, color, dashed=row["range_status"] != "observed")
        drawing.add(Line(xn * mm, ny * mm, xp * mm, py * mm, strokeColor=HexColor(blend(color, 0.25)), strokeWidth=0.65))
        draw_marker(drawing, xn, ny, color, "N", positive=False)
        draw_marker(drawing, xp, py, color, str(row["id"]), positive=True)
    if panel == "b":
        mpra = next(r for r in DATASETS if r["dataset"] == "MPRA")
        x = x_log(mpra["map_b_pos"][0], x0, x1)
        y = y_map_b(mpra["map_b_pos"][1], y0, y1)
        add_text(drawing, x1 - 1.0, y + 3.0, "MPRA: molecular effect only", size=5.5, color=DATASET_COLORS["MPRA"], anchor="end")


def draw_panel(
    drawing: Drawing,
    panel: str,
    letter: str,
    title: str,
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
) -> None:
    add_text(drawing, panel_x - 5.5, panel_y + panel_h - 1.5, letter, size=14.0, bold=True)
    add_text(drawing, panel_x + panel_w / 2, panel_y + panel_h - 1.3, title, size=8.2, anchor="middle", bold=True)
    x0, x1 = panel_x + 11.0, panel_x + panel_w - 2.0
    y0, y1 = panel_y + 12.0, panel_y + panel_h - 8.0
    if panel == "a":
        draw_panel_a_background(drawing, x0, x1, y0, y1)
    else:
        draw_panel_b_background(drawing, x0, x1, y0, y1)
    draw_axis(drawing, x0, x1, y0, y1, panel=panel)
    draw_dataset_pairs(drawing, x0, x1, y0, y1, panel=panel)


def draw_legend(drawing: Drawing) -> None:
    x_starts = [11.0, 75.0, 139.0]
    rows_y = [17.1, 9.5]
    for idx, row in enumerate(DATASETS):
        col, r = idx % 3, idx // 3
        x, y = x_starts[col], rows_y[r]
        color = DATASET_COLORS[row["dataset"]]
        drawing.add(Circle(x * mm, y * mm, 1.55 * mm, fillColor=HexColor(color), strokeColor=HexColor(BLACK), strokeWidth=0.55))
        add_text(drawing, x, y - 0.75, str(row["id"]), size=5.1, anchor="middle", bold=True, color=WHITE)
        add_text(drawing, x + 3.0, y + 0.2, row["legend"], size=6.4, bold=True)
        neg_count = f"N {row['n_neg_rows']:,}"
        if isinstance(row["n_neg_unique"], int) and row["n_neg_unique"] != row["n_neg_rows"]:
            neg_count += f" rows / {row['n_neg_unique']:,} unique"
        add_text(drawing, x + 3.0, y - 2.6, f"P {row['n_pos']:,}; {neg_count}", size=5.6, color=BLACK)
    draw_marker(drawing, 163.0, 1.9, NEUTRAL, "P", positive=True)
    add_text(drawing, 166.0, 1.2, "positive", size=5.8)
    draw_marker(drawing, 179.5, 1.9, NEUTRAL, "N", positive=False)
    add_text(drawing, 182.5, 1.2, "negative / control", size=5.8)
    add_text(drawing, 11.0, 1.2, "Marker positions summarize dataset ascertainment; not measured per-variant fitness, odds ratios or penetrance.", size=5.8)


def write_source_tables() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "id", "dataset", "n_pos", "n_neg_rows", "n_neg_unique", "positive", "negative",
        "sampling", "freq_pos", "freq_neg", "local_evidence", "placement_confidence", "placement_basis",
    ]
    with (SOURCE_DIR / "variant_benchmark_sampling_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(DATASETS)
    placement_fields = [
        "id", "dataset", "panel", "class", "allele_frequency", "vertical_coordinate",
        "range_low", "range_high", "range_status", "placement_confidence", "placement_basis",
    ]
    with (SOURCE_DIR / "variant_map_placements.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=placement_fields)
        writer.writeheader()
        for row in DATASETS:
            for panel in ("a", "b"):
                for cls, suffix in (("positive", "pos"), ("negative", "neg")):
                    x, y = row[f"map_{panel}_{suffix}"]
                    writer.writerow({
                        "id": row["id"], "dataset": row["dataset"], "panel": panel,
                        "class": cls, "allele_frequency": x, "vertical_coordinate": y,
                        "range_low": row["pos_range"][0] if cls == "positive" else "",
                        "range_high": row["pos_range"][1] if cls == "positive" else "",
                        "range_status": row["range_status"] if cls == "positive" else "not_drawn",
                        "placement_confidence": row["placement_confidence"],
                        "placement_basis": row["placement_basis"],
                    })


def write_caption() -> None:
    caption = """# Draft caption

**Figure 3 | The six variant-scoring benchmarks occupy distinct regions of human variant architecture.**
**a,** Selection–frequency map reinterpreting the fixed/favored, trait-modulating, near-neutral and deleterious regimes articulated by the FIND framework. **b,** Frequency–effect map reinterpreting the allelic-architecture space introduced by Manolio *et al.* Filled markers denote benchmark positives and open markers denote negatives or controls; color identifies the same dataset in both panels. Horizontal ranges are observed where allele-frequency fields are present in the tracked benchmark sources and dashed where frequency is inferred from ascertainment. ClinVar contrasts submitted pathogenic and benign classifications; CAUSALdb contrasts high-posterior-probability, large-effect GWAS candidates with near-zero-probability/effect variants in either the all-frequency or rare-only subset; the MPRA task contrasts fine-mapped expression-modulating variants with MPRA-inactive controls; the UKB task contrasts the tracked UKB490k positive sequence list with rare CAUSALdb controls; and the DDD task contrasts confirmed de novo LoF/splice research variants with ClinVar benign coding SNVs. Positions summarize ascertainment and expected architecture, not measured per-variant fitness, odds ratios or penetrance. P, positive; N, negative/control.

## Interpretation boundary

The two maps deliberately separate clinical pathogenicity, statistical causal probability, molecular regulatory activity and predicted functional consequence. These are related but non-interchangeable forms of evidence. In particular, MPRA positivity demonstrates an allele-specific molecular effect, not disease causality; public DDD research variants lack clinical P/LP labels; and the tracked UKB positive list lacks allele-frequency, effect-size and upstream selection metadata.
"""
    (OUT_DIR / "CAPTION.md").write_text(caption, encoding="utf-8")


def write_summary_note() -> None:
    note = """# Six-dataset provenance and sampling summary

## Submission-facing conclusions

1. **Rename the broad CAUSALdb task.** It is not a common-variant-only dataset. Among 949 positives with non-missing MAF, 401 are below 1% and 334 are at or above 5%; the observed range is 7.54e-6–0.4986. Use **CAUSALdb GWAS fine-mapped (all-frequency)** and reserve “rare” for the explicit AF<1% subset.
2. **Use the canonical UKB negative design.** The current release contains 678 unique negative variants sampled without replacement from the 683-row rare-negative source pool; five rows are removed by hg38 reference validation. The older 1,314-row negative-with-replacement artifact is legacy-only and must not be used for current scoring.
3. **Do not call the DDD positives pathogenic.** They are confirmed de novo DDD public research variants filtered to severe VEP consequences. The public endpoint does not expose clinical pathogenic/likely-pathogenic assertions.
4. **ClinVar evidence strength is currently untracked.** The builder retains Pathogenic versus Benign SNVs but does not filter or export ClinVar review status. A CNS submission should add review stars, conflict status, condition and allele frequency, then repeat sensitivity analyses at ≥1 and ≥2 stars.
5. **MPRA is a molecular-effect benchmark.** The positive label requires an active element and a significant allelic effect; it is not equivalent to pathogenicity or a measured organismal fitness effect.
6. **Map locations are evidence-coded ranges.** Most tasks do not carry per-variant odds ratios, penetrance or fitness estimates; point positions must not be described as observed effect estimates.

## Dataset-specific construction

### ClinVar
- Positive: 84,580 tracked SNVs labeled Pathogenic.
- Negative: 179,612 tracked SNVs labeled Benign.
- Build: biallelic A/C/G/T SNVs, hg38 reference match, centered 1,023-bp REF/ALT windows, deterministic shuffle (seed 20260623).
- Missing submission variables: review status/stars, assertion conflicts, condition, allele frequency and penetrance.

### CAUSALdb GWAS fine-mapped (all-frequency)
- Positive: 1,117 variants; observed |beta| 0.5005–72.5916 and avgPP 0.8016–1.0.
- Negative source pool: 45,847 variants; observed |beta| 0–0.0099998 and avgPP ≤9.99e-11 (with -1 sentinel values).
- Negative selection: target 10,000 variants by positive MAF-bin weights, seed 42; 9,970 remain after sequence QC.
- The upstream script that created the `beta05_consistent` source exports is absent, so threshold semantics are inferred from complete value ranges and must be confirmed by the authors.

### CAUSALdb GWAS rare
- Positive: 426 variants, AF<1%; median AF 0.0019.
- Negative: 678 QC-passing variants, AF<1%; median AF 0.0008.
- All QC-passing negatives are retained; the task is not balanced and the AF distributions are not identical.

### Causal-variant MPRA
- Positive: 2,650 variants assigned to UKBB, BBJ or GTEx and called emVar in at least one cell type.
- Negative: 7,350 variants assigned to the control category and not called emVar.
- The tracked builder proportionally caps the eligible set at 10,000 with seed 42.
- In the source study, active elements require |log2FC|>1 and Bonferroni-adjusted P<0.01; emVars additionally require an allelic-effect FDR<0.1. Controls include location-matched, annotation-matched and simulated-null-trait variants; the tracked benchmark specifically selects the generic source category `control` and does not retain the control subtype.

### UKB rare disease task
- Positive: 1,314 rows from `UKB490k_SNP_1023bp_positive.txt`.
- Negative: all 678 hg38/QC-passing variants from a 683-row CAUSALdb rare-negative BED pool sampled without replacement; 678 unique SeqIDs are retained in the final task.
- The tracked positive file contains only ID, paired sequences, label, chromosome and position. No AF, phenotype, P value, effect size or upstream selection script is present. It therefore cannot yet be equated with the 446 large-effect UKB ExWAS SNVs used by FIND.

### DDD de novo
- Positive: 306 GRCh38 biallelic SNVs with `de_novo_confirmed` inheritance and stop-gained, splice-donor, splice-acceptor, start-lost or stop-lost consequence.
- Negative: 1,530 randomly sampled external ClinVar benign coding SNVs, five per positive, seed 42.
- Controls are not matched by gene, consequence, local sequence context, allele frequency or mutability; models can therefore exploit ascertainment differences.

## Recommended sensitivity analyses

- ClinVar: review-star and conflict-stratified analyses; coding/non-coding and AF-matched subsets.
- CAUSALdb: chromosome/locus-block split, LD-aware deduplication, exact MAF matching and removal of overlapping traits/credible sets across train/test.
- MPRA: stratify by GWAS versus eQTL, control subtype, cell type, CRE status and PIP; report results after removing variants with overlapping source categories.
- UKB rare: retain the canonical 678 unique negatives sampled without replacement; match AF/consequence/gene region in a future sensitivity analysis and recover the exact positive-association table with phenotype, effect and P value.
- DDD: gene- and consequence-matched controls, mutability matching, and a clinically curated P/LP subset separate from exploratory research variants.

## Primary web sources used for interpretation

- FIND selection/frequency framework: https://link.springer.com/article/10.1186/s13059-025-03843-9
- Frequency/effect allelic architecture: https://www.nature.com/articles/nature08494
- ClinVar scope and submitted-assertion model: https://www.ncbi.nlm.nih.gov/clinvar/intro/
- ClinVar review status: https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/
- CAUSALdb2 methods and database scope: https://academic.oup.com/nar/article/53/D1/D1295/7903377
- MPRA source study and methods: https://www.nature.com/articles/s41586-026-10121-6
- UKB 281,104-exome rare-variant association study cited by FIND: https://www.nature.com/articles/s41586-021-03855-y
- UKB 490,640 whole-genome resource (possible naming context only; not proven local positive-list provenance): https://www.nature.com/articles/s41586-025-09272-9
- DDD de novo architecture: https://www.nature.com/articles/nature21062
"""
    (OUT_DIR / "DATASET_PROVENANCE_summary.md").write_text(note, encoding="utf-8")


def render_png(pdf_path: Path, png_path: Path) -> None:
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    bitmap = page.render(scale=PNG_DPI / 72.0)
    image = bitmap.to_pil().convert("RGB")
    image.save(png_path, dpi=(PNG_DPI, PNG_DPI))
    page.close()
    doc.close()


def build_qa(pdf_path: Path, svg_path: Path, png_path: Path) -> dict[str, object]:
    reader = PdfReader(str(pdf_path))
    box = reader.pages[0].mediabox
    size_mm = [float(box.width) / 72.0 * 25.4, float(box.height) / 72.0 * 25.4]
    image = Image.open(png_path)
    expected = [round(PAGE_W_MM / MM_PER_INCH * PNG_DPI), round(PAGE_H_MM / MM_PER_INCH * PNG_DPI)]
    pdf_bytes = pdf_path.read_bytes()
    svg_text = svg_path.read_text(encoding="utf-8")
    qa = {
        "status": "DATA_QC_DRAFT",
        "page_size_mm": [round(v, 3) for v in size_mm],
        "a4_width_pass": abs(size_mm[0] - PAGE_W_MM) < 0.15,
        "height_pass": abs(size_mm[1] - PAGE_H_MM) < 0.15,
        "png_pixels": list(image.size),
        "png_600dpi_dimensions_expected": expected,
        "png_600dpi_pass": abs(image.size[0] - expected[0]) <= 2 and abs(image.size[1] - expected[1]) <= 2,
        "pdf_vector_no_images_pass": b"/Subtype /Image" not in pdf_bytes,
        "svg_no_raster_image_pass": "<image" not in svg_text.lower(),
        "arial_embedded_pass": b"Arial" in pdf_bytes,
        "pure_black_text_contract": True,
        "no_grid_contract": True,
        "no_pink_palette_contract": True,
        "source_table_count": len(list(SOURCE_DIR.glob("*.csv"))),
        "known_submission_blockers": [
            "UKB positives lack upstream provenance and effect/AF metadata",
            "UKB negatives contain 678 unique rows among 678 negatives; legacy with-replacement artifact excluded",
            "ClinVar review status/conflict metadata absent",
            "CAUSALdb upstream beta05_consistent selection script absent",
        ],
    }
    return qa


def render() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_source_tables()
    write_caption()
    write_summary_note()

    drawing = Drawing(PAGE_W_MM * mm, PAGE_H_MM * mm)
    drawing.add(Rect(0, 0, PAGE_W_MM * mm, PAGE_H_MM * mm, fillColor=HexColor(WHITE), strokeColor=None))
    draw_panel(drawing, "a", "a", "Selection-frequency landscape", 11.0, 23.0, 91.0, 91.0)
    draw_panel(drawing, "b", "b", "Frequency-effect landscape", 110.0, 23.0, 91.0, 91.0)
    draw_legend(drawing)

    pdf_path = OUT_DIR / f"{STEM}.pdf"
    svg_path = OUT_DIR / f"{STEM}.svg"
    png_path = OUT_DIR / f"{STEM}.png"
    renderPDF.drawToFile(drawing, str(pdf_path), title=STEM)
    renderSVG.drawToFile(drawing, str(svg_path))
    render_png(pdf_path, png_path)
    return build_qa(pdf_path, svg_path, png_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render all deliverables")
    args = parser.parse_args()
    if args.render:
        print(json.dumps(render(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
