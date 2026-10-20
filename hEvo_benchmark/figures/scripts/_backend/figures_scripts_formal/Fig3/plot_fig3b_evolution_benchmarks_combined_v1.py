#!/usr/bin/env python3
"""Add population adaptation and fixed human-lineage acceleration to Fig. 3b.

The renderer produces both an actual-slot standalone panel and an A4-integrated
Figure 3 preview. HAR/HAQER are deliberately placed in a non-numeric terminal
cap because the tracked benchmark is region-level classification rather than a
per-variant derived-allele-frequency analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

import plot_figure3_abcdefghijk_a4_integrated_v2 as base


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig3"
    / "figure3_abcdefghijk_a4_integrated_v11"
)
SOURCE_DIR = OUT_DIR / "source_data"

MAIN_STEM = "Figure3_abcdefghijk_A4_integrated_v11_DATA_QC_DRAFT"
MAIN_ONLY_STEM = "Figure3_abcdefghijk_A4_integrated_v11_EVOLUTION_ONLY_DATA_QC_DRAFT"
PANEL_STEM = "Figure3b_evolution_benchmarks_combined_v1_DATA_QC_DRAFT"
PANEL_ONLY_STEM = "Figure3b_evolution_only_v1_DATA_QC_DRAFT"

TEAL = "#2A7373"
TEAL_DARK = "#155A5A"
TEAL_WASH = "#EAF3F3"
BLUE = "#356FA8"
BLUE_WASH = "#DDE9F4"

ADAPTIVE_MEDIAN = 0.16169581511555275
ADAPTIVE_LOW = 0.0121798875702685
ADAPTIVE_HIGH = 0.3440037476577139


def add_break(drawing: Drawing, x: float, y: float) -> None:
    for offset in (0.0, 1.8):
        drawing.add(
            Line(
                (x + offset - 0.8) * mm,
                (y - 0.9) * mm,
                (x + offset + 0.8) * mm,
                (y + 0.9) * mm,
                strokeColor=HexColor(base.BLACK),
                strokeWidth=0.55,
            )
        )


def add_diamond(
    drawing: Drawing,
    x: float,
    y: float,
    *,
    radius: float,
    fill: str,
    stroke: str,
) -> None:
    drawing.add(
        Polygon(
            [
                x * mm,
                (y + radius) * mm,
                (x + radius) * mm,
                y * mm,
                x * mm,
                (y - radius) * mm,
                (x - radius) * mm,
                y * mm,
            ],
            fillColor=HexColor(fill),
            strokeColor=HexColor(stroke),
            strokeWidth=0.55,
        )
    )


def add_square(
    drawing: Drawing,
    x: float,
    y: float,
    *,
    filled: bool,
) -> None:
    size = 2.5
    drawing.add(
        Rect(
            (x - size / 2) * mm,
            (y - size / 2) * mm,
            size * mm,
            size * mm,
            fillColor=HexColor(TEAL if filled else base.WHITE),
            strokeColor=HexColor(TEAL_DARK),
            strokeWidth=0.60,
        )
    )


def draw_revised_b_panel(
    drawing: Drawing,
    placements: pd.DataFrame,
    *,
    x: float,
    width: float,
    letter_y: float,
    title_y: float,
    y0: float,
    y1: float,
) -> None:
    """Draw the Fig. 3b selection landscape in its real manuscript slot."""
    base.core.add_text(drawing, x, letter_y, "b", 13.0, bold=True)
    base.core.add_text(
        drawing,
        x + width / 2,
        title_y,
        "Fitness effects, allele frequency and lineage fixation",
        7.0,
        bold=True,
        anchor="middle",
    )

    x0 = x + 15.0
    x_end = x + width - 2.5
    fixed_w = 13.5
    break_gap = 4.8
    fixed_x0 = x_end - fixed_w
    poly_x1 = fixed_x0 - break_gap
    span = y1 - y0

    # The original six variant benchmarks retain their source-backed positions.
    base.draw_architecture_background(drawing, "a", x0, poly_x1, y0, y1)
    base.draw_architecture_data(drawing, placements, "a", x0, poly_x1, y0, y1)

    # Fixed human-lineage changes are a categorical terminal cap, not a DAF bin.
    cap_y0 = y0 + 0.705 * span
    drawing.add(
        Rect(
            fixed_x0 * mm,
            cap_y0 * mm,
            fixed_w * mm,
            (y1 - cap_y0) * mm,
            rx=1.2 * mm,
            ry=1.2 * mm,
            fillColor=HexColor(TEAL_WASH),
            strokeColor=HexColor(TEAL_DARK),
            strokeWidth=0.55,
        )
    )
    base.core.add_text(
        drawing,
        fixed_x0 + fixed_w / 2,
        y1 - 2.2,
        "Fixed lineage",
        5.25,
        bold=True,
        anchor="middle",
        color=TEAL_DARK,
    )
    base.core.add_text(
        drawing,
        fixed_x0 + fixed_w / 2,
        y1 - 4.2,
        "change",
        5.25,
        bold=True,
        anchor="middle",
        color=TEAL_DARK,
    )

    har_y = y0 + 0.875 * span
    haqer_y = y0 + 0.785 * span
    glyph_x = fixed_x0 + 2.5
    label_x = glyph_x + 2.2
    add_square(drawing, glyph_x, har_y, filled=False)
    add_square(drawing, glyph_x, haqer_y, filled=True)
    base.core.add_text(drawing, label_x, har_y - 0.65, "HAR", 5.45, bold=True, color=TEAL_DARK)
    base.core.add_text(drawing, label_x, haqer_y - 0.65, "HAQER", 5.35, bold=True, color=TEAL_DARK)

    # A numeric population-adaptation marker occupies the polymorphic wedge.
    adapt_y = y0 + 0.905 * span
    xa = base.x_log(ADAPTIVE_LOW, x0, poly_x1)
    xb = base.x_log(ADAPTIVE_HIGH, x0, poly_x1)
    xm = base.x_log(ADAPTIVE_MEDIAN, x0, poly_x1)
    base.core.add_line(drawing, xa, adapt_y, xb, adapt_y, color=BLUE, width=0.62)
    base.core.add_line(drawing, xa, adapt_y - 0.48, xa, adapt_y + 0.48, color=BLUE, width=0.45)
    base.core.add_line(drawing, xb, adapt_y - 0.48, xb, adapt_y + 0.48, color=BLUE, width=0.45)
    add_diamond(drawing, xm, adapt_y, radius=1.45, fill=BLUE, stroke=base.BLACK)
    base.core.add_text(
        drawing,
        (xa + xb) / 2,
        adapt_y - 3.0,
        "Adaptive haplotypes",
        5.45,
        anchor="middle",
        color=base.BLACK,
    )

    # Axes: a log-frequency segment plus a visibly discontinuous fixed cap.
    base.core.add_line(drawing, x0, y0, poly_x1, y0, width=0.70)
    base.core.add_line(drawing, fixed_x0, y0, x_end, y0, width=0.70)
    base.core.add_line(drawing, x0, y0, x0, y1, width=0.70)
    add_break(drawing, poly_x1 + 1.45, y0)

    for value, label in (
        (1e-6, "de novo"),
        (1e-3, "0.1%"),
        (1e-2, "1%"),
        (5e-2, "5%"),
        (5e-1, "50%"),
    ):
        px = base.x_log(value, x0, poly_x1)
        base.core.add_line(drawing, px, y0, px, y0 - 0.9, width=0.48)
        base.core.add_text(drawing, px, y0 - 3.2, label, 5.8, anchor="middle")

    fixed_center = fixed_x0 + fixed_w / 2
    base.core.add_line(drawing, fixed_center, y0, fixed_center, y0 - 0.9, width=0.48)
    base.core.add_text(drawing, fixed_center, y0 - 3.2, "fixed", 5.8, anchor="middle")
    base.core.add_text(
        drawing,
        (x0 + x_end) / 2,
        y0 - 7.0,
        "Derived allele frequency / lineage fixation status",
        6.35,
        anchor="middle",
    )

    for fraction, label in ((0.06, "deleterious"), (0.50, "limited"), (0.94, "adaptive")):
        py = y0 + fraction * span
        base.core.add_line(drawing, x0 - 0.9, py, x0, py, width=0.48)
        base.core.add_text(drawing, x0 - 1.5, py - 0.65, label, 5.8, anchor="end")
    base.add_rotated_text(
        drawing,
        x0 - 11.5,
        (y0 + y1) / 2,
        "Fitness effect (schematic)",
        size=6.35,
    )


def render_standalone(placements: pd.DataFrame) -> tuple[Path, Path, Path]:
    width_mm, height_mm = 98.0, 75.0
    drawing = Drawing(width_mm * mm, height_mm * mm)
    drawing.add(
        Rect(
            0,
            0,
            width_mm * mm,
            height_mm * mm,
            fillColor=HexColor(base.WHITE),
            strokeColor=None,
        )
    )
    draw_revised_b_panel(
        drawing,
        placements,
        x=0.0,
        width=98.0,
        letter_y=72.0,
        title_y=71.7,
        y0=10.0,
        y1=67.8,
    )
    return render_artifact(drawing, PANEL_STEM)


def draw_evolution_only_panel(drawing: Drawing) -> None:
    """Compact alternative containing only adaptation and acceleration tasks."""
    width_mm, height_mm = 98.0, 46.0
    base.core.add_text(drawing, 0.0, 43.0, "b", 13.0, bold=True)
    base.core.add_text(
        drawing,
        width_mm / 2,
        42.7,
        "Evolutionary adaptation and human-lineage acceleration",
        6.9,
        bold=True,
        anchor="middle",
    )

    y0, y1 = 10.2, 37.8
    x0, poly_x1 = 8.0, 72.0
    fixed_x0, fixed_x1 = 78.0, 96.0
    block_y0, block_y1 = 18.0, 35.5

    # Population-level adaptation block.
    drawing.add(
        Rect(
            x0 * mm,
            block_y0 * mm,
            (poly_x1 - x0) * mm,
            (block_y1 - block_y0) * mm,
            rx=1.2 * mm,
            ry=1.2 * mm,
            fillColor=HexColor("#EEF3F8"),
            strokeColor=HexColor(BLUE),
            strokeWidth=0.55,
        )
    )
    base.core.add_text(
        drawing,
        (x0 + poly_x1) / 2,
        block_y1 - 3.0,
        "Population adaptation",
        5.9,
        bold=True,
        anchor="middle",
        color="#234E73",
    )

    xa = base.x_log(ADAPTIVE_LOW, x0 + 4.0, poly_x1 - 3.0)
    xb = base.x_log(ADAPTIVE_HIGH, x0 + 4.0, poly_x1 - 3.0)
    xm = base.x_log(ADAPTIVE_MEDIAN, x0 + 4.0, poly_x1 - 3.0)
    adapt_y = 25.5
    base.core.add_line(drawing, xa, adapt_y, xb, adapt_y, color=BLUE, width=0.72)
    base.core.add_line(drawing, xa, adapt_y - 0.55, xa, adapt_y + 0.55, color=BLUE, width=0.48)
    base.core.add_line(drawing, xb, adapt_y - 0.55, xb, adapt_y + 0.55, color=BLUE, width=0.48)
    add_diamond(drawing, xm, adapt_y, radius=1.55, fill=BLUE, stroke=base.BLACK)
    base.core.add_text(
        drawing,
        (xa + xb) / 2,
        adapt_y - 3.2,
        "Adaptive haplotypes",
        5.75,
        anchor="middle",
    )
    base.core.add_text(
        drawing,
        poly_x1 - 2.2,
        block_y0 + 1.6,
        "14 loci | median 16.2% | 1.2–34.4%",
        5.15,
        anchor="end",
        color="#234E73",
    )

    # Lineage-level terminal block.
    drawing.add(
        Rect(
            fixed_x0 * mm,
            block_y0 * mm,
            (fixed_x1 - fixed_x0) * mm,
            (block_y1 - block_y0) * mm,
            rx=1.2 * mm,
            ry=1.2 * mm,
            fillColor=HexColor(TEAL_WASH),
            strokeColor=HexColor(TEAL_DARK),
            strokeWidth=0.55,
        )
    )
    base.core.add_text(
        drawing,
        (fixed_x0 + fixed_x1) / 2,
        block_y1 - 2.4,
        "Fixed lineage",
        5.45,
        bold=True,
        anchor="middle",
        color=TEAL_DARK,
    )
    gx = fixed_x0 + 3.2
    add_square(drawing, gx, 27.7, filled=False)
    add_square(drawing, gx, 22.8, filled=True)
    base.core.add_text(drawing, gx + 2.5, 27.0, "HAR", 5.7, bold=True, color=TEAL_DARK)
    base.core.add_text(drawing, gx + 2.5, 22.1, "HAQER", 5.5, bold=True, color=TEAL_DARK)
    base.core.add_text(
        drawing,
        (fixed_x0 + fixed_x1) / 2,
        block_y0 + 1.5,
        "region-level",
        5.2,
        anchor="middle",
        color=TEAL_DARK,
    )

    # Shared frequency/fixation axis.
    base.core.add_line(drawing, x0, y0, poly_x1, y0, width=0.68)
    base.core.add_line(drawing, fixed_x0, y0, fixed_x1, y0, width=0.68)
    add_break(drawing, 74.0, y0)
    for value, label in ((1e-3, "0.1%"), (1e-2, "1%"), (5e-2, "5%"), (5e-1, "50%")):
        px = base.x_log(value, x0, poly_x1)
        base.core.add_line(drawing, px, y0, px, y0 - 0.8, width=0.45)
        base.core.add_text(drawing, px, y0 - 2.8, label, 5.4, anchor="middle")
    fixed_center = (fixed_x0 + fixed_x1) / 2
    base.core.add_line(drawing, fixed_center, y0, fixed_center, y0 - 0.8, width=0.45)
    base.core.add_text(drawing, fixed_center, y0 - 2.8, "fixed", 5.4, anchor="middle")
    base.core.add_text(
        drawing,
        (x0 + fixed_x1) / 2,
        2.4,
        "Derived allele frequency / lineage fixation status",
        6.0,
        anchor="middle",
    )


def draw_evolution_only_integrated(drawing: Drawing, *, x: float, width: float) -> None:
    """Evolution-only alternative redrawn natively in the full Figure 3 slot."""
    base.core.add_text(drawing, x, 291.3, "b", 13.0, bold=True)
    base.core.add_text(
        drawing,
        x + width / 2,
        291.0,
        "Evolutionary adaptation and human-lineage acceleration",
        6.9,
        bold=True,
        anchor="middle",
    )

    axis_y = 228.8
    x0, poly_x1 = x + 8.0, x + 72.0
    fixed_x0, fixed_x1 = x + 78.0, x + 96.0
    block_y0, block_y1 = 238.0, 283.2

    drawing.add(
        Rect(
            x0 * mm,
            block_y0 * mm,
            (poly_x1 - x0) * mm,
            (block_y1 - block_y0) * mm,
            rx=1.2 * mm,
            ry=1.2 * mm,
            fillColor=HexColor("#EEF3F8"),
            strokeColor=HexColor(BLUE),
            strokeWidth=0.55,
        )
    )
    base.core.add_text(
        drawing,
        (x0 + poly_x1) / 2,
        block_y1 - 3.1,
        "Population adaptation",
        5.9,
        bold=True,
        anchor="middle",
        color="#234E73",
    )
    xa = base.x_log(ADAPTIVE_LOW, x0 + 4.0, poly_x1 - 3.0)
    xb = base.x_log(ADAPTIVE_HIGH, x0 + 4.0, poly_x1 - 3.0)
    xm = base.x_log(ADAPTIVE_MEDIAN, x0 + 4.0, poly_x1 - 3.0)
    adapt_y = 260.6
    base.core.add_line(drawing, xa, adapt_y, xb, adapt_y, color=BLUE, width=0.72)
    base.core.add_line(drawing, xa, adapt_y - 0.55, xa, adapt_y + 0.55, color=BLUE, width=0.48)
    base.core.add_line(drawing, xb, adapt_y - 0.55, xb, adapt_y + 0.55, color=BLUE, width=0.48)
    add_diamond(drawing, xm, adapt_y, radius=1.55, fill=BLUE, stroke=base.BLACK)
    base.core.add_text(
        drawing,
        (xa + xb) / 2,
        adapt_y - 3.4,
        "Adaptive haplotypes",
        5.75,
        anchor="middle",
    )
    base.core.add_text(
        drawing,
        poly_x1 - 2.2,
        block_y0 + 2.1,
        "14 loci | median 16.2% | 1.2–34.4%",
        5.15,
        anchor="end",
        color="#234E73",
    )

    drawing.add(
        Rect(
            fixed_x0 * mm,
            block_y0 * mm,
            (fixed_x1 - fixed_x0) * mm,
            (block_y1 - block_y0) * mm,
            rx=1.2 * mm,
            ry=1.2 * mm,
            fillColor=HexColor(TEAL_WASH),
            strokeColor=HexColor(TEAL_DARK),
            strokeWidth=0.55,
        )
    )
    base.core.add_text(
        drawing,
        (fixed_x0 + fixed_x1) / 2,
        block_y1 - 3.1,
        "Fixed lineage",
        5.45,
        bold=True,
        anchor="middle",
        color=TEAL_DARK,
    )
    glyph_x = fixed_x0 + 3.2
    add_square(drawing, glyph_x, 264.8, filled=False)
    add_square(drawing, glyph_x, 253.0, filled=True)
    base.core.add_text(drawing, glyph_x + 2.5, 264.1, "HAR", 5.7, bold=True, color=TEAL_DARK)
    base.core.add_text(drawing, glyph_x + 2.5, 252.3, "HAQER", 5.5, bold=True, color=TEAL_DARK)
    base.core.add_text(
        drawing,
        (fixed_x0 + fixed_x1) / 2,
        block_y0 + 2.1,
        "region-level",
        5.2,
        anchor="middle",
        color=TEAL_DARK,
    )

    base.core.add_line(drawing, x0, axis_y, poly_x1, axis_y, width=0.68)
    base.core.add_line(drawing, fixed_x0, axis_y, fixed_x1, axis_y, width=0.68)
    add_break(drawing, x + 74.0, axis_y)
    for value, label in ((1e-3, "0.1%"), (1e-2, "1%"), (5e-2, "5%"), (5e-1, "50%")):
        px = base.x_log(value, x0, poly_x1)
        base.core.add_line(drawing, px, axis_y, px, axis_y - 0.8, width=0.45)
        base.core.add_text(drawing, px, axis_y - 2.8, label, 5.4, anchor="middle")
    fixed_center = (fixed_x0 + fixed_x1) / 2
    base.core.add_line(drawing, fixed_center, axis_y, fixed_center, axis_y - 0.8, width=0.45)
    base.core.add_text(drawing, fixed_center, axis_y - 2.8, "fixed", 5.4, anchor="middle")
    base.core.add_text(
        drawing,
        (x0 + fixed_x1) / 2,
        axis_y - 6.6,
        "Derived allele frequency / lineage fixation status",
        6.0,
        anchor="middle",
    )


def render_evolution_only() -> tuple[Path, Path, Path]:
    width_mm, height_mm = 98.0, 46.0
    drawing = Drawing(width_mm * mm, height_mm * mm)
    drawing.add(
        Rect(
            0,
            0,
            width_mm * mm,
            height_mm * mm,
            fillColor=HexColor(base.WHITE),
            strokeColor=None,
        )
    )
    draw_evolution_only_panel(drawing)
    return render_artifact(drawing, PANEL_ONLY_STEM)


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
    import subprocess

    subprocess.run(
        [str(poppler), "-png", "-singlefile", "-r", "600", str(pdf), str(png.with_suffix(""))],
        check=True,
    )
    return pdf, svg, png


def render_integrated(
    clinvar: pd.DataFrame,
    static: pd.DataFrame,
    temporal: pd.DataFrame,
    checks: dict,
    placements: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    original = base.draw_architecture_panel

    def revised_panel(
        drawing: Drawing,
        placement_frame: pd.DataFrame,
        *,
        panel: str,
        letter: str,
        title: str,
        x: float,
        width: float,
    ) -> None:
        if panel != "a":
            original(
                drawing,
                placement_frame,
                panel=panel,
                letter=letter,
                title=title,
                x=x,
                width=width,
            )
            return
        draw_revised_b_panel(
            drawing,
            placement_frame,
            x=x,
            width=width,
            letter_y=291.3,
            title_y=291.0,
            y0=225.5,
            y1=287.2,
        )

    base.draw_architecture_panel = revised_panel
    base.OUT_DIR = OUT_DIR
    base.SOURCE_DIR = SOURCE_DIR
    base.MAIN_STEM = MAIN_STEM
    try:
        return base.render_main(clinvar, static, temporal, checks, placements)
    finally:
        base.draw_architecture_panel = original


def render_integrated_evolution_only(
    clinvar: pd.DataFrame,
    static: pd.DataFrame,
    temporal: pd.DataFrame,
    checks: dict,
    placements: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    original = base.draw_architecture_panel

    def revised_panel(
        drawing: Drawing,
        placement_frame: pd.DataFrame,
        *,
        panel: str,
        letter: str,
        title: str,
        x: float,
        width: float,
    ) -> None:
        if panel != "a":
            original(
                drawing,
                placement_frame,
                panel=panel,
                letter=letter,
                title=title,
                x=x,
                width=width,
            )
            return
        draw_evolution_only_integrated(drawing, x=x, width=width)

    base.draw_architecture_panel = revised_panel
    base.OUT_DIR = OUT_DIR
    base.SOURCE_DIR = SOURCE_DIR
    base.MAIN_STEM = MAIN_ONLY_STEM
    try:
        return base.render_main(clinvar, static, temporal, checks, placements)
    finally:
        base.draw_architecture_panel = original


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_supporting_files(
    placements: pd.DataFrame,
    integrated: tuple[Path, Path, Path],
    standalone: tuple[Path, Path, Path],
    evolution_only: tuple[Path, Path, Path],
    integrated_evolution_only: tuple[Path, Path, Path],
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    placements.to_csv(SOURCE_DIR / "variant_map_placements_v11_display.csv", index=False)
    source_rows = [
        {
            "benchmark": "Adaptive haplotypes",
            "grain": "allele-centered phased haplotype pair",
            "frequency_status": "observed ALT fraction across 14 common loci",
            "frequency_median": ADAPTIVE_MEDIAN,
            "frequency_low": ADAPTIVE_LOW,
            "frequency_high": ADAPTIVE_HIGH,
            "display_status": "numeric interval in polymorphic adaptive region",
            "evidence_note": "Source curation remains a submission gate.",
        },
        {
            "benchmark": "HAR",
            "grain": "accelerated genomic region",
            "frequency_status": "historical human-lineage substitutions; mostly fixed",
            "frequency_median": "",
            "frequency_low": "",
            "frequency_high": "",
            "display_status": "categorical fixed-lineage terminal cap; open square",
            "evidence_note": "Acceleration may reflect selection, gBGC or loss of constraint.",
        },
        {
            "benchmark": "HAQER",
            "grain": "quickly evolved genomic region",
            "frequency_status": "ancient human-lineage divergence followed by hominin constraint",
            "frequency_median": "",
            "frequency_low": "",
            "frequency_high": "",
            "display_status": "categorical fixed-lineage terminal cap; filled square",
            "evidence_note": "Directional-positive-selection evidence is stronger than for generic HARs.",
        },
    ]
    with (SOURCE_DIR / "Figure3b_evolution_benchmark_display_source.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    caption = """# Figure 3b revision caption addendum

**b,** Conceptual placement of variant- and region-level evolutionary benchmarks. The adaptive-haplotype diamond is positioned by the observed adaptive-ALT fraction among the 14 common locally tracked loci (median 16.2%; range 1.2–34.4%); its vertical coordinate is schematic rather than a measured selection coefficient. HAR and HAQER benchmarks are shown in a discontinuous `fixed` terminal cap because they classify human-lineage accelerated regions rather than present-day polymorphic alleles. The open HAR square denotes mixed mechanistic evidence, including positive selection, GC-biased gene conversion and loss of constraint; the filled HAQER square denotes the stronger published evidence for episodic directional positive selection before the human–Neanderthal split followed by hominin constraint. The terminal-cap positions do not represent per-region derived-allele frequencies or fitness coefficients.
"""
    (OUT_DIR / "Figure3b_evolution_benchmarks_CAPTION_ADDENDUM.md").write_text(
        caption, encoding="utf-8"
    )

    qa = {
        "status": "DATA_QC_DRAFT",
        "submission_status": "NOT_SUBMISSION_READY",
        "design_contract": {
            "takeaway": "Separate polymorphic adaptation from fixed human-lineage acceleration.",
            "polymorphic_marker": "adaptive haplotypes with observed frequency interval",
            "fixed_cap": "HAR and HAQER region-level classification",
            "non_color_encoding": "diamond vs open square vs filled square",
            "gridlines": False,
            "text_color": "pure black except teal module labels",
        },
        "known_submission_gates": [
            "Tracked HAQER benchmark contains 1,541 positive sequences whereas the draft methods describe 3,275 integrated intervals.",
            "HAR is 511 bp and HAQER is 512 bp in the tracked builder.",
            "Current negative sampling is intergenic but not explicitly matched for GC, CpG, repeats, conservation or mappability.",
            "Current builder stores replicate seeds as an integer although source results contain five negative sets.",
            "Adaptive-locus provenance and the use of global ALT fraction as a DAF proxy require final curation.",
        ],
        "artifacts": {
            "integrated": {str(p): sha256(p) for p in integrated},
            "standalone": {str(p): sha256(p) for p in standalone},
            "evolution_only": {str(p): sha256(p) for p in evolution_only},
            "integrated_evolution_only": {
                str(p): sha256(p) for p in integrated_evolution_only
            },
        },
    }


def main() -> None:
    base.core.configure_fonts()
    clinvar, static, temporal, checks = base.core.read_sources()
    placements, _summary = base.read_architecture_sources()
    standalone = render_standalone(placements)
    evolution_only = render_evolution_only()
    integrated = render_integrated(clinvar, static, temporal, checks, placements)
    integrated_evolution_only = render_integrated_evolution_only(
        clinvar, static, temporal, checks, placements
    )
    write_supporting_files(
        placements,
        integrated,
        standalone,
        evolution_only,
        integrated_evolution_only,
    )
    print(OUT_DIR)
    for path in (*standalone, *evolution_only, *integrated, *integrated_evolution_only):
        print(path)


if __name__ == "__main__":
    main()
