#!/usr/bin/env python3
"""Render the integrated Figure 3 (a-k) on an A4 manuscript canvas.

The renderer is source-table driven and keeps three evidence layers separate:
ClinVar stratum discrimination (a-c), temporal ClinVar reclassification (d-f),
and five static variant-effect benchmarks (g-k).  No source estimate, interval,
baseline, model choice, or denominator is recomputed by this layout script.
"""

from __future__ import annotations

# [Integration core] Base data preparation and panel plotting for Fig. 3 v1; the v2 entry point reuses shared functions from this module.
# [Input] ClinVar, static-variant, and temporal source data. [Plotting] render/build functions return a ReportLab Drawing.
# [Status] Retained as a compatibility core; the current main entry point is plot_figure3_abcdefghijk_a4_integrated_v2.py.

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from figures.scripts.common.font_config import register_reportlab_arial  # noqa: E402
from figures.scripts.common.hevo_model_style import canonical_model_name  # noqa: E402

TABLE_DIR = PROJECT_ROOT / "tables"
OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig3"
    / "figure3_abcdefghijk_a4_integrated_v1"
)
SOURCE_DIR = OUT_DIR / "source_data"

CLINVAR_SOURCE = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig3"
    / "fig3abc_clinvar_main_full_family_v5"
    / "source_data"
    / "Figure3abc_main_18_algorithm_representatives_plot_source.csv"
)
STATIC_SOURCE = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig3"
    / "fig3f_to_j_static_variant_scoring_v2"
    / "source_data"
    / "Figure3f-j_main_representatives_plot_source.csv"
)
TEMPORAL_SOURCE = TABLE_DIR / "S32_retrospective_enrichment_of_clinvar_reclassification_events_by_cutoff_year.csv"

STEM = "Figure3_abcdefghijk_A4_integrated_v1_DATA_QC_DRAFT"
STATUS = "DATA_QC_DRAFT"
WIDTH_MM = 210.0
HEIGHT_MM = 297.0
PNG_DPI = 600

BLACK = "#000000"
WHITE = "#FFFFFF"
HEVO = "#B65D1D"
NT = "#65717F"
EVO2 = "#0E7C86"
GPN = "#0B2E83"
NEUTRAL = "#B8C0CB"
HEVO_WASH = "#F6E8DD"

FAMILY_COLORS = {
    "hEVO": HEVO,
    "NT": NT,
    "Evo 2": EVO2,
    "GPN": GPN,
    "Other": NEUTRAL,
}

CLINVAR_META = {
    "all": {
        "letter": "a",
        "title": "ClinVar all",
        "scale": (0.20, 1.00),
        "ticks": (0.20, 0.60, 1.00),
    },
    "coding": {
        "letter": "b",
        "title": "ClinVar coding",
        "scale": (0.40, 1.00),
        "ticks": (0.40, 0.70, 1.00),
    },
    "noncoding": {
        "letter": "c",
        "title": "ClinVar noncoding",
        "scale": (0.00, 1.00),
        "ticks": (0.00, 0.50, 1.00),
    },
}

TEMPORAL_META = {
    "blb_stable_vs_upgraded": {
        "letter": "d",
        "title": "LB/B → later upgrade",
        "limits": (0.15, 2.15),
        "ticks": (0.5, 1.0, 1.5, 2.0),
    },
    "plp_stable_vs_downgraded": {
        "letter": "e",
        "title": "P/LP → later downgrade",
        "limits": (0.00, 7.65),
        "ticks": (0.0, 2.0, 4.0, 6.0),
    },
    "vus_resolved_path_vs_benign": {
        "letter": "f",
        "title": "VUS → P/LP",
        "limits": (0.70, 2.06),
        "ticks": (0.8, 1.2, 1.6, 2.0),
    },
}

STATIC_META = {
    "causal": {
        "letter": "g",
        "title": "CausalDB disease",
        "table": "S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv",
        # GPN-Star-M reaches 0.150 AUPRC with a reported CI upper bound of
        # 0.169; keep the full interval visible in the main-figure panel.
        # Include the current hEvo-large confidence interval through 0.20.
        "scale": (0.07, 0.20),
        "ticks": (0.08, 0.14, 0.20),
    },
    "rare_causal": {
        "letter": "h",
        "title": "Rare causal variants",
        "table": "S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv",
        # The updated hEvo-large row reaches the upper 0.8 range.
        "scale": (0.30, 0.65),
        "ticks": (0.30, 0.45, 0.60),
    },
    "mpra": {
        "letter": "i",
        "title": "MPRA",
        "table": "S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv",
        # Keep the complete current hEvo-large confidence interval visible.
        "scale": (0.20, 0.35),
        "ticks": (0.20, 0.28, 0.34),
    },
    "rare_ukb": {
        "letter": "j",
        "title": "UKB rare variants",
        "table": "S28_zero_shot_classification_of_uk_biobank_rare_variants.csv",
        # The current GitLab S30 snapshot spans AUPRC means 0.623-0.724
        # and reported 95% CI upper bounds through 0.751. Keep the full
        # interval visible; the previous 0.40-0.65 range clipped all CI
        # upper ends and the prevalence baseline at the right edge.
        "scale": (0.58, 0.80),
        "ticks": (0.60, 0.70, 0.80),
    },
    "ddd_clinvar": {
        "letter": "k",
        "title": "DDD de novo vs ClinVar benign",
        "table": "S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv",
        "scale": (0.00, 1.00),
        "ticks": (0.00, 0.50, 1.00),
    },
}

TEMPORAL_MODEL_STYLE = {
    "hEvo-large": {
        "label": "hEvo-large",
        "color": HEVO,
        "marker": "o",
        "fill": HEVO,
        "dash": None,
        "width": 0.95,
    },
    "hEvo-base": {
        "label": "hEvo-base",
        "color": HEVO,
        "marker": "o",
        "fill": WHITE,
        "dash": [2.4, 1.5],
        "width": 0.80,
    },
    "evo2_7b_base": {
        "label": "EVO 2-7B",
        "color": EVO2,
        "marker": "^",
        "fill": EVO2,
        "dash": None,
        "width": 0.72,
    },
    "GPN-MSA": {
        "label": "GPN-MSA",
        "color": GPN,
        "marker": "D",
        "fill": GPN,
        "dash": None,
        "width": 0.72,
    },
    "Genos-1.2B": {
        "label": "Genos-1.2B",
        "color": NT,
        "marker": "s",
        "fill": NT,
        "dash": None,
        "width": 0.68,
    },
    "GPN-Star-V": {
        "label": "GPN-Star-V",
        "color": "#6D5AA8",
        "marker": "^",
        "fill": "#6D5AA8",
        "dash": None,
        "width": 0.72,
    },
    "GPN-Star-M": {
        "label": "GPN-Star-M",
        "color": "#3C73B8",
        "marker": "D",
        "fill": "#3C73B8",
        "dash": None,
        "width": 0.72,
    },
    "GPN-Star-P": {
        "label": "GPN-Star-P",
        "color": "#168B8B",
        "marker": "s",
        "fill": "#168B8B",
        "dash": None,
        "width": 0.72,
    },
}
for _temporal_model, _temporal_style in TEMPORAL_MODEL_STYLE.items():
    _temporal_style["label"] = canonical_model_name(_temporal_model)

TEMPORAL_DRAW_ORDER = (
    "Genos-1.2B",
    "evo2_7b_base",
    "GPN-MSA",
    "hEvo-base",
    "hEvo-large",
    "GPN-Star-P",
    "GPN-Star-V",
    "GPN-Star-M",
)
TEMPORAL_LEGEND_ORDER = (
    "hEvo-large",
    "hEvo-base",
    "evo2_7b_base",
    "GPN-MSA",
    "Genos-1.2B",
    "GPN-Star-V",
    "GPN-Star-M",
    "GPN-Star-P",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_fonts() -> None:
    register_reportlab_arial()


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    size: float,
    *,
    bold: bool = False,
    anchor: str = "start",
    color: str = BLACK,
) -> None:
    drawing.add(
        String(
            x * mm,
            y * mm,
            text,
            fontName="Arial-Bold" if bold else "Arial",
            fontSize=size,
            fillColor=HexColor(color),
            textAnchor=anchor,
        )
    )


def add_line(
    drawing: Drawing,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = BLACK,
    width: float = 0.55,
    dash: list[float] | None = None,
) -> None:
    line = Line(
        x0 * mm,
        y0 * mm,
        x1 * mm,
        y1 * mm,
        strokeColor=HexColor(color),
        strokeWidth=width,
    )
    if dash:
        line.strokeDashArray = dash
    drawing.add(line)


def add_marker(
    drawing: Drawing,
    x: float,
    y: float,
    model: str,
    radius_mm: float = 0.63,
) -> None:
    style = TEMPORAL_MODEL_STYLE[model]
    stroke = HexColor(style["color"])
    fill = HexColor(style["fill"])
    r = radius_mm * mm
    marker = style["marker"]
    if marker == "o":
        drawing.add(
            Circle(
                x * mm, y * mm, r, fillColor=fill, strokeColor=stroke, strokeWidth=0.50
            )
        )
    elif marker == "s":
        drawing.add(
            Rect(
                (x - radius_mm) * mm,
                (y - radius_mm) * mm,
                2 * r,
                2 * r,
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=0.50,
            )
        )
    elif marker == "^":
        drawing.add(
            Polygon(
                [
                    x * mm,
                    (y + radius_mm * 1.15) * mm,
                    (x - radius_mm * 1.04) * mm,
                    (y - radius_mm * 0.85) * mm,
                    (x + radius_mm * 1.04) * mm,
                    (y - radius_mm * 0.85) * mm,
                ],
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=0.50,
            )
        )
    elif marker == "D":
        drawing.add(
            Polygon(
                [
                    x * mm,
                    (y + radius_mm * 1.08) * mm,
                    (x - radius_mm * 1.08) * mm,
                    y * mm,
                    x * mm,
                    (y - radius_mm * 1.08) * mm,
                    (x + radius_mm * 1.08) * mm,
                    y * mm,
                ],
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=0.50,
            )
        )


def scale(value: float, low: float, high: float, p0: float, p1: float) -> float:
    clipped = min(max(float(value), low), high)
    return p0 + (clipped - low) / (high - low) * (p1 - p0)


def format_tick(value: float, low: float, high: float) -> str:
    span = high - low
    if span <= 0.30:
        return f"{value:.2f}"
    if abs(value - round(value)) < 1e-8:
        return f"{value:.0f}"
    return f"{value:.1f}"


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    for path in (CLINVAR_SOURCE, STATIC_SOURCE, TEMPORAL_SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)

    clinvar = pd.read_csv(CLINVAR_SOURCE)
    static = pd.read_csv(STATIC_SOURCE)
    temporal = pd.read_csv(TEMPORAL_SOURCE)

    clin_required = {
        "task",
        "model",
        "model_display",
        "family",
        "auprc_mean",
        "auprc_ci_lower",
        "auprc_ci_upper",
        "num_samples",
        "baseline",
    }
    static_required = {
        "task",
        "model",
        "model_display",
        "family",
        "score",
        "ci_lower",
        "ci_upper",
        "baseline_value",
    }
    temporal_required = {
        "model_label",
        "task_name",
        "cutoff_year",
        "precision_at_k_mean",
        "enrichment_vs_baseline_mean",
        "enrichment_vs_baseline_ci_low",
        "enrichment_vs_baseline_ci_high",
        "top_fraction",
        "n_boot",
        "seed_count",
    }
    for frame, required, label in (
        (clinvar, clin_required, "clinvar"),
        (static, static_required, "static"),
        (temporal, temporal_required, "temporal"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label}: missing columns {missing}")

    if set(clinvar["task"]) != set(CLINVAR_META):
        raise ValueError(
            "ClinVar task set is not the expected all/coding/noncoding set"
        )
    if not all(len(clinvar[clinvar["task"].eq(task)]) == 17 for task in CLINVAR_META):
        raise ValueError(
            "Each ClinVar task must contain 17 eligible algorithm representatives"
        )
    if set(static["task"]) != set(STATIC_META):
        raise ValueError("Static task set is not the expected five-benchmark set")
    if not all(len(static[static["task"].eq(task)]) == 17 for task in STATIC_META):
        raise ValueError(
            "Each static task must contain 17 eligible algorithm representatives"
        )
    if set(temporal["task_name"]) != set(TEMPORAL_META):
        raise ValueError("Temporal task set does not match Candidate A")
    if set(temporal["model_label"]) != set(TEMPORAL_MODEL_STYLE):
        raise ValueError("Temporal model set does not match Candidate A")
    if (
        len(temporal) != 216
        or temporal.duplicated(["task_name", "model_label", "cutoff_year"]).any()
    ):
        raise ValueError("Temporal table is not the complete 8 x 3 x 9 grid")
    if set(temporal["top_fraction"].astype(float)) != {0.01}:
        raise ValueError("Temporal figure supports only the reported top 1%")

    temporal["event_prevalence"] = (
        temporal["precision_at_k_mean"] / temporal["enrichment_vs_baseline_mean"]
    )
    spread = (
        temporal.groupby(["task_name", "cutoff_year"])["event_prevalence"]
        .agg(lambda values: float(values.max() - values.min()))
        .max()
    )
    if spread >= 1e-12:
        raise ValueError("Implied temporal event prevalence differs across models")

    static_stats: dict[str, dict[str, float | int | str]] = {}
    raw_hashes: dict[str, str] = {}
    for task, meta in STATIC_META.items():
        raw_path = TABLE_DIR / str(meta["table"])
        raw = pd.read_csv(raw_path)
        first = raw.iloc[0]
        baseline = float(first["baseline"])
        plotted_baseline = float(
            static.loc[static["task"].eq(task), "baseline_value"].iloc[0]
        )
        if abs(baseline - plotted_baseline) > 1e-12:
            raise ValueError(f"{task}: processed baseline differs from raw table")
        static_stats[task] = {
            "num_samples": int(first["num_samples"]),
            "num_positive": int(first["num_positive"]),
            "num_negative": int(first["num_negative"]),
            "baseline": baseline,
            "table": raw_path.name,
        }
        raw_hashes[raw_path.name] = sha256(raw_path)

    checks = {
        "clinvar_rows": int(len(clinvar)),
        "static_rows": int(len(static)),
        "temporal_rows": int(len(temporal)),
        "clinvar_models_per_task": {
            task: int(len(clinvar[clinvar["task"].eq(task)])) for task in CLINVAR_META
        },
        "static_models_per_task": {
            task: int(len(static[static["task"].eq(task)])) for task in STATIC_META
        },
        "temporal_complete_grid": True,
        "temporal_top_fraction": 0.01,
        "temporal_n_boot": sorted(temporal["n_boot"].astype(int).unique().tolist()),
        "temporal_seed_count": sorted(
            temporal["seed_count"].astype(int).unique().tolist()
        ),
        "temporal_implied_prevalence_max_model_spread": float(spread),
        "source_hashes": {
            CLINVAR_SOURCE.name: sha256(CLINVAR_SOURCE),
            STATIC_SOURCE.name: sha256(STATIC_SOURCE),
            TEMPORAL_SOURCE.name: sha256(TEMPORAL_SOURCE),
            **raw_hashes,
        },
        "static_task_stats": static_stats,
    }
    return clinvar, static, temporal, checks


def draw_family_legend(drawing: Drawing) -> None:
    x = 6.0
    y = 293.1
    for family, width in (
        ("hEVO", 20.5),
        ("NT", 15.5),
        ("Evo 2", 20.5),
        ("GPN", 17.5),
        ("Other", 20.0),
    ):
        color = FAMILY_COLORS[family]
        drawing.add(
            Rect(
                x * mm,
                (y - 1.15) * mm,
                4.0 * mm,
                1.75 * mm,
                fillColor=HexColor(color),
                strokeColor=None,
            )
        )
        add_text(
            drawing, x + 5.0, y - 0.65, "EVO 2" if family == "Evo 2" else family, 6.2
        )
        x += width
    add_text(
        drawing,
        204.0,
        y - 0.65,
        "local AUPRC scales | whisker: 95% CI | black tick: prevalence",
        6.0,
        anchor="end",
    )


def draw_bar_panel(
    drawing: Drawing,
    rows: pd.DataFrame,
    *,
    x: float,
    bottom: float,
    width: float,
    top: float,
    letter: str,
    title: str,
    n_samples: int,
    baseline: float,
    limits: tuple[float, float],
    ticks: tuple[float, ...],
    score_col: str,
    low_col: str,
    high_col: str,
) -> None:
    rows = rows.sort_values(
        [score_col, "model_display"], ascending=[False, True], kind="stable"
    )
    if len(rows) != 17:
        raise ValueError(f"{letter}: expected 17 rows, found {len(rows)}")

    label_width = 29.2 if width < 80 else 39.5
    value_width = 8.1
    plot_x0 = x + label_width
    plot_x1 = x + width - value_width
    value_x = x + width - 0.7
    axis_y = top - 10.6
    row_top = top - 14.3
    row_bottom = bottom + 2.1
    row_pitch = (row_top - row_bottom) / max(1, len(rows) - 1)
    low, high = limits

    add_text(drawing, x, top - 1.0, letter, 13.0, bold=True)
    add_text(drawing, x + width / 2, top - 1.2, title, 7.2, bold=True, anchor="middle")
    has_rowwise_prevalence = {"num_samples", "baseline_value"}.issubset(rows.columns)
    if has_rowwise_prevalence:
        sample_counts = pd.to_numeric(rows["num_samples"], errors="raise")
        baseline_values = pd.to_numeric(rows["baseline_value"], errors="raise")
        if (sample_counts <= 0).any() or not np.isfinite(baseline_values).all():
            raise ValueError(f"{letter}: invalid row-level sample count or prevalence")
        if sample_counts.nunique() == 1 and baseline_values.nunique() == 1:
            header_text = f"AUPRC | N={int(sample_counts.iloc[0]):,} | base={float(baseline_values.iloc[0]):.3f}"
        else:
            header_text = "AUPRC | model-specific N and prevalence"
    else:
        header_text = f"AUPRC | N={n_samples:,} | base={baseline:.3f}"
    add_text(
        drawing,
        x + width / 2,
        top - 5.7,
        header_text,
        6.0,
        anchor="middle",
    )

    add_line(drawing, plot_x0, axis_y, plot_x1, axis_y, width=0.65)
    for tick in ticks:
        tx = scale(float(tick), low, high, plot_x0, plot_x1)
        add_line(drawing, tx, axis_y, tx, axis_y - 1.1, width=0.50)
        add_text(
            drawing,
            tx,
            axis_y + 1.0,
            format_tick(float(tick), low, high),
            6.0,
            anchor="middle",
        )

    bar_height = min(1.35, row_pitch * 0.52)
    for idx, (_, row) in enumerate(rows.iterrows()):
        y = row_top - idx * row_pitch
        is_hevo = str(row["family"]) == "hEVO"
        if is_hevo:
            drawing.add(
                Rect(
                    x * mm,
                    (y - row_pitch * 0.39) * mm,
                    width * mm,
                    (row_pitch * 0.78) * mm,
                    fillColor=HexColor(HEVO_WASH),
                    strokeColor=None,
                )
            )
        add_text(
            drawing,
            x + 0.7,
            y - 0.72,
            str(row["model_display"]),
            6.0,
            bold=is_hevo,
        )
        mean = float(row[score_col])
        mean_x = scale(mean, low, high, plot_x0, plot_x1)
        color = FAMILY_COLORS.get(str(row["family"]), NEUTRAL)
        drawing.add(
            Rect(
                plot_x0 * mm,
                (y - bar_height / 2) * mm,
                max(0.0, mean_x - plot_x0) * mm,
                bar_height * mm,
                fillColor=HexColor(color),
                strokeColor=None,
            )
        )
        ci_low = row[low_col]
        ci_high = row[high_col]
        if pd.notna(ci_low) and pd.notna(ci_high):
            lx = scale(float(ci_low), low, high, plot_x0, plot_x1)
            hx = scale(float(ci_high), low, high, plot_x0, plot_x1)
            add_line(drawing, lx, y, hx, y, width=0.62)
            add_line(drawing, lx, y - 0.62, lx, y + 0.62, width=0.52)
            add_line(drawing, hx, y - 0.62, hx, y + 0.62, width=0.52)
        row_baseline = (
            float(row["baseline_value"])
            if has_rowwise_prevalence
            else baseline
        )
        baseline_x = scale(row_baseline, low, high, plot_x0, plot_x1)
        add_line(drawing, baseline_x, y - 0.82, baseline_x, y + 0.82, width=0.58)
        add_text(
            drawing, value_x, y - 0.72, f"{mean:.3f}", 6.0, bold=is_hevo, anchor="end"
        )


def draw_temporal_legend(drawing: Drawing, y: float) -> None:
    # Eight entries fit on one A4 line at the compact panel typography.  Keep
    # the model names explicit so the legend remains machine-readable in SVG.
    widths = [26.0, 25.0, 26.0, 25.0, 25.0, 26.0, 26.0, 26.0]
    x = (WIDTH_MM - sum(widths)) / 2
    for model, item_width in zip(TEMPORAL_LEGEND_ORDER, widths):
        style = TEMPORAL_MODEL_STYLE[model]
        add_line(
            drawing,
            x,
            y,
            x + 6.0,
            y,
            color=style["color"],
            width=float(style["width"]),
            dash=style["dash"],
        )
        add_marker(drawing, x + 3.0, y, model, radius_mm=0.55)
        add_text(drawing, x + 7.3, y - 0.72, style["label"], 6.0)
        x += item_width


def draw_temporal_panel(
    drawing: Drawing,
    temporal: pd.DataFrame,
    *,
    task: str,
    x: float,
    width: float,
    top: float,
    bottom: float,
    inner_left: float = 10.3,
    inner_right: float = 1.2,
) -> None:
    meta = TEMPORAL_META[task]
    subset = temporal[temporal["task_name"].eq(task)]
    x0 = x + inner_left
    x1 = x + width - inner_right
    main_y0 = bottom + 14.5
    main_y1 = top - 10.7
    prevalence_y0 = bottom + 4.8
    prevalence_y1 = bottom + 10.2
    low, high = meta["limits"]

    add_text(drawing, x, top - 4.6, str(meta["letter"]), 13.0, bold=True)
    add_text(
        drawing,
        x + width / 2,
        top - 4.5,
        str(meta["title"]),
        7.0,
        bold=True,
        anchor="middle",
    )
    if task == "blb_stable_vs_upgraded":
        add_text(drawing, x0, main_y1 + 1.7, "Top-1% enrichment", 6.0)

    add_line(drawing, x0, main_y0, x0, main_y1, width=0.62)
    add_line(drawing, x0, main_y0, x1, main_y0, width=0.58)
    for tick in meta["ticks"]:
        ty = scale(float(tick), low, high, main_y0, main_y1)
        add_line(drawing, x0 - 0.9, ty, x0, ty, width=0.48)
        label = (
            f"{float(tick):.0f}"
            if task == "plp_stable_vs_downgraded"
            else f"{float(tick):.1f}"
        )
        add_text(drawing, x0 - 1.4, ty - 0.68, label, 6.0, anchor="end")
    null_y = scale(1.0, low, high, main_y0, main_y1)
    add_line(drawing, x0, null_y, x1, null_y, width=0.48, dash=[2.4, 1.7])

    for model in TEMPORAL_DRAW_ORDER:
        style = TEMPORAL_MODEL_STYLE[model]
        rows = subset[subset["model_label"].eq(model)].sort_values("cutoff_year")
        coords: list[tuple[float, float]] = []
        for row in rows.itertuples(index=False):
            px = scale(float(row.cutoff_year), 2017.0, 2025.0, x0, x1)
            py = scale(
                float(row.enrichment_vs_baseline_mean), low, high, main_y0, main_y1
            )
            ly = scale(
                float(row.enrichment_vs_baseline_ci_low), low, high, main_y0, main_y1
            )
            hy = scale(
                float(row.enrichment_vs_baseline_ci_high), low, high, main_y0, main_y1
            )
            add_line(drawing, px, ly, px, hy, color=style["color"], width=0.38)
            add_line(
                drawing, px - 0.48, ly, px + 0.48, ly, color=style["color"], width=0.38
            )
            add_line(
                drawing, px - 0.48, hy, px + 0.48, hy, color=style["color"], width=0.38
            )
            coords.append((px, py))
        for (xa, ya), (xb, yb) in zip(coords[:-1], coords[1:]):
            add_line(
                drawing,
                xa,
                ya,
                xb,
                yb,
                color=style["color"],
                width=float(style["width"]),
                dash=style["dash"],
            )
        for px, py in coords:
            add_marker(drawing, px, py, model, radius_mm=0.52)

    add_line(drawing, x0, prevalence_y0, x0, prevalence_y1, width=0.50)
    add_line(drawing, x0, prevalence_y0, x1, prevalence_y0, width=0.50)
    baseline = (
        subset.groupby("cutoff_year", as_index=False)["event_prevalence"]
        .first()
        .sort_values("cutoff_year")
    )
    percentages = 100.0 * baseline["event_prevalence"].to_numpy(float)
    if task == "plp_stable_vs_downgraded":
        p_low, p_high = -0.45, 7.0
    else:
        p_low, p_high = 47.5, 52.5
    coords = []
    for year, value in zip(baseline["cutoff_year"], percentages):
        px = scale(float(year), 2017.0, 2025.0, x0, x1)
        py = scale(float(value), p_low, p_high, prevalence_y0, prevalence_y1)
        coords.append((px, py))
    for (xa, ya), (xb, yb) in zip(coords[:-1], coords[1:]):
        add_line(drawing, xa, ya, xb, yb, width=0.54)
    for px, py in coords:
        drawing.add(
            Circle(
                px * mm,
                py * mm,
                0.38 * mm,
                fillColor=HexColor(WHITE),
                strokeColor=HexColor(BLACK),
                strokeWidth=0.45,
            )
        )
    if task == "blb_stable_vs_upgraded":
        add_text(drawing, x0, prevalence_y1 + 0.7, "Evaluated-set positives (%)", 6.0)
        add_text(drawing, x1, prevalence_y1 - 0.4, "50.0%", 6.0, anchor="end")
    elif task == "vus_resolved_path_vs_benign":
        add_text(drawing, x1, prevalence_y1 - 0.4, "50.0%", 6.0, anchor="end")
    else:
        add_text(drawing, x0 + 0.6, prevalence_y1 - 0.4, f"{percentages[0]:.2f}%", 6.0)
        add_text(
            drawing,
            x1,
            prevalence_y0 + 0.7,
            f"{percentages[-1]:.2f}%",
            6.0,
            anchor="end",
        )

    for year in (2017, 2021, 2025):
        px = scale(float(year), 2017.0, 2025.0, x0, x1)
        add_line(drawing, px, prevalence_y0, px, prevalence_y0 - 0.8, width=0.45)
        add_text(drawing, px, prevalence_y0 - 2.9, str(year), 6.0, anchor="middle")


def build_integrated_source(
    clinvar: pd.DataFrame,
    static: pd.DataFrame,
    temporal: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for task, meta in CLINVAR_META.items():
        part = clinvar[clinvar["task"].eq(task)].copy()
        part.insert(0, "panel_letter", meta["letter"])
        part.insert(1, "panel_group", "ClinVar stratum")
        part = part.rename(
            columns={
                "auprc_mean": "estimate",
                "auprc_ci_lower": "ci_lower",
                "auprc_ci_upper": "ci_upper",
                "baseline": "baseline_or_prevalence",
            }
        )
        parts.append(part)
    for task, meta in TEMPORAL_META.items():
        part = temporal[temporal["task_name"].eq(task)].copy()
        part.insert(0, "panel_letter", meta["letter"])
        part.insert(1, "panel_group", "Temporal ClinVar")
        part = part.rename(
            columns={
                "model_label": "model",
                "enrichment_vs_baseline_mean": "estimate",
                "enrichment_vs_baseline_ci_low": "ci_lower",
                "enrichment_vs_baseline_ci_high": "ci_upper",
                "event_prevalence": "baseline_or_prevalence",
            }
        )
        parts.append(part)
    for task, meta in STATIC_META.items():
        part = static[static["task"].eq(task)].copy()
        part.insert(0, "panel_letter", meta["letter"])
        part.insert(1, "panel_group", "Static variant benchmark")
        part = part.rename(
            columns={"score": "estimate", "baseline_value": "baseline_or_prevalence"}
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False)


def render(
    clinvar: pd.DataFrame,
    static: pd.DataFrame,
    temporal: pd.DataFrame,
    checks: dict,
) -> tuple[Path, Path, Path]:
    drawing = Drawing(WIDTH_MM * mm, HEIGHT_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            WIDTH_MM * mm,
            HEIGHT_MM * mm,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )
    draw_family_legend(drawing)

    margin = 5.0
    gap3 = 3.0
    width3 = (WIDTH_MM - 2 * margin - 2 * gap3) / 3.0
    x3 = [margin + idx * (width3 + gap3) for idx in range(3)]
    gap2 = 5.0
    width2 = (WIDTH_MM - 2 * margin - gap2) / 2.0
    x2 = [margin, margin + width2 + gap2]

    # a-c: ClinVar all, coding and noncoding.
    for x, (task, meta) in zip(x3, CLINVAR_META.items()):
        task_rows = clinvar[clinvar["task"].eq(task)]
        draw_bar_panel(
            drawing,
            task_rows,
            x=x,
            bottom=222.0,
            width=width3,
            top=287.0,
            letter=meta["letter"],
            title=meta["title"],
            n_samples=int(task_rows.iloc[0]["num_samples"]),
            baseline=float(task_rows.iloc[0]["baseline"]),
            limits=meta["scale"],
            ticks=meta["ticks"],
            score_col="auprc_mean",
            low_col="auprc_ci_lower",
            high_col="auprc_ci_upper",
        )

    # d-f: compact Candidate A, with the formula moved to the caption.
    draw_temporal_legend(drawing, 217.6)
    for x, task in zip(x3, TEMPORAL_META):
        draw_temporal_panel(
            drawing,
            temporal,
            task=task,
            x=x,
            width=width3,
            top=217.0,
            bottom=165.5,
        )

    # g-i: common-to-functional disease variant benchmarks.
    for x, task in zip(x3, ("causal", "rare_causal", "mpra")):
        meta = STATIC_META[task]
        stats = checks["static_task_stats"][task]
        draw_bar_panel(
            drawing,
            static[static["task"].eq(task)],
            x=x,
            bottom=95.0,
            width=width3,
            top=160.5,
            letter=meta["letter"],
            title=meta["title"],
            n_samples=int(stats["num_samples"]),
            baseline=float(stats["baseline"]),
            limits=meta["scale"],
            ticks=meta["ticks"],
            score_col="score",
            low_col="ci_lower",
            high_col="ci_upper",
        )

    # j-k: clinical/rare-disease endpoints receive the wider bottom row.
    for x, task in zip(x2, ("rare_ukb", "ddd_clinvar")):
        meta = STATIC_META[task]
        stats = checks["static_task_stats"][task]
        draw_bar_panel(
            drawing,
            static[static["task"].eq(task)],
            x=x,
            bottom=9.5,
            width=width2,
            top=89.5,
            letter=meta["letter"],
            title=meta["title"],
            n_samples=int(stats["num_samples"]),
            baseline=float(stats["baseline"]),
            limits=meta["scale"],
            ticks=meta["ticks"],
            score_col="score",
            low_col="ci_lower",
            high_col="ci_upper",
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{STEM}.pdf"
    svg = OUT_DIR / f"{STEM}.svg"
    png = OUT_DIR / f"{STEM}.png"
    renderPDF.drawToFile(drawing, str(pdf), title=STEM)
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
    subprocess.run(
        [
            str(poppler),
            "-png",
            "-singlefile",
            "-r",
            str(PNG_DPI),
            str(pdf),
            str(png.with_suffix("")),
        ],
        check=True,
    )
    return pdf, svg, png


def inspect_pdf_fonts(pdf: Path) -> dict:
    reader = PdfReader(str(pdf))
    names = set()
    embedded = []
    for page in reader.pages:
        resources = page.get("/Resources")
        if not resources or "/Font" not in resources:
            continue
        fonts = resources["/Font"].get_object()
        for font_ref in fonts.values():
            font = font_ref.get_object()
            names.add(str(font.get("/BaseFont", "")))
            descriptor = font.get("/FontDescriptor")
            if descriptor is not None:
                descriptor = descriptor.get_object()
                embedded.append(
                    any(
                        key in descriptor
                        for key in ("/FontFile", "/FontFile2", "/FontFile3")
                    )
                )
    return {
        "font_names": sorted(names),
        "all_detected_font_descriptors_embedded": bool(embedded and all(embedded)),
    }


def write_supporting_files(
    clinvar: pd.DataFrame,
    static: pd.DataFrame,
    temporal: pd.DataFrame,
    checks: dict,
    pdf: Path,
    svg: Path,
    png: Path,
) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for source in (CLINVAR_SOURCE, STATIC_SOURCE, TEMPORAL_SOURCE):
        shutil.copy2(source, SOURCE_DIR / source.name)
    for meta in STATIC_META.values():
        source = TABLE_DIR / str(meta["table"])
        shutil.copy2(source, SOURCE_DIR / source.name)
    integrated = build_integrated_source(clinvar, static, temporal)
    integrated.to_csv(
        SOURCE_DIR / "Figure3_abcdefghijk_integrated_plot_source.csv", index=False
    )

    reader = PdfReader(str(pdf))
    page = reader.pages[0]
    width_cm = float(page.mediabox.width) / 72.0 * 2.54
    height_cm = float(page.mediabox.height) / 72.0 * 2.54
    with Image.open(png) as image:
        pixels = image.size
        dpi = image.info.get("dpi", (None, None))
    svg_text = svg.read_text(encoding="utf-8").lower()
    qa = {
        "status": STATUS,
        "submission_status": "NOT_SUBMISSION_READY",
        "data_checks": checks,
        "artifact": {
            "pdf": str(pdf),
            "svg": str(svg),
            "png": str(png),
            "width_cm": width_cm,
            "height_cm": height_cm,
            "expected_width_cm": 21.0,
            "expected_height_cm": 29.7,
            "png_pixels": pixels,
            "png_dpi": dpi,
            **inspect_pdf_fonts(pdf),
            "svg_text_preserved": "<text" in svg_text,
            "forbidden_pink_absent": not any(
                token in svg_text
                for token in ("#ff6fae", "#d45087", "#f7dfe9", "#e78ac3")
            ),
            "hashes": {
                "pdf": sha256(pdf),
                "svg": sha256(svg),
                "png": sha256(png),
            },
        },
        "layout_contract": {
            "row_1": "a-c ClinVar all/coding/noncoding",
            "row_2": "d-f temporal ClinVar Candidate A",
            "row_3": "g-i CausalDB/rare causal/MPRA",
            "row_4": "j-k UKB rare/DDD de novo",
            "minimum_text_pt": 6.0,
            "gridlines": "none",
            "text_color": BLACK,
            "family_palette": FAMILY_COLORS,
        },
        "evidence_boundaries": [
            "ClinVar and static benchmark panels show locked best-parameter algorithm representatives, with both hEVO scales retained.",
            "Each AUPRC panel has its own explicitly labelled scale; bar lengths are not comparable across panels.",
            "Temporal panels reproduce the reported Top-1% summary only; Top-5% and threshold-free analyses require event-level scores.",
            "Temporal cutoffs have unequal follow-up horizons and represent retrospective reclassification discrimination, not prospective clinical validation.",
            "LB/B upgrade and VUS resolution are balanced at 50% prevalence and therefore have a 2x enrichment ceiling.",
            "The temporal source reports only n_boot=20; a submission analysis should use at least 1,000 variant- or cluster-level replicates.",
            "Confidence-interval provenance and prediction-level manifests remain submission gates for the current source snapshot.",
        ],
    }

    caption = """# Figure 3 caption draft

**Figure 3 | Zero-shot discrimination of clinical and disease-associated human variants.** **a-c,** AUPRC for ClinVar pathogenic/likely pathogenic versus benign/likely benign variants across all variants, coding variants and noncoding variants. **d-f,** reported enrichment of later ClinVar reclassification events among the top 1% of model-ranked variants across historical cutoff years: LB/B variants later upgraded, P/LP variants later downgraded, and VUS variants subsequently resolved as P/LP rather than LB/B. Enrichment is precision at the top 1% divided by the cutoff-specific event prevalence; error bars are the reported bootstrap 95% confidence intervals from five seeds and `n_boot = 20`. Lower strips show the corresponding event prevalence. **g-k,** AUPRC on CausalDB disease variants, rare causal variants, MPRA-tested causal variants, UKB disease-associated rare variants and DDD de novo LoF/splice variants versus matched benchmark negatives. For AUPRC panels, models are ordered independently by performance, horizontal whiskers denote reported 95% confidence intervals and black ticks mark positive-class prevalence; each panel uses its own labelled scale. hEvo-base and hEvo-large are retained as the two focal model scales, whereas parameter-scale variants of other algorithms are represented by the checkpoint locked in the source selection summary. The dynamic ClinVar panels are retrospective temporal reclassification analyses with unequal follow-up horizons and should not be interpreted as prospective clinical validation.
"""
    (OUT_DIR / "Figure3_abcdefghijk_CAPTION_DRAFT.md").write_text(
        caption, encoding="utf-8"
    )


def main() -> None:
    configure_fonts()
    clinvar, static, temporal, checks = read_sources()
    pdf, svg, png = render(clinvar, static, temporal, checks)
    write_supporting_files(clinvar, static, temporal, checks, pdf, svg, png)
    print(f"Rendered {pdf}")
    print(f"Rendered {svg}")
    print(f"Rendered {png}")


if __name__ == "__main__":
    main()
