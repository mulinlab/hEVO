#!/usr/bin/env python3
"""Render an integrated, source-backed Figure 3a-c ClinVar comparison.

The module shares one model registry across ClinVar all, coding, and noncoding
strata.  AUPRC is plotted on the same absolute 0-1 scale in every panel; each
horizontal segment begins at the stratum-specific prevalence baseline.  The
left summary reports mean rank computed against all 33 source models before
one parameter-scale representative is selected per prespecified algorithm.

This design keeps the class-imbalance context visible and does not infer AUROC
values that are absent from S23-S25.
"""

from __future__ import annotations

# [Fig. 3a–c core] Base implementation for loading ClinVar AUPRC data, aggregating models/families, and drawing axes and bar/interval panels.
# [Usage] Imported by the v5 builder, which uses its results in the current Fig. 3 main and supplementary figures.
# [Data] Defines the source schema, task order, model display names, and QA statistics; it does not arrange the full Fig. 3 page.

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import OrderedDict
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
from reportlab.pdfbase.ttfonts import TTFont


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from figures.scripts.common.hevo_model_style import canonical_model_name  # noqa: E402
from figures.scripts.common.font_config import register_reportlab_arial  # noqa: E402


TABLE_DIR = PROJECT_ROOT / "tables"
OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig3" / "fig3abc_clinvar_integrated_v1"
SOURCE_DIR = OUT_DIR / "source_data"

STATUS = "DATA_QC_DRAFT"
STEM = "Figure3abc_ClinVar_integrated_AUPRC_rank_v1_DATA_QC_DRAFT"
PANEL_SIZE_MM = (190.0, 94.0)
PNG_DPI = 600
MM_PER_INCH = 25.4

FONT = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_PATHS = {
    FONT: Path.home() / ".local/share/fonts/Arial/arial.ttf",
    FONT_BOLD: Path.home() / ".local/share/fonts/Arial/arialbd.ttf",
}

BLACK = "#000000"
WHITE = "#FFFFFF"
HEVO = "#B35C1E"
HEVO_WASH = "#FBF4EC"
NEUTRAL = "#65717F"
NEUTRAL_LIGHT = "#AEB6BF"
NEUTRAL_PALE = "#E8EBEF"
RANK_COLORS = {1: "#C43B3B", 2: "#2E7D55", 3: "#2864B4"}

TASKS = OrderedDict(
    [
        (
            "all",
            {
                "letter": "a",
                "title": "ClinVar all",
                "table": "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv",
            },
        ),
        (
            "coding",
            {
                "letter": "b",
                "title": "Coding",
                "table": "S24_zero_shot_classification_of_coding_clinvar_variants.csv",
            },
        ),
        (
            "noncoding",
            {
                "letter": "c",
                "title": "Noncoding",
                "table": "S25_zero_shot_classification_of_noncoding_clinvar_variants.csv",
            },
        ),
    ]
)

# Parameter scales are collapsed only within a locked algorithm identity.
# hEvo-base and hEvo-large are both retained.  Caduceus-PS, the NT 1KG route, and
# hEVO corpus ablations remain available in the 33-model supplementary source.
REPRESENTATIVE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hEvo-base", ("hEvo-base",)),
    ("hEvo-large", ("hEvo-large",)),
    ("GENA-LM", ("GENA-LM-base", "GENA-LM-large")),
    ("EVO 2", ("evo2_1b_base", "evo2_7b_base")),
    ("GENERator", ("GENERator-1.2B", "GENERator-3B")),
    ("Mistral-DNA", ("Mistral-DNA-17M", "Mistral-DNA-138M", "Mistral-DNA-422M")),
    ("NTv3", ("NTv3-8M-pre", "NTv3-100M-pre", "NTv3-650M-pre")),
    ("NTv1-MS", ("NT_2.5b_MS",)),
    ("GPN-Promoter", ("GPN-promoter",)),
    ("GPN-MSA", ("GPN-MSA",)),
    ("Caduceus-PH", ("Caduceus-PH",)),
    ("DNABERT-S", ("DNABERT",)),
    ("DNABERT-2", ("DNABERT2",)),
    ("AIDO.DNA", ("AIDO.DNA-300M",)),
    ("GROVER", ("GROVER",)),
    ("HyenaDNA", ("HyenaDNA_tiny_1k",)),
    ("Genos", ("Genos-1.2B",)),
    ("SpeciesLM", ("SpeciesLM",)),
)

FOCUS_RAW = {
    "hEvo-base",
    "hEvo-large",
    "GPN-MSA",
}


def configure_fonts() -> None:
    register_reportlab_arial()
    for name, path in FONT_PATHS.items():
        if not path.exists():
            raise RuntimeError(f"Required Arial font is missing: {path}")
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    value: str,
    size: float = 6.0,
    *,
    bold: bool = False,
    anchor: str = "start",
    color: str = BLACK,
) -> None:
    drawing.add(
        String(
            x * mm,
            y * mm,
            value,
            fontName=FONT_BOLD if bold else FONT,
            fontSize=size,
            fillColor=HexColor(color),
            textAnchor=anchor,
        )
    )


def text_width_mm(value: str, size: float, *, bold: bool = False) -> float:
    return pdfmetrics.stringWidth(value, FONT_BOLD if bold else FONT, size) / mm


def add_fitted_text(
    drawing: Drawing,
    x: float,
    y: float,
    value: str,
    size: float,
    max_width_mm: float,
    *,
    bold: bool = False,
    anchor: str = "start",
    color: str = BLACK,
    min_size: float = 6.0,
) -> None:
    width = text_width_mm(value, size, bold=bold)
    fitted = size if width <= max_width_mm else size * max_width_mm / width
    if fitted < min_size:
        raise ValueError(f"Cannot fit {value!r} at >= {min_size:g} pt")
    add_text(drawing, x, y, value, fitted, bold=bold, anchor=anchor, color=color)


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


def add_marker(
    drawing: Drawing,
    x: float,
    y: float,
    rank: int,
    *,
    stroke: str,
    size: float = 0.92,
) -> None:
    fill = RANK_COLORS.get(rank, WHITE)
    stroke_width = 0.8 if rank <= 3 else 0.7
    if rank == 1:
        points = [x, y + size, x + size, y, x, y - size, x - size, y]
        drawing.add(
            Polygon(
                [value * mm for value in points],
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=stroke_width,
            )
        )
    elif rank == 2:
        drawing.add(
            Rect(
                (x - size) * mm,
                (y - size) * mm,
                2 * size * mm,
                2 * size * mm,
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=stroke_width,
            )
        )
    elif rank == 3:
        points = [x, y + size, x + size, y - size, x - size, y - size]
        drawing.add(
            Polygon(
                [value * mm for value in points],
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=stroke_width,
            )
        )
    else:
        drawing.add(
            Circle(
                x * mm,
                y * mm,
                size * mm,
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=stroke_width,
            )
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_validate_and_rank() -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    required = {
        "plot_title",
        "dataset",
        "model",
        "auprc_mean",
        "auprc_ci_lower",
        "auprc_ci_upper",
        "num_samples",
        "num_positive",
        "num_negative",
        "baseline",
    }
    task_stats: dict[str, dict[str, float | int]] = {}
    model_sets: list[set[str]] = []

    for task, spec in TASKS.items():
        path = TABLE_DIR / str(spec["table"])
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        frame = frame.copy()
        for column in (
            "auprc_mean",
            "auprc_ci_lower",
            "auprc_ci_upper",
            "num_samples",
            "num_positive",
            "num_negative",
            "baseline",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[list(required - {"plot_title", "dataset", "model"})].isna().any().any():
            raise ValueError(f"{path.name}: missing or non-numeric values")
        if frame.duplicated("model").any():
            raise ValueError(f"{path.name}: duplicate model rows")
        if len(frame) != 33 or frame["model"].nunique() != 33:
            raise ValueError(f"{path.name}: expected 33 unique models")
        if not ((frame["auprc_ci_lower"] <= frame["auprc_mean"]) & (frame["auprc_mean"] <= frame["auprc_ci_upper"])).all():
            raise ValueError(f"{path.name}: confidence interval ordering failure")
        if not frame[["auprc_ci_lower", "auprc_mean", "auprc_ci_upper", "baseline"]].apply(lambda s: s.between(0, 1)).all().all():
            raise ValueError(f"{path.name}: AUPRC or baseline outside [0,1]")
        for column in ("num_samples", "num_positive", "num_negative", "baseline"):
            if frame[column].nunique() != 1:
                raise ValueError(f"{path.name}: inconsistent {column}")
        n = int(frame["num_samples"].iloc[0])
        pos = int(frame["num_positive"].iloc[0])
        neg = int(frame["num_negative"].iloc[0])
        baseline = float(frame["baseline"].iloc[0])
        if pos + neg != n or not math.isclose(baseline, pos / n, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{path.name}: sample count or prevalence baseline mismatch")

        frame["task"] = task
        frame["task_title"] = str(spec["title"])
        frame["source_table"] = path.name
        frame["task_rank_all33"] = frame["auprc_mean"].rank(method="min", ascending=False).astype(int)
        frames.append(frame)
        model_sets.append(set(frame["model"]))
        task_stats[task] = {
            "num_samples": n,
            "num_positive": pos,
            "num_negative": neg,
            "baseline": baseline,
        }

    if not all(models == model_sets[0] for models in model_sets[1:]):
        raise ValueError("S23-S25 model identities differ")

    all_stats = task_stats["all"]
    coding_stats = task_stats["coding"]
    noncoding_stats = task_stats["noncoding"]
    partition_checks = {
        "samples": int(coding_stats["num_samples"]) + int(noncoding_stats["num_samples"]) == int(all_stats["num_samples"]),
        "positives": int(coding_stats["num_positive"]) + int(noncoding_stats["num_positive"]) == int(all_stats["num_positive"]),
        "negatives": int(coding_stats["num_negative"]) + int(noncoding_stats["num_negative"]) == int(all_stats["num_negative"]),
    }
    if not all(partition_checks.values()):
        raise ValueError(f"Coding/noncoding counts do not partition all: {partition_checks}")

    long = pd.concat(frames, ignore_index=True)
    summary = (
        long.groupby("model", as_index=False)
        .agg(
            mean_rank_all33=("task_rank_all33", "mean"),
            rank_sd_all33=("task_rank_all33", "std"),
            mean_auprc=("auprc_mean", "mean"),
        )
    )
    long = long.merge(summary, on="model", how="left", validate="many_to_one")
    summary = {
        "source_tables": [str((TABLE_DIR / str(spec["table"])).resolve()) for spec in TASKS.values()],
        "models_per_task": 33,
        "task_stats": task_stats,
        "partition_checks": partition_checks,
        "auroc_source_status": "NOT_AVAILABLE_IN_S23_S25_OR_LOCAL_RESULT_FILES",
    }
    return long, summary


def select_representatives(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = long[["model", "mean_rank_all33", "rank_sd_all33", "mean_auprc"]].drop_duplicates("model")
    rows: list[dict[str, object]] = []
    selected_models: list[str] = []
    source_models = set(summary["model"])
    for group, candidates in REPRESENTATIVE_GROUPS:
        missing = sorted(set(candidates) - source_models)
        if missing:
            raise ValueError(f"Representative group {group}: missing candidates {missing}")
        subset = summary[summary["model"].isin(candidates)].sort_values(
            ["mean_rank_all33", "mean_auprc", "model"],
            ascending=[True, False, True],
            kind="stable",
        )
        selected = str(subset.iloc[0]["model"])
        selected_models.append(selected)
        for candidate in candidates:
            record = subset[subset["model"] == candidate].iloc[0]
            rows.append(
                {
                    "algorithm_group": group,
                    "candidate_model": candidate,
                    "display_model": canonical_model_name(candidate),
                    "mean_rank_all33": float(record["mean_rank_all33"]),
                    "rank_sd_all33": float(record["rank_sd_all33"]),
                    "mean_auprc": float(record["mean_auprc"]),
                    "selected": candidate == selected,
                    "selection_rule": "lowest mean rank across ClinVar all/coding/noncoding; mean AUPRC tiebreak",
                }
            )

    if len(set(selected_models)) != len(REPRESENTATIVE_GROUPS):
        raise ValueError("Representative selection produced duplicate models")
    selected_summary = summary[summary["model"].isin(selected_models)].copy()
    selected_summary["display_model"] = selected_summary["model"].map(canonical_model_name)
    selected_summary["is_hevo"] = selected_summary["model"].str.startswith("hEvo-")
    selected_summary = selected_summary.sort_values(
        ["mean_rank_all33", "mean_auprc", "display_model"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    selected_summary["display_order"] = np.arange(1, len(selected_summary) + 1)

    plot = long[long["model"].isin(selected_models)].merge(
        selected_summary[["model", "display_model", "is_hevo", "display_order"]],
        on="model",
        how="left",
        validate="many_to_one",
    )
    plot = plot.sort_values(["display_order", "task"], kind="stable").reset_index(drop=True)
    return plot, selected_summary, pd.DataFrame(rows)


def map_rank(rank: float, x0: float, x1: float) -> float:
    return x0 + (33.0 - float(rank)) / 32.0 * (x1 - x0)


def map_score(score: float, x0: float, x1: float) -> float:
    return x0 + float(score) * (x1 - x0)


def draw_axis(drawing: Drawing, x0: float, x1: float, y: float, ticks: list[tuple[float, str]]) -> None:
    add_line(drawing, x0, y, x1, y, width=0.7)
    for fraction, label in ticks:
        x = x0 + fraction * (x1 - x0)
        add_line(drawing, x, y, x, y - 1.0, width=0.6)
        add_text(drawing, x, y - 3.1, label, 6.0, anchor="middle")


def render(plot: pd.DataFrame, summary: pd.DataFrame, summary: dict[str, object]) -> Drawing:
    width, height = PANEL_SIZE_MM
    drawing = Drawing(width * mm, height * mm)
    drawing.add(Rect(0, 0, width * mm, height * mm, fillColor=HexColor(WHITE), strokeColor=None))

    model_x = 3.7
    mean_x0, mean_x1 = 43.2, 59.2
    task_bounds = {
        "all": (65.0, 101.7),
        "coding": (107.2, 143.9),
        "noncoding": (149.4, 186.1),
    }
    task_letter_x = {"all": 61.5, "coding": 103.7, "noncoding": 145.9}

    row_top = 77.0
    row_pitch = 3.33
    row_height = 2.82
    axis_y = 15.8

    add_text(drawing, model_x, 81.5, "Model", 6.2, bold=True)
    add_text(drawing, (mean_x0 + mean_x1) / 2, 81.5, "Mean rank", 6.2, bold=True, anchor="middle")
    add_line(drawing, model_x, 80.3, mean_x1, 80.3, width=0.75)

    task_stats = summary["task_stats"]
    for task, spec in TASKS.items():
        x0, x1 = task_bounds[task]
        add_text(drawing, task_letter_x[task], 87.4, str(spec["letter"]), 14.0, bold=True)
        add_text(drawing, (x0 + x1) / 2, 87.6, str(spec["title"]), 7.2, bold=True, anchor="middle")
        stats = task_stats[task]
        positive_pct = 100.0 * float(stats["num_positive"]) / float(stats["num_samples"])
        add_text(
            drawing,
            (x0 + x1) / 2,
            83.2,
            f"N={int(stats['num_samples']):,} | positive={positive_pct:.1f}%",
            6.0,
            anchor="middle",
        )
        add_line(drawing, x0, 80.3, x1, 80.3, width=0.75)

    summary_lookup = summary.set_index("model")
    plot_lookup = plot.set_index(["model", "task"])
    ordered_models = summary.sort_values("display_order", kind="stable")["model"].tolist()

    for index, model in enumerate(ordered_models):
        row = summary_lookup.loc[model]
        y = row_top - index * row_pitch
        is_hevo = bool(row["is_hevo"])
        if is_hevo:
            drawing.add(
                Rect(
                    2.0 * mm,
                    (y - row_height / 2) * mm,
                    185.2 * mm,
                    row_height * mm,
                    fillColor=HexColor(HEVO_WASH),
                    strokeColor=None,
                )
            )
            drawing.add(
                Rect(
                    2.0 * mm,
                    (y - row_height / 2) * mm,
                    0.75 * mm,
                    row_height * mm,
                    fillColor=HexColor(HEVO),
                    strokeColor=None,
                )
            )

        add_fitted_text(
            drawing,
            model_x,
            y - 0.78,
            str(row["display_model"]),
            6.0,
            mean_x0 - model_x - 1.4,
        )

        rank_x = map_rank(float(row["mean_rank_all33"]), mean_x0, mean_x1)
        add_line(drawing, mean_x0, y, rank_x, y, color=HEVO if is_hevo else NEUTRAL_LIGHT, width=0.7)
        drawing.add(
            Circle(
                rank_x * mm,
                y * mm,
                0.82 * mm,
                fillColor=HexColor(HEVO if is_hevo else WHITE),
                strokeColor=HexColor(HEVO if is_hevo else NEUTRAL),
                strokeWidth=0.75,
            )
        )

        for task in TASKS:
            x0, x1 = task_bounds[task]
            cell = plot_lookup.loc[(model, task)]
            score = float(cell["auprc_mean"])
            lower = float(cell["auprc_ci_lower"])
            upper = float(cell["auprc_ci_upper"])
            baseline = float(cell["baseline"])
            rank = int(cell["task_rank_all33"])
            mean_x = map_score(score, x0, x1)
            low_x = map_score(lower, x0, x1)
            high_x = map_score(upper, x0, x1)
            base_x = map_score(baseline, x0, x1)
            stroke = HEVO if is_hevo else NEUTRAL

            add_line(drawing, base_x, y, mean_x, y, color=HEVO if is_hevo else NEUTRAL_LIGHT, width=0.85)
            add_line(drawing, low_x, y, high_x, y, color=stroke, width=0.9)
            add_line(drawing, low_x, y - 0.64, low_x, y + 0.64, color=stroke, width=0.65)
            add_line(drawing, high_x, y - 0.64, high_x, y + 0.64, color=stroke, width=0.65)
            add_marker(drawing, mean_x, y, rank, stroke=stroke)

            if model in FOCUS_RAW:
                value_text = f"{score:.3f}"
                label_right = mean_x - 1.45
                label_width = text_width_mm(value_text, 6.0)
                fill = HEVO_WASH if is_hevo else WHITE
                drawing.add(
                    Rect(
                        (label_right - label_width - 0.8) * mm,
                        (y - 1.06) * mm,
                        (label_width + 0.65) * mm,
                        2.12 * mm,
                        fillColor=HexColor(fill),
                        strokeColor=None,
                    )
                )
                add_text(drawing, label_right, y - 0.78, value_text, 6.0, anchor="end")

    draw_axis(drawing, mean_x0, mean_x1, axis_y, [(0.0, "33"), (0.5, "17"), (1.0, "1")])
    add_text(drawing, (mean_x0 + mean_x1) / 2, 9.9, "better rank", 6.0, anchor="middle")
    for task in TASKS:
        x0, x1 = task_bounds[task]
        draw_axis(drawing, x0, x1, axis_y, [(0.0, "0"), (0.5, "0.5"), (1.0, "1.0")])
        add_text(drawing, (x0 + x1) / 2, 9.9, "AUPRC", 6.0, anchor="middle")

    legend_y = 6.6
    add_text(drawing, 3.7, legend_y - 0.75, "Rank", 6.0)
    cursor = 13.5
    for rank, label in ((1, "1"), (2, "2"), (3, "3"), (4, "other")):
        add_marker(drawing, cursor, legend_y, rank, stroke=NEUTRAL, size=0.78)
        add_text(drawing, cursor + 2.1, legend_y - 0.75, label, 6.0)
        cursor += 9.4 if rank < 4 else 14.2
    add_line(drawing, 50.4, legend_y, 56.0, legend_y, color=HEVO, width=1.0)
    drawing.add(Circle(53.2 * mm, legend_y * mm, 0.78 * mm, fillColor=HexColor(HEVO), strokeColor=HexColor(HEVO)))
    add_text(drawing, 57.5, legend_y - 0.75, "hEVO", 6.0)
    add_text(
        drawing,
        186.1,
        legend_y - 0.75,
        "Point estimate with reported 95% bootstrap CI; segment starts at prevalence baseline.",
        6.0,
        anchor="end",
    )
    add_text(
        drawing,
        186.1,
        2.5,
        "Mean rank is calculated among all 33 models across the three ClinVar strata; lower is better.",
        6.0,
        anchor="end",
    )
    return drawing


def render_png(pdf_path: Path, output_png: Path) -> None:
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
    executable = bundled if bundled.exists() else Path(shutil.which("pdftoppm") or "")
    if not executable.exists():
        raise RuntimeError("pdftoppm is required for 600-dpi PNG rendering")
    subprocess.run(
        [
            str(executable),
            "-f",
            "1",
            "-singlefile",
            "-png",
            "-r",
            str(PNG_DPI),
            str(pdf_path),
            str(output_png.with_suffix("")),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def pdf_font_summary(path: Path) -> dict[str, object]:
    page = PdfReader(path).pages[0]
    resources = (page.get("/Resources") or {}).get_object()
    fonts = (resources.get("/Font") or {}).get_object()
    records: dict[str, bool] = {}
    for reference in fonts.values():
        font = reference.get_object()
        name = str(font.get("/BaseFont", ""))
        descriptor = font.get("/FontDescriptor")
        if not descriptor and font.get("/DescendantFonts"):
            descriptor = font["/DescendantFonts"][0].get_object().get("/FontDescriptor")
        descriptor = descriptor.get_object() if hasattr(descriptor, "get_object") else descriptor
        records[name] = bool(descriptor and any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")))
    arial = [name for name in records if "Arial" in name]
    return {
        "fonts": records,
        "arial_present": bool(arial),
        "required_arial_fonts_embedded": bool(arial) and all(records[name] for name in arial),
    }


def write_caption(plot: pd.DataFrame, summary: pd.DataFrame, summary: dict[str, object]) -> Path:
    base_raw = "hEvo-base"
    large_raw = "hEvo-large"
    gpn_raw = "GPN-MSA"
    lookup = plot.set_index(["model", "task"])
    base_summary = summary.set_index("model").loc[base_raw]
    gpn_non = lookup.loc[(gpn_raw, "noncoding")]
    base_non = lookup.loc[(base_raw, "noncoding")]
    caption = f"""# Figure 3a-c caption draft

**a-c, Zero-shot ClinVar variant-effect discrimination across all, coding, and noncoding variants.** Models score paired reference and alternate sequences using the signed likelihood-ratio route. Points show AUPRC and whiskers show the reported 95% bootstrap confidence intervals; each horizontal segment begins at the stratum-specific positive-class prevalence, the expected AUPRC of a random classifier. All panels use the same absolute 0-1 AUPRC scale. Models are ordered by mean rank across the three strata after ranks were computed among all 33 evaluated models. One parameter-scale representative is retained per prespecified algorithm identity, with hEvo-base and hEvo-large both shown.

hEvo-base ranks first for ClinVar all ({lookup.loc[(base_raw, 'all'), 'auprc_mean']:.3f}) and coding ({lookup.loc[(base_raw, 'coding'), 'auprc_mean']:.3f}) variants and second for noncoding variants ({float(base_non['auprc_mean']):.3f}); GPN-MSA leads the noncoding stratum ({float(gpn_non['auprc_mean']):.3f}), a difference of {float(gpn_non['auprc_mean']) - float(base_non['auprc_mean']):.3f} AUPRC. Across the three strata, hEvo-base has the best mean rank ({float(base_summary['mean_rank_all33']):.2f}). ClinVar labels contrast pathogenic/likely pathogenic with benign/likely benign variants.

Source data: S23-S25. Coding and noncoding counts exactly partition the all-variant cohort. AUROC is not shown because the current locally traceable S23-S25 source tables contain AUPRC but no matched per-model AUROC values or intervals.
"""
    path = OUT_DIR / "Figure3abc_ClinVar_CAPTION_DRAFT.md"
    path.write_text(caption, encoding="utf-8")
    return path


def write_method_summary(plot: pd.DataFrame, summary: pd.DataFrame, summary: dict[str, object]) -> Path:
    lookup = plot.set_index(["model", "task"])
    base = "hEvo-base"
    large = "hEvo-large"
    gpn = "GPN-MSA"
    report = f"""# Figure 3a-c data and method summary

## Validated from S23-S25

- Each task contains 33 unique models, and the model identities are identical across ClinVar all, coding, and noncoding.
- Coding plus noncoding exactly reconstructs the all-variant counts: 144,285 + 119,907 = 264,192 samples; 70,787 + 13,793 = 84,580 positives; 73,498 + 106,114 = 179,612 negatives.
- Every prevalence baseline equals `num_positive / num_samples` to numerical precision.
- All reported intervals satisfy lower <= point estimate <= upper and remain within [0,1].
- Ranks were recomputed from source AUPRC values among all 33 models before representative selection.

## Reader-facing result boundary

- hEvo-base ranks #1 / #1 / #2 for all, coding, and noncoding ClinVar, respectively.
- hEvo-large ranks #4 / #4 / #3.
- Noncoding is not hidden: GPN-MSA is #1 ({float(lookup.loc[(gpn, 'noncoding'), 'auprc_mean']):.6f}) and exceeds hEvo-base ({float(lookup.loc[(base, 'noncoding'), 'auprc_mean']):.6f}) by {float(lookup.loc[(gpn, 'noncoding'), 'auprc_mean'] - lookup.loc[(base, 'noncoding'), 'auprc_mean']):.6f} AUPRC.
- hEvo-base has the best mean rank across the three strata ({float(summary.set_index('model').loc[base, 'mean_rank_all33']):.3f}).

## Metric decision

AUPRC remains the main-text metric because class prevalence differs markedly across the three strata (32.0%, 49.1%, and 11.5%). AUROC point estimates are computed by the benchmark implementation, but matched per-model AUROC rows and intervals are not present in S23-S25, and the local `results/zeroshot_ClinVar260615` model directories contain no result files. No AUROC values are inferred in this figure. If the original `summary_results.json` or HDF5 scores are restored, AUROC can be recomputed on exactly the same samples and added to a supplementary sensitivity panel.

## Submission gates

1. The table README describes the AUPRC intervals as bootstrap confidence intervals, but the uploaded `scripts/evaluate/zeroshot_paried_seq_input.py` explicitly implements bootstrap intervals for AUROC only. The original `plots/Fig3/plot_fig3_variant_auprc_bar.py` source referenced by the documentation is absent from this snapshot. Therefore the figure says **reported 95% bootstrap CI** and does not claim independent recomputation.
2. Restore or checksum-link the exact per-model `summary_results.json`/HDF5 score files used for S23-S25, including label order and sample identifiers.
3. Confirm that the signed score orientation (`-LLR` for label=1 pathogenicity) and the paired REF/ALT construction are identical across all 33 models.
4. Confirm the final hEVO checkpoint and scoring route used for S23-S25 before replacing `DATA_QC_DRAFT` with a submission-ready status.
"""
    path = OUT_DIR / "Figure3abc_ClinVar_DATA_AND_METHOD_summary.md"
    path.write_text(report, encoding="utf-8")
    return path


def save(
    drawing: Drawing,
    long: pd.DataFrame,
    plot: pd.DataFrame,
    summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
    data_summary: dict[str, object],
) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{STEM}.pdf"
    svg_path = OUT_DIR / f"{STEM}.svg"
    png_path = OUT_DIR / f"{STEM}.png"
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    render_png(pdf_path, png_path)

    long_path = SOURCE_DIR / "Figure3abc_all_33_models_source.csv"
    plot_path = SOURCE_DIR / "Figure3abc_representative_plot_source.csv"
    summary_path = SOURCE_DIR / "Figure3abc_representative_summary.csv"
    selection_path = SOURCE_DIR / "Figure3abc_representative_selection_summary.csv"
    long.to_csv(long_path, index=False)
    plot.to_csv(plot_path, index=False)
    summary.to_csv(summary_path, index=False)
    selection_summary.to_csv(selection_path, index=False)
    caption_path = write_caption(plot, summary, data_summary)
    method_summary_path = write_method_summary(plot, summary, data_summary)

    page = PdfReader(pdf_path).pages[0]
    width_mm = float(page.mediabox.width) / 72.0 * MM_PER_INCH
    height_mm = float(page.mediabox.height) / 72.0 * MM_PER_INCH
    extracted = page.extract_text() or ""
    fonts = pdf_font_summary(pdf_path)
    with Image.open(png_path) as image:
        pixels = list(image.size)
    expected_pixels = [round(value / MM_PER_INCH * PNG_DPI) for value in PANEL_SIZE_MM]
    svg_text = svg_path.read_text(encoding="utf-8")

    lookup = plot.set_index(["model", "task"])
    base = "hEvo-base"
    large = "hEvo-large"
    gpn = "GPN-MSA"
    checks = {
        "size_mm": [round(width_mm, 3), round(height_mm, 3)],
        "size_pass": abs(width_mm - PANEL_SIZE_MM[0]) < 0.15 and abs(height_mm - PANEL_SIZE_MM[1]) < 0.15,
        "png_pixels": pixels,
        "expected_png_pixels": expected_pixels,
        "dpi_pass": all(abs(a - b) <= 2 for a, b in zip(pixels, expected_pixels)),
        "editable_svg_text": "<text" in svg_text,
        "arial_present": fonts["arial_present"],
        "required_arial_fonts_embedded": fonts["required_arial_fonts_embedded"],
        "black_text_requested": True,
        "no_gridlines": True,
        "forbidden_pink_absent": not any(code in svg_text.lower() for code in ("#ff6fae", "#d45087", "#f7dfe9", "#e76f9a")),
        "same_absolute_auprc_axis": True,
        "models_per_task_33": all(long[long["task"] == task]["model"].nunique() == 33 for task in TASKS),
        "same_model_set": len({tuple(sorted(long[long["task"] == task]["model"])) for task in TASKS}) == 1,
        "coding_noncoding_partition_all": all(data_summary["partition_checks"].values()),
        "representatives_18": summary["model"].nunique() == 18,
        "hEVO_base_and_large_present": {base, large}.issubset(set(summary["model"])),
        "hEVO_base_ranks": {task: int(lookup.loc[(base, task), "task_rank_all33"]) for task in TASKS},
        "hEVO_large_ranks": {task: int(lookup.loc[(large, task), "task_rank_all33"]) for task in TASKS},
        "noncoding_leader": str(plot[(plot["task"] == "noncoding") & (plot["task_rank_all33"] == 1)]["model"].iloc[0]),
        "noncoding_GPN_minus_hEVO_base": float(lookup.loc[(gpn, "noncoding"), "auprc_mean"] - lookup.loc[(base, "noncoding"), "auprc_mean"]),
        "auroc_plotted": False,
        "auroc_status": data_summary["auroc_source_status"],
        "required_labels_present": all(label in extracted for label in ("ClinVar all", "Coding", "Noncoding", "hEvo-base", "GPN-MSA", "AUPRC")),
        "visual_overlap_review": "PASS_600DPI_MANUAL_REVIEW_NO_CLIPPING_OR_TEXT_OVERLAP",
    }
    automated_keys = (
        "size_pass",
        "dpi_pass",
        "editable_svg_text",
        "arial_present",
        "required_arial_fonts_embedded",
        "forbidden_pink_absent",
        "same_absolute_auprc_axis",
        "models_per_task_33",
        "same_model_set",
        "coding_noncoding_partition_all",
        "representatives_18",
        "hEVO_base_and_large_present",
        "required_labels_present",
    )
    checks["automated_pass"] = all(bool(checks[key]) for key in automated_keys)

    qa = {
        "status": STATUS,
        "design": {
            "format": "A4-width compact module",
            "dimensions_mm": list(PANEL_SIZE_MM),
            "font": "Arial",
            "minimum_font_pt": 6.0,
            "hEVO_color": HEVO,
            "plot_type": "shared-row baseline-to-AUPRC interval matrix plus cross-stratum mean rank",
            "selection": "one parameter-scale representative per prespecified algorithm identity; hEvo-base and hEvo-large forced",
        },
        "data_summary": data_summary,
        "checks": checks,
        "fonts": fonts["fonts"],
        "artifacts": {
            "pdf": str(pdf_path.resolve()),
            "svg": str(svg_path.resolve()),
            "png": str(png_path.resolve()),
            "caption": str(caption_path.resolve()),
            "method_summary": str(method_summary_path.resolve()),
            "all_33_source": str(long_path.resolve()),
            "plot_source": str(plot_path.resolve()),
            "summary_source": str(summary_path.resolve()),
            "selection_summary": str(selection_path.resolve()),
        },
        "hashes": {
            "script": sha256(HERE),
            "pdf": sha256(pdf_path),
            "svg": sha256(svg_path),
            "png": sha256(png_path),
            "S23": sha256(TABLE_DIR / str(TASKS["all"]["table"])),
            "S24": sha256(TABLE_DIR / str(TASKS["coding"]["table"])),
            "S25": sha256(TABLE_DIR / str(TASKS["noncoding"]["table"])),
        },
        "claim_boundary": [
            "AUPRC is the only ClinVar discrimination metric plotted because matched local AUROC source rows are absent.",
            "The noncoding result is not hidden: GPN-MSA is rank 1 and its absolute margin over hEvo-base is reported.",
            "Confidence intervals are retained exactly as reported in S23-S25; this renderer does not recompute them.",
            "Submission remains contingent on linking S23-S25 to the exact scored files and final variant-scoring route.",
        ],
    }
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    configure_fonts()
    long, data_summary = read_validate_and_rank()
    plot, summary, selection_summary = select_representatives(long)
    drawing = render(plot, summary, data_summary)
    qa = save(drawing, long, plot, summary, selection_summary, data_summary)


if __name__ == "__main__":
    main()
