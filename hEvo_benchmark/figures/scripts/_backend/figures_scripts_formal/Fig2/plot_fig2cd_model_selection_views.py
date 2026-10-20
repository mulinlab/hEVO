#!/usr/bin/env python3
"""Render source-backed Figure 2c/d benchmark comparison views.

Outputs per benchmark:
  1. Main-text panel: one independently selected parameter-scale winner per
     prespecified algorithm identity.
  2. Supplementary all-model panel: all models in the canonical source table.
  3. Supplementary family panel: hEVO, NT, Evo 2 and GPN series.

The task cells show the reported MCC mean and 95% CI on a common 0-1 scale.
Task ranks are recomputed from the displayed source values so that any source
rank discrepancy is visible in the summary rather than silently inherited.
"""

from __future__ import annotations

# [Panels c/d] Plot benchmark model-selection views: c shows the NT benchmark and d the genomic benchmark.
# [Input] Benchmark source data such as S21/S22; validate tasks, models, and intervals before building the dot/interval matrix.
# [Output] Panel Drawing, standalone PDF/SVG/PNG, plot source, selection summary, and QA; the A4 integration layer reuses the Drawing and data functions.

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from figures.scripts.common.hevo_model_style import (  # noqa: E402
    BLACK,
    MODEL_FAMILY_STYLE,
    canonical_model_name,
    model_family,
    model_registry_record,
)
import plot_fig2a_split_benchmark_routes as base  # noqa: E402


TABLE_DIR = PROJECT_ROOT / "tables"
OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig2" / "fig2cd_model_selection_v2"
SOURCE_DIR = OUT_DIR / "source_data"
STATUS = "DATA_QC_DRAFT"
PNG_DPI = 600
MM_PER_INCH = 25.4

WHITE = "#FFFFFF"
NEUTRAL = "#65717F"
NEUTRAL_LIGHT = "#D8DDE3"
NEUTRAL_PALE = "#F5F6F7"
HEVO_WASH = "#FBF2EA"
GROUP_LINE = "#AEB6BF"
RANK_COLORS = {1: "#C43B3B", 2: "#2E7D55", 3: "#2864B4"}


GB_TASKS = (
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
)
GB_SHARED_TASKS = GB_TASKS[:5]
GB_OPTIONAL_DEMO_TASKS = GB_TASKS[5:]
NT_HISTONE = (
    "H2AFZ",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
    "H4K20me1",
)
NT_REGULATORY = (
    "enhancers",
    "enhancers_types",
    "promoter_all",
    "promoter_no_tata",
    "promoter_tata",
)
NT_SPLICE = (
    "splice_sites_acceptors",
    "splice_sites_all",
    "splice_sites_donors",
)
NT_TASKS = NT_HISTONE + NT_REGULATORY + NT_SPLICE


@dataclass(frozen=True)
class BenchmarkConfig:
    key: str
    letter: str
    context: str
    table_file: str
    tasks: tuple[str, ...]
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    aliases: dict[str, str]


CONFIGS = {
    "gb": BenchmarkConfig(
        key="gb",
        letter="c",
        context="Genomic Benchmark | 7 tasks",
        table_file="S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv",
        tasks=GB_TASKS,
        groups=(("Genomic Benchmark", GB_TASKS),),
        aliases={
            "human_enhancers_cohn": "Human\nenhancers\n(Cohn)",
            "human_enhancers_ensembl": "Human\nenhancers\n(Ensembl)",
            "human_ensembl_regulatory": "Human\nregulatory\n(Ensembl)",
            "human_nontata_promoters": "Human\nnon-TATA\npromoters",
            "human_ocr_ensembl": "Open chromatin\n(Ensembl)",
            "demo_coding_vs_intergenomic_seqs": "Coding versus\nintergenomic",
            "demo_human_or_worm": "Human versus\nworm",
        },
    ),
    "nt": BenchmarkConfig(
        key="nt",
        letter="d",
        context="NT Benchmark | 18 tasks",
        table_file="S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv",
        tasks=NT_TASKS,
        groups=(
            ("Histone marks", NT_HISTONE),
            ("Regulatory elements", NT_REGULATORY),
            ("Splice sites", NT_SPLICE),
        ),
        aliases={
            "H2AFZ": "H2AFZ",
            "H3K27ac": "H3K27ac",
            "H3K27me3": "H3K27me3",
            "H3K36me3": "H3K36me3",
            "H3K4me1": "H3K4me1",
            "H3K4me2": "H3K4me2",
            "H3K4me3": "H3K4me3",
            "H3K9ac": "H3K9ac",
            "H3K9me3": "H3K9me3",
            "H4K20me1": "H4K20me1",
            "enhancers": "Enhancer",
            "enhancers_types": "Enhancer types",
            "promoter_all": "Promoter all",
            "promoter_no_tata": "Promoter non-TATA",
            "promoter_tata": "Promoter TATA",
            "splice_sites_acceptors": "Splice acceptor",
            "splice_sites_all": "Splice site all",
            "splice_sites_donors": "Splice donor",
        },
    ),
}


# A parameter-scale family is collapsed only when the architecture/data route
# is otherwise the same. NT-v1, NT-v2 and NT-v3 are distinct generations.
# The compact c/d panels retain NTv1-MS-2.5B and omit NTv2-MS-500M by the
# locked manuscript selection; complete supplementary views retain both.
# GPN-MSA is retained as its own representation alongside GPN-Promoter.
# The main-figure representative roster follows the supplied Figure 2 reference.
MAIN_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hEVO", ("hEvo-base", "hEvo-large")),
    ("GENA-LM", ("GENA-LM-base", "GENA-LM-large")),
    ("Evo 2", ("evo2_1b_base", "evo2_7b_base")),
    ("GENERator", ("GENERator-1.2B", "GENERator-3B")),
    ("Mistral-DNA", ("Mistral-DNA-17M", "Mistral-DNA-138M", "Mistral-DNA-422M")),
    ("NT v3", ("NTv3-8M-pre", "NTv3-100M-pre", "NTv3-650M-pre")),
    ("NTv1-MS", ("NT_2.5b_1kG",)),
    ("GPN-Promoter", ("GPN-promoter",)),
    ("GPN-MSA", ("GPN-MSA",)),
    ("GPN-Star", ("GPN-Star-M",)),
    ("Caduceus-PS", ("Caduceus-PS",)),
    ("DNABERT", ("DNABERT",)),
    ("DNABERT-2", ("DNABERT2",)),
    ("AIDO.DNA", ("AIDO.DNA-300M",)),
    ("GROVER", ("GROVER",)),
    ("HyenaDNA", ("HyenaDNA_tiny_1k",)),
    ("Genos", ("Genos-1.2B",)),
    ("SpeciesLM", ("SpeciesLM",)),
)

FAMILY_RAW = {
    "hEVO": ("hEvo-base", "hEvo-large"),
    "NT": (
        "NTv3-8M-pre",
        "NTv3-100M-pre",
        "NTv3-650M-pre",
        "NT_500m_MS",
        "NT_2.5b_MS",
        "NT_500m_1kG",
        "NT_2.5b_1kG",
    ),
    "Evo 2": ("evo2_1b_base", "evo2_7b_base"),
    "GPN": ("GPN-promoter",),
}

def display_name(raw: str) -> str:
    return canonical_model_name(raw)


def family_color(family: str) -> str:
    return MODEL_FAMILY_STYLE[family]["color"]


def rank_shape(rank: float) -> str:
    value = int(round(float(rank)))
    return {1: "diamond", 2: "square", 3: "triangle"}.get(value, "circle")


def read_and_validate(
    config: BenchmarkConfig,
    *,
    allow_incomplete: bool = False,
    display_exclusions: tuple[tuple[str, str], ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    path = TABLE_DIR / config.table_file
    frame = pd.read_csv(path)
    required = {
        "benchmark",
        "task_group",
        "model",
        "dataset",
        "mcc_mean",
        "mcc_lower",
        "mcc_upper",
        "task_rank",
        "task_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    frame = frame.copy()
    for column in ("mcc_mean", "mcc_lower", "mcc_upper", "task_rank", "task_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(required - {"benchmark", "task_group", "model", "dataset"})].isna().any().any():
        raise ValueError(f"{path.name}: missing or non-numeric benchmark values")
    if frame.duplicated(["model", "dataset"]).any():
        raise ValueError(f"{path.name}: duplicate model-task cells")
    source_tasks = tuple(frame["dataset"].drop_duplicates())
    if set(source_tasks) != set(config.tasks):
        raise ValueError(f"{path.name}: configured and source tasks differ")
    frame = frame[frame["dataset"].isin(config.tasks)].copy()
    source_frame = frame.copy()
    source_rows = len(source_frame)
    source_model_count = int(source_frame["model"].nunique())
    source_observed_cells = set(zip(source_frame["model"].astype(str), source_frame["dataset"].astype(str)))
    source_task_counts = source_frame.groupby("dataset")["model"].nunique()
    source_reported_counts = source_frame.groupby("dataset")["task_count"].first().astype(int)
    if not source_reported_counts.equals(source_task_counts.astype(int)):
        raise ValueError(f"{path.name}: source task_count does not match its source rows")
    source_rank_reported = source_frame["task_rank"].astype(int)
    source_rank_recomputed = source_frame.groupby("dataset")["mcc_mean"].rank(
        method="min", ascending=False
    ).astype(int)
    source_rank_mismatch_count = int((source_rank_reported != source_rank_recomputed).sum())

    requested_exclusions = set(display_exclusions)
    if requested_exclusions and config.key != "gb":
        raise ValueError("display_exclusions are supported only for the Genomic Benchmark")
    missing_exclusions = requested_exclusions - source_observed_cells
    if missing_exclusions:
        raise ValueError(f"{path.name}: requested display exclusions are absent: {sorted(missing_exclusions)}")
    applied_exclusions = sorted(requested_exclusions)
    if applied_exclusions:
        excluded = pd.Series(
            [(str(model), str(task)) in requested_exclusions for model, task in zip(frame["model"], frame["dataset"])],
            index=frame.index,
        )
        frame = frame.loc[~excluded].copy()
        frame["task_count"] = frame.groupby("dataset")["model"].transform("nunique").astype(int)
        frame["task_rank"] = frame.groupby("dataset")["mcc_mean"].rank(
            method="min", ascending=False
        ).astype(int)

    model_count = int(frame["model"].nunique())
    if model_count <= 0:
        raise ValueError(f"{path.name}: no models found")
    expected = model_count * len(config.tasks)
    observed_by_task = frame.groupby("dataset")["model"].nunique()
    reported_by_task = frame.groupby("dataset")["task_count"].nunique()
    if not reported_by_task.eq(1).all():
        raise ValueError(f"{path.name}: task_count is not consistent within each task")
    reported_counts = frame.groupby("dataset")["task_count"].first().astype(int)
    if not reported_counts.equals(observed_by_task.astype(int)):
        raise ValueError(f"{path.name}: task_count does not match the source rows by task")

    task_counts_by_model = frame.groupby("model")["dataset"].nunique()
    incomplete_models = {
        str(model): [task for task in config.tasks if not ((frame["model"] == model) & (frame["dataset"] == task)).any()]
        for model in frame["model"].unique()
        if int(task_counts_by_model[model]) != len(config.tasks)
    }
    missing_cells = expected - len(frame)
    if missing_cells and not allow_incomplete:
        raise ValueError(f"{path.name}: expected {expected} cells, found {len(frame)}")
    if allow_incomplete and config.key == "gb":
        # Representative candidates must have the shared five human tasks.
        # Models without source records for the optional demo tasks remain
        # incomplete in those cells; recorded values are retained as supplied.
        main_candidates = {model for _, candidates in MAIN_GROUPS for model in candidates}
        candidate_tasks = GB_SHARED_TASKS
        incomplete_candidates = {
            model: [task for task in candidate_tasks if not ((frame["model"] == model) & (frame["dataset"] == task)).any()]
            for model in sorted(main_candidates & set(frame["model"].astype(str)))
            if any(not ((frame["model"] == model) & (frame["dataset"] == task)).any() for task in candidate_tasks)
        }
        if incomplete_candidates:
            raise ValueError(
                f"{path.name}: representative-selection candidates have incomplete tasks: {incomplete_candidates}"
            )
    if not ((frame["mcc_lower"] <= frame["mcc_mean"]) & (frame["mcc_mean"] <= frame["mcc_upper"])).all():
        raise ValueError(f"{path.name}: confidence interval ordering violation")
    if not frame[["mcc_lower", "mcc_mean", "mcc_upper"]].apply(lambda s: s.between(-1, 1)).all().all():
        raise ValueError(f"{path.name}: MCC or confidence bound outside [-1, 1]")

    frame["task_rank_reported"] = frame["task_rank"].astype(int)
    frame["task_rank_recomputed"] = (
        frame.groupby("dataset")["mcc_mean"].rank(method="min", ascending=False).astype(int)
    )
    frame["task_rank_match"] = frame["task_rank_reported"] == frame["task_rank_recomputed"]
    frame["rank_marker"] = frame["task_rank_recomputed"].map(rank_shape)
    frame["task_order"] = frame["dataset"].map({task: i for i, task in enumerate(config.tasks)})
    frame["task_alias"] = frame["dataset"].map(config.aliases)
    frame["model_display"] = frame["model"].map(display_name)
    frame["family"] = frame["model"].map(model_family)

    model_summary = frame.groupby(["model", "model_display", "family"], as_index=False).agg(
        mean_mcc=("mcc_mean", "mean"),
        mean_task_rank=("task_rank_recomputed", "mean"),
    )
    model_summary = model_summary.sort_values(["mean_mcc", "model_display"], ascending=[False, True], kind="stable")
    model_summary["all_model_rank"] = np.arange(1, len(model_summary) + 1)

    mismatches = frame.loc[
        ~frame["task_rank_match"],
        ["dataset", "model", "mcc_mean", "task_rank_reported", "task_rank_recomputed"],
    ].to_dict("records")
    data_summary = {
        "source_path": str(path.resolve()),
        "source_sha256": base.sha256(path),
        "source_rows": int(source_rows),
        "rows": int(len(frame)),
        "models": int(frame["model"].nunique()),
        "tasks": int(frame["dataset"].nunique()),
        "duplicate_cells": 0,
        "expected_cells": int(expected),
        "missing_cells": int(missing_cells),
        "source_missing_cells": int(source_model_count * len(config.tasks) - source_rows),
        "display_excluded_cells": [
            {"model": model, "dataset": task} for model, task in applied_exclusions
        ],
        "source_rank_mismatch_count": source_rank_mismatch_count,
        "incomplete_models": incomplete_models,
        "task_model_counts": {str(task): int(count) for task, count in observed_by_task.items()},
        "source_task_model_counts": {str(task): int(count) for task, count in source_task_counts.items()},
        "ci_order_pass": True,
        "mcc_bounds_pass": True,
        "mcc_mean_range": [float(frame["mcc_mean"].min()), float(frame["mcc_mean"].max())],
        "rank_mismatch_count": int((~frame["task_rank_match"]).sum()),
        "rank_mismatches": mismatches,
        "rank_glyph_source": "task_rank_recomputed",
        "selection_rule": (
            "all models with complete shared five-human-task coverage; GPN-MSA demo cells are suppressed, "
            "and other demo scores remain visible when available"
            if config.key == "gb"
            else f"all models with complete {len(config.tasks)}-task coverage"
        ),
    }
    return frame.sort_values(["model", "task_order"]), model_summary.reset_index(drop=True), data_summary


def select_main(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = set(summary["model"])
    rows: list[dict[str, object]] = []
    selected_models: list[str] = []
    score = summary.set_index("model")["mean_mcc"].to_dict()
    rank = summary.set_index("model")["all_model_rank"].to_dict()
    for algorithm, candidates in MAIN_GROUPS:
        missing = sorted(set(candidates) - available)
        if missing:
            raise ValueError(f"Representative group {algorithm} missing candidates: {missing}")
        winner = sorted(candidates, key=lambda model: (-float(score[model]), model))[0]
        selected_for_group = list(candidates) if algorithm == "hEVO" else [winner]
        selected_models.extend(selected_for_group)
        for candidate in candidates:
            rows.append(
                {
                    "algorithm_identity": algorithm,
                    "candidate_model": candidate,
                    "candidate_display": display_name(candidate),
                    "mean_mcc": float(score[candidate]),
                    "all_model_rank": int(rank[candidate]),
                    "selected": candidate in selected_for_group,
                    "selection_rule": (
                        "retain both focal hEVO scales"
                        if algorithm == "hEVO"
                        else "highest unweighted mean MCC across displayed benchmark tasks"
                    ),
                }
            )
    selected = summary[summary["model"].isin(selected_models)].copy()
    selected = selected.sort_values(["mean_mcc", "model_display"], ascending=[False, True], kind="stable")
    selected["plot_rank"] = np.arange(1, len(selected) + 1)
    summary = pd.DataFrame(rows).sort_values(["algorithm_identity", "mean_mcc"], ascending=[True, False])
    return selected.reset_index(drop=True), summary.reset_index(drop=True)


def prepare_all(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy().sort_values(["mean_mcc", "model_display"], ascending=[False, True])
    result["plot_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def prepare_families(summary: pd.DataFrame) -> pd.DataFrame:
    available = set(summary["model"])
    pieces: list[pd.DataFrame] = []
    for family_index, (family, models) in enumerate(FAMILY_RAW.items()):
        missing = sorted(set(models) - available)
        if missing:
            raise ValueError(f"Family {family} missing models: {missing}")
        piece = summary[summary["model"].isin(models)].copy()
        piece = piece.sort_values(["mean_mcc", "model_display"], ascending=[False, True])
        piece["family_block"] = family
        piece["family_block_order"] = family_index
        piece["family_row_order"] = np.arange(len(piece))
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)
    result["plot_rank"] = np.arange(1, len(result) + 1)
    return result


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    value: str,
    size: float = 6.0,
    *,
    bold: bool = False,
    anchor: str = "start",
    fill: str = BLACK,
) -> None:
    base.add_text(drawing, x, y, value, size, bold=bold, anchor=anchor, color=fill)


def add_rotated_text(drawing: Drawing, x: float, y: float, value: str, size: float = 6.0) -> None:
    radians = math.radians(55.0)
    label = String(
        0,
        0,
        value,
        fontName=base.FONT,
        fontSize=size,
        fillColor=HexColor(BLACK),
        textAnchor="start",
    )
    group = Group(label)
    group.transform = (
        math.cos(radians),
        math.sin(radians),
        -math.sin(radians),
        math.cos(radians),
        x * mm,
        y * mm,
    )
    drawing.add(group)


def add_multiline_center(drawing: Drawing, x: float, y: float, value: str, size: float = 6.0) -> None:
    lines = value.split("\n")
    spacing = 2.05
    start = y + spacing * (len(lines) - 1) / 2
    for index, line in enumerate(lines):
        add_text(drawing, x, start - index * spacing, line, size, anchor="middle")


def add_marker(drawing: Drawing, x: float, y: float, shape: str, fill: str, stroke: str) -> None:
    if shape == "diamond":
        radius = 0.70
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
                strokeWidth=0.65,
            )
        )
    elif shape == "square":
        half = 0.57
        drawing.add(
            Rect(
                (x - half) * mm,
                (y - half) * mm,
                2 * half * mm,
                2 * half * mm,
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=0.65,
            )
        )
    elif shape == "triangle":
        radius = 0.72
        drawing.add(
            Polygon(
                [
                    x * mm,
                    (y + radius) * mm,
                    (x + 0.80 * radius) * mm,
                    (y - 0.60 * radius) * mm,
                    (x - 0.80 * radius) * mm,
                    (y - 0.60 * radius) * mm,
                ],
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=0.65,
            )
        )
    else:
        drawing.add(
            Circle(
                x * mm,
                y * mm,
                0.43 * mm,
                fillColor=HexColor(fill),
                strokeColor=HexColor(stroke),
                strokeWidth=0.62,
            )
        )


def draw_family_legend(drawing: Drawing, right: float, y: float, *, focal_only: bool) -> None:
    labels = ("hEVO", "Other models") if focal_only else ("hEVO", "Evo 2", "NT", "GPN", "Other")
    widths = {label: base.text_width_mm(label, 6.0) + 6.0 for label in labels}
    cursor = right - sum(widths.values())
    for label in labels:
        shade = family_color("hEVO") if label == "hEVO" else NEUTRAL if focal_only else family_color(label)
        drawing.add(Circle((cursor + 1.0) * mm, (y + 0.15) * mm, 0.72 * mm, fillColor=HexColor(shade), strokeColor=None))
        add_text(drawing, cursor + 2.5, y - 0.55, label, 6.0)
        cursor += widths[label]


def row_positions(rows: pd.DataFrame, top: float, bottom: float, family_mode: bool) -> dict[str, float]:
    if not family_mode:
        if len(rows) == 1:
            return {str(rows.iloc[0]["model"]): (top + bottom) / 2}
        step = (top - bottom) / (len(rows) - 1)
        return {str(row.model): top - index * step for index, row in rows.iterrows()}
    gaps = max(int(rows["family_block"].nunique()) - 1, 0)
    units = max(len(rows) - 1, 1) + gaps * 0.75
    step = (top - bottom) / units
    positions: dict[str, float] = {}
    cursor = top
    previous = None
    for row in rows.itertuples(index=False):
        if previous is not None:
            cursor -= step
            if row.family_block != previous:
                cursor -= step * 0.75
        positions[str(row.model)] = cursor
        previous = row.family_block
    return positions


def draw_view(
    config: BenchmarkConfig,
    cells: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    view: str,
    letter: str | None,
    height_mm: float,
) -> Drawing:
    width_mm = 190.0
    drawing = Drawing(width_mm * mm, height_mm * mm)
    drawing.add(Rect(0, 0, drawing.width, drawing.height, fillColor=HexColor(WHITE), strokeColor=None))

    family_mode = view == "family"
    focal_only = view == "main"
    x_model = 11.0 if letter else 7.0
    model_width = 40.0 if config.key == "gb" else 37.0
    x_mean = x_model + model_width + 2.5
    mean_width = 27.0
    x_tasks = x_mean + mean_width + 4.0
    right = 187.0
    task_width = (right - x_tasks) / len(config.tasks)

    title_y = height_mm - 7.2
    if letter:
        add_text(drawing, 2.0, title_y - 0.4, letter, 14.0, bold=True)
    add_text(drawing, x_model, title_y, config.context, 7.0, bold=True)
    draw_family_legend(drawing, right, title_y + 0.1, focal_only=focal_only)

    group_y = height_mm - 12.1
    add_text(drawing, x_mean + mean_width / 2, group_y, "Mean MCC", 6.0, bold=True, anchor="middle")
    task_index = {task: index for index, task in enumerate(config.tasks)}
    for group_name, group_tasks in config.groups:
        first = min(task_index[task] for task in group_tasks)
        last = max(task_index[task] for task in group_tasks)
        left = x_tasks + first * task_width
        end = x_tasks + (last + 1) * task_width
        add_text(drawing, (left + end) / 2, group_y, group_name, 6.0, bold=True, anchor="middle")
        drawing.add(Line(left * mm, (group_y - 1.2) * mm, end * mm, (group_y - 1.2) * mm, strokeColor=HexColor(GROUP_LINE), strokeWidth=0.55))

    if config.key == "nt":
        # Rotated labels start low enough that their ascenders do not enter the
        # benchmark-group labels above them.
        label_y = height_mm - 30.0
        row_top = height_mm - 37.5
        for index, task in enumerate(config.tasks):
            add_rotated_text(drawing, x_tasks + index * task_width + 0.45, label_y, config.aliases[task], 6.0)
    else:
        label_y = height_mm - 17.7
        row_top = height_mm - 24.0
        for index, task in enumerate(config.tasks):
            add_multiline_center(drawing, x_tasks + (index + 0.5) * task_width, label_y, config.aliases[task], 6.0)

    bottom = 11.2
    positions = row_positions(rows, row_top, bottom, family_mode)
    if len(positions) > 1:
        sorted_y = sorted(positions.values(), reverse=True)
        nominal_step = float(np.median(np.diff(sorted_y) * -1))
    else:
        nominal_step = 3.4
    band_h = min(max(nominal_step * 0.84, 2.6), 3.5)

    lookup = cells.set_index(["model", "dataset"])
    row_lookup = rows.set_index("model")
    prior_family = None
    for sequence_index, row in enumerate(rows.itertuples(index=False)):
        model = str(row.model)
        y = positions[model]
        family = str(row.family)
        shade = (
            family_color("hEVO") if family == "hEVO" else NEUTRAL
        ) if focal_only else family_color(family)
        fill = HEVO_WASH if family == "hEVO" else (NEUTRAL_PALE if sequence_index % 2 else WHITE)
        drawing.add(Rect((x_model - 1.4) * mm, (y - band_h / 2) * mm, (right - x_model + 1.4) * mm, band_h * mm, fillColor=HexColor(fill), strokeColor=None))

        if family_mode and family != prior_family:
            add_text(drawing, x_model - 1.1, y + band_h / 2 + 1.0, str(row.family_block), 6.0, bold=True, fill=shade)
            drawing.add(Line((x_model - 1.1) * mm, (y + band_h / 2 + 0.35) * mm, (x_model + 9.0) * mm, (y + band_h / 2 + 0.35) * mm, strokeColor=HexColor(shade), strokeWidth=0.85))
        prior_family = family

        drawing.add(Circle((x_model + 0.4) * mm, y * mm, 0.69 * mm, fillColor=HexColor(shade), strokeColor=None))
        base.add_fitted_text(
            drawing,
            x_model + 2.1,
            y - 0.72,
            str(row.model_display),
            6.1,
            model_width - 3.0,
            bold=family == "hEVO",
            color=BLACK,
            min_size=6.0,
        )

        mean_left = x_mean + 0.8
        mean_right = x_mean + 17.0
        drawing.add(Line(mean_left * mm, y * mm, mean_right * mm, y * mm, strokeColor=HexColor(NEUTRAL_LIGHT), strokeWidth=0.70))
        mean_x = mean_left + float(np.clip(row.mean_mcc, 0.0, 0.70)) / 0.70 * (mean_right - mean_left)
        drawing.add(Line(mean_left * mm, y * mm, mean_x * mm, y * mm, strokeColor=HexColor(shade), strokeWidth=0.95))
        drawing.add(Circle(mean_x * mm, y * mm, 0.58 * mm, fillColor=HexColor(shade), strokeColor=None))
        add_text(drawing, x_mean + 18.2, y - 0.70, f"{float(row.mean_mcc):.3f}", 6.0)

        for task_number, task in enumerate(config.tasks):
            cell = lookup.loc[(model, task)]
            left = x_tasks + task_number * task_width + 0.45
            span = task_width - 0.90
            drawing.add(Line(left * mm, y * mm, (left + span) * mm, y * mm, strokeColor=HexColor(NEUTRAL_LIGHT), strokeWidth=0.48))
            lo = left + float(np.clip(cell.mcc_lower, 0.0, 1.0)) * span
            hi = left + float(np.clip(cell.mcc_upper, 0.0, 1.0)) * span
            mid = left + float(np.clip(cell.mcc_mean, 0.0, 1.0)) * span
            drawing.add(Line(lo * mm, y * mm, hi * mm, y * mm, strokeColor=HexColor(shade), strokeWidth=0.80))
            rank_value = int(cell.task_rank_recomputed)
            top_three = rank_value <= 3
            marker_color = RANK_COLORS[rank_value] if top_three and focal_only else shade
            add_marker(
                drawing,
                mid,
                y,
                str(cell.rank_marker),
                marker_color if top_three else WHITE,
                marker_color,
            )

    footer_y = 4.8
    add_text(drawing, x_model, footer_y, "Global task rank", 6.0, bold=True)
    cursor = x_model + 22.0
    for rank_value, shape, label in ((1, "diamond", "1"), (2, "square", "2"), (3, "triangle", "3"), (4, "circle", "other")):
        legend_color = RANK_COLORS[rank_value] if focal_only and rank_value <= 3 else BLACK
        add_marker(drawing, cursor, footer_y + 0.7, shape, legend_color if shape != "circle" else WHITE, legend_color)
        add_text(drawing, cursor + 1.5, footer_y, label, 6.0)
        cursor += 7.2 if label != "other" else 11.5
    drawing.add(Line(cursor * mm, (footer_y + 0.7) * mm, (cursor + 7.0) * mm, (footer_y + 0.7) * mm, strokeColor=HexColor(BLACK), strokeWidth=0.78))
    add_marker(drawing, cursor + 3.5, footer_y + 0.7, "circle", WHITE, BLACK)
    add_text(drawing, cursor + 8.7, footer_y, "MCC mean + 95% CI", 6.0)
    add_text(drawing, right, footer_y, "Mean: unweighted across labeled tasks", 6.0, anchor="end")
    return drawing


def save_view(
    config: BenchmarkConfig,
    cells: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    view: str,
    letter: str | None,
    height_mm: float,
    stem: str,
) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    subset = cells[cells["model"].isin(rows["model"])].copy()
    subset = subset.merge(rows[["model", "plot_rank"]], on="model", how="left")
    subset = subset.sort_values(["plot_rank", "task_order"])
    source_path = SOURCE_DIR / f"{stem}_plot_source.csv"
    subset.to_csv(source_path, index=False)

    drawing = draw_view(config, subset, rows, view=view, letter=letter, height_mm=height_mm)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    svg_path = OUT_DIR / f"{stem}.svg"
    png_path = OUT_DIR / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf_path))
    renderSVG.drawToFile(drawing, str(svg_path))
    base.render_png(pdf_path, png_path)

    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    size_mm = [
        float(page.mediabox.width) / 72.0 * MM_PER_INCH,
        float(page.mediabox.height) / 72.0 * MM_PER_INCH,
    ]
    extracted = "".join((page.extract_text() or "") for page in reader.pages)
    fonts = base.pdf_font_summary(pdf_path)
    with Image.open(png_path) as image:
        pixels = list(image.size)
    expected_pixels = [round(190.0 / MM_PER_INCH * PNG_DPI), round(height_mm / MM_PER_INCH * PNG_DPI)]
    checks = {
        "one_page": len(reader.pages) == 1,
        "size_mm": [round(value, 3) for value in size_mm],
        "size_pass": abs(size_mm[0] - 190.0) < 0.15 and abs(size_mm[1] - height_mm) < 0.15,
        "png_pixels": pixels,
        "expected_png_pixels": expected_pixels,
        "dpi_pass": all(abs(actual - expected) <= 2 for actual, expected in zip(pixels, expected_pixels)),
        "arial_present": fonts["arial_present"],
        "required_arial_fonts_embedded": fonts["required_arial_fonts_embedded"],
        "editable_svg_text": "<text" in svg_path.read_text(encoding="utf-8"),
        "replacement_character_absent": "\ufffd" not in extracted,
        "models_in_view": int(rows["model"].nunique()),
        "tasks_in_view": int(subset["dataset"].nunique()),
        "source_rows": int(len(subset)),
        "source_rows_expected": int(len(rows) * len(config.tasks)),
        "source_complete": len(subset) == len(rows) * len(config.tasks),
        "no_duplicate_cells": not subset.duplicated(["model", "dataset"]).any(),
    }
    checks["all_checks_pass"] = all(
        bool(checks[key])
        for key in (
            "one_page",
            "size_pass",
            "dpi_pass",
            "arial_present",
            "required_arial_fonts_embedded",
            "editable_svg_text",
            "replacement_character_absent",
            "source_complete",
            "no_duplicate_cells",
        )
    )
    return {
        "view": view,
        "pdf": str(pdf_path.resolve()),
        "svg": str(svg_path.resolve()),
        "png": str(png_path.resolve()),
        "plot_source": str(source_path.resolve()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("gb", "nt", "both"), default="both")
    args = parser.parse_args()
    base.configure_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    keys = ("gb", "nt") if args.benchmark == "both" else (args.benchmark,)
    combined: dict[str, object] = {"status": STATUS, "benchmarks": {}, "outputs": []}
    model_sets: dict[str, set[str]] = {}
    for key in keys:
        config = CONFIGS[key]
        cells, summary, data_summary = read_and_validate(config)
        model_sets[key] = set(summary["model"])
        main_rows, selection_summary = select_main(summary)
        all_rows = prepare_all(summary)
        family_rows = prepare_families(summary)

        normalized_path = SOURCE_DIR / f"{key}_normalized_full_source.csv"
        selection_path = SOURCE_DIR / f"{key}_representative_selection_summary.csv"
        summary_path = SOURCE_DIR / f"{key}_model_summary.csv"
        cells.to_csv(normalized_path, index=False)
        selection_summary.to_csv(selection_path, index=False)
        summary.to_csv(summary_path, index=False)
        registry_path = SOURCE_DIR / "model_name_registry.csv"
        registry_rows = pd.DataFrame(
            [model_registry_record(model) for model in sorted(set(summary["model"]))]
        )
        registry_rows.to_csv(registry_path, index=False)

        main_stem = (
            "Fig2c_GB_best_algorithm_representatives_DATA_QC_DRAFT"
            if key == "gb"
            else "Fig2d_NT_best_algorithm_representatives_DATA_QC_DRAFT"
        )
        all_stem = (
            "FigS2_GB_all_models_DATA_QC_DRAFT"
            if key == "gb"
            else "FigS2_NT_all_models_DATA_QC_DRAFT"
        )
        family_stem = (
            "FigS2_GB_family_series_DATA_QC_DRAFT"
            if key == "gb"
            else "FigS2_NT_family_series_DATA_QC_DRAFT"
        )
        outputs = [
            save_view(config, cells, main_rows, view="main", letter=config.letter, height_mm=96.0, stem=main_stem),
            save_view(config, cells, all_rows, view="all", letter=None, height_mm=144.0, stem=all_stem),
            save_view(config, cells, family_rows, view="family", letter=None, height_mm=91.0, stem=family_stem),
        ]
        combined["outputs"].extend(outputs)
        combined["benchmarks"][key] = {
            "data_summary": data_summary,
            "normalized_source": str(normalized_path.resolve()),
            "model_summary": str(summary_path.resolve()),
            "selection_summary": str(selection_path.resolve()),
            "model_name_registry": str(registry_path.resolve()),
            "main_selected_models": main_rows[["model", "model_display", "mean_mcc", "all_model_rank"]].to_dict("records"),
            "main_model_count": int(len(main_rows)),
            "all_model_count": int(len(all_rows)),
            "family_model_count": int(len(family_rows)),
        }

    if set(keys) == {"gb", "nt"}:
        combined["cross_benchmark_model_sets_identical"] = model_sets["gb"] == model_sets["nt"]
    combined["all_artifact_checks_pass"] = all(
        bool(output["checks"]["all_checks_pass"]) for output in combined["outputs"]
    )
    combined["submission_gate"] = (
        "PASS"
        if combined["all_artifact_checks_pass"]
        and all(
            item["data_summary"]["rank_mismatch_count"] == 0
            for item in combined["benchmarks"].values()
        )
        else "DATA_QC_DRAFT"
    )


if __name__ == "__main__":
    main()
