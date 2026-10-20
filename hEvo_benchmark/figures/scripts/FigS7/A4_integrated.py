#!/usr/bin/env python3
"""Render FigS7's six-task AUPRC and AUROC model matrices from local tables."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

HERE = Path(__file__).resolve()
BENCHMARK = HERE.parents[3]
SCRIPT_ROOT = BENCHMARK / "figures" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from common import supplementary_v8_base as b  # noqa: E402

OUT = BENCHMARK / "figures" / "FigS7"
TABLES = BENCHMARK / "tables"
SRC = OUT / "source_data"
for directory in (OUT, SRC):
    directory.mkdir(parents=True, exist_ok=True)
b.TABLES, b.OUT, b.SRC = TABLES, OUT, SRC
b.CHECKS = {"figure": "FigS7", "sources": {}, "figures": {},
            "text_bounds": {}, "known_limitations": []}

TASKS = [
    ("ClinVar", 23, 1, "#D44A37"),
    ("DDD", 30, 2, "#A62A2A"),
    ("GWAS", 26, 3, "#6F9B45"),
    ("Rare GWAS", 27, 4, "#C49A00"),
    ("UKB rare", 28, 5, "#E07B26"),
    ("MPRA", 29, 6, "#16856B"),
]
EXPECTED_MODEL_COUNT = 33
POINTS: list[dict[str, object]] = []


def load_data() -> tuple[dict[int, pd.DataFrame], list[str]]:
    tables = {number: b.load(number) for number in (23, 26, 27, 28, 29, 30)}
    for number, frame in tables.items():
        if frame["model"].duplicated().any():
            raise ValueError(f"S{number} has duplicate model rows")
        positive = pd.to_numeric(frame["num_positive"], errors="raise")
        negative = pd.to_numeric(frame["num_negative"], errors="raise")
        samples = pd.to_numeric(frame["num_samples"], errors="raise")
        if not (positive + negative).eq(samples).all():
            raise ValueError(f"S{number} has inconsistent class and sample counts")
        baseline = pd.to_numeric(frame["baseline"], errors="raise")
        np.testing.assert_allclose(baseline, positive / samples, rtol=0, atol=1e-12)
        for metric in ("auprc", "auroc"):
            for suffix in ("mean", "ci_lower", "ci_upper"):
                values = pd.to_numeric(frame[f"{metric}_{suffix}"], errors="raise")
                if not values.between(0, 1).all():
                    raise ValueError(f"S{number} {metric}_{suffix} is outside [0, 1]")
            if (frame[f"{metric}_ci_lower"] > frame[f"{metric}_ci_upper"]).any():
                raise ValueError(f"S{number} {metric} interval bounds are reversed")
        tables[number] = frame.set_index("model")

    model_sets = [set(frame.index) for frame in tables.values()]
    roster = set.intersection(*model_sets)
    if len(roster) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"S7 expects {EXPECTED_MODEL_COUNT} shared models across all six tasks; "
            f"found {len(roster)}"
        )
    if any(models != roster for models in model_sets):
        differences = {
            f"S{number}": {
                "missing": sorted(roster - set(frame.index)),
                "extra": sorted(set(frame.index) - roster),
            }
            for number, frame in tables.items()
            if set(frame.index) != roster
        }
        raise ValueError(f"S7 source tables do not contain the same model roster: {differences}")

    task_ranks = pd.DataFrame({
        label: tables[number]["auprc_mean"].rank(ascending=False, method="average")
        for label, number, _, _ in TASKS
    })
    mean_rank = task_ranks.mean(axis=1)
    models = []
    for _, family_models in b.GROUPS:
        family_members = [model for model in family_models if model in roster]
        models.extend(sorted(family_members,
                             key=lambda model: (mean_rank[model], b.display(model))))
    if len(models) != EXPECTED_MODEL_COUNT or set(models) != roster:
        raise ValueError("S7 family-group order does not cover the local model roster")

    ordering = pd.DataFrame({
        "model": models,
        "display_name": [b.display(model) for model in models],
        "family": [b.FAMILY[model] for model in models],
        "mean_task_rank": [float(mean_rank[model]) for model in models],
    })
    b.save(ordering, "S7_shared_model_order")
    b.save(task_ranks.loc[models].rename_axis("model").reset_index(),
           "S7_six_task_AUPRC_ranks_for_ordering")

    raw_rows = []
    for task, number, task_number, _ in TASKS:
        raw_rows.append(tables[number].loc[models].reset_index().assign(
            task=task, dataset_number=task_number, source_table=f"S{number}"))
    b.save(pd.concat(raw_rows, ignore_index=True), "S7_all_models_raw_metrics_and_intervals")
    return tables, models


def badge(page, x, y, number, color):
    page.marker(x, y, 1.5, color)
    page.text(x, y + .72, str(number), 6, True, "#FFFFFF", "center")


def row_labels(page, models, y_positions, pitch=2.65, right=200):
    previous_family = None
    for model in models:
        y = y_positions[model]
        family = b.FAMILY[model]
        if model in b.FOCUS:
            page.rect(8.7, y - pitch * .48, right - 8.7, pitch * .96, b.ROW_HIGHLIGHT)
        if previous_family is not None and family != previous_family:
            page.line(8.7, y - pitch * .5 - .35, right, y - pitch * .5 - .35,
                      .3, "#969696")
        marker = {"Evo 2": "triangle", "NT": "square", "GPN": "diamond"}.get(family, "circle")
        page.marker(9.6, y, .62, b.COL[family], open_=model == "hEvo-base", shape=marker)
        page.text(11.35, y + .67, b.display(model), 6, bold=model in b.FOCUS)
        previous_family = family


def axis(page, x, y, width, low, high, ticks, bottom):
    page.line(x, y, x + width, y, .45)
    for tick in ticks:
        xx = x + (tick - low) / (high - low) * width
        page.line(xx, y, xx, y - 1, .35)
        page.text(xx, y - 1.65, f"{tick:.2f}", 5.5, align="center")
        page.line(xx, y + .8, xx, bottom, .2, "#E3E3E3")


def metric_dot(page, row, model, task, metric, x, y, width, low, high,
               value_x, rank=None):
    mean = float(row[f"{metric}_mean"])
    ci_low = float(row[f"{metric}_ci_lower"])
    ci_high = float(row[f"{metric}_ci_upper"])
    if not low <= ci_low <= mean <= ci_high <= high:
        raise ValueError(f"S7 {metric} interval is clipped for {model}/{task}")
    transform = lambda value: x + (value - low) / (high - low) * width
    xx, xl, xh = transform(mean), transform(ci_low), transform(ci_high)
    family = b.FAMILY[model]
    color = b.COL[family]
    if metric == "auprc":
        baseline = float(row.baseline)
        if low <= baseline <= high:
            xb = transform(baseline)
            page.line(xb, y - .73, xb, y + .73, .38, "#161616")
    page.line(xl, y, xh, y, .5, color)
    page.line(xl, y - .37, xl, y + .37, .4, color)
    page.line(xh, y - .37, xh, y + .37, .4, color)
    marker = {"Evo 2": "triangle", "NT": "square", "GPN": "diamond"}.get(family, "circle")
    page.marker(xx, y, .58 if model in b.FOCUS else .48, color,
                open_=model == "hEvo-base", shape=marker)
    page.text(value_x, y + .64, f"{mean:.3f}", 5.75,
              bold=rank is not None and rank <= 3, align="right")
    if rank is not None and rank <= 3:
        page.text(value_x - 7.6, y + .6, str(int(rank)), 4.8,
                  color="#444444", align="right")
    POINTS.append({"figure": "FigS7", "task": task, "metric": metric,
                   "model": model, "mean": mean, "lower": ci_low,
                   "upper": ci_high, "x_mm": xx, "interval_left_mm": xl,
                   "interval_right_mm": xh, "axis_min": low, "axis_max": high})


def render(tables, models):
    POINTS.clear()
    page = b.Page("FigureS7_variant_prediction_A4",
                  "Supplementary Figure S7. Six-task variant prediction")
    limits = {
        "auprc": [(.28, .96, (.30, .60, .90)), (.12, .86, (.20, .50, .80)),
                  (.065, .21, (.08, .14, .20)), (.28, .66, (.30, .45, .60)),
                  (.58, .82, (.60, .70, .80)), (.205, .355, (.22, .28, .34))],
        "auroc": [(.43, 1.0, (.50, .70, .90)), (.42, 1.0, (.50, .70, .90)),
                  (.40, .69, (.40, .50, .60)), (.37, .70, (.40, .50, .60)),
                  (.42, .73, (.45, .55, .65)), (.40, .64, (.40, .50, .60))],
    }
    for block, metric in enumerate(("auprc", "auroc")):
        top = 12 + block * 120
        page.panel("ab"[block], 9, top, metric.upper())
        y_positions = b.rows(models, top + 23.7, pitch=2.65, gap=.65)
        bottom = max(y_positions.values()) + 1.65
        row_labels(page, models, y_positions, pitch=2.65)
        for column, (task, number, task_number, color) in enumerate(TASKS):
            column_x = 47 + column * 25.5
            badge(page, column_x + 10, top + 5.2, task_number, color)
            page.text(column_x + 10, top + 10.7, task, 6, True, align="center")
            low, high, ticks = limits[metric][column]
            x, width = column_x + .3, 14.8
            axis(page, x, top + 19.8, width, low, high, ticks, bottom)
            if metric == "auroc":
                chance_x = x + (.5 - low) / (high - low) * width
                page.line(chance_x, top + 20.8, chance_x, bottom, .4, "#929292")
            ranks = tables[number][f"{metric}_mean"].rank(ascending=False, method="min")
            for model in models:
                metric_dot(page, tables[number].loc[model], model, task, metric,
                           x, y_positions[model], width, low, high,
                           column_x + 24.7, ranks[model])
        page.line(8.7, bottom, 200, bottom, .45)

    b.family_legend(page, bottom + 7)
    page.text(135, bottom + 7, "1, 2, 3  Within-task rank", 5.75)
    source_pdf = page.finish()
    expected_points = EXPECTED_MODEL_COUNT * len(TASKS) * 2
    if len(POINTS) != expected_points:
        raise ValueError(f"Expected {expected_points} plotted cells; rendered {len(POINTS)}")
    b.save(pd.DataFrame(POINTS), "S7_rendered_points")
    b.CHECKS["figures"]["S7"] = {
        "models": len(models), "tasks": len(TASKS), "metrics": 2,
        "panels": ["a AUPRC", "b AUROC"], "plotted_values": len(POINTS),
        "excluded_models": [],
        "model_order": models,
        "row_order": "family groups; within-family mean rank across six AUPRC tasks",
        "source_tables": [f"S{number}" for _, number, _, _ in TASKS],
        "data_changed": False,
    }
    return source_pdf


def export() -> dict[str, str]:
    tables, models = load_data()
    source_pdf = render(tables, models)
    integrated = OUT / "A4_integrated.pdf"
    if source_pdf.resolve() != integrated.resolve():
        shutil.copyfile(source_pdf, integrated)
    reader = PdfReader(str(integrated))
    text = reader.pages[0].extract_text() or ""
    required = ("AUPRC", "AUROC", "ClinVar", "DDD", "GWAS", "Rare GWAS",
                "UKB rare", "MPRA")
    missing = [label for label in required if label not in text]
    if len(reader.pages) != 1 or missing:
        raise ValueError(f"FigS7 page/panel check failed: pages={len(reader.pages)}, missing={missing}")

    pdftoppm = shutil.which("pdftoppm")
    pdftocairo = shutil.which("pdftocairo")
    if not pdftoppm or not pdftocairo:
        raise FileNotFoundError("Poppler pdftoppm and pdftocairo are required for PNG/SVG exports")
    subprocess.run([pdftoppm, "-f", "1", "-l", "1", "-singlefile", "-png", "-r", "600",
                    str(integrated), str(OUT / "A4_integrated")], check=True, capture_output=True)
    subprocess.run([pdftocairo, "-f", "1", "-l", "1", "-svg", str(integrated),
                    str(OUT / "A4_integrated.svg")], check=True, capture_output=True)
    svg = OUT / "A4_integrated.svg"
    b.CHECKS["output"] = {
        "pages": 1, "page_mm": [210, 297], "panels": ["a AUPRC", "b AUROC"],
        "local_source_data": True, "models": len(models),
        "plotted_scores": len(POINTS),
    }
    return {"pdf": str(integrated), "png": str(OUT / "A4_integrated.png"),
            "svg": str(svg), "models": str(len(models)), "points": str(len(POINTS))}


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))
