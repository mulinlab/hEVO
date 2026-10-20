#!/usr/bin/env python3
"""Render FigS9 from the local S36-S39 tables."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.metrics import calinski_harabasz_score

HERE = Path(__file__).resolve()
BENCHMARK = HERE.parents[3]
SCRIPT_ROOT = BENCHMARK / "figures" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from common import supplementary_v8_base as b  # noqa: E402

OUT = BENCHMARK / "figures" / "FigS9"
TABLES = BENCHMARK / "tables"
SRC = OUT / "source_data"
for directory in (OUT, SRC):
    directory.mkdir(parents=True, exist_ok=True)
b.TABLES, b.OUT, b.SRC = TABLES, OUT, SRC
b.CHECKS = {
    "figure": "FigS9",
    "sources": {},
    "figures": {},
    "text_bounds": {},
    "known_limitations": [],
}

TITLE = (
    "Supplementary Figure S9. Human evolutionary sequence classification "
    "and adaptive-allele preference across all models"
)
MODEL_ORDER = [
    "hEvo-large", "hEvo-base",
    "evo2_7b_base", "evo2_1b_base",
    "NT_2.5b_1kG", "NT_2.5b_MS", "NT_500m_MS", "NTv3-650M-pre",
    "NTv3-100M-pre", "NT_500m_1kG", "NTv3-8M-pre",
    "GPN-Star-M", "GPN-Star-P", "GPN-Star-V", "GPN-promoter", "GPN-MSA",
    "GENA-LM-large", "Mistral-DNA-422M", "GENA-LM-base", "DNABERT2",
    "Mistral-DNA-17M", "Caduceus-PS", "GROVER", "Genos-1.2B",
    "SpeciesLM", "AIDO.DNA-300M", "Caduceus-PH", "HyenaDNA_tiny_1k",
    "Mistral-DNA-138M", "GENERator-1.2B", "GENERator-3B",
]
OMITTED_BY_REQUEST = {"DNABERT"}
OMITTED_FROM_REFERENCE = {"hEvo-open-large", "hEvo-open-base"}
FAMILY_MARK = {"Evo 2": "triangle", "NT": "square", "GPN": "diamond"}
EXPECTED_RUNS = 5
EXPECTED_LOCI = 30


def _metric_rank(data: pd.DataFrame, column: str) -> pd.Series:
    return data[column].rank(ascending=False, method="min").astype(int)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read S36-S39, apply the requested model roster, and compute S39 CH."""
    s36, s37, s38, s39 = (b.load(number) for number in (36, 37, 38, 39))
    if s36.duplicated(["model", "run"]).any() or s37.duplicated(["model", "run"]).any():
        raise ValueError("S36/S37 contain duplicate model/run rows")
    if s38["model_id"].duplicated().any():
        raise ValueError("S38 contains duplicate model rows")
    if s39.duplicated(["model_id", "variant_id"]).any():
        raise ValueError("S39 contains duplicate model/locus rows")

    s36_models, s37_models, s38_models = (
        set(s36["model"]), set(s37["model"]), set(s38["model_id"])
    )
    if not (s36_models == s37_models == s38_models):
        raise ValueError("S36-S38 source tables do not have the same model roster")
    if s36_models != set(MODEL_ORDER) | OMITTED_FROM_REFERENCE | OMITTED_BY_REQUEST:
        raise ValueError(
            "FigS9 reference roster does not reconcile with S36-S38: "
            f"missing={sorted(s36_models - set(MODEL_ORDER) - OMITTED_FROM_REFERENCE - OMITTED_BY_REQUEST)}, "
            f"unaccounted={sorted((set(MODEL_ORDER) | OMITTED_FROM_REFERENCE | OMITTED_BY_REQUEST) - s36_models)}"
        )
    if not set(MODEL_ORDER).issubset(s39["model_id"]):
        raise ValueError("S39 is missing one or more models displayed in FigS9")
    if len(MODEL_ORDER) != 31:
        raise ValueError(f"FigS9 after DNABERT removal should contain 31 models, got {len(MODEL_ORDER)}")

    preferences = s38.set_index("model_id")
    selected_preferences = preferences.reindex(MODEL_ORDER)
    if selected_preferences["status"].ne("PASS").any():
        raise ValueError("Every displayed model must have a PASS row in S38")
    if not pd.to_numeric(selected_preferences["n_loci"], errors="raise").eq(EXPECTED_LOCI).all():
        raise ValueError(f"S38 must report {EXPECTED_LOCI} loci for every displayed model")

    data = pd.DataFrame(index=pd.Index(MODEL_ORDER, name="model"))
    data["model_display"] = [b.display(model) for model in MODEL_ORDER]
    data["family"] = [b.FAMILY[model] for model in MODEL_ORDER]
    data["preference_accuracy"] = pd.to_numeric(
        selected_preferences["accuracy"], errors="raise"
    )
    data["preference_ci_low"] = pd.to_numeric(
        selected_preferences["accuracy_ci_low"], errors="raise"
    )
    data["preference_ci_high"] = pd.to_numeric(
        selected_preferences["accuracy_ci_high"], errors="raise"
    )
    data["positive_loci"] = pd.to_numeric(
        selected_preferences["positive_loci"], errors="raise"
    ).astype(int)
    data["n_loci"] = EXPECTED_LOCI
    if not (
        data["preference_ci_low"].le(data["preference_accuracy"])
        & data["preference_accuracy"].le(data["preference_ci_high"])
    ).all():
        raise ValueError("S38 preference intervals do not contain their point estimates")

    run_records: list[pd.DataFrame] = []
    for label, source in (("HAR", s37), ("HAQER", s36)):
        frame = source.loc[source["model"].isin(MODEL_ORDER)].copy()
        frame["mcc_mean"] = pd.to_numeric(frame["mcc_mean"], errors="raise")
        frame["mcc_low"] = pd.to_numeric(frame["mcc_low"], errors="raise")
        frame["mcc_high"] = pd.to_numeric(frame["mcc_high"], errors="raise")
        if not frame["mcc_mean"].between(-1, 1).all():
            raise ValueError(f"{label} MCC values fall outside [-1, 1]")
        if not (
            frame["mcc_low"].le(frame["mcc_mean"])
            & frame["mcc_mean"].le(frame["mcc_high"])
        ).all():
            raise ValueError(f"{label} S-table intervals do not contain their MCC values")
        grouped = frame.groupby("model", sort=False)
        counts = grouped.size().reindex(MODEL_ORDER)
        if not counts.eq(EXPECTED_RUNS).all():
            raise ValueError(f"{label} must contain {EXPECTED_RUNS} runs per displayed model")

        summaries = []
        for model in MODEL_ORDER:
            rows = grouped.get_group(model)
            means = rows["mcc_mean"].to_numpy(dtype=float)
            stored_mean = pd.to_numeric(rows["mean_mcc_for_sort"], errors="raise").to_numpy(dtype=float)
            if not np.allclose(stored_mean, stored_mean[0], rtol=0, atol=1e-12):
                raise ValueError(f"{label} mean_mcc_for_sort changes across runs for {model}")
            if not np.isclose(float(means.mean()), stored_mean[0], rtol=0, atol=2e-6):
                raise ValueError(f"{label} aggregate MCC does not reconcile for {model}")
            summaries.append({
                "model": model,
                f"{label.lower()}_mean_mcc": float(stored_mean[0]),
                f"{label.lower()}_run_min": float(means.min()),
                f"{label.lower()}_run_max": float(means.max()),
            })
            run_records.append(
                rows.assign(dataset=label)[
                    ["dataset", "model", "run", "mcc_mean", "mcc_low", "mcc_high"]
                ]
            )
        summary = pd.DataFrame(summaries).set_index("model")
        data = data.join(summary)

    s39_selected = s39.loc[
        s39["model_id"].isin(MODEL_ORDER) & s39["benchmark_status"].eq("PASS")
    ].copy()
    s39_selected["selected_logprob"] = pd.to_numeric(
        s39_selected["selected_logprob"], errors="coerce"
    )
    s39_selected["other_logprob"] = pd.to_numeric(
        s39_selected["other_logprob"], errors="coerce"
    )
    ch_records = []
    for model in MODEL_ORDER:
        rows = s39_selected.loc[s39_selected["model_id"].eq(model)]
        if len(rows) != EXPECTED_LOCI:
            raise ValueError(f"S39 must contain {EXPECTED_LOCI} passing loci for {model}")
        values = rows[["selected_logprob", "other_logprob"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"S39 has non-finite raw logprob values for {model}")
        directional_correct = int(pd.to_numeric(rows["directional_correct"], errors="raise").sum())
        if directional_correct != int(data.at[model, "positive_loci"]):
            raise ValueError(f"S38/S39 directional counts disagree for {model}")
        score_values = np.concatenate([values[:, 0], values[:, 1]]).reshape(-1, 1)
        labels = np.concatenate([
            np.ones(EXPECTED_LOCI, dtype=int),
            np.zeros(EXPECTED_LOCI, dtype=int),
        ])
        ch_value = float(calinski_harabasz_score(score_values, labels))
        if not np.isfinite(ch_value) or ch_value <= 0:
            raise ValueError(f"S39 produced an invalid CH index for {model}")
        data.at[model, "score_separation_ch_index"] = ch_value
        ch_records.append({
            "model": model,
            "model_display": data.at[model, "model_display"],
            "passing_loci": len(rows),
            "positive_loci": directional_correct,
            "score_separation_ch_index": ch_value,
            "definition": "Calinski-Harabasz on 30 selected and 30 other raw logprob values",
        })

    data["preference_pct"] = data["preference_accuracy"] * 100
    data["preference_ci_low_pct"] = data["preference_ci_low"] * 100
    data["preference_ci_high_pct"] = data["preference_ci_high"] * 100
    ranks = pd.DataFrame(index=data.index)
    for column in (
        "har_mean_mcc", "haqer_mean_mcc", "preference_accuracy", "score_separation_ch_index"
    ):
        ranks[column + "_rank"] = _metric_rank(data, column)

    run_data = pd.concat(run_records, ignore_index=True)
    b.save(data.reset_index(), "S9_plotted_values")
    b.save(run_data, "S9_run_level_MCC")
    b.save(pd.DataFrame(ch_records), "S9_CH_from_S39")
    b.save(ranks.reset_index(), "S9_within_metric_ranks")
    order = data.reset_index()[["model", "model_display", "family"]].copy()
    order.insert(0, "row_order", np.arange(1, len(order) + 1))
    b.save(order, "S9_model_order")
    return data, ranks


def _axis_transform(value: float, low: float, high: float, logarithmic: bool) -> float:
    if logarithmic:
        if value <= 0:
            raise ValueError(f"Log-scale axis received a nonpositive value: {value}")
        return (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
    return (value - low) / (high - low)


def draw_axis(
    page,
    *,
    x: float,
    y: float,
    width: float,
    low: float,
    high: float,
    ticks: tuple[float, ...],
    top: float,
    tick_format,
    logarithmic: bool = False,
) -> None:
    page.line(x, y, x + width, y, .42, "#202020")
    for tick in ticks:
        xx = x + _axis_transform(tick, low, high, logarithmic) * width
        page.line(xx, top, xx, y, .22, "#E4E4E4")
        page.line(xx, y, xx, y + .9, .32, "#202020")
        page.text(xx, y + 3.65, tick_format(tick), 5.15, align="center")


def draw_rank(page, value_x: float, row_y: float, rank: int) -> None:
    if rank <= 3:
        page.text(value_x + .48, row_y + .10, rank, 4.15, color="#303030")


def draw_mcc_distribution(
    page,
    *,
    model: str,
    value: float,
    run_values: np.ndarray,
    row_y: float,
    axis: dict[str, float],
    rank: int,
) -> None:
    if not np.isfinite(run_values).all() or not axis["low"] <= run_values.min() <= run_values.max() <= axis["high"]:
        raise ValueError(f"MCC range falls outside its axis for {model}")
    color = b.COL[b.FAMILY[model]]
    shape = FAMILY_MARK.get(b.FAMILY[model], "circle")
    transform = lambda number: axis["x"] + _axis_transform(
        float(number), axis["low"], axis["high"], False
    ) * axis["width"]
    lower, upper = transform(run_values.min()), transform(run_values.max())
    page.line(lower, row_y, upper, row_y, .40, color)
    page.line(lower, row_y - .31, lower, row_y + .31, .34, color)
    page.line(upper, row_y - .31, upper, row_y + .31, .34, color)
    for replicate in run_values:
        page.marker(transform(replicate), row_y, .37, color, open_=True, shape=shape)
    page.marker(
        transform(value), row_y, .66, color,
        open_=model == "hEvo-base", shape=shape,
    )
    page.text(axis["value_x"], row_y + .57, f"{value:.3f}", 5.65, rank <= 3, align="right")
    draw_rank(page, axis["value_x"], row_y, rank)


def draw_preference(
    page,
    *,
    model: str,
    mean_pct: float,
    low_pct: float,
    high_pct: float,
    row_y: float,
    axis: dict[str, float],
    rank: int,
) -> None:
    if not axis["low"] <= low_pct <= mean_pct <= high_pct <= axis["high"]:
        raise ValueError(f"S38 preference interval falls outside its axis for {model}")
    color = b.COL[b.FAMILY[model]]
    shape = FAMILY_MARK.get(b.FAMILY[model], "circle")
    transform = lambda number: axis["x"] + _axis_transform(
        float(number), axis["low"], axis["high"], False
    ) * axis["width"]
    lower, upper = transform(low_pct), transform(high_pct)
    page.line(lower, row_y, upper, row_y, .40, color)
    page.line(lower, row_y - .34, lower, row_y + .34, .34, color)
    page.line(upper, row_y - .34, upper, row_y + .34, .34, color)
    page.marker(
        transform(mean_pct), row_y, .60, color,
        open_=model == "hEvo-base", shape=shape,
    )
    page.text(axis["value_x"], row_y + .57, f"{mean_pct:.1f}", 5.65, rank <= 3, align="right")
    draw_rank(page, axis["value_x"], row_y, rank)


def _format_ch(value: float) -> str:
    return re.sub(r"e([+-])0+([0-9]+)", r"e\1\2", f"{value:.3g}")


def draw_ch(
    page,
    *,
    model: str,
    value: float,
    row_y: float,
    axis: dict[str, float],
    rank: int,
) -> None:
    if not axis["low"] <= value <= axis["high"]:
        raise ValueError(f"CH index falls outside its log axis for {model}: {value}")
    color = b.COL[b.FAMILY[model]]
    shape = FAMILY_MARK.get(b.FAMILY[model], "circle")
    xx = axis["x"] + _axis_transform(value, axis["low"], axis["high"], True) * axis["width"]
    page.marker(xx, row_y, .60, color, open_=model == "hEvo-base", shape=shape)
    page.text(axis["value_x"], row_y + .57, _format_ch(value), 5.55, rank <= 3, align="right")
    draw_rank(page, axis["value_x"], row_y, rank)


def render(data: pd.DataFrame, ranks: pd.DataFrame, run_data: pd.DataFrame) -> Path:
    page = b.Page("FigureS9_HAR_HAQER_adaptive_alleles_A4", TITLE)
    first_y, row_pitch, family_gap = 28.0, 3.0, .78
    row_y: dict[str, float] = {}
    y = first_y
    previous_family = None
    for model in MODEL_ORDER:
        family = b.FAMILY[model]
        if previous_family is not None and family != previous_family:
            y += family_gap
        row_y[model] = y
        y += row_pitch
        previous_family = family
    last_y = row_y[MODEL_ORDER[-1]]
    axis_y = last_y + 4.55
    plot_top = first_y - 1.0

    # Panel titles and column headers.
    page.text(45.0, 13.0, "a", 10.5, True)
    page.text(63.0, 13.3, "HAR", 6.7, True, align="center")
    page.text(85.0, 13.0, "b", 10.5, True)
    page.text(103.0, 13.3, "HAQER", 6.7, True, align="center")
    page.text(125.0, 13.0, "c", 10.5, True)
    page.text(165.0, 13.3, "Adaptive-allele benchmark", 6.7, True, align="center")

    page.text(11.2, 21.2, "Model", 6.1, True)
    page.text(63.0, 21.2, "Mean MCC", 6.1, True, align="center")
    page.text(103.0, 21.2, "Mean MCC", 6.1, True, align="center")
    page.text(144.0, 17.2, "Selected-allele", 6.1, True, align="center")
    page.text(144.0, 20.7, "preference (%)", 6.1, True, align="center")
    page.text(185.0, 17.2, "Score separation", 6.1, True, align="center")
    page.text(185.0, 20.7, "CH index (log scale)", 6.1, True, align="center")
    for left, right in ((44.8, 81.0), (84.8, 121.0), (124.8, 202.0)):
        page.line(left, 24.7, right, 24.7, .50, "#202020")

    # Row labels and separators preserve the family order and colors in the PDF.
    previous_family = None
    for model in MODEL_ORDER:
        center_y = row_y[model]
        family = b.FAMILY[model]
        if model in b.FOCUS:
            page.rect(8.7, center_y - row_pitch * .49, 194.3, row_pitch * .98, b.ROW_HIGHLIGHT)
        if previous_family is not None and family != previous_family:
            page.line(8.7, center_y - row_pitch * .5 - family_gap * .5,
                      203.0, center_y - row_pitch * .5 - family_gap * .5, .32, "#969696")
        page.marker(
            9.55, center_y, .67 if model in b.FOCUS else .55, b.COL[family],
            open_=model == "hEvo-base",
            shape=FAMILY_MARK.get(family, "circle"),
        )
        page.text(
            11.2, center_y + .62, data.at[model, "model_display"], 6.05,
            model in b.FOCUS,
        )
        previous_family = family

    har_axis = {"x": 45.0, "width": 25.0, "low": 0.0, "high": 1.0, "value_x": 80.2}
    haqer_axis = {"x": 85.0, "width": 25.0, "low": 0.0, "high": 1.0, "value_x": 120.2}
    preference_axis = {"x": 125.0, "width": 27.0, "low": 0.0, "high": 100.0, "value_x": 163.7}
    ch_axis = {"x": 170.0, "width": 22.0, "low": 1e-6, "high": 100.0, "value_x": 201.0}
    draw_axis(
        page, **{key: har_axis[key] for key in ("x", "width", "low", "high")},
        y=axis_y, ticks=(0.0, .5, 1.0), top=plot_top,
        tick_format=lambda value: f"{value:g}",
    )
    draw_axis(
        page, **{key: haqer_axis[key] for key in ("x", "width", "low", "high")},
        y=axis_y, ticks=(0.0, .5, 1.0), top=plot_top,
        tick_format=lambda value: f"{value:g}",
    )
    draw_axis(
        page, **{key: preference_axis[key] for key in ("x", "width", "low", "high")},
        y=axis_y, ticks=(0, 40, 80), top=plot_top,
        tick_format=lambda value: f"{value:.0f}",
    )
    ch_tick_labels = {1e-4: "1e-4", 1e-2: "0.01", 1.0: "1", 100.0: "100"}
    draw_axis(
        page, **{key: ch_axis[key] for key in ("x", "width", "low", "high")},
        y=axis_y, ticks=(1e-4, 1e-2, 1.0, 100.0), top=plot_top,
        tick_format=lambda value: ch_tick_labels[value], logarithmic=True,
    )

    rendered: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        values = data.loc[model]
        rank = ranks.loc[model]
        y = row_y[model]
        for dataset, axis in (("HAR", har_axis), ("HAQER", haqer_axis)):
            metric = f"{dataset.lower()}_mean_mcc"
            run_values = run_data.loc[
                run_data["dataset"].eq(dataset) & run_data["model"].eq(model), "mcc_mean"
            ].to_numpy(dtype=float)
            draw_mcc_distribution(
                page,
                model=model,
                value=float(values[metric]),
                run_values=run_values,
                row_y=y,
                axis=axis,
                rank=int(rank[metric + "_rank"]),
            )
            rendered.append({
                "model": model,
                "metric": metric,
                "estimate": float(values[metric]),
                "range_min": float(run_values.min()),
                "range_max": float(run_values.max()),
                "rank": int(rank[metric + "_rank"]),
                "row_y_mm": y,
            })

        preference_rank = int(rank["preference_accuracy_rank"])
        draw_preference(
            page,
            model=model,
            mean_pct=float(values["preference_pct"]),
            low_pct=float(values["preference_ci_low_pct"]),
            high_pct=float(values["preference_ci_high_pct"]),
            row_y=y,
            axis=preference_axis,
            rank=preference_rank,
        )
        rendered.append({
            "model": model,
            "metric": "preference_pct",
            "estimate": float(values["preference_pct"]),
            "lower": float(values["preference_ci_low_pct"]),
            "upper": float(values["preference_ci_high_pct"]),
            "rank": preference_rank,
            "row_y_mm": y,
        })

        ch_rank = int(rank["score_separation_ch_index_rank"])
        ch_value = float(values["score_separation_ch_index"])
        draw_ch(page, model=model, value=ch_value, row_y=y, axis=ch_axis, rank=ch_rank)
        rendered.append({
            "model": model,
            "metric": "score_separation_ch_index",
            "estimate": ch_value,
            "rank": ch_rank,
            "row_y_mm": y,
        })

    page.line(8.7, last_y + 2.1, 203.0, last_y + 2.1, .45)
    legend_y = axis_y + 13.5
    b.family_legend(page, legend_y)
    page.text(143.0, legend_y, "1, 2, 3   Within-metric rank", 5.55)
    pdf = page.finish()

    expected_points = len(MODEL_ORDER) * 4
    if len(rendered) != expected_points:
        raise ValueError(f"Expected {expected_points} plotted estimates, got {len(rendered)}")
    b.save(pd.DataFrame(rendered), "S9_rendered_points")
    b.CHECKS["figures"]["S9"] = {
        "models": len(MODEL_ORDER),
        "omitted_by_request": sorted(OMITTED_BY_REQUEST),
        "omitted_from_reference_roster": sorted(OMITTED_FROM_REFERENCE),
        "metrics": [
            "HAR mean MCC",
            "HAQER mean MCC",
            "S38 selected-allele preference (%)",
            "S39 raw-score Calinski-Harabasz index",
        ],
        "HAR_HAQER_interval": "observed min-to-max across five negative-control run means; all five run means are plotted",
        "preference_interval": "S38 exact binomial interval",
        "CH_definition": "Calinski-Harabasz on 30 selected and 30 other raw S39 logprob values",
        "plotted_estimates": expected_points,
        "GPN-MSA_values": data.loc["GPN-MSA", [
            "har_mean_mcc", "haqer_mean_mcc", "preference_pct", "score_separation_ch_index",
        ]].to_dict(),
        "row_order": "the displayed family-grouped order in the supplied reference PDF, without DNABERT",
    }
    return pdf


def export(output_dir: Path | None = None, tables_dir: Path | None = None) -> dict[str, str]:
    global OUT, TABLES, SRC
    if output_dir is not None:
        OUT = Path(output_dir).resolve()
        SRC = OUT / "source_data"
        for directory in (OUT, SRC):
            directory.mkdir(parents=True, exist_ok=True)
        b.OUT, b.SRC = OUT, SRC
    if tables_dir is not None:
        TABLES = Path(tables_dir).resolve()
        b.TABLES = TABLES

    data, ranks = load_data()
    run_data = pd.read_csv(SRC / "S9_run_level_MCC.csv")
    source_pdf = render(data, ranks, run_data)
    integrated = OUT / "A4_integrated.pdf"
    if source_pdf.resolve() != integrated.resolve():
        shutil.copyfile(source_pdf, integrated)
    reader = PdfReader(str(integrated))
    text = reader.pages[0].extract_text() or ""
    required = (
        "HAR", "HAQER", "Adaptive-allele benchmark", "Selected-allele",
        "Score separation", "GPN-MSA", "DNABERT2",
    )
    required = required + tuple(_format_ch(float(value)) for value in data["score_separation_ch_index"])
    missing = [token for token in required if token not in text]
    standalone_dnabert = re.search(r"\bDNABERT\b", text)
    if len(reader.pages) != 1 or missing or standalone_dnabert:
        raise ValueError(
            f"FigS9 page/content check failed: pages={len(reader.pages)}, "
            f"missing={missing}, standalone_DNABERT={standalone_dnabert is not None}"
        )

    pdftoppm = shutil.which("pdftoppm")
    pdftocairo = shutil.which("pdftocairo")
    if not pdftoppm or not pdftocairo:
        raise FileNotFoundError("Poppler pdftoppm and pdftocairo are required for PNG/SVG exports")
    subprocess.run(
        [pdftoppm, "-f", "1", "-l", "1", "-singlefile", "-png", "-r", "600",
         str(integrated), str(OUT / "A4_integrated")],
        check=True, capture_output=True,
    )
    subprocess.run(
        [pdftocairo, "-f", "1", "-l", "1", "-svg", str(integrated),
         str(OUT / "A4_integrated.svg")],
        check=True, capture_output=True,
    )
    b.CHECKS["output"] = {
        "pages": 1,
        "page_mm": [210, 297],
        "local_source_data": True,
        "models": len(MODEL_ORDER),
        "plotted_estimates": len(MODEL_ORDER) * 4,
        "formats": ["PDF", "PNG", "SVG"],
    }
    return {
        "pdf": str(integrated),
        "png": str(OUT / "A4_integrated.png"),
        "svg": str(OUT / "A4_integrated.svg"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    outputs = export(args.output_dir, args.tables_dir)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
