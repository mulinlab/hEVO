#!/usr/bin/env python3
"""Render FigS8 ancestry metrics from the local S33-S35 source tables."""

from __future__ import annotations

import argparse
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
from common import supplementary_v8_base as b

OUT = BENCHMARK / "figures" / "FigS8"
TABLES = BENCHMARK / "tables"
SRC = OUT / "source_data"
for directory in (OUT, SRC):
    directory.mkdir(parents=True, exist_ok=True)
b.TABLES, b.OUT, b.SRC = TABLES, OUT, SRC
b.CHECKS = {
    "figure": "FigS8",
    "sources": {},
    "figures": {},
    "text_bounds": {},
    "known_limitations": [],
}

TITLE = (
    "Supplementary Figure S8. Ancestry representation and prediction across all models"
)
MODEL_ORDER = [
    "hEvo-large",
    "hEvo-base",
    "evo2_7b_base",
    "evo2_1b_base",
    "NT_2.5b_1kG",
    "NTv3-650M-pre",
    "NT_2.5b_MS",
    "NT_500m_MS",
    "NTv3-100M-pre",
    "NT_500m_1kG",
    "NTv3-8M-pre",
    "GPN-Star-M",
    "GPN-Star-P",
    "GPN-Star-V",
    "GPN-promoter",
    "GPN-MSA",
    "DNABERT",
    "DNABERT2",
    "GENERator-1.2B",
    "GENA-LM-large",
    "GENERator-3B",
    "GENA-LM-base",
    "Mistral-DNA-422M",
    "AIDO.DNA-300M",
    "GROVER",
    "Genos-1.2B",
    "Mistral-DNA-17M",
    "SpeciesLM",
    "HyenaDNA_tiny_1k",
    "Caduceus-PS",
    "Caduceus-PH",
    "Mistral-DNA-138M",
]
EXCLUDED_MODELS = {"hEvo-open-large", "hEvo-open-base"}
FAMILY_MARK = {
    "Evo 2": "triangle",
    "NT": "square",
    "GPN": "diamond",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join the exact metrics displayed in FigS8 and retain the PDF roster."""
    s33, s34, s35 = b.load(33), b.load(34), b.load(35)
    for number, frame in ((33, s33), (34, s34)):
        if frame["model"].duplicated().any():
            raise ValueError(f"S{number} contains duplicate model rows")
    s35_60 = s35.loc[pd.to_numeric(s35["shot"], errors="raise").eq(60)].copy()
    if s35_60["model"].duplicated().any():
        raise ValueError("S35 has duplicate model rows for shot=60")

    frames = [s33.set_index("model"), s34.set_index("model"), s35_60.set_index("model")]
    model_sets = [set(frame.index) for frame in frames]
    if not all(models == model_sets[0] for models in model_sets[1:]):
        differences = {
            name: {
                "missing": sorted(model_sets[0] - models),
                "extra": sorted(models - model_sets[0]),
            }
            for name, models in zip(("S34", "S35 shot=60"), model_sets[1:])
            if models != model_sets[0]
        }
        raise ValueError(f"S33-S35 do not share the same model roster: {differences}")
    roster = model_sets[0]
    if set(MODEL_ORDER) | EXCLUDED_MODELS != roster:
        raise ValueError(
            "FigS8 model order does not match S33-S35: "
            f"missing={sorted(roster - set(MODEL_ORDER) - EXCLUDED_MODELS)}, "
            f"unaccounted={sorted((set(MODEL_ORDER) | EXCLUDED_MODELS) - roster)}"
        )
    if len(MODEL_ORDER) != 32:
        raise ValueError(
            f"Reference FigS8 roster should contain 32 models, got {len(MODEL_ORDER)}"
        )

    # Validate source display metadata across tables, then render through the
    # manuscript-wide canonical naming registry.
    source_display_names = s33.set_index("model")["model_display"]
    for frame, label, columns in (
        (frames[1], "S34", ["model_display"]),
        (frames[2], "S35", ["model_display"]),
    ):
        for column in columns:
            if (
                not frame[column]
                .reindex(MODEL_ORDER)
                .equals(source_display_names.reindex(MODEL_ORDER))
            ):
                raise ValueError(
                    f"Model display names disagree between S33 and {label}"
                )

    display_names = pd.Series(
        {model: b.display(model) for model in MODEL_ORDER}, name="model_display"
    )
    data = pd.DataFrame(index=pd.Index(MODEL_ORDER, name="model"))
    data["model_display"] = display_names.reindex(MODEL_ORDER)
    data["family"] = [b.FAMILY[model] for model in MODEL_ORDER]
    data["tsne_ch_index"] = pd.to_numeric(
        frames[0]["tsne_ch_index"].reindex(MODEL_ORDER), errors="raise"
    )
    data["tsne_silhouette"] = pd.to_numeric(
        frames[0]["tsne_silhouette"].reindex(MODEL_ORDER), errors="raise"
    )
    data["classification_mcc"] = pd.to_numeric(
        frames[1]["classification_mcc"].reindex(MODEL_ORDER), errors="raise"
    )
    data["pearson_r_mean"] = pd.to_numeric(
        frames[2]["pearson_r_mean"].reindex(MODEL_ORDER), errors="raise"
    )
    data["pearson_r_std"] = pd.to_numeric(
        frames[2]["pearson_r_std"].reindex(MODEL_ORDER), errors="raise"
    )
    data["few_shot_population_samples"] = 60

    numeric = [
        "tsne_ch_index",
        "tsne_silhouette",
        "classification_mcc",
        "pearson_r_mean",
        "pearson_r_std",
    ]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("S33-S35 contain non-finite values in a plotted FigS8 metric")
    if (data["tsne_ch_index"] < 0).any():
        raise ValueError("S33 Calinski-Harabasz values must be nonnegative")
    if not data["tsne_silhouette"].between(-1, 1).all():
        raise ValueError("S33 silhouette values must be within [-1, 1]")
    if not data["classification_mcc"].between(-1, 1).all():
        raise ValueError("S34 MCC values must be within [-1, 1]")
    if (
        not data["pearson_r_mean"].between(-1, 1).all()
        or (data["pearson_r_std"] < 0).any()
    ):
        raise ValueError(
            "S35 Pearson r or standard deviation is outside its valid range"
        )

    ranks = pd.DataFrame(index=data.index)
    for column in (
        "tsne_ch_index",
        "tsne_silhouette",
        "classification_mcc",
        "pearson_r_mean",
    ):
        ranks[column + "_rank"] = (
            data[column].rank(ascending=False, method="min").astype(int)
        )

    b.save(data.reset_index(), "S8_plotted_values")
    b.save(ranks.reset_index(), "S8_within_metric_ranks")
    order = data.reset_index()[["model", "model_display", "family"]].copy()
    order.insert(0, "row_order", np.arange(1, len(order) + 1))
    b.save(order, "S8_model_order")
    return data, ranks


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
) -> None:
    """Draw the light grid, bottom axis, ticks, and labels used in the reference."""
    page.line(x, y, x + width, y, 0.42, "#202020")
    for tick in ticks:
        xx = x + (tick - low) / (high - low) * width
        page.line(xx, top, xx, y, 0.22, "#E4E4E4")
        page.line(xx, y, xx, y + 0.9, 0.32, "#202020")
        page.text(xx, y + 3.7, tick_format(tick), 5.15, align="center")


def draw_metric(
    page,
    *,
    model: str,
    value: float,
    std: float,
    rank: int,
    row_y: float,
    x: float,
    width: float,
    low: float,
    high: float,
    value_x: float,
    label: str,
    show_std: bool = False,
    open_marker: bool = False,
) -> None:
    if not low <= value <= high:
        raise ValueError(
            f"{label} value is outside its displayed axis for {model}: {value}"
        )
    color = b.COL[b.FAMILY[model]]
    shape = FAMILY_MARK.get(b.FAMILY[model], "circle")
    transform = lambda number: x + (number - low) / (high - low) * width
    xx = transform(value)
    if show_std:
        lower, upper = value - std, value + std
        if lower < low or upper > high:
            raise ValueError(
                f"{label} mean±SD is clipped by the displayed axis for {model}"
            )
        xl, xh = transform(lower), transform(upper)
        page.line(xl, row_y, xh, row_y, 0.38, color)
        page.line(xl, row_y - 0.34, xl, row_y + 0.34, 0.35, color)
        page.line(xh, row_y - 0.34, xh, row_y + 0.34, 0.35, color)
    page.marker(xx, row_y, 0.56, color, open_=open_marker, shape=shape)

    if label == "ch":
        value_text = f"{value:,.1f}"
    else:
        value_text = f"{value:.3f}"
    page.text(value_x, row_y + 0.57, value_text, 5.25, bold=rank <= 3, align="right")
    if rank <= 3:
        page.text(value_x + 0.55, row_y + 0.10, rank, 4.15, color="#303030")


def render(data: pd.DataFrame, ranks: pd.DataFrame) -> Path:
    page = b.Page("FigureS8_ancestry_prediction_A4", TITLE)
    first_y, row_pitch, family_gap = 33.0, 3.02, 0.72
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
    plot_top = first_y - 1.2

    # Panel headings and metric labels.
    page.text(44.0, 13.0, "a", 8.6, True)
    page.text(84.7, 13.8, "Ancestry separation", 7.0, True, align="center")
    page.text(84.7, 17.4, "2D t-SNE", 7.0, True, align="center")
    page.text(129.5, 13.0, "b", 8.6, True)
    page.text(148.1, 13.8, "Five-group ancestry", 6.8, True, align="center")
    page.text(148.1, 17.4, "classification", 6.8, True, align="center")
    page.text(169.8, 13.0, "c", 8.6, True)
    page.text(185.0, 13.8, "Few-shot ancestry", 6.8, True, align="center")
    page.text(185.0, 17.4, "proportion prediction", 6.8, True, align="center")
    page.text(185.0, 22.0, "60 samples per population", 5.15, align="center")

    page.text(9.5, 26.0, "Model", 5.55, True)
    page.text(61.4, 22.6, "Calinski-Harabasz", 5.45, True, align="center")
    page.text(61.4, 26.0, "index", 5.45, True, align="center")
    page.text(105.3, 26.0, "Silhouette", 5.45, True, align="center")
    page.text(142.2, 26.0, "MCC", 5.45, True, align="center")
    page.text(183.0, 26.0, "Pearson r", 5.45, True, align="center")
    for left, right in ((44.0, 128.0), (132.0, 165.0), (172.0, 203.0)):
        page.line(left, 29.0, right, 29.0, 0.42, "#202020")

    # Shared family grouping, separators, and the two hEvo row highlights.
    previous_family = None
    for model in MODEL_ORDER:
        center_y = row_y[model]
        family = b.FAMILY[model]
        if model in b.FOCUS:
            page.rect(
                8.7,
                center_y - row_pitch * 0.49,
                194.3,
                row_pitch * 0.98,
                b.ROW_HIGHLIGHT,
            )
        if previous_family is not None and family != previous_family:
            separator_y = center_y - row_pitch * 0.5 - family_gap * 0.5
            page.line(8.7, separator_y, 203.0, separator_y, 0.30, "#969696")
        color = b.COL[family]
        page.marker(
            9.55,
            center_y,
            0.60 if model in b.FOCUS else 0.53,
            color,
            open_=model == "hEvo-base",
            shape=FAMILY_MARK.get(family, "circle"),
        )
        page.text(
            11.25,
            center_y + 0.60,
            data.at[model, "model_display"],
            5.35,
            model in b.FOCUS,
        )
        previous_family = family

    # Axis ranges and point columns follow the reference panel geometry.
    ch = dict(x=44.7, width=33.0, low=0.0, high=1800.0, value_x=90.0)
    sil = dict(x=96.0, width=18.5, low=-0.05, high=0.45, value_x=125.7)
    mcc = dict(x=132.2, width=19.3, low=0.92, high=0.99, value_x=163.0)
    pearson = dict(x=172.6, width=19.0, low=0.78, high=0.98, value_x=199.7)
    draw_axis(
        page,
        x=ch["x"],
        y=axis_y,
        width=ch["width"],
        low=ch["low"],
        high=ch["high"],
        ticks=(0, 900, 1800),
        top=plot_top,
        tick_format=lambda value: f"{value:.0f}",
    )
    draw_axis(
        page,
        x=sil["x"],
        y=axis_y,
        width=sil["width"],
        low=sil["low"],
        high=sil["high"],
        ticks=(0, 0.2, 0.4),
        top=plot_top,
        tick_format=lambda value: f"{value:.1f}".rstrip("0").rstrip("."),
    )
    draw_axis(
        page,
        x=mcc["x"],
        y=axis_y,
        width=mcc["width"],
        low=mcc["low"],
        high=mcc["high"],
        ticks=(0.94, 0.96, 0.98),
        top=plot_top,
        tick_format=lambda value: f"{value:.2f}",
    )
    draw_axis(
        page,
        x=pearson["x"],
        y=axis_y,
        width=pearson["width"],
        low=pearson["low"],
        high=pearson["high"],
        ticks=(0.80, 0.88, 0.96),
        top=plot_top,
        tick_format=lambda value: f"{value:.2f}".rstrip("0").rstrip("."),
    )

    rendered = []
    for model in MODEL_ORDER:
        values = data.loc[model]
        row = ranks.loc[model]
        for label, metric, axis in (
            ("ch", "tsne_ch_index", ch),
            ("silhouette", "tsne_silhouette", sil),
            ("mcc", "classification_mcc", mcc),
            ("pearson", "pearson_r_mean", pearson),
        ):
            mean = float(values[metric])
            sd = float(values["pearson_r_std"]) if label == "pearson" else 0.0
            metric_rank = int(row[metric + "_rank"])
            draw_metric(
                page,
                model=model,
                value=mean,
                std=sd,
                rank=metric_rank,
                row_y=row_y[model],
                x=axis["x"],
                width=axis["width"],
                low=axis["low"],
                high=axis["high"],
                value_x=axis["value_x"],
                label=label,
                show_std=label == "pearson",
                open_marker=model == "hEvo-base",
            )
            rendered.append(
                {
                    "model": model,
                    "model_display": values["model_display"],
                    "metric": metric,
                    "estimate": mean,
                    "standard_deviation": sd if label == "pearson" else None,
                    "rank": metric_rank,
                    "x_mm": axis["x"]
                    + (mean - axis["low"])
                    / (axis["high"] - axis["low"])
                    * axis["width"],
                    "row_y_mm": row_y[model],
                }
            )

    page.line(8.7, last_y + 2.1, 203.0, last_y + 2.1, 0.45)
    legend_y = axis_y + 13.5
    b.family_legend(page, legend_y)
    page.text(143.0, legend_y, "1, 2, 3   Within-metric rank", 5.55)
    pdf = page.finish()

    if len(rendered) != len(MODEL_ORDER) * 4:
        raise ValueError(
            f"Expected {len(MODEL_ORDER) * 4} plotted values, got {len(rendered)}"
        )
    b.save(pd.DataFrame(rendered), "S8_rendered_points")
    b.CHECKS["figures"]["S8"] = {
        "models": len(MODEL_ORDER),
        "excluded_from_reference_roster": sorted(EXCLUDED_MODELS),
        "metrics": [
            "tsne_ch_index",
            "tsne_silhouette",
            "classification_mcc",
            "pearson_r_mean",
        ],
        "panels": [
            "a Ancestry separation / 2D t-SNE",
            "b Five-group ancestry classification",
            "c Few-shot ancestry proportion prediction",
        ],
        "few_shot_samples_per_population": 60,
        "Pearson_r_interval": "mean ± source-table standard deviation",
        "plotted_values": len(rendered),
        "GPN-MSA_values": data.loc[
            "GPN-MSA",
            [
                "tsne_ch_index",
                "tsne_silhouette",
                "classification_mcc",
                "pearson_r_mean",
            ],
        ].to_dict(),
        "row_order": "the 32-model order shown in the supplied reference PDF",
        "source_tables": ["S33", "S34", "S35 (shot=60)"],
    }
    return pdf


def export(
    output_dir: Path | None = None, tables_dir: Path | None = None
) -> dict[str, str]:
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
    source_pdf = render(data, ranks)
    integrated = OUT / "A4_integrated.pdf"
    if source_pdf.resolve() != integrated.resolve():
        shutil.copyfile(source_pdf, integrated)
    reader = PdfReader(str(integrated))
    text = reader.pages[0].extract_text() or ""
    required = (
        "Ancestry separation",
        "Five-group ancestry",
        "Few-shot ancestry",
        "GPN-MSA",
        "445.8",
        "-0.025",
        "0.978",
        "0.929",
    )
    missing = [token for token in required if token not in text]
    if len(reader.pages) != 1 or missing:
        raise ValueError(
            f"FigS8 page/content check failed: pages={len(reader.pages)}, missing={missing}"
        )

    pdftoppm = shutil.which("pdftoppm")
    pdftocairo = shutil.which("pdftocairo")
    if not pdftoppm or not pdftocairo:
        raise FileNotFoundError(
            "Poppler pdftoppm and pdftocairo are required for PNG/SVG exports"
        )
    subprocess.run(
        [
            pdftoppm,
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            "-png",
            "-r",
            "600",
            str(integrated),
            str(OUT / "A4_integrated"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            pdftocairo,
            "-f",
            "1",
            "-l",
            "1",
            "-svg",
            str(integrated),
            str(OUT / "A4_integrated.svg"),
        ],
        check=True,
        capture_output=True,
    )
    b.CHECKS["output"] = {
        "pages": 1,
        "page_mm": [210, 297],
        "local_source_data": True,
        "models": len(MODEL_ORDER),
        "plotted_scores": len(MODEL_ORDER) * 4,
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
