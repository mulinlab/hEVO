#!/usr/bin/env python3
"""Build the genomic-element CH-index panel from S11–S16.

The three datasets are kept on independent raw-CH axes.  The representative
model list is selected using within-dataset t-SNE ranks, so the selection does
not average incomparable raw CH scales. Fig.2 retains the representatives
visible in the prior panel, including hEvo-large/base and unchanged selections
from other model families; FigS6 retains the full 34-model roster.
"""

from __future__ import annotations

# === Comment index ===
# [Role] Load the three embedding CH source and summary tables from S11–S16 for Fig. 2c.
# [Data] Each set contains 204/68 rows, 34 models, seeds 42/43/44, and PCA/t-SNE results; the tasks have 5/3/5 classes, respectively.
# [Plotting] Use a separate raw-CH x-axis for each dataset; Fig. 2c shows 18 representative models, with only hEvo-large and hEvo-base from the hEvo family.
# [Style change] Use black text and axes while retaining family colors for markers and error bars. Arial fonts and sizes are set in FONT_* constants; marker radii are 0.9/0.75.
# [Export] build_drawing returns a Drawing that can be embedded in the full page; S11–S16 and Fig. S6 retain results for all 34 models.

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics

from figures.scripts.common.hevo_model_style import canonical_model_name


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
TABLE_DIR = PROJECT_ROOT / "tables"

DATASETS = {
    "ENCODE_CCRE_V4": {
        "label": "ENCODE cCRE v4",
        "source": "S11_seed_level_representation_separation_for_encode_candidate_cis_regulatory_elements.csv",
        "summary": "S12_summary_of_representation_separation_for_encode_candidate_cis_regulatory_elements.csv",
        "n_rows": 5000,
        "n_classes": 5,
    },
    "GENCODE_V45": {
        "label": "GENCODE v45",
        "source": "S13_seed_level_representation_separation_for_gencode_gene_structure_classes.csv",
        "summary": "S14_summary_of_representation_separation_for_gencode_gene_structure_classes.csv",
        "n_rows": 3000,
        "n_classes": 3,
    },
        "UCSC_REFSEQ_CURATED": {
        "label": "UCSC RefSeq",
        "source": "S15_seed_level_representation_separation_for_curated_refseq_gene_structure_classes.csv",
        "summary": "S16_summary_of_representation_separation_for_curated_refseq_gene_structure_classes.csv",
        "n_rows": 5000,
        "n_classes": 5,
    },
}

GROUP_CANDIDATES = {
    "hEvo-large": ["hEvo-large"],
    "hEvo-base": ["hEvo-base"],
    "GENA-LM": ["GENA-LM-base", "GENA-LM-large"],
    "GPN-Promoter": ["GPN-promoter"],
    "NT v3": ["NTv3-8M-pre", "NTv3-100M-pre", "NTv3-650M-pre"],
    "Evo 2": ["evo2_1b_base", "evo2_7b_base"],
    "Caduceus-Ph": ["Caduceus-PH", "Caduceus-PS"],
    "GENERator": ["GENERator-1.2B", "GENERator-3B"],
    "NTv2-MS": ["NT_500m_MS"],
    "GROVER": ["GROVER"],
    "HyenaDNA": ["HyenaDNA_tiny_1k"],
    "AIDO.DNA": ["AIDO.DNA-300M"],
    "Mistral-DNA": ["Mistral-DNA-17M", "Mistral-DNA-138M", "Mistral-DNA-422M"],
    "Genos": ["Genos-1.2B"],
    "SpeciesLM": ["SpeciesLM"],
    "DNABERT": ["DNABERT"],
    "DNABERT-2": ["DNABERT2"],
}

# Freeze the representatives visible in the pre-replacement Fig.2c PDF. An old
# code comment claimed 18 rows, but the released panel contains 17 unique labels.
FIG2_FIXED_REPRESENTATIVES = {
    "hEvo-large": "hEvo-large",
    "hEvo-base": "hEvo-base",
    "GENA-LM": "GENA-LM-base",
    "GPN-Promoter": "GPN-promoter",
    "NT v3": "NTv3-8M-pre",
    "Evo 2": "evo2_7b_base",
    "Caduceus-Ph": "Caduceus-PS",
    "GENERator": "GENERator-1.2B",
    "NTv2-MS": "NT_500m_MS",
    "GROVER": "GROVER",
    "HyenaDNA": "HyenaDNA_tiny_1k",
    "AIDO.DNA": "AIDO.DNA-300M",
    "Mistral-DNA": "Mistral-DNA-422M",
    "Genos": "Genos-1.2B",
    "SpeciesLM": "SpeciesLM",
    "DNABERT": "DNABERT",
    "DNABERT-2": "DNABERT2",
}

FAMILY_COLORS = {
    "hEVO": "#B35C1E",
    "NT": "#65717F",
    "Evo 2": "#0E7C86",
    "GPN": "#0B2E83",
    "Other": "#9AA3AF",
}

FAMILY_MODEL_ORDER = {
    "hEVO": [
        "hEvo-large",
        "hEvo-base",
        "hEvo-1kG-sliding128-86M",
        "hEvo-fullUKB-86M",
        "hEvo-60k-86M",
        "hEvo-1kG-86M",
    ],
    "NT": [
        "NTv3-100M-pre",
        "NTv3-8M-pre",
        "NTv3-650M-pre",
        "NT_500m_MS",
        "NT_2.5b_1kG",
        "NT_500m_1kG",
        "NT_2.5b_MS",
    ],
    "Evo 2": ["evo2_7b_base", "evo2_1b_base"],
    "GPN": ["GPN-promoter"],
}

METHOD_STYLES = {"PCA": False, "tSNE": True}
BLACK = "#000000"
GREY = "#000000"
LIGHT = "#000000"
WHITE = "#FFFFFF"

FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_SIZE_PANEL = 14.0
FONT_SIZE_TITLE = 10.0
FONT_SIZE_PANEL_TITLE = 7.2
FONT_SIZE_DATASET = 6.3
FONT_SIZE_LABEL = 7.0
FONT_SIZE_FAMILY = 4.2
FONT_SIZE_LEGEND = 6.8
FONT_SIZE_AXIS = 6.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_family(model: str) -> str:
    value = model.lower()
    if value.startswith("hevo-"):
        return "hEVO"
    if value.startswith("nt"):
        return "NT"
    if value.startswith("evo2") or value.startswith("evo 2"):
        return "Evo 2"
    if value.startswith("gpn-"):
        return "GPN"
    return "Other"


def display_name(model: str) -> str:
    return canonical_model_name(model)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    summaries: list[pd.DataFrame] = []
    validations: dict[str, object] = {}
    model_sets: list[set[str]] = []
    for dataset, config in DATASETS.items():
        source_path = TABLE_DIR / config["source"]
        summary_path = TABLE_DIR / config["summary"]
        source = pd.read_csv(source_path)
        summary = pd.read_csv(summary_path)
        required_source = {
            "model_name",
            "seed",
            "method",
            "n_rows",
            "n_classes",
            "ch_index",
        }
        required_summary = {
            "model_name",
            "method",
            "n_seeds",
            "n_rows",
            "ch_index_mean",
            "ch_index_sd",
        }
        if not required_source.issubset(source.columns):
            raise ValueError(f"{dataset}: source columns are incomplete")
        if not required_summary.issubset(summary.columns):
            raise ValueError(f"{dataset}: summary columns are incomplete")
        if len(source) != 204 or len(summary) != 68:
            raise ValueError(
                f"{dataset}: expected 204/68 rows, got {len(source)}/{len(summary)}"
            )
        if set(source["seed"].unique()) != {42, 43, 44}:
            raise ValueError(f"{dataset}: seed set is not 42/43/44")
        if set(source["method"].unique()) != {"PCA", "tSNE"}:
            raise ValueError(f"{dataset}: source method set is invalid")
        if set(summary["method"].unique()) != {"PCA", "tSNE"}:
            raise ValueError(f"{dataset}: summary method set is invalid")
        if set(source["n_rows"].unique()) != {config["n_rows"]} or set(
            source["n_classes"].unique()
        ) != {config["n_classes"]}:
            raise ValueError(f"{dataset}: n_rows/n_classes contract failed")
        if source["ch_index"].isna().any() or summary["ch_index_mean"].isna().any():
            raise ValueError(f"{dataset}: CH index contains missing values")
        if set(summary["n_seeds"].unique()) != {3}:
            raise ValueError(f"{dataset}: summary seed count is not 3")
        models = set(source["model_name"].astype(str))
        model_sets.append(models)
        summary = summary.copy()
        summary["dataset"] = dataset
        summary["dataset_label"] = config["label"]
        summaries.append(summary)
        validations[dataset] = {
            "source_rows": len(source),
            "summary_rows": len(summary),
            "models": len(models),
            "seeds": sorted(int(x) for x in source["seed"].unique()),
            "methods": sorted(source["method"].unique().tolist()),
            "n_rows": sorted(int(x) for x in source["n_rows"].unique()),
            "n_classes": sorted(int(x) for x in source["n_classes"].unique()),
            "source_sha256": sha256(source_path),
            "summary_sha256": sha256(summary_path),
        }
    shared_models = set.intersection(*model_sets)
    model_count = len(shared_models)
    if model_count < 1 or any(len(models) != model_count for models in model_sets):
        counts = [len(models) for models in model_sets]
        raise ValueError(
            "The three S11-S16 datasets do not share one model set: "
            f"dataset counts={counts}, shared={model_count}"
        )
    combined = pd.concat(summaries, ignore_index=True)
    combined["rank"] = combined.groupby(["dataset", "method"])["ch_index_mean"].rank(
        ascending=False, method="min"
    )
    combined["rank_pct"] = 1.0 - (combined["rank"] - 1.0) / max(model_count - 1.0, 1.0)
    combined["family"] = combined["model_name"].map(model_family)
    combined["display_label"] = combined["model_name"].map(display_name)

    selected_rows: list[dict[str, object]] = []
    available = set(combined["model_name"])
    for group, candidates in GROUP_CANDIDATES.items():
        candidates = [candidate for candidate in candidates if candidate in available]
        if not candidates:
            raise ValueError(f"No current model found for representative group {group}")
        preferred = FIG2_FIXED_REPRESENTATIVES[group]
        if preferred not in candidates:
            raise ValueError(
                f"Frozen Fig.2 representative {preferred} for {group} is unavailable; "
                "do not silently substitute another model"
            )
        scores = (
            combined[
                (combined["model_name"] == preferred)
                & (combined["method"] == "tSNE")
            ]
            .groupby("model_name", as_index=False)
            .agg(
                mean_tsne_rank_pct=("rank_pct", "mean"),
                mean_tsne_ch=("ch_index_mean", "mean"),
            )
        )
        scores["group"] = group
        scores = scores.sort_values(
            ["mean_tsne_rank_pct", "mean_tsne_ch", "model_name"],
            ascending=[False, False, True],
            kind="stable",
        )
        winner = scores.iloc[0]
        selected_rows.append(
            {
                "algorithm_group": group,
                "selected_model": winner["model_name"],
                "display_label": display_name(str(winner["model_name"])),
                "mean_tsne_rank_pct": float(winner["mean_tsne_rank_pct"]),
                "mean_tsne_ch": float(winner["mean_tsne_ch"]),
                "candidate_models": ";".join(candidates),
            }
        )
    summary = pd.DataFrame(selected_rows)
    order = summary.sort_values(
        ["mean_tsne_rank_pct", "mean_tsne_ch", "algorithm_group"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    order["representative_rank"] = np.arange(1, len(order) + 1)
    selected = combined.merge(
        order[["algorithm_group", "selected_model", "representative_rank"]],
        left_on="model_name",
        right_on="selected_model",
        how="inner",
    )
    selected = selected.sort_values(
        ["representative_rank", "dataset", "method"], kind="stable"
    )
    return (
        combined,
        selected,
        {
            "datasets": validations,
            "model_count": model_count,
            "selection_summary": order.to_dict(orient="records"),
        },
    )


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
    requested = FONT_BOLD if bold else FONT_REGULAR
    fallback = "Arial-Bold" if bold else "Arial"
    font_name = (
        requested if requested in pdfmetrics.getRegisteredFontNames() else fallback
    )
    drawing.add(
        String(
            x * mm,
            y * mm,
            text,
            fontName=font_name,
            fontSize=size,
            fillColor=HexColor(color),
            textAnchor=anchor,
        )
    )


def family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, FAMILY_COLORS["Other"])


def map_x(value: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (float(value) - lo) / max(hi - lo, 1e-9) * (x1 - x0)


def draw_method_point(
    drawing: Drawing, x: float, y: float, color: str, filled: bool, radius: float = 0.9
) -> None:
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            radius * mm,
            fillColor=HexColor(color if filled else WHITE),
            strokeColor=HexColor(color),
            strokeWidth=0.5,
        )
    )


def draw_family_point(
    drawing: Drawing, x: float, y: float, color: str, family: str, *, filled: bool, radius: float
) -> None:
    """Draw Fig.2c points with the reference's family-specific marker shapes."""

    shape = {"Evo 2": "triangle", "NT": "square", "GPN": "diamond"}.get(family, "circle")
    fill = HexColor(color if filled else WHITE)
    stroke = HexColor(color)
    if shape == "circle":
        draw_method_point(drawing, x, y, color, filled, radius=radius)
    elif shape == "square":
        drawing.add(
            Rect(
                (x - radius) * mm,
                (y - radius) * mm,
                2 * radius * mm,
                2 * radius * mm,
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=0.5,
            )
        )
    elif shape == "diamond":
        drawing.add(
            Polygon(
                [x * mm, (y + radius) * mm,
                 (x + radius) * mm, y * mm,
                 x * mm, (y - radius) * mm,
                 (x - radius) * mm, y * mm],
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=0.5,
            )
        )
    else:
        drawing.add(
            Polygon(
                [(x - radius) * mm, (y - radius * 0.65) * mm,
                 (x + radius) * mm, (y - radius * 0.65) * mm,
                 x * mm, (y + radius) * mm],
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=0.5,
            )
        )


def draw_shared_legend(drawing: Drawing, x: float, y: float) -> None:
    """在 b 面板顶部绘制供 a/b 共用的 marker 与模型家族图例。"""

    cursor = x
    method_items = (
        ("PCA", False, FAMILY_COLORS["Other"], 9.5),
        ("t-SNE", True, FAMILY_COLORS["Other"], 13.0),
    )
    family_items = (
        ("hEVO", "hEVO", 11.0),
        ("NT", "NT", 7.5),
        ("Evo 2", "Evo 2", 11.0),
        ("GPN", "GPN", 9.5),
        ("Other", "Other", 11.0),
    )
    for label, filled, color, step in method_items:
        draw_method_point(drawing, cursor, y, color, filled, radius=0.72)
        add_text(drawing, cursor + 1.6, y - 0.7, label, FONT_SIZE_LEGEND, color=BLACK)
        cursor += step
    for label, family, step in family_items:
        draw_method_point(drawing, cursor, y, family_color(family), True, radius=0.72)
        add_text(drawing, cursor + 1.6, y - 0.7, label, FONT_SIZE_LEGEND, color=BLACK)
        cursor += step


def format_axis_tick(value: float) -> str:
    """Use compact labels so the three narrow dataset facets remain balanced."""

    if abs(value) < 0.5:
        return "0"
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:g}k"
    return f"{value:g}"


def draw_axes(
    drawing: Drawing,
    x0: float,
    x1: float,
    y: float,
    maximum: float,
    *,
    ticks: tuple[float, ...] = (),
    comma_ticks: bool = False,
) -> None:
    drawing.add(
        Line(
            x0 * mm,
            y * mm,
            x1 * mm,
            y * mm,
            strokeColor=HexColor(GREY),
            strokeWidth=0.45,
        )
    )
    for tick in ticks:
        if tick > maximum:
            continue
        x = map_x(tick, 0, maximum, x0, x1)
        drawing.add(
            Line(
                x * mm,
                (y - 1.2) * mm,
                x * mm,
                (y + 1.2) * mm,
                strokeColor=HexColor(GREY),
                strokeWidth=0.4,
            )
        )
        label = (
            f"{int(tick):,}"
            if comma_ticks and float(tick).is_integer()
            else format_axis_tick(float(tick))
        )
        add_text(
            drawing, x, y - 3.0, label, FONT_SIZE_AXIS, anchor="middle", color=GREY
        )


def draw_panel_a(
    drawing: Drawing,
    data: pd.DataFrame,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    panel_letter: str = "a",
    standalone_legend: bool = False,
    model_label_size: float | None = None,
    model_order: tuple[str, ...] | None = None,
) -> None:
    """Draw the Fig.2c reference-style PCA CH lollipop comparison.

    Values come from the active S12/S14/S16 PCA summaries. Displayed order is
    the product of each model's within-facet PCA rank, as in the reference PDF.
    """
    # Fig.2c geometry is calibrated against the supplied A4 reference PDF:
    # 190 mm wide, compact vertical title/rank header, and three equal-width
    # plotting lanes that span nearly the full panel width.
    add_text(drawing, x + 1.3, y + height - 3.6, panel_letter, FONT_SIZE_PANEL, bold=True)
    add_text(
        drawing,
        x + width - 1.4,
        y + height - 2.7,
        "Ordered by rank product across datasets",
        5.3,
        anchor="end",
    )
    add_text(drawing, x + 5.0, y + height - 6.2, "RP rank", 5.3, bold=True)

    pca = data.loc[data["method"].eq("PCA")].copy()
    if pca.empty or pca[["model_name", "dataset", "ch_index_mean"]].isna().any().any():
        raise ValueError("Fig.2c PCA values must be complete for every selected model and dataset")
    expected_facets = set(DATASETS)
    facet_counts = pca.groupby("model_name")["dataset"].agg(lambda values: set(values))
    if not facet_counts.map(lambda values: values == expected_facets).all():
        raise ValueError("Fig.2c selected models must have PCA values for all three datasets")

    pca["facet_rank"] = pca.groupby("dataset")["ch_index_mean"].rank(
        ascending=False, method="min"
    )
    ranks = pca.pivot(index="model_name", columns="dataset", values="facet_rank")
    values = pca.pivot(index="model_name", columns="dataset", values="ch_index_mean")
    meta = (
        pca[["model_name", "display_label", "family"]]
        .drop_duplicates("model_name")
        .set_index("model_name")
    )
    ranked_models = (
        pd.DataFrame(
            {
                "rank_product": ranks.prod(axis=1),
                "mean_rank": ranks.mean(axis=1),
                "display_label": meta["display_label"],
            }
        )
        .sort_values(["rank_product", "mean_rank", "display_label"], kind="stable")
        .index.tolist()
    )
    if model_order is None:
        ordered_models = ranked_models
    else:
        if set(model_order) != set(ranked_models):
            raise ValueError("Fig.2c fixed model order must contain each displayed model exactly once")
        ordered_models = list(model_order)

    facet_specs = (
        ("ENCODE_CCRE_V4", 30.0, 210.0, (40.0, 80.0, 160.0), 39.0, 72.9, 85.2),
        ("GENCODE_V45", 10.0, 2500.0, (10.0, 100.0, 1000.0), 90.0, 124.0, 136.2),
        ("UCSC_REFSEQ_CURATED", 35.0, 1600.0, (50.0, 200.0, 1000.0), 141.0, 174.9, 187.2),
    )
    row_top = y + height - 9.0
    row_bottom = y + 8.0
    row_ys = np.linspace(row_top, row_bottom, len(ordered_models))
    axis_y = y + 6.1
    label_size = 6.0 if model_label_size is None else model_label_size

    for rank, model in enumerate(ordered_models, start=1):
        yy = float(row_ys[rank - 1])
        if model in {"hEvo-large", "hEvo-base"}:
            drawing.add(
                Rect(
                    (x + 1.7) * mm,
                    (yy - 0.91) * mm,
                    (width - 3.2) * mm,
                    1.82 * mm,
                    fillColor=HexColor("#FCF2E7"),
                    strokeColor=None,
                )
            )
        add_text(drawing, x + 7.5, yy - 0.63, str(rank), 5.3, anchor="middle")
        add_text(
            drawing,
            x + 10.0,
            yy - 0.61,
            str(meta.loc[model, "display_label"]),
            label_size,
            bold=model in {"hEvo-large", "hEvo-base"},
        )

    for dataset, minimum, maximum, ticks, x0_offset, x1_offset, label_x_offset in facet_specs:
        x0, x1 = x + x0_offset, x + x1_offset
        add_text(
            drawing,
            (x0 + x1) / 2.0 + 7.05,
            y + height - 6.17,
            DATASETS[dataset]["label"],
            6.3,
            bold=True,
            anchor="middle",
        )
        header_rule_y = y + height - 7.3
        drawing.add(
            Line(
                x0 * mm,
                header_rule_y * mm,
                x1 * mm,
                header_rule_y * mm,
                strokeColor=HexColor(BLACK),
                strokeWidth=0.4,
            )
        )
        for rank, model in enumerate(ordered_models, start=1):
            yy = float(row_ys[rank - 1])
            value = float(values.loc[model, dataset])
            if value <= 0:
                raise ValueError(f"Fig.2c {model}/{dataset} value {value} is not positive for its log axis")
            px = x0 + (np.log10(value) - np.log10(minimum)) / (np.log10(maximum) - np.log10(minimum)) * (x1 - x0)
            family = str(meta.loc[model, "family"])
            color = family_color(family)
            draw_family_point(
                drawing,
                px,
                yy,
                color,
                family,
                filled=model != "hEvo-base",
                radius=0.67,
            )
            add_text(
                drawing,
                x + label_x_offset + 0.5,
                yy - 0.61,
                f"{value:,.1f}",
                5.75,
                anchor="end",
                bold=model in {"hEvo-large", "hEvo-base"},
                color=color,
            )
        drawing.add(
            Line(
                x0 * mm,
                axis_y * mm,
                x1 * mm,
                axis_y * mm,
                strokeColor=HexColor(BLACK),
                strokeWidth=0.45,
            )
        )
        for tick in ticks:
            tick_x = x0 + (np.log10(tick) - np.log10(minimum)) / (np.log10(maximum) - np.log10(minimum)) * (x1 - x0)
            drawing.add(
                Line(
                    tick_x * mm,
                    axis_y * mm,
                    tick_x * mm,
                    (axis_y - 0.8) * mm,
                    strokeColor=HexColor(BLACK),
                    strokeWidth=0.4,
                )
            )
            tick_label = f"{int(tick):,}" if tick >= 1000 else f"{int(tick)}"
            add_text(drawing, tick_x, y + 3.13, tick_label, 5.3, anchor="middle")

    add_text(
        drawing,
        x + 113.0,
        y + 0.28,
        "Class separation (2D PCA, CH index)",
        6.0,
        anchor="middle",
    )


def draw_panel_b(
    drawing: Drawing,
    data: pd.DataFrame,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    add_text(drawing, x, y + height - 1.5, "b", FONT_SIZE_PANEL, bold=True)
    add_text(
        drawing,
        x + 7,
        y + height - 1.5,
        "Family members: model-level CH",
        FONT_SIZE_TITLE,
        bold=True,
    )
    draw_shared_legend(drawing, x + 0.8, y + height - 4.0)
    families = ["hEVO", "NT", "Evo 2", "GPN"]
    family_models = {
        family: [
            model
            for model in FAMILY_MODEL_ORDER[family]
            if model in set(data["model_name"])
        ]
        for family in families
    }
    n_rows = sum(len(models) for models in family_models.values())
    n_slots = n_rows + len(families)
    chart_y_top = y + height - 11.5
    chart_y_bottom = y + 7.0
    row_step = (chart_y_top - chart_y_bottom) / max(n_slots - 1, 1)
    gap = 2.2
    facet_w = (width - gap * 2) / 3
    for j, dataset in enumerate(DATASETS):
        fx = x + j * (facet_w + gap)
        subset = data[data["dataset"] == dataset]
        maximum = float(np.ceil(subset["ch_index_mean"].max() / 500.0) * 500.0)
        label_w = 15.0 if j == 0 else 1.0
        x0, x1 = fx + label_w, fx + facet_w - 1.5
        add_text(
            drawing,
            (x0 + x1) / 2,
            chart_y_top + 4.0,
            DATASETS[dataset]["label"],
            FONT_SIZE_DATASET,
            bold=True,
            anchor="middle",
        )
        row_index = 0
        for family in families:
            models = family_models[family]
            if not models:
                continue
            header_y = chart_y_top - row_index * row_step
            add_text(
                drawing,
                (x0 + x1) / 2,
                header_y,
                family,
                FONT_SIZE_FAMILY,
                bold=True,
                anchor="middle",
                color=BLACK,
            )
            row_index += 1
            for model in models:
                yy = chart_y_top - row_index * row_step
                if j == 0:
                    add_text(
                        drawing,
                        fx,
                        yy - 1.0,
                        display_name(model),
                        FONT_SIZE_LABEL,
                        color=BLACK,
                    )
                model_rows = subset[subset["model_name"] == model]
                points = {}
                for method in ("PCA", "tSNE"):
                    current = model_rows[model_rows["method"] == method]
                    if current.empty:
                        continue
                    value = float(current.iloc[0]["ch_index_mean"])
                    sd = float(current.iloc[0]["ch_index_sd"])
                    color = family_color(family)
                    px = map_x(value, 0, maximum, x0, x1)
                    lo = map_x(max(0.0, value - sd), 0, maximum, x0, x1)
                    hi = map_x(value + sd, 0, maximum, x0, x1)
                    drawing.add(
                        Line(
                            lo * mm,
                            yy * mm,
                            hi * mm,
                            yy * mm,
                            strokeColor=HexColor(color),
                            strokeWidth=0.4,
                        )
                    )
                    draw_method_point(
                        drawing, px, yy, color, METHOD_STYLES[method], radius=0.75
                    )
                    points[method] = px
                if set(points) == {"PCA", "tSNE"}:
                    drawing.add(
                        Line(
                            min(points.values()) * mm,
                            yy * mm,
                            max(points.values()) * mm,
                            yy * mm,
                            strokeColor=HexColor("#9AA3AF"),
                            strokeWidth=0.55,
                        )
                    )
                row_index += 1
        draw_axes(
            drawing,
            x0,
            x1,
            chart_y_bottom - 1.0,
            maximum,
            ticks=(0.0, maximum / 2.0, maximum),
        )
    add_text(
        drawing,
        x + width / 2,
        y + 0.3,
        "Calinski–Harabasz index (dataset-specific raw scale)",
        FONT_SIZE_AXIS,
        anchor="middle",
        color=BLACK,
    )


def build_drawing(combined: pd.DataFrame, selected: pd.DataFrame) -> Drawing:
    drawing = Drawing(190 * mm, 76 * mm)
    drawing.add(
        Rect(0, 0, 190 * mm, 76 * mm, fillColor=HexColor(WHITE), strokeColor=None)
    )
    draw_panel_a(drawing, selected, 0, 0, 93, 76)
    draw_panel_b(drawing, combined, 97, 0, 93, 76)
    return drawing


def write_sources(
    output_dir: Path,
    combined: pd.DataFrame,
    selected: pd.DataFrame,
    metadata: dict[str, object],
) -> dict[str, object]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    combined_path = source_dir / "Fig2ab_S12_S16_all_model_summary.csv"
    selected_path = source_dir / "Fig2ab_S12_S16_representative_model_summary.csv"
    family_path = source_dir / "Fig2ab_S12_S16_family_summary.csv"
    summary_path = source_dir / "Fig2ab_S12_S16_selection_summary.csv"
    combined.to_csv(combined_path, index=False)
    selected.to_csv(selected_path, index=False)
    family = combined.groupby(["dataset", "family", "method"], as_index=False).agg(
        n_models=("model_name", "nunique"),
        ch_q1=("ch_index_mean", lambda x: float(np.quantile(x, 0.25))),
        ch_median=("ch_index_mean", "median"),
        ch_q3=("ch_index_mean", lambda x: float(np.quantile(x, 0.75))),
    )
    family.to_csv(family_path, index=False)
    pd.DataFrame(metadata["selection_summary"]).to_csv(summary_path, index=False)
    return {
        "all_model_summary": str(combined_path.resolve()),
        "representative_summary": str(selected_path.resolve()),
        "family_summary": str(family_path.resolve()),
        "selection_summary": str(summary_path.resolve()),
        "hashes": {
            name: sha256(Path(path))
            for name, path in {
                "all_model_summary": combined_path,
                "representative_summary": selected_path,
                "family_summary": family_path,
                "selection_summary": summary_path,
            }.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "formal" / "Fig2" / "fig2ab_s11_s16",
    )
    parser.add_argument("--stem", default="Fig2ab_S11_S16_DATA_QC_DRAFT")
    args = parser.parse_args()
    combined, selected, metadata = load_data()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    drawing = build_drawing(combined, selected)
    from reportlab.graphics import renderPDF, renderSVG

    pdf = args.output_dir / f"{args.stem}.pdf"
    svg = args.output_dir / f"{args.stem}.svg"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    source_info = write_sources(args.output_dir, combined, selected, metadata)
    qa = {
        "status": "DATA_QC_DRAFT",
        "legacy_input_used": False,
        "datasets": metadata["datasets"],
        "models_in_source": int(combined["model_name"].nunique()),
        "representative_models": 18,
        "methods": ["PCA", "tSNE"],
        "metric": "Calinski-Harabasz index",
        "selection_rule": "one representative per legacy algorithm group; highest mean within-dataset t-SNE rank percentile, then mean raw t-SNE CH, then raw model ID",
        "raw_ch_axes": "independent per dataset; no cross-dataset raw CH averaging",
        "outputs": {"pdf": str(pdf.resolve()), "svg": str(svg.resolve())},
        "source_data": source_info,
        "script_sha256": sha256(HERE),
    }


if __name__ == "__main__":
    main()
