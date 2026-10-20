#!/usr/bin/env python3
"""Render one best scale per distinct DNA language-model algorithm.

Only parameter-scale variants are collapsed.  Architectures, generations,
pretraining corpora/objectives and GPN targets remain separate algorithms.
The representative is selected by the highest t-SNE Calinski-Harabasz index.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect
from reportlab.lib.units import mm


HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import plot_figS2a_ucsc_element_separability as base  # noqa: E402


WIDTH_MM = 180.0
HEIGHT_MM = 76.0
AXIS_MIN = 0.0
AXIS_MAX = 5000.0


@dataclass(frozen=True)
class AlgorithmRule:
    group: str
    candidates: tuple[str, ...]
    display_template: str


# Corpus/objective variants remain separate.  hEVO corpus ablations are not
# independent algorithms and are excluded; only the base/large main pair enters.
RULES = (
    # Both main hEVO scales are shown by author request; corpus/stride ablations
    # remain excluded from the cross-algorithm comparison.
    AlgorithmRule("hEvo-large", ("hEvo-large",), "hEvo-large"),
    AlgorithmRule("hEvo-base", ("hEvo-base",), "hEvo-base"),
    AlgorithmRule("GENA-LM", ("GENA-LM-base-110M", "GENA-LM-large-336M"), "{model}"),
    AlgorithmRule("GPN-Promoter", ("GPN-Promoter",), "GPN-Promoter"),
    AlgorithmRule("GPN-MSA", ("GPN-MSA",), "GPN-MSA"),
    AlgorithmRule(
        "NT v3",
        ("NTv3-8M", "NTv3-100M", "NTv3-650M"),
        "{model}",
    ),
    AlgorithmRule("Evo 2", ("EVO 2-1B", "EVO 2-7B"), "{model}"),
    AlgorithmRule("Caduceus-PH", ("Caduceus-PH",), "Caduceus-PH"),
    AlgorithmRule(
        "GENERator",
        ("GENERator-1.2B", "GENERator-3B"),
        "{model}",
    ),
    # Main Figure 2a keeps the NT-v2 500M checkpoint; the NT-v1 2.5B
    # multispecies checkpoint remains available in complete supplementary views.
    AlgorithmRule("NTv2-MS", ("NTv2-MS-500M",), "{model}"),
    AlgorithmRule("GROVER", ("GROVER",), "GROVER"),
    AlgorithmRule("HyenaDNA", ("HyenaDNA-1K",), "HyenaDNA-1K"),
    AlgorithmRule("AIDO.DNA", ("AIDO.DNA-300M",), "AIDO.DNA-300M"),
    AlgorithmRule(
        "Mistral-DNA",
        ("Mistral-DNA-17M", "Mistral-DNA-138M", "Mistral-DNA-422M"),
        "{model}",
    ),
    AlgorithmRule("Genos", ("Genos-1.2B",), "Genos-1.2B"),
    AlgorithmRule("SpeciesLM", ("SpeciesLM-Metazoa",), "SpeciesLM-Metazoa"),
    AlgorithmRule("DNABERT", ("DNABERT-S",), "DNABERT-S"),
    AlgorithmRule("DNABERT-2", ("DNABERT-2-117M",), "DNABERT-2-117M"),
)


def x_map(value: float, x0: float, x1: float) -> float:
    clipped = min(max(float(value), AXIS_MIN), AXIS_MAX)
    return x0 + clipped / AXIS_MAX * (x1 - x0)


def selected_scale(model: str, group: str) -> str:
    if group == "hEVO":
        return model.removeprefix("hEVO-")
    if group == "GENA-LM":
        return model.removeprefix("GENA-LM-")
    if group == "NT v3":
        return model.removeprefix("NTv3-").removesuffix("-pre")
    if group == "Evo 2":
        return model.removeprefix("Evo 2-")
    if group == "GENERator":
        return model.removeprefix("GENERator-")
    if group in {"NTv1-MS", "NTv2-MS"}:
        return model.split("-", 1)[-1]
    if group == "NT-1KGP":
        return model.removeprefix("NT-").removesuffix("-1KGP")
    if group == "Mistral-DNA":
        return model.removeprefix("Mistral-DNA-")
    return model


def select_representatives(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = set(summary["model_name"].unique())
    selected_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for rule in RULES:
        missing = set(rule.candidates) - available
        if missing:
            raise ValueError(f"Missing candidates for {rule.group}: {sorted(missing)}")
        candidates = summary[summary["model_name"].isin(rule.candidates)].copy()
        tsne = candidates[candidates["method"] == "t-SNE"].sort_values(
            ["ch_index_median", "model_name"], ascending=[False, True], kind="stable"
        )
        winner = str(tsne.iloc[0]["model_name"])
        scale = selected_scale(winner, rule.group)
        label = rule.display_template.format(scale=scale, model=winner)
        winner_rows = candidates[candidates["model_name"] == winner]
        for row in winner_rows.to_dict("records"):
            selected_rows.append(
                {
                    **row,
                    "algorithm_group": rule.group,
                    "display_label": label,
                    "selected_model": winner,
                    "candidate_count": len(rule.candidates),
                }
            )
        tsne_lookup = dict(
            zip(tsne["model_name"].astype(str), tsne["ch_index_median"].astype(float))
        )
        for candidate in rule.candidates:
            summary_rows.append(
                {
                    "algorithm_group": rule.group,
                    "candidate_model": candidate,
                    "candidate_count": len(rule.candidates),
                    "tsne_ch_index": tsne_lookup[candidate],
                    "selected": candidate == winner,
                    "selection_rule": "highest t-SNE CH within parameter-scale variants",
                }
            )

    selected = pd.DataFrame(selected_rows)
    summary = pd.DataFrame(summary_rows)
    ranks = (
        selected[selected["method"] == "t-SNE"]
        .sort_values(["ch_index_median", "display_label"], ascending=[False, True], kind="stable")
        [["algorithm_group"]]
        .reset_index(drop=True)
    )
    ranks["representative_rank"] = np.arange(1, len(ranks) + 1)
    selected = selected.merge(ranks, on="algorithm_group", how="left")
    return selected, summary


def build_drawing(
    table: pd.DataFrame,
) -> tuple[Drawing, pd.DataFrame, pd.DataFrame, list[str]]:
    base.configure_fonts()
    summary = base.summarize(table)
    selected, summary = select_representatives(summary)
    if float(selected["ch_index_q3"].max()) > AXIS_MAX:
        raise ValueError("Observed CH index exceeds the locked 0-5,000 comparison scale")
    order = (
        selected[["algorithm_group", "representative_rank"]]
        .drop_duplicates()
        .sort_values("representative_rank", kind="stable")["algorithm_group"]
        .tolist()
    )

    drawing = Drawing(WIDTH_MM * mm, HEIGHT_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            WIDTH_MM * mm,
            HEIGHT_MM * mm,
            fillColor=base.color(base.WHITE),
            strokeColor=None,
        )
    )
    base.add_text(drawing, 2.0, 70.0, "a", 14.0, bold=True)
    base.add_text(
        drawing,
        176.0,
        71.0,
        "Representative models | five UCSC classes",
        6.2,
        anchor="end",
    )
    base.draw_open_point(drawing, 72.0, 67.0, base.BLACK)
    base.add_text(drawing, 74.2, 66.3, "PCA", 6.1)
    base.draw_filled_point(drawing, 88.0, 67.0, base.BLACK)
    base.add_text(drawing, 90.2, 66.3, "t-SNE", 6.1)
    family_x = 109.0
    for family, advance in (("hEVO", 19.0), ("Evo 2", 19.0), ("NT", 15.0), ("GPN", 17.0)):
        family_color = base.MODEL_FAMILY_STYLE[family]["color"]
        base.draw_filled_point(drawing, family_x, 67.0, family_color)
        base.add_text(drawing, family_x + 2.2, 66.3, family, 6.1)
        family_x += advance

    label_x = 7.0
    plot_x0 = 58.0
    plot_x1 = 176.0
    row_top = 61.5
    # Match the model-row pitch used by the compact NT and Genomic panels.
    row_bottom = row_top - 2.15 * (len(order) - 1)
    row_step = (row_top - row_bottom) / (len(order) - 1)

    for index, group in enumerate(order):
        y = row_top - index * row_step
        rows = selected[selected["algorithm_group"] == group].set_index("method")
        model = str(rows.iloc[0]["selected_model"])
        label = str(rows.iloc[0]["display_label"])
        family = base.model_family(model)
        family_color = base.MODEL_FAMILY_STYLE[family]["color"]
        pca = float(rows.loc["PCA", "ch_index_median"])
        tsne = float(rows.loc["t-SNE", "ch_index_median"])
        x_pca = x_map(pca, plot_x0, plot_x1)
        x_tsne = x_map(tsne, plot_x0, plot_x1)

        if family == "hEVO":
            drawing.add(
                Rect(
                    4.0 * mm,
                    (y - row_step * 0.42) * mm,
                    172.0 * mm,
                    row_step * 0.84 * mm,
                    fillColor=base.color(base.HEVO_WASH),
                    strokeColor=None,
                )
            )
        base.draw_filled_point(drawing, 5.1, y, family_color)
        base.add_text(drawing, label_x, y - 0.72, label, 6.25)
        drawing.add(
            Line(
                x_pca * mm,
                y * mm,
                x_tsne * mm,
                y * mm,
                strokeColor=base.color(base.CONNECTOR_GREY),
                strokeWidth=0.65,
            )
        )
        base.draw_open_point(drawing, x_pca, y, family_color)
        base.draw_filled_point(drawing, x_tsne, y, family_color)

    axis_y = 16.7
    drawing.add(
        Line(
            plot_x0 * mm,
            axis_y * mm,
            plot_x1 * mm,
            axis_y * mm,
            strokeColor=base.color(base.BLACK),
            strokeWidth=0.7,
        )
    )
    for value in range(0, 5001, 1000):
        x = x_map(value, plot_x0, plot_x1)
        drawing.add(
            Line(
                x * mm,
                axis_y * mm,
                x * mm,
                (axis_y - 1.2) * mm,
                strokeColor=base.color(base.BLACK),
                strokeWidth=0.6,
            )
        )
        base.add_text(drawing, x, axis_y - 3.8, f"{value:,}", 6.0, anchor="middle")
    base.add_text(
        drawing,
        (plot_x0 + plot_x1) / 2,
        0.7,
        "Calinski-Harabasz index",
        7.1,
        anchor="middle",
    )
    return drawing, selected, summary, order


def render(table: pd.DataFrame, output_dir: Path, stem: str) -> dict[str, object]:
    drawing, selected, summary, order = build_drawing(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    base.render_png_from_pdf(pdf, png)

    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_csv = source_dir / "FigS2a_best_algorithm_representatives_plot_source.csv"
    summary_csv = source_dir / "FigS2a_best_algorithm_representatives_selection_summary.csv"
    selected.sort_values(["representative_rank", "method"], kind="stable").to_csv(
        source_csv, index=False
    )
    summary.to_csv(summary_csv, index=False)

    provenance_values = set(
        table.get("provenance_status", pd.Series(dtype=str)).dropna().astype(str)
    )
    status = (
        "LEGACY_VECTOR_RECOVERED_QC_DRAFT"
        if "LEGACY_VECTOR_RECOVERED_QC_DRAFT" in provenance_values
        else "SOURCE_BACKED"
    )
    with Image.open(png) as image:
        png_size = list(image.size)
    qa = {
        "status": status,
        "panel": "Figure S2a representative algorithms",
        "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
        "representatives": len(order),
        "selection_rule": "highest t-SNE CH among parameter-scale variants; both main hEVO scales retained by author request",
        "hEVO_ablation_policy": "corpus/stride ablations excluded from algorithm comparison",
        "excluded_by_author_request": ["NTv1-1000G", "Caduceus-PS"],
        "distinct_models": ["NTv2-MS", "NTv3", "GPN-Promoter", "GPN-MSA"],
        "axis": {"minimum": AXIS_MIN, "maximum": AXIS_MAX, "shared_with_family_view": True},
        "font": "Arial",
        "minimum_font_pt": 6.0,
        "gridlines": False,
        "text_color": base.BLACK,
        "pdf_inspection": base.inspect_pdf(pdf),
        "png_pixels": png_size,
        "exports": {"pdf": str(pdf), "svg": str(svg), "png": str(png)},
        "hashes": {
            "input": base.sha256(Path(table.attrs["source_path"])),
            "script": base.sha256(HERE),
            "plot_source": base.sha256(source_csv),
            "selection_summary": base.sha256(summary_csv),
        },
    }
    return qa


def parse_args() -> argparse.Namespace:
    default_out = base.PROJECT_ROOT / "figures" / "formal" / "Fig2" / "figS2a_ucsc_elements_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument(
        "--stem",
        default="FigS2b_UCSC_best_algorithm_representatives_LEGACY_VECTOR_RECOVERED_QC_DRAFT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = base.read_source(args.input_csv)
    table.attrs["source_path"] = str(args.input_csv.resolve())
    qa = render(table, args.output_dir, args.stem)


if __name__ == "__main__":
    main()
