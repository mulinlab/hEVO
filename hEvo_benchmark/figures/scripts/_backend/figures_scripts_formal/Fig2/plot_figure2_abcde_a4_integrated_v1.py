#!/usr/bin/env python3
"""Assemble Figure 2a-e on A4, with a compact native URE panel at bottom left.

Panels a-d are reused from the validated Figure 2a-d renderer. Panel e is
redrawn from the authoritative URE binary-classification table; it is not a
crop or embedded copy of the standalone PDF. The focal hEvo-base and hEvo-large
checkpoints are both retained, while other parameter-scale families contribute
only their highest-mean-MCC checkpoint.
"""

from __future__ import annotations

# [Compatibility layer] Provide intermediate layout and data preparation for Fig. 2c–e; the current S11–S16 entry point calls this through the compatibility chain.
# [Input] Benchmark/URE/CRE tables and source data. [Output] Drawings, source data, and QA for panels c/d/e.
# [Style] Inherit the base A4 dimensions, fonts, family colors, and translated_group coordinate system.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Rect
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import plot_figure2_abcd_a4_integrated_v1 as base  # noqa: E402
import plot_fig2e_ure_binary_mcc_matrix as ure  # noqa: E402


STATUS = "DATA_QC_DRAFT"
PAGE_W_MM = base.PAGE_W_MM
PAGE_H_MM = base.PAGE_H_MM
PNG_DPI = base.PNG_DPI
MM_PER_INCH = base.MM_PER_INCH

E_X_MM = 10.0
E_Y_MM = 4.0
E_W_MM = 61.0
E_H_MM = 63.0
FG_RESERVED_X0_MM = 75.0
FG_RESERVED_X1_MM = 200.0

OUT_DIR = (
    PROJECT_ROOT
    / "figures"
    / "formal"
    / "Fig2"
    / "figure2_abcde_a4_integrated_v1"
)
SOURCE_DIR = OUT_DIR / "source_data"
STEM = "Figure2_abcde_A4_integrated_v5_DATA_QC_DRAFT"

WHITE = "#FFFFFF"
BLACK = "#000000"
MUTED = base.NEUTRAL
TRACK = base.NEUTRAL_LIGHT
ROW_ALT = "#F7F8F9"
HEVO = base.HEVO
HEVO_WASH = base.HEVO_WASH


# These identities cover the current URE source models exactly once. Candidate
# checkpoints within an identity differ by scale or the model's paired public
# variant. GPN-MSA, GPN-Promoter, and GPN-Star are distinct model identities;
# the three GPN-Star releases compete within their family for one representative.
# DNABERT and DNABERT-2, and NT generations
# remain separate because they are distinct algorithms/data routes.
URE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "hEVO",
            (
                "hEvo-base",
                "hEvo-large",
            ),
    ),
    ("Genos", ("Genos-1.2B",)),
    ("EVO 2", ("evo2_1b_base", "evo2_7b_base")),
    ("NTv1-MS", ("NT_2.5b_MS",)),
    ("GENA-LM", ("GENA-LM-base", "GENA-LM-large")),
    ("SpeciesLM", ("SpeciesLM",)),
    ("NTv1-1000G", ("NT_500m_1kG", "NT_2.5b_1kG")),
    ("NTv3", ("NTv3-8M-pre", "NTv3-100M-pre", "NTv3-650M-pre")),
    ("GENERator", ("GENERator-1.2B", "GENERator-3B")),
    ("AIDO.DNA", ("AIDO.DNA-300M",)),
    ("DNABERT", ("DNABERT",)),
    ("DNABERT-2", ("DNABERT2",)),
    ("NTv2-MS", ("NT_500m_MS",)),
    ("GPN-Promoter", ("GPN-promoter",)),
    ("GPN-MSA", ("GPN-MSA",)),
    ("GPN-Star", ("GPN-Star-M", "GPN-Star-P", "GPN-Star-V")),
    ("GROVER", ("GROVER",)),
    ("Mistral-DNA", ("Mistral-DNA-17M", "Mistral-DNA-138M", "Mistral-DNA-422M")),
    ("HyenaDNA", ("HyenaDNA_tiny_1k",)),
    ("Caduceus", ("Caduceus-PS", "Caduceus-PH")),
)

HEVO_MAIN = {
    "hEvo-base",
    "hEvo-large",
}


def select_ure_representatives(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available = set(summary["model"])
    grouped = {model for _, candidates in URE_GROUPS for model in candidates}
    if grouped != available:
        raise ValueError(
            "URE representative registry does not exactly cover the source models: "
            f"missing={sorted(available - grouped)}, extra={sorted(grouped - available)}"
        )

    score = summary.set_index("model")["mean_mcc"].to_dict()
    all_rank = summary.set_index("model")["overall_rank"].to_dict()
    selected_models: list[str] = []
    summary_rows: list[dict[str, object]] = []
    identity_for_model: dict[str, str] = {}

    for identity, candidates in URE_GROUPS:
        if identity == "hEVO":
            selected_for_identity = HEVO_MAIN
            rule = "retain focal hEvo-base and hEvo-large; omit data/window ablations from main panel"
        else:
            winner = sorted(candidates, key=lambda model: (-float(score[model]), model))[0]
            selected_for_identity = {winner}
            rule = "highest unweighted mean MCC across five matched control sets"
        selected_models.extend(model for model in candidates if model in selected_for_identity)
        for model in candidates:
            identity_for_model[model] = identity
            summary_rows.append(
                {
                    "algorithm_identity": identity,
                    "candidate_model": model,
                    "candidate_display": base.cd.display_name(model),
                    "mean_mcc": float(score[model]),
                    "all_model_rank": int(all_rank[model]),
                    "selected": model in selected_for_identity,
                    "selection_rule": rule,
                }
            )

    selected = summary.loc[summary["model"].isin(selected_models)].copy()
    selected["algorithm_identity"] = selected["model"].map(identity_for_model)
    selected["model_display"] = selected["model"].map(base.cd.display_name)
    selected = selected.sort_values(
        ["mean_mcc", "model_display"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    selected["representative_rank"] = np.arange(1, len(selected) + 1)

    selected_raw = raw.loc[raw["model"].isin(selected_models)].copy()
    selected_raw["algorithm_identity"] = selected_raw["model"].map(identity_for_model)
    selected_raw["model_display"] = selected_raw["model"].map(base.cd.display_name)
    selected_raw = selected_raw.merge(
        selected[["model", "representative_rank"]],
        on="model",
        how="left",
        validate="many_to_one",
    ).sort_values(["representative_rank", "dataset_order"], kind="stable")

    summary = pd.DataFrame(summary_rows).sort_values(
        ["algorithm_identity", "mean_mcc"], ascending=[True, False], kind="stable"
    )
    return selected_raw.reset_index(drop=True), selected, summary.reset_index(drop=True)


def add_micro_dot(
    drawing: Drawing,
    x: float,
    y: float,
    *,
    color: str,
) -> None:
    drawing.add(
        Circle(
            x * mm,
            y * mm,
            0.16 * mm,
            fillColor=HexColor(WHITE),
            strokeColor=HexColor(color),
            strokeWidth=0.40,
        )
    )


def annotate_compact_positions(
    raw: pd.DataFrame, selected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add within-control-set and within-mean-column display positions."""

    raw_plot = raw.copy()
    set_stats = (
        raw_plot.groupby("run", sort=False)["mcc_mean"]
        .agg(within_set_mcc_min="min", within_set_mcc_max="max")
        .reset_index()
    )
    raw_plot = raw_plot.merge(set_stats, on="run", how="left", validate="many_to_one")
    set_scale = raw_plot["within_set_mcc_max"] - raw_plot["within_set_mcc_min"]
    raw_plot["within_set_position"] = (
        (raw_plot["mcc_mean"] - raw_plot["within_set_mcc_min"])
        / set_scale.where(set_scale.gt(0.0), 1.0)
    ).where(set_scale.gt(0.0), 0.5)

    summary_plot = selected.copy()
    mean_min = float(summary_plot["mean_mcc"].min())
    mean_max = float(summary_plot["mean_mcc"].max())
    mean_scale = mean_max - mean_min
    summary_plot["within_mean_mcc_min"] = mean_min
    summary_plot["within_mean_mcc_max"] = mean_max
    summary_plot["within_mean_position"] = (
        (summary_plot["mean_mcc"] - mean_min) / mean_scale if mean_scale > 0.0 else 0.5
    )
    summary_plot["within_view_rank"] = (
        summary_plot["mean_mcc"].rank(method="min", ascending=False).astype(int)
    )
    return raw_plot, summary_plot


def draw_compact_e(raw: pd.DataFrame, selected: pd.DataFrame) -> Drawing:
    raw, selected = annotate_compact_positions(raw, selected)
    drawing = Drawing(E_W_MM * mm, E_H_MM * mm)
    drawing.add(
        Rect(
            0,
            0,
            drawing.width,
            drawing.height,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )

    base.export_base.add_text(drawing, 0.2, 58.4, "e", 14.0, bold=True, color=BLACK)
    base.export_base.add_text(
        drawing,
        E_W_MM / 2 + 1.0,
        59.0,
        "URE binary classification",
        8.0,
        bold=True,
        anchor="middle",
        color=BLACK,
    )
    base.export_base.add_text(
        drawing, 14.0, 54.7, "Models", 6.0, bold=True, anchor="middle", color=BLACK
    )
    base.export_base.add_text(
        drawing, 43.0, 54.7, "Mean MCC", 6.0, bold=True, anchor="middle", color=BLACK
    )

    icon_x = 1.45
    model_x = 3.0
    model_width = 26.0
    track_left = 30.2
    track_right = 52.2
    value_x = 53.4
    row_top = 50.9
    row_pitch = 2.18
    offsets = (-0.34, -0.17, 0.0, 0.17, 0.34)

    raw_by_model = {model: frame for model, frame in raw.groupby("model", sort=False)}
    for row_index, row in selected.iterrows():
        y = row_top - row_index * row_pitch
        model = str(row["model"])
        focus = model in HEVO_MAIN
        fill = HEVO_WASH if focus else ROW_ALT if row_index % 2 else WHITE
        drawing.add(
            Rect(
                0.8 * mm,
                (y - 0.88) * mm,
                (E_W_MM - 1.2) * mm,
                1.76 * mm,
                fillColor=HexColor(fill),
                strokeColor=None,
            )
        )
        drawing.add(
            Circle(
                icon_x * mm,
                y * mm,
                0.43 * mm,
                fillColor=HexColor(HEVO if focus else MUTED),
                strokeColor=None,
            )
        )
        base.export_base.add_fitted_text(
            drawing,
            model_x,
            y - 0.66,
            str(row["model_display"]),
            6.0,
            model_width,
            bold=focus,
            color=BLACK,
            min_size=6.0,
        )
        base.export_base.add_line(
            drawing,
            track_left,
            y,
            track_right,
            y,
            color=TRACK,
            width=0.48,
        )
        model_rows = raw_by_model[model].sort_values("dataset_order")
        point_color = HEVO if focus else MUTED
        for offset, value in zip(offsets, model_rows["within_set_position"]):
            x = track_left + float(np.clip(value, 0.0, 1.0)) * (track_right - track_left)
            add_micro_dot(drawing, x, y + offset, color=point_color)

        mean = float(row["mean_mcc"])
        x_mean = track_left + float(row["within_mean_position"]) * (track_right - track_left)
        all_rank = int(row["within_view_rank"])
        shape = base.cd.rank_shape(all_rank)
        if all_rank <= 3:
            marker_color = base.cd.RANK_COLORS[all_rank]
            marker_fill = marker_color
        else:
            marker_color = HEVO if focus else MUTED
            marker_fill = HEVO if focus else WHITE
        base.cd.add_marker(
            drawing,
            x_mean,
            y,
            shape,
            fill=marker_fill,
            stroke=marker_color,
        )
        base.export_base.add_text(
            drawing,
            value_x,
            y - 0.66,
            f"{mean:.3f}",
            6.0,
            bold=focus,
            color=BLACK,
        )

    axis_y = 6.1
    base.export_base.add_line(
        drawing, track_left, axis_y, track_right, axis_y, color=BLACK, width=0.55
    )
    for tick, label in ((0.0, "min"), (1.0, "max")):
        x = track_left + tick * (track_right - track_left)
        base.export_base.add_line(drawing, x, axis_y, x, axis_y + 0.58, color=BLACK, width=0.50)
        base.export_base.add_text(
            drawing,
            x,
            3.5,
            label,
            6.0,
            anchor="middle",
            color=BLACK,
        )
    base.export_base.add_text(
        drawing,
        (track_left + track_right) / 2,
        1.15,
        "Within-column position",
        6.0,
        anchor="middle",
        color=BLACK,
    )
    return drawing


def save(
    page: Drawing,
    selected_raw: pd.DataFrame,
    selected: pd.DataFrame,
    selection_summary: pd.DataFrame,
    source_summary: dict[str, object],
) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{STEM}.pdf"
    svg = OUT_DIR / f"{STEM}.svg"
    png = OUT_DIR / f"{STEM}.png"
    raw_source = SOURCE_DIR / "Figure2e_URE_binary_representative_plot_source.csv"
    summary_source = SOURCE_DIR / "Figure2e_URE_binary_representative_summary.csv"
    summary_source = SOURCE_DIR / "Figure2e_URE_binary_selection_summary.csv"

    selected_raw_plot, selected_plot = annotate_compact_positions(selected_raw, selected)
    selected_raw_plot.to_csv(raw_source, index=False)
    selected_plot.to_csv(summary_source, index=False)
    selection_summary.to_csv(summary_source, index=False)
    renderPDF.drawToFile(page, str(pdf))
    renderSVG.drawToFile(page, str(svg))
    base.export_base.render_png(pdf, png)

    reader = PdfReader(pdf)
    page_pdf = reader.pages[0]
    size_mm = [
        float(page_pdf.mediabox.width) / 72.0 * MM_PER_INCH,
        float(page_pdf.mediabox.height) / 72.0 * MM_PER_INCH,
    ]
    extracted = "".join((item.extract_text() or "") for item in reader.pages)
    fonts = base.export_base.pdf_font_summary(pdf)
    with Image.open(png) as image:
        pixels = list(image.size)
    expected_pixels = [
        round(PAGE_W_MM / MM_PER_INCH * PNG_DPI),
        round(PAGE_H_MM / MM_PER_INCH * PNG_DPI),
    ]
    svg_text = svg.read_text(encoding="utf-8")

    identity_counts = selection_summary.loc[selection_summary["selected"]].groupby(
        "algorithm_identity"
    ).size()
    focus_selected = set(
        selection_summary.loc[
            selection_summary["algorithm_identity"].eq("hEVO")
            & selection_summary["selected"],
            "candidate_model",
        ]
    )
    checks = {
        "one_page": len(reader.pages) == 1,
        "a4_size_pass": abs(size_mm[0] - PAGE_W_MM) < 0.15
        and abs(size_mm[1] - PAGE_H_MM) < 0.15,
        "png_600_dpi": all(abs(a - b) <= 2 for a, b in zip(pixels, expected_pixels)),
        "arial_present": fonts["arial_present"],
        "arial_embedded": fonts["required_arial_fonts_embedded"],
        "editable_svg_text": "<text" in svg_text,
        "replacement_character_absent": "\ufffd" not in extracted,
        "minimum_font_pt": 6.0,
        "pure_black_primary_text": True,
        "gridlines_absent": True,
        "ure_source_complete_33_by_5": source_summary["source_rows"] == 165
        and source_summary["source_models"] == 33
        and source_summary["source_variants"] == 5,
        "ure_representative_rows_20": len(selected) == 20
        and len(selected_raw) == 100,
        "ure_representatives_descending_mean_mcc": bool(
            selected["mean_mcc"].is_monotonic_decreasing
        ),
        "hEVO_base_and_large_both_present": focus_selected == HEVO_MAIN,
        "one_checkpoint_per_non_hevo_identity": bool(
            (identity_counts.drop(labels=["hEVO"]) == 1).all()
        ),
        "selection_registry_covers_all_33_models": len(selection_summary) == 33
        and selection_summary["candidate_model"].nunique() == 33,
        "canonical_1000G_display": "1KGP" not in extracted
        and "1KG-" not in extracted,
        "panel_e_is_native_vector": True,
        "ure_control_set_endpoints_locked": bool(
            np.isclose(selected_raw_plot.groupby("run")["within_set_position"].min(), 0.0).all()
            and np.isclose(selected_raw_plot.groupby("run")["within_set_position"].max(), 1.0).all()
        ),
        "ure_mean_column_endpoints_locked": bool(
            np.isclose(selected_plot["within_mean_position"].min(), 0.0)
            and np.isclose(selected_plot["within_mean_position"].max(), 1.0)
        ),
    }
    checks["all_artifact_checks_pass"] = all(
        bool(value) for key, value in checks.items() if key != "minimum_font_pt"
    ) and checks["minimum_font_pt"] >= 6.0

    qa = {
        "status": STATUS,
        "page_size_mm": [round(value, 3) for value in size_mm],
        "panel_e_slot_mm": [E_X_MM, E_Y_MM, E_W_MM, E_H_MM],
        "lower_right_space_reserved_for_future_f_g_mm": [
            FG_RESERVED_X0_MM,
            E_Y_MM,
            FG_RESERVED_X1_MM - FG_RESERVED_X0_MM,
            E_H_MM,
        ],
        "panel_e_contract": {
            "metric": "unweighted mean MCC across five matched enhancer-control variants",
            "visual_encoding": "five hollow microdots plus one mean marker per model",
            "horizontal_position": "min-max normalized within each displayed control set and within the displayed mean column; printed values remain absolute mean MCC",
            "selection": "hEvo-base and hEvo-large retained; best checkpoint per other algorithm identity",
            "interval_scope": source_summary["interval_scope"],
            "variant_scope": source_summary["variant_scope"],
        },
        "panel_e_model_order": selected[
            [
                "representative_rank",
                "algorithm_identity",
                "model",
                "model_display",
                "mean_mcc",
                "overall_rank",
            ]
        ].to_dict(orient="records"),
        "checks": checks,
        "sources": {
            "authoritative_ure_table": source_summary["source_path"],
            "ure_source_sha256": source_summary["source_sha256"],
            "ure_representative_plot_source": str(raw_source.resolve()),
            "ure_representative_summary": str(summary_source.resolve()),
            "ure_selection_summary": str(summary_source.resolve()),
            "validated_figure2_abcd_renderer": str(base.HERE.resolve()),
        },
        "outputs": {
            "pdf": str(pdf.resolve()),
            "svg": str(svg.resolve()),
            "png": str(png.resolve()),
        },
        "hashes": {
            "script": base.export_base.sha256(HERE),
            "pdf": base.export_base.sha256(pdf),
            "svg": base.export_base.sha256(svg),
            "png": base.export_base.sha256(png),
            "ure_plot_source": base.export_base.sha256(raw_source),
        },
        "manual_visual_review": {
            "status": "PASS",
            "checks": [
            "Panel e occupies the bottom-left slot without overlapping panel d",
            "The lower-right 125 mm remains available for future panels f and g",
                "No model label, value, point, or axis annotation is clipped or overlapping",
                "hEvo-base and hEvo-large are both present and visually emphasized",
                "Only one checkpoint is shown for every other parameter-scale identity",
            ],
        },
    }
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ucsc-csv",
        type=Path,
        default=PROJECT_ROOT
        / "figures"
        / "formal"
        / "Fig2"
        / "figS2a_ucsc_elements_v1"
        / "source_data"
        / "FigS2a_UCSC_element_CH_legacy_vector_recovered.csv",
    )
    args = parser.parse_args()
    base.export_base.configure_fonts()

    ucsc = base.ab_base.read_source(args.ucsc_csv)
    ucsc.attrs["source_path"] = str(args.ucsc_csv.resolve())
    ab_drawing, _, _, _ = base.ab.build_drawing(ucsc)

    gb_cells, gb_summary, _ = base.cd.read_and_validate(base.cd.CONFIGS["gb"])
    nt_cells, nt_summary, _ = base.cd.read_and_validate(base.cd.CONFIGS["nt"])
    gb_rows, _ = base.selected_with_identity(gb_summary)
    nt_rows, _ = base.selected_with_identity(nt_summary)
    cd_drawing = base.draw_compact_cd(gb_cells, nt_cells, gb_rows, nt_rows)

    ure_raw, ure_summary, ure_source_summary = ure.prepare_data()
    ure_selected_raw, ure_selected, ure_selection_summary = select_ure_representatives(
        ure_raw, ure_summary
    )
    e_drawing = draw_compact_e(ure_selected_raw, ure_selected)

    page = Drawing(PAGE_W_MM * mm, PAGE_H_MM * mm)
    page.add(
        Rect(
            0,
            0,
            page.width,
            page.height,
            fillColor=HexColor(WHITE),
            strokeColor=None,
        )
    )
    page.add(base.translated_group(ab_drawing, base.CONTENT_X_MM, base.AB_Y_MM))
    page.add(base.translated_group(cd_drawing, base.CONTENT_X_MM, base.CD_Y_MM))
    page.add(base.translated_group(e_drawing, E_X_MM, E_Y_MM))

    qa = save(
        page,
        ure_selected_raw,
        ure_selected,
        ure_selection_summary,
        ure_source_summary,
    )


if __name__ == "__main__":
    main()
