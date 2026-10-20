"""Render the six-panel Fig. 4 page from its local supplementary tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.matplotlib_common import configure_matplotlib, export_figure
from common.panel_runtime import input_files, panel_tables_dir
from common.paths import figure_dir
from Fig4 import (
    _har_haqer,
    e_haqer_gpn_comparison,
    g_adaptive_world_map,
    h_adaptive_allele_preference,
)
from Fig4._backend import BACKEND
from Fig4._backend import load_data as load_ancestry_data

PAGE_W_PT, PAGE_H_PT = 595.2755905512, 841.8897637795
PAGE_W_MM, PAGE_H_MM = 210.0, 297.0
TOP_PANEL_HEIGHT_PT = 140.393701

ANCESTRY_CLASSIFICATION_MODELS = [
    "hEvo-large",
    "evo2_7b_base",
    "NTv3-650M-pre",
    "NT_500m_1kG",
    "DNABERT",
    "DNABERT2",
    "GPN-MSA",
    "GPN-Star-V",
    "GENA-LM-large",
    "GENERator-3B",
    "NT_500m_MS",
    "AIDO.DNA-300M",
    "Mistral-DNA-422M",
    "GPN-promoter",
    "Genos-1.2B",
    "GROVER",
    "hEvo-base",
    "Caduceus-PS",
    "SpeciesLM",
    "Caduceus-PH",
    "HyenaDNA_tiny_1k",
]
FEWSHOT_MODELS = [
    "hEvo-large",
    "evo2_7b_base",
    "NT_2.5b_1kG",
    "GENERator-3B",
    "NTv3-650M-pre",
    "GPN-Star-P",
    "GPN-MSA",
    "DNABERT",
    "NT_500m_MS",
    "GENA-LM-base",
    "AIDO.DNA-300M",
    "Genos-1.2B",
    "DNABERT2",
    "Mistral-DNA-422M",
    "hEvo-base",
    "GPN-promoter",
    "GROVER",
    "SpeciesLM",
    "Caduceus-PS",
    "Caduceus-PH",
    "HyenaDNA_tiny_1k",
]
HAR_MODELS = [
    "hEvo-large",
    "GPN-Star-M",
    "hEvo-base",
    "GENA-LM-large",
    "evo2_7b_base",
    "NT_2.5b_1kG",
    "DNABERT2",
    "Caduceus-PS",
    "DNABERT",
    "Caduceus-PH",
    "Mistral-DNA-422M",
    "Genos-1.2B",
    "NTv3-650M-pre",
    "SpeciesLM",
    "GROVER",
    "AIDO.DNA-300M",
    "NT_500m_MS",
    "GENERator-3B",
    "GPN-promoter",
    "HyenaDNA_tiny_1k",
    "GPN-MSA",
]
HAQER_MODELS = [
    "hEvo-large",
    "GPN-Star-M",
    "hEvo-base",
    "GENA-LM-large",
    "evo2_7b_base",
    "Genos-1.2B",
    "DNABERT2",
    "DNABERT",
    "NT_2.5b_1kG",
    "Mistral-DNA-422M",
    "GROVER",
    "NTv3-650M-pre",
    "GPN-promoter",
    "AIDO.DNA-300M",
    "SpeciesLM",
    "NT_500m_MS",
    "Caduceus-PS",
    "HyenaDNA_tiny_1k",
    "GENERator-1.2B",
    "Caduceus-PH",
    "GPN-MSA",
]


def _axes_from_top(
    fig, left_pt: float, top_pt: float, width_pt: float, height_pt: float
):
    """Place an axes using page points measured from the upper-left corner."""

    return fig.add_axes(
        [
            left_pt / PAGE_W_PT,
            1.0 - (top_pt + height_pt) / PAGE_H_PT,
            width_pt / PAGE_W_PT,
            height_pt / PAGE_H_PT,
        ]
    )


def _page_text(fig, text: str, x_pt: float, y_pt: float, **kwargs):
    return fig.text(x_pt / PAGE_W_PT, 1.0 - y_pt / PAGE_H_PT, text, **kwargs)


def _draw_reference_endpoint(ax, rows):
    """Draw the 60-sample ranking with the reference PDF's SD whiskers."""

    BACKEND.y_model_axis(ax, rows, font_size=6.0)
    _shade_reference_hevo_rows(ax, rows)
    for y, row in rows.reset_index(drop=True).iterrows():
        family = BACKEND.model_family(row["model"])
        style = BACKEND.MODEL_FAMILY_STYLE[family]
        color = style["color"]
        marker = style["marker"]
        ax.errorbar(
            float(row["pearson_r_mean"]),
            y,
            xerr=float(row["pearson_r_std"]),
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=1.4,
            capthick=1.0,
            zorder=2,
        )
        is_hevo = row["model"] in {"hEvo-large", "hEvo-base"}
        face = "white" if row["model"] == "hEvo-base" else color
        edge = color if row["model"] == "hEvo-base" else "#000000"
        ax.scatter(
            float(row["pearson_r_mean"]),
            y,
            s=16.0 if is_hevo else 14.0,
            marker=marker,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.55,
            zorder=3,
        )
    ax.set_ylim(len(rows) - 0.5, -0.5)
    BACKEND.trim_spines(ax)


def _shade_reference_hevo_rows(ax, rows):
    for y, model in enumerate(rows["model"]):
        if model in {"hEvo-large", "hEvo-base"}:
            ax.axhspan(y - 0.47, y + 0.47, color="#FBF1E7", zorder=0, lw=0)


def _draw_reference_mcc(ax, rows):
    BACKEND.y_model_axis(ax, rows, font_size=6.0)
    _shade_reference_hevo_rows(ax, rows)
    for y, row in rows.reset_index(drop=True).iterrows():
        family = BACKEND.model_family(row["model"])
        style = BACKEND.MODEL_FAMILY_STYLE[family]
        is_hevo = row["model"] in {"hEvo-large", "hEvo-base"}
        face = "white" if row["model"] == "hEvo-base" else style["color"]
        edge = style["color"] if row["model"] == "hEvo-base" else "#000000"
        ax.scatter(
            float(row["classification_mcc"]),
            y,
            s=16.0 if is_hevo else 14.0,
            marker=style["marker"],
            facecolor=face,
            edgecolor=edge,
            linewidth=0.55,
            zorder=3,
        )
    ax.set_ylim(len(rows) - 0.5, -0.5)
    BACKEND.trim_spines(ax)


def build_panel(tables_dir: Path | None = None):
    """Build Fig. 4a-f with the reference PDF's compact A4 composition."""

    root = panel_tables_dir("Fig4", tables_dir)
    configure_matplotlib()
    fig = plt.figure(figsize=(PAGE_W_MM / 25.4, PAGE_H_MM / 25.4), facecolor="white")

    _, classification, fewshot = load_ancestry_data(root)
    classification_rows = classification.loc[
        classification["model"].isin(ANCESTRY_CLASSIFICATION_MODELS)
    ].copy()
    if len(classification_rows) != len(ANCESTRY_CLASSIFICATION_MODELS):
        raise ValueError(
            "S34 is missing one or more models displayed in reference panel a"
        )
    classification_rows = classification_rows.sort_values(
        ["classification_mcc", "model_display"], ascending=[False, True]
    ).reset_index(drop=True)
    classification_rows["plot_label"] = classification_rows["model"].map(
        BACKEND.compact_display
    )
    endpoint = (
        fewshot.loc[fewshot["shot"].eq(60)]
        .sort_values("pearson_r_mean", ascending=False)
        .reset_index(drop=True)
    )
    endpoint_rows = endpoint.loc[endpoint["model"].isin(FEWSHOT_MODELS)].copy()
    if len(endpoint_rows) != len(FEWSHOT_MODELS):
        raise ValueError(
            "S35 is missing one or more models displayed in reference panel b"
        )
    endpoint_rows = endpoint_rows.sort_values(
        ["pearson_r_mean", "model_display"], ascending=[False, True]
    ).reset_index(drop=True)
    endpoint_rows["plot_label"] = endpoint_rows["model"].map(BACKEND.compact_display)
    reference_label_overrides = {
        "NT_500m_1kG": "NTv1-500M-1kG",
        "NT_2.5b_1kG": "NTv1-2.5B-1kG",
    }
    classification_rows["plot_label"] = classification_rows["model"].map(
        lambda model: reference_label_overrides.get(
            model, BACKEND.compact_display(model)
        )
    )
    endpoint_rows["plot_label"] = endpoint_rows["model"].map(
        lambda model: reference_label_overrides.get(
            model, BACKEND.compact_display(model)
        )
    )

    # a: five-group ancestry classification, ranked by the S34 MCC values.
    a_slot = fig.add_gridspec(
        1,
        1,
        left=85.039 / PAGE_W_PT,
        right=172.913 / PAGE_W_PT,
        bottom=1.0 - (56.692913 + TOP_PANEL_HEIGHT_PT) / PAGE_H_PT,
        top=1.0 - 56.692913 / PAGE_H_PT,
    )[0, 0]
    a_ax = fig.add_subplot(a_slot)
    _draw_reference_mcc(a_ax, classification_rows)
    a_ax.set_xlim(0.928, 0.991)
    a_ax.set_xticks([0.94, 0.96, 0.98])
    a_ax.set_xlabel("MCC", fontsize=6.2, labelpad=2.2)
    a_ax.tick_params(axis="x", labelsize=5.7)
    a_ax.tick_params(axis="y", pad=0.9)

    # b: all few-shot trajectories plus the 60-sample endpoint ranking.
    curve_ax = _axes_from_top(fig, 218.268, 56.692913, 153.071, TOP_PANEL_HEIGHT_PT)
    rank_ax = _axes_from_top(
        fig, 438.866457, 56.692913, 119.558740, TOP_PANEL_HEIGHT_PT
    )
    BACKEND.draw_fewshot_curve(
        curve_ax, fewshot, endpoint_rows["model"].tolist(), family_colours=True
    )
    _draw_reference_endpoint(rank_ax, endpoint_rows)
    rank_ax.set_xlabel("Mean Pearson r", fontsize=6.2, labelpad=2.98)
    rank_ax.xaxis.get_label().set_x(0.50217)
    rank_ax.set_xlim(0.853814, 0.957249)
    rank_ax.set_xticks([0.86, 0.90, 0.94])
    curve_ax.set_ylabel("Mean Pearson r", fontsize=6.0, labelpad=2.32)
    curve_ax.set_xlabel("Training samples per population", fontsize=6.0, labelpad=2.34)
    curve_ax.tick_params(axis="both", labelsize=5.7, length=2.0, width=0.5, pad=1.35)
    rank_ax.tick_params(axis="x", labelsize=5.7, length=2.0, width=0.5, pad=1.85)
    rank_ax.tick_params(axis="y", labelsize=6.0, length=2.0, width=0.5, pad=1.1)

    # c/d: HAR and HAQER negative-control robustness, retaining all source rows.
    c_ax = _har_haqer.draw_dataset(
        fig,
        fig.add_gridspec(
            1,
            1,
            left=99.213 / PAGE_W_PT,
            right=295.055 / PAGE_W_PT,
            bottom=1.0 - (235.275590 + TOP_PANEL_HEIGHT_PT) / PAGE_H_PT,
            top=1.0 - 235.275590 / PAGE_H_PT,
        )[0, 0],
        root,
        "HAR",
        letter="",
        integrated=True,
        model_roster=HAR_MODELS,
    )
    d_ax = _har_haqer.draw_dataset(
        fig,
        fig.add_gridspec(
            1,
            1,
            left=362.583 / PAGE_W_PT,
            right=558.425 / PAGE_W_PT,
            bottom=1.0 - (235.275590 + TOP_PANEL_HEIGHT_PT) / PAGE_H_PT,
            top=1.0 - 235.275590 / PAGE_H_PT,
        )[0, 0],
        root,
        "HAQER",
        letter="",
        integrated=True,
        model_roster=HAQER_MODELS,
    )

    # e: the compact twelve-window HAQER browser shown in the reference PDF.
    e_haqer_gpn_comparison.build_panel(
        root,
        figure=fig,
        rect=[
            99.213 / PAGE_W_PT,
            1.0 - (552.261417 / PAGE_H_PT),
            459.212598 / PAGE_W_PT,
            142.396850 / PAGE_H_PT,
        ],
        reference_style=True,
    )

    # f: adaptive-allele world map, model preference and raw-score CH index.
    g_adaptive_world_map.build_panel(
        root,
        figure=fig,
        letter="f",
        repo_root=HERE.parents[4],
        vertical_shift_pts=-56.09,
    )

    # Page-level titles and letters match the reference's compact hierarchy.
    _page_text(
        fig,
        "Five-group ancestry",
        60.10,
        24.93,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="baseline",
    )
    _page_text(
        fig,
        "classification",
        71.39,
        32.55,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="baseline",
    )
    _page_text(
        fig,
        "Few-shot ancestry-proportion prediction",
        315.216,
        25.91,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="baseline",
    )
    _page_text(
        fig,
        "60 samples per population",
        465.857,
        39.72,
        fontsize=5.6,
        ha="left",
        va="baseline",
    )
    _page_text(
        fig,
        "HAR",
        189.54,
        222.91,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="baseline",
    )
    _page_text(
        fig,
        "HAQER",
        447.86,
        222.91,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="baseline",
    )
    BACKEND.add_family_key(fig, y=0.5287, fontsize=6.0)

    for letter, x_pt, y_pt in [
        ("a", 19.842, 26.812),
        ("b", 187.087, 26.812),
        ("c", 19.842, 223.820),
        ("d", 297.638, 223.820),
        ("e", 19.842, 412.580),
        ("f", 19.796, 590.660),
    ]:
        _page_text(
            fig,
            letter,
            x_pt,
            y_pt,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="baseline",
        )

    # The panel builders keep their data products attached to the Figure so
    # export_panel can write reproducible source summaries beside the render.
    fig._hevo_fig4_ch_source_data = fig._hevo_fig4_g_source_data
    fig._hevo_fig4_map_source_data = fig._hevo_fig4_g_map_source_data
    return fig


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    root = panel_tables_dir("Fig4", tables_dir)
    fig = build_panel(root)
    output_root = Path(output_dir or figure_dir("Fig4"))
    output_root.mkdir(parents=True, exist_ok=True)
    h_adaptive_allele_preference.write_source_data(
        fig._hevo_fig4_ch_source_data,
        output_root / "g_adaptive_allele_preference_ch_source_data.csv",
    )
    g_adaptive_world_map.write_map_source_data(
        fig._hevo_fig4_map_source_data,
        output_root / "g_adaptive_world_map_source_data.csv",
    )
    track_inputs = e_haqer_gpn_comparison.source_files()
    result = export_figure(
        fig,
        output_root,
        "A4_integrated",
        input_files(
            root / "S34_supervised_classification_of_1000_genomes_ancestry_groups.csv",
            root / "S35_few_shot_regression_of_ancestry_proportions.csv",
            root
            / "S37_classification_of_human_accelerated_regions_across_negative_controls.csv",
            root
            / "S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv",
            root
            / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv",
            root
            / "S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv",
            HERE.parents[4] / g_adaptive_world_map.LOCI_RELATIVE,
            g_adaptive_world_map.ASSET_PATH,
            *track_inputs,
        ),
        width_mm=PAGE_W_MM,
        height_mm=PAGE_H_MM,
    )
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=None
    )
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig4"))
    args = parser.parse_args()
    export_panel(args.output_dir, args.tables_dir)


if __name__ == "__main__":
    main()
