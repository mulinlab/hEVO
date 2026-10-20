"""Supplementary continuous selected-allele preference heatmap.

The heatmap remains an independent supplemental output and is not placed in
the active A4 a-h page. It uses a per-model signed percentile of ``abs(locus_score)``. This
preserves the biological zero baseline and avoids comparing raw likelihood
scales across heterogeneous model families.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.matplotlib_common import configure_matplotlib, export_figure
from common.model_names import canonical_model_name
from common.paths import figure_dir, tables_dir as default_tables_dir
from Fig4.f_adaptation_auprc import _sort_models


def _label(model_id: str, display: str, status: str) -> str:
    value = canonical_model_name(model_id)
    return value + ("*" if status == "BLOCKED" else "")


def _ordered_loci(scores: pd.DataFrame) -> pd.DataFrame:
    loci = scores[["variant_id", "gene", "qa_status"]].drop_duplicates("variant_id")
    loci = loci.assign(_gene_sort=loci["gene"].fillna("").astype(str))
    return (
        loci.sort_values(["_gene_sort", "variant_id"], kind="stable")
        .drop(columns="_gene_sort")
        .reset_index(drop=True)
    )


def prepare_heatmap_data(tables_dir: Path | None = None):
    root = Path(tables_dir) if tables_dir is not None else default_tables_dir()
    summary = pd.read_csv(
        root / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv"
    )
    scores = pd.read_csv(
        root / "S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv"
    )
    order = _sort_models(summary)
    loci = _ordered_loci(scores)
    models = order["model_id"].tolist()
    variants = loci["variant_id"].tolist()

    grid = pd.MultiIndex.from_product(
        [models, variants], names=["model", "variant_id"]
    ).to_frame(index=False)
    score_fields = scores[
        [
            "model_id",
            "variant_id",
            "gene",
            "qa_status",
            "benchmark_status",
            "locus_score",
        ]
    ].rename(columns={"model_id": "model", "locus_score": "raw_locus_score"})
    visual = grid.merge(score_fields, on=["model", "variant_id"], how="left")
    visual["raw_locus_score"] = pd.to_numeric(
        visual["raw_locus_score"], errors="coerce"
    )
    valid = visual["raw_locus_score"].notna() & visual["benchmark_status"].eq("PASS")
    visual["abs_score"] = visual["raw_locus_score"].abs()
    visual["abs_rank"] = np.nan
    visual.loc[valid, "abs_rank"] = visual.loc[valid].groupby("model")[
        "abs_score"
    ].rank(method="average", ascending=True)
    n_valid = visual.loc[valid].groupby("model").size()
    visual["abs_percentile"] = visual["abs_rank"] / visual["model"].map(n_valid)
    nonzero = valid & visual["raw_locus_score"].ne(0)
    visual["normalized_visual_score"] = np.nan
    visual.loc[nonzero, "normalized_visual_score"] = (
        np.sign(visual.loc[nonzero, "raw_locus_score"])
        * visual.loc[nonzero, "abs_percentile"]
    )
    visual.loc[valid & visual["raw_locus_score"].eq(0), "normalized_visual_score"] = 0.0
    visual["score_sign"] = "NA"
    visual.loc[valid & visual["raw_locus_score"].lt(0), "score_sign"] = "negative"
    visual.loc[valid & visual["raw_locus_score"].eq(0), "score_sign"] = "zero"
    visual.loc[valid & visual["raw_locus_score"].gt(0), "score_sign"] = "positive"
    return visual, order, loci


def write_visualization_scores(visual: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "model",
        "variant_id",
        "gene",
        "qa_status",
        "raw_locus_score",
        "normalized_visual_score",
        "score_sign",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visual[columns].to_csv(output_path, sep="\t", index=False)


def build_panel(
    tables_dir: Path | None = None,
    *,
    figure=None,
    slot=None,
    integrated: bool = False,
    letter: str | None = None,
    prepared=None,
):
    configure_matplotlib()
    if prepared is None:
        visual, order, loci = prepare_heatmap_data(tables_dir)
    else:
        visual, order, loci = prepared
    models = order["model_id"].tolist()
    variants = loci["variant_id"].tolist()
    matrix = (
        visual.pivot(index="model", columns="variant_id", values="normalized_visual_score")
        .reindex(index=models, columns=variants)
        .to_numpy(dtype=float)
    )
    masked = np.ma.masked_invalid(matrix)

    if figure is None:
        figure = plt.figure(figsize=(127 / 25.4, 70 / 25.4), facecolor="white")
    ax = figure.add_subplot(slot) if slot is not None else figure.add_subplot(111)
    cmap = LinearSegmentedColormap.from_list(
        "adaptive_preference",
        ["#91A6B5", "#FFFFFF", "#72A78A"],
        N=256,
    )
    cmap.set_bad("#FAFAFA")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        norm=norm,
        zorder=1,
    )
    ax.set_yticks(np.arange(len(models)))
    if integrated:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(
            [
                _label(row.model_id, row.display_model, row.status)
                for row in order.itertuples()
            ],
            fontsize=4.0,
        )
    ax.set_xticks(np.arange(len(loci)))
    ax.set_xticklabels(loci["variant_id"], rotation=90, fontsize=3.7)
    ax.tick_params(axis="both", length=0, pad=1)

    for row_index, model_id in enumerate(models):
        if model_id in {"hEvo-base", "hEvo-large"}:
            ax.add_patch(
                plt.Rectangle(
                    (-0.5, row_index - 0.5),
                    len(loci),
                    1,
                    fill=False,
                    edgecolor="#D62728" if model_id == "hEvo-large" else "#1F77B4",
                    linewidth=0.65,
                    zorder=3,
                )
            )
    for x, qa_status in enumerate(loci["qa_status"]):
        if str(qa_status).startswith("WARN"):
            ax.plot(
                x,
                -0.78,
                marker="v",
                markersize=2.0,
                color="#E67E22",
                clip_on=False,
                zorder=4,
            )
    ax.set_xlabel(
        "Benchmark locus (orange triangle = warning locus)",
        fontsize=5.0,
        labelpad=2,
    )
    ax.set_title(
        "Selected-allele preference by model and locus",
        loc="left",
        fontsize=6.1,
        pad=3,
        fontweight="bold",
    )
    ax.text(
        -0.045,
        1.05,
        letter or "i",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        va="bottom",
    )
    colorbar = figure.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.012,
        ticks=[-1.0, 0.0, 1.0],
    )
    colorbar.ax.set_yticklabels(
        ["Alternative", "0", "Selected"], fontsize=3.8
    )
    colorbar.set_label(
        "Relative selected-allele preference", fontsize=4.2, labelpad=2
    )
    if not integrated:
        ax.legend(
            handles=[
                Patch(
                    facecolor="#FAFAFA",
                    edgecolor="#C9CED6",
                    label="NA = unavailable",
                )
            ],
            loc="upper left",
            bbox_to_anchor=(0, -0.27),
            frameon=False,
            fontsize=4.0,
            handlelength=0.9,
        )
    return ax


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    root = Path(tables_dir) if tables_dir is not None else default_tables_dir()
    out = Path(output_dir) if output_dir is not None else figure_dir("Fig4")
    out.mkdir(parents=True, exist_ok=True)
    visual, order, loci = prepare_heatmap_data(root)
    score_path = out / "heatmap_visualization_scores.tsv"
    write_visualization_scores(visual, score_path)
    fig = plt.figure(figsize=(127 / 25.4, 70 / 25.4), facecolor="white")
    build_panel(root, figure=fig, slot=111, prepared=(visual, order, loci))
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.27, top=0.90)
    result = export_figure(
        fig,
        out,
        "i_adaptive_allele_preference_locus_matrix",
        [
            root / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv",
            root / "S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv",
        ],
        width_mm=127,
        height_mm=70,
    )
    plt.close(fig)
    result["heatmap_visualization_scores"] = str(score_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(export_panel(args.output_dir, args.tables_dir))


if __name__ == "__main__":
    main()
