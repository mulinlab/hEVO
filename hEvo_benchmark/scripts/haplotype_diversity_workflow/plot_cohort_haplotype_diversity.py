#!/usr/bin/env python3
"""Render the actual teacher-order cohort-balanced haplotype diversity plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig1_style import PURE_BLACK, apply_pub_rcparams, style_black_axes
from workflow_config import actual_s4_source, plot_output_dir

VALUE_COL = "mean_cohort_balanced_unique_haplotypes_per_window"
LOW_COL = "mean_cohort_balanced_unique_haplotypes_ci_low"
HIGH_COL = "mean_cohort_balanced_unique_haplotypes_ci_high"
FOLD_COL = "relative_to_ukb_reference"
HAPLOTYPE_BAR_COLOR = "#70A6D4"


def load_plot_data(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source, sep="\t" if source.suffix == ".tsv" else ",")
    required = {"scenario", "scenario_label", VALUE_COL, LOW_COL, HIGH_COL, FOLD_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {source}: {sorted(missing)}")
    df = df.copy()
    labels = {
        "UKB_reference": "UKB full",
        "S1_UKB": "UKB subset",
        "S2_UK10K": "+ UK10K",
        "S3_EAS2K": "+ EAS2K",
        "S4_SG10K": "+ SG10K",
        "S5_AFR1K": "+ AFR1K",
        "S6_TOPMed": "+ TOPMed",
        "S7_HGDP": "+ HGDP",
        "S8_1KGP": "+ 1KGP",
    }
    df["display_label"] = df["scenario"].map(labels).fillna(df["scenario_label"])
    df["xerr_low"] = (df[VALUE_COL] - df[LOW_COL]).clip(lower=0.0)
    df["xerr_high"] = (df[HIGH_COL] - df[VALUE_COL]).clip(lower=0.0)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=actual_s4_source())
    parser.add_argument("--out-dir", type=Path, default=plot_output_dir())
    parser.add_argument("--out-stem", default="cohort_haplotype_diversity")
    args = parser.parse_args()
    df = load_plot_data(args.source)
    plot_df = (
        df.loc[~df["scenario"].eq("UKB_reference")].iloc[::-1].reset_index(drop=True)
    )
    y = np.arange(len(plot_df), dtype=float)
    apply_pub_rcparams(base_font_size=11)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(
        y,
        plot_df[VALUE_COL],
        height=0.56,
        color=HAPLOTYPE_BAR_COLOR,
        edgecolor=PURE_BLACK,
        linewidth=0.5,
        xerr=np.vstack([plot_df["xerr_low"], plot_df["xerr_high"]]),
        error_kw={
            "ecolor": PURE_BLACK,
            "elinewidth": 1.0,
            "capsize": 2.8,
            "capthick": 1.0,
        },
    )
    offset = plot_df[HIGH_COL].max() * 0.018
    for i, row in plot_df.iterrows():
        ax.text(
            row[HIGH_COL] + offset,
            y[i],
            f"{row[FOLD_COL]:.2f}x",
            va="center",
            ha="left",
            fontsize=11,
            color=PURE_BLACK,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_label"], fontsize=11, color=PURE_BLACK)
    ax.set_xlabel(
        "Cohort-balanced unique haplotypes per 100 kb window",
        fontsize=13,
        color=PURE_BLACK,
    )
    ax.set_xlim(0, plot_df[HIGH_COL].max() * 1.18)
    style_black_axes(
        ax, labelsize=13, tick_labelsize=11, spine_width=1.0, tick_width=1.0
    )
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / f"{args.out_stem}.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / f"{args.out_stem}.svg", bbox_inches="tight")
    fig.savefig(args.out_dir / f"{args.out_stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_dir / args.out_stem}")


if __name__ == "__main__":
    main()
