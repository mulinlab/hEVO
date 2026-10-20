#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_config import results_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact haplotype diversity plotting source."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=results_root()
        / "haplotype_diversity"
        / "fig1_haplotype_diversity_summary.tsv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=results_root()
        / "haplotype_diversity"
        / "fig1_haplotype_diversity_plot_source.tsv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary, sep="\t")
    value_col = "mean_cohort_balanced_unique_haplotypes_per_window"
    low_col = "mean_cohort_balanced_unique_haplotypes_ci_low"
    high_col = "mean_cohort_balanced_unique_haplotypes_ci_high"
    base = df.loc[df["scenario"].eq("UKB_reference"), value_col].iloc[0]
    df["relative_to_ukb_reference"] = df[value_col] / base
    df["step_gain"] = df[value_col].diff()
    df.loc[df["scenario"].eq("UKB_reference"), "step_gain"] = pd.NA
    df["is_cumulative_series"] = ~df["scenario"].eq("UKB_reference")
    out = df[
        [
            "scenario",
            "scenario_label",
            "is_cumulative_series",
            "n_windows",
            value_col,
            low_col,
            high_col,
            "relative_to_ukb_reference",
            "step_gain",
            "total_novel_haplotype_windows_vs_ukb_full",
            "mean_novel_haplotype_windows_vs_ukb_full",
            "total_snv_records_scanned",
        ]
    ].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
