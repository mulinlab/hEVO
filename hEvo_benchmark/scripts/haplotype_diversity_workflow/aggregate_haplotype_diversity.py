#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from workflow_config import results_root


SCENARIO_ORDER = [
    "UKB_reference",
    "S1_UKB",
    "S2_UK10K",
    "S3_EAS2K",
    "S4_SG10K",
    "S5_AFR1K",
    "S6_TOPMed",
    "S7_HGDP",
    "S8_1KGP",
]


def bootstrap_mean(
    values: np.ndarray, repeats: int, seed: int
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for i in range(repeats):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate chr-level haplotype diversity windows into Fig1 source data."
    )
    parser.add_argument(
        "--in-dir", type=Path, default=results_root() / "haplotype_diversity"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=results_root()
        / "haplotype_diversity"
        / "fig1_haplotype_diversity_summary.tsv",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_files = sorted(
        args.in_dir.glob("window_haplotype_diversity.chr*.shard*-of-*.tsv.gz")
    )
    files = shard_files or sorted(
        args.in_dir.glob("window_haplotype_diversity.chr*.tsv.gz")
    )
    if not files:
        raise FileNotFoundError(f"No chr window files found in {args.in_dir}")
    tables = []
    for path in files:
        table = pd.read_csv(path, sep="\t")
        if not table.empty:
            tables.append(table)
    if not tables:
        raise ValueError(f"All window files in {args.in_dir} were empty")
    df = pd.concat(tables, ignore_index=True)

    rows = []
    for scenario, sub in df.groupby("scenario", sort=False):
        mean_richness, ci_low, ci_high = bootstrap_mean(
            sub["rarefied_unique_mean"].to_numpy(dtype=float),
            repeats=args.bootstrap_repeats,
            seed=args.seed + SCENARIO_ORDER.index(scenario)
            if scenario in SCENARIO_ORDER
            else args.seed,
        )
        balanced_mean, balanced_low, balanced_high = bootstrap_mean(
            sub["balanced_rarefied_unique_mean"].to_numpy(dtype=float),
            repeats=args.bootstrap_repeats,
            seed=args.seed + 100 + SCENARIO_ORDER.index(scenario)
            if scenario in SCENARIO_ORDER
            else args.seed + 100,
        )
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": sub["scenario_label"].iloc[0],
                "cohorts": sub["cohorts"].iloc[0],
                "n_windows": len(sub),
                "n_haplotypes": int(sub["n_haplotypes"].median()),
                "rarefy_target_haplotypes": int(
                    sub["rarefy_target_haplotypes"].median()
                ),
                "mean_rarefied_unique_haplotypes_per_window": mean_richness,
                "mean_rarefied_unique_haplotypes_ci_low": ci_low,
                "mean_rarefied_unique_haplotypes_ci_high": ci_high,
                "balanced_rarefy_target_per_unit": int(
                    sub["balanced_rarefy_target_per_unit"].median()
                ),
                "mean_cohort_balanced_unique_haplotypes_per_window": balanced_mean,
                "mean_cohort_balanced_unique_haplotypes_ci_low": balanced_low,
                "mean_cohort_balanced_unique_haplotypes_ci_high": balanced_high,
                "total_unique_haplotype_windows": int(sub["n_unique_haplotypes"].sum()),
                "total_novel_haplotype_windows_vs_ukb_full": int(
                    sub["n_novel_vs_ukb_full"].sum()
                ),
                "mean_novel_haplotype_windows_vs_ukb_full": float(
                    sub["n_novel_vs_ukb_full"].mean()
                ),
                "total_snv_records_scanned": int(sub["n_snv_records"].sum()),
                "total_raw_records_scanned": int(sub["n_raw_records"].sum()),
            }
        )

    out = pd.DataFrame(rows)
    out["scenario"] = pd.Categorical(
        out["scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    out = out.sort_values("scenario")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
