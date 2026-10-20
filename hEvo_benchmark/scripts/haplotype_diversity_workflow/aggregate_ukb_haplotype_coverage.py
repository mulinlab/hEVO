#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate UKB-full haplotype coverage shard outputs."
    )
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument(
        "--marker-thresholds",
        default="0.01,0.001,0.0001,0.00001,0.000001",
        help="Comma-separated reference haplotype frequency thresholds to force into the curve.",
    )
    parser.add_argument("--n-grid", type=int, default=300)
    return parser.parse_args()


def load_rows(in_dir: Path) -> pd.DataFrame:
    paths = sorted(in_dir.glob("ukb_haplotype_coverage.chr*.tsv.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No ukb_haplotype_coverage.chr*.tsv.gz files found in {in_dir}"
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(f"All shard files in {in_dir} are empty")
    return pd.concat(frames, ignore_index=True)


def build_thresholds(
    freq: pd.Series, marker_thresholds: list[float], n_grid: int
) -> np.ndarray:
    positive = freq[freq > 0].to_numpy(dtype=float)
    if len(positive) == 0:
        raise ValueError("No positive UKB-full haplotype frequencies found")
    lo = float(positive.min())
    hi = float(positive.max())
    grid = np.geomspace(hi, lo, n_grid)
    markers = np.array([t for t in marker_thresholds if lo <= t <= hi], dtype=float)
    return np.sort(np.unique(np.concatenate([grid, markers])))[::-1]


def aggregate_curve(df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["covered_by_final_cohort_union"] = np.maximum(
        df["covered_by_ukb_subset"].to_numpy(dtype=int),
        df["covered_by_final_cohort"].to_numpy(dtype=int),
    )
    df["random_ukb_downsample_expected"] = 1.0 - np.power(
        1.0 - df["ukb_full_haplotype_frequency"].to_numpy(dtype=float),
        df["n_ukb_subset_haplotypes_in_window"].to_numpy(dtype=float),
    )
    targets = [
        ("Selected UKB subset", "covered_by_ukb_subset"),
        ("Random UKB downsample", "random_ukb_downsample_expected"),
    ]
    freq = df["ukb_full_haplotype_frequency"].to_numpy(dtype=float)
    weight = df["ukb_full_haplotype_count"].to_numpy(dtype=float)
    for target, column in targets:
        covered = df[column].to_numpy(dtype=float)
        for threshold in thresholds:
            keep = freq >= threshold
            denom = weight[keep].sum()
            if denom <= 0:
                weighted = np.nan
                set_cov = np.nan
                n_ref = 0
            else:
                weighted = float((weight[keep] * covered[keep]).sum() / denom)
                set_cov = float(covered[keep].mean())
                n_ref = int(keep.sum())
            rows.append(
                {
                    "target": target,
                    "ukb_haplotype_frequency_threshold": threshold,
                    "threshold_weighted_coverage": weighted,
                    "threshold_set_coverage": set_cov,
                    "n_reference_haplotypes_above_threshold": n_ref,
                    "reference_weight_above_threshold": float(denom),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["covered_by_final_cohort_union"] = np.maximum(
        df["covered_by_ukb_subset"].to_numpy(dtype=int),
        df["covered_by_final_cohort"].to_numpy(dtype=int),
    )
    df["random_ukb_downsample_expected"] = 1.0 - np.power(
        1.0 - df["ukb_full_haplotype_frequency"].to_numpy(dtype=float),
        df["n_ukb_subset_haplotypes_in_window"].to_numpy(dtype=float),
    )
    marker_rows = curve[
        curve["ukb_haplotype_frequency_threshold"].isin(
            [0.01, 0.001, 0.0001, 0.00001, 0.000001]
        )
    ]
    for target, column in [
        ("Selected UKB subset", "covered_by_ukb_subset"),
        ("Random UKB downsample", "random_ukb_downsample_expected"),
    ]:
        total_weight = df["ukb_full_haplotype_count"].sum()
        weighted = float(
            (df["ukb_full_haplotype_count"] * df[column]).sum() / total_weight
        )
        rows.append(
            {
                "target": target,
                "n_windows": int(
                    df[["chrom", "window_start", "window_end"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_ukb_full_reference_haplotypes": int(len(df)),
                "weighted_coverage_all_observed_reference_haplotypes": weighted,
                "set_coverage_all_observed_reference_haplotypes": float(
                    df[column].mean()
                ),
            }
        )
        for _, marker in marker_rows[marker_rows["target"] == target].iterrows():
            rows[-1][
                f"weighted_coverage_ge_{marker['ukb_haplotype_frequency_threshold']:.0e}"
            ] = marker["threshold_weighted_coverage"]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    marker_thresholds = [
        float(x) for x in args.marker_thresholds.split(",") if x.strip()
    ]
    df = load_rows(args.in_dir)
    thresholds = build_thresholds(
        df["ukb_full_haplotype_frequency"], marker_thresholds, args.n_grid
    )
    curve = aggregate_curve(df, thresholds)
    summary = summarize(df, curve)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.out_prefix.with_suffix(".curve.tsv"), sep="\t", index=False)
    summary.to_csv(args.out_prefix.with_suffix(".summary.tsv"), sep="\t", index=False)
    print(f"Wrote {args.out_prefix.with_suffix('.curve.tsv')}")
    print(f"Wrote {args.out_prefix.with_suffix('.summary.tsv')}")


if __name__ == "__main__":
    main()
