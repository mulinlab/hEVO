#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate empirical UKB downsampling VCF coverage."
    )
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--n-grid", type=int, default=300)
    parser.add_argument("--marker-thresholds", default="0.01,0.001,0.0001,0.00001")
    return parser.parse_args()


def load_rows(in_dir: Path) -> pd.DataFrame:
    paths = sorted(in_dir.glob("ukb_downsampling_empirical.chr*.tsv.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No ukb_downsampling_empirical.chr*.tsv.gz files found in {in_dir}"
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(
            f"All empirical downsampling shard files in {in_dir} are empty"
        )
    return pd.concat(frames, ignore_index=True)


def thresholds_from_freq(
    freq: pd.Series, n_grid: int, marker_thresholds: list[float]
) -> np.ndarray:
    positive = freq[freq > 0].to_numpy(dtype=float)
    grid = np.geomspace(float(positive.max()), float(positive.min()), n_grid)
    markers = np.array(
        [t for t in marker_thresholds if positive.min() <= t <= positive.max()]
    )
    return np.sort(np.unique(np.concatenate([grid, markers])))[::-1]


def aggregate(df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    freq = df["reference_haplotype_frequency"].to_numpy(dtype=float)
    weight = df["reference_haplotype_count"].to_numpy(dtype=float)
    selected = df["covered_by_selected_ukb_subset"].to_numpy(dtype=float)
    random_mean = df["random_cover_count"].to_numpy(dtype=float) / df[
        "random_repeats"
    ].to_numpy(dtype=float)
    for target, covered in [
        ("Selected UKB subset", selected),
        ("Random UKB downsample", random_mean),
    ]:
        for threshold in thresholds:
            keep = freq >= threshold
            denom = weight[keep].sum()
            rows.append(
                {
                    "target": target,
                    "reference_haplotype_frequency_threshold": threshold,
                    "threshold_weighted_coverage": float(
                        (weight[keep] * covered[keep]).sum() / denom
                    ),
                    "threshold_set_coverage": float(covered[keep].mean()),
                    "n_reference_haplotypes_above_threshold": int(keep.sum()),
                    "reference_weight_above_threshold": float(denom),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    if df["n_selected_samples"].nunique() != 1:
        raise ValueError("Coverage shards have inconsistent selected sample counts")
    rows = []
    for target in ["Selected UKB subset", "Random UKB downsample"]:
        sub = curve[curve["target"] == target]
        row = {
            "target": target,
            "n_windows": int(
                df[["chrom", "window_start", "window_end"]].drop_duplicates().shape[0]
            ),
            "n_reference_haplotypes": int(len(df)),
            "n_reference_samples": int(df["n_reference_samples"].median()),
            "n_selected_samples": int(df["n_selected_samples"].median()),
            "random_repeats": int(df["random_repeats"].median()),
        }
        for threshold in [0.01, 0.001, 0.0001, 0.00001]:
            if threshold < sub["reference_haplotype_frequency_threshold"].min():
                continue
            nearest = sub.iloc[
                (sub["reference_haplotype_frequency_threshold"] - threshold)
                .abs()
                .argmin()
            ]
            row[f"weighted_coverage_ge_{threshold:.0e}"] = nearest[
                "threshold_weighted_coverage"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    markers = [float(x) for x in args.marker_thresholds.split(",") if x.strip()]
    df = load_rows(args.in_dir)
    thresholds = thresholds_from_freq(
        df["reference_haplotype_frequency"], args.n_grid, markers
    )
    curve = aggregate(df, thresholds)
    summary = summarize(df, curve)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.out_prefix.with_suffix(".curve.tsv"), sep="\t", index=False)
    summary.to_csv(args.out_prefix.with_suffix(".summary.tsv"), sep="\t", index=False)
    print(f"Wrote {args.out_prefix.with_suffix('.curve.tsv')}")
    print(f"Wrote {args.out_prefix.with_suffix('.summary.tsv')}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
