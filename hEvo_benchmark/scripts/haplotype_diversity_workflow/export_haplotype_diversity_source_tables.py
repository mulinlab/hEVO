#!/usr/bin/env python3
"""Export the pinned haplotype-diversity sources into GitLab source tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from workflow_config import actual_s3_source, actual_s4_source

EXPECTED_S3_SOURCE_SHA256 = (
    "82e9a6b9b671b9b902245f53290718187be72c08dfb625696328a7ee2fd58f06"
)
EXPECTED_S4_SOURCE_SHA256 = (
    "8f0f06c9c2e61a6201b65876fa369d3bad784d7b371310c81bd5a55b0663ce52"
)
S3_COLUMNS = [
    "target",
    "reference_haplotype_frequency_threshold",
    "threshold_weighted_coverage",
    "threshold_set_coverage",
    "n_reference_haplotypes_above_threshold",
    "reference_weight_above_threshold",
]
S4_SCENARIOS = [
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
S4_COLUMNS = [
    "scenario",
    "scenario_label",
    "is_cumulative_series",
    "n_windows",
    "mean_cohort_balanced_unique_haplotypes_per_window",
    "mean_cohort_balanced_unique_haplotypes_ci_low",
    "mean_cohort_balanced_unique_haplotypes_ci_high",
    "relative_to_ukb_reference",
    "step_gain",
    "total_novel_haplotype_windows_vs_ukb_full",
    "mean_novel_haplotype_windows_vs_ukb_full",
    "total_snv_records_scanned",
]
S4_OUTPUT_COLUMNS = S4_COLUMNS + ["display_label", "xerr_low", "xerr_high"]
DISPLAY_LABELS = {
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-source", type=Path, default=actual_s3_source())
    parser.add_argument("--s4-source", type=Path, default=actual_s4_source())
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "tables"
    )
    parser.add_argument(
        "--provenance-out",
        type=Path,
        help="Optional external path for source/output hashes; omitted for a clean public checkout.",
    )
    parser.add_argument("--allow-source-hash-change", action="store_true")
    return parser.parse_args()


def require_columns(table: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def validate_source_hash(source: Path, expected: str, allow_change: bool) -> str:
    if not source.exists():
        raise FileNotFoundError(source)
    observed = sha256(source)
    if observed != expected and not allow_change:
        raise ValueError(
            f"Pinned source hash mismatch for {source}: expected {expected}, observed {observed}. Use --allow-source-hash-change only for a deliberate source revision."
        )
    return observed


def build_s3(source: Path) -> pd.DataFrame:
    table = pd.read_csv(source, sep="\t")
    require_columns(table, S3_COLUMNS, source)
    if list(table.columns) != S3_COLUMNS:
        table = table[S3_COLUMNS]
    if len(table) != 606:
        raise ValueError(f"Expected 606 S3 rows, found {len(table)}")
    if set(table["target"].dropna().unique()) != {
        "Selected UKB subset",
        "Random UKB downsample",
    }:
        raise ValueError("Unexpected S3 target set")
    if table.isna().any().any():
        raise ValueError("S3 contains missing values")
    for col in S3_COLUMNS[1:]:
        values = pd.to_numeric(table[col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"S3 column {col} contains non-finite values")
    for col in ["threshold_weighted_coverage", "threshold_set_coverage"]:
        if ((table[col] < 0) | (table[col] > 1)).any():
            raise ValueError(f"S3 column {col} is outside [0, 1]")
    return table


def build_s4(source: Path) -> pd.DataFrame:
    table = pd.read_csv(source, sep="\t")
    require_columns(table, S4_COLUMNS, source)
    table = table[S4_COLUMNS].copy()
    if table["scenario"].tolist() != S4_SCENARIOS:
        raise ValueError(f"Unexpected S4 scenario order: {table['scenario'].tolist()}")
    if len(table) != len(S4_SCENARIOS) or not (table["n_windows"] == 577).all():
        raise ValueError("S4 must contain nine 577-window scenarios")
    numeric = [c for c in S4_COLUMNS[3:] if c != "is_cumulative_series"]
    for col in numeric:
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        if col == "step_gain":
            values = values[1:]
        if not np.isfinite(values).all():
            raise ValueError(f"S4 column {col} contains non-finite values")
    if (
        table["mean_cohort_balanced_unique_haplotypes_ci_low"]
        > table["mean_cohort_balanced_unique_haplotypes_per_window"]
    ).any():
        raise ValueError("S4 lower confidence limit exceeds mean")
    if (
        table["mean_cohort_balanced_unique_haplotypes_ci_high"]
        < table["mean_cohort_balanced_unique_haplotypes_per_window"]
    ).any():
        raise ValueError("S4 upper confidence limit is below mean")
    table["display_label"] = table["scenario"].map(DISPLAY_LABELS)
    value = table["mean_cohort_balanced_unique_haplotypes_per_window"]
    table["xerr_low"] = value - table["mean_cohort_balanced_unique_haplotypes_ci_low"]
    table["xerr_high"] = table["mean_cohort_balanced_unique_haplotypes_ci_high"] - value
    return table[S4_OUTPUT_COLUMNS]


def main() -> None:
    args = parse_args()
    s3_hash = validate_source_hash(
        args.s3_source, EXPECTED_S3_SOURCE_SHA256, args.allow_source_hash_change
    )
    s4_hash = validate_source_hash(
        args.s4_source, EXPECTED_S4_SOURCE_SHA256, args.allow_source_hash_change
    )
    s3 = build_s3(args.s3_source)
    s4 = build_s4(args.s4_source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    s3_out = (
        args.out_dir
        / "S5_coverage_of_uk_biobank_reference_haplotypes_by_the_selected_subset.csv"
    )
    s4_out = (
        args.out_dir
        / "S6_local_haplotype_diversity_after_sequential_cohort_integration.csv"
    )
    s3.to_csv(s3_out, index=False)
    s4.to_csv(s4_out, index=False)
    records = [
        {
            "artifact": "S3",
            "source": str(args.s3_source),
            "source_sha256": s3_hash,
            "source_rows": len(s3),
            "output": str(s3_out),
            "output_sha256": sha256(s3_out),
            "output_rows": len(s3),
        },
        {
            "artifact": "S4",
            "source": str(args.s4_source),
            "source_sha256": s4_hash,
            "source_rows": len(pd.read_csv(args.s4_source, sep="\t")),
            "output": str(s4_out),
            "output_sha256": sha256(s4_out),
            "output_rows": len(s4),
        },
    ]
    if args.provenance_out is not None:
        args.provenance_out.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_out.write_text(
            "artifact\tsource\tsource_sha256\tsource_rows\toutput\toutput_sha256\toutput_rows\n"
            + "\n".join(
                "\t".join(
                    str(row[k])
                    for k in [
                        "artifact",
                        "source",
                        "source_sha256",
                        "source_rows",
                        "output",
                        "output_sha256",
                        "output_rows",
                    ]
                )
                for row in records
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps({"s3": records[0], "s4": records[1]}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
