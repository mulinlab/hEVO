#!/usr/bin/env python3
"""Validate the published S4/S5 table contract and report SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_S3_ROWS = 606
EXPECTED_S4_SCENARIOS = [
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s3",
        type=Path,
        default=root
        / "tables/S5_coverage_of_uk_biobank_reference_haplotypes_by_the_selected_subset.csv",
    )
    parser.add_argument(
        "--s4",
        type=Path,
        default=root
        / "tables/S6_local_haplotype_diversity_after_sequential_cohort_integration.csv",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def check(args: argparse.Namespace) -> dict:
    s3 = pd.read_csv(args.s3)
    s4 = pd.read_csv(args.s4)
    errors = []
    if len(s3) != EXPECTED_S3_ROWS:
        errors.append(f"S3 rows={len(s3)} expected={EXPECTED_S3_ROWS}")
    if set(s3["target"].unique()) != {"Selected UKB subset", "Random UKB downsample"}:
        errors.append("S3 target set mismatch")
    if s3.isna().any().any():
        errors.append("S3 contains NA")
    if (
        (s3["threshold_weighted_coverage"] < 0)
        | (s3["threshold_weighted_coverage"] > 1)
    ).any():
        errors.append("S3 weighted coverage outside [0,1]")
    if s4["scenario"].tolist() != EXPECTED_S4_SCENARIOS:
        errors.append("S4 scenario order mismatch")
    if len(s4) != 9 or not (s4["n_windows"] == 577).all():
        errors.append("S4 row/window contract mismatch")
    for col in [
        "mean_cohort_balanced_unique_haplotypes_per_window",
        "mean_cohort_balanced_unique_haplotypes_ci_low",
        "mean_cohort_balanced_unique_haplotypes_ci_high",
        "relative_to_ukb_reference",
        "xerr_low",
        "xerr_high",
    ]:
        values = pd.to_numeric(s4[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"S4 non-finite column={col}")
    if (s4["xerr_low"] < 0).any() or (s4["xerr_high"] < 0).any():
        errors.append("S4 negative error width")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "s3": {"path": str(args.s3), "rows": len(s3), "sha256": sha256(args.s3)},
        "s4": {"path": str(args.s4), "rows": len(s4), "sha256": sha256(args.s4)},
    }
    return result


def main() -> None:
    args = parse_args()
    result = check(args)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
