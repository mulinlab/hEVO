#!/usr/bin/env python3
"""Summarize one allele-preference score TSV without rerunning inference."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="Score TSV (original positional interface)"
    )
    parser.add_argument(
        "--output", type=Path, help="Default: summary_results.json beside score TSV"
    )
    args = parser.parse_args()
    from scipy.stats import beta

    path = args.path
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    passed = [
        r
        for r in rows
        if r.get("status") == "PASS" and r.get("locus_score") not in (None, "")
    ]
    correct = sum(float(r["locus_score"]) > 0 for r in passed)
    n = len(passed)
    alpha = 0.05
    if n:
        low = 0.0 if correct == 0 else beta.ppf(alpha / 2, correct, n - correct + 1)
        high = (
            1.0 if correct == n else beta.ppf(1 - alpha / 2, correct + 1, n - correct)
        )
    else:
        low, high = None, None
    summary = {
        "model_id": rows[0].get("model_id") if rows else None,
        "n_loci": n,
        "n_loci_expected": 30,
        "pass_rows": len(passed),
        "fail_rows": len(rows) - len(passed),
        "directional_correct": int(correct),
        "directional_accuracy": float(correct / n) if n else None,
        "directional_accuracy_clopper_pearson_95ci": [low, high],
        "score_file": str(path),
    }
    (args.output or path.with_name("summary_results.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
