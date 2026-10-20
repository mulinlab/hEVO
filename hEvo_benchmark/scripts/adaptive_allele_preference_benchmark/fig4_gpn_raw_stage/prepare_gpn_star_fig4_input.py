#!/usr/bin/env python3
"""Create the normalized 30-locus input used by the GPN-Star raw-logit run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "task_name",
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "selected_allele",
    "other_allele",
    "score_mode",
    "source_row",
    "input_status",
    "failure_reason",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 30:
        raise ValueError(f"expected 30 loci, found {len(rows)}")

    output = []
    for index, row in enumerate(rows):
        selected = row["selected_allele"].upper()
        ref = row["vcf_ref"].upper()
        alt = row["vcf_alt"].upper()
        if selected not in {ref, alt} or len(ref) != 1 or len(alt) != 1:
            raise ValueError(
                f"row {index}: selected/ref/alt are not a biallelic SNV: "
                f"{selected=} {ref=} {alt=}"
            )
        other = alt if selected == ref else ref
        output.append(
            {
                "task_name": "adaptive_allele_preference_fig4f",
                "variant_id": row["variant_id"],
                "chrom": row["chrom"],
                "pos": row["pos"],
                "ref": ref,
                "alt": alt,
                "label": "-1",
                "selected_allele": selected,
                "other_allele": other,
                "score_mode": "selected_other",
                "source_row": str(index),
                "input_status": "PASS",
                "failure_reason": "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {args.output} rows={len(output)}")


if __name__ == "__main__":
    main()
