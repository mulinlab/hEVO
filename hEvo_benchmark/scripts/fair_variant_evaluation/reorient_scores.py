#!/usr/bin/env python3
"""Reorient standardized LLR scores after a model-blind allele swap.

The source CausalDB scores use the beta-directed paired allele order.  When
the pair is canonicalized to hg38 REF>ALT, an LLR changes sign exactly when
the pair was swapped.  This utility only applies that deterministic sign
change; it never selects, removes, or ranks variants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", type=Path, required=True, help="Standardized beta-oriented TSV"
    )
    parser.add_argument(
        "--canonical-dataset",
        type=Path,
        required=True,
        help="Canonicalized CausalDB CSV",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    scores = read_tsv(args.scores)
    with args.canonical_dataset.open("r", encoding="utf-8", newline="") as handle:
        dataset = list(csv.DictReader(handle))
    if not scores or not dataset:
        raise ValueError("scores and canonical dataset must be non-empty")
    dataset_by_id = {row["seqID"]: row for row in dataset}
    score_by_id = {row["variant_id"]: row for row in scores}
    if len(dataset_by_id) != len(dataset) or len(score_by_id) != len(scores):
        raise ValueError("duplicate variant IDs")
    if set(dataset_by_id) != set(score_by_id):
        raise ValueError(
            f"ID mismatch: dataset={len(dataset_by_id)} scores={len(score_by_id)}"
        )

    output_rows = []
    flipped = 0
    for row in dataset:
        sid = row["seqID"]
        score_row = score_by_id[sid]
        if int(score_row["label"]) != int(row["label"]):
            raise ValueError(f"label mismatch for {sid}")
        sign = -1.0 if row.get("source_orientation") == "swapped" else 1.0
        if sign < 0:
            flipped += 1
        output_rows.append(
            {
                "model": args.model,
                "variant_id": sid,
                "chrom": score_row["chrom"],
                "pos": score_row["pos"],
                "label": row["label"],
                "benchmark_score": sign * float(score_row["benchmark_score"]),
                "score_mode": "neg_llr",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    manifest = {
        "status": "PASS",
        "model": args.model,
        "input_scores": str(args.scores.resolve()),
        "input_scores_sha256": sha256(args.scores),
        "canonical_dataset": str(args.canonical_dataset.resolve()),
        "canonical_dataset_sha256": sha256(args.canonical_dataset),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "rows": len(output_rows),
        "sign_flipped_rows": flipped,
        "score_mode": "neg_llr",
        "selection_used_model_scores": False,
        "transformation": "benchmark_score *= -1 for source_orientation=swapped",
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
