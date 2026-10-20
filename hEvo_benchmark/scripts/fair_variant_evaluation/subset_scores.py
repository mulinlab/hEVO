#!/usr/bin/env python3
"""Project a complete score table onto a frozen model-blind dataset subset."""

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


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="frozen subset CSV")
    parser.add_argument(
        "--scores", type=Path, required=True, help="complete standardized score TSV"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        dataset = list(csv.DictReader(handle))
    score_rows = read(args.scores)
    if not dataset or not score_rows:
        raise ValueError("dataset and scores must be non-empty")
    dataset_ids = [
        row.get("seqID") or row.get("SeqID") or row.get("variant_id") or ""
        for row in dataset
    ]
    scores_by_id = {row.get("variant_id", ""): row for row in score_rows}
    if not all(dataset_ids) or len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("frozen subset has missing or duplicate IDs")
    if len(scores_by_id) != len(score_rows):
        raise ValueError("complete score table has missing or duplicate IDs")
    if not set(dataset_ids) <= set(scores_by_id):
        raise ValueError("complete score table does not cover frozen subset")
    output_rows: list[dict[str, str]] = []
    for row, sid in zip(dataset, dataset_ids):
        score = scores_by_id[sid]
        if str(row["label"]) != str(score["label"]):
            raise ValueError(f"label mismatch for {sid}")
        output_rows.append(score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    manifest = {
        "status": "PASS",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "complete_scores": str(args.scores.resolve()),
        "complete_scores_sha256": sha256(args.scores),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "rows": len(output_rows),
        "selection_model_scores_used": False,
        "operation": "ID projection onto a pre-frozen dataset subset",
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
