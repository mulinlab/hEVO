#!/usr/bin/env python3
"""Aggregate R0 variant scores to the preregistered R1 locus estimand."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(
            csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
        )


def variant_id(row: dict[str, str]) -> str:
    return row.get("variant_id") or row.get("seqID") or row.get("SeqID") or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="R0 CSV")
    parser.add_argument(
        "--scores", type=Path, required=True, help="standardized TSV scores"
    )
    parser.add_argument("--output", type=Path, required=True, help="R1 TSV")
    parser.add_argument("--summary", type=Path, required=True, help="R1 JSON summary")
    parser.add_argument(
        "--method", choices=("first", "mean", "maximum"), default="mean"
    )
    args = parser.parse_args()
    dataset = read_rows(args.dataset)
    scores = read_rows(args.scores)
    score_by_id = {variant_id(row): float(row["benchmark_score"]) for row in scores}
    if len(score_by_id) != len(scores):
        raise ValueError("score file contains duplicate or empty variant_id")
    loci: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in dataset:
        loci[(row["chrom"].replace("chr", ""), row["pos"])].append(row)
    output_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for (chrom, pos), rows in sorted(loci.items()):
        ids = [variant_id(row) for row in rows]
        if any(item not in score_by_id for item in ids):
            missing.extend(item for item in ids if item not in score_by_id)
            continue
        values = [score_by_id[item] for item in ids]
        if args.method == "first":
            score = values[0]
        elif args.method == "maximum":
            score = max(values)
        else:
            score = float(np.mean(values))
        output_rows.append(
            {
                "locus_id": f"chr{chrom}:{pos}",
                "chrom": chrom,
                "pos": pos,
                "label": max(int(row["label"]) for row in rows),
                "benchmark_score": score,
                "variant_ids": ",".join(ids),
            }
        )
    if missing:
        raise ValueError(
            f"missing scores for {len(missing)} variants; first={missing[:5]}"
        )
    if not output_rows or {row["label"] for row in output_rows} != {0, 1}:
        raise ValueError("R1 output must contain both labels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    labels = np.asarray([int(row["label"]) for row in output_rows])
    values = np.asarray([float(row["benchmark_score"]) for row in output_rows])
    summary = {
        "schema": "hevo_fair_variant_evaluation_locus_metrics_v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "scores": str(args.scores.resolve()),
        "scores_sha256": sha256(args.scores),
        "aggregation_method": args.method,
        "evaluation_unit": "locus",
        "rows": len(output_rows),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "baseline_auprc": float(labels.mean()),
        "auprc": float(average_precision_score(labels, values)),
        "auroc": float(roc_auc_score(labels, values)),
        "status": "PASS",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
