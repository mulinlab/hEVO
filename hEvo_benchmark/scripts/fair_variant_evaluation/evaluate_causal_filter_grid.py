#!/usr/bin/env python3
"""Evaluate every arm in a model-blind CausalDB filter-grid manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def bootstrap_ci(
    labels: list[int], scores: list[float], seed: int, n_bootstrap: int
) -> tuple[float | None, float | None]:
    if n_bootstrap <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(labels)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, n)
        sampled_labels = [labels[index] for index in indices]
        if len(set(sampled_labels)) < 2:
            continue
        values.append(
            float(
                average_precision_score(
                    sampled_labels, [scores[index] for index in indices]
                )
            )
        )
    if not values:
        raise ValueError("bootstrap produced no two-class samples")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--scores-root",
        type=Path,
        required=True,
        help="directory containing C0_full standardized score TSVs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model-count", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    score_files = sorted(args.scores_root.glob("*.tsv"))
    if not score_files:
        raise ValueError(f"no score files in {args.scores_root}")
    # Load each complete score table once. Re-reading the 32 files for every
    # arm makes the all-candidate bootstrap needlessly I/O bound.
    score_tables: dict[Path, tuple[dict[str, dict[str, str]], str]] = {}
    for score_path in score_files:
        with score_path.open("r", encoding="utf-8", newline="") as handle:
            score_rows = list(csv.DictReader(handle, delimiter="\t"))
        by_id = {row.get("variant_id", ""): row for row in score_rows}
        if len(by_id) != len(score_rows) or "" in by_id:
            raise ValueError(f"score table has missing or duplicate IDs: {score_path}")
        score_tables[score_path] = (by_id, sha256(score_path))
    for arm in manifest["arms"]:
        if arm.get("status") != "PASS":
            continue
        dataset = Path(str(arm["dataset"]))
        rows = read_csv(dataset)
        ids = [row.get("seqID") or row.get("variant_id") or "" for row in rows]
        if not ids or not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid IDs in {dataset}")
        labels = {sid: int(row["label"]) for sid, row in zip(ids, rows)}
        if set(labels.values()) != {0, 1}:
            raise ValueError(f"both labels are required in {dataset}")
        for score_path in score_files:
            by_id, score_hash = score_tables[score_path]
            if len(by_id) != len(score_rows) or not set(ids) <= set(by_id):
                raise ValueError(
                    f"score coverage mismatch for {arm['arm_id']}/{score_path.name}"
                )
            scores = [float(by_id[sid]["benchmark_score"]) for sid in ids]
            score_labels = [int(by_id[sid]["label"]) for sid in ids]
            if score_labels != [labels[sid] for sid in ids]:
                raise ValueError(
                    f"label mismatch for {arm['arm_id']}/{score_path.name}"
                )
            ci_lower, ci_upper = bootstrap_ci(
                score_labels,
                scores,
                args.seed + len(records),
                args.bootstrap,
            )
            records.append(
                {
                    "arm": arm["arm_id"],
                    "model": score_path.stem,
                    "rows": len(rows),
                    "positive": sum(labels.values()),
                    "negative": len(rows) - sum(labels.values()),
                    "baseline_auprc": sum(labels.values()) / len(rows),
                    "auprc": float(average_precision_score(score_labels, scores)),
                    "auroc": float(roc_auc_score(score_labels, scores)),
                    "auprc_ci_lower": ci_lower,
                    "auprc_ci_upper": ci_upper,
                    "bootstrap_n": args.bootstrap,
                    "bootstrap_seed": args.seed + len(records),
                    "bootstrap_resampling_unit": "row_iid",
                    "score_sha256": score_hash,
                    "selection_model_scores_used": False,
                    "status": "PASS",
                }
            )
    if not records:
        raise ValueError("no evaluable PASS arms")
    by_arm: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_arm.setdefault(str(record["arm"]), []).append(record)
    for arm_records in by_arm.values():
        ordered = sorted(
            arm_records, key=lambda row: (-float(row["auprc"]), str(row["model"]))
        )
        for rank, record in enumerate(ordered, start=1):
            record["rank"] = rank
            record["roster_status"] = (
                "FULL_ROSTER"
                if len(ordered) == args.expected_model_count
                else "PARTIAL_ROSTER"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            sorted(records, key=lambda row: (str(row["arm"]), int(row["rank"])))
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "arms": len(by_arm),
                "records": len(records),
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
