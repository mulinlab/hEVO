#!/usr/bin/env python3
"""Summarize fair-summary score files with counts, baseline, CIs, and ranks."""

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


def bootstrap(
    y: np.ndarray, score: np.ndarray, groups: list[list[int]], seed: int, n_boot: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        choices = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([np.asarray(groups[i], dtype=int) for i in choices])
        if y[idx].sum() in (0, len(idx)):
            continue
        values.append(float(average_precision_score(y[idx], score[idx])))
    if not values:
        raise ValueError("bootstrap produced no two-class samples")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def variant_id(row: dict[str, str]) -> str:
    return row.get("variant_id") or row.get("seqID") or row.get("SeqID") or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--score", action="append", required=True, help="ARM:MODEL:SCORE_TSV"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--expected-model-count", type=int, default=32)
    parser.add_argument(
        "--roster-label",
        default="full-benchmark",
        help="Label for the evaluated roster in rank_status (default: full-benchmark)",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    # A single-arm prospective builder writes the arm record at the manifest
    # root; multi-arm builders write an ``arms`` list.  Normalize both
    # schemas here without weakening any row/ID/label checks below.
    arm_records = manifest.get("arms")
    if arm_records is None:
        arm_records = [manifest]
    arm_info = {item["arm_id"]: item for item in arm_records}
    records: list[dict[str, object]] = []
    for specification in args.score:
        try:
            arm, model, score_path_text = specification.split(":", 2)
        except ValueError as exc:
            raise ValueError("--score must be ARM:MODEL:SCORE_TSV") from exc
        if arm not in arm_info:
            raise ValueError(f"unknown arm: {arm}")
        score_path = Path(score_path_text)
        rows = read_rows(score_path)
        if not rows:
            raise ValueError(f"empty score file: {score_path}")
        y = np.asarray([int(row["label"]) for row in rows], dtype=int)
        score = np.asarray([float(row["benchmark_score"]) for row in rows], dtype=float)
        if not np.isfinite(score).all() or set(y) != {0, 1}:
            raise ValueError(f"invalid score/labels in {score_path}")
        # Support both manifest schemas used by the summary: older manifests
        # nested counts under ``summary``, while the frozen causal manifest
        # stores the same counts directly on each arm record.
        summary = arm_info[arm].get("summary", arm_info[arm])
        if "rows" not in summary:
            raise ValueError(f"arm manifest lacks row count for {arm}")
        if len(rows) != int(summary["rows"]):
            raise ValueError(f"row-count mismatch for {arm}/{model}")
        groups = [[i] for i in range(len(rows))]
        if arm.startswith("R"):
            loci: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
            for i, row in enumerate(rows):
                loci[(row["chrom"].replace("chr", ""), row["pos"])].append(i)
            groups = list(loci.values())
        ap = float(average_precision_score(y, score))
        is_locus = arm_info[arm].get("evaluation_unit") == "locus"
        ci_lower, ci_upper = bootstrap(y, score, groups, 20260913, args.bootstrap)
        task = arm_info[arm].get("task", "causal")
        record = {
            "arm": arm,
            "task": task,
            "model": model,
            "evaluation_unit": "locus" if is_locus else "variant",
            "rows": len(rows),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "baseline_auprc": float(y.mean()),
            "auprc": ap,
            "auroc": float(roc_auc_score(y, score)),
            "auprc_ci_lower": ci_lower,
            "auprc_ci_upper": ci_upper,
            "score_file": str(score_path.resolve()),
            "score_file_sha256": sha256(score_path),
            "bootstrap_resampling_unit": "locus_iid"
            if is_locus
            else ("locus_cluster" if arm.startswith("R") else "row_iid"),
            "status": "PASS",
        }
        records.append(record)
    by_arm: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_arm[str(record["arm"])].append(record)
    for arm, arm_records in by_arm.items():
        ordered = sorted(
            arm_records,
            key=lambda record: (-float(record["auprc"]), str(record["model"])),
        )
        for rank, record in enumerate(ordered, start=1):
            record["rank"] = rank
        if len(arm_records) < args.expected_model_count:
            for record in arm_records:
                record["rank_status"] = "PARTIAL_ROSTER"
        else:
            for record in arm_records:
                record["rank_status"] = (
                    "FULL_ROSTER"
                    if args.roster_label == "full-benchmark"
                    else f"FULL_{args.roster_label.upper().replace('-', '_')}_ROSTER"
                )
    if not records:
        raise ValueError("no score records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                records,
                key=lambda record: (str(record["arm"]), int(record.get("rank", 0))),
            )
        )
    print(
        json.dumps(
            {
                "status": "PASS",
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
