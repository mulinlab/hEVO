#!/usr/bin/env python3
"""Compute preregistered paired AUPRC differences for two score files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def key(row: dict[str, str]) -> str:
    return row.get("variant_id") or row.get("locus_id") or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--left-score", type=Path, required=True)
    parser.add_argument("--right-model", required=True)
    parser.add_argument("--right-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    arm_records = manifest.get("arms")
    if arm_records is None:
        arm_records = [manifest]
    arm_info = {item["arm_id"]: item for item in arm_records}
    if args.arm not in arm_info:
        raise ValueError(f"unknown arm: {args.arm}")
    left, right = read(args.left_score), read(args.right_score)
    left_by, right_by = (
        {key(row): row for row in left},
        {key(row): row for row in right},
    )
    if not left_by or len(left_by) != len(left) or len(right_by) != len(right):
        raise ValueError("score files contain empty or duplicate IDs")
    if set(left_by) != set(right_by):
        raise ValueError("paired score files have different ID sets")
    ordered = list(left_by)
    labels = np.asarray([int(left_by[item]["label"]) for item in ordered], dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("paired scores must contain both labels")
    if any(
        int(left_by[item]["label"]) != int(right_by[item]["label"]) for item in ordered
    ):
        raise ValueError("paired score labels differ")
    left_values = np.asarray(
        [float(left_by[item]["benchmark_score"]) for item in ordered]
    )
    right_values = np.asarray(
        [float(right_by[item]["benchmark_score"]) for item in ordered]
    )
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("paired scores contain non-finite values")
    groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(ordered):
        row = left_by[item]
        groups[(row.get("chrom", ""), row.get("pos", ""))].append(index)
    group_list = list(groups.values())
    if arm_info[args.arm].get("evaluation_unit") == "locus":
        group_list = [[index] for index in range(len(ordered))]
    observed_left = float(average_precision_score(labels, left_values))
    observed_right = float(average_precision_score(labels, right_values))
    rng = np.random.default_rng(args.seed)
    deltas: list[float] = []
    for _ in range(args.bootstrap):
        choices = rng.integers(0, len(group_list), len(group_list))
        indices = np.concatenate(
            [np.asarray(group_list[i], dtype=int) for i in choices]
        )
        if labels[indices].sum() in (0, len(indices)):
            continue
        deltas.append(
            float(
                average_precision_score(labels[indices], right_values[indices])
                - average_precision_score(labels[indices], left_values[indices])
            )
        )
    if not deltas:
        raise ValueError("paired bootstrap produced no two-class samples")
    output = {
        "schema": "hevo_fair_variant_evaluation_paired_auprc_v1",
        "arm": args.arm,
        "evaluation_unit": arm_info[args.arm].get("evaluation_unit", "variant"),
        "rows": len(ordered),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "baseline_auprc": float(labels.mean()),
        "left_model": args.left_model,
        "left_auprc": observed_left,
        "right_model": args.right_model,
        "right_auprc": observed_right,
        "delta_right_minus_left": observed_right - observed_left,
        "delta_ci_lower": float(np.quantile(deltas, 0.025)),
        "delta_ci_upper": float(np.quantile(deltas, 0.975)),
        "bootstrap_n": args.bootstrap,
        "bootstrap_seed": args.seed,
        "bootstrap_resampling_unit": "locus_iid"
        if arm_info[args.arm].get("evaluation_unit") == "locus"
        else "locus_cluster"
        if args.arm.startswith("R")
        else "row_iid",
        "left_score_sha256": sha256(args.left_score),
        "right_score_sha256": sha256(args.right_score),
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
