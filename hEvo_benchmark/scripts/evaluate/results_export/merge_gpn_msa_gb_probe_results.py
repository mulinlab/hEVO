#!/usr/bin/env python3
"""Merge a completed coordinate-aware GPN-MSA Genomic Benchmark rerun into S18."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .common import read_csv, sha256, write_csv
except ImportError:  # direct script execution
    from common import read_csv, sha256, write_csv


TASKS = (
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
)
TABLE_NAME = (
    "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv"
)


def result_for_task(run_dir: Path, task: str) -> tuple[dict[str, Any], list[float]]:
    manifest_path = run_dir / task / "results" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "COMPLETED":
        raise ValueError(f"{task}: run manifest is not COMPLETED")
    if manifest.get("model_name") != "GPN-MSA":
        raise ValueError(f"{task}: expected model_name GPN-MSA")
    if manifest.get("dataset_name") != f"GB_official_coord_{task}":
        raise ValueError(
            f"{task}: unexpected dataset name {manifest.get('dataset_name')!r}"
        )

    result_path = Path(manifest["result_json"])
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    actual_hash = sha256(result_path)
    if actual_hash != manifest.get("result_sha256"):
        raise ValueError(f"{task}: result hash does not match run manifest")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(f"{task}: expected one model in result JSON")
    model_result = next(iter(payload.values()))
    expected_dataset = f"GB_official_coord_{task}"
    if not isinstance(model_result, dict) or set(model_result) != {expected_dataset}:
        raise ValueError(f"{task}: result JSON task key does not match run manifest")
    projections = model_result[expected_dataset]
    metrics = (
        projections.get("no_projection") if isinstance(projections, dict) else None
    )
    mcc = metrics.get("mcc") if isinstance(metrics, dict) else None
    if not isinstance(mcc, list) or len(mcc) < 3:
        raise ValueError(f"{task}: result JSON has no MCC mean/lower/upper")
    values = [float(value) for value in mcc[:3]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{task}: MCC contains a non-finite value")
    return manifest, values


def competition_ranks(rows: list[dict[str, str]]) -> None:
    tasks: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        tasks.setdefault(row["dataset"], []).append(row)
    for task_rows in tasks.values():
        task_rows.sort(key=lambda row: (-float(row["mcc_mean"]), row["model"]))
        last_score: float | None = None
        current_rank = 0
        for position, row in enumerate(task_rows, 1):
            score = float(row["mcc_mean"])
            if last_score is None or score != last_score:
                current_rank = position
                last_score = score
            row["task_rank"] = str(current_rank)
            row["task_count"] = str(len(task_rows))


def merge(run_dir: Path, tables_dir: Path) -> list[dict[str, str]]:
    table_path = tables_dir / TABLE_NAME
    fieldnames, rows = read_csv(table_path)
    updated: list[dict[str, str]] = []
    for task in TASKS:
        matches = [
            row for row in rows if row["dataset"] == task and row["model"] == "GPN-MSA"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"S18 must have one existing GPN-MSA row for {task}; got {len(matches)}"
            )
        _, metrics = result_for_task(run_dir, task)
        row = matches[0]
        row["mcc_mean"], row["mcc_lower"], row["mcc_upper"] = (
            format(value, ".15g") for value in metrics
        )
        updated.append(
            {
                "task": task,
                "mcc_mean": row["mcc_mean"],
                "mcc_lower": row["mcc_lower"],
                "mcc_upper": row["mcc_upper"],
            }
        )

    competition_ranks(rows)
    task_order = list(dict.fromkeys(row["dataset"] for row in rows))
    task_position = {task: index for index, task in enumerate(task_order)}
    rows.sort(
        key=lambda row: (
            task_position[row["dataset"]],
            int(row["task_rank"]),
            row["model"],
        )
    )
    write_csv(table_path, fieldnames, rows)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "tables",
    )
    args = parser.parse_args()
    for row in merge(args.run_dir, args.tables_dir):
        print(
            f"{row['task']}: MCC={row['mcc_mean']} "
            f"[{row['mcc_lower']}, {row['mcc_upper']}]"
        )


if __name__ == "__main__":
    main()
