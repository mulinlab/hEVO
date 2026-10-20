#!/usr/bin/env python3
"""Merge coordinate-aware native GPN-MSA NT-v2 probe results into S17."""

from __future__ import annotations

import argparse
import json
import math
import os
import h5py
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import read_csv, sha256, write_csv as _write_csv
except ImportError:  # direct script execution
    from common import read_csv, sha256, write_csv as _write_csv

TABLE_NAME = "S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv"
EXPECTED_TASKS = 18


def validate_embedding_cache(manifest: dict[str, Any], task: str) -> dict[str, Any]:
    args = manifest.get("cli_args", {})
    cache_dir = Path(args.get("hdf5_dir", ""))
    cache_path = cache_dir / f"GPN-MSA_{task}.h5"
    if not cache_path.is_file():
        raise FileNotFoundError(f"{task}: missing native embedding cache {cache_path}")
    counts = {
        "train": int(manifest["split"]["train_rows"]),
        "val": int(manifest["split"]["validation_rows"]),
    }
    audit: dict[str, Any] = {
        "path": str(cache_path),
        "sha256": sha256(cache_path),
        "splits": {},
    }
    with h5py.File(cache_path, "r") as h5:
        if set(h5.keys()) != {"train", "val"}:
            raise ValueError(
                f"{task}: embedding cache must contain train and val groups"
            )
        for split, expected_rows in counts.items():
            group = h5[split]
            embeddings = group["embeddings"]
            labels = group["labels"]
            sample_ids = group["sample_ids"]
            if embeddings.shape != (expected_rows, 768):
                raise ValueError(
                    f"{task}/{split}: unexpected embedding shape {embeddings.shape}"
                )
            if labels.shape != (expected_rows,) or sample_ids.shape != (expected_rows,):
                raise ValueError(
                    f"{task}/{split}: embeddings, labels and sample IDs are misaligned"
                )
            if not np.isfinite(embeddings[:]).all():
                raise ValueError(f"{task}/{split}: non-finite cached embedding")
            expected_cache_metadata = {
                "embedding_backend": "gpn.msa.embedding.EmbeddingInference",
                "gpn_window_size": 128,
                "gpn_center_window_size": 1000,
                "gpn_center_window_policy": "full_interval_up_to_configured_max",
            }
            for key, expected in expected_cache_metadata.items():
                actual = group.attrs.get(key)
                if isinstance(actual, bytes):
                    actual = actual.decode("utf-8")
                if actual != expected:
                    raise ValueError(
                        f"{task}/{split}: cache metadata {key}={actual!r}; expected {expected!r}"
                    )
            mismatch_count = int(group.attrs.get("target_sequence_mismatch_count", -1))
            if mismatch_count < 0:
                raise ValueError(f"{task}/{split}: missing sequence verification count")
            mismatch_report = []
            if mismatch_count:
                if "target_sequence_mismatch_report" not in group:
                    raise ValueError(
                        f"{task}/{split}: mismatch count has no row-level report"
                    )
                raw_report = group["target_sequence_mismatch_report"][()]
                if isinstance(raw_report, bytes):
                    raw_report = raw_report.decode("utf-8")
                mismatch_report = json.loads(raw_report)
                if len(mismatch_report) != mismatch_count:
                    raise ValueError(
                        f"{task}/{split}: mismatch report row count differs from cache metadata"
                    )
            missing_target_rows = []
            for item in mismatch_report:
                coverage = float(item.get("msa_target_coverage", -1))
                chrom = str(item.get("chrom", ""))
                if coverage == 0 and chrom == "Y":
                    missing_target_rows.append(item)
                else:
                    raise ValueError(
                        f"{task}/{split}: target-covered interval does not match benchmark sequence: {item}"
                    )
            audit["splits"][split] = {
                "rows": expected_rows,
                "feature_dim": int(embeddings.shape[1]),
                "chrY_target_missing_rows": len(missing_target_rows),
                "chrY_target_missing_examples": missing_target_rows[:10],
                "other_sequence_mismatch_rows": mismatch_count
                - len(missing_target_rows),
                "finite": True,
            }
    return audit


def read_task_result(
    run_root: Path, task: str, expected_inputs: dict[str, dict[str, Any]]
) -> tuple[list[float], dict[str, Any]]:
    task_root = run_root / "results" / task
    manifest_path = task_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "COMPLETED":
        raise ValueError(f"{task}: run is not COMPLETED")
    if manifest.get("model_name") != "GPN-MSA" or manifest.get("dataset_name") != task:
        raise ValueError(f"{task}: unexpected model/task in manifest")
    args = manifest.get("cli_args", {})
    expected_msa_path = Path(
        os.environ.get(
            "GPN_MSA_89_PATH",
            str(
                Path(
                    os.environ.get(
                        "DNAMODELS_BENCHMARK_ROOT", Path.home() / "DNAModelsBenchmark"
                    )
                )
                / "datasets/GPN_Datasets/multiz100way/89.zarr.zip"
            ),
        )
    )
    if (
        Path(args.get("gpn_msa_path", "")).expanduser().resolve()
        != expected_msa_path.expanduser().resolve()
    ):
        raise ValueError(f"{task}: manifest does not point to the expected 89-way MSA")
    if args.get("gpn_window_size") != 128 or args.get("gpn_center_window_size") != 1000:
        raise ValueError(f"{task}: unexpected GPN-MSA pooling parameters")
    if args.get("seq_col") != 1 or args.get("label_col") != 2 or args.get("sep") != ",":
        raise ValueError(f"{task}: unexpected NT v2 input-column configuration")
    protocol = manifest.get("protocol", {})
    if (
        protocol.get("probe") != "unprojected_linear_classifier"
        or protocol.get("seed") != 40
        or protocol.get("epochs") != 50
    ):
        raise ValueError(f"{task}: unexpected probe protocol")
    expected_files = [expected_inputs["train"], expected_inputs["test"]]
    observed_files = manifest.get("dataset_files", [])
    if len(observed_files) != 2:
        raise ValueError(f"{task}: expected train and test inputs in run manifest")
    for observed, expected in zip(observed_files, expected_files):
        if (
            observed.get("path") != expected["path"]
            or observed.get("sha256") != expected["sha256"]
        ):
            raise ValueError(f"{task}: run input path/hash differs from input manifest")
    result_path = Path(manifest["result_json"])
    if not result_path.is_file() or sha256(result_path) != manifest.get(
        "result_sha256"
    ):
        raise ValueError(f"{task}: result is missing or checksum mismatch")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"GPN-MSA"}:
        raise ValueError(f"{task}: expected one GPN-MSA result")
    model_results = payload["GPN-MSA"]
    if not isinstance(model_results, dict) or set(model_results) != {task}:
        raise ValueError(f"{task}: result task key does not match")
    projections = model_results[task]
    metrics = (
        projections.get("no_projection") if isinstance(projections, dict) else None
    )
    mcc = metrics.get("mcc") if isinstance(metrics, dict) else None
    if not isinstance(mcc, list) or len(mcc) < 3:
        raise ValueError(f"{task}: result JSON has no MCC mean/CI")
    values = [float(x) for x in mcc[:3]]
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"{task}: non-finite MCC result")
    return values, manifest


def rank_tasks(rows: list[dict[str, str]]) -> None:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["dataset"]].append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: (-float(row["mcc_mean"]), row["model"]))
        last_score: float | None = None
        rank = 0
        for position, row in enumerate(task_rows, 1):
            score = float(row["mcc_mean"])
            if last_score is None or score != last_score:
                rank = position
                last_score = score
            row["task_rank"] = str(rank)
            row["task_count"] = str(len(task_rows))


def merge(run_root: Path, tables_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    table_path = tables_dir / TABLE_NAME
    fieldnames, rows = read_csv(table_path)
    task_order = list(dict.fromkeys(row["dataset"] for row in rows))
    if len(task_order) != EXPECTED_TASKS:
        raise ValueError(
            f"Expected {EXPECTED_TASKS} tasks in S17, found {len(task_order)}"
        )
    input_manifest = json.loads(
        (run_root / "input_manifest.json").read_text(encoding="utf-8")
    )
    expected_inputs = {
        task: {
            entry["split"]: entry
            for entry in input_manifest["files"]
            if entry["task"] == task
        }
        for task in task_order
    }
    if any(set(expected_inputs[task]) != {"train", "test"} for task in task_order):
        raise ValueError("input_manifest.json is missing a task train/test file")
    updates = []
    cache_audits = {}
    for task in task_order:
        matches = [r for r in rows if r["dataset"] == task and r["model"] == "GPN-MSA"]
        if len(matches) != 1:
            raise ValueError(f"{task}: expected exactly one existing GPN-MSA S17 row")
        old = [float(matches[0][key]) for key in ("mcc_mean", "mcc_lower", "mcc_upper")]
        values, manifest = read_task_result(run_root, task, expected_inputs[task])
        cache_audits[task] = validate_embedding_cache(manifest, task)
        matches[0]["mcc_mean"], matches[0]["mcc_lower"], matches[0]["mcc_upper"] = [
            format(x, ".15g") for x in values
        ]
        updates.append(
            {
                "task": task,
                "old_mcc_mean": old[0],
                "new_mcc_mean": values[0],
                "delta": values[0] - old[0],
                "new_mcc_ci": values,
                "result_sha256": manifest["result_sha256"],
            }
        )
    rank_tasks(rows)
    order = {task: i for i, task in enumerate(task_order)}
    rows.sort(
        key=lambda row: (order[row["dataset"]], int(row["task_rank"]), row["model"])
    )
    summary = {
        "status": "PASS",
        "table": str(table_path),
        "old_table_sha256": sha256(table_path),
        "updates": updates,
        "embedding_cache_audits": cache_audits,
    }
    if not dry_run:
        _write_csv(
            table_path,
            fieldnames,
            rows,
            atomic=True,
            atomic_style="suffix",
        )
        summary["new_table_sha256"] = sha256(table_path)
        summary["written"] = True
    else:
        summary["written"] = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "tables",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-json", type=Path, default=None)
    args = parser.parse_args()
    summary = merge(args.run_root, args.tables_dir, dry_run=args.dry_run)
    if args.audit_json and not args.dry_run:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
