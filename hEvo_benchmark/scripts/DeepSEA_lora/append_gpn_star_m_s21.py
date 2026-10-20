#!/usr/bin/env python3
"""Append the completed GPN-Star-M run to the public S21 source table.

This utility only reformats the run's category summary. It does not train,
score, or copy the large runtime artifacts into the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


DNAMODELS_ROOT = Path(
    os.environ.get("DNAMODELS_BENCHMARK_ROOT", str(Path.home() / "DNAModelsBenchmark"))
)
DEFAULT_RUN = (
    DNAMODELS_ROOT
    / "runs/deepsea_gpn_star_m_v1/training/deepsea_1epoch_full_40gpu_evo2params_20261005_230146_134556"
)
DEFAULT_TABLE = (
    Path(__file__).resolve().parents[2]
    / "tables"
    / "S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv"
)
CATEGORIES = ("DHS", "HM", "TF")
MAPPING_FIELDS = (
    "train_rows",
    "train_mapped_rows",
    "train_mapping_coverage",
    "test_rows",
    "test_mapped_rows",
    "test_mapping_coverage",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_rows(run_root: Path) -> list[dict[str, str]]:
    summary = load_json(run_root / "summary" / "category_metrics_seed_summary.json")
    manifest = load_json(run_root / "seed_42" / "run_manifest.json")
    if (
        summary.get("status") != "completed"
        or summary.get("model_name") != "GPN-Star-M"
    ):
        raise ValueError("The category summary is not a completed GPN-Star-M run")
    if summary.get("run_root") != str(run_root):
        raise ValueError("The category summary points to a different run directory")
    if summary.get("seeds") != [42] or manifest.get("seed") != 42:
        raise ValueError("S21 update expects the single completed seed 42")
    if manifest.get("world_size") != 40 or manifest.get("max_length") != 1000:
        raise ValueError(
            "Run manifest does not match the 40-GPU, 1,000-bp GPN-Star-M protocol"
        )

    mapping = summary.get("coordinate_mapping", {})
    train_rows = int(mapping["train_rows"])
    train_mapped = int(mapping["train_mapped_rows"])
    test_rows = int(mapping["test_rows"])
    test_mapped = int(mapping["test_mapped_rows"])
    if train_rows != 4_400_000 or test_rows != 455_024:
        raise ValueError("Unexpected DeepSEA train/test row counts")
    if not (0 < train_mapped <= train_rows and 0 < test_mapped <= test_rows):
        raise ValueError("Invalid coordinate-mapping counts")

    by_seed = summary.get("by_seed", [])
    if {row.get("category") for row in by_seed} != set(CATEGORIES) or len(by_seed) != 3:
        raise ValueError("Expected one seed-42 summary row for DHS, HM, and TF")
    mean_auroc = sum(float(row["macro_auroc"]) for row in by_seed) / len(CATEGORIES)
    run_id = "S21_run_23"
    rows = []
    for row in sorted(by_seed, key=lambda item: CATEGORIES.index(item["category"])):
        if int(row["seed"]) != 42 or not 0 < int(row["test_samples"]) <= test_rows:
            raise ValueError(
                "Category summary has invalid seed or evaluated sample count"
            )
        rows.append(
            {
                "benchmark": "DeepSEA LoRA evaluation",
                "model": "GPN-Star-M",
                "task": row["category"],
                "metric": "AUROC",
                "auroc": format(float(row["macro_auroc"]), ".15g"),
                "Mean": format(mean_auroc, ".15g"),
                "run_id": run_id,
                "test_samples": str(int(row["test_samples"])),
                "num_labels": str(int(row["num_labels"])),
                "macro_auprc": format(float(row["macro_auprc"]), ".15g"),
                "micro_auroc": format(float(row["micro_auroc"]), ".15g"),
                "micro_auprc": format(float(row["micro_auprc"]), ".15g"),
                "bce_loss": format(float(row["bce_loss"]), ".15g"),
                "valid_label_count": str(int(row["valid_label_count"])),
                "max_length": "1000",
                "pooling": "mean-pooling",
                "seed": "42",
                "train_rows": str(train_rows),
                "train_mapped_rows": str(train_mapped),
                "train_mapping_coverage": format(train_mapped / train_rows, ".12g"),
                "test_rows": str(test_rows),
                "test_mapped_rows": str(test_mapped),
                "test_mapping_coverage": format(test_mapped / test_rows, ".12g"),
            }
        )
    return rows


def update_table(table: Path, rows: list[dict[str, str]]) -> None:
    with table.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    for field in MAPPING_FIELDS:
        if field not in fields:
            fields.append(field)
    new_run = rows[0]["run_id"]
    if any(
        row.get("run_id") == new_run and row.get("model") != "GPN-Star-M"
        for row in existing
    ):
        raise ValueError(f"{new_run} is already assigned to another model")
    if any(
        row.get("model") == "GPN-Star-M" and row.get("run_id") != new_run
        for row in existing
    ):
        raise ValueError(
            "S21 already contains another GPN-Star-M run; review before adding a second run"
        )
    existing = [row for row in existing if row.get("run_id") != new_run]
    normalized = []
    for row in existing:
        normalized.append({field: row.get(field, "") for field in fields})
    normalized.extend({field: row.get(field, "") for field in fields} for row in rows)
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    rows = build_rows(args.run_root.resolve())
    update_table(args.table.resolve(), rows)
    print(
        json.dumps(
            {
                "table": str(args.table.resolve()),
                "rows_added": len(rows),
                "run_id": rows[0]["run_id"],
                "mapping": {field: rows[0][field] for field in MAPPING_FIELDS},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
