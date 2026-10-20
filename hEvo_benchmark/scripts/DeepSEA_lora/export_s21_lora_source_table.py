#!/usr/bin/env python3
"""Export the active S21 table from completed DeepSEA LoRA summaries.

The exporter only reformats existing JSON/CSV results. It never trains or
re-evaluates a model and never modifies the <source-project-root> source tree.
Rows remain run-level so repeated runs are not silently averaged or replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "tables"
    / "S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv"
)
TASK_ORDER = {"DHS": 0, "HM": 1, "TF": 2}
MODEL_DISPLAY = {
    "hEvo-base": "hEvo-base",
    "hEvo-large": "hEvo-large",
    "NT_2.5b_MS": "NT-2.5B-MS",
    "NT_2.5b_1kG": "NT-2.5B-1kG",
    "NT_500m_1kG": "NT-500M-1kG",
}
RECOVERED_RUN_ID = "S21_run_22"
BASE_RUN_ID_START = 2  # S21_run_01 was retired; preserve surviving public IDs.


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: %s" % path)
    return value


def manifest_for(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run_manifest.json"
    return read_json(path) if path.is_file() else {}


def load_rows(
    source_root: Path, summary_path: Path | None = None, expected_runs: int = 20
) -> List[Dict[str, Any]]:
    summary_path = summary_path or (
        source_root / "results" / "DeepSEA_lora" / "category_summary.csv"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        summary = list(csv.DictReader(handle))
    required = {
        "model_name",
        "category",
        "run_dir",
        "test_samples",
        "num_labels",
        "macro_auroc",
        "macro_auprc",
        "micro_auroc",
        "micro_auprc",
        "bce_loss",
        "valid_label_count",
    }
    missing = required - set(summary[0] if summary else [])
    if missing:
        raise ValueError("Missing category summary columns: %s" % sorted(missing))
    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_run[row["run_dir"]].append(row)
    if not by_run:
        raise ValueError("Empty DeepSEA LoRA category summary")
    if len(by_run) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} completed run directories, found {len(by_run)}"
        )
    result: List[Dict[str, Any]] = []
    for run_offset, (run_dir_string, run_rows) in enumerate(sorted(by_run.items())):
        public_run_number = BASE_RUN_ID_START + run_offset
        run_dir = Path(run_dir_string).resolve()
        if not run_dir.is_relative_to(source_root.resolve()):
            raise ValueError(f"Run directory is outside --source-root: {run_dir}")
        if len(run_rows) != len(TASK_ORDER) or {
            row["category"] for row in run_rows
        } != set(TASK_ORDER):
            raise ValueError(f"Run must have exactly DHS, HM and TF once: {run_dir}")
        metrics_path = run_dir / "category_metrics.json"
        metrics = read_json(metrics_path)
        if (
            metrics.get("status") != "completed"
            or Path(str(metrics.get("run_dir", ""))).resolve() != run_dir
        ):
            raise ValueError(
                f"Incomplete or mismatched category metrics: {metrics_path}"
            )
        categories = metrics.get("categories")
        if not isinstance(categories, dict) or set(categories) != set(TASK_ORDER):
            raise ValueError(f"Invalid category roster: {metrics_path}")
        metric_names = (
            "macro_auroc",
            "macro_auprc",
            "micro_auroc",
            "micro_auprc",
            "bce_loss",
        )
        for row in run_rows:
            if row["model_name"] != metrics.get("model_name"):
                raise ValueError(
                    f"{run_dir.name}: model_name differs from category_metrics.json"
                )
            category = categories[row["category"]]
            for name in metric_names:
                if not math.isclose(
                    float(row[name]), float(category[name]), rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"{run_dir.name}/{row['category']}: {name} differs from category_metrics.json"
                    )
            if int(row["test_samples"]) != int(metrics["test_samples"]):
                raise ValueError(
                    f"{run_dir.name}: test_samples differs from category_metrics.json"
                )
            if int(row["num_labels"]) != int(
                metrics["category_definition"]["categories"][row["category"]]["count"]
            ):
                raise ValueError(
                    f"{run_dir.name}/{row['category']}: num_labels differs from category_metrics.json"
                )
            if int(row["valid_label_count"]) != int(category["valid_label_count"]):
                raise ValueError(
                    f"{run_dir.name}/{row['category']}: valid_label_count differs from category_metrics.json"
                )
        manifest = manifest_for(run_dir)
        model_info = manifest.get("model", {})
        if not isinstance(model_info, dict):
            model_info = {}
        values = [float(row["macro_auroc"]) for row in run_rows]
        mean_auroc = sum(values) / len(values)
        for row in sorted(
            run_rows, key=lambda item: TASK_ORDER.get(item["category"], 99)
        ):
            model_name = str(row["model_name"])
            model_name = MODEL_DISPLAY.get(model_name, model_name)
            result.append(
                {
                    "benchmark": "DeepSEA LoRA evaluation",
                    "model": model_name,
                    "task": row["category"],
                    "metric": "AUROC",
                    "auroc": row["macro_auroc"],
                    "Mean": "%.12g" % mean_auroc,
                    "run_id": f"S21_run_{public_run_number:02d}",
                    "test_samples": row["test_samples"],
                    "num_labels": row["num_labels"],
                    "macro_auprc": row["macro_auprc"],
                    "micro_auroc": row["micro_auroc"],
                    "micro_auprc": row["micro_auprc"],
                    "bce_loss": row["bce_loss"],
                    "valid_label_count": row["valid_label_count"],
                    "max_length": manifest.get("max_length"),
                    "pooling": model_info.get("pooling"),
                    "seed": manifest.get("seed"),
                }
            )
    if len(result) != expected_runs * len(TASK_ORDER):
        raise ValueError(
            f"Expected {expected_runs * len(TASK_ORDER)} run-level rows, got {len(result)}"
        )
    if len({row["run_id"] for row in result}) != expected_runs:
        raise ValueError(f"Expected {expected_runs} completed run IDs")
    if len({row["model"] for row in result}) != expected_runs:
        raise ValueError(
            f"Expected one unique model per original run ({expected_runs})"
        )
    return result


def recovered_open_large_rows(
    recovery_dir: Path, run_manifest: Path
) -> List[Dict[str, Any]]:
    """Append the completed hEvo-open-large test-shard recovery as one run."""
    manifest = read_json(run_manifest)
    model = manifest.get("model", {})
    if (
        model.get("name") != "hEvo-open-large"
        or manifest.get("stage") != "final"
        or manifest.get("dataset", {}).get("test_labels_shape") != [455024, 919]
    ):
        raise ValueError(
            "hEvo-open-large manifest does not match the formal DeepSEA test protocol"
        )
    results = {task: read_json(recovery_dir / f"{task}.json") for task in TASK_ORDER}
    if not all(
        value.get("status") == "completed"
        and value.get("model_name") == "hEvo-open-large"
        and value.get("category") == task
        and value.get("source") == "complete_32_rank_test_prediction_shards"
        and value.get("source_job_id") == "6718312"
        and value.get("test_samples") == 455024
        for task, value in results.items()
    ):
        raise ValueError(
            "Recovered DeepSEA category results are incomplete or mismatched"
        )
    expected_labels = {"DHS": 125, "HM": 104, "TF": 690}
    for task, result in results.items():
        if (
            result["num_labels"] != expected_labels[task]
            or not 0 < result["valid_label_count"] <= expected_labels[task]
        ):
            raise ValueError(f"Invalid recovered label coverage for {task}")
        for key in (
            "macro_auroc",
            "macro_auprc",
            "micro_auroc",
            "micro_auprc",
            "bce_loss",
        ):
            value = float(result[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid recovered {key} for {task}")
    mean_auroc = sum(float(value["macro_auroc"]) for value in results.values()) / len(
        TASK_ORDER
    )
    rows = []
    for task in TASK_ORDER:
        result = results[task]
        rows.append(
            {
                "benchmark": "DeepSEA LoRA evaluation",
                "model": "hEvo-open-large",
                "task": task,
                "metric": "AUROC",
                "auroc": result["macro_auroc"],
                "Mean": "%.12g" % mean_auroc,
                "run_id": RECOVERED_RUN_ID,
                "test_samples": result["test_samples"],
                "num_labels": result["num_labels"],
                "macro_auprc": result["macro_auprc"],
                "micro_auroc": result["micro_auroc"],
                "micro_auprc": result["micro_auprc"],
                "bce_loss": result["bce_loss"],
                "valid_label_count": result["valid_label_count"],
                "max_length": manifest["max_length"],
                "pooling": model["pooling"],
                "seed": manifest["seed"],
            }
        )
    return rows


def frozen_base_rows(path: Path) -> List[Dict[str, Any]]:
    """Read the checked 20-run base roster when the external summary is stale."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["run_id"] != RECOVERED_RUN_ID
        ]
    runs = {row["run_id"] for row in rows}
    models = {row["model"] for row in rows}
    expected_runs = {f"S21_run_{number:02d}" for number in range(BASE_RUN_ID_START, 22)}
    if len(rows) != 60 or runs != expected_runs or len(models) != 20:
        raise ValueError(
            "Frozen S21 base roster must contain run_02 through run_21: 60 rows, 20 runs, and 20 models"
        )
    if any(
        {row["task"] for row in rows if row["run_id"] == run} != set(TASK_ORDER)
        for run in runs
    ):
        raise ValueError("Frozen S21 base roster has an incomplete DHS/HM/TF run")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="External source project root containing results/DeepSEA_lora",
    )
    parser.add_argument(
        "--category-summary",
        type=Path,
        default=None,
        help="Explicit authoritative category_summary.csv; overrides --source-root result summary",
    )
    parser.add_argument(
        "--base-table",
        type=Path,
        help="checked S21 table containing the frozen 20-run base roster",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=20,
        help="Required original run count; current S21 base roster is 20",
    )
    parser.add_argument(
        "--open-large-recovery-dir",
        type=Path,
        help="Directory containing recovered DHS/HM/TF JSON results",
    )
    parser.add_argument(
        "--open-large-run-manifest",
        type=Path,
        help="Formal hEvo-open-large DeepSEA run_manifest.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary_path = (
        args.category_summary.resolve() if args.category_summary is not None else None
    )
    if args.expected_runs < 1:
        parser.error("--expected-runs must be positive")
    if args.output.resolve() == DEFAULT_OUTPUT.resolve() and (
        args.expected_runs != 20
        or args.open_large_recovery_dir is None
        or args.open_large_run_manifest is None
    ):
        parser.error(
            "The published S21 requires 20 original runs plus the recovered hEvo-open-large run"
        )
    if (args.open_large_recovery_dir is None) != (args.open_large_run_manifest is None):
        parser.error(
            "Both hEvo-open-large recovery and run-manifest paths are required together"
        )
    rows = (
        frozen_base_rows(args.base_table.resolve())
        if args.base_table is not None
        else load_rows(args.source_root.resolve(), summary_path, args.expected_runs)
    )
    if args.open_large_recovery_dir is not None:
        rows.extend(
            recovered_open_large_rows(
                args.open_large_recovery_dir.resolve(),
                args.open_large_run_manifest.resolve(),
            )
        )
    expected_total_runs = args.expected_runs + (
        1 if args.open_large_recovery_dir is not None else 0
    )
    run_ids = {row["run_id"] for row in rows}
    models = {row["model"] for row in rows}
    if len(rows) != expected_total_runs * len(TASK_ORDER):
        raise ValueError(
            f"Expected {expected_total_runs * len(TASK_ORDER)} rows, found {len(rows)}"
        )
    if len(run_ids) != expected_total_runs or len(models) != expected_total_runs:
        raise ValueError(f"Expected {expected_total_runs} unique run IDs and models")
    if any(
        {row["task"] for row in rows if row["run_id"] == run_id} != set(TASK_ORDER)
        for run_id in run_ids
    ):
        raise ValueError("Every S21 run must contain exactly DHS, HM and TF")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("S21 output rows do not match the active table schema")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "base_roster_source": str(args.base_table.resolve())
                if args.base_table is not None
                else str(
                    summary_path
                    or args.source_root.resolve()
                    / "results"
                    / "DeepSEA_lora"
                    / "category_summary.csv"
                ),
                "rows": len(rows),
                "run_count": len({row["run_id"] for row in rows}),
                "model_count": len({row["model"] for row in rows}),
                "tasks": sorted(
                    {row["task"] for row in rows}, key=lambda x: TASK_ORDER.get(x, 99)
                ),
                "pos_weight_enabled": False,
                "loss_name": "binary_cross_entropy_with_logits",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
