#!/usr/bin/env python3
"""Replace the displayed hEvo-large rows with a checkpoint-backed probe run.

The source tables intentionally keep the display label ``hEvo-large``.  The
checkpoint identity, result manifests, hashes, and task mappings are written
to a separate provenance JSON so that the display table does not silently
collapse the new checkpoint identity into the historical model name.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .common import read_csv as _read_csv, sha256, write_csv as _write_csv
except ImportError:  # direct script execution
    from common import read_csv as _read_csv, sha256, write_csv as _write_csv


MODEL_NAME = "hEvo-large"
DISPLAY_MODEL = "hEvo-large"
MODEL_RELATIVE_PATH = "hEvo_benchmark/models/hEvo-large"

TABLE_NAMES = {
    "S17": "S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv",
    "S18": "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv",
    "S19": "S19_binary_classification_across_five_ure_enhancer_control_sets.csv",
    "S20": "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv",
    "S36": "S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv",
    "S37": "S37_classification_of_human_accelerated_regions_across_negative_controls.csv",
}

URE_BINARY_MAP = {
    "URE_dataset_2": 1,
    "URE_dataset_3": 2,
    "URE_dataset_4": 3,
    "URE_dataset_5": 4,
    "URE_dataset_7": 5,
}

URE_MULTI_MAP = {
    "URE_dataset_4": 1,
    "URE_dataset_6": 2,
    "URE_dataset_7": 3,
    "URE_dataset_8": 4,
    "URE_dataset_9": 5,
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    return _read_csv(path, require_header=True)


def write_csv_atomic(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    _write_csv(
        path,
        fieldnames,
        rows,
        atomic=True,
        atomic_style="named",
        create_parent=True,
        extrasaction="raise",
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def fmt(value: float, precision: int = 16) -> str:
    return format(float(value), f".{precision}g")


def fmt_rank(value: int, old: str) -> str:
    return f"{value}.0" if old.endswith(".0") else str(value)


def relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return f"<external-root>/{path.name}"


def table_path(tables_dir: Path, key: str) -> Path:
    return tables_dir / TABLE_NAMES[key]


def model_result_path(
    results_root: Path, group: str, task_dir: str, task_token: str | None = None
) -> Path:
    if task_token is None:
        task_token = task_dir
    filename = f"results_{MODEL_NAME}_{task_token}.json"
    return results_root / group / task_dir / MODEL_NAME / filename


def load_result(
    path: Path,
    repo_root: Path,
    expected_dataset: str | None = None,
) -> dict[str, Any]:
    manifest_path = path.parent / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        result_json = json.load(handle)
    actual_result_hash = sha256(path)

    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("run_status") != "COMPLETED":
            raise ValueError(f"result is not completed: {manifest_path}")
        if manifest.get("model_name") != MODEL_NAME:
            raise ValueError(f"wrong model in manifest: {manifest_path}")
        if (
            expected_dataset is not None
            and manifest.get("dataset_name") != expected_dataset
        ):
            raise ValueError(
                f"dataset mismatch for {path}: {manifest.get('dataset_name')} != {expected_dataset}"
            )
        recorded_result_hash = manifest.get("result_sha256")
        if recorded_result_hash and recorded_result_hash != actual_result_hash:
            raise ValueError(f"result hash mismatch: {path}")
        manifest_sha256 = sha256(manifest_path)
        repo_manifest_path: str | None = relative_path(manifest_path, repo_root)
        manifest_status = "RUN_MANIFEST_COMPLETED"
    else:
        json_model = result_json.get("model_name")
        json_dataset = result_json.get("dataset_name")
        if json_model != MODEL_NAME:
            raise ValueError(f"wrong model in result JSON: {path}")
        if expected_dataset is not None and json_dataset != expected_dataset:
            raise ValueError(
                f"dataset mismatch for {path}: {json_dataset} != {expected_dataset}"
            )
        manifest = {
            "run_status": "RESULT_JSON_ONLY",
            "model_name": json_model,
            "dataset_name": json_dataset,
            "protocol": {
                "pooling": result_json.get("pooling"),
                "max_length": result_json.get("max_length"),
                "evaluator_sha256": result_json.get("evaluator_sha256"),
            },
            "split": {
                "train_rows": result_json.get("train_samples"),
                "validation_rows": result_json.get("test_samples"),
            },
        }
        manifest_sha256 = None
        repo_manifest_path = None
        manifest_status = "RESULT_JSON_ONLY"

    if MODEL_NAME in result_json:
        model_payload = result_json[MODEL_NAME]
        if not isinstance(model_payload, dict) or len(model_payload) != 1:
            raise ValueError(f"unexpected nested result schema: {path}")
        dataset_payload = next(iter(model_payload.values()))
        metrics = dataset_payload.get("no_projection")
    else:
        metrics = result_json.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"missing metrics in {path}")
    metrics = dict(metrics)
    bootstrap = result_json.get("bootstrap", {})
    if isinstance(bootstrap, dict):
        for metric_name in ("accuracy", "mcc", "auc"):
            point = metrics.get(metric_name)
            interval = bootstrap.get(metric_name)
            if (
                isinstance(point, (int, float))
                and isinstance(interval, list)
                and len(interval) == 3
            ):
                metrics[metric_name] = [
                    float(point),
                    float(interval[1]),
                    float(interval[2]),
                ]

    return {
        "path": path,
        "manifest_path": manifest_path,
        "metrics": metrics,
        "manifest": manifest,
        "result_sha256": actual_result_hash,
        "manifest_sha256": manifest_sha256,
        "manifest_status": manifest_status,
        "repo_result_path": relative_path(path, repo_root),
        "repo_manifest_path": repo_manifest_path,
        "result_json_metadata": {
            key: result_json.get(key)
            for key in (
                "model_sha256",
                "dataset_sha256",
                "evaluator_sha256",
                "pooling",
                "max_length",
                "shard_count",
            )
            if key in result_json
        },
    }


def require_triplet(metrics: dict[str, Any], name: str) -> list[float]:
    values = metrics.get(name)
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"expected mean/low/high triplet for {name}")
    return [float(value) for value in values]


def require_f1(metrics: dict[str, Any]) -> list[float]:
    values = metrics.get("per_class_f1")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError("expected five per-class F1 values")
    return [float(value) for value in values]


def replace_metric_triplet(
    row: dict[str, str], prefix: str, values: list[float], precision: int = 16
) -> None:
    row[f"{prefix}_mean"] = fmt(values[0], precision)
    row[f"{prefix}_low"] = fmt(values[1], precision)
    row[f"{prefix}_high"] = fmt(values[2], precision)


def replace_task_mcc(
    row: dict[str, str], values: list[float], precision: int = 16
) -> None:
    row["mcc_mean"] = fmt(values[0], precision)
    row["mcc_lower"] = fmt(values[1], precision)
    row["mcc_upper"] = fmt(values[2], precision)


def rank_by_task(rows: list[dict[str, str]]) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda item: -float(item["mcc_mean"]))
        for rank, row in enumerate(ordered, start=1):
            row["task_rank"] = fmt_rank(rank, row["task_rank"])
            row["task_count"] = fmt_rank(len(group), row["task_count"])


def rank_by_model(rows: list[dict[str, str]], metric: str) -> None:
    values: dict[str, list[float]] = defaultdict(list)
    first_index: dict[str, int] = {}
    for index, row in enumerate(rows):
        model = row["model"]
        values[model].append(float(row[metric]))
        first_index.setdefault(model, index)
    means = {
        model: sum(model_values) / len(model_values)
        for model, model_values in values.items()
    }
    ordered_models = sorted(
        means,
        key=lambda model: (-means[model], first_index[model]),
    )
    model_order = {model: rank for rank, model in enumerate(ordered_models, start=1)}
    for row in rows:
        if "model_order" in row:
            row["model_order"] = fmt_rank(model_order[row["model"]], row["model_order"])
        row["mean_mcc_for_sort"] = fmt(means[row["model"]])


def result_artifact_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    manifest = artifact["manifest"]
    protocol = manifest.get("protocol", {})
    split = manifest.get("split", {})
    return {
        "dataset_name": manifest.get("dataset_name"),
        "result_path": artifact["repo_result_path"],
        "manifest_path": artifact["repo_manifest_path"],
        "result_sha256": artifact["result_sha256"],
        "manifest_sha256": artifact["manifest_sha256"],
        "manifest_status": artifact["manifest_status"],
        "result_json_metadata": artifact["result_json_metadata"],
        "run_status": manifest.get("run_status"),
        "started_at_utc": manifest.get("started_at_utc"),
        "finished_at_utc": manifest.get("finished_at_utc"),
        "protocol": protocol,
        "split": {
            "mode": split.get("mode"),
            "train_rows": split.get("train_rows"),
            "validation_rows": split.get("validation_rows"),
        },
    }


def replace_s22(
    path: Path, results_root: Path, repo_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields, rows = read_csv(path)
    targets = [row for row in rows if row["model"] == DISPLAY_MODEL]
    if len(targets) != 18:
        raise ValueError(f"S17 expected 18 hEvo-large rows, found {len(targets)}")
    artifacts = []
    for row in targets:
        dataset = row["dataset"]
        result_path = model_result_path(
            results_root, "Probe_NT_Pareto", dataset, "NT_benchmark_v2"
        )
        artifact = load_result(result_path, repo_root, "NT_benchmark_v2")
        replace_task_mcc(row, require_triplet(artifact["metrics"], "mcc"), 6)
        artifacts.append(
            {"table": "S17", "task": dataset, **result_artifact_summary(artifact)}
        )
    rank_by_task(rows)
    return {
        "path": path,
        "fields": fields,
        "rows": rows,
        "before_rows": len(rows),
    }, artifacts


def replace_s23(
    path: Path, results_root: Path, repo_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields, rows = read_csv(path)
    targets = [row for row in rows if row["model"] == DISPLAY_MODEL]
    if len(targets) != 7:
        raise ValueError(f"S18 expected 7 hEvo-large rows, found {len(targets)}")
    artifacts = []
    for row in targets:
        dataset = row["dataset"]
        result_path = model_result_path(
            results_root,
            "Probe_GenomicBenchmark",
            dataset,
            f"GenomicBenchmark_{dataset}",
        )
        artifact = load_result(
            result_path,
            repo_root,
            f"GenomicBenchmark_{dataset}",
        )
        replace_task_mcc(row, require_triplet(artifact["metrics"], "mcc"))
        artifacts.append(
            {"table": "S18", "task": dataset, **result_artifact_summary(artifact)}
        )
    rank_by_task(rows)
    return {
        "path": path,
        "fields": fields,
        "rows": rows,
        "before_rows": len(rows),
    }, artifacts


def replace_s24(
    path: Path, results_root: Path, repo_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields, rows = read_csv(path)
    targets = [row for row in rows if row["model"] == DISPLAY_MODEL]
    if len(targets) != 5:
        raise ValueError(f"S19 expected 5 hEvo-large rows, found {len(targets)}")
    artifacts = []
    for row in targets:
        benchmark_raw = row["benchmark_raw"]
        dataset_index = URE_BINARY_MAP[benchmark_raw]
        task_dir = f"dataset_{dataset_index}"
        expected_dataset = f"URE_binary_classification_{task_dir}"
        result_path = model_result_path(
            results_root,
            "Probe_URE_binary_classification",
            task_dir,
            f"URE_binary_classification_{task_dir}",
        )
        artifact = load_result(result_path, repo_root, expected_dataset)
        metrics = artifact["metrics"]
        replace_metric_triplet(row, "acc", require_triplet(metrics, "accuracy"))
        replace_metric_triplet(row, "mcc", require_triplet(metrics, "mcc"))
        replace_metric_triplet(row, "auc", require_triplet(metrics, "auc"))
        row["run_dir"] = (
            f"<repo-root>/hEvo_benchmark/results/Probe_URE_binary_classification/"
            f"{task_dir}/{MODEL_NAME}"
        )
        artifacts.append(
            {
                "table": "S19",
                "task": benchmark_raw,
                "dataset_index": dataset_index,
                **result_artifact_summary(artifact),
            }
        )
    rank_by_model(rows, "mcc_mean")
    return {
        "path": path,
        "fields": fields,
        "rows": rows,
        "before_rows": len(rows),
    }, artifacts


def replace_s20(
    path: Path, results_root: Path, repo_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields, rows = read_csv(path)
    targets = [row for row in rows if row["model"] == DISPLAY_MODEL]
    if len(targets) != 25:
        raise ValueError(f"S20 expected 25 hEvo-large rows, found {len(targets)}")
    artifacts_by_dataset: dict[str, dict[str, Any]] = {}
    artifacts = []
    for row in targets:
        dataset = row["dataset"]
        dataset_index = URE_MULTI_MAP[dataset]
        task_dir = f"dataset_{dataset_index}"
        if dataset not in artifacts_by_dataset:
            result_path = model_result_path(
                results_root,
                "Probe_URE_multi_classification",
                task_dir,
                f"URE_multi_classification_{task_dir}",
            )
            artifacts_by_dataset[dataset] = load_result(
                result_path,
                repo_root,
                f"URE_multi_classification_{task_dir}",
            )
            artifacts.append(
                {
                    "table": "S20",
                    "task": dataset,
                    "dataset_index": dataset_index,
                    **result_artifact_summary(artifacts_by_dataset[dataset]),
                }
            )
        values = require_f1(artifacts_by_dataset[dataset]["metrics"])
        class_id = int(row["class_id"])
        row["per_class_f1"] = fmt(values[class_id])
    return {
        "path": path,
        "fields": fields,
        "rows": rows,
        "before_rows": len(rows),
    }, artifacts


def replace_evolution_table(
    path: Path,
    results_root: Path,
    repo_root: Path,
    table_key: str,
    prefix: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields, rows = read_csv(path)
    targets = [row for row in rows if row["model"] == DISPLAY_MODEL]
    if len(targets) != 5:
        raise ValueError(
            f"{table_key} expected 5 hEvo-large rows, found {len(targets)}"
        )
    artifacts = []
    for replicate, row in enumerate(targets, start=1):
        historical_row_label = row.get("benchmark_raw")
        task_dir = f"{prefix}_rep{replicate}"
        dataset_name = f"Evolution_{task_dir}"
        result_path = model_result_path(
            results_root,
            "Probe_Evolution",
            task_dir,
            f"Evolution_{task_dir}",
        )
        artifact = load_result(result_path, repo_root, dataset_name)
        metrics = artifact["metrics"]
        replace_metric_triplet(row, "acc", require_triplet(metrics, "accuracy"))
        replace_metric_triplet(row, "mcc", require_triplet(metrics, "mcc"))
        replace_metric_triplet(row, "auc", require_triplet(metrics, "auc"))
        # Keep the task directory in the provenance artifact while using the
        # same canonical replicate label as the historical model rows.
        canonical_task = f"{prefix}_combined_rep{replicate}"
        row["run"] = canonical_task
        if "run_dir" in row:
            row["run_dir"] = (
                f"<repo-root>/hEvo_benchmark/results/Probe_Evolution/"
                f"{task_dir}/{MODEL_NAME}"
            )
        row["benchmark_raw"] = canonical_task
        artifacts.append(
            {
                "table": table_key,
                "task": task_dir,
                "historical_row_label_replaced": historical_row_label,
                **result_artifact_summary(artifact),
            }
        )
    rank_by_model(rows, "mcc_mean")
    return {
        "path": path,
        "fields": fields,
        "rows": rows,
        "before_rows": len(rows),
    }, artifacts


def validate_table_shape(table: dict[str, Any], expected_data_rows: int) -> None:
    rows = table["rows"]
    if len(rows) != expected_data_rows:
        raise ValueError(
            f"{table['path']} changed row count: {len(rows)} != {expected_data_rows}"
        )
    if any(row["model"] != DISPLAY_MODEL for row in rows if row["model"] == MODEL_NAME):
        raise ValueError(f"raw model name leaked into display table: {table['path']}")


def build_provenance(
    repo_root: Path,
    model_path: Path,
    table_results: dict[str, dict[str, Any]],
    artifacts: list[dict[str, Any]],
    before_hashes: dict[str, str],
    provenance_path: Path,
) -> dict[str, Any]:
    model_provenance_path = model_path / "provenance.json"
    model_config_path = model_path / "config.json"
    with model_provenance_path.open("r", encoding="utf-8") as handle:
        model_provenance = json.load(handle)
    return {
        "schema": "hevo_large_sequence_classification_s_table_provenance_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "display_model": DISPLAY_MODEL,
        "source_model": MODEL_NAME,
        "model": {
            "path": "<repo-root>/" + MODEL_RELATIVE_PATH,
            "config_sha256": sha256(model_config_path),
            "provenance_sha256": sha256(model_provenance_path),
            "weight_sha256": model_provenance.get("output_weight_sha256"),
            "stage": model_provenance.get("stage"),
            "backbone_frozen": model_provenance.get("backbone_frozen"),
        },
        "table_contract": {
            "S17": {"file": TABLE_NAMES["S17"], "display_rows_replaced": 18},
            "S18": {"file": TABLE_NAMES["S18"], "display_rows_replaced": 7},
            "S19": {"file": TABLE_NAMES["S19"], "display_rows_replaced": 5},
            "S20": {"file": TABLE_NAMES["S20"], "display_rows_replaced": 25},
            "S36": {"file": TABLE_NAMES["S36"], "display_rows_replaced": 5},
            "S37": {"file": TABLE_NAMES["S37"], "display_rows_replaced": 5},
        },
        "ure_mapping": {
            "S19": URE_BINARY_MAP,
            "S20": URE_MULTI_MAP,
        },
        "evolution_mapping": {
            "S36": [
                "HAQER_combined_rep1",
                "HAQER_combined_rep2",
                "HAQER_combined_rep3",
                "HAQER_combined_rep4",
                "HAQER_combined_rep5",
            ],
            "S37": [
                "HAR_combined_rep1",
                "HAR_combined_rep2",
                "HAR_combined_rep3",
                "HAR_combined_rep4",
                "HAR_combined_rep5",
            ],
            "historical_negset_mapping": {
                "HAQER": {
                    "negset2": "rep1",
                    "negset7": "rep2",
                    "negset10": "rep3",
                    "negset13": "rep4",
                    "negset19": "rep5",
                },
                "HAR": {
                    "negset2": "rep1",
                    "negset5": "rep2",
                    "negset6": "rep3",
                    "negset8": "rep4",
                    "negset10": "rep5",
                },
            },
        },
        "source_results": artifacts,
        "table_hashes": {
            key: {
                "path": TABLE_NAMES[key],
                "sha256_before": before_hashes[key],
                "sha256_after": sha256(table_results[key]["path"]),
                "rows_before": table_results[key]["before_rows"],
                "rows_after": len(table_results[key]["rows"]),
            }
            for key in TABLE_NAMES
        },
        "validation": {
            "no_imputation": True,
            "display_model_only": True,
            "result_artifacts_completed": len(artifacts),
            "expected_result_artifacts": 45,
            "provenance_path": "<repo-root>/hEvo_benchmark/tables/"
            + provenance_path.name,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--provenance", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all source results and table contracts without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    tables_dir = (args.tables_dir or repo_root / "hEvo_benchmark/tables").resolve()
    results_root = (args.results_root or repo_root / "hEvo_benchmark/results").resolve()
    model_path = (
        args.model_path or repo_root / "hEvo_benchmark/models/hEvo-large"
    ).resolve()
    provenance_path = (
        args.provenance
        or tables_dir / "hEvo_large_sequence_classification_provenance.json"
    ).resolve()

    table_paths = {key: table_path(tables_dir, key) for key in TABLE_NAMES}
    before_hashes = {key: sha256(path) for key, path in table_paths.items()}
    table_results: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []

    table, found = replace_s22(table_paths["S17"], results_root, repo_root)
    table_results["S17"] = table
    artifacts.extend(found)
    table, found = replace_s23(table_paths["S18"], results_root, repo_root)
    table_results["S18"] = table
    artifacts.extend(found)
    table, found = replace_s24(table_paths["S19"], results_root, repo_root)
    table_results["S19"] = table
    artifacts.extend(found)
    table, found = replace_s20(table_paths["S20"], results_root, repo_root)
    table_results["S20"] = table
    artifacts.extend(found)
    table, found = replace_evolution_table(
        table_paths["S36"], results_root, repo_root, "S36", "HAQER"
    )
    table_results["S36"] = table
    artifacts.extend(found)
    table, found = replace_evolution_table(
        table_paths["S37"], results_root, repo_root, "S37", "HAR"
    )
    table_results["S37"] = table
    artifacts.extend(found)

    expected_rows = {
        "S17": 540,
        "S18": 210,
        "S19": 150,
        "S20": 750,
        "S36": 170,
        "S37": 170,
    }
    for key, expected in expected_rows.items():
        validate_table_shape(table_results[key], expected)
    if len(artifacts) != 45:
        raise ValueError(
            f"expected 45 completed result artifacts, found {len(artifacts)}"
        )

    if not args.check_only:
        for table in table_results.values():
            write_csv_atomic(table["path"], table["fields"], table["rows"])
    provenance = build_provenance(
        repo_root,
        model_path,
        table_results,
        artifacts,
        before_hashes,
        provenance_path,
    )
    if not args.check_only:
        write_json_atomic(provenance_path, provenance)

    print(
        json.dumps(
            {
                "mode": "check-only" if args.check_only else "write",
                "tables": {
                    key: len(value["rows"]) for key, value in table_results.items()
                },
                "result_artifacts": len(artifacts),
                "provenance": relative_path(provenance_path, repo_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
