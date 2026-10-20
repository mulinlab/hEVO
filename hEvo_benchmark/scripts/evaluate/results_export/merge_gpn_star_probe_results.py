#!/usr/bin/env python3
"""Merge completed GPN-Star embedding-probe outputs into the source tables.

The input run directory is deliberately supplied at runtime because it lives
outside the benchmark repository on the HPC filesystem. Re-running this script
replaces the current GPN-Star rows in the target tables with the available
completed outputs, so newly finished jobs can be merged without duplicates.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

try:
    from .common import read_csv as _read_csv, write_csv as _write_csv
except ImportError:  # direct script execution
    from common import read_csv as _read_csv, write_csv as _write_csv


MODELS = {"GPN-Star-M", "GPN-Star-P", "GPN-Star-V"}
EVOLUTION_NEGATIVE_SETS = {
    "HAQER": {2: 1, 7: 2, 10: 3, 13: 4, 19: 5},
    "HAR": {2: 1, 5: 2, 6: 3, 8: 4, 10: 5},
}


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    return _read_csv(path)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    _write_csv(path, fieldnames, rows)


def finite_number(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        return ""
    return format(number, ".15g")


def result_records(root: Path) -> List[Tuple[str, str, dict]]:
    records = []
    for result_path in sorted(root.glob("**/results_*.json")):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or len(data) != 1:
            continue
        model, task_map = next(iter(data.items()))
        if model not in MODELS or not isinstance(task_map, dict) or len(task_map) != 1:
            continue
        task, projections = next(iter(task_map.items()))
        metrics = (
            projections.get("no_projection") if isinstance(projections, dict) else None
        )
        if not isinstance(metrics, dict) or not isinstance(metrics.get("mcc"), list):
            continue
        if len(metrics["mcc"]) < 3 or not all(
            math.isfinite(float(v)) for v in metrics["mcc"][:3]
        ):
            continue
        manifest_path = result_path.parent / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("run_status") != "COMPLETED":
                continue
        records.append((model, task, metrics))
    return records


def merge_rows(path: Path, model_column: str, new_rows: List[Dict[str, str]]) -> int:
    fieldnames, old_rows = read_csv(path)
    rows = [row for row in old_rows if row.get(model_column) not in MODELS]
    rows.extend(new_rows)
    if len({tuple(row.get(k, "") for k in fieldnames) for row in rows}) != len(rows):
        raise ValueError(f"duplicate rows detected while merging {path}")
    write_csv(path, fieldnames, rows)
    return len(new_rows)


def re_rank_task_table(path: Path) -> None:
    fieldnames, rows = read_csv(path)
    runs = list(dict.fromkeys(row["run"] for row in rows))
    model_scores = defaultdict(list)
    for row in rows:
        model_scores[row["model"]].append(float(row["mcc_mean"]))
    model_means = {model: mean(values) for model, values in model_scores.items()}
    ordered_models = sorted(model_means, key=lambda model: (-model_means[model], model))
    model_order = {model: index + 1 for index, model in enumerate(ordered_models)}
    for row in rows:
        row["mean_mcc_for_sort"] = finite_number(model_means[row["model"]])
        if "model_order" in fieldnames:
            row["model_order"] = str(model_order[row["model"]])
    run_order = {run: index for index, run in enumerate(runs)}
    rows.sort(key=lambda row: (run_order[row["run"]], model_order[row["model"]]))
    write_csv(path, fieldnames, rows)


def probe_row(
    template: Dict[str, str], model: str, task: str, metrics: dict, run_dir: str
) -> Dict[str, str]:
    row = dict(template)
    row["model"] = model
    if "dataset" in row:
        row["dataset"] = task
    if "run" in row:
        row["run"] = task
    if "run_dir" in row:
        row["run_dir"] = run_dir
    for metric, columns in (
        ("accuracy", ("acc_mean", "acc_low", "acc_high")),
        ("mcc", ("mcc_mean", "mcc_low", "mcc_high")),
        ("auc", ("auc_mean", "auc_low", "auc_high")),
    ):
        values = metrics.get(metric)
        if isinstance(values, list) and len(values) >= 3:
            for column, value in zip(columns, values[:3]):
                row[column] = finite_number(value)
    return row


def assign_competition_ranks(task_rows: List[Dict[str, str]]) -> None:
    """Keep equal published MCC values at the same (minimum) rank."""

    last_score = None
    current_rank = 0
    for position, row in enumerate(task_rows, 1):
        score = float(row["mcc_mean"])
        if last_score is None or score != last_score:
            current_rank = position
            last_score = score
        row["task_rank"] = str(current_rank)
        row["task_count"] = str(len(task_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()

    tables = args.benchmark_root / "tables"
    counts = {}

    # NT v2: H2AFZ came from the completed full-task template validation;
    # other tasks use the all-task run directory, preferring it if duplicated.
    nt_roots = [
        args.run_root
        / "gpn_probe_template_smoke_20260926"
        / "results"
        / "Probe_NT_Pareto",
        args.run_root
        / "gpn_embedding_probe_nt_all_20260926"
        / "results"
        / "Probe_NT_Pareto",
    ]
    nt_records = {}
    for root in nt_roots:
        for model, task, metrics in result_records(root):
            nt_records[(model, task)] = (metrics, root)
    path = (
        tables
        / "S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv"
    )
    _, old_rows = read_csv(path)
    nt_rows = []
    for (model, task), (metrics, root) in sorted(nt_records.items()):
        template = next((row for row in old_rows if row["dataset"] == task), None)
        if template is None:
            continue
        row = dict(template)
        row["model"] = model
        for column, value in zip(
            ("mcc_mean", "mcc_lower", "mcc_upper"), metrics["mcc"][:3]
        ):
            row[column] = finite_number(value)
        nt_rows.append(row)
    counts["S17"] = merge_rows(path, "model", nt_rows)
    fieldnames, rows = read_csv(path)
    task_order = list(dict.fromkeys(row["dataset"] for row in rows))
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["dataset"]].append(row)
    for task, task_rows in by_task.items():
        task_rows.sort(key=lambda row: (-float(row["mcc_mean"]), row["model"]))
        assign_competition_ranks(task_rows)
    rows = [row for task in task_order for row in by_task[task]]
    write_csv(path, fieldnames, rows)

    # Genomic Benchmark: only coordinate-supported human tasks are merged.
    gb_root = (
        args.run_root
        / "gpn_star_supplementary_20260926"
        / "GenomicBenchmark_sourcecoords"
        / "gpn_star_probe_run"
    )
    path = (
        tables
        / "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv"
    )
    _, old_rows = read_csv(path)
    gb_rows = []
    for model, raw_task, metrics in result_records(gb_root / "results"):
        task = raw_task[len("GBsrc_") :] if raw_task.startswith("GBsrc_") else raw_task
        template = next((row for row in old_rows if row["dataset"] == task), None)
        if template is None:
            continue
        row = dict(template)
        row["model"] = model
        for column, value in zip(
            ("mcc_mean", "mcc_lower", "mcc_upper"), metrics["mcc"][:3]
        ):
            row[column] = finite_number(value)
        gb_rows.append(row)
    counts["S18"] = merge_rows(path, "model", gb_rows)
    fieldnames, rows = read_csv(path)
    task_order = list(dict.fromkeys(row["dataset"] for row in rows))
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["dataset"]].append(row)
    for task, task_rows in by_task.items():
        task_rows.sort(key=lambda row: (-float(row["mcc_mean"]), row["model"]))
        assign_competition_ranks(task_rows)
    rows = [row for task in task_order for row in by_task[task]]
    write_csv(path, fieldnames, rows)

    classification_root = (
        args.run_root
        / "gpn_star_supplementary_20260926"
        / "results"
        / "classification"
        / "Probe_GPNStar_MSA_sequence_classification"
    )

    # Binary URE and HAR/HAQER use their existing run-level source-table schemas.
    run_table_specs = [
        (
            "URE_binary_classification",
            "S19_binary_classification_across_five_ure_enhancer_control_sets.csv",
        ),
        (
            "HAQER",
            "S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv",
        ),
        (
            "HAR",
            "S37_classification_of_human_accelerated_regions_across_negative_controls.csv",
        ),
    ]
    for subdir, filename in run_table_specs:
        path = tables / filename
        _, old_rows = read_csv(path)
        new_rows = []
        for model, task, metrics in result_records(classification_root):
            if subdir == "URE_binary_classification":
                expected = task.startswith("URE_binary_")
                table_task = (
                    task.replace("URE_binary_", "URE_", 1) if expected else task
                )
            elif subdir == "HAQER":
                expected = task.startswith("v1_HAQERs_")
                table_task = task
                if expected:
                    match = re.search(r"negset(\d+)$", task)
                    negative_set = int(match.group(1)) if match else None
                    replicate = EVOLUTION_NEGATIVE_SETS[subdir].get(negative_set)
                    if replicate is None:
                        raise ValueError(f"Unmapped HAQER result task: {task}")
                    table_task = f"HAQER_combined_rep{replicate}"
            else:
                expected = task.startswith("v1_7_HARs_")
                table_task = task
                if expected:
                    match = re.search(r"negset(\d+)$", task)
                    negative_set = int(match.group(1)) if match else None
                    replicate = EVOLUTION_NEGATIVE_SETS[subdir].get(negative_set)
                    if replicate is None:
                        raise ValueError(f"Unmapped HAR result task: {task}")
                    table_task = f"HAR_combined_rep{replicate}"
            if not expected:
                continue
            template = next((row for row in old_rows if row["run"] == table_task), None)
            if template is None:
                continue
            run_dir = f"<gpn-star-supplementary-root>/results/classification/Probe_GPNStar_MSA_sequence_classification/{task}/{model}"
            new_rows.append(probe_row(template, model, table_task, metrics, run_dir))
        counts[filename.split("_")[0]] = merge_rows(path, "model", new_rows)
        re_rank_task_table(path)

    # Multiclass URE stores one row per class; class labels are copied from the
    # current source table so label order remains tied to the benchmark table.
    path = (
        tables
        / "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv"
    )
    _, old_rows = read_csv(path)
    labels = {}
    for row in old_rows:
        labels[(row["dataset"], row["class_id"])] = row["class_label"]
    new_rows = []
    for model, task, metrics in result_records(classification_root):
        if not task.startswith("URE_multi_"):
            continue
        dataset = task.replace("URE_multi_", "URE_")
        template = next((row for row in old_rows if row["dataset"] == dataset), None)
        if template is None:
            continue
        f1_values = metrics.get("per_class_f1")
        if not isinstance(f1_values, list) or len(f1_values) != 5:
            continue
        for class_id, value in enumerate(f1_values):
            class_key = str(class_id)
            if (dataset, class_key) not in labels:
                raise ValueError(f"class {class_id} label not found for {dataset}")
            new_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "class_id": class_key,
                    "class_label": labels[(dataset, class_key)],
                    "per_class_f1": finite_number(value),
                }
            )
    counts["S20"] = merge_rows(path, "model", new_rows)
    fieldnames, rows = read_csv(path)
    datasets = list(dict.fromkeys(row["dataset"] for row in old_rows))
    rows.sort(
        key=lambda row: (
            datasets.index(row["dataset"]),
            row["model"],
            int(row["class_id"]),
        )
    )
    write_csv(path, fieldnames, rows)

    # Unsupervised genomic-element summaries are built from the active matching
    # inputs and the completed three-seed candidate tables.
    for task, prefix in (("ENCODE", "S11"), ("GENCODEv45", "S13"), ("UCSC", "S15")):
        source_path = (
            args.run_root
            / "gpn_star_supplementary_20260926"
            / "unsupervised"
            / task
            / "tables_candidate"
            / "source.csv"
        )
        for target_name, source_name, model_col in (
            (
                f"{prefix}_seed_level_representation_separation_for_"
                + {
                    "ENCODE": "encode_candidate_cis_regulatory_elements",
                    "GENCODEv45": "gencode_gene_structure_classes",
                    "UCSC": "curated_refseq_gene_structure_classes",
                }[task]
                + ".csv",
                "source.csv",
                "model_name",
            ),
            (
                f"S{int(prefix[1:]) + 1}_summary_of_representation_separation_for_"
                + {
                    "ENCODE": "encode_candidate_cis_regulatory_elements",
                    "GENCODEv45": "gencode_gene_structure_classes",
                    "UCSC": "curated_refseq_gene_structure_classes",
                }[task]
                + ".csv",
                "summary.csv",
                "model_name",
            ),
        ):
            target_path = tables / target_name
            incoming = list(
                csv.DictReader(
                    (source_path.parent / source_name).open(
                        newline="", encoding="utf-8"
                    )
                )
            )
            if {row["model_name"] for row in incoming} != MODELS:
                raise ValueError(
                    f"{task}: candidate table does not contain all three GPN-Star models"
                )
            expected_rows = 18 if source_name == "source.csv" else 6
            if len(incoming) != expected_rows:
                raise ValueError(
                    f"{task}: expected {expected_rows} GPN-Star rows in {source_name}, got {len(incoming)}"
                )
            fieldnames, old_rows = read_csv(target_path)
            rows = [row for row in old_rows if row.get(model_col) not in MODELS]
            normalized = []
            for row in incoming:
                clean = {
                    key: ("" if value.lower() == "nan" else value)
                    for key, value in row.items()
                }
                normalized.append(clean)
            rows.extend(normalized)
            if source_name == "source.csv":
                rows.sort(
                    key=lambda row: (row["model_name"], int(row["seed"]), row["method"])
                )
            else:
                rows.sort(key=lambda row: (row["model_name"], row["method"]))
            write_csv(target_path, fieldnames, rows)
            counts[target_path.stem.split("_")[0]] = len(normalized)

    for table, added in counts.items():
        print(f"{table}: refreshed {added} GPN-Star rows")


if __name__ == "__main__":
    main()
