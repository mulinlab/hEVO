#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from config_utils import (
    default_config_path,
    find_benchmark_root,
    load_config,
    read_yaml,
)


BENCHMARK_ROOT = find_benchmark_root(__file__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the monolithic benchmark config into project/models/tasks YAML files."
    )
    parser.add_argument(
        "--config-path", type=Path, default=default_config_path(__file__)
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=None,
        help="Optional source config. Use this to rebuild from a monolithic backup.",
    )
    parser.add_argument("--configs-dir", type=Path, default=BENCHMARK_ROOT / "configs")
    parser.add_argument(
        "--backup-suffix",
        default=".monolith.bak",
        help="Backup suffix for the original config.yaml.",
    )
    return parser.parse_args()


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def normalize_tasks(tasks: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for task in tasks:
        task_copy = dict(task)
        subtasks = []
        for subtask in task.get("subtasks", []):
            subtask_copy = dict(subtask)
            if task_copy.get("supername") == "NT_Pareto":
                train_path = str(subtask_copy.get("train_path", ""))
                test_path = str(subtask_copy.get("test_path", ""))
                subtask_copy["train_path"] = train_path.replace(
                    "train.txt", "train.csv"
                )
                subtask_copy["test_path"] = test_path.replace("test.txt", "test.csv")
            subtasks.append(subtask_copy)
        task_copy["subtasks"] = subtasks
        normalized.append(task_copy)
    return normalized


def main() -> None:
    args = parse_args()
    source_config = args.source_config or args.config_path
    raw_source = read_yaml(source_config)
    if isinstance(raw_source, dict) and "ConfigFiles" in raw_source:
        raw_config = load_config(source_config)
    else:
        raw_config = raw_source

    project_yaml = {
        "DownloadConfig": raw_config.get("DownloadConfig", {}),
        "ProjectPaths": {
            "benchmark_root": "${BENCHMARK_ROOT}",
            "sbatch_root": "${BENCHMARK_ROOT}/.sbatch",
            "slurm_root": "${BENCHMARK_ROOT}/slurm",
            "results_root": "${BENCHMARK_ROOT}/results",
            "probe_runs_root": "${BENCHMARK_ROOT}/Probe_runs",
            "datasets_h5_root": "${BENCHMARK_ROOT}/datasets_H5",
            "datasets_emb_h5_root": "${BENCHMARK_ROOT}/datasets_emb_H5",
        },
        "CliDefaults": {
            "create_probe_single_seq": {
                "supername": "NT_Pareto",
                "protocol": "auto",
                "sbatch_root": "${BENCHMARK_ROOT}/.sbatch",
                "slurm_root": "${BENCHMARK_ROOT}/slurm",
                "results_root": "${BENCHMARK_ROOT}/results",
                "hdf5_root": "${BENCHMARK_ROOT}/datasets_emb_H5",
                "log_root": "${BENCHMARK_ROOT}/Probe_runs",
            },
            "create_zeroshot_paired_seq": {
                "evaluate_script": "${BENCHMARK_ROOT}/scripts/evaluate/zeroshot_paried_seq_input.py",
                "sbatch_root": "${BENCHMARK_ROOT}/.sbatch",
                "slurm_root": "${BENCHMARK_ROOT}/slurm",
                "results_root": "${BENCHMARK_ROOT}/results",
                "h5_root": "${BENCHMARK_ROOT}/datasets_H5",
                "log_root": "${BENCHMARK_ROOT}/Probe_runs",
            },
            "create_popu_classification": {
                "script_path": "${BENCHMARK_ROOT}/scripts/population_classify/extract_single_sample_emb.py",
                "dataset_dir": "${BENCHMARK_ROOT}/datasets/Ancestry_Classification/jsonl_2000v_1500s_511bp",
                "results_dir": "${BENCHMARK_ROOT}/results/1kg_popu_classification/2000v_1500s_511bp",
                "log_dir": "${BENCHMARK_ROOT}/.sbatch/2000v_1500s_511bp",
                "slurm_dir": "${BENCHMARK_ROOT}/slurm/popu_emb_2000v",
            },
        },
    }
    models_yaml = {"Models": raw_config.get("Models", [])}
    tasks_yaml = {"Tasks": normalize_tasks(raw_config.get("Tasks", []))}
    manifest_yaml = {
        "PathVars": {
            "BENCHMARK_ROOT": "${CONFIG_DIR}",
        },
        "ConfigFiles": {
            "project": "configs/project.yaml",
            "models": "configs/models.yaml",
            "tasks": "configs/tasks.yaml",
        },
    }

    project_path = args.configs_dir / "project.yaml"
    models_path = args.configs_dir / "models.yaml"
    tasks_path = args.configs_dir / "tasks.yaml"

    dump_yaml(project_path, project_yaml)
    dump_yaml(models_path, models_yaml)
    dump_yaml(tasks_path, tasks_yaml)

    original_text = args.config_path.read_text(encoding="utf-8")
    backup_path = args.config_path.with_name(args.config_path.name + args.backup_suffix)
    backup_path.write_text(original_text, encoding="utf-8")
    dump_yaml(args.config_path, manifest_yaml)

    print(f"Wrote: {project_path}")
    print(f"Wrote: {models_path}")
    print(f"Wrote: {tasks_path}")
    print(f"Updated manifest: {args.config_path}")
    print(f"Backup saved to: {backup_path}")


if __name__ == "__main__":
    main()
