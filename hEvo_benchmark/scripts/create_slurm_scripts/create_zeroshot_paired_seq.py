from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jinja2

CURRENT_FILE = Path(__file__).resolve()
if str(CURRENT_FILE.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_FILE.parent))

from cli_utils import (  # noqa: E402 - local path bootstrap
    print_banner,
    print_directory,
    print_settings,
    print_task_model_summary,
)
from config_utils import (  # noqa: E402 - local path bootstrap
    choose_path,
    choose_value,
    resolve_supername,
    default_config_path,
    find_benchmark_root,
    get_cli_defaults,
    load_config,
)


TOOL_NAME = "create_zeroshot_paired_seq"
BENCHMARK_ROOT = find_benchmark_root(__file__)
DEFAULT_PYTHON_EXEC = sys.executable

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=zeroshot_{{ model_name }}_{{ dataset_name }}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output={{ sbatch_dir }}/%x.%j.out
#SBATCH --error={{ sbatch_dir }}/%x.%j.err

set -euo pipefail

nvidia-smi
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "========================================================="

{{ gpn_msa_setup }}
{% if evaluator_cwd %}cd "{{ evaluator_cwd }}"
{% endif %}
{{ python_exec }} {{ evaluate_script }} \
  --model_paths {{ model_path }} \
  --model_name {{ model_name }} \
  --dataset_files {{ train_file }} \
  --dataset_name {{ dataset_name }} \
  --sep "," \
  --h5_path {{ h5_path }} \
  --batch_size 32 \
  --num_workers 4 \
  --ref_col {{ ref_col }} \
  --alt_col {{ alt_col }} \
  --label_col {{ label_col }} \
  --out_dir {{ out_dir }} \
  --log_dir {{ log_dir }} \
  --max_length {{ max_length }} \
  --precompute_batch_size 16{{ use_abs_arg }}{{ gpn_msa_args }}

echo "========================================================="
echo "Job completed At:"
date
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for paired zero-shot evaluation."
    )
    parser.add_argument("--supername", default=None)
    parser.add_argument(
        "--config-path", type=Path, default=default_config_path(__file__)
    )
    parser.add_argument("--python-exec", default=None)
    parser.add_argument("--evaluate-script", type=Path, default=None)
    parser.add_argument(
        "--gpn-msa-evaluate-script", type=Path, default=None,
        help="Coordinate-aware native MSA evaluator used only for GPN-MSA jobs.",
    )
    parser.add_argument("--sbatch-root", type=Path, default=None)
    parser.add_argument("--slurm-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--h5-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument(
        "--evaluator-cwd",
        type=Path,
        default=None,
        help="Optional working directory for evaluators that resolve relative model configs.",
    )
    parser.add_argument(
        "--gpn-msa-path",
        type=Path,
        default=None,
        help="Optional GPN-MSA Zarr/zip asset passed only to GPN-MSA jobs.",
    )
    parser.add_argument(
        "--gpn-zarr2-path",
        type=Path,
        default=None,
        help="Optional directory containing the Zarr v2 compatibility packages.",
    )
    parser.add_argument(
        "--gpn-package-path",
        type=Path,
        default=None,
        help="Optional GPN source checkout root containing the gpn package.",
    )
    parser.add_argument(
        "--use-abs",
        action="store_true",
        help="Pass --use_abs to the evaluator for absolute-LLR scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    defaults = get_cli_defaults(config, TOOL_NAME)

    python_exec = choose_value(
        args.python_exec, defaults.get("python_exec"), DEFAULT_PYTHON_EXEC
    )
    evaluate_script = choose_path(
        args.evaluate_script,
        defaults.get("evaluate_script"),
        BENCHMARK_ROOT / "scripts" / "evaluate" / "zeroshot_paried_seq_input.py",
    )
    sbatch_root = choose_path(
        args.sbatch_root, defaults.get("sbatch_root"), BENCHMARK_ROOT / ".sbatch"
    )
    slurm_root = choose_path(
        args.slurm_root, defaults.get("slurm_root"), BENCHMARK_ROOT / "slurm"
    )
    results_root = choose_path(
        args.results_root, defaults.get("results_root"), BENCHMARK_ROOT / "results"
    )
    h5_root = choose_path(
        args.h5_root, defaults.get("h5_root"), BENCHMARK_ROOT / "datasets_H5"
    )
    log_root = choose_path(
        args.log_root, defaults.get("log_root"), BENCHMARK_ROOT / "Probe_runs"
    )
    evaluator_cwd = (
        args.evaluator_cwd.resolve() if args.evaluator_cwd is not None else None
    )

    task_map = {task["supername"]: task for task in config["Tasks"]}
    supername = resolve_supername(
        choose_value(args.supername, defaults.get("supername"), ""),
        list(task_map.keys()),
    )

    sbatch_dir = sbatch_root / f"zeroshot_{supername}"
    slurm_dir = slurm_root / f"zeroshot_{supername}"
    results_dir = results_root / f"zeroshot_{supername}"
    h5_path = h5_root / supername
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    h5_path.mkdir(parents=True, exist_ok=True)

    task_info = task_map[supername]

    print_banner(
        "Zero-shot Paired Seq",
        "Generating paired-sequence zero-shot SLURM scripts with portable benchmark-root defaults.",
    )
    print_settings(
        "Resolved Settings",
        {
            "supername": supername,
            "benchmark_root": BENCHMARK_ROOT,
            "config_path": args.config_path,
            "python_exec": python_exec,
            "evaluate_script": evaluate_script,
            "sbatch_dir": sbatch_dir,
            "slurm_dir": slurm_dir,
            "results_dir": results_dir,
            "h5_path": h5_path,
            "log_root": log_root,
            "evaluator_cwd": evaluator_cwd,
        },
        value_styles={"supername": "bold red"},
    )

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(SBATCH_TEMPLATE)
    task_model_counts = {sub_task["name"]: 0 for sub_task in task_info["subtasks"]}

    for model_info in config["Models"]:
        model_name = model_info["name"]
        model_evaluate_script = evaluate_script
        if model_name == "GPN-MSA":
            if args.gpn_msa_path is None:
                raise ValueError("GPN-MSA native scoring requires --gpn-msa-path")
            if args.gpn_msa_evaluate_script is not None:
                model_evaluate_script = args.gpn_msa_evaluate_script.resolve()
            evaluator_text = model_evaluate_script.read_text(encoding="utf-8")
            if "--gpn_msa_path" not in evaluator_text or "precompute_and_cache_gpn_msa" not in evaluator_text:
                raise ValueError(
                    "GPN-MSA needs a native MSA evaluator. Set --gpn-msa-evaluate-script "
                    "to the coordinate-aware scorer used by the recorded run."
                )
        model_path = model_info["url"]
        max_length = int(model_info.get("max_length", 1024))
        log_dir = log_root / model_name
        log_dir.mkdir(parents=True, exist_ok=True)

        for sub_task in task_info["subtasks"]:
            task_name = sub_task["name"]
            train_file = os.path.join(
                task_info["path"], sub_task["train_path"].lstrip("/")
            )
            out_dir = results_dir / task_name / model_name
            out_dir.mkdir(parents=True, exist_ok=True)

            script_path = slurm_dir / f"zeroshot_{model_name}_{task_name}.sh"
            script_content = template.render(
                python_exec=python_exec,
                evaluate_script=str(model_evaluate_script),
                model_name=model_name,
                model_path=model_path,
                dataset_name=task_name,
                train_file=train_file,
                h5_path=str(h5_path),
                out_dir=str(out_dir),
                log_dir=str(log_dir),
                max_length=max_length,
                ref_col=sub_task.get("ref_col", 2),
                alt_col=sub_task.get("alt_col", 3),
                label_col=sub_task.get("label_col", 3),
                gpn_msa_setup=(
                    'export PYTHONPATH="'
                    + str(args.gpn_zarr2_path)
                    + ":"
                    + str(args.gpn_package_path)
                    + ':${PYTHONPATH:-}"\n'
                    + "export WANDB_MODE=disabled"
                    if model_name == "GPN-MSA"
                    and args.gpn_msa_path is not None
                    and args.gpn_zarr2_path is not None
                    else ""
                ),
                evaluator_cwd=str(evaluator_cwd) if evaluator_cwd is not None else None,
                gpn_msa_args=(
                    " \\\n  --chrom_col "
                    + str(sub_task.get("chrom_col", 4))
                    + " \\\n  --pos_col "
                    + str(sub_task.get("pos_col", 5))
                    + " \\\n  --gpn_msa_path "
                    + str(args.gpn_msa_path)
                    + " \\\n  --gpn_window_size 128"
                    if model_name == "GPN-MSA" and args.gpn_msa_path is not None
                    else ""
                ),
                use_abs_arg=(" \\\n  --use_abs" if args.use_abs and not model_name.startswith("Mistral-DNA-") else ""),
                sbatch_dir=str(sbatch_dir),
            )
            script_path.write_text(script_content, encoding="utf-8")
            task_model_counts[task_name] += 1

    print_directory("SBATCH Script Root", slurm_dir)
    print_task_model_summary("Task Coverage", task_model_counts)


if __name__ == "__main__":
    main()
