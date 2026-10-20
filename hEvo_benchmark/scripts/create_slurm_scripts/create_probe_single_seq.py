from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jinja2
from cli_utils import (
    print_banner,
    print_directory,
    print_settings,
    print_task_model_summary,
)
from config_utils import (
    choose_path,
    choose_value,
    resolve_supername,
    default_config_path,
    find_benchmark_root,
    get_cli_defaults,
    load_config,
)

CURRENT_FILE = Path(__file__).resolve()
if str(CURRENT_FILE.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_FILE.parent))


TOOL_NAME = "create_probe_single_seq"
BENCHMARK_ROOT = find_benchmark_root(__file__)
DEFAULT_PYTHON_EXEC = sys.executable

SBATCH_TEMPLATE = """#!/bin/bash

#SBATCH --job-name=Probe_{{ model_name }}_{{ dataset_name }}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={{ 8 if model_name in ["GPN-MSA", "GPN-Star-V", "GPN-Star-M", "GPN-Star-P"] else 4 }}
#SBATCH --mem={{ "128G" if model_name in ["GPN-MSA", "GPN-Star-V", "GPN-Star-M", "GPN-Star-P"] else "64G" }}
#SBATCH --time=48:00:00
#SBATCH --output={{ sbatch_dir }}/%x.%j.out
#SBATCH --error={{ sbatch_dir }}/%x.%j.err

set -euo pipefail
{% if model_name == "GPN-MSA" %}: "${GPN_MSA_PATH:?Set GPN_MSA_PATH before submitting this job}"
test -f "${GPN_MSA_PATH}"
{% elif model_name in ["GPN-Star-V", "GPN-Star-M", "GPN-Star-P"] %}: "${GPN_STAR_MSA_STORES_CSV:?Set comma-separated GPN_STAR_MSA_STORES_CSV}"
: "${GPN_STAR_MSA_SPECIES_CSV:?Set comma-separated GPN_STAR_MSA_SPECIES_CSV}"
IFS=',' read -r -a gpn_star_msa_stores <<< "${GPN_STAR_MSA_STORES_CSV}"
IFS=',' read -r -a gpn_star_msa_species <<< "${GPN_STAR_MSA_SPECIES_CSV}"
gpn_star_store_count=$(awk -F, '{print NF}' <<< "${GPN_STAR_MSA_STORES_CSV}")
gpn_star_species_count=$(awk -F, '{print NF}' <<< "${GPN_STAR_MSA_SPECIES_CSV}")
if [[ "${gpn_star_store_count}" == "0" || "${gpn_star_store_count}" != "${gpn_star_species_count}" ]]; then
  echo "GPN-Star MSA store and species-count lists must be nonempty and have equal lengths" >&2
  exit 2
fi
for store in "${gpn_star_msa_stores[@]}"; do test -d "${store}"; done
{% endif %}
nvidia-smi
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Allocated GPUs: $SLURM_JOB_GPUS"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "========================================================="

{{ python_exec }} {{ evaluate_script }} \
{% if protocol_arg %}  --protocol {{ protocol }} \
{% endif %}  --model_paths {{ model_path }} \
  --model_name {{ model_name }} \
  --dataset_files {{ train_file }},{{ test_file }} \
  --dataset_name {{ dataset_name }} \
  --hdf5_dir {{ hdf5_dir }} \
  --batch_size 32 \
  --epochs 50 \
  --num_workers 0 \
  --out_dir {{ out_dir }} \
  --log_dir {{ log_dir }} \
  --max_length {{ max_length }} \
  --precompute_batch_size {{ 16 if model_name == "GPN-MSA" else 8 if model_name in ["GPN-Star-V", "GPN-Star-M", "GPN-Star-P"] else 20 }} \
  --label_col {{ label_col }} \
  --seq_col {{ seq_col }} \
  --sep {{ sep }} \
  --pooling mean{% if model_name == "GPN-MSA" %} \\
  --gpn_msa_path "${GPN_MSA_PATH}" \\
  --gpn_window_size "${GPN_MSA_WINDOW_SIZE:-128}" \\
  --gpn_center_window_size "${GPN_MSA_CENTER_WINDOW_SIZE:-{{ center_window_size }}}"{% elif model_name in ["GPN-Star-V", "GPN-Star-M", "GPN-Star-P"] %} \
  --gpn_star_msa_stores "${gpn_star_msa_stores[@]}" \
  --gpn_star_msa_species "${gpn_star_msa_species[@]}" \
  --gpn_star_window_size "${GPN_STAR_WINDOW_SIZE:-1024}" \
  --gpn_star_center_window_size "${GPN_STAR_CENTER_WINDOW_SIZE:-{{ center_window_size }}}"{% endif %}

echo "========================================================="
echo "Job completed At:"
date
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for single-sequence probe evaluation."
    )
    parser.add_argument("--supername", default=None)
    parser.add_argument(
        "--config-path", type=Path, default=default_config_path(__file__)
    )
    parser.add_argument("--python-exec", default=None)
    parser.add_argument("--evaluate-script", type=Path, default=None)
    parser.add_argument(
        "--protocol",
        choices=("auto", "standard", "pareto"),
        default=None,
        help="Probe protocol. Auto selects Pareto for NT_Pareto and standard otherwise.",
    )
    parser.add_argument("--sbatch-root", type=Path, default=None)
    parser.add_argument("--slurm-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--hdf5-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    return parser.parse_args()


def resolve_evaluator(
    args: argparse.Namespace, defaults: dict, supername: str
) -> tuple[Path, str, bool]:
    """Resolve the evaluator script and protocol with legacy path support."""
    unified_script = (
        BENCHMARK_ROOT / "scripts" / "evaluate" / "probe_single_seq_input.py"
    )
    protocol = choose_value(args.protocol, defaults.get("protocol"), "auto")
    if protocol == "auto":
        protocol = "pareto" if supername == "NT_Pareto" else "standard"
    if protocol not in {"standard", "pareto"}:
        raise ValueError(
            f"Unsupported probe protocol {protocol!r}; expected auto, standard, or pareto"
        )

    # Explicit CLI paths remain the strongest override. Older configs may
    # still provide evaluate_script; honor that when no protocol was selected.
    if args.evaluate_script is not None:
        return args.evaluate_script, protocol, False
    if args.protocol is None and defaults.get("evaluate_script"):
        evaluate_script = choose_path(
            None, defaults.get("evaluate_script"), unified_script
        )
        return (
            evaluate_script,
            protocol,
            evaluate_script.resolve() == unified_script.resolve(),
        )
    return unified_script, protocol, True


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    defaults = get_cli_defaults(config, TOOL_NAME)

    python_exec = choose_value(
        args.python_exec, defaults.get("python_exec"), DEFAULT_PYTHON_EXEC
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
    hdf5_root = choose_path(
        args.hdf5_root,
        defaults.get("hdf5_root"),
        BENCHMARK_ROOT / "datasets_emb_H5",
    )
    log_root = choose_path(
        args.log_root, defaults.get("log_root"), BENCHMARK_ROOT / "Probe_runs"
    )

    task_map = {task["supername"]: task for task in config["Tasks"]}
    supername = resolve_supername(
        choose_value(args.supername, defaults.get("supername"), ""),
        list(task_map.keys()),
    )
    evaluate_script, protocol, protocol_arg = resolve_evaluator(
        args, defaults, supername
    )

    sbatch_dir = sbatch_root / f"Probe_{supername}"
    slurm_dir = slurm_root / f"Probe_{supername}"
    results_dir = results_root / f"Probe_{supername}"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    task_info = task_map[supername]

    print_banner(
        "Probe Single Seq",
        "Generating SLURM scripts from the benchmark root with portable relative defaults.",
    )
    print_settings(
        "Resolved Settings",
        {
            "supername": supername,
            "benchmark_root": BENCHMARK_ROOT,
            "config_path": args.config_path,
            "python_exec": python_exec,
            "evaluate_script": evaluate_script,
            "protocol": protocol if protocol_arg else "(selected by evaluator path)",
            "sbatch_dir": sbatch_dir,
            "slurm_dir": slurm_dir,
            "results_dir": results_dir,
            "hdf5_root": hdf5_root,
            "log_root": log_root,
        },
        value_styles={"supername": "bold green"},
    )

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(SBATCH_TEMPLATE)
    task_model_counts = {sub_task["name"]: 0 for sub_task in task_info["subtasks"]}

    for model_info in config["Models"]:
        model_name = model_info["name"]
        model_path = model_info["url"]
        max_length = int(model_info.get("max_length", 1024))
        log_dir = log_root / model_name
        log_dir.mkdir(parents=True, exist_ok=True)

        for sub_task in task_info["subtasks"]:
            task_name = sub_task["name"]
            center_window_size = int(sub_task.get("center_window_size", max_length))
            train_file = os.path.join(
                task_info["path"], sub_task["train_path"].lstrip("/")
            )
            test_file = os.path.join(
                task_info["path"], sub_task["test_path"].lstrip("/")
            )
            out_dir = results_dir / task_name / model_name
            hdf5_dir = hdf5_root / supername / task_name
            out_dir.mkdir(parents=True, exist_ok=True)
            hdf5_dir.mkdir(parents=True, exist_ok=True)

            script_path = slurm_dir / f"probe_{model_name}_{task_name}.sh"
            script_content = template.render(
                python_exec=python_exec,
                evaluate_script=str(evaluate_script),
                protocol=protocol,
                protocol_arg=protocol_arg,
                model_name=model_name,
                model_path=model_path,
                dataset_name=task_name,
                train_file=train_file,
                test_file=test_file,
                hdf5_dir=str(hdf5_dir),
                out_dir=str(out_dir),
                log_dir=str(log_dir),
                max_length=max_length,
                center_window_size=center_window_size,
                label_col=sub_task.get("label_col", 3),
                seq_col=sub_task.get("sequence_col", 0),
                sep=sub_task.get("sep", "space"),
                sbatch_dir=str(sbatch_dir),
            )
            script_path.write_text(script_content, encoding="utf-8")
            task_model_counts[task_name] += 1

    print_directory("SBATCH Script Root", slurm_dir)
    print_task_model_summary("Task Coverage", task_model_counts)


if __name__ == "__main__":
    main()
