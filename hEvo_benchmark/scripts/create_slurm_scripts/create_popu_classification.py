from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jinja2

CURRENT_FILE = Path(__file__).resolve()
if str(CURRENT_FILE.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_FILE.parent))

from cli_utils import print_banner, print_settings  # noqa: E402 - local path/runtime configuration precedes this import
from config_utils import (  # noqa: E402 - local path bootstrap
    default_config_path,
    find_benchmark_root,
    get_cli_defaults,
    load_config,
)


TOOL_NAME = "create_popu_classification"
BENCHMARK_ROOT = find_benchmark_root(__file__)
DEFAULT_PYTHON_EXEC = sys.executable

SBATCH_TEMPLATE = """#!/bin/bash
set -euo pipefail

#SBATCH --job-name=1kg_emb_{{ model_name }}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-749
#SBATCH --output={{ log_dir }}/{{ model_name }}_emb_%a.out

nvidia-smi
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "========================================================="

FILES=({{ dataset_dir }}/*.jsonl)
IDX1=$((SLURM_ARRAY_TASK_ID * 2))
IDX2=$((SLURM_ARRAY_TASK_ID * 2 + 1))

OUTPUT_DIR="{{ output_dir }}"
mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="{{ python_bin }}"
SCRIPT_PATH="{{ script_path }}"
COMMON_ARGS="--model_path {{ model_path }} --model_name {{ model_name }} --output_dir $OUTPUT_DIR --batch_size {{ batch_size }} --max_length 511 --aggregation multi_stats --topk 500"

INPUT_FILE1=${FILES[$IDX1]}
if [ -f "$INPUT_FILE1" ]; then
    echo "--- [Task 1/2] Processing index $IDX1 ---"
    echo "Input: $INPUT_FILE1"
    $PYTHON_BIN $SCRIPT_PATH --input_file "$INPUT_FILE1" $COMMON_ARGS
else
    echo "Warning: Index $IDX1 file not found."
fi

INPUT_FILE2=${FILES[$IDX2]}
if [ -f "$INPUT_FILE2" ]; then
    echo "--- [Task 2/2] Processing index $IDX2 ---"
    echo "Input: $INPUT_FILE2"
    $PYTHON_BIN $SCRIPT_PATH --input_file "$INPUT_FILE2" $COMMON_ARGS
else
    echo "Note: Index $IDX2 exceeds file list or file not found."
fi

echo "========================================================="
echo "Job completed At: $(date)"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for population embedding extraction."
    )
    parser.add_argument(
        "--config-path", type=Path, default=default_config_path(__file__)
    )
    parser.add_argument("--python-exec", default=None)
    parser.add_argument("--script-path", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--slurm-dir", type=Path, default=None)
    return parser.parse_args()


def choose_path(
    cli_value: Path | None, config_value: str | None, fallback: Path
) -> Path:
    if cli_value is not None:
        return cli_value
    if config_value:
        return Path(config_value)
    return fallback


def choose_value(cli_value: str | None, config_value: str | None, fallback: str) -> str:
    return cli_value or config_value or fallback


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    defaults = get_cli_defaults(config, TOOL_NAME)

    python_exec = choose_value(
        args.python_exec, defaults.get("python_exec"), DEFAULT_PYTHON_EXEC
    )
    script_path = choose_path(
        args.script_path,
        defaults.get("script_path"),
        BENCHMARK_ROOT
        / "scripts"
        / "population_classify"
        / "extract_single_sample_emb.py",
    )
    dataset_dir = choose_path(
        args.dataset_dir,
        defaults.get("dataset_dir"),
        BENCHMARK_ROOT
        / "datasets"
        / "Ancestry_Classification"
        / "jsonl_2000v_1500s_511bp",
    )
    results_dir = choose_path(
        args.results_dir,
        defaults.get("results_dir"),
        BENCHMARK_ROOT / "results" / "1kg_popu_classification" / "2000v_1500s_511bp",
    )
    log_dir_root = choose_path(
        args.log_dir,
        defaults.get("log_dir"),
        BENCHMARK_ROOT / ".sbatch" / "2000v_1500s_511bp",
    )
    slurm_dir = choose_path(
        args.slurm_dir,
        defaults.get("slurm_dir"),
        BENCHMARK_ROOT / "slurm" / "popu_emb_2000v",
    )
    slurm_dir.mkdir(parents=True, exist_ok=True)

    print_banner(
        "Population Classification",
        "Generating population embedding SLURM scripts with benchmark-root relative defaults.",
    )
    print_settings(
        "Resolved Settings",
        {
            "benchmark_root": BENCHMARK_ROOT,
            "config_path": args.config_path,
            "python_exec": python_exec,
            "script_path": script_path,
            "dataset_dir": dataset_dir,
            "results_dir": results_dir,
            "log_dir": log_dir_root,
            "slurm_dir": slurm_dir,
        },
    )

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(SBATCH_TEMPLATE)
    generated_paths: list[Path] = []

    for model_info in config["Models"]:
        model_name = model_info["name"]
        model_path = model_info["url"]
        batch_size = 1 if "evo2" in model_path.lower() else 16

        output_dir = results_dir / f"{model_name}_popu_emb"
        log_dir = log_dir_root / f"{model_name}_popu_emb"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        script_content = template.render(
            model_name=model_name,
            model_path=model_path,
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            log_dir=str(log_dir),
            python_bin=python_exec,
            script_path=str(script_path),
            batch_size=batch_size,
        )

        output_script = slurm_dir / f"run_{model_name}.sh"
        output_script.write_text(script_content, encoding="utf-8")
        generated_paths.append(output_script)

    preview = {path.name: path for path in generated_paths[:20]}
    if preview:
        print_settings("Generated Scripts", preview)
    print_settings("Generation Summary", {"num_scripts": len(generated_paths)})


if __name__ == "__main__":
    main()
