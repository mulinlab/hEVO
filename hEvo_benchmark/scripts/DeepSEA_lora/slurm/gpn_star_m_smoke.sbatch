#!/usr/bin/env bash
#SBATCH --job-name=ds-gpnm-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
HPC_HOME="${HPC_HOME:-${HOME}}"
REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
PYTHON="${LLMBM_PYTHON:-${HPC_HOME}/micromamba/envs/llmbm/bin/python}"
CODE="${REPO}/hEvo_benchmark/scripts/DeepSEA_lora/run_deepsea_lora.py"
ROOT="${DEEPSEA_GPN_STAR_RUN_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/deepsea_gpn_star_m_v1}"
CONFIG="${REPO}/hEvo_benchmark/configs/deepsea_gpn_star_m.yaml"
DATA="${DEEPSEA_DATA_ROOT:-${HPC_HOME}/hEvo/dataset/deepsea}"
mkdir -p "$ROOT/training/smoke"
exec "$PYTHON" "$CODE" --config "$CONFIG" \
  --model_name GPN-Star-M --model_scope all --dataset_root "$DATA" \
  --output_root "$ROOT/training" --output_dir "$ROOT/training/smoke" \
  --gpn_coordinate_root "$ROOT/coordinates" --train_samples 4 --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --mixed_precision bf16 \
  --smoke_test
