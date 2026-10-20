#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/DeepSEA_lora/evo2_category_%j.out
#SBATCH --error=logs/DeepSEA_lora/evo2_category_%j.err

set -euo pipefail
ROOT="<source-project-root>"
RUN_DIR=${RUN_DIR:?RUN_DIR must point to an Evo2 run directory}
cd "${ROOT}/scripts/DeepSEA_lora"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
<micromamba-root>/bin/micromamba run --clean-env -n llmbm python evaluate_evo2_category_metrics.py \
  --run_dir "${RUN_DIR}" --config "${ROOT}/config.yaml" \
  --dataset_root "${ROOT}/datasets/DeepSEA" \
  --metadata "${ROOT}/datasets/DeepSEA/deepsea_metadata.tsv"
