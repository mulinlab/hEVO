#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/DeepSEA_lora/category_%j.out
#SBATCH --error=logs/DeepSEA_lora/category_%j.err

set -euo pipefail

ROOT="<source-project-root>"
SCRIPT=${ROOT}/scripts/DeepSEA_lora/evaluate_category_metrics.py
RUN_DIR=${RUN_DIR:?RUN_DIR must point to one completed DeepSEA LoRA run directory}

cd "${ROOT}/scripts/DeepSEA_lora"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[category-eval] run_dir=${RUN_DIR}"
echo "[category-eval] host=$(hostname) date=$(date -Is)"

"<micromamba-root>/bin/micromamba" run --clean-env -n llmbm python -m torch.distributed.run \
  --standalone --nproc_per_node=4 "${SCRIPT}" \
  --run_dir "${RUN_DIR}" \
  --config "${ROOT}/config.yaml" \
  --dataset_root "${ROOT}/datasets/DeepSEA" \
  --metadata "${ROOT}/datasets/DeepSEA/deepsea_metadata.tsv"
