#!/usr/bin/env bash
#SBATCH --job-name=gpn-star-vep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID}"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${MODEL_REVISION:?Set MODEL_REVISION}"
: "${MSA_STORES_CSV:?Set comma-separated MSA_STORES_CSV}"
: "${MSA_SPECIES_CSV:?Set comma-separated MSA_SPECIES_CSV}"
: "${INPUT_TSV:?Set INPUT_TSV}"
: "${OUTPUT_TSV:?Set OUTPUT_TSV}"

IFS=',' read -r -a msa_stores <<< "${MSA_STORES_CSV}"
IFS=',' read -r -a msa_species <<< "${MSA_SPECIES_CSV}"
args=(
  --model_paths "${MODEL_DIR}" --model_name "${MODEL_ID}"
  --dataset_files "${INPUT_TSV}" --dataset_name "${TASK_NAME:-gpn_star}"
  --h5_path "${RUN_ROOT:-.}/h5" --out_dir "$(dirname "${OUTPUT_TSV}")"
  --log_dir "${RUN_ROOT:-.}/logs" --bootstrap 0
  --gpn-star-model-id "${MODEL_ID}" --gpn-star-model-dir "${MODEL_DIR}"
  --gpn-star-model-revision "${MODEL_REVISION}" --gpn-star-input "${INPUT_TSV}"
  --gpn-star-output "${OUTPUT_TSV}" --gpn-star-window-bp "${WINDOW_BP:-1024}"
  --gpn-star-manifest "${RUN_ROOT:-.}/manifests/${MODEL_ID}/${TASK_NAME:-gpn_star}.json"
  --gpn-star-msa-dataset "${MSA_DATASET:-}" --gpn-star-msa-revision "${MSA_REVISION:-}"
)
for store in "${msa_stores[@]}"; do args+=(--gpn-star-msa-store "${store}"); done
args+=(--gpn-star-msa-species "${msa_species[@]}")

micromamba -n llmbm run python \
  "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT}/scripts/evaluate/zeroshot_paried_seq_input.py" \
  "${args[@]}"
