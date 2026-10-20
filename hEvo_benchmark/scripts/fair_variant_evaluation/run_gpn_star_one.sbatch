#!/usr/bin/env bash
#SBATCH --job-name=fair-gpn-star
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# One frozen dataset × one native GPN-Star model.  Paths and model revisions
# are caller-supplied; no score-directed sampling is possible in this recipe.
set -euo pipefail

: "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT}"
: "${INPUT_DATASET:?Set INPUT_DATASET to the frozen paired CSV}"
: "${RUN_ROOT:?Set RUN_ROOT to an external run directory}"
: "${MODEL_ID:?Set MODEL_ID}"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${MODEL_REVISION:?Set MODEL_REVISION}"
: "${MSA_STORES_CSV:?Set semicolon-separated MSA_STORES_CSV}"
: "${MSA_SPECIES_CSV:?Set semicolon-separated MSA_SPECIES_CSV}"

test -f "${INPUT_DATASET}"
test -d "${MODEL_DIR}"
TASK_NAME="${TASK_NAME:-CausalDB_hg38_canonical_refalt}"
mkdir -p "${RUN_ROOT}/inputs/${MODEL_ID}" "${RUN_ROOT}/scores/${MODEL_ID}" \
  "${RUN_ROOT}/results/${MODEL_ID}/${TASK_NAME}" \
  "${RUN_ROOT}/logs/${MODEL_ID}/${TASK_NAME}" \
  "${RUN_ROOT}/manifests/${MODEL_ID}"
NORMALIZED="${RUN_ROOT}/inputs/${MODEL_ID}/${TASK_NAME}.tsv"
OUTPUT="${RUN_ROOT}/scores/${MODEL_ID}/${TASK_NAME}.tsv"

/mnt/hpc/home/xuqinyi/micromamba/envs/llmbm/bin/python \
  "${BENCHMARK_ROOT}/scripts/evaluate/adapters/gpn_star_input.py" \
  --input "${INPUT_DATASET}" --output "${NORMALIZED}" \
  --task-name "${TASK_NAME}" --score-mode neg_llr

IFS=';' read -r -a msa_stores <<< "${MSA_STORES_CSV}"
IFS=';' read -r -a msa_species <<< "${MSA_SPECIES_CSV}"
args=(
  --model_paths "${MODEL_DIR}" --model_name "${MODEL_ID}"
  --dataset_files "${NORMALIZED}" --dataset_name "${TASK_NAME}"
  --h5_path "${RUN_ROOT}/h5" --out_dir "${RUN_ROOT}/results/${MODEL_ID}/${TASK_NAME}"
  --log_dir "${RUN_ROOT}/logs/${MODEL_ID}/${TASK_NAME}" --bootstrap 0
  --gpn-star-model-id "${MODEL_ID}" --gpn-star-model-dir "${MODEL_DIR}"
  --gpn-star-model-revision "${MODEL_REVISION}"
  --gpn-star-input "${NORMALIZED}" --gpn-star-output "${OUTPUT}"
  --gpn-star-manifest "${RUN_ROOT}/manifests/${MODEL_ID}/${TASK_NAME}.json"
  --gpn-star-window-bp "${WINDOW_BP:-1024}"
  --precompute_batch_size "${PRECOMPUTE_BATCH_SIZE:-4}"
  --gpn-star-msa-dataset "${MSA_DATASET:-}"
  --gpn-star-msa-revision "${MSA_REVISION:-}"
)
for store in "${msa_stores[@]}"; do args+=(--gpn-star-msa-store "${store}"); done
args+=(--gpn-star-msa-species "${msa_species[@]}")

/mnt/hpc/home/xuqinyi/micromamba/envs/llmbm/bin/python \
  "${BENCHMARK_ROOT}/scripts/evaluate/zeroshot_paried_seq_input.py" \
  "${args[@]}"
