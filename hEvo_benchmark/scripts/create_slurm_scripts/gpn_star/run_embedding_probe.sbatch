#!/usr/bin/env bash
#SBATCH --job-name=gpn-star-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

: "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT}"
: "${MODEL_ID:?Set MODEL_ID to GPN-Star-V, GPN-Star-M, or GPN-Star-P}"
: "${MODEL_DIR:?Set MODEL_DIR to the local GPN-Star checkpoint}"
: "${TASK_NAME:?Set TASK_NAME}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the coordinate-aware train CSV}"
: "${TEST_FILE:?Set TEST_FILE to the coordinate-aware test CSV}"
: "${RUN_ROOT:?Set RUN_ROOT to an external run directory}"
: "${MSA_STORES_CSV:?Set comma-separated local GPN-Star MSA Zarr stores}"
: "${MSA_SPECIES_CSV:?Set comma-separated species counts matching MSA_STORES_CSV}"

case "${MODEL_ID}" in
  GPN-Star-V|GPN-Star-M|GPN-Star-P) ;;
  *) echo "Unsupported GPN-Star MODEL_ID: ${MODEL_ID}" >&2; exit 2 ;;
esac

test -d "${MODEL_DIR}"
test -f "${TRAIN_FILE}"
test -f "${TEST_FILE}"
IFS=',' read -r -a msa_stores <<< "${MSA_STORES_CSV}"
IFS=',' read -r -a msa_species <<< "${MSA_SPECIES_CSV}"
if (( ${#msa_stores[@]} == 0 || ${#msa_stores[@]} != ${#msa_species[@]} )); then
  echo "MSA store and species-count lists must be nonempty and have equal lengths" >&2
  exit 2
fi
for store in "${msa_stores[@]}"; do test -d "${store}"; done

mkdir -p "${RUN_ROOT}/cache" "${RUN_ROOT}/results" "${RUN_ROOT}/logs"
args=(
  --model_paths "${MODEL_DIR}"
  --model_name "${MODEL_ID}"
  --dataset_files "${TRAIN_FILE},${TEST_FILE}"
  --dataset_name "${TASK_NAME}"
  --hdf5_dir "${RUN_ROOT}/cache"
  --out_dir "${RUN_ROOT}/results"
  --log_dir "${RUN_ROOT}/logs"
  --batch_size "${BATCH_SIZE:-32}"
  --precompute_batch_size "${PRECOMPUTE_BATCH_SIZE:-16}"
  --epochs "${EPOCHS:-50}"
  --num_workers "${NUM_WORKERS:-0}"
  --max_length "${MAX_LENGTH:-1000}"
  --sep ","
  --seq_col 1
  --label_col 2
  --pooling mean
  --gpn_star_msa_stores "${msa_stores[@]}"
  --gpn_star_msa_species "${msa_species[@]}"
  --gpn_star_window_size "${GPN_STAR_WINDOW_SIZE:-1024}"
  --gpn_star_center_window_size "${GPN_STAR_CENTER_WINDOW_SIZE:-1000}"
  --bootstrap "${BOOTSTRAP:-1000}"
)
if [[ "${RECOMPUTE:-0}" == "1" ]]; then args+=(--recompute); fi

micromamba -n llmbm run python \
  "${BENCHMARK_ROOT}/scripts/evaluate/probe_single_seq_input.py" \
  "${args[@]}"
