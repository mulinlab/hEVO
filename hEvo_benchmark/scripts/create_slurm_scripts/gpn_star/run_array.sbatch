#!/usr/bin/env bash
#SBATCH --job-name=gpn-star-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail

: "${TASK_MANIFEST:?Set TASK_MANIFEST to a TSV with task_name, input, score_mode columns}"
: "${RUN_ROOT:?Set RUN_ROOT to an external run directory}"
: "${MODEL_ID:?Set MODEL_ID}"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${MODEL_REVISION:?Set MODEL_REVISION}"
: "${MSA_STORES_CSV:?Set comma-separated MSA_STORES_CSV}"
: "${MSA_SPECIES_CSV:?Set comma-separated MSA_SPECIES_CSV}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

task_line=$(awk -F '\t' -v target="$((SLURM_ARRAY_TASK_ID + 1))" '
  !/^#/ && NF >= 3 && tolower($1) != "task_name" { count += 1; if (count == target) print; }
' "${TASK_MANIFEST}")
if [[ -z "${task_line}" ]]; then
  echo "No task row for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
IFS=$'\t' read -r task_name input_path score_mode <<< "${task_line}"

input_dir="${RUN_ROOT}/inputs"
score_dir="${RUN_ROOT}/scores/${MODEL_ID}"
mkdir -p "${input_dir}" "${score_dir}"
normalized="${input_dir}/${task_name}.tsv"
output="${score_dir}/${task_name}.tsv"

micromamba -n llmbm run python \
  "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT}/scripts/evaluate/adapters/gpn_star_input.py" \
  --input "${input_path}" --output "${normalized}" \
  --task-name "${task_name}" --score-mode "${score_mode}"

IFS=',' read -r -a msa_stores <<< "${MSA_STORES_CSV}"
IFS=',' read -r -a msa_species <<< "${MSA_SPECIES_CSV}"
args=(
  --model_paths "${MODEL_DIR}" --model_name "${MODEL_ID}"
  --dataset_files "${normalized}" --dataset_name "${task_name}"
  --h5_path "${RUN_ROOT}/h5" --out_dir "${RUN_ROOT}/results/${MODEL_ID}/${task_name}"
  --log_dir "${RUN_ROOT}/logs/${MODEL_ID}/${task_name}" --bootstrap 0
  --gpn-star-model-id "${MODEL_ID}" --gpn-star-model-dir "${MODEL_DIR}"
  --gpn-star-model-revision "${MODEL_REVISION}"
  --gpn-star-input "${normalized}" --gpn-star-output "${output}"
  --gpn-star-manifest "${RUN_ROOT}/manifests/${MODEL_ID}/${task_name}.json"
  --gpn-star-window-bp "${WINDOW_BP:-1024}"
  --precompute_batch_size "${PRECOMPUTE_BATCH_SIZE:-32}"
  --gpn-star-msa-dataset "${MSA_DATASET:-}"
  --gpn-star-msa-revision "${MSA_REVISION:-}"
)
for store in "${msa_stores[@]}"; do args+=(--gpn-star-msa-store "${store}"); done
args+=(--gpn-star-msa-species "${msa_species[@]}")

micromamba -n llmbm run python \
  "${BENCHMARK_ROOT}/scripts/evaluate/zeroshot_paried_seq_input.py" \
  "${args[@]}"
