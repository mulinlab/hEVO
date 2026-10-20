#!/usr/bin/env bash
# Run one large native-MSA task as four independent GPU steps on one node.
#SBATCH --job-name=gpn-star-node4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID}"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${MODEL_REVISION:?Set MODEL_REVISION}"
: "${MSA_STORES_CSV:?Set comma-separated MSA_STORES_CSV}"
: "${MSA_SPECIES_CSV:?Set comma-separated MSA_SPECIES_CSV}"
: "${INPUT_TSV:?Set normalized input TSV}"
: "${RUN_ROOT:?Set external run root}"
: "${BENCHMARK_ROOT:?Set benchmark root}"

TASK_NAME="${TASK_NAME:-ClinVar260615_all}"
SHARDS="${SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
shard_root="${RUN_ROOT}/shards/${MODEL_ID}/${TASK_NAME}/${SLURM_JOB_ID}"
score_dir="${RUN_ROOT}/scores/${MODEL_ID}"
log_dir="${RUN_ROOT}/logs/${MODEL_ID}/${TASK_NAME}/${SLURM_JOB_ID}"
mkdir -p "${shard_root}" "${score_dir}" "${log_dir}"

micromamba -n llmbm run python \
  "${BENCHMARK_ROOT}/scripts/evaluate/compat/gpn_star_split.py" \
  --input "${INPUT_TSV}" --output-dir "${shard_root}" --prefix "${TASK_NAME}" \
  --shards "${SHARDS}"

# Resolve the llmbm interpreter once.  Calling micromamba concurrently from
# four GPU steps contends on the shared user cache lock and serializes startup.
  python_bin="$(micromamba -n llmbm run python -c 'import sys; print(sys.executable)')"

IFS=',' read -r -a msa_stores <<< "${MSA_STORES_CSV}"
IFS=',' read -r -a msa_species <<< "${MSA_SPECIES_CSV}"
pids=()
outputs=()

run_shard() {
  local index="$1"
  local input="${shard_root}/${TASK_NAME}_$(printf '%02d' "${index}").tsv"
  local output="${score_dir}/${TASK_NAME}.shard_$(printf '%02d' "${index}").tsv"
  local log="${log_dir}/shard_$(printf '%02d' "${index}").log"
  local args=(
    --model_paths "${MODEL_DIR}" --model_name "${MODEL_ID}"
    --dataset_files "${input}" --dataset_name "${TASK_NAME}"
    --h5_path "${RUN_ROOT}/h5" --out_dir "${RUN_ROOT}/results/${MODEL_ID}/${TASK_NAME}/shard_$(printf '%02d' "${index}")"
    --log_dir "${log_dir}" --bootstrap 0
    --gpn-star-model-id "${MODEL_ID}" --gpn-star-model-dir "${MODEL_DIR}"
    --gpn-star-model-revision "${MODEL_REVISION}" --gpn-star-input "${input}"
    --gpn-star-output "${output}" --gpn-star-window-bp "${WINDOW_BP:-1024}"
    --gpn-star-manifest "${RUN_ROOT}/manifests/${MODEL_ID}/${TASK_NAME}/shard_$(printf '%02d' "${index}").json"
    --gpn-star-device auto --gpn-star-msa-dataset "${MSA_DATASET:-}"
    --gpn-star-msa-revision "${MSA_REVISION:-}"
  )
  for store in "${msa_stores[@]}"; do args+=(--gpn-star-msa-store "${store}"); done
  args+=(--gpn-star-msa-species "${msa_species[@]}")
  srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
    "${python_bin}" \
    "${BENCHMARK_ROOT}/scripts/evaluate/zeroshot_paried_seq_input.py" \
    "${args[@]}" >"${log}" 2>&1
}

for ((index = 0; index < SHARDS; index += 1)); do
  outputs+=("${score_dir}/${TASK_NAME}.shard_$(printf '%02d' "${index}").tsv")
  run_shard "${index}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  echo "At least one GPN-Star shard failed; see ${log_dir}" >&2
  exit 1
fi

merge_args=(
  --output "${score_dir}/${TASK_NAME}.tsv"
  --task-name "${TASK_NAME}"
)
for output in "${outputs[@]}"; do merge_args+=(--input "${output}"); done
micromamba -n llmbm run python \
  "${BENCHMARK_ROOT}/scripts/evaluate/compat/gpn_star_merge.py" \
  "${merge_args[@]}"
