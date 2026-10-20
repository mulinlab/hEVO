#!/usr/bin/env bash
#SBATCH --job-name=fair-variant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# One immutable dataset × model evaluation.  All large files are external to
# the repository.  Submit an array with --array=0-9%10 to honor the 10-GPU
# concurrency cap, or submit individual jobs for a smaller pilot.
set -euo pipefail

: "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT}"
: "${DATASET_FILE:?Set DATASET_FILE}"
: "${DATASET_KIND:?Set DATASET_KIND to causal or rare}"
: "${ARM_ID:?Set ARM_ID, e.g. C0 or R0}"
: "${MODEL_ID:?Set MODEL_ID}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${RUN_ROOT:?Set external RUN_ROOT}"

test -f "${DATASET_FILE}"
test -e "${MODEL_PATH}"
mkdir -p "${RUN_ROOT}/h5/${ARM_ID}" "${RUN_ROOT}/evaluator/${ARM_ID}/${MODEL_ID}" \
  "${RUN_ROOT}/logs/${ARM_ID}/${MODEL_ID}" "${RUN_ROOT}/scores/${ARM_ID}"

# Use the named HPC environment for both evaluation and score
# standardization.  Activation is used here because it preserves stdout/stderr
# on the cluster's micromamba installation.
eval "$(micromamba shell hook -s bash)"
micromamba activate llmbm

if [[ "${DATASET_KIND}" == "causal" ]]; then
  ref_col=2
  alt_col=3
  label_col=4
else
  ref_col=1
  alt_col=2
  label_col=3
fi

python \
  "${BENCHMARK_ROOT}/scripts/evaluate/zeroshot_paried_seq_input.py" \
  --model_paths "${MODEL_PATH}" \
  --model_name "${MODEL_ID}" \
  --dataset_files "${DATASET_FILE}" \
  --dataset_name "${ARM_ID}" \
  --h5_path "${RUN_ROOT}/h5" \
  --out_dir "${RUN_ROOT}/evaluator/${ARM_ID}/${MODEL_ID}" \
  --log_dir "${RUN_ROOT}/logs/${ARM_ID}/${MODEL_ID}" \
  --sep comma \
  --idx_col 0 --ref_col "${ref_col}" --alt_col "${alt_col}" --label_col "${label_col}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --precompute_batch_size "${PRECOMPUTE_BATCH_SIZE:-32}" \
  --max_length "${MAX_LENGTH:-1024}" --center_index "${CENTER_INDEX:-511}" --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --bootstrap 0 --score_mode neg_llr --recompute

python \
  "${BENCHMARK_ROOT}/scripts/fair_variant_evaluation/standardize_scores.py" \
  --dataset "${DATASET_FILE}" \
  --input "${RUN_ROOT}/h5/${ARM_ID}/${MODEL_ID}.hdf5" \
  --model "${MODEL_ID}" \
  --output "${RUN_ROOT}/scores/${ARM_ID}/${MODEL_ID}.tsv" \
  --variant-index "${CENTER_INDEX:-511}"

printf 'PASS\t%s\t%s\t%s\n' "${ARM_ID}" "${MODEL_ID}" "${RUN_ROOT}/scores/${ARM_ID}/${MODEL_ID}.tsv"
