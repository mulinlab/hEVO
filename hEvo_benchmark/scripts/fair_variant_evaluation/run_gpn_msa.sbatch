#!/usr/bin/env bash
#SBATCH --job-name=fair-gpn-msa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Coordinate-aware GPN-MSA evaluation for a frozen CausalDB CSV.  All paths
# are supplied by the caller so the recipe is portable and traceable.
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the external <source-project-root> root}"
: "${DATASET_FILE:?Set DATASET_FILE to the frozen paired CSV}"
: "${MODEL_DIR:?Set MODEL_DIR to the GPN-MSA checkpoint}"
: "${MSA_ZIP:?Set MSA_ZIP to the multiz100way Zarr v2 archive}"
: "${ZARR2_PATH:?Set ZARR2_PATH to the zarr<3 support tree}"
: "${RUN_ROOT:?Set RUN_ROOT to an external run directory}"
: "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT to the benchmark root}"

test -f "${DATASET_FILE}"
test -d "${MODEL_DIR}"
test -f "${MSA_ZIP}"
test -d "${ZARR2_PATH}/zarr"
export PYTHONPATH="${ZARR2_PATH}:${PROJECT_ROOT}/gpn:${PYTHONPATH:-}"
export WANDB_MODE=disabled
mkdir -p "${RUN_ROOT}/gpn_msa_h5" "${RUN_ROOT}/gpn_msa_results" "${RUN_ROOT}/gpn_msa_logs"

DATASET_NAME="${DATASET_NAME:-CausalDB_hg38_canonical_refalt}"

/mnt/hpc/home/xuqinyi/micromamba/envs/llmbm/bin/python \
  "${PROJECT_ROOT}/scripts/zeroshot_paried_seq_input.py" \
  --model_paths "${MODEL_DIR}" --model_name GPN-MSA \
  --dataset_files "${DATASET_FILE}" --dataset_name "${DATASET_NAME}" \
  --sep ',' --h5_path "${RUN_ROOT}/gpn_msa_h5" \
  --batch_size "${BATCH_SIZE:-32}" --precompute_batch_size "${PRECOMPUTE_BATCH_SIZE:-32}" --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
  --ref_col 2 --alt_col 3 --label_col 4 --idx_col 0 \
  --chrom_col "${CHROM_COL:-11}" --pos_col "${POS_COL:-12}" --max_length 1024 \
  --gpn_msa_path "zip:///::${MSA_ZIP}" --gpn_window_size 128 \
  --recompute --bootstrap 0 \
  --out_dir "${RUN_ROOT}/gpn_msa_results" --log_dir "${RUN_ROOT}/gpn_msa_logs"
