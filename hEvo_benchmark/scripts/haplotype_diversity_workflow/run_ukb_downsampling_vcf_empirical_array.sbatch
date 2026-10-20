#!/bin/bash
#SBATCH --job-name=ukbdsemp
#SBATCH --partition=cpuPartition
#SBATCH --array=1-990%80
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ukb_downsampling_empirical/%x_%A_%a.out
#SBATCH --error=logs/ukb_downsampling_empirical/%x_%A_%a.err

set -euo pipefail

PROJECT="${HAPDIV_PROJECT_ROOT:-<source-project-root>}"
SCRIPT_DIR="${HAPDIV_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-<micromamba-root>/envs/llmbm/bin/python}"
OUT_DIR="${OUT_DIR:-${PROJECT}/results/ukb_downsampling_vcf_empirical}"
WINDOW_SIZE="${WINDOW_SIZE:-100000}"
WINDOW_SUBSAMPLE_RATE="${WINDOW_SUBSAMPLE_RATE:-0.02}"
REFERENCE_SAMPLE_SIZE="${REFERENCE_SAMPLE_SIZE:-5000}"
RANDOM_REPEATS="${RANDOM_REPEATS:-1}"
N_SHARDS="${N_SHARDS:-45}"

mkdir -p "${PROJECT}/logs/ukb_downsampling_empirical" "${OUT_DIR}"

task0=$((SLURM_ARRAY_TASK_ID - 1))
chrom=$((task0 / N_SHARDS + 1))
shard_index=$((task0 % N_SHARDS))

target_args=()
if [[ -n "${TARGET_SAMPLE_SIZE:-}" ]]; then
  target_args=(--target-sample-size "${TARGET_SAMPLE_SIZE}")
fi

"${PYTHON}" "${SCRIPT_DIR}/compute_ukb_downsampling_vcf_empirical.py" \
  --chrom "${chrom}" \
  --n-shards "${N_SHARDS}" \
  --shard-index "${shard_index}" \
  --window-size "${WINDOW_SIZE}" \
  --window-subsample-rate "${WINDOW_SUBSAMPLE_RATE}" \
  --reference-sample-size "${REFERENCE_SAMPLE_SIZE}" \
  ${target_args[@]+"${target_args[@]}"} \
  --random-repeats "${RANDOM_REPEATS}" \
  --out-dir "${OUT_DIR}"
