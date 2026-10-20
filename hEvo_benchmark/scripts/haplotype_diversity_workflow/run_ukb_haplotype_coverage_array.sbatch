#!/bin/bash
#SBATCH --job-name=ukbcov
#SBATCH --partition=cpuPartition
#SBATCH --array=1-990%160
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/haplotype_coverage/%x_%A_%a.out
#SBATCH --error=logs/haplotype_coverage/%x_%A_%a.err

set -euo pipefail

PROJECT="${HAPDIV_PROJECT_ROOT:-<source-project-root>}"
SCRIPT_DIR="${HAPDIV_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-<micromamba-root>/envs/llmbm/bin/python}"
OUT_DIR="${OUT_DIR:-${PROJECT}/results/ukb_haplotype_coverage_vcf}"
WINDOW_SIZE="${WINDOW_SIZE:-100000}"
WINDOW_SUBSAMPLE_RATE="${WINDOW_SUBSAMPLE_RATE:-0.02}"
MAX_SAMPLES_UKB_FULL="${MAX_SAMPLES_UKB_FULL:-2000}"
MAX_SAMPLES_TARGET="${MAX_SAMPLES_TARGET:-2000}"
N_SHARDS="${N_SHARDS:-45}"

mkdir -p "${PROJECT}/logs/haplotype_coverage" "${OUT_DIR}"

task0=$((SLURM_ARRAY_TASK_ID - 1))
chrom=$((task0 / N_SHARDS + 1))
shard_index=$((task0 % N_SHARDS))

"${PYTHON}" "${SCRIPT_DIR}/compute_ukb_haplotype_coverage.py" \
  --chrom "${chrom}" \
  --n-shards "${N_SHARDS}" \
  --shard-index "${shard_index}" \
  --window-size "${WINDOW_SIZE}" \
  --window-subsample-rate "${WINDOW_SUBSAMPLE_RATE}" \
  --max-samples-ukb-full "${MAX_SAMPLES_UKB_FULL}" \
  --max-samples-target "${MAX_SAMPLES_TARGET}" \
  --out-dir "${OUT_DIR}"
