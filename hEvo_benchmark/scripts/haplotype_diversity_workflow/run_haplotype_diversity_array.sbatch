#!/bin/bash
#SBATCH --job-name=hapdiv
#SBATCH --partition=cpuPartition
#SBATCH --array=1-220%80
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/haplotype_diversity/%x_%A_%a.out
#SBATCH --error=logs/haplotype_diversity/%x_%A_%a.err

set -euo pipefail

PROJECT="${HAPDIV_PROJECT_ROOT:-<source-project-root>}"
SCRIPT_DIR="${HAPDIV_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-<micromamba-root>/envs/llmbm/bin/python}"
OUT_DIR="${OUT_DIR:-${PROJECT}/results/haplotype_diversity}"
WINDOW_SIZE="${WINDOW_SIZE:-100000}"
WINDOW_SUBSAMPLE_RATE="${WINDOW_SUBSAMPLE_RATE:-0.02}"
MAX_SAMPLES_PER_COHORT="${MAX_SAMPLES_PER_COHORT:-100}"
RAREFACTION_REPEATS="${RAREFACTION_REPEATS:-10}"

mkdir -p "${PROJECT}/logs/haplotype_diversity" "${OUT_DIR}"

N_SHARDS="${N_SHARDS:-10}"
task0=$((SLURM_ARRAY_TASK_ID - 1))
chrom=$((task0 / N_SHARDS + 1))
shard_index=$((task0 % N_SHARDS))

"${PYTHON}" "${SCRIPT_DIR}/compute_window_haplotype_diversity.py" \
  --chrom ${chrom} \
  --n-shards ${N_SHARDS} \
  --shard-index ${shard_index} \
  --window-size ${WINDOW_SIZE} \
  --window-subsample-rate ${WINDOW_SUBSAMPLE_RATE} \
  --max-samples-per-cohort ${MAX_SAMPLES_PER_COHORT} \
  --rarefaction-repeats ${RAREFACTION_REPEATS} \
  --out-dir "${OUT_DIR}"
