#!/bin/bash
set -euo pipefail

PROJECT="${HAPDIV_PROJECT_ROOT:-<source-project-root>}"
SCRIPT_DIR="${HAPDIV_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-<micromamba-root>/envs/llmbm/bin/python}"

N_SHARDS="${N_SHARDS:-45}"
CONCURRENCY="${CONCURRENCY:-80}"
WINDOW_SIZE="${WINDOW_SIZE:-100000}"
WINDOW_SUBSAMPLE_RATE="${WINDOW_SUBSAMPLE_RATE:-0.02}"
REFERENCE_SAMPLE_SIZE="${REFERENCE_SAMPLE_SIZE:-5000}"
TARGET_SAMPLE_SIZE="${TARGET_SAMPLE_SIZE:-}"
RANDOM_REPEATS="${RANDOM_REPEATS:-1}"
OUT_DIR="${OUT_DIR:-${PROJECT}/results/ukb_downsampling_vcf_empirical}"

TOTAL_TASKS=$((22 * N_SHARDS))

mkdir -p "${PROJECT}/logs/ukb_downsampling_empirical" "${OUT_DIR}"

export_args="ALL,PYTHON=${PYTHON},N_SHARDS=${N_SHARDS},WINDOW_SIZE=${WINDOW_SIZE},WINDOW_SUBSAMPLE_RATE=${WINDOW_SUBSAMPLE_RATE},REFERENCE_SAMPLE_SIZE=${REFERENCE_SAMPLE_SIZE},RANDOM_REPEATS=${RANDOM_REPEATS},OUT_DIR=${OUT_DIR}"
if [[ -n "${TARGET_SAMPLE_SIZE}" ]]; then
  export_args="${export_args},TARGET_SAMPLE_SIZE=${TARGET_SAMPLE_SIZE}"
fi

sbatch \
  --array="1-${TOTAL_TASKS}%${CONCURRENCY}" \
  --export="${export_args}" \
  "${SCRIPT_DIR}/run_ukb_downsampling_vcf_empirical_array.sbatch"
