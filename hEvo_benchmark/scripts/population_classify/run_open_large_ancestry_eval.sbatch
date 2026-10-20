#!/usr/bin/env bash
# Submit with: sbatch --dependency=afterok:<array_job_id> run_open_large_ancestry_eval.sbatch
#SBATCH --job-name=hevo-open-large-ancestry-eval
#SBATCH --partition=cpuPartition
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=open_large_eval_%j.out

set -euo pipefail
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:?Set DNAMODELS_BENCHMARK_ROOT to the external suite checkout}"
"${PYTHON:-python}" "${SCRIPT_DIR}/run_open_large_ancestry_eval.py"
