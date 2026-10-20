#!/usr/bin/env bash
#SBATCH --job-name=deepsea-gpnm-summary
#SBATCH --partition=cpuPartition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
HPC_HOME="${HPC_HOME:-${HOME}}"
REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
PYTHON="${LLMBM_PYTHON:-${HPC_HOME}/micromamba/envs/llmbm/bin/python}"
CODE="${REPO}/hEvo_benchmark/scripts/DeepSEA_lora/aggregate_gpn_star_m_deepsea.py"
RESULT_ROOT="${RESULT_ROOT:-${DEEPSEA_GPN_STAR_RUN_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/deepsea_gpn_star_m_v1}/training/deepsea_1epoch_stride4_40gpu}"
exec "$PYTHON" "$CODE" --root "$RESULT_ROOT" --seeds 42
