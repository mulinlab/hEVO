#!/usr/bin/env bash
#SBATCH --job-name=ds-gpnm-index
#SBATCH --partition=cpuPartition
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
HPC_HOME="${HPC_HOME:-${HOME}}"
REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
PYTHON="${LLMBM_PYTHON:-${HPC_HOME}/micromamba/envs/llmbm/bin/python}"
CODE="${REPO}/hEvo_benchmark/scripts/DeepSEA_lora/map_deepsea_to_hg38.py"
ROOT="${GPN_DEEPSEA_COORDINATE_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/deepsea_gpn_star_m_v1/coordinates}"
mkdir -p "$ROOT" "${ROOT%/coordinates}/slurm"
exec "$PYTHON" "$CODE" build_index --index "$ROOT/hg38.sr.mmi"
