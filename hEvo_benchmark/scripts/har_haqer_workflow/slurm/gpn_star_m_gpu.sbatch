#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
set -euo pipefail

HPC_HOME="${HPC_HOME:-${HOME}}"
REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")/../../../.." rev-parse --show-toplevel)}"
PYTHON="${LLMBM_PYTHON:-${HPC_HOME}/micromamba/envs/llmbm/bin/python}"
WORKFLOW="${REPO}/hEvo_benchmark/scripts/har_haqer_workflow/gpn_star_m_peak_workflow.py"
STAGE=${1:?Pass a workflow stage}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

exec "${PYTHON}" "${WORKFLOW}" "${STAGE}"
