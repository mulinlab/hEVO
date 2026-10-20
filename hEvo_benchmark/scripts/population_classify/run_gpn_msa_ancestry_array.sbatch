#!/usr/bin/env bash
# Run 40 persistent GPU workers; each worker reuses the Zarr window cache and model.
# Smoke run: sbatch --array=0 --output="$RUN_ROOT/logs/smoke_%A_%a.out" this-file "$RUN_ROOT" smoke
# Full run:  sbatch --output="$RUN_ROOT/logs/array_%A_%a.out" this-file "$RUN_ROOT" full

#SBATCH --job-name=gpn-msa-ancestry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-39%40

set -euo pipefail

RUN_ROOT="${1:?RUN_ROOT is required}"
MODE="${2:-full}"
SCRIPT_DIR="${HEVO_POPULATION_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/scripts/population_classify}"
if [[ -z "${HEVO_POPULATION_SCRIPT_DIR:-}" && ! -f "${SCRIPT_DIR}/extract_gpn_msa_ancestry_emb.py" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}/hEvo_benchmark/scripts/population_classify"
fi
[[ -f "${SCRIPT_DIR}/extract_gpn_msa_ancestry_emb.py" ]] || { echo "Set HEVO_POPULATION_SCRIPT_DIR to scripts/population_classify" >&2; exit 2; }
PYTHON="${HEVO_LLBM_PYTHON:-python}"

TASK_TMPDIR="${RUN_ROOT}/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TASK_TMPDIR}"
export TMPDIR="${TASK_TMPDIR}"
export TEMP="${TASK_TMPDIR}"
export TMP="${TASK_TMPDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
trap 'rm -rf "${TASK_TMPDIR}"' EXIT

if [[ "${MODE}" == "smoke" ]]; then
  [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]] || exit 0
  "${PYTHON}" "${SCRIPT_DIR}/extract_gpn_msa_ancestry_emb.py" \
    --run-root "${RUN_ROOT}" \
    --worker-index 0 \
    --worker-count 1 \
    --max-samples 2 \
    --max-records 16 \
    --batch-size 4 \
    --output-dir "${RUN_ROOT}/smoke/embeddings" \
    --manifest-dir "${RUN_ROOT}/smoke/manifests"
elif [[ "${MODE}" == "full" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/extract_gpn_msa_ancestry_emb.py" \
    --run-root "${RUN_ROOT}" \
    --worker-index "${SLURM_ARRAY_TASK_ID}" \
    --worker-count 40 \
    --batch-size "${GPN_MSA_BATCH_SIZE:-16}" \
    --topk 500
else
  echo "Unsupported mode: ${MODE}" >&2
  exit 2
fi
