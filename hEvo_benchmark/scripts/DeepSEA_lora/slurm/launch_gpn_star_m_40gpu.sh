#!/usr/bin/env bash
set -euo pipefail

REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")/../../../.." rev-parse --show-toplevel)}"
HPC_HOME="${HPC_HOME:-${HOME}}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
ROOT="${DEEPSEA_GPN_STAR_RUN_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/deepsea_gpn_star_m_v1}"
SCRIPTS="$REPO/hEvo_benchmark/scripts/DeepSEA_lora/slurm"
SMOKE_JOB=${1:?usage: launch_gpn_star_m_40gpu.sh SMOKE_JOB_ID}
RUN_TOKEN="$(date +%Y%m%d_%H%M%S)_$$"
RESULT_RUN_NAME="deepsea_1epoch_stride4_40gpu_${RUN_TOKEN}"
RESULT_RUN_DIR="$ROOT/training/$RESULT_RUN_NAME"
MASTER_PORT=$((20000 + $$ % 30000))

mkdir -p "$ROOT/slurm" "$RESULT_RUN_DIR"
job_ids=()
for group in 0 1 2 3 4; do
  job_id=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" \
    --export="ALL,RUN_TOKEN=${RUN_TOKEN},RUN_GROUP_INDEX=${group},MASTER_PORT=${MASTER_PORT},RESULT_RUN_DIR=${RESULT_RUN_DIR}" \
    "$SCRIPTS/gpn_star_m_train_fullnode_group.sbatch")
  job_ids+=("$job_id")
done

dependency=$(IFS=:; echo "${job_ids[*]}")
summary_job=$(sbatch --parsable --dependency="afterok:${dependency}" \
  --export="ALL,RESULT_ROOT=${RESULT_RUN_DIR}" \
  "$SCRIPTS/gpn_star_m_aggregate_deepsea.sbatch")

printf 'RUN_TOKEN=%s\nRUN_DIR=%s\nTRAIN_JOBS=%s\nSUMMARY_JOB=%s\n' \
  "$RUN_TOKEN" "$RESULT_RUN_DIR" "${job_ids[*]}" "$summary_job"
