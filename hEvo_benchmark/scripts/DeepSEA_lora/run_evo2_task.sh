#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${SLURM_PROCID:-${SLURM_LOCALID}}"
echo "[Evo2 task] rank=${SLURM_PROCID:-unset} local_id=${SLURM_LOCALID:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
exec <micromamba-root>/bin/micromamba run --clean-env -n llmbm env \
  "RANK=${SLURM_PROCID:-0}" \
  "WORLD_SIZE=${SLURM_NTASKS:-1}" \
  "LOCAL_RANK=0" \
  "MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR is required}" \
  "MASTER_PORT=${MASTER_PORT:?MASTER_PORT is required}" \
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  python "$@"
