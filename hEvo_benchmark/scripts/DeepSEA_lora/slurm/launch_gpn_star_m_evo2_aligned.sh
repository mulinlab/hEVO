#!/usr/bin/env bash
set -euo pipefail

REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")/../../../.." rev-parse --show-toplevel)}"
HPC_HOME="${HPC_HOME:-${HOME}}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
ROOT="${DEEPSEA_GPN_STAR_RUN_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/deepsea_gpn_star_m_v1}"
SCRIPTS="$REPO/hEvo_benchmark/scripts/DeepSEA_lora/slurm"
RUN_TOKEN="$(date +%Y%m%d_%H%M%S)_$$"
RESULT_RUN_DIR="$ROOT/training/deepsea_1epoch_full_40gpu_evo2params_${RUN_TOKEN}"
MASTER_PORT=$((20000 + $$ % 30000))

mkdir -p "$ROOT/slurm" "$ROOT/training/.rendezvous" "$RESULT_RUN_DIR/slurm"
job_ids=()
for group in 0 1 2 3 4; do
  export_args="ALL,RUN_TOKEN=${RUN_TOKEN},RUN_GROUP_INDEX=${group},MASTER_PORT=${MASTER_PORT},RESULT_RUN_DIR=${RESULT_RUN_DIR},SEED=42,TRAIN_SAMPLES=4400000,TRAIN_ROW_STRIDE=1,EPOCHS=1,GRADIENT_ACCUMULATION_STEPS=13,TARGET_GLOBAL_BATCH_SIZE=520,LEARNING_RATE=2e-5,WEIGHT_DECAY=0.01,WARMUP_RATIO=0.03,LR_SCHEDULE=constant,EVAL_EVERY_STEPS=5000"
  job_id=$(sbatch --parsable \
    --export="$export_args" \
    --output="$RESULT_RUN_DIR/slurm/train_%j.out" \
    --error="$RESULT_RUN_DIR/slurm/train_%j.err" \
    "$SCRIPTS/gpn_star_m_train_fullnode_group.sbatch")
  job_ids+=("$job_id")
done

dependency=$(IFS=:; echo "${job_ids[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:${dependency}" \
  --export="ALL,RESULT_ROOT=${RESULT_RUN_DIR}" \
  --output="$RESULT_RUN_DIR/slurm/summary_%j.out" \
  --error="$RESULT_RUN_DIR/slurm/summary_%j.err" \
  "$SCRIPTS/gpn_star_m_aggregate_deepsea.sbatch")

RUN_TOKEN="$RUN_TOKEN" RESULT_RUN_DIR="$RESULT_RUN_DIR" MASTER_PORT="$MASTER_PORT" \
  TRAIN_JOBS="${job_ids[*]}" SUMMARY_JOB="$summary_job" \
  "${LLMBM_PYTHON:-${HOME}/micromamba/envs/llmbm/bin/python}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_RUN_DIR"])
payload = {
    "run_token": os.environ["RUN_TOKEN"],
    "result_run_dir": str(root),
    "training_jobs": os.environ["TRAIN_JOBS"].split(),
    "summary_job": os.environ["SUMMARY_JOB"],
    "model": "GPN-Star-M",
    "seed": 42,
    "train_samples": 4_400_000,
    "train_row_stride": 1,
    "epochs": 1,
    "world_size": 40,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 13,
    "effective_global_batch_size": 520,
    "precision": "bf16",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_schedule": "constant_after_warmup",
    "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
    "max_length": 1000,
    "eval_every_steps": 5000,
}
(root / "submission_manifest.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
PY

printf 'RUN_TOKEN=%s\nRUN_DIR=%s\nTRAIN_JOBS=%s\nSUMMARY_JOB=%s\n' \
  "$RUN_TOKEN" "$RESULT_RUN_DIR" "${job_ids[*]}" "$summary_job"
