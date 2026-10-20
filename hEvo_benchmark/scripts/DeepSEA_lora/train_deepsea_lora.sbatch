#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/DeepSEA_lora/%x_%j.out
#SBATCH --error=logs/DeepSEA_lora/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT="<source-project-root>"
MAMBA="<micromamba-root>/bin/micromamba"
SCRIPT=${PROJECT}/scripts/DeepSEA_lora/run_deepsea_lora.py
CONFIG=${PROJECT}/config.yaml
DATASET=${PROJECT}/datasets/DeepSEA
RESULTS=${PROJECT}/results/DeepSEA_lora

: "${MODEL_NAME:?MODEL_NAME must be exported by sbatch}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be exported by sbatch}"

TRAIN_BATCH=1
GRAD_ACCUM=32
MAX_LENGTH=${MAX_LENGTH_OVERRIDE:-1024}
ATTENTION_BACKEND=${ATTENTION_BACKEND_OVERRIDE:-eager}
MIXED_PRECISION=${MIXED_PRECISION_OVERRIDE:-fp16}
AUTO_BATCH=${AUTO_BATCH_OVERRIDE:-1}

case "${MODEL_NAME}" in
  hEvo-base) TRAIN_BATCH=8; GRAD_ACCUM=4 ;;
  hEvo-large) TRAIN_BATCH=2; GRAD_ACCUM=16 ;;
  NT_500m_1kG) TRAIN_BATCH=2; GRAD_ACCUM=16 ;;
  NT_2.5b_1kG) TRAIN_BATCH=1; GRAD_ACCUM=32 ;;
esac

if [ "${AUTO_BATCH}" = "0" ]; then
  # Reproduce the pre-FA2 eager baseline used by the cancelled hEvo runs.
  TRAIN_BATCH=16
  GRAD_ACCUM=2
fi

AUTO_BATCH_ARG=()
if [ "${AUTO_BATCH}" = "1" ]; then
  AUTO_BATCH_ARG+=(--auto_batch)
fi

TRAIN_BATCH=${TRAIN_BATCH_OVERRIDE:-${TRAIN_BATCH}}
GRAD_ACCUM=${GRAD_ACCUM_OVERRIDE:-${GRAD_ACCUM}}
STAGE=${STAGE:-screen}
NUM_WORKERS=${NUM_WORKERS_OVERRIDE:-2}
EVAL_BATCH=${EVAL_BATCH_OVERRIDE:-8}
mkdir -p "${OUTPUT_DIR}" "${PROJECT}/.sbatch/DeepSEA_lora"

"${MAMBA}" -n llmbm run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${SCRIPT}" \
  --config "${CONFIG}" \
  --model_name "${MODEL_NAME}" \
  --dataset_root "${DATASET}" \
  --output_root "${RESULTS}" \
  --output_dir "${OUTPUT_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --attention_backend "${ATTENTION_BACKEND}" \
    --mixed_precision "${MIXED_PRECISION}" \
  --per_device_train_batch_size "${TRAIN_BATCH}" \
  --per_device_eval_batch_size "${EVAL_BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --gradient_checkpointing \
    ${AUTO_BATCH_ARG[@]-} \
  --stage "${STAGE}" \
  --logging_steps 100 \
  --eval_every_steps 5000 \
  --eval_subset_size 4096 \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor 2 \
  --target_global_batch_size 128 \
  --seed 42 \
  --train_samples 4400000 \
  --epochs 3
