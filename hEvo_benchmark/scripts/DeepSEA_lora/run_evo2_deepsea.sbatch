#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
# One full Evo2 replica per GPU; DDP aggregates four data shards.
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=21-00:00:00
#SBATCH --exclusive
#SBATCH --output=logs/DeepSEA_lora/%x_%j.out
#SBATCH --error=logs/DeepSEA_lora/%x_%j.err

set -euo pipefail
PROJECT="<source-project-root>"
MAMBA="<micromamba-root>/bin/micromamba"
MODEL_NAME=${MODEL_NAME:?MODEL_NAME must be exported}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be exported}
SMOKE_ONLY=${SMOKE_ONLY:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-128}
SMOKE_ACCUM=${SMOKE_ACCUM:-8}

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29501}

mkdir -p "${OUTPUT_DIR}" "${PROJECT}/.sbatch/DeepSEA_lora"
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  srun --gpu-bind=closest --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 \
    bash <source-project-root>/scripts/DeepSEA_lora/run_evo2_task.sh \
    \
    "${PROJECT}/scripts/DeepSEA_lora/run_evo2_deepsea.py" \
    --model_name "${MODEL_NAME}" \
    --config "${PROJECT}/config.yaml" \
    --dataset_root "${PROJECT}/datasets/DeepSEA" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length 1024 \
    --train_samples 128 \
    --epochs 1 \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${SMOKE_ACCUM}" \
    --learning_rate 2e-5 \
    --eval_every_steps 8 \
    --eval_subset_size 128 \
    --logging_steps 1 \
    --num_workers 0 \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --seed 42 --smoke_only
else
  srun --gpu-bind=closest --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 \
    bash <source-project-root>/scripts/DeepSEA_lora/run_evo2_task.sh \
    \
    "${PROJECT}/scripts/DeepSEA_lora/run_evo2_deepsea.py" \
    --model_name "${MODEL_NAME}" \
    --config "${PROJECT}/config.yaml" \
    --dataset_root "${PROJECT}/datasets/DeepSEA" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length 1024 \
    --train_samples 4400000 \
    --epochs 1 \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --eval_every_steps 5000 \
    --eval_subset_size 4096 \
    --logging_steps 100 \
    --num_workers 0 \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --seed 42
fi
