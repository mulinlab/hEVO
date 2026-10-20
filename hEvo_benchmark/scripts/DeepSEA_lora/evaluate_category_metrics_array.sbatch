#!/bin/bash
#SBATCH --job-name=s21-openlarge-cat
#SBATCH --partition=gpu
#SBATCH --array=0-4%5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=category_s21_run24_%A_%a.out
#SBATCH --error=category_s21_run24_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

if [[ -n "${HEVO_BENCHMARK_ROOT:-}" ]]; then
  PROJECT=${HEVO_BENCHMARK_ROOT}
elif [[ -f "${SLURM_SUBMIT_DIR}/scripts/DeepSEA_lora/evaluate_category_metrics.py" ]]; then
  PROJECT=${SLURM_SUBMIT_DIR}
else
  PROJECT=${SLURM_SUBMIT_DIR}/hEvo_benchmark
fi
RUN_DIR=${DEEPSEA_RUN_DIR:?Set DEEPSEA_RUN_DIR to the DeepSEA run results directory}
DATASET_ROOT=${DEEPSEA_DATASET_ROOT:-${DNAMODELS_BENCHMARK_ROOT:-$HOME/DNAModelsBenchmark}/datasets/DeepSEA}
METADATA=${DEEPSEA_METADATA:-${DATASET_ROOT}/deepsea_metadata.tsv}
TORCHRUN=${LLMBM_TORCHRUN:-$HOME/micromamba/envs/llmbm/bin/torchrun}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=5
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export PROJECT RUN_DIR DATASET_ROOT METADATA TORCHRUN SHARD_INDEX NUM_SHARDS MASTER_ADDR MASTER_PORT
export BENCHMARK_ROOT=${PROJECT}

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -lc '
  exec "${TORCHRUN}" \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node=4 \
    --node_rank="${SLURM_NODEID}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT}/scripts/DeepSEA_lora/evaluate_category_metrics.py" \
    --run_dir "${RUN_DIR}" \
    --config "${PROJECT}/configs/models.yaml" \
    --dataset_root "${DATASET_ROOT}" \
    --metadata "${METADATA}" \
    --batch_size 16 \
    --shard_index "${SHARD_INDEX}" \
    --num_shards "${NUM_SHARDS}" \
    --output "${RUN_DIR}/category_metrics.json"
'
