#!/usr/bin/env bash
# Submit as:
# sbatch run_population_embedding_array.sh DATA_DIR MODEL_PATH MODEL_NAME OUTPUT_DIR

#SBATCH --job-name=popu_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1499

set -euo pipefail

DATA_DIR="${1:?DATA_DIR is required}"
MODEL_PATH="${2:?MODEL_PATH is required}"
MODEL_NAME="${3:?MODEL_NAME is required}"
OUTPUT_DIR="${4:?OUTPUT_DIR is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t FILES < <(find "${DATA_DIR}" -maxdepth 1 -type f -name "*.jsonl" | sort)
INPUT_FILE="${FILES[${SLURM_ARRAY_TASK_ID}]}"

micromamba run -n llmbm python "${SCRIPT_DIR}/extract_single_sample_emb.py" \
  --input_file "${INPUT_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --max_length "${MAX_LENGTH:-511}" \
  --aggregation "${AGGREGATION:-multi_stats}" \
  --topk "${TOPK:-500}"

