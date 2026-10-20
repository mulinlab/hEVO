#!/usr/bin/env bash
# Slurm MaxArraySize is 1001: submit 0-999 and 0-499 with ARRAY_OFFSET=1000.
# One Slurm array task processes one 1KG sample on one allocated GPU.
#SBATCH --job-name=hevo-open-large-ancestry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=open_large_%A_%a.out

set -euo pipefail

ROOT="${DNAMODELS_BENCHMARK_ROOT:?Set DNAMODELS_BENCHMARK_ROOT to the external suite checkout}"
SCRIPT_DIR="${ROOT}/scripts/populaton_classify"
DATA_DIR="${ROOT}/datasets/1kg_popu_classification/jsonl_2000v_1500s_511bp"
BENCHMARK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${HEVO_OPEN_LARGE_MODEL:-${BENCHMARK_ROOT}/models/hEvo-open-large}"
OUTPUT_DIR="${ROOT}/results/1kg_popu_classification/2000v_1500s_511bp/hEvo-open-large_popu_emb"
PYTHON="${PYTHON:-python}"

mapfile -t FILES < <(find "${DATA_DIR}" -maxdepth 1 -type f -name '*.jsonl' | sort)
[[ ${#FILES[@]} -eq 1500 ]] || { echo "Expected 1500 input samples, found ${#FILES[@]}" >&2; exit 1; }
FILE_INDEX=$((SLURM_ARRAY_TASK_ID + ${ARRAY_OFFSET:-0}))
(( FILE_INDEX >= 0 && FILE_INDEX < ${#FILES[@]} )) || { echo "Invalid input index: ${FILE_INDEX}" >&2; exit 1; }
INPUT_FILE="${FILES[${FILE_INDEX}]}"
mkdir -p "${OUTPUT_DIR}"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
"${PYTHON}" "${SCRIPT_DIR}/extract_single_sample_emb.py" \
  --input_file "${INPUT_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name hEvo-open-large \
  --batch_size 16 \
  --max_length 512 \
  --aggregation multi_stats \
  --topk 500

"${PYTHON}" - "${OUTPUT_DIR}/$(basename "${INPUT_FILE}" .jsonl).npy" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
x = np.load(path, allow_pickle=False)
assert x.shape == (3844,), (path, x.shape)
assert np.isfinite(x).all(), path
print(f'Validated {path}: shape={x.shape}, norm={np.linalg.norm(x):.6g}')
PY
