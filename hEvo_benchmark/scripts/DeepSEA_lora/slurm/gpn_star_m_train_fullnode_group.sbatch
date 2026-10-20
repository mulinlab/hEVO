#!/usr/bin/env bash
#SBATCH --job-name=deepsea-gpnm-ddp40
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6-00:00:00
set -euo pipefail
HPC_HOME="${HPC_HOME:-${HOME}}"
REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
PYTHON="${LLMBM_PYTHON:-${HPC_HOME}/micromamba/envs/llmbm/bin/python}"
CODE="${REPO}/hEvo_benchmark/scripts/DeepSEA_lora/run_deepsea_lora.py"
ROOT="${DEEPSEA_GPN_STAR_RUN_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/deepsea_gpn_star_m_v1}"
CONFIG="${REPO}/hEvo_benchmark/configs/deepsea_gpn_star_m.yaml"
DATA="${DEEPSEA_DATA_ROOT:-${HPC_HOME}/hEvo/dataset/deepsea}"
GPN_STAR_ASSETS_ROOT="${GPN_STAR_ASSETS_ROOT:-${HPC_HOME}/gpn_star_assets}"
SEED=${SEED:-42}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-4400000}
TRAIN_ROW_STRIDE=${TRAIN_ROW_STRIDE:-1}
EPOCHS=${EPOCHS:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-13}
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-520}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULE=${LR_SCHEDULE:-constant}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-5000}
: "${RUN_TOKEN:?RUN_TOKEN must be set by launch_gpn_star_m_40gpu.sh}"
: "${RUN_GROUP_INDEX:?RUN_GROUP_INDEX must be between 0 and 4}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${RESULT_RUN_DIR:?RESULT_RUN_DIR is required}"
if (( RUN_GROUP_INDEX < 0 || RUN_GROUP_INDEX > 4 )); then
  echo "Invalid RUN_GROUP_INDEX=$RUN_GROUP_INDEX" >&2
  exit 2
fi

COORD_DIR="$ROOT/training/.rendezvous/$RUN_TOKEN"
OUT="$RESULT_RUN_DIR/seed_${SEED}"
mkdir -p "$COORD_DIR" "$OUT"
export PYTHON CODE ROOT CONFIG DATA OUT SEED RUN_TOKEN RUN_GROUP_INDEX MASTER_PORT
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

HOSTS_FILE="$COORD_DIR/hosts.${RUN_GROUP_INDEX}"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$HOSTS_FILE.tmp.${SLURM_JOB_ID}"
mv "$HOSTS_FILE.tmp.${SLURM_JOB_ID}" "$HOSTS_FILE"

deadline=$((SECONDS + 3600))
while :; do
  ready=1
  for group in 0 1 2 3 4; do
    [[ -s "$COORD_DIR/hosts.${group}" ]] || ready=0
  done
  if (( ready )); then break; fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for all five full-node allocations in $COORD_DIR" >&2
    exit 1
  fi
  sleep 5
done

if (( RUN_GROUP_INDEX == 0 )); then
  head -n 1 "$COORD_DIR/hosts.0" > "$COORD_DIR/master_host.tmp"
  mv "$COORD_DIR/master_host.tmp" "$COORD_DIR/master_host"
fi
while [[ ! -s "$COORD_DIR/master_host" ]]; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for rendezvous leader host" >&2
    exit 1
  fi
  sleep 2
done
MASTER_ADDR=$(<"$COORD_DIR/master_host")
export MASTER_ADDR

srun --kill-on-bad-exit=1 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=16 bash -c '
  exec "$PYTHON" -m torch.distributed.run \
    --nnodes=10 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$RUN_TOKEN" \
    "$CODE" --config "$CONFIG" --model_name GPN-Star-M --model_scope all \
    --dataset_root "$DATA" --output_root "$ROOT/training" --output_dir "$OUT" \
    --gpn_coordinate_root "$ROOT/coordinates" \
    --gpn_msa_243 "${GPN_STAR_ASSETS_ROOT}/msa/hg38_cactus447way/243/all.zarr" \
    --gpn_msa_204 "${GPN_STAR_ASSETS_ROOT}/msa/hg38_cactus447way/204/all.zarr" \
    --stage custom --epochs "$EPOCHS" --train_samples "$TRAIN_SAMPLES" --train_row_stride "$TRAIN_ROW_STRIDE" --seed "$SEED" \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --target_global_batch_size "$TARGET_GLOBAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" --weight_decay "$WEIGHT_DECAY" --warmup_ratio "$WARMUP_RATIO" \
    --lr_schedule "$LR_SCHEDULE" --mixed_precision bf16 --gradient_checkpointing \
    --num_workers 0 --eval_subset_size 4096 --eval_every_steps "$EVAL_EVERY_STEPS" --logging_steps 100
'
