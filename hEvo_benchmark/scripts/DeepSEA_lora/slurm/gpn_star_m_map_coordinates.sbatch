#!/usr/bin/env bash
#SBATCH --job-name=ds-gpnm-map
#SBATCH --partition=cpuPartition
#SBATCH --nodes=6
#SBATCH --nodelist=cpu02,cpu04,cpu11,cpu13,cpu14,cpu16
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
set -euo pipefail
HPC_HOME="${HPC_HOME:-${HOME}}"
REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
PYTHON="${LLMBM_PYTHON:-${HPC_HOME}/micromamba/envs/llmbm/bin/python}"
CODE="${REPO}/hEvo_benchmark/scripts/DeepSEA_lora/map_deepsea_to_hg38.py"
DATA="${DEEPSEA_DATA_ROOT:-${HPC_HOME}/hEvo/dataset/deepsea}"
ROOT="${GPN_DEEPSEA_COORDINATE_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/deepsea_gpn_star_m_v1/coordinates}"
export PYTHON CODE DATA ROOT

srun --ntasks=12 --ntasks-per-node=2 bash -c '
  exec "$PYTHON" "$CODE" map --dataset-root "$DATA" --output-dir "$ROOT" \
    --index "$ROOT/hg38.sr.mmi" --shard-index "$SLURM_PROCID" --shard-count 12
'
