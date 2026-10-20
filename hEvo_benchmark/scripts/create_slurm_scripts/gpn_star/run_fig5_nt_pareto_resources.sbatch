#!/usr/bin/env bash
#SBATCH --job-name=fig5-gpn-nt-eff
#SBATCH --partition=gpu
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
set -euo pipefail

HPC_HOME="${HPC_HOME:-${HOME}}"
PROJECT="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")/../../../.." rev-parse --show-toplevel)}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
GPN_STAR_ASSETS_ROOT="${GPN_STAR_ASSETS_ROOT:-${HPC_HOME}/gpn_star_assets}"
PYTHON="${LLMBM_PYTHON:-${HPC_HOME}/micromamba/envs/llmbm/bin/python}"
CODE=${PROJECT}/hEvo_benchmark/scripts/evaluate/runtime/profile_gpn_star_nt_pareto_resources.py
TEST=${PROJECT}/hEvo_benchmark/datasets/NT_benchmark_v2/_combined/test.csv
TRAIN=${PROJECT}/hEvo_benchmark/datasets/NT_benchmark_v2/_combined/train.csv
S17=${PROJECT}/hEvo_benchmark/tables/S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv
OUTPUT=${PROJECT}/hEvo_benchmark/results/Probe_NT_Pareto
RUN_ROOT="${GPN_STAR_RESOURCE_RUN_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/fig5_gpn_star_nt_resource_profiles}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    model_id=GPN-Star-V
    model_dir="${GPN_STAR_ASSETS_ROOT}/models/GPN-Star-V"
    revision=0c949f132d35619a3eb188b402848c998a3313ae
    msa_store=("${GPN_STAR_ASSETS_ROOT}/msa/multiz100way-pigz/100/all.zarr")
    msa_species=(100)
    msa_dataset=songlab/multiz100way-pigz
    msa_revision=6a9d42a35e7debbba845979dea6064f14d5cb3f9
    ;;
  1)
    model_id=GPN-Star-M
    model_dir="${GPN_STAR_ASSETS_ROOT}/models/GPN-Star-M"
    revision=0d8f5164fba8e8108200c160370b86652bfb6244
    msa_store=("${GPN_STAR_ASSETS_ROOT}/msa/hg38_cactus447way/243/all.zarr" "${GPN_STAR_ASSETS_ROOT}/msa/hg38_cactus447way/204/all.zarr")
    msa_species=(243 204)
    msa_dataset=songlab/hg38_cactus447way
    msa_revision=f523f0467bc143b62fb5b3b9a91c510e8dc2a7f6
    ;;
  2)
    model_id=GPN-Star-P
    model_dir="${GPN_STAR_ASSETS_ROOT}/models/GPN-Star-P"
    revision=16773016130e826cd3d72e91ed312a3cb27ba2b5
    msa_store=("${GPN_STAR_ASSETS_ROOT}/msa/hg38_cactus447way/243/all.zarr")
    msa_species=(243)
    msa_dataset=songlab/hg38_cactus447way
    msa_revision=f523f0467bc143b62fb5b3b9a91c510e8dc2a7f6
    ;;
  *) exit 2 ;;
esac

mkdir -p "${RUN_ROOT}/manifests"
test -d "${model_dir}"
for store in "${msa_store[@]}"; do test -d "${store}"; done

args=(
  --model-id "${model_id}"
  --model-dir "${model_dir}"
  --model-revision "${revision}"
  --msa-dataset "${msa_dataset}"
  --msa-revision "${msa_revision}"
  --train "${TRAIN}"
  --test "${TEST}"
  --s17-table "${S17}"
  --output-root "${OUTPUT}"
  --manifest "${RUN_ROOT}/manifests/${model_id}.json"
  --batch-size 8
  --window-size 1024
  --center-window-size 1000
)
for store in "${msa_store[@]}"; do args+=(--msa-store "${store}"); done
for species in "${msa_species[@]}"; do args+=(--msa-species "${species}"); done

exec "${PYTHON}" "${CODE}" "${args[@]}"
