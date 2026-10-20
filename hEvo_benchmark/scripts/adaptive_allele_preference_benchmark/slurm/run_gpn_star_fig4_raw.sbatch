#!/usr/bin/env bash
#SBATCH --job-name=fig4f-gpn-raw
#SBATCH --partition=gpu
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/runs/fig4_gpn_raw_scores_20260919/slurm/%A_%a.out
#SBATCH --error=/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/runs/fig4_gpn_raw_scores_20260919/slurm/%A_%a.err

set -euo pipefail

PYTHON=/mnt/hpc/home/xuqinyi/micromamba/envs/llmbm/bin/python
CODE=/mnt/hpc/home/xuqinyi/hEvo_GitLab/hEvo_benchmark/scripts/adaptive_allele_preference_benchmark/fig4_gpn_raw_stage/gpn_star_raw_logits_adapter.py
INPUT=/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/runs/fig4_gpn_raw_scores_20260919/inputs/fig4f_gpn_star.tsv
ROOT=/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/runs/fig4_gpn_raw_scores_20260919

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    model_id=GPN-Star-V
    model_dir=/mnt/hpc/home/xuqinyi/gpn_star_assets/models/GPN-Star-V
    revision=0c949f132d35619a3eb188b402848c998a3313ae
    msa_store=(/mnt/hpc/home/xuqinyi/gpn_star_assets/msa/multiz100way-pigz/100/all.zarr)
    msa_species=(100)
    msa_dataset=songlab/multiz100way-pigz
    msa_revision=6a9d42a35e7debbba845979dea6064f14d5cb3f9
    ;;
  1)
    model_id=GPN-Star-M
    model_dir=/mnt/hpc/home/xuqinyi/gpn_star_assets/models/GPN-Star-M
    revision=0d8f5164fba8e8108200c160370b86652bfb6244
    msa_store=(/mnt/hpc/home/xuqinyi/gpn_star_assets/msa/hg38_cactus447way/243/all.zarr /mnt/hpc/home/xuqinyi/gpn_star_assets/msa/hg38_cactus447way/204/all.zarr)
    msa_species=(243 204)
    msa_dataset=songlab/hg38_cactus447way
    msa_revision=f523f0467bc143b62fb5b3b9a91c510e8dc2a7f6
    ;;
  2)
    model_id=GPN-Star-P
    model_dir=/mnt/hpc/home/xuqinyi/gpn_star_assets/models/GPN-Star-P
    revision=16773016130e826cd3d72e91ed312a3cb27ba2b5
    msa_store=(/mnt/hpc/home/xuqinyi/gpn_star_assets/msa/hg38_cactus447way/243/all.zarr)
    msa_species=(243)
    msa_dataset=songlab/hg38_cactus447way
    msa_revision=f523f0467bc143b62fb5b3b9a91c510e8dc2a7f6
    ;;
  *) exit 2 ;;
esac

mkdir -p "${ROOT}/scores" "${ROOT}/manifests" "${ROOT}/slurm"
args=(
  --model-id "${model_id}"
  --model-dir "${model_dir}"
  --model-revision "${revision}"
  --input "${INPUT}"
  --output "${ROOT}/scores/${model_id}.tsv"
  --manifest "${ROOT}/manifests/${model_id}.json"
  --msa-dataset "${msa_dataset}"
  --msa-revision "${msa_revision}"
  --window-bp 1024
  --batch-size 8
  --device cuda
)
for store in "${msa_store[@]}"; do args+=(--msa-store "${store}"); done
args+=(--msa-species "${msa_species[@]}")

exec "${PYTHON}" "${CODE}" "${args[@]}"
