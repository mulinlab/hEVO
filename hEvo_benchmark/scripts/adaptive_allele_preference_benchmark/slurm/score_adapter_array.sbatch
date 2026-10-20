#!/usr/bin/env bash
#SBATCH --job-name=aap_adapter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%4
set -euo pipefail

ROOT=${AAP_ROOT:?set AAP_ROOT to the benchmark root}
MAMBA_EXE=${MAMBA_EXE:?set MAMBA_EXE to micromamba}
export PYTHONPATH="$ROOT/ModelGenerator:${PYTHONPATH:-}"
cd "$ROOT"

case "${SLURM_ARRAY_TASK_ID}" in
  0) MODEL_ID=SpeciesLM; KIND=masked; MODEL_DIR=models/specieslm-metazoa-upstream-k6 ;;
  1) MODEL_ID=AIDO.DNA-300M; KIND=masked; MODEL_DIR=models/AIDO.DNA-300M ;;
  2) MODEL_ID=evo2_1b_base; KIND=causal; MODEL_DIR=models/Evo2/evo2_1b_base/evo2_1b_base.pt ;;
  3) MODEL_ID=evo2_7b_base; KIND=causal; MODEL_DIR=models/Evo2/evo2_7b_base/evo2_7b_base.pt ;;
  *) echo "unsupported array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT=runs/adaptive_allele_preference_benchmark_v1/expanded_adapter_20260905/${MODEL_ID}/locus_scores.tsv
mkdir -p "$(dirname "$OUT")"

"$MAMBA_EXE" -n llmbm run python \
  hEvo_benchmark/scripts/adaptive_allele_preference_benchmark/05_score_generic_model.py \
  --model-dir "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --kind "$KIND" \
  --loci runs/adaptive_allele_preference_benchmark_v1/expanded_loci_master_20260905/benchmark_loci_expanded.tsv \
  --reference-fasta datasets/hg38.fa \
  --output "$OUT" \
