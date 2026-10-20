#!/usr/bin/env bash
#SBATCH --job-name=aap_unified
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-28%6
set -euo pipefail

ROOT=${AAP_ROOT:?set AAP_ROOT to the benchmark root}
MAMBA_EXE=${MAMBA_EXE:?set MAMBA_EXE to micromamba}
export PYTHONPATH="$ROOT/ModelGenerator:${PYTHONPATH:-}"
cd "$ROOT"

MODEL_IDS=(
  HyenaDNA_tiny_1k NT_500m_1kG NT_500m_MS NT_2.5b_1kG NT_2.5b_MS
  NTv3-8M-pre NTv3-100M-pre NTv3-650M-pre GPN-MSA GPN-promoter
  Genos-1.2B SpeciesLM AIDO.DNA-300M DNABERT DNABERT2
  evo2_1b_base evo2_7b_base hEvo-base hEvo-large GENA-LM-base
  GENA-LM-large GROVER Mistral-DNA-17M Mistral-DNA-138M Mistral-DNA-422M
  Caduceus-PH Caduceus-PS Generator-1.2B Generator-3B
)
KINDS=(
  causal masked masked masked masked masked masked masked masked masked
  causal masked masked masked masked causal causal masked masked masked
  masked masked causal causal causal masked masked causal causal
)
MODEL_DIRS=(
  models/LongSafari/hyenadna-tiny-1k-seqlen-hf-causal models/InstaDeepAI/NT-500m-1000g models/InstaDeepAI/NT-500m-MS
  models/InstaDeepAI/NT-2.5b-1000g models/InstaDeepAI/NT-2.5b-MS models/InstaDeepAI/NTv3_8M_pre
  models/InstaDeepAI/NTv3_100M_pre models/InstaDeepAI/NTv3_650M_pre models/gpn-msa-sapiens models/gpn-animal-promoter
  models/Genos-1.2B models/specieslm-metazoa-upstream-k6 models/AIDO.DNA-300M models/DNABERT-S models/DNABERT-2-117M
  models/Evo2/evo2_1b_base/evo2_1b_base.pt models/Evo2/evo2_7b_base/evo2_7b_base.pt models/hEvo/hEvo-base models/hEvo/hEvo-large
  models/GENA-LM/GENA-LM-base models/GENA-LM/GENA-LM-large models/GROVER models/Mistral-DNA/Mistral-DNA-v1-17M-hg38
  models/Mistral-DNA/Mistral-DNA-v1-138M-hg38 models/Mistral-DNA/Mistral-DNA-v1-422M-hg38
  models/Caduceus/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 models/Caduceus/caduceus-ps_seqlen-131k_d_model-256_n_layer-16
  models/GENERator-v2/GENERator-v2-eukaryote-1.2b-base models/GENERator-v2/GENERator-v2-eukaryote-3b-base
)
IDX=$SLURM_ARRAY_TASK_ID
MODEL_ID=${MODEL_IDS[$IDX]}
KIND=${KINDS[$IDX]}
MODEL_DIR=${MODEL_DIRS[$IDX]}

OUT=runs/adaptive_allele_preference_benchmark_v1/expanded_unified_rerun_20260905/$MODEL_ID/locus_scores.tsv
mkdir -p "$(dirname "$OUT")"
set +e
"$MAMBA_EXE" -n llmbm run python hEvo_benchmark/scripts/adaptive_allele_preference_benchmark/05_score_generic_model.py --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" --kind "$KIND" --loci runs/adaptive_allele_preference_benchmark_v1/expanded_loci_master_20260905/benchmark_loci_expanded.tsv --reference-fasta datasets/hg38.fa --output "$OUT" --window-bp 1024 --device cuda > "$OUT.log" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "$MODEL_ID failed with exit code $RC; see $OUT.log" >&2
fi
exit "$RC"
