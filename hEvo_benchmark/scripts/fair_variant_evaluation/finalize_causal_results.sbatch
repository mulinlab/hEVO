#!/usr/bin/env bash
#SBATCH --job-name=fair-causal-final
#SBATCH --partition=cpuPartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Finalize only after every required native/model rerun succeeds. This job
# performs no sampling; C3-C5 are projections onto already frozen CSVs.
set -euo pipefail

: "${BENCHMARK_ROOT:?Set BENCHMARK_ROOT}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${SOURCE_DATASET:?Set SOURCE_DATASET to the pre-canonical CausalDB CSV}"
: "${STAR_ROOT:?Set STAR_ROOT}"
: "${MSA_ROOT:?Set MSA_ROOT}"
: "${GENOS_ROOT:?Set GENOS_ROOT}"
: "${EVO1_ROOT:?Set EVO1_ROOT}"
: "${EVO7_ROOT:?Set EVO7_ROOT}"
: "${HYENA_ROOT:?Set HYENA_ROOT}"

PYTHON="${BENCHMARK_ROOT}/scripts/fair_variant_evaluation"
CANONICAL="${DATA_ROOT}/canonical_refalt/CausalDB_disease_hg38_canonical_refalt.csv"
SCORES="${DATA_ROOT}/canonical_refalt_scores"
mkdir -p "${SCORES}/C0_full" "${RUN_ROOT}/subsets" "${RUN_ROOT}/final"
test -s "${CANONICAL}"

# Convert all native GPN-Star outputs against the canonical CausalDB CSV,
# never against the normalized variant_id TSV.
for spec in \
  "GPN-Star-M:${STAR_ROOT}/M/scores/GPN-Star-M/CausalDB_hg38_canonical_refalt.tsv" \
  "GPN-Star-P:${STAR_ROOT}/P/scores/GPN-Star-P/CausalDB_hg38_canonical_refalt.tsv" \
  "GPN-Star-V:${STAR_ROOT}/V/scores/GPN-Star-V/CausalDB_hg38_canonical_refalt.tsv"; do
  model="${spec%%:*}"
  input="${spec#*:}"
  test -s "${input}"
  micromamba -n llmbm run python "${PYTHON}/standardize_scores.py" \
    --dataset "${CANONICAL}" --input "${input}" --model "${model}" \
    --output "${SCORES}/C0_full/${model}.tsv"
done

GPN_MSA_H5="${MSA_ROOT}/gpn_msa_h5/CausalDB_hg38_canonical_refalt/GPN-MSA.official_msa_llr.hdf5"
test -s "${GPN_MSA_H5}"
micromamba -n llmbm run python "${PYTHON}/standardize_scores.py" \
  --dataset "${CANONICAL}" --input "${GPN_MSA_H5}" --model GPN-MSA \
  --output "${SCORES}/C0_full/GPN-MSA.tsv"

# Fresh canonical causal-LM outputs replace provisional sign-only transforms.
declare -A CAUSAL_LM_FILES=(
  [Genos-1.2B]="${GENOS_ROOT}/scores/C0_canonical/Genos-1.2B.tsv"
  [evo2_1b_base]="${EVO1_ROOT}/scores/C0_canonical/evo2_1b_base.tsv"
  [evo2_7b_base]="${EVO7_ROOT}/scores/C0_canonical/evo2_7b_base.tsv"
  [HyenaDNA_tiny_1k]="${HYENA_ROOT}/scores/C0_canonical/HyenaDNA_tiny_1k.tsv"
)
for model in "${!CAUSAL_LM_FILES[@]}"; do
  test -s "${CAUSAL_LM_FILES[$model]}"
  cp "${CAUSAL_LM_FILES[$model]}" "${SCORES}/C0_full/${model}.tsv"
done

declare -a CANONICAL_MODELS=(
  AIDO.DNA-300M Caduceus-PH Caduceus-PS DNABERT DNABERT2
  GENA-LM-base GENA-LM-large GENERator-1.2B GENERator-3B GPN-MSA
  GPN-promoter GROVER Genos-1.2B HyenaDNA_tiny_1k Mistral-DNA-138M
  Mistral-DNA-17M Mistral-DNA-422M NT_2.5b_1kG NT_2.5b_MS NT_500m_1kG
  NT_500m_MS NTv3-100M-pre NTv3-650M-pre NTv3-8M-pre SpeciesLM
  evo2_1b_base evo2_7b_base hEvo-base hEvo-large
)
declare -a EXTENSION_MODELS=(GPN-Star-M GPN-Star-P GPN-Star-V)
declare -a ALL_MODELS=("${CANONICAL_MODELS[@]}" "${EXTENSION_MODELS[@]}")

for arm in C3_common_maf_1to1 C4_common_highPP_maf_1to1 C5_common_highBeta_maf_1to1; do
  for model in "${ALL_MODELS[@]}"; do
    micromamba -n llmbm run python "${PYTHON}/subset_scores.py" \
      --dataset "${DATA_ROOT}/canonical_refalt/${arm}.csv" \
      --scores "${SCORES}/C0_full/${model}.tsv" \
      --output "${SCORES}/${arm}/${model}.tsv" \
      > "${RUN_ROOT}/subsets/${arm}.${model}.json"
  done
done

# Evaluate the entire predeclared PP/beta/MAF grid against the same complete
# score roster. This remains exploratory and cannot overwrite C0-C5 ranks.
GRID_MANIFEST="${DATA_ROOT}/filter_grid/filter_grid_manifest.json"
test -s "${GRID_MANIFEST}"
micromamba -n llmbm run python "${PYTHON}/evaluate_causal_filter_grid.py" \
  --manifest "${GRID_MANIFEST}" \
  --scores-root "${SCORES}/C0_full" \
  --output "${RUN_ROOT}/final/causal_filter_grid_rankings.tsv" \
  --expected-model-count 32 --bootstrap "${GRID_BOOTSTRAP_N:-500}" --seed 20260917

micromamba -n llmbm run python "${PYTHON}/write_causal_manifest.py" \
  --full "${CANONICAL}" \
  --source "${SOURCE_DATASET}" \
  --canonicalization-manifest "${DATA_ROOT}/canonical_refalt/canonicalization_manifest.json" \
  --arm "C3_common_maf_1to1:${DATA_ROOT}/canonical_refalt/C3_common_maf_1to1.csv" \
  --arm "C4_common_highPP_maf_1to1:${DATA_ROOT}/canonical_refalt/C4_common_highPP_maf_1to1.csv" \
  --arm "C5_common_highBeta_maf_1to1:${DATA_ROOT}/canonical_refalt/C5_common_highBeta_maf_1to1.csv" \
  --output "${RUN_ROOT}/final/causal_formal_manifest.json"

declare -a SCORE_ARGS=()
for model in "${ALL_MODELS[@]}"; do
  SCORE_ARGS+=(--score "C0_full:${model}:${SCORES}/C0_full/${model}.tsv")
  for arm in C3_common_maf_1to1 C4_common_highPP_maf_1to1 C5_common_highBeta_maf_1to1; do
    SCORE_ARGS+=(--score "${arm}:${model}:${SCORES}/${arm}/${model}.tsv")
  done
done
micromamba -n llmbm run python "${PYTHON}/summarize_fair_results.py" \
  --manifest "${RUN_ROOT}/final/causal_formal_manifest.json" \
  "${SCORE_ARGS[@]}" --output "${RUN_ROOT}/final/causal_rankings.tsv" \
  --bootstrap "${BOOTSTRAP_N:-1000}" --expected-model-count 32

for arm in C0_full C3_common_maf_1to1 C4_common_highPP_maf_1to1 C5_common_highBeta_maf_1to1; do
  for right in GPN-Star-M GPN-MSA Genos-1.2B; do
    micromamba -n llmbm run python "${PYTHON}/compare_paired_scores.py" \
      --arm "${arm}" --manifest "${RUN_ROOT}/final/causal_formal_manifest.json" \
      --left-model hEvo-large --left-score "${SCORES}/${arm}/hEvo-large.tsv" \
      --right-model "${right}" --right-score "${SCORES}/${arm}/${right}.tsv" \
      --output "${RUN_ROOT}/final/${arm}.hEvo-large_vs_${right}.json" \
      --bootstrap "${PAIRED_BOOTSTRAP_N:-2000}" --seed 20260914
  done
done

printf 'PASS\tcausal_finalizer\t%s\n' "${RUN_ROOT}/final/causal_rankings.tsv"
