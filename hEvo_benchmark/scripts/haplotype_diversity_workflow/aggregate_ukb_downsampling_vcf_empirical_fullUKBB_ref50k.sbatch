#!/bin/bash
#SBATCH --job-name=ukbdsemp_aggr
#SBATCH --partition=cpuPartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ukb_downsampling_empirical/%x_%j.out
#SBATCH --error=logs/ukb_downsampling_empirical/%x_%j.err

set -euo pipefail

PROJECT="${HAPDIV_PROJECT_ROOT:-<source-project-root>}"
SCRIPT_DIR="${HAPDIV_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="<micromamba-root>/envs/llmbm/bin/python"
RESULT_DIR=${PROJECT}/results/ukb_downsampling_vcf_empirical_fullUKBB_ref50k
RESULT_PREFIX=${RESULT_DIR}/ukb_downsampling_empirical

cd "${PROJECT}"

"${PYTHON}" "${SCRIPT_DIR}/aggregate_ukb_downsampling_vcf_empirical.py" \
  --in-dir "${RESULT_DIR}" \
  --out-prefix "${RESULT_PREFIX}"

MPLBACKEND=Agg "${PYTHON}" "${SCRIPT_DIR}/plot_ukb_haplotype_coverage.py" \
  --result-prefix "${RESULT_PREFIX}" \
  --out-dir "${HAPDIV_PLOT_OUTPUT:-${PROJECT}/results/haplotype_diversity_plots}"
