#!/bin/bash
#SBATCH --job-name=hapdiv_agg
#SBATCH --partition=cpuPartition
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/haplotype_diversity/%x_%j.out
#SBATCH --error=logs/haplotype_diversity/%x_%j.err

set -euo pipefail

PROJECT="${HAPDIV_PROJECT_ROOT:-<source-project-root>}"
SCRIPT_DIR="${HAPDIV_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-<micromamba-root>/envs/llmbm/bin/python}"
OUT_DIR=${PROJECT}/results/haplotype_diversity

mkdir -p "${PROJECT}/logs/haplotype_diversity" "${OUT_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/aggregate_haplotype_diversity.py" \
  --in-dir ${OUT_DIR} \
  --out "${OUT_DIR}/fig1_haplotype_diversity_summary.tsv"
