#!/usr/bin/env bash
set -euo pipefail

REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
HPC_HOME="${HPC_HOME:-${HOME}}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
RUN="${DEEPSEA_GPN_STAR_RUN_ROOT:-${DNAMODELS_BENCHMARK_ROOT}/runs/deepsea_gpn_star_m_v1}"
SCRIPTS="$REPO/hEvo_benchmark/scripts/DeepSEA_lora/slurm"
SLURM_USER="${HEVO_SLURM_USER:-$(id -un)}"
mkdir -p "$RUN/slurm" "$RUN/coordinates" "$RUN/training/smoke" "$RUN/training"

index_job=$(sbatch --parsable "$SCRIPTS/gpn_star_m_build_index.sbatch")
prepare_job=$(sbatch --parsable "$SCRIPTS/gpn_star_m_prepare_coordinates.sbatch")
map_job=$(sbatch --parsable --dependency="afterok:${index_job}:${prepare_job}" \
  "$SCRIPTS/gpn_star_m_map_coordinates.sbatch")
qc_job=$(sbatch --parsable --dependency="afterok:${map_job}" \
  "$SCRIPTS/gpn_star_m_summarize_coordinates.sbatch")
smoke_job=$(sbatch --parsable --dependency="afterok:${qc_job}" \
  "$SCRIPTS/gpn_star_m_smoke.sbatch")
training_submission=$(bash "$SCRIPTS/launch_gpn_star_m_40gpu.sh" "$smoke_job")

cat <<EOF
Submitted GPN-Star-M DeepSEA workflow
  hg38 minimap2 index: ${index_job}
  row-offset / coordinate arrays: ${prepare_job}
  distributed mapping job (12 shards on 6 CPU nodes): ${map_job}
  mapping quality gate: ${qc_job}
  1-GPU MSA/LoRA smoke test: ${smoke_job}
  one seed distributed across 5 coordinated 8-GPU full-node jobs (40 GPUs):
${training_submission}

Monitor with: squeue -u ${SLURM_USER}
Run root: ${RUN}
EOF
