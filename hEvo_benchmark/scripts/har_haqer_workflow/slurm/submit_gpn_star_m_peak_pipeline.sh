#!/usr/bin/env bash
set -euo pipefail

HPC_HOME="${HPC_HOME:-${HOME}}"
REPO="${HEVO_REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")/../../../.." rev-parse --show-toplevel)}"
DNAMODELS_BENCHMARK_ROOT="${DNAMODELS_BENCHMARK_ROOT:-${HPC_HOME}/DNAModelsBenchmark}"
ROOT="${GPN_STAR_PEAK_RUN:-${DNAMODELS_BENCHMARK_ROOT}/runs/gpn_star_m_har_haqer_v1}"
SLURM_DIR="${REPO}/hEvo_benchmark/scripts/har_haqer_workflow/slurm"
mkdir -p "${ROOT}/slurm" "${ROOT}/reports"

submit() {
    local script=$1
    shift
    local stage=$1
    shift
    sbatch --parsable "$@" "${script}" "${stage}" | cut -d ';' -f 1
}

GPU="${SLURM_DIR}/gpn_star_m_gpu.sbatch"
CPU="${SLURM_DIR}/gpn_star_m_cpu.sbatch"
EXPORT="ALL,GPN_STAR_PEAK_RUN=${ROOT}"
EXTRACT_ARRAY='0-39%40'
SCAN_ARRAY='0-47%40'

preflight=$(submit "${GPU}" preflight --export="${EXPORT}" --job-name=gpnM-preflight)
prepare=$(submit "${CPU}" prepare --export="${EXPORT}" --dependency="afterok:${preflight}" --job-name=gpnM-prepare)
extract=$(submit "${GPU}" extract-window --export="${EXPORT}" --dependency="afterok:${prepare}" --job-name=gpnM-extract --array="${EXTRACT_ARRAY}")
merge=$(submit "${CPU}" merge-window --export="${EXPORT}" --dependency="afterok:${extract}" --job-name=gpnM-merge)
train=$(submit "${GPU}" train --export="${EXPORT}" --dependency="afterok:${merge}" --job-name=gpnM-train)
prepare_scan=$(submit "${CPU}" prepare-scan --export="${EXPORT}" --dependency="afterok:${train}" --job-name=gpnM-scan-prep)
scan=$(submit "${GPU}" scan --export="${EXPORT}" --dependency="afterok:${prepare_scan}" --job-name=gpnM-scan --array="${SCAN_ARRAY}")
evaluate=$(submit "${CPU}" evaluate --export="${EXPORT}" --dependency="afterok:${scan}" --job-name=gpnM-evaluate)

{
    printf 'stage\tjob_id\tdependency\n'
    printf 'preflight\t%s\t-\n' "${preflight}"
    printf 'prepare\t%s\tafterok:%s\n' "${prepare}" "${preflight}"
    printf 'extract_window\t%s\tafterok:%s; array=%s\n' "${extract}" "${prepare}" "${EXTRACT_ARRAY}"
    printf 'merge_window\t%s\tafterok:%s\n' "${merge}" "${extract}"
    printf 'train\t%s\tafterok:%s\n' "${train}" "${merge}"
    printf 'prepare_scan\t%s\tafterok:%s\n' "${prepare_scan}" "${train}"
    printf 'scan\t%s\tafterok:%s; array=%s\n' "${scan}" "${prepare_scan}" "${SCAN_ARRAY}"
    printf 'evaluate\t%s\tafterok:%s\n' "${evaluate}" "${scan}"
} > "${ROOT}/slurm/pipeline_jobs.tsv"

printf 'GPN-Star-M HAQER/HAR pipeline submitted.\nRun root: %s\n' "${ROOT}"
cat "${ROOT}/slurm/pipeline_jobs.tsv"
