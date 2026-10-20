#!/usr/bin/bash
# ============================================================
# Shapeit5 UKBB WGS Phasing Pipeline (Merged)
# ============================================================
# Usage:
#   bash shapeit5_phasing_pipeline.sh <chr> [step]
#   Example: bash shapeit5_phasing_pipeline.sh 22
#         bash shapeit5_phasing_pipeline.sh 22 step05_common_phasing
# ============================================================

set -euo pipefail

# ----------------------------------------------------------
# 0. Argument parsing and configuration
# ----------------------------------------------------------
usage() {
    cat <<'USAGE'
Usage: shapeit5_phasing_pipeline.sh <chr> [start_step]

Required environment:
  WORK_DIR       Pipeline working directory
  SHAPEIT5_DIR   Directory containing phase_common_static, phase_rare_static,
                 ligate_static, and switch_static

Optional environment:
  UKBB_PGEN, GMAP_FILE, PED_FILE, BLOCK_DIR, RESULT_DIR
  PLINK2, BCFTOOLS, TABIX, PARALLEL, GLIMPSE2_CHUNK
  MICROMAMBA, MICROMAMBA_ROOT, PYTHON_ENV (default: llmbm)

Steps: step01 step02 step03 step04 step05 step06 step07 step08 step09
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

CHR=${1:-22}
START_STEP=${2:-step01}

# Project and input paths. All defaults are relative to WORK_DIR so that the
# workflow is portable across HPC accounts and scratch locations.
WORK_DIR="${WORK_DIR:-}"
if [[ -z "${WORK_DIR}" ]]; then
    echo "[ERROR] WORK_DIR must be set; see --help" >&2
    exit 2
fi

# Input data paths
UKBB_PGEN="${UKBB_PGEN:-${WORK_DIR}/input/chr${CHR}/chr${CHR}}"
GMAP_FILE="${GMAP_FILE:-${WORK_DIR}/gmap_file/chr${CHR}.b38.gmap.gz}"
PED_FILE="${PED_FILE:-${WORK_DIR}/ukb.related_column3.fam}"

# Output subdirectories
BLOCK_DIR="${BLOCK_DIR:-${WORK_DIR}/chr_chunk_file_self}"
RESULT_DIR="${RESULT_DIR:-${WORK_DIR}/plink_version_chr${CHR}_result}"

# Software paths. Tool locations are explicit environment overrides; bare
# command names are resolved from PATH by validate_config.
PLINK2="${PLINK2:-plink2}"
BCFTOOLS="${BCFTOOLS:-bcftools}"
TABIX="${TABIX:-tabix}"
PARALLEL="${PARALLEL:-parallel}"
GLIMPSE2_CHUNK="${GLIMPSE2_CHUNK:-}"
SHAPEIT5_DIR="${SHAPEIT5_DIR:-}"
PHASE_COMMON="${PHASE_COMMON:-${SHAPEIT5_DIR}/phase_common_static}"
PHASE_RARE="${PHASE_RARE:-${SHAPEIT5_DIR}/phase_rare_static}"
LIGATE="${LIGATE:-${SHAPEIT5_DIR}/ligate_static}"
SWITCH="${SWITCH:-${SHAPEIT5_DIR}/switch_static}"

# Python is always launched through the requested HPC micromamba environment.
MICROMAMBA="${MICROMAMBA:-<micromamba-root>/bin/micromamba}"
MICROMAMBA_ROOT="${MICROMAMBA_ROOT:-<micromamba-root>}"
PYTHON_ENV="${PYTHON_ENV:-llmbm}"

# Parallel configuration
PARALLEL_JOBS_PGEN="${PARALLEL_JOBS_PGEN:-30}"
PARALLEL_JOBS_BCF="${PARALLEL_JOBS_BCF:-50}"
PARALLEL_JOBS_PHASE="${PARALLEL_JOBS_PHASE:-20}"
PARALLEL_JOBS_SER="${PARALLEL_JOBS_SER:-8}"

# Filtering parameters
HWE_THRESHOLD="${HWE_THRESHOLD:-1e-30}"
GENO_THRESHOLD="${GENO_THRESHOLD:-0.1}"
MIN_AC_FILTER="${MIN_AC_FILTER:-10}"   # final filter AC>=10
COMMON_MAF="${COMMON_MAF:-0.01}"

# ----------------------------------------------------------
# Environment and input validation
# ----------------------------------------------------------
require_file() {
    if [[ ! -s "$1" ]]; then
        echo "[ERROR] Required file is missing or empty: $1" >&2
        exit 2
    fi
}

require_executable() {
    local candidate="$1"
    if [[ "$candidate" == */* ]]; then
        if [[ ! -x "$candidate" ]]; then
            echo "[ERROR] Required executable is not executable: $candidate" >&2
            exit 2
        fi
    elif ! command -v "$candidate" >/dev/null 2>&1; then
        echo "[ERROR] Required executable is not on PATH: $candidate" >&2
        exit 2
    fi
}

run_python() {
    "$MICROMAMBA" -r "$MICROMAMBA_ROOT" -n "$PYTHON_ENV" run python "$@"
}

validate_config() {
    require_executable "$MICROMAMBA"
    if [[ -z "${SHAPEIT5_DIR}" ]]; then
        echo "[ERROR] SHAPEIT5_DIR must be set; see --help" >&2
        exit 2
    fi
    if [[ -z "${GLIMPSE2_CHUNK}" ]]; then
        echo "[ERROR] GLIMPSE2_CHUNK must be set; see --help" >&2
        exit 2
    fi
    require_file "${UKBB_PGEN}.pgen"
    require_file "${UKBB_PGEN}.pvar"
    require_file "${UKBB_PGEN}.psam"
    require_file "$GMAP_FILE"
    require_file "$PED_FILE"
    require_executable "$PLINK2"
    require_executable "$BCFTOOLS"
    require_executable "$TABIX"
    require_executable "$PARALLEL"
    require_executable "$GLIMPSE2_CHUNK"
    require_executable "$PHASE_COMMON"
    require_executable "$PHASE_RARE"
    require_executable "$LIGATE"
    require_executable "$SWITCH"
    run_python -c "import pandas"
}

# ----------------------------------------------------------
# Check if the previous step succeeded
# ----------------------------------------------------------
check_step() {
    local step_name=$1
    local exit_code=$2
    if [ $exit_code -ne 0 ]; then
        echo "[ERROR] ${step_name} failed with exit code ${exit_code}" >&2
        exit $exit_code
    fi
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ${step_name} DONE"
}

# ----------------------------------------------------------
# Step 01: Create chromosome block list
# ----------------------------------------------------------
step01_create_block_list() {
    echo "========================================"
    echo "  Step 01: Create Chromosome Block List"
    echo "========================================"
    mkdir -p ${BLOCK_DIR}

    local block_file="${BLOCK_DIR}/chr${CHR}_1Mb_step.list"

    if [ -f "${block_file}" ] && [ -s "${block_file}" ]; then
        echo "[SKIP] Block list already exists: ${block_file}"
        return 0
    fi

    local length=$(zcat ${GMAP_FILE} | awk 'END {print int($1 / 1000000) + 1}')
    echo "  Chromosome ${CHR} length: ${length} Mb"

    local start=0
    > ${block_file}
    while [ ${start} -lt ${length} ]; do
        local end=$((start + 1))
        echo -e "${start}\t${end}" >> ${block_file}
        start=${end}
    done

    echo "  Block list created: ${block_file} ($(wc -l < ${block_file}) blocks)"
    echo ""
}

# ----------------------------------------------------------
# Step 02: pgen to VCF (chunked conversion + filtering + merging)
# ----------------------------------------------------------
step02_pgen_to_vcf() {
    echo "========================================"
    echo "  Step 02: Convert pgen to VCF"
    echo "========================================"
    mkdir -p ${RESULT_DIR}

    local chunk_list="${BLOCK_DIR}/chr${CHR}_1Mb_step.list"
    local merge_vcf="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.vcf.gz"

    if [ -f "${merge_vcf}" ]; then
        echo "[SKIP] Merged VCF already exists: ${merge_vcf}"
        return 0
    fi

    # --- Step 02a: pgen to VCF by 1Mb blocks with filtering ---
    echo "  [02a] Converting pgen to VCF by 1Mb blocks..."
    date +"%Y-%m-%d-%H:%M"

    cat ${chunk_list} | ${PARALLEL} -j ${PARALLEL_JOBS_PGEN} --colsep '\t' \
        ${PLINK2} \
        --pgen ${UKBB_PGEN}.pgen \
        --psam ${UKBB_PGEN}.psam \
        --pvar ${UKBB_PGEN}.pvar \
        --chr ${CHR} \
        --from-mb {1} \
        --to-mb {2} \
        --hwe ${HWE_THRESHOLD} \
        --geno ${GENO_THRESHOLD} \
        --min-ac 1 \
        --threads 8 \
        --export vcf bgz \
        --out ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}_{2}Mb.filtered

    check_step "step02a_pgen_to_vcf" $?

    # --- Step 02b: Split multi-allelic variants ---
    echo "  [02b] Normalizing multi-allelic variants..."
    date +"%Y-%m-%d-%H:%M"

    cat ${chunk_list} | ${PARALLEL} -j ${PARALLEL_JOBS_BCF} --colsep '\t' \
        ${BCFTOOLS} norm \
        -m -any ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}_{2}Mb.filtered.vcf.gz \
        -Oz -o ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}_{2}Mb.norm.vcf.gz \
        --threads 8

    check_step "step02b_bcftools_norm" $?

    # --- Step 02c: Merge all chunk VCFs ---
    echo "  [02c] Concatenating chunk VCFs..."
    date +"%Y-%m-%d-%H:%M"

    local concat_list="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.list"
    > ${concat_list}

    while read start end; do
        local chunk_file="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.${start}_${end}Mb.norm.vcf.gz"
        if [ -f "${chunk_file}" ]; then
            echo "${chunk_file}" >> ${concat_list}
        fi
    done < ${chunk_list}

    ${BCFTOOLS} concat \
        -n \
        -f ${concat_list} \
        -Oz -o ${merge_vcf} \
        --threads 32

    check_step "step02c_bcftools_concat" $?
    echo ""
}

# ----------------------------------------------------------
# Step 03: Build index + GLIMPSE2 chunking
# ----------------------------------------------------------
step03_glimpse_chunk() {
    echo "========================================"
    echo "  Step 03: GLIMPSE2 Chunking"
    echo "========================================"

    local merge_vcf="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.vcf.gz"
    local chunk_file="${BLOCK_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.chunk_4cm.list"

    # --- Step 03a: Build tabix index ---
    echo "  [03a] Building tabix index..."
    ${TABIX} -p vcf --threads 8 ${merge_vcf}
    check_step "step03a_tabix" $?

    # --- Step 03b: GLIMPSE2 chunk ---
    if [ -f "${chunk_file}" ] && [ -s "${chunk_file}" ]; then
        echo "[SKIP] Chunk file already exists: ${chunk_file}"
        return 0
    fi

    echo "  [03b] Running GLIMPSE2_chunk..."
    date +"%Y-%m-%d-%H:%M"

    ${GLIMPSE2_CHUNK} \
        --input ${merge_vcf} \
        --output ${chunk_file} \
        --window-cm 4 \
        --window-mb 4 \
        --window-count 30000 \
        --buffer-cm 0.5 \
        --buffer-mb 0.5 \
        --buffer-count 3000 \
        --sequential \
        --map ${GMAP_FILE} \
        --region ${CHR} \
        --threads 12

    check_step "step03b_glimpse2_chunk" $?
    echo ""
}

# ----------------------------------------------------------
# Step 04: Convert chunk format (rare + common)
# ----------------------------------------------------------
step04_chunk_format() {
    echo "========================================"
    echo "  Step 04: Convert Chunk Format"
    echo "========================================"

    local chunk_file="${BLOCK_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.chunk_4cm.list"
    local rare_chunk="${chunk_file}_rare"
    local common_chunk="${chunk_file}_common"

    # --- Step 04a: Create rare chunk format ---
    echo "  [04a] Creating rare chunk format..."
    if [ ! -f "${rare_chunk}" ] || [ ! -s "${rare_chunk}" ]; then
        cat ${chunk_file} | cut -f 1,3,4,8 | sed 's/:/\t/g' | sed 's/-/\t/g' > ${rare_chunk}
        echo "  Rare chunk: ${rare_chunk} ($(wc -l < ${rare_chunk}) blocks)"
    else
        echo "  [SKIP] Rare chunk exists: ${rare_chunk}"
    fi

    # --- Step 04b: Create common chunk format (merge every 3 blocks) ---
    echo "  [04b] Creating common chunk format..."
    if [ ! -f "${common_chunk}" ] || [ ! -s "${common_chunk}" ]; then
        run_python -c "
import pandas as pd
import sys

def rare_to_common_style(input_file):
    rare_file = pd.read_table(input_file, header=None, dtype=int)
    rare_file.columns = ['chunk_index', 'common_chr', 'common_start', 'common_end',
                         'rare_chr', 'rare_start', 'rare_end', 'variantCount']

    common_result = []
    rare_result = []
    common_current_start = None
    rare_current_start = None
    index_group_com = []
    index_group_rare = []

    for index, pos in rare_file.iterrows():
        com_start, com_end = pos['common_start'], pos['common_end']
        rare_start, rare_end = pos['rare_start'], pos['rare_end']
        index_group_com.append(index)
        index_group_rare.append(index)

        # Common merge: every 3 blocks
        if common_current_start is None:
            common_current_start = com_start
        else:
            if len(index_group_com) % 3 == 0:
                common_result.append([common_current_start, com_end])
                common_current_start = None
                index_group_com = []
            elif index == rare_file.index[-1]:
                common_result[-1][1] = com_end

        # Rare merge: every 3 blocks
        if rare_current_start is None:
            rare_current_start = rare_start
        else:
            if len(index_group_rare) % 3 == 0:
                rare_result.append([rare_current_start, rare_end])
                rare_current_start = None
                index_group_rare = []
            elif index == rare_file.index[-1]:
                rare_result[-1][1] = rare_end

    com_df = pd.DataFrame(common_result, columns=['com_merge_start', 'com_merge_end'])
    rare_df = pd.DataFrame(rare_result, columns=['rare_merge_start', 'rare_merge_end'])
    final = pd.concat([
        rare_file[['chunk_index', 'common_chr']],
        com_df,
        rare_file[['rare_chr']],
        rare_df
    ], axis=1).dropna().astype(int)
    final.to_csv(sys.argv[2], index=False, header=False, sep='\t')

rare_to_common_style('${rare_chunk}', '${common_chunk}')
"
        echo "  Common chunk: ${common_chunk} ($(wc -l < ${common_chunk}) blocks)"
    else
        echo "  [SKIP] Common chunk exists: ${common_chunk}"
    fi

    echo ""
}

# ----------------------------------------------------------
# Step 05: Common Variant Phasing + index building
# ----------------------------------------------------------
step05_common_phasing() {
    echo "========================================"
    echo "  Step 05: Common Variant Phasing"
    echo "========================================"

    local merge_vcf="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.vcf.gz"
    local common_chunk="${BLOCK_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.chunk_4cm.list_common"

    # --- Step 05a: SHAPEIT5 phase_common ---
    echo "  [05a] Running SHAPEIT5_phase_common..."
    date +"%Y-%m-%d-%H:%M"

    cat ${common_chunk} | ${PARALLEL} -j ${PARALLEL_JOBS_PHASE} --colsep '\t' \
        ${PHASE_COMMON} \
        --input ${merge_vcf} \
        --map ${GMAP_FILE} \
        --output ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}.{3}_{4}.common.vcf.gz \
        --log ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}.{3}_{4}.common.log \
        --filter-maf ${COMMON_MAF} \
        --region ${CHR}:{3}-{4} \
        --pedigree ${PED_FILE} \
        --thread 80

    check_step "step05a_phase_common" $?

    # --- Step 05b: Build tabix index (for each chunk) ---
    echo "  [05b] Building tabix index for common chunks..."
    date +"%Y-%m-%d-%H:%M"

    cat ${common_chunk} | ${PARALLEL} -j ${PARALLEL_JOBS_PHASE} --colsep '\t' \
        ${TABIX} -p vcf --threads 8 \
        ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}.{3}_{4}.common.vcf.gz

    check_step "step05b_tabix_common" $?
    echo ""
}

# ----------------------------------------------------------
# Step 06: Common Result Ligate + index
# ----------------------------------------------------------
step06_common_ligate() {
    echo "========================================"
    echo "  Step 06: Ligate Common Phasing Results"
    echo "========================================"

    local ligate_vcf="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.common_result.ligate.vcf.gz"

    if [ -f "${ligate_vcf}" ]; then
        echo "[SKIP] Ligated VCF already exists: ${ligate_vcf}"
        return 0
    fi

    # --- Step 06a: Create ligate file list ---
    echo "  [06a] Creating ligate file list..."
    local ligate_list="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.common_result.ligate_merge.list"

    find ${RESULT_DIR} -name "chr${CHR}.ukbb_from_pgen.*.*_*.common.vcf.gz" \
        | sort -V > ${ligate_list}

    echo "  Ligate list: ${ligate_list} ($(wc -l < ${ligate_list}) files)"

    # --- Step 06b: SHAPEIT5_ligate ---
    echo "  [06b] Running SHAPEIT5_ligate..."
    date +"%Y-%m-%d-%H:%M"

    ${LIGATE} \
        --input ${ligate_list} \
        --output ${ligate_vcf} \
        --log ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.common_result.ligate.log \
        --thread 32

    check_step "step06b_ligate" $?

    # --- Step 06c: Build tabix index (as rare phasing scaffold) ---
    echo "  [06c] Building tabix index for ligated VCF..."
    ${TABIX} -p vcf --threads 8 ${ligate_vcf}
    check_step "step06c_tabix_ligate" $?
    echo ""
}

# ----------------------------------------------------------
# Step 07: Rare Variant Phasing
# ----------------------------------------------------------
step07_rare_phasing() {
    echo "========================================"
    echo "  Step 07: Rare Variant Phasing"
    echo "========================================"

    local merge_vcf="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.vcf.gz"
    local ligate_vcf="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.common_result.ligate.vcf.gz"
    local rare_chunk="${BLOCK_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.chunk_4cm.list_rare"

    echo "  [07] Running SHAPEIT5_phase_rare..."
    date +"%Y-%m-%d-%H:%M"

    cat ${rare_chunk} | ${PARALLEL} -j ${PARALLEL_JOBS_PHASE} --colsep '\t' \
        ${PHASE_RARE} \
        --input-plain ${merge_vcf} \
        --input-region ${CHR}:{6}-{7} \
        --scaffold ${ligate_vcf} \
        --scaffold-region ${CHR}:{3}-{4} \
        --map ${GMAP_FILE} \
        --output ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}.{6}_{7}.rare.vcf.gz \
        --log ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}.{6}_{7}.rare.log \
        --input-maf ${COMMON_MAF} \
        --pedigree ${PED_FILE} \
        --thread 160

    check_step "step07_phase_rare" $?
    echo ""
}

# ----------------------------------------------------------
# Step 08: Rare Result filtering (AC>=10) + merging
# ----------------------------------------------------------
step08_rare_filter_concat() {
    echo "========================================"
    echo "  Step 08: Filter (AC>=10) & Concat Rare Results"
    echo "========================================"

    local rare_chunk="${BLOCK_DIR}/chr${CHR}.ukbb_from_pgen.norm.merge.chunk_4cm.list_rare"
    local final_vcf="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.filtered_ac_10.phased.vcf.gz"

    if [ -f "${final_vcf}" ]; then
        echo "[SKIP] Final phased VCF already exists: ${final_vcf}"
        return 0
    fi

    # --- Step 08a: Filter each chunk (AC>=10) ---
    echo "  [08a] Filtering each chunk (AC>=10)..."
    date +"%Y-%m-%d-%H:%M"

    cat ${rare_chunk} | ${PARALLEL} -j ${PARALLEL_JOBS_BCF} --colsep '\t' \
        ${BCFTOOLS} view \
        --min-ac ${MIN_AC_FILTER} \
        ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}.{6}_{7}.rare.vcf.gz \
        -Oz -o ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.{1}.{6}_{7}.filtered_ac_10.rare.vcf.gz \
        --threads 32

    check_step "step08a_filter" $?

    # --- Step 08b: Create concat list ---
    echo "  [08b] Creating concat list..."
    local concat_list="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.filtered_ac_10.rare_result.concat.list"
    > ${concat_list}

    while read index chr_1 com_start com_end chr_2 rare_start rare_end variantCount; do
        local f="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.${index}.${rare_start}_${rare_end}.filtered_ac_10.rare.vcf.gz"
        if [ -f "${f}" ]; then
            echo "${f}" >> ${concat_list}
        fi
    done < ${rare_chunk}

    # --- Step 08c: bcftools concat ---
    echo "  [08c] Concatenating filtered rare results..."
    date +"%Y-%m-%d-%H:%M"

    ${BCFTOOLS} concat \
        -n \
        -f ${concat_list} \
        -Oz -o ${final_vcf} \
        --threads 32

    check_step "step08c_concat" $?
    echo "  Final phased VCF: ${final_vcf}"
    echo ""
}

# ----------------------------------------------------------
# Step 09: Switch Error Rate (SER) estimation (optional)
# ----------------------------------------------------------
step09_switch_error() {
    echo "========================================"
    echo "  Step 09: Switch Error Rate Estimation"
    echo "========================================"

    local phased_vcf="${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.filtered_ac_10.phased.vcf.gz"
    local chunk_list="${BLOCK_DIR}/chr${CHR}_1Mb_step.list"

    # Two steps required:
    # 9a: Re-phase after removing parents (using step09.1 logic)
    # 9b: Compare SER of full phasing vs parent-removed phasing

    echo "  [09a] Removing parents and re-phasing (standalone SER preparation)..."
    echo "  [NOTE] This step requires a separate run. Use --ser-mode for full pipeline."
    echo ""

    # --- Simplified SER: run switch on existing phased results ---
    echo "  [09b] Measuring SER on existing phased results..."

    # Convert to pgen for faster reading
    ${PLINK2} --vcf ${phased_vcf} --make-pgen \
        --out ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.phased \
        --threads 40

    # Extract by 1Mb and run switch (requires control data)
    # Full SER needs two sets of phased results (with vs without parents)
    echo "  SER measurement requires parent-removed phased data as control."
    echo "  Run with --full-ser to execute the complete SER workflow."
    echo ""
}

# ----------------------------------------------------------
# Main workflow dispatcher
# ----------------------------------------------------------
main() {
    validate_config

    # Ensure output directories exist
    mkdir -p ${BLOCK_DIR} ${RESULT_DIR}

    # Define step execution order and corresponding functions
    declare -A STEPS=(
        ["step01"]="step01_create_block_list"
        ["step02"]="step02_pgen_to_vcf"
        ["step03"]="step03_glimpse_chunk"
        ["step04"]="step04_chunk_format"
        ["step05"]="step05_common_phasing"
        ["step06"]="step06_common_ligate"
        ["step07"]="step07_rare_phasing"
        ["step08"]="step08_rare_filter_concat"
        ["step09"]="step09_switch_error"
    )

    local step_order=("step01" "step02" "step03" "step04" "step05" "step06" "step07" "step08" "step09")

    # Find starting step index
    local start_idx=0
    local start_found=0
    for i in "${!step_order[@]}"; do
        if [ "${step_order[$i]}" == "${START_STEP}" ]; then
            start_idx=$i
            start_found=1
            break
        fi
    done

    if [[ "$start_found" -ne 1 ]]; then
        echo "[ERROR] Unknown start step: ${START_STEP}" >&2
        usage >&2
        exit 2
    fi

    echo "============================================"
    echo "  Shapeit5 UKBB WGS Phasing Pipeline"
    echo "  Chromosome: ${CHR}"
    echo "  Start Step: ${START_STEP}"
    echo "  Work Dir:   ${WORK_DIR}"
    echo "============================================"
    echo ""

    # Execute sequentially from the specified step
    for ((i=start_idx; i<${#step_order[@]}; i++)); do
        local step=${step_order[$i]}
        echo ">>> Executing ${step}..."
        ${STEPS[$step]}
    done

    echo "============================================"
    echo "  Pipeline Complete!"
    echo "  Final output: ${RESULT_DIR}/chr${CHR}.ukbb_from_pgen.filtered_ac_10.phased.vcf.gz"
    echo "============================================"
}

# Run
main
