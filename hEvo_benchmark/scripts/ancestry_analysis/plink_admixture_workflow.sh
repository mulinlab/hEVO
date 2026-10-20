#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Usage: plink_admixture_workflow.sh VCF_DIR OUT_DIR [K=5]
Inputs: chr1.vcf.gz through chr22.vcf.gz in VCF_DIR.
Outputs: merged/QC/pruned PLINK files and ADMIXTURE estimates in OUT_DIR.
Requirements: plink, GNU parallel, admixture; THREADS defaults to 22.
USAGE
    exit 0
fi

VCF_DIR="${1:?VCF_DIR is required}"
OUT_DIR="${2:?OUT_DIR is required}"
K="${3:-5}"
THREADS="${THREADS:-22}"

TMP_DIR="${OUT_DIR}/temp_plink_bin"
MERGE_LIST="${OUT_DIR}/merge_list.txt"
mkdir -p "${TMP_DIR}" "${OUT_DIR}"
rm -f "${MERGE_LIST}"

parallel -j "${THREADS}" \
  "plink --vcf ${VCF_DIR}/chr{}.vcf.gz --make-bed --out ${TMP_DIR}/chr{} --const-fid 0 --allow-extra-chr" ::: {1..22}

for chrom in $(seq 2 22); do
  echo "${TMP_DIR}/chr${chrom}" >> "${MERGE_LIST}"
done

plink --bfile "${TMP_DIR}/chr1" \
  --merge-list "${MERGE_LIST}" \
  --make-bed \
  --out "${OUT_DIR}/merged" \
  --set-missing-var-ids @:#

plink --bfile "${OUT_DIR}/merged" \
  --geno 0.05 \
  --mind 0.05 \
  --maf 0.01 \
  --make-bed \
  --out "${OUT_DIR}/qc"

plink --bfile "${OUT_DIR}/qc" \
  --indep-pairwise 50 5 0.2 \
  --out "${OUT_DIR}/prune_list"

plink --bfile "${OUT_DIR}/qc" \
  --extract "${OUT_DIR}/prune_list.prune.in" \
  --make-bed \
  --out "${OUT_DIR}/final_for_admixture"

(cd "${OUT_DIR}" && admixture --cv -j"${THREADS}" final_for_admixture.bed "${K}")

