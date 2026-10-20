#!/bin/bash
set -euo pipefail

PROJECT="<source-project-root>"
SBATCH_FILE=${PROJECT}/scripts/DeepSEA_lora/train_deepsea_lora.sbatch
RESULTS=${PROJECT}/results/DeepSEA_lora
STAMP=$(date +%Y%m%d_%H%M%S)
STAGE=${STAGE:-screen}

for MODEL_NAME in hEvo-base hEvo-large NT_500m_1kG NT_2.5b_1kG; do
  RUN_DIR=${RESULTS}/${MODEL_NAME}__lora-r8-a16-d0.05__len1024__seed42__${STAGE}__${STAMP}
  JOB_NAME=deepsea_lora_${MODEL_NAME}
  sbatch --job-name="${JOB_NAME}" \
    --export=ALL,MODEL_NAME="${MODEL_NAME}",OUTPUT_DIR="${RUN_DIR}",STAGE="${STAGE}" \
    "${SBATCH_FILE}"
done
