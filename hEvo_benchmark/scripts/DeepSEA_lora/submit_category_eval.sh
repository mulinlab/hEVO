#!/bin/bash
set -euo pipefail

SCRIPT="<source-project-root>/scripts/DeepSEA_lora/evaluate_category_metrics_single.sbatch"
ROOT="<source-project-root>/results/DeepSEA_lora"

submit() {
  local job_name="$1"
  local run_name="$2"
  sbatch --job-name="${job_name}" \
    --export="ALL,RUN_DIR=${ROOT}/${run_name}" \
    "${SCRIPT}"
}

submit cat_aido_old 'AIDO.DNA-300M__lora-r8-a16-d0.05__len1024__seed42__screen__20260814'
submit cat_dnabert2 'DNABERT2__lora-r8-a16-d0.05__len512__seed42__screen__20260814'
submit cat_gena_base 'GENA-LM-base__lora-r8-a16-d0.05__len512__seed42__screen__20260814'
submit cat_gena_large 'GENA-LM-large__lora-r8-a16-d0.05__len512__seed42__screen__legacy-20260816_155025'
submit cat_generator 'GENERator-1.2B__lora-r8-a16-d0.05__len512__seed42__screen__legacy-20260816_155025'
submit cat_grover 'GROVER__bert-targets__lora-r8-a16-d0.05__len512__seed42__screen-20260816_164217'
submit cat_hevo340 'hEvo-large__lora-r8-a16-d0.05__len1024__seed42__screen__eager-fp16-legacy__20260815'
submit cat_hevo86 'hEvo-base__lora-r8-a16-d0.05__len1024__seed42__screen__eager-fp16-legacy__20260815'
submit cat_hevo86fa 'hEvo-base__lora-r8-a16-d0.05__len1024__seed42__screen__bf16-torchflash__20260815'
submit cat_mistral17 'Mistral-DNA-17M__lora-r8-a16-d0.05__len512__seed42__screen__legacy-20260816_155025'
submit cat_mistral138 'Mistral-DNA-138M__bf16__lora-r8-a16-d0.05__len512__seed42__screen-'
submit cat_mistral422 'Mistral-DNA-422M__lora-r8-a16-d0.05__len512__seed42__screen__legacy-20260816_155025'
submit cat_nt25 'NT_2.5b_1kG__lora-r8-a16-d0.05__len1024__seed42__screen__20260814_143027'
submit cat_nt25ms 'NT_2.5b_MS__lora-r8-a16-d0.05__len1024__seed42__screen__legacy-20260816_155025'
submit cat_nt500 'NT_500m_1kG__lora-r8-a16-d0.05__len1024__seed42__screen__20260814_143027'
submit cat_ntv3 'NTv3-8M-pre__custom-masklm__lora-r8-a16-d0.05__len1024__seed42__screen-20260816_164217'
