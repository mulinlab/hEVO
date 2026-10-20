# HAR/HAQER genomic track workflow

This directory contains the GPN-Star-M window-scoring and peak-calling
workflow for HAR and HAQER regions. HAR/HAQER classification is handled by the
[shared evaluation workflow](../evaluate/README.md).

## Files

| Path | Purpose |
|---|---|
| `gpn_star_m_peak_workflow.py` | Prepare inputs, extract GPN-Star-M window features, train/select the scoring head, scan regions, and evaluate calls. |
| `slurm/` | GPU/CPU jobs and the dependency-linked pipeline submitter. |
| `core/` | Shared head training, window scanning, feature, and peak-calling modules. |

## Run on HPC

Run from the repository root after setting the model, MSA, reference, and
external run paths for the current environment:

```bash
bash hEvo_benchmark/scripts/har_haqer_workflow/slurm/submit_gpn_star_m_peak_pipeline.sh
```

The pipeline runs a GPU preflight, prepares data, extracts window features in
40 single-GPU array tasks (maximum 40 running at once), trains and selects the
head, then scans validation and test regions in 48 single-GPU array tasks
(maximum 40 at once). Caller configurations are selected on validation
chromosomes only. Slurm job IDs and logs are recorded in the run directory.

The 243-way store carries the hg38 target sequence and is checked against the
reference FASTA. The 204-way store is source-only, so preflight checks its
species/chromosome layout and verifies that every requested context lies
within its stored axis; out-of-range windows are marked invalid.

All generated arrays, weights, scan scores, calls, and reports stay outside the
Git repository at
`$DNAMODELS_BENCHMARK_ROOT/runs/gpn_star_m_har_haqer_v1/`. Set
`DNAMODELS_BENCHMARK_ROOT` before submitting jobs; the scripts default to
`$HOME/DNAModelsBenchmark`.

The test report includes window AUPRC/AUROC/center-top1, summit-hit
precision/recall/F1, interval overlap, and a separate HAR window-evaluation
description: HAR positive centers versus nearby non-overlapping genomic
flanks/backgrounds.

[Back to parent guide](../README.md)
