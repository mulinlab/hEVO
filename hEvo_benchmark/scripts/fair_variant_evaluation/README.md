# Matched causal and rare variant workflows

This directory contains input builders, paired-score evaluators, and
summarizers for matched causal and rare-variant comparisons.

## Input construction

The causal workflow builds positive and control arms from supplied source files.
Controls are matched by chromosome and allele-frequency strata. The rare-variant
workflow uses a supplied source table and constructs fixed-strata summaries from
the supplied score files.

Builders take input tables, reference FASTA, random seed, and output paths
through command-line arguments. They write canonical hg38 REF/ALT sequence
pairs and identifiers used by downstream evaluators.

## Score protocol

The hEvo evaluation route uses paired 1,023-bp inputs with center index 511 and
the configured signed score orientation. Native GPN-Star evaluation uses its
coordinate-aware MSA input format and records that protocol with its run
metadata.

Use `reorient_scores.py` for scoring methods whose pair-swap operation exactly
reverses the score sign. For other scoring methods, evaluate the canonical
REF/ALT pair directly.

## Evaluation

Submit run_one.sbatch with an input file, model, and output directory:

    BENCHMARK_ROOT=<benchmark-root> DATASET_FILE=<dataset-file> DATASET_KIND=causal ARM_ID=C0 MODEL_ID=hEvo-large MODEL_PATH=<model-path> RUN_ROOT=<run-root> sbatch --export=ALL run_one.sbatch

The job writes paired scores and a standardized score table under the supplied
run directory. Summarizers join by variant identifier and use the task label
and score-orientation fields in the input tables.

## Sensitivity summaries

summarize_rare_strata.py groups the supplied rare-variant rows by transition
class, allele bases, SNV class, CpG context, GC-content interval, and chromosome.
Membership is defined by input fields.

build_causal_filter_grid.py constructs arms from posterior-probability,
effect-size, and allele-frequency criteria. The evaluation script reads the
generated arm tables and corresponding score files.

All workflow inputs and outputs are selected through command-line paths.
Configure reference files, checkpoints, score caches, and run directories in
their designated data and runtime locations.

[Back to parent guide](../README.md)
