# Haplotype diversity workflow

This workflow scans phased VCFs, computes window-level haplotype diversity,
aggregates chromosome or cohort shards, prepares UK Biobank coverage curves,
and exports source tables for figures. Inputs and output locations are
configurable through the example environment file and command-line options.

## Environment and inputs

The scripts use Python with `numpy`, `pandas`, `pysam`, and `matplotlib`.
Slurm templates read cohort and VCF settings from `cohort_manifest.tsv` and
environment variables. Set paths for the project, input VCFs, work directory,
and output directory before submitting jobs.

```bash
source config.env.example
export HAPDIV_PROJECT_ROOT=<project-root>
export HEVO_DATA_ROOT=<data-root>
```

Use `--in-dir`, `--out-dir`, `--manifest`, and source-path options to override
the configured locations for individual scripts.

## Window diversity

Run a small local job with the same command-line interface used by the array
workflow:

```bash
micromamba -n llmbm run python compute_window_haplotype_diversity.py \
  --chrom 22 --max-windows 5 --max-samples-per-cohort 50 \
  --rarefaction-repeats 5 --out-dir <output-directory>
```

Submit the chromosome-array workflow and aggregate its shards with:

```bash
bash submit_haplotype_diversity_array.sh
sbatch aggregate_haplotype_diversity.sbatch
```

Common settings include `N_SHARDS`, `CONCURRENCY`, `WINDOW_SIZE`,
`WINDOW_SUBSAMPLE_RATE`, `MAX_SAMPLES_PER_COHORT`, `RAREFACTION_REPEATS`, and
`OUT_DIR`. The cohort manifest defines display labels, balancing units, and
input directories.

## UK Biobank coverage

The empirical coverage workflow has separate array and aggregation scripts:

```bash
bash submit_ukb_downsampling_vcf_empirical_array.sh
sbatch aggregate_ukb_downsampling_vcf_empirical_fullUKBB_ref50k.sbatch
```

The aggregate step writes curve and summary tables. Coverage is represented
on the unit interval; preserve the source values when exporting tables.

## Source-table export and figures

Run the exporter from the benchmark root. It writes S5 and S6 source tables to
`tables/` and accepts explicit input paths when working from a different
directory:

```bash
micromamba -n llmbm run python \
  scripts/haplotype_diversity_workflow/export_haplotype_diversity_source_tables.py
```

Use the plotting scripts to render the coverage and cohort-diversity panels:

```bash
micromamba -n llmbm run python \
  scripts/haplotype_diversity_workflow/plot_ukb_haplotype_coverage.py
micromamba -n llmbm run python \
  scripts/haplotype_diversity_workflow/plot_cohort_haplotype_diversity.py
```

Set `HAPDIV_PLOT_OUTPUT` to control figure output location. The workflow keeps
generated shards and run files under the configured output directory.

[Back to parent guide](../README.md)
