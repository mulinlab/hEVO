# Preprocessing statistics tables

This directory contains source tables used by Fig.1 population and haplotype
summaries. The two subdirectories contain cohort-level and 1000 Genomes
comparison tables built with the chr1–22 window workflow.

Both use 1,024-bp windows, 128-bp stride, and phased haplotypes. Population
columns are AFR, EUR, EAS, SAS, and Others; the 1000 Genomes tables omit AMR.

Each dataset directory contains:

- fig1b_dedup_by_chrom_real.csv: chromosome-level deduplication summaries;
- fig1c_rank_abundance_real.csv and fig1c_rank_abundance_histogram.csv:
  rank-abundance values and histogram bins;
- fig1d_population_abundance_real.csv and related top-sequence tables:
  population-level sequence counts;
- fig1d_population_abundance_summary_stats.csv: population summary fields;
- fig1e_population_upset_real.csv and fig1e_population_upset_full_table.csv:
  population-combination counts.

The preprocessing workflow source code is in the [manually gated HF repository](https://huggingface.co/MulinLab/hEvo-pretraining-code/tree/main/hEvo_benchmark/scripts/preprocess_stats); MulinLab reviews access requests. The generated result tables and SHA256 manifest remain public here.

[Back to parent guide](../README.md) · [Public code and workflow index](https://github.com/mulinlab/hEVO/tree/main/hEvo_benchmark/scripts)
