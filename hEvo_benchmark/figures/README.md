# hEvo publication figures

Figure renderers and shared plotting helpers are under `figures/scripts/`.
Renderers read local source tables from `tables/` and write PDF, SVG, and PNG
to the matching `figures/Fig*` directories.

Run the active figure suite from the benchmark directory:

```sh
micromamba -r <micromamba-root> -n llmbm run python figures/scripts/run_all_figures.py --tables-dir tables --output-dir figures
micromamba -r <micromamba-root> -n llmbm run python figures/merge_a4_figures.py
```

For the publication build, set `--output-dir figures` to write the official
figure folders. The batch runner requires that canonical output because the
FigS1, FigS2, and FigS7 builders write directly to their registered
folders. The merge command creates a 18-page PDF ordered Fig. 1–5, then
Fig. S1–S13.

The publication supplementary set is FigS1–FigS13. Each integrated PDF is a single
A4 portrait page. FigS1 has five panels (a–e); FigS2 is one unlettered
architecture schematic; FigS3 has panels a/b; FigS4 has three dataset columns;
FigS5 has panels a/b; FigS6 is one unlettered DeepSEA summary table; and FigS7
has two panels (a/b: AUPRC/AUROC).

## Source tables and workflows

- Fig. 1 uses cohort and haplotype tables S1–S10. FigS1 uses S1–S3 and S7–S9
  for its panels and source cross-checks.
- Fig. 2 and FigS3–FigS6 use genomic-element and model-evaluation tables
  S11–S21. FigS3 uses S17–S18, FigS4 uses S11–S16, FigS5 uses S19–S20, and
  FigS6 uses S21. FigS7 uses the variant-task tables S23 and S26–S30.
- FigS2 uses the local hEvo architecture configs, training/evaluation source
  code, and model metadata S47.
- Fig. 3 uses variant-benchmark tables S22–S32.
- Fig. 4 uses ancestry, evolutionary-region, and adaptive-allele tables
  S33–S39. Fig. 4e compares hEvo-large with GPN-Star-M using the checked-in
  source bundle in `tables/fig4e_haqer_tracks/`, led by
  `Fig4e_track_source_data.csv`.
- Fig. 5 uses ablation and efficiency tables S41–S44, with model
  metadata from S47 and cross-task ranks from Table S49.

S8, S9, and S10 remain source tables for the main figures. FigS8 summarizes
ancestry analyses and FigS9 summarizes accelerated regions and adaptive alleles. Figures FigS10–FigS13 are included in the default 18-page publication merge.
Table names, field definitions, and figure inputs are listed in the table guide.

## Figure output

Each figure folder contains the integrated output.
Figures with standalone panels also export those panels beside the integrated
page.

[Back to parent guide](../README.md)
