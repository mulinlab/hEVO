# Final S1 six-PC cohort ancestry annotation

The cohort annotation reads PC1–PC6 from S1, using the frozen distance-weighted kNN with 15 neighbors, sigmoid
calibration, probability threshold 0.80, probability margin 0.30 and the
population-envelope distance rule (quantile 0.999, five neighbors).
The reference model uses 2,791 1000 Genomes/HGDP individuals.

The final resource contains 63,809 individuals in the S2 sample order.
The 3,188 internal Chinese-cohort individuals receive EAS through an explicit
cohort annotation. Other samples retain model assignments. Accepted AMR,
MID and OCE predictions, and rejected predictions, are displayed as Others.
Probabilities describe model predictions.

## Environment and model resources

Create the pinned environment in `env/environment.yaml`. The model, reference
metadata and fixed calibration folds are in `resources/final_6pc/`.
`protocol.json` is a required runtime resource record: model files and dependency
versions are checked before use. Reference labels and calibration folds are
retained for reproducibility. Training and output directories remain outside
the repository; an existing output directory is never overwritten.

## Run

Copy `config.example.yaml`, set its two input paths and output directories,
then run from this directory:

```bash
python run.py --config /path/to/config.yaml predict --dry-run
python run.py --config /path/to/config.yaml predict
python run.py --config /path/to/config.yaml check
```

`check` validates resource versions, sample IDs, coverage and current-table
agreement.
Relative input paths are resolved relative to the configuration file.

`train` refits the same six-PC protocol using the bundled reference identities
and the five population-grouped calibration folds. It writes a separate
model and never replaces the bundled model:

```bash
python run.py --config /path/to/config.yaml train --dry-run
python run.py --config /path/to/config.yaml train
```

## Table and coordinate utilities

- `sync_ancestry_tables.py` joins predictions by ID to update S2/S4, R02/R28
  and the cohort-composition sheet S1 in the 32-table publication workbook.
  Its required backup/output directory must be outside the repository.
- `correct_cohort_sources.py` establishes cohort membership using source VCF
  sample sets; it preserves the separate ancestry fields.
- `generate_pca_facet_pdf.py` renders supplied PC coordinates by cohort.

The `population_classify` workflow remains the independent DNA-model
embedding benchmark for S33–S35. Sequence-sharing tables are documented in the table guide.
