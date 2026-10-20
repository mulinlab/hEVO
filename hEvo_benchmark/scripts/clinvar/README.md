# ClinVar subset reconstruction

This directory exports S23–S25 from all/coding/noncoding evaluation
results. Run commands from the repository
root with Python >=3.10, numpy, pandas, h5py, scipy and scikit-learn. Raw inference
is handled by [evaluate](../evaluate/README.md).

## Choose an interface

- `export_clinvar_all_subsets.py`: hEvo-large export; preserves its
  existing CLI and delegates explicit `--input-manifest` inputs to reconstruction.
- `rebuild_gpn_subsets.py`: multi-model GPN reconstruction, with identity, labels,
  sequence hashes, scored/skipped coverage and score-direction validation.

An input manifest identifies the dataset root and each model's TSV/HDF5 cache,
its score protocol, checksum and optional sidecar manifest. Supply a JSON manifest with a `models` array. Each model has `model`, `format`
(`tsv` or `hdf5`), and `path`; TSV additionally requires the checksummed sidecar
`manifest`. A minimal TSV input looks like:

```json
{"models": [{"model": "GPN-Star-M", "format": "tsv", "path": "/path/to/all_scores.tsv", "manifest": "/path/to/all_scores.manifest.json"}]}
```

For an HDF5 source, preserve the recorded `scoring_method`; an explicitly
reconciled new all-run summary uses `all_summary`. Supply source data with
`--data-dir` and the existing S23–S25 directory with `--tables-dir`; paths
inside the manifest resolve from the invocation directory. Use absolute paths
for portability.

```bash
python hEvo_benchmark/scripts/clinvar/rebuild_gpn_subsets.py --help
python hEvo_benchmark/scripts/clinvar/export_clinvar_all_subsets.py \
  --input-manifest "$CLINVAR_MANIFEST" \
  --output-dir "$CLINVAR_OUTPUT_DIR" --validate-only
```

Run validation first. Without `--apply`, the reconstruction exports metrics to the selected output directory. `--apply` explicitly writes the
formal tables. `--validate-only` checks sources without bootstrap or table edits.
The exporter requires explicit source-summary, score-cache, coding and
noncoding CSV, table and provenance paths; see its `--help`.

## Protocol and outputs

Subset membership uses chromosome/position/REF/ALT and labels. Per-model evaluated prevalence and excluded variants are preserved.
Scores use the recorded protocol and orientation; AUPRC is average precision,
AUROC is ROC AUC, and subgroup CIs use the specified row bootstrap.
The output directory contains metrics and per-model subset scores. Reconstruction needs CPU/RAM for the full caches; no GPU is required.

Identity mismatch, duplicate IDs, incomplete scored/skipped coverage, checksum
mismatch and incompatible protocol block export. Use matching scoring protocols when comparing models.

[Workflow index](../README.md)
