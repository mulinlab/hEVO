# GPN-Star variant reconstruction and shard tools

This directory prepares non-ClinVar datasets and merges/exports their
results. The inference implementation is the [GPN-Star adapter](../adapters/gpn_star_adapter.py).
Run from the repository root with Python >=3.10,
numpy, pandas, h5py, scipy and scikit-learn. Scoring additionally needs GPN-Star,
PyTorch, pinned checkpoints, hg38 MSA stores and GPU resources.

## Execution order

1. `prepare_nonclinvar_runs.py`: build filtered, checksummed inputs for causal,
   rare-causal, rare-variant, MPRA and DDD tasks.
2. `shard_variant_csv.py`: partition a CSV by row index; keep the shard count and
   task/model provenance when submitting inference jobs.
3. Use the official scoring interface and the configured model/MSA assets.
4. `merge_score_shards.py`: compatibility wrapper around the maintained
   [TSV merger](../runtime/gpn_star_merge.py). This path supports external launchers.
   `merge_hdf5_score_shards.py` instead merges HDF5 outputs; these formats are distinct.
5. `export_nonclinvar_results.py`: validate and export completed results
   to the selected repository's tables. This is a writing/export stage.

```bash
python hEvo_benchmark/scripts/evaluate/gpn_star_variant/prepare_nonclinvar_runs.py --help
python hEvo_benchmark/scripts/evaluate/gpn_star_variant/shard_variant_csv.py --help
python hEvo_benchmark/scripts/evaluate/gpn_star_variant/merge_score_shards.py --help
```

Preparation requires `--repo-root`, `--output-root` and `--rare-causal-h5-root`.
HDF5 merging requires repeated `--input`, `--output`, `--task-name` and `--manifest`.
Export requires `--run-root`, `--repo-root`, `--gpn-comparison`, `--rare-h5-root`
and `--ddd-h5-root`. See each CLI's help for the full schema and constraints.
Input CSVs retain original variant IDs, paired sequences and labels; scored
records retain coordinate/REF/ALT identity and status. Outputs include filtered
views, TSV/HDF5 caches, metric summaries and checksummed manifests.

Preparation and merging run on CPU; scoring uses the configured GPU runtime.

[Workflow index](../README.md) · [Job generation](../../create_slurm_scripts/README.md)
