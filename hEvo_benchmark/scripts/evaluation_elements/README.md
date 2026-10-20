# Genomic-element embedding evaluation

Evaluation code for the ENCODE, GENCODE, and UCSC RefSeq curated tasks. Frozen
task definitions are configured through the benchmark task settings.

- `evaluation/`: source-taxonomy model preflight, embedding extraction, PCA/t-SNE,
  CH calculation, aggregation, and Slurm task preparation. Frozen model adapters
  are in `evaluation/shared/`; Evo2 configuration is in `evaluation/evo2_configs/`.
- `taxonomy_v2/`: reuse source embeddings by exact sequence ID, reduce selected
  embeddings, and summarize CH for the merged taxonomy.
- `ucsc_refseq_curated/`: prepare the model roster, extract embeddings, run
  PCA/t-SNE/UMAP, submit jobs, and aggregate results.

Run commands from the repository root, for example:

```bash
python hEvo_benchmark/scripts/evaluation_elements/evaluation/evaluation_jobs.py --help
python hEvo_benchmark/scripts/evaluation_elements/taxonomy_v2/reuse_embeddings.py --help
python hEvo_benchmark/scripts/evaluation_elements/ucsc_refseq_curated/submit_pipeline.py --help
```

Use the `llmbm` environment for numerical dependencies. Model extraction also
requires the corresponding model runtimes and checkpoints. Supply external run
paths through each command's arguments; generated embeddings, coordinates,
manifests, logs, and result tables belong in those run directories. Historical
run snapshots and their provenance records are not modified by this relocation.
