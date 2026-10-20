# DeepSEA LoRA evaluation

This directory contains the training, evaluation, and export scripts for the
DeepSEA low-rank adaptation workflow. Dataset files, model checkpoints, and
generated run outputs are provided through local paths at runtime.

The public hEvo identifiers are `hEvo-base`, `hEvo-open-base`, and
`hEvo-large`. Their model locations are configured in
[`configs/models.yaml`](../../configs/models.yaml). The `hEvo-open-base`
backbone is self-supervised; DeepSEA training fits task-specific supervised
LoRA heads.

## Commands

Run from the benchmark root with the `llmbm` environment:

```bash
micromamba -r <micromamba-root> -n llmbm run python \
  scripts/DeepSEA_lora/run_deepsea_lora.py --help

micromamba -r <micromamba-root> -n llmbm run python \
  scripts/DeepSEA_lora/evaluate_category_metrics.py --help
```

Use the Slurm templates in this directory for scheduled jobs. Pass data,
model, and output locations through command-line arguments or environment
configuration. The S21 source table is documented in
[`tables/README.md`](../../tables/README.md); its exporter accepts a source
directory and reads per-run category metric files.

The hEvo-open-large FP16 category evaluation can use the five-shard, 40-GPU array
launcher in `evaluate_category_metrics_array.sbatch`. Set `DEEPSEA_RUN_DIR`
to the run results directory before submission. The job writes logs in the
submission directory; `sbatch --output` and `--error` can select other log paths.
From the benchmark directory:

```bash
export DEEPSEA_RUN_DIR=/path/to/run/results
sbatch scripts/DeepSEA_lora/evaluate_category_metrics_array.sbatch
```

After all array tasks
complete successfully, aggregate and verify all 40 rank outputs with:

```bash
micromamba -n llmbm run python scripts/DeepSEA_lora/evaluate_category_metrics.py \
  --run_dir "$DEEPSEA_RUN_DIR" \
  --metadata "$DEEPSEA_DATASET_ROOT/deepsea_metadata.tsv" \
  --aggregate_only --expected_shards 40
```

To add the completed GPN-Star-M result to S21 from its validated summary:

```bash
micromamba -r <micromamba-root> -n llmbm run python \
  scripts/DeepSEA_lora/append_gpn_star_m_s21.py
```

## GPN-Star-M native MSA run

The published GPN-Star-M result in S21 is run `S21_run_23`, seed 42. The
Evo2-aligned launcher is
[`slurm/launch_gpn_star_m_evo2_aligned.sh`](slurm/launch_gpn_star_m_evo2_aligned.sh).
It uses 4.4 million training rows, one epoch, `row_stride=1`, BF16, 40 GPUs,
per-GPU batch 1 with 13-step accumulation (effective batch 520), learning rate
`2e-5`, weight decay `0.01`, warmup 0.03, and a constant schedule after
warmup. GPN-Star-M retains its native 1,000-bp MSA input and mean pooling. The
coordinate mapper uses hg38 and the 243-way target MSA followed by the 204-way
source-only MSA through GPN-Star's native `GenomeMSA` interface. Of 4,400,000
training rows, 4,230,062 map to hg38 (96.14%); 436,931 of 455,024 test rows
map (96.02%). DHS, HM, and TF metrics are written together in the
validated category summary. Runtime artifacts remain outside the repository
under `$DNAMODELS_BENCHMARK_ROOT/runs/deepsea_gpn_star_m_v1/`. Set
`DNAMODELS_BENCHMARK_ROOT` before submitting jobs; the launch scripts default
to `$HOME/DNAModelsBenchmark`.

## Smoke test

Use `run_deepsea_lora.py --smoke_test` or the Evo2 workflow's
`run_evo2_deepsea.py --smoke_only` with its Slurm launcher. The GPN-Star-M
launcher also runs its MSA smoke preflight before submitting the full training
job.

## Portable inputs and summaries

Training/category CLIs accept explicit config, dataset, metadata, checkpoint/run
and output paths. Config defaults now use the checked-in `configs/models.yaml`;
data/output defaults use the benchmark root or `DNAMODELS_BENCHMARK_ROOT` when set.
Defaults include 919 labels and model-specific scoring and training settings.

```bash
python hEvo_benchmark/scripts/DeepSEA_lora/summarize_category_metrics.py \
  --results-root "$DEEPSEA_RESULTS_ROOT" \
  --output-csv "$DEEPSEA_SUMMARY_CSV" --output-json "$DEEPSEA_SUMMARY_JSON"
```

This summary reads per-run `category_metrics.json` files on CPU.
`inspect_peft_keys.py RUN_DIR MODEL_NAME MODEL_PATH` accepts its positional
arguments and supports `--help`. Set `MODELGENERATOR_ROOT` or
`DNAMODELS_BENCHMARK_ROOT` to the external ModelGenerator checkout used by AIDO.

[Back to parent guide](../README.md)
