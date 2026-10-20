# Evaluation entry points

The public evaluator directory keeps two Python command entry points at its root;
shared implementations and compatibility commands live in purpose-specific
subdirectories:

```text
evaluate/
├── probe_single_seq_input.py
├── zeroshot_paried_seq_input.py
├── adapters/       model/input adapters
├── compat/         historical command/import paths
├── core/           probe protocol and provenance logic
├── gpn_star_variant/ GPN-Star task preparation, sharding and export
├── results_export/ validation, table export and shard aggregation
└── runtime/        GPN-Star split/merge and resource profiling
```

The benchmark has two public evaluator families:

- `probe_single_seq_input.py` is the frozen-backbone embedding probe. Select
  `--protocol standard` for the regular probe or `--protocol pareto` for the
  Pareto/resource route used by `NT_Pareto` and Fig. 5. The
  `compat/probe_single_seq_input_Pareto.py` path remains available as a
  compatibility entry point. With `--protocol standard`, `GPN-MSA` reads
  coordinate-bearing NT
  benchmark records and uses `gpn.msa.embedding.EmbeddingInference` against the
  aligned MSA store.
- `zeroshot_paried_seq_input.py` is the shared paired-sequence variant
  evaluator. It uses the configured LLR scorer for compatible models, routes
  Mistral-DNA and GENERator to their native variant scorers, and dispatches
  `GPN-Star-V`, `GPN-Star-M` and `GPN-Star-P` to the official native-MSA
  protocol. GPN-Star embedding probes use the coordinate-aware
  `EmbeddingInference` interface and the matching hg38 MSA store.

Coordinate-aware GPN-MSA/GPN-Star probes use `--protocol standard --stage full`.
The Pareto and sequence-only sharded routes do not support these models. Use the
native GPN-Star resource runner for resource measurements.

The shared single-sequence variant evaluator has two scoring routes: MLM
models mask the changed target tokens in the reference context and sum the
ALT-minus-REF token log-probability contrasts; causal models compare their
sequence likelihoods or use the native scorers below. GPN-MSA and
GPN-Star retain their coordinate-aware native MSA interfaces.

The hEvo model names are `hEvo-base`, `hEvo-large`, `hEvo-open-base`, and
`hEvo-open-large`. Their configured paths are supplied through
[`configs/models.yaml`](../../configs/models.yaml); each name maps to its own
checkpoint configuration.

`merge_gpn_msa_nt_native_results.py` validates completed coordinate-aware
GPN-MSA NT-v2 probe outputs and updates the corresponding rows and task ranks
in supplementary table S17. Supporting tools are grouped by purpose:
`results_export/` contains table export, shard aggregation, and native ClinVar
result handling; `runtime/` contains GPN-Star splitting, merging, and resource
profiling; `gpn_star_variant/` handles non-ClinVar inputs, shards, and exports;
and `compat/` contains wrappers for earlier script paths. Score calibration is
implemented in [calibrate_gpn_star_scores.py](../revision_consistency/calibrate_gpn_star_scores.py).

`hEvo-open-base` and `hEvo-open-large` were independently pretrained on public
1000 Genomes Project and HGDP sequences, independently of benchmark task labels.
In paired REF/ALT evaluation, they use the same zero-shot LLR contract as
other masked language models.

`adapters/gpn_star_input.py` normalizes variant tables, and
`adapters/gpn_star_adapter.py` calls the native GPN-Star VEP protocol. Configure
checkpoint and MSA paths in the runtime settings. Each run writes manifests,
scores, and logs to its selected output directory.

## Native paired-variant scoring for Mistral-DNA and GENERator

The shared `zeroshot_paried_seq_input.py` evaluator routes the five configured
Mistral-DNA and GENERator model names to their model-native paired-variant
scorers. These paths apply to the paired-variant tasks supported by that
evaluator:

- Mistral-DNA: run the causal LM on each paired REF/ALT sequence, max-pool
  token logits over the sequence, and score the L2 distance between pooled
  vectors. It uses the full centered 1,023 bp input, within the benchmark's
  1,024 bp maximum and the model's 512-token limit. The evaluator follows the
  tutorial's executable `AutoModelForCausalLM` implementation, which returns
  4,096-dimensional vocabulary logits.
- GENERator: read the hg38 sequence upstream of each variant, align the
  upstream length to a 6-mer boundary, predict the next token, marginalize
  probability by the token's first DNA base, and score `log(P_ref/P_alt)`.
  Its default and maximum context is 1,024 bp under the shared benchmark
  protocol; 6-mer alignment leaves at most 1,020 bp as model input. It needs an
  indexed hg38 FASTA.

Task score modes refer to canonical ALT-minus-REF LLR. GENERator caches retain
their REF-minus-ALT quantity: the default disruption mode uses it directly,
`--score_mode raw_llr` negates it, and `--use_abs` or `--score_mode abs_llr`
uses its magnitude. These score transformations share the same raw cache.
Mistral-DNA distances are evaluated directly as distance scores.

For GPN-MSA jobs, select the coordinate-aware native MSA evaluator with
`--gpn-msa-evaluate-script`; other models retain the shared evaluator.

`runtime/native_clinvar_models.py` is a separate ClinVar-specific runner for the same
model-native scoring definitions. It supports isolated ClinVar runs. Example:

```bash
micromamba run -n llmbm python hEvo_benchmark/scripts/evaluate/runtime/native_clinvar_models.py \
  --family generator \
  --model-name GENERator-1.2B \
  --model-path "$DNAMODELS_BENCHMARK_ROOT/models/GENERator-v2/GENERator-v2-eukaryote-1.2b-base" \
  --output-dir hEvo_benchmark/results/native_protocol_recheck_20261005/GENERator_1024bp/GENERator-1.2B
```

For Mistral-DNA set `--family mistral` and provide the desired model path/name.
Each run writes per-variant TSV scores, metrics JSON, and a provenance manifest.

## Standard probe and paired-variant examples

Run from the repository root. CSV column indices in these examples are zero-based;
use the values in tasks.yaml for the selected dataset. A standard single-sequence
CSV needs the configured sequence and label columns, with distinct train/test files.
A paired-variant CSV needs ID, REF sequence, ALT sequence and label columns, plus
coordinates when required by a native MSA scorer.

```bash
python hEvo_benchmark/scripts/evaluate/probe_single_seq_input.py \
  --model_paths "$MODEL_DIR" --model_name "$MODEL_ID" \
  --protocol standard \
  --dataset_files "$TRAIN_CSV,$TEST_CSV" --dataset_name "$TASK_ID" \
  --seq_col 1 --label_col 2 --sep ',' --max_length 1024 \
  --hdf5_dir "$EMBEDDING_CACHE_DIR" --out_dir "$PROBE_RESULT_DIR" \
  --log_dir "$PROBE_LOG_DIR"

python hEvo_benchmark/scripts/evaluate/zeroshot_paried_seq_input.py \
  --model_paths "$MODEL_DIR" --model_name "$MODEL_ID" \
  --dataset_files "$VARIANT_CSV" --dataset_name "$TASK_ID" \
  --idx_col 0 --ref_col 1 --alt_col 2 --label_col 3 --sep ',' \
  --h5_path "$VARIANT_CACHE_DIR" --out_dir "$VARIANT_RESULT_DIR" \
  --log_dir "$VARIANT_LOG_DIR"
```

These commands create caches, result/metric JSON, and run provenance. They require
model checkpoints and input files. Use scheduler generators for repeated
model/task runs. Set HG38_FASTA for GENERator and configure the documented MSA
paths for GPN.

`create_probe_single_seq.py` selects the Pareto protocol automatically for the
`NT_Pareto` task family and the standard protocol for other task families. Set
`--protocol` to override that choice. An explicit `--evaluate-script` takes
precedence and is invoked without a protocol flag.

[Back to parent guide](../README.md)
