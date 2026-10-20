# GPN-Star ancestry workflow

These scripts prepare coordinate-aware GPN-Star embeddings for ancestry
classification and assemble the corresponding tables and figure inputs. MSA
stores, run data, and output locations are supplied by the local environment.

## Workflow

1. Run `preflight_gpn_star_ancestry.py` with an input JSONL and MSA root to
   validate the inputs and configure the cache.
2. Use `extract_gpn_star_ancestry_emb.py` to extract embeddings for the input
   variants.
3. Use `roll_gpn_star_ancestry_arrays.py` to submit and coordinate array jobs
   when running on Slurm.
4. Run `finalize_gpn_star_ancestry.py` to assemble the embedding outputs and
   generate the table and figure inputs.

The MSA root must contain the store layouts required by the selected
GPN-Star adapters. Set `GPN_STAR_MSA_ROOT` or pass `--msa-root` explicitly.
Input, cache, benchmark, run, and output roots are configurable through the
command-line options; generated embeddings and manifests are written to the
specified run directory.

## Example preflight

Run from this directory with the required Python environment:

```bash
python preflight_gpn_star_ancestry.py \
  --input-file <input.jsonl> \
  --output-json <preflight-report.json> \
  --msa-root <gpn-star-msa-root>
```

For local extraction, invoke `extract_gpn_star_ancestry_emb.py` with the
matching run configuration. On Slurm, use the `.sbatch` templates in this
directory and then pass the run and benchmark roots to the finalizer.

## GPN-MSA 89-way ancestry workflow

The GPN-MSA ancestry workflow uses the published `89.zarr` alignment (human
plus 89 auxiliary species) and the matching `gpn-msa-sapiens` checkpoint. It
validates the shared 10,000-locus roster and reference alleles, then caches
the 128-bp auxiliary-species windows so all GPU workers can reuse them. Each
worker applies the sample's GT[0] haplotype to the human sequence, extracts
forward/reverse hidden-state deltas, and writes one checksummed 2,308-value
vector per sample.

Run `preflight_gpn_msa_ancestry.py` first. Then submit a one-GPU smoke task and
the full 40-worker array to Slurm's `gpu` partition with
`run_gpn_msa_ancestry_array.sbatch`. The full array is capped at 40 concurrent
one-GPU tasks. After all 1,500 sample vectors are complete, run
`finalize_gpn_msa_ancestry.py` to calculate S33-S35 and refresh the Fig4 ancestry outputs.
The preflight report, per-sample manifests, result vectors, table backups, and
finalization report are stored together under the chosen run root.

Submit from the benchmark directory or repository root. Use
`HEVO_POPULATION_SCRIPT_DIR` for a separate script checkout and
`HEVO_LLBM_PYTHON` to select an interpreter; the default interpreter is `python`
from the active environment. The GPN-Star rolling scheduler monitors the
current account; set `HEVO_SLURM_USER` to monitor a specified scheduler account.
For example, from the benchmark directory:

```bash
sbatch scripts/population_classify/run_gpn_msa_ancestry_array.sbatch /path/to/run smoke
sbatch scripts/population_classify/run_gpn_msa_ancestry_array.sbatch /path/to/run full
```

The pooled embedding follows the maintained GPN-MSA embedding interface:
the official tokenizer and reverse-complement handling, last-layer hidden
states, and an exact five-position center mean. The public CLI currently
expects a readable Zarr store path, so the provided adapter uses GPN's
`GenomeMSA` window methods over the published ZIP-backed Zarr without
unpacking the full archive.

[Back to parent guide](../README.md)

## Standard ancestry tasks and regression

`classify.py --base-dir "$ANCESTRY_EMBEDDINGS" --results-dir "$ANCESTRY_RESULTS"`
performs fold-local scaling and five-fold logistic classification. Input directories
contain model sample embeddings with the identities/labels understood by
`embedding_loader.py`. This is CPU analysis; extraction requires model runtimes
and GPU. GPN preflight/extraction additionally requires coordinate-bearing JSONL,
correct REF alleles, the pinned MSA store and model checkpoint.

`Regression_with_admixture.py` runs few-shot ancestry-proportion regression. Run from the repository root:

```bash
python hEvo_benchmark/scripts/population_classify/Regression_with_admixture.py \
  --base-dir "$ANCESTRY_EMBEDDINGS" --gt-path "$ADMIXTURE_Q" \
  --report-dir "$ANCESTRY_REGRESSION_REPORT" --target-model hEvo-large \
  --min-shots 10 --max-shots 60 --num-repeats 10 --seed-start 10 --num-workers 1
```

Ground truth is a whitespace table with SampleID, AFR, AMR, EUR, EAS, SAS and
SuperPop columns, aligned to embedding sample IDs. The regression uses the
repeated sampling, Ridge, scaling and softmax to estimate ancestry proportions.
Finalizers use this local script by default; `--regression-script` still overrides
it.

`run_open_large_ancestry_eval.py` provides external-suite orchestration
script. It requires
`DNAMODELS_BENCHMARK_ROOT` and original analysis modules/run assets. Sample roster,
shape or ground-truth identity mismatch should be resolved before exporting S33–S35.
