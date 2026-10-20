# Derived results and manuscript tools

This directory calculates rank and calibration sensitivity results from model
predictions and provides manuscript tools. Run commands from the repository root with Python
>=3.10, numpy, pandas, scipy and scikit-learn. Word tools additionally require
lxml/python-docx; plotting requires matplotlib.

## Rank and resource summaries

Use `hEvo_benchmark/tables/scripts/build_fig5_cross_task_ranks.py` to summarize current
S17–S37 task ranks, then `hEvo_benchmark/tables/scripts/build_pareto_supplementary_tables.py` for NT resource
results and `hEvo_benchmark/tables/scripts/build_fig1_family_radar.py` for family summaries. Pass the retained
result directories through each command's options.

`build_revision_evidence.py` generates derived rank comparisons and evaluation
metadata. Supply the source project and a writable output directory; `--apply`
also updates the corresponding repository tables. Use `--help` for its available
inputs and output options.

## Calibration sensitivity

The calibrated score is the raw LLR minus the forward-reference
pentanucleotide/ALT neutral mean. Task-specific score orientation is applied
after calibration. Coordinates are one-based hg38. Raw/cLLR comparisons use
records with available reference and calibration contexts.

```bash
python hEvo_benchmark/scripts/revision_consistency/run_cllr_sensitivity.py \
  --runs-root "$RUNS_ROOT" --assets-root "$GPN_ASSETS_ROOT" \
  --source-project "$SOURCE_PROJECT" --reference-fasta "$HG38_FASTA" \
  --output-dir "$SENSITIVITY_OUTPUT_DIR"
python hEvo_benchmark/scripts/revision_consistency/finalize_cllr_sensitivity.py --help
```

The outputs contain per-record calibrated scores, model/task summaries and
sensitivity plots. `finalize_cllr_sensitivity.py` adds native allele-preference
results using the same source-project, reference and output paths. For a single
input, use `scripts/revision_consistency/calibrate_gpn_star_scores.py` with `--input`,
`--calibration-table`, `--calibration-sha256`, `--reference-fasta`, `--model-id`,
`--model-revision` and `--output`. The former
`scripts/evaluate/compat/calibrate_gpn_star_scores.py` path remains available as a
compatibility entry point.

DDD candidates are grouped by positive variant ID for within-site top-1 ranking.
Retrospective ClinVar calculations use all five seeds and event-specific score
directions. Keep sensitivity outputs in a separate directory from primary scores.

## Manuscript copies

```bash
python hEvo_benchmark/scripts/revision_consistency/build_revised_documents.py \
  --output-dir "$DOCUMENT_OUTPUT_DIR"
```

This creates Word copies with tracked text edits. `revise_docx.py` provides the
Word-editing functions; `revise_markdown.py` updates the corresponding Markdown
text. These tools require the original manuscript and retained source records.
Keep original documents separate from the output directory.

[Workflow index](../README.md)
