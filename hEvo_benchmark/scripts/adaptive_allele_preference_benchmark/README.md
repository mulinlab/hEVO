# Adaptive Allele Preference Benchmark

This workflow evaluates whether a model prefers the literature-supported selected allele over the other allele at a curated locus. Its score is `log P(selected | context) - log P(other | context)`, and its input grain is a model/locus pair.

| Stage | Scripts/files | Responsibility |
|---|---|---|
| Model roster | `model_registry_29.yaml`, `model_registry_34.yaml` | Checkpoint rosters |
| Scoring | `05_score_generic_model.py` | Generic masked/causal inference and model-family adapters |
| Direction correction | `allele_direction.py` | Idempotent selected/other orientation correction; REF/ALT remains unchanged |
| Aggregation | `06_summarize_cross_model.py` | Preference rate, exact binomial intervals, Wilcoxon tests, locus bootstrap and paired comparisons |
| Reporting | `07_write_cross_model_report.py` | Registry-aware cross-model report |
| Table export | `summarize_final_tables.py` | Release tables and model-name normalization |
| Single-file summary | `summarize_adaptation.py` | Compact PASS/FAIL summary from a score TSV; positional file argument |
| GPN raw-score stage | `fig4_gpn_raw_stage/` | Separate raw-logit inference/merge protocol; see its guide |


Use each CLI script's `--help` for options.

## Figure data lineage

The adaptive-allele panels in Fig4 and FigS9 use this benchmark's results:

- `tables/S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv`: selected-allele preference and exact binomial intervals.
- `tables/S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv`: locus scores, raw selected/other log probabilities, heatmaps and Calinski-Harabasz score separation.

FigS9 additionally uses S36/S37 for HAR/HAQER, which are separate evaluations.

## Reproduce scoring and aggregate results

Run from the repository root. Scoring requires PyTorch/Transformers, pysam,
model-native dependencies, an indexed hg38 FASTA and the curated TSV roster.
Use a GPU, the declared model context, and selected-minus-other score orientation.

```bash
python hEvo_benchmark/scripts/adaptive_allele_preference_benchmark/05_score_generic_model.py \
  --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" --kind masked \
  --loci "$ADAPTIVE_LOCI_TSV" --reference-fasta "$HG38_FASTA" \
  --output "$ADAPTIVE_SCORE_TSV"
python hEvo_benchmark/scripts/adaptive_allele_preference_benchmark/06_summarize_cross_model.py \
  --score-files "$ADAPTIVE_SCORE_TSV" --output-dir "$ADAPTIVE_RESULT_DIR"
```

Each scoring TSV contains model/locus identity, alleles, selected/other likelihoods,
locus_score, status and token checks. Aggregate only complete, unique model/locus
grids with finite scores. Aggregation writes model_summary.tsv, bootstrap_summary.tsv,
leave-one-locus-out.tsv and paired_model_tests.tsv plus QA/result summaries.
`summarize_final_tables.py` exports S38/S39 from a completed result directory and
registry; use its help for explicit paths. Raw GPN scoring follows the dedicated
stage guide.

## Slurm launchers

Batch scripts are in [`slurm/`](slurm/). They cover generic adapter arrays and
the Fig. 4 GPN-Star raw-logit run. Set the required environment variables and
update local model and input paths for the target cluster before submission.
