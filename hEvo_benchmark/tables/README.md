# Benchmark tables and figure source data

This directory contains S-series CSV tables used for benchmark results,
metadata, and figure rendering. The [R-series guide](repository_tables/README.md)
provides annotated reader tables, field definitions, and the R-to-S crosswalk.
Related S and R tables are views of the same underlying results.

## Navigation

- [Table index](#table-index): available S-series tables, grouped by subject.
- [Data definitions](#data-definitions): sample identifiers, ancestry, models, and evaluation protocols.
- [Figure inputs and aggregation](#figure-inputs-and-aggregation): ranking rules and figure-specific selections.
- [Sources and companion results](#sources-and-companion-results): URE evidence and sensitivity tables.
- [Reproduction](#reproduction): configurations, builders, and workflow guides.

The S-series filenames and the worksheet numbers in the supplementary Excel
workbook use separate numbering schemes. Match tables by title and content;
use the R-series guide to identify related CSV and reader tables.

## Table index

### Cohort and haplotype resource (S1–S9)

| Table | File | Topic |
|---|---|---|
| S1 | [S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv](S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv) | Cohort PCA coordinates |
| S2 | [S2_principal_component_coordinates_of_the_final_63_809_participant_cohort.csv](S2_principal_component_coordinates_of_the_final_63_809_participant_cohort.csv) | Cohort PCA coordinates |
| S3 | [S3_chromosome_22_phasing_switch_error_by_minor_allele_count.csv](S3_chromosome_22_phasing_switch_error_by_minor_allele_count.csv) | Phasing comparison |
| S4 | [S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv](S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv) | Cohort sources and population composition |
| S5 | [S5_coverage_of_uk_biobank_reference_haplotypes_by_the_selected_subset.csv](S5_coverage_of_uk_biobank_reference_haplotypes_by_the_selected_subset.csv) | Reference haplotype coverage |
| S6 | [S6_local_haplotype_diversity_after_sequential_cohort_integration.csv](S6_local_haplotype_diversity_after_sequential_cohort_integration.csv) | Haplotype diversity |
| S7 | [S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv](S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv) | Haplotype windows and sequence deduplication |
| S8 | [S8_rank_abundance_profile_of_distinct_haplotype_sequences.csv](S8_rank_abundance_profile_of_distinct_haplotype_sequences.csv) | Sequence rank abundance |
| S9 | [S9_haplotype_sequence_sharing_across_five_ancestry_groups.csv](S9_haplotype_sequence_sharing_across_five_ancestry_groups.csv) | Population-level haplotype sharing |

### Model-family summaries and genomic-element representations (S10–S16)

| Table | File | Topic |
|---|---|---|
| S10 | [S10_best_model_family_ranks_across_functional_variant_and_evolutionary_tasks.csv](S10_best_model_family_ranks_across_functional_variant_and_evolutionary_tasks.csv) | Model-family ranks |
| S11 | [S11_seed_level_representation_separation_for_encode_candidate_cis_regulatory_elements.csv](S11_seed_level_representation_separation_for_encode_candidate_cis_regulatory_elements.csv) | ENCODE representation-separation source data |
| S12 | [S12_summary_of_representation_separation_for_encode_candidate_cis_regulatory_elements.csv](S12_summary_of_representation_separation_for_encode_candidate_cis_regulatory_elements.csv) | ENCODE representation-separation summaries |
| S13 | [S13_seed_level_representation_separation_for_gencode_gene_structure_classes.csv](S13_seed_level_representation_separation_for_gencode_gene_structure_classes.csv) | GENCODE representation-separation source data |
| S14 | [S14_summary_of_representation_separation_for_gencode_gene_structure_classes.csv](S14_summary_of_representation_separation_for_gencode_gene_structure_classes.csv) | GENCODE representation-separation summaries |
| S15 | [S15_seed_level_representation_separation_for_curated_refseq_gene_structure_classes.csv](S15_seed_level_representation_separation_for_curated_refseq_gene_structure_classes.csv) | RefSeq representation-separation source data |
| S16 | [S16_summary_of_representation_separation_for_curated_refseq_gene_structure_classes.csv](S16_summary_of_representation_separation_for_curated_refseq_gene_structure_classes.csv) | RefSeq representation-separation summaries |

### Sequence classification and DeepSEA (S17–S21)

| Table | File | Topic |
|---|---|---|
| S17 | [S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv](S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv) | Nucleotide Transformer classification |
| S18 | [S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv](S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv) | Genomic Benchmark classification |
| S19 | [S19_binary_classification_across_five_ure_enhancer_control_sets.csv](S19_binary_classification_across_five_ure_enhancer_control_sets.csv) | URE binary classification |
| S20 | [S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv](S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv) | URE multiclass classification |
| S21 | [S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv](S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv) | DeepSEA multi-label evaluation |

### Variant benchmarks (S22–S32)

| Table | File | Topic |
|---|---|---|
| S22 | [S22_schematic_coordinates_for_variant_effect_and_allele_frequency_benchmarks.csv](S22_schematic_coordinates_for_variant_effect_and_allele_frequency_benchmarks.csv) | Variant benchmark coordinates |
| S23 | [S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv](S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv) | ClinVar classification |
| S24 | [S24_zero_shot_classification_of_coding_clinvar_variants.csv](S24_zero_shot_classification_of_coding_clinvar_variants.csv) | Coding ClinVar classification |
| S25 | [S25_zero_shot_classification_of_noncoding_clinvar_variants.csv](S25_zero_shot_classification_of_noncoding_clinvar_variants.csv) | Noncoding ClinVar classification |
| S26 | [S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv](S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv) | Candidate causal variant classification |
| S27 | [S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv](S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv) | Rare candidate causal variant classification |
| S28 | [S28_zero_shot_classification_of_uk_biobank_rare_variants.csv](S28_zero_shot_classification_of_uk_biobank_rare_variants.csv) | UK Biobank rare-variant classification |
| S29 | [S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv](S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv) | MPRA-supported variant classification |
| S30 | [S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv](S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv) | DDD and ClinVar-control classification |
| S31 | [S31_within_site_top_rank_discrimination_of_ddd_de_novo_alleles.csv](S31_within_site_top_rank_discrimination_of_ddd_de_novo_alleles.csv) | DDD within-site allele ranking |
| S32 | [S32_retrospective_enrichment_of_clinvar_reclassification_events_by_cutoff_year.csv](S32_retrospective_enrichment_of_clinvar_reclassification_events_by_cutoff_year.csv) | ClinVar reclassification event enrichment |

### Ancestry and evolutionary-region benchmarks (S33–S39)

| Table | File | Topic |
|---|---|---|
| S33 | [S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv](S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv) | Unsupervised ancestry metrics |
| S34 | [S34_supervised_classification_of_1000_genomes_ancestry_groups.csv](S34_supervised_classification_of_1000_genomes_ancestry_groups.csv) | Supervised ancestry classification |
| S35 | [S35_few_shot_regression_of_ancestry_proportions.csv](S35_few_shot_regression_of_ancestry_proportions.csv) | Ancestry-proportion regression |
| S36 | [S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv](S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv) | HAQER classification |
| S37 | [S37_classification_of_human_accelerated_regions_across_negative_controls.csv](S37_classification_of_human_accelerated_regions_across_negative_controls.csv) | HAR classification |
| S38 | [S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv](S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv) | Adaptive-allele model summaries |
| S39 | [S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv](S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv) | Adaptive-allele locus scores |
### Training conditions and computational resources (S41–S44)

| Table | File | Topic |
|---|---|---|
| S41 | [S41_effects_of_training_corpus_and_windowing_on_18_sequence_classification_tasks.csv](S41_effects_of_training_corpus_and_windowing_on_18_sequence_classification_tasks.csv) | Training-corpus and windowing conditions |
| S42 | [S42_effects_of_corpus_windowing_and_allelic_refinement_on_clinvar_scoring.csv](S42_effects_of_corpus_windowing_and_allelic_refinement_on_clinvar_scoring.csv) | ClinVar scoring conditions |
| S43 | [S43_model_level_prediction_performance_and_computational_resource_use.csv](S43_model_level_prediction_performance_and_computational_resource_use.csv) | Model-level performance, 45-task mean rank/coverage, and arithmetic-mean resource metrics across 18 NT tasks |
| S44 | [S44_task_level_prediction_performance_and_computational_resource_use.csv](S44_task_level_prediction_performance_and_computational_resource_use.csv) | Task-level performance and resource metrics, with source paths and hashes |

### Dataset, model, and protocol metadata (S45–S50)

| Table | File | Topic |
|---|---|---|
| S45 | [S45_overview_of_datasets_labels_and_benchmark_sample_sizes.csv](S45_overview_of_datasets_labels_and_benchmark_sample_sizes.csv) | Dataset and task definitions |
| S46 | [S46_architecture_and_hyperparameters_of_the_primary_hevo_models.csv](S46_architecture_and_hyperparameters_of_the_primary_hevo_models.csv) | hEvo model configurations |
| S47 | [S47_architecture_training_scope_and_provenance_of_evaluated_models.csv](S47_architecture_training_scope_and_provenance_of_evaluated_models.csv) | Evaluated-model architecture, training scope, and parameter counts |
| S48 | [S48_parameters_used_across_model_evaluation_protocols.csv](S48_parameters_used_across_model_evaluation_protocols.csv) | Evaluation protocol parameters |
| S49 | [S49_model_task_level_scores_and_ranks_for_fig5.csv](S49_model_task_level_scores_and_ranks_for_fig5.csv) | Task-level model scores and ranks across benchmark tasks |
| S50 | [S50_data_sources_and_evidence_categories_used_to_construct_the_ure_benchmark_datasets.csv](S50_data_sources_and_evidence_categories_used_to_construct_the_ure_benchmark_datasets.csv) | URE benchmark data sources and evidence categories |

## Data definitions

### Sample identifiers and cohort ancestry

S1 and S2 use `IID` sample identifiers for cross-table matching. R01 and R02
use file-local `Record` numbers, which are separate from those sample IDs.

In S2, `Group` identifies the source cohort; `Ancestry` and
`Ancestry_label_source` describe the ancestry annotation and its basis.
S4 and R02/R28 summarize annotations from the S1 six-PC classifier, with all
3,188 internal Chinese cohort participants annotated as EAS. The
[ancestry workflow](../scripts/ancestry_harmonization/README.md) documents the method.

### Genomic-element representations (S11–S16)

These tables contain seed-level and summary Calinski–Harabasz (CH) results for
ENCODE, GENCODE, and RefSeq annotations. CH is the reported separation metric.
The three analyses appear in Fig. 2c.

### hEvo architecture and training settings (S46)

S46 contains architecture and shared training hyperparameters for four hEvo
checkpoints. Public-resource variants use the corresponding scale-specific
settings. Trainer optimizer and scheduler fields describe Trainer arguments;
DeepSpeed configuration can override them. Checkpoint `dtype` is distinct from
`stage1_precision`.

The companion [training-budget table](S46_training_run_budgets.csv) contains
nine Stage I configuration entries. Step and sequence budgets are independent
fields, including for hEvo-open-base and hEvo-open-large. Nominal nucleotide
positions equal the sequence budget multiplied by 1,024.

### Evaluated models and parameter counts (S47)

S47 contains 34 model records: 31 single-sequence models and three GPN-Star
MSA models. S23 and S26–S30 contain 33 models; DNABERT is registered in S47
but does not occur in those result tables.

Fields cover tokenization, architecture, parameter counts, training scope,
literature, inference interfaces, embedding readouts, runtime defaults,
checkpoint and tokenizer identities, licenses, and MSA metadata.
`model_lm_type` describes the inference interface and differs from
`training_objective`. `NA` denotes an unrecorded source field; the three
Mistral-DNA records use `NA` for their training objective.

Sources include `models/models.tsv` in the configured source project,
[`models.yaml`](../configs/models.yaml), [`gpn_star.yaml`](../configs/gpn_star.yaml),
and pinned checkpoint metadata.

Parameter counts include unique stored parameters, counting shared tensor
aliases once:

| Checkpoints | Parameters |
|---|---:|
| hEvo-base and hEvo-open-base | 87,094,922 each |
| hEvo-large and hEvo-open-large | 319,462,410 each |

S43 and S44 report total predictor parameters: inference backbone plus the
task-specific probe. For GPN-Star, these counts exclude the unused masked-LM
output head; S47 records the full 203,022,215-parameter checkpoint.

### Evaluation protocols (S48)

S48 contains 12 configuration items across six protocol columns: sequence
probe classification, paired-sequence variant evaluation, DeepSEA LoRA,
adaptive-allele preference, adaptive-allele locus scoring, and ancestry
embedding analysis.

The definitions distinguish task-directed paired-variant LLR, native
Mistral-DNA pooled-logit L2, GENERator `log(P_ref/P_alt)`, and coordinate-aware
GPN-Star MSA scoring. Adaptive-allele preference uses its own protocol.

## Figure inputs and aggregation

### Task counts and cross-task ranks (S49 / Fig. 5)

The benchmark distinguishes 48 manuscript task endpoints, 54 configured task
entries in [`tasks.yaml`](../configs/tasks.yaml), and 45 ranking units in S49.
S49 contains 1,523 model/task rows and uses minimum ranks for ties, with higher
scores ranked better.

| Task group | Ranking units | Source |
|---|---:|---|
| Nucleotide Transformer | 18 | S17 |
| Genomic Benchmark, human-coordinate tasks | 5 | Eligible S18 rows |
| URE binary classification | 5 | S19 |
| URE multiclass classification | 5 | S20 |
| ClinVar, all variants | 1 | S23 |
| Other variant tasks | 6 | S26–S31 |
| Ancestry analyses | 3 | S33–S35 |
| HAR and HAQER | 2 | S36–S37 |

URE multiclass scores average class-specific F1 within each dataset. Ancestry
regression averages the 10–60-shot sweep; HAR and HAQER average five
negative-control runs. Each of
GPN-Star-M, GPN-Star-P, and GPN-Star-V covers all 45 ranking units.

The two S18 tasks `demo_human_or_worm` and
`demo_coding_vs_intergenomic_seqs` are excluded from this ranking protocol.
S24 and S25 are ClinVar subsets and are not ranked separately from S23.
S32 reports event enrichment rather than a fixed task-level score;
S21 is a separate LoRA evaluation.

### Computational resources (S43–S44)

S43 and S44 cover 32 models across 18 NT tasks. S43 combines
45-task mean rank and coverage with arithmetic-mean resource metrics across
the 18 NT tasks; S44 provides the task-level records.

In S43, `missing_tasks=none` means every NT task has a recorded MCC. In S44,
`missing_efficiency_fields=none` means parameter count, throughput, latency,
and VRAM are present. Accuracy, AUROC, and their 95% confidence intervals are
populated for all 576 model/task rows. FLOPs remain `NA` where the source did not
report them.

### Other figure selections

| Figure or analysis | Inputs and selection |
|---|---|
| Model-family radar | S10; best eligible checkpoint selected separately for each axis |
| DeepSEA / FigS6 | S21; hEvo-open-large S21_run_24 alongside hEvo-open-base |
| Ancestry / FigS8 | S33–S35, shot=60 where applicable; 32 models, excluding hEvo-open-base and hEvo-open-large |
| Evolution / FigS9 | S36–S39; 31 models, excluding hEvo-open-base, hEvo-open-large, and DNABERT |
| HAQER tracks / Fig. 4e | [Track source data](fig4e_haqer_tracks/Fig4e_track_source_data.csv); hEvo-large and GPN-Star-M |

S21 includes GPN-Star-M LoRA evaluation S21_run_23 (seed 42; DHS/HM/TF),
using 1,000-bp native-MSA inputs. Dedicated fields record train/test coordinate
mapping counts and coverage.

S21 stores evaluation metrics by model, run, and target group. The
`scripts/DeepSEA_lora/README.md` documents the evaluation and export workflow;
`scripts/DeepSEA_lora/export_s21_lora_source_table.py` rebuilds the public S21
table from its base table, run outputs, and run metadata.

## Sources and companion results

### URE source evidence (S50)

S50 records 47 URE data sources and evidence categories accompanying the
S19/S20 benchmarks.

### Sensitivity analyses

The following companion results are in `revision_consistency/`:

| File | Content |
|---|---|
| [common_task_rank_sensitivity.csv](revision_consistency/common_task_rank_sensitivity.csv) | Ranks on the common-task intersection |
| [cllr_sensitivity_summary.csv](revision_consistency/cllr_sensitivity_summary.csv) | Paired raw and mutation-rate-calibrated GPN-Star summaries |
| [retrospective_cllr_by_seed.csv](revision_consistency/retrospective_cllr_by_seed.csv) | Retrospective calibration by seed |
| [retrospective_cllr_summary.csv](revision_consistency/retrospective_cllr_summary.csv) | Aggregated retrospective calibration |

These CSVs use portable `source://`, `runs://`, `assets://`, and `output://`
provenance labels and input hashes.

## Reproduction

Run commands from `hEvo_benchmark` unless the linked guide specifies otherwise.

| Purpose | Entry point |
|---|---|
| Configure benchmark tasks | [Task configuration](../configs/tasks.yaml) |
| Run evaluations and exporters | [Workflow guide](../scripts/README.md) |
| Render figures from CSVs | [Figure guide](../figures/README.md) |
| Build dataset overview S45 | [build_dataset_supplementary_table.py](scripts/build_dataset_supplementary_table.py) |
| Build Fig. 5 task ranks S49 | [build_fig5_cross_task_ranks.py](scripts/build_fig5_cross_task_ranks.py) |
| Build resource summaries S43–S44 | [build_pareto_supplementary_tables.py](scripts/build_pareto_supplementary_tables.py) |
| Build model-family summaries | [build_fig1_family_radar.py](scripts/build_fig1_family_radar.py) |
| Build model and evaluation tables S46–S48 | [build_model_supplementary_tables.py](scripts/build_model_supplementary_tables.py) |
| Reconcile the publication workbook | [build_publication_supplementary_tables.py](scripts/build_publication_supplementary_tables.py) |
| Export available S tables to a workbook | [build_supplementary_tables_excel.py](scripts/build_supplementary_tables_excel.py) |
| Select active DeepSEA runs for derived summaries and figures | [deepsea_run_selection.py](scripts/deepsea_run_selection.py) |
| Synchronize training settings and budgets | [sync_hevo_training_tables.py](scripts/sync_hevo_training_tables.py) |
| Verify the training-table synchronizer | [test_sync_hevo_training_tables.py](scripts/test_sync_hevo_training_tables.py) |

To synchronize the supplementary workbook's training-configuration sheet S30
with S46/R19 and their budget companions, run:

```bash
python tables/scripts/sync_hevo_training_tables.py
```

Public-resource step and sequence budgets can be entered independently in the
workbook or budget CSV. The synchronizer preserves existing entries and rejects
conflicting filled values.

[Back to parent guide](../README.md)
