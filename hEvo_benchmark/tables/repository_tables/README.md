# Reader-facing repository tables

This directory contains the annotated R01–R28 tables. Use these files to
inspect or reuse table data: each TSV includes table-level notes and a
`## Column n | name | definition` comment for every field.

The parent directory contains S-series CSV figure inputs. The crosswalk below
links each R table to its corresponding source tables.

## Reading the TSV files

- Ignore comment lines beginning with `##`; the first non-comment line is the
  tab-separated header.
- `NA` denotes unavailable or inapplicable values.
- `Record` numbers are local to each R file. S1/S2 retain the original `IID`
  sample identifiers for figure joins; R-file record numbers are separate.
- Numeric values use source-table precision.

## R-to-S crosswalk

| Reader table | Rows | Columns | Related S table or figure | Relationship and interpretation |
|---|---:|---:|---|---|
| [R01](R01_prescreening_genetic_principal_components.tsv) | 550,237 | 12 | S1 | Same PC coordinate values, with reader-facing `Record` and `Source cohort` fields. The 10 PC columns match S1 row by row. |
| [R02](R02_training_resource_genetic_principal_components.tsv) | 63,809 | 6 | S2 | Coordinates and file-local records are provided; source cohort and five-category ancestry are separate fields. |
| [R03](R03_phasing_switch_errors_by_minor_allele_count.tsv) | 9 | 5 | S3 | Chromosome 22 phasing switch-error summary by minor allele count. |
| [R04](R04_reference_haplotype_coverage.tsv) | 606 | 6 | S5 | Reader-annotated view of the 606 reference-haplotype coverage records. Frequency is a fraction; the two coverage measures have different denominators. |
| [R05](R05_local_diversity_during_cohort_integration.tsv) | 9 | 7 | S6 | Reader-facing cohort-integration summary across nine scenarios; see the R05 notes for the sampling and reference baseline. |
| [R06](R06_distinct_sequence_abundance_profile.tsv) | 14,801 | 2 | S8 | Rank-abundance profile containing 14,801 records. |
| [R07](R07_ancestry_sharing_of_distinct_sequences.tsv) | 31 | 4 | S9 | Sequence-sharing counts across the ancestry categories in S9. |
| [R08](R08_family_level_benchmark_summaries.tsv) | 52 | 8 | S10 | Family-level view of the 13 radar axes. The best eligible checkpoint is selected separately for each axis. |
| [R09](R09_encode_ccre_representation_separation.tsv) | 204 | 11 | S11–S12 | ENCODE cCRE seed-level separation and within-model summaries. |
| [R10](R10_gencode_representation_separation.tsv) | 204 | 11 | S13–S14 | GENCODE seed-level separation and within-model summaries. |
| [R11](R11_refseq_representation_separation.tsv) | 204 | 11 | S15–S16 | RefSeq seed-level separation and within-model summaries. |
| [R12](R12_conceptual_placement_of_variant_benchmarks.tsv) | 6 | 5 | S22 | Biological definitions for six schematic benchmark placements. |
| [R13](R13_within_position_discrimination_of_ddd_alternative_alleles.tsv) | 33 | 8 | S31 | Within-position DDD alternative-allele comparison. |
| [R14](R14_adaptive_allele_scores_at_individual_loci.tsv) | 990 | 12 | S39 | Per-locus native log probabilities and selected-versus-other score differences. |
| [R15](R15_direction_independent_adaptive_allele_score_separation.tsv) | 33 | 3 | Related adaptive-allele analysis; distinct from S38's directional preference | One-dimensional CH separation of selected and alternative allele scores. |
| [R16](R16_published_haqer_contexts_displayed_in_figure_4.tsv) | 12 | 9 | Fig. 4e reader table | Twelve HAQER illustration contexts with coordinates, split, gene context, and literature reference. |
| [R17](R17_haqer_illustration_track_values.tsv) | 3,047 | 11 | Fig. 4e reader table | Track values for the illustrated intervals; the current renderer reads its dedicated source bundle under `tables/fig4e_haqer_tracks/`. |
| [R18](R18_benchmark_dataset_inventory.tsv) | 62 | 15 | S45 | Reader-facing inventory of 62 dataset records. Cohort, sample, and sequence-record counts are different units. |
| [R19](R19_hevo_architecture_and_training_configuration.tsv) | 4 | 49 | S46 | Architecture and shared Stage I/II training settings for four hEvo checkpoints; [run-budget companion](R19_training_run_budgets.tsv) contains nine Stage I configuration entries, with independent budget fields for the two public-resource models. |
| [R20](R20_benchmark_evaluation_protocols.tsv) | 12 | 7 | S48 | Definitions and parameters for 12 evaluation protocols. |
| [R21](R21_model_checkpoint_and_alignment_provenance.tsv) | 34 | 10 | S47 | Model checkpoint and alignment provenance. |
| [R22](R22_clinvar_ablation_records.tsv) | 72 | 12 | S42 | Current ClinVar ablation estimates and confidence-interval endpoints, copied from S42 with scored class counts and evaluation seed. Existing source strata are retained; manuscript overall analyses use `all`. |
| [R23](R23_task_scores_and_ranks_underlying_figure_5.tsv) | 1,523 | 9 | S49 | Scores and ranks for 45 ranking units. |
| [R24](R24_deepsea_adaptation_settings_and_run_history.tsv) | 69 | 15 | S21 | DeepSEA adaptation settings and evaluation records. |
| [R25](R25_genomic_benchmark_demo_task_alignment_records.tsv) | 2 | 6 | S18 task records; excluded from S49 ranking | Two source-export demo-task records outside the five-task human-coordinate evaluation and S49 ranks. `demo_human_or_worm` mixes human and worm sequences; `demo_coding_vs_intergenomic_seqs` compares coding and intergenic sequences. |
| [R26](R26_detailed_task_level_resource_and_metric_records.tsv) | 576 | 22 | S44 | 576 task-level performance and resource records. |
| [R27](R27_gpn_star_calibration_across_variant_and_adaptive_allele_endpoints.tsv) | 39 | 15 | S23, S26–S31, S38–S39; calibration sensitivity records | Matched-subset raw and mutation-rate-calibrated sensitivity results. |
| [R28](R28_cohort_ancestry_composition.tsv) | 9 | 11 | S4 | Cohort ancestry counts and Others breakdown, consistent with the current sample labels in R02. |

[Back to parent guide](../README.md)
