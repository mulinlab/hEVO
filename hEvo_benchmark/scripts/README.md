# Benchmark scripts

Run the examples below from the **repository root**. Python >=3.10 is required;
use a scientific Python environment with numpy, pandas, scipy, scikit-learn,
h5py, PyYAML and matplotlib for CPU analysis. Inference additionally
requires PyTorch/Transformers and model-specific packages/checkpoints. Slurm,
PLINK, ADMIXTURE, bcftools, R and C++ requirements depend on the workflow.
The examples use the `llmbm` environment. Document editing additionally needs lxml/python-docx.

## Choose a benchmark

| Purpose | Entry / workflow guide | Source tables |
|---|---|---|
| NT, Genomic Benchmark, URE classification | [Shared evaluators](evaluate/README.md) | S17–S20 |
| DeepSEA multilabel LoRA | [DeepSEA](DeepSEA_lora/README.md) | S21 |
| ClinVar and causal/rare/MPRA/DDD variants | [Shared evaluators](evaluate/README.md) | S23–S31 |
| ClinVar coding/noncoding reconstruction | [ClinVar](clinvar/README.md) | S24–S25 |
| GPN-Star non-ClinVar preparation/export | [GPN-Star variants](evaluate/gpn_star_variant/README.md) | S26–S31 |
| Variant controls and within-site ranking | [Fair variant evaluation](fair_variant_evaluation/README.md) | S26–S31 |
| Ancestry separation/classification/few-shot regression | [Population classification](population_classify/README.md) | S33–S35 |
| HAR/HAQER classification | [Shared evaluators](evaluate/README.md) | S36–S37 |
| HAR/HAQER GPN-Star-M genomic track and peak workflow | [HAR/HAQER](har_haqer_workflow/README.md), [core](har_haqer_workflow/core/README.md) | — |
| Adaptive allele preference | [Allele preference](adaptive_allele_preference_benchmark/README.md), [GPN raw stage](adaptive_allele_preference_benchmark/fig4_gpn_raw_stage/README.md) | S38–S39 |
| Pareto/resource evaluation | [Shared evaluators](evaluate/README.md) | S10, S43–S44, S49 |
| Ablations and calibrated sensitivity | [Sensitivity analysis](revision_consistency/README.md) | S41–S42, R27, sensitivity tables |
| Evaluator and generator tests | [Test guide](tests/README.md) | — |
| Generate probe/variant/ancestry Slurm jobs | [Job generation](create_slurm_scripts/README.md) | Scheduler support |

[Benchmark inputs and outputs](benchmark_coverage.tsv) lists tasks, entry
points and result tables. Representation-separation tools are in
[`evaluation_elements/`](evaluation_elements/README.md), and result-table
builders are in [`../tables/scripts/`](../tables/scripts/).

## Supporting workflows

| Workflow | Guide |
|---|---|
| Final S1 six-PC cohort ancestry annotation and coordinate plots | [Ancestry harmonization](ancestry_harmonization/README.md) |
| Phasing comparison | [Phasing](phasing/README.md) |
| Conventional ancestry estimates | [Ancestry analysis](ancestry_analysis/README.md) |
| Haplotype diversity and UKB reference coverage | [Haplotype diversity](haplotype_diversity_workflow/README.md) |
| Cohort-aware preprocessing and statistics | [Restricted source code on HF](https://huggingface.co/MulinLab/hEvo-pretraining-code/tree/main/hEvo_benchmark/scripts/preprocess_stats) · [Public result tables](../tables/preprocess_stats/README.md) |

## Calling and configuration

Use the selected workflow's CLI to configure paths. Configure
[project paths](../configs/project.yaml), [model paths](../configs/models.yaml)
and [task columns](../configs/tasks.yaml). Use inputs that match the selected
task schema and protocol. See [evaluation help](evaluate/README.md) before
choosing the standard probe, Pareto probe or native variant scorer.

```bash
python hEvo_benchmark/scripts/evaluate/probe_single_seq_input.py --help
python hEvo_benchmark/scripts/evaluate/zeroshot_paried_seq_input.py --help
python hEvo_benchmark/scripts/create_slurm_scripts/create_probe_single_seq.py --help
```

The main probe/paired-variant/DeepSEA training/category-evaluation entry points
show help before importing model runtimes. Other analysis tools require their
workflow's CPU dependencies even for help. Job generators create scripts; the
separate Slurm submission scripts submit jobs. Table exporters/finalizers write tables
or figures and should receive explicit run/output locations when reviewing a run.


[Source tables](../tables/README.md) · [Annotated results](../tables/repository_tables/README.md)
