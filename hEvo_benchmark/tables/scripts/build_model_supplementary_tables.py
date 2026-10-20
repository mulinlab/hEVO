#!/usr/bin/env python3
"""Build model and evaluation-parameter supplementary tables S46-S48."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml


GITLAB_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("SOURCE_PROJECT_ROOT", str(GITLAB_ROOT)))
TABLES_DIR = GITLAB_ROOT / "tables"

HEVO_DIRS = {
    "hEvo-base": "hEvo-base",
    "hEvo-large": "hEvo-large",
    "hEvo-open-base": "hEvo-open-base",
    "hEvo-open-large": "hEvo-open-large",
}

HEVO_ROSTER_MODELS = set(HEVO_DIRS) | {"hEvo-open-base", "hEvo-open-large"}
PRESERVED_ROSTER_MODELS = HEVO_ROSTER_MODELS | {"GPN-Star-M", "GPN-Star-P", "GPN-Star-V"}
CHECKPOINT_PARAMETER_COUNTS = {
    "hEvo-base": 87_094_922,
    "hEvo-large": 319_462_410,
    "hEvo-open-base": 87_094_922,
    "hEvo-open-large": 319_462_410,
}
PRIMARY_SLIDING = ["hEvo-base", "hEvo-large", "hEvo-open-base", "hEvo-open-large"]
NATIVE_CAUSAL_MODEL_IDS = {
    "Mistral-DNA-17M",
    "Mistral-DNA-138M",
    "Mistral-DNA-422M",
    "GENERator-1.2B",
    "GENERator-3B",
}

HEVO_METADATA = {
    "hEvo-base": {
        "training_cohort": "final 60k cohort",
        "haplotype_window_bp": 1024,
        "sliding_stride_bp": 128,
        "model_size_label": "87.095M",
        "primary_display_model": True,
    },
    "hEvo-large": {
        "training_cohort": "final 60k cohort",
        "haplotype_window_bp": 1024,
        "sliding_stride_bp": 128,
        "model_size_label": "319.462M",
        "primary_display_model": True,
    },
}

for public_model, full_model in (("hEvo-open-base", "hEvo-base"), ("hEvo-open-large", "hEvo-large")):
    HEVO_METADATA[public_model] = {
        **HEVO_METADATA[full_model],
        "training_cohort": "1000 Genomes Project + HGDP",
        "primary_display_model": False,
    }


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_value(config: dict, *keys: str):
    for key in keys:
        if key in config:
            return config[key]
    return ""


def hevo_row(model: str) -> dict:
    model_dir = GITLAB_ROOT / "models" / HEVO_DIRS[model]
    config = read_json(model_dir / "config.json")
    metadata = HEVO_METADATA[model]
    public_model_dir = f"<benchmark-root>/models/{HEVO_DIRS[model]}"
    return {
        "model": model,
        "model_directory": public_model_dir,
        "training_cohort": metadata["training_cohort"],
        "haplotype_window_bp": metadata["haplotype_window_bp"],
        "sliding_stride_bp": metadata["sliding_stride_bp"],
        "tokenization": "1-mer",
        "model_size_label": metadata["model_size_label"],
        "architectures": "|".join(config.get("architectures", [])),
        "hidden_size": config.get("hidden_size", ""),
        "num_hidden_layers": config.get("num_hidden_layers", ""),
        "num_attention_heads": config.get("num_attention_heads", ""),
        "intermediate_size": config.get("intermediate_size", ""),
        "max_position_embeddings": config.get("max_position_embeddings", ""),
        "vocab_size": config.get("vocab_size", ""),
        "type_vocab_size": config.get("type_vocab_size", ""),
        "hidden_act": config.get("hidden_act", ""),
        "attention_dropout": config.get("attention_probs_dropout_prob", ""),
        "hidden_dropout": config.get("hidden_dropout_prob", ""),
        "dtype": config_value(config, "dtype", "torch_type", "torch_dtype"),
        "attention_implementation": config.get("attn_implementation", ""),
        "use_flash_attention": config.get("use_flash_attention", ""),
        "use_frequency_embedding": config.get("use_frequency_embedding", ""),
        "frequency_embedding_dim": config.get("frequency_embedding_dim", ""),
        "problem_type": config.get("problem_type", ""),
        "primary_display_model": metadata["primary_display_model"],
        "config_source": f"{public_model_dir}/config.json",
    }


def build_s46() -> pd.DataFrame:
    rows = [hevo_row(model) for model in PRIMARY_SLIDING]
    snapshot_path = TABLES_DIR / "S46_architecture_and_hyperparameters_of_the_primary_hevo_models.csv"
    if snapshot_path.exists():
        snapshot = pd.read_csv(snapshot_path, keep_default_na=False)
        if snapshot["model"].duplicated().any():
            raise ValueError("S46 contains duplicate model identifiers")
        indexed = snapshot.set_index("model")
        architecture_fields = set(rows[0])
        extra_fields = [column for column in snapshot.columns if column not in architecture_fields]
        for row in rows:
            if row["model"] in indexed.index:
                row.update({column: indexed.loc[row["model"], column] for column in extra_fields})
    return pd.DataFrame(rows)


def apply_checkpoint_parameter_counts(roster: pd.DataFrame) -> pd.DataFrame:
    if "model" not in roster.columns:
        return roster
    for model, count in CHECKPOINT_PARAMETER_COUNTS.items():
        mask = roster["model"].eq(model)
        if "parameter_count" in roster.columns:
            roster.loc[mask, "parameter_count"] = f"{count / 1_000_000:.3f}M"
        if "parameter_count_exact" in roster.columns:
            roster.loc[mask, "parameter_count_exact"] = count
    return roster


def apply_native_causal_interfaces(roster: pd.DataFrame) -> pd.DataFrame:
    """Record the implemented causal inference interface for native scorers."""
    if "model" not in roster.columns or "model_lm_type" not in roster.columns:
        return roster
    roster = roster.copy()
    mask = roster["model"].isin(NATIVE_CAUSAL_MODEL_IDS)
    roster.loc[mask, "model_lm_type"] = "causal"
    return roster


def normalize_model_roster_metadata(roster: pd.DataFrame) -> pd.DataFrame:
    """Use NA where cited source materials contain no value for a metadata field."""
    roster = roster.copy()
    replacements = {
        "预印": "Preprint",
        "Not reported": "NA",
        "Not specified in cited model/checkpoint documentation.": "NA",
        "NOT_VERIFIABLE": "NA",
        "NOT_AVAILABLE": "NA",
    }
    for column in roster.columns:
        roster[column] = roster[column].map(
            lambda value: replacements.get(value.strip(), value.strip())
            if isinstance(value, str)
            else value
        )
        roster[column] = roster[column].replace("", "NA")
    roster = roster.fillna("NA")
    if "venue" in roster.columns:
        roster["venue"] = roster["venue"].replace(
            {
                "nature machine intelligence": "Nature Machine Intelligence",
                "Nature methods": "Nature Methods",
                "Nucleic Acids Research,": "Nucleic Acids Research",
            }
        )
    return roster


def build_model_roster() -> pd.DataFrame:
    source_path = PROJECT_ROOT / "models" / "models.tsv"
    existing_path = TABLES_DIR / "S47_architecture_training_scope_and_provenance_of_evaluated_models.csv"
    if not source_path.exists():
        if not existing_path.exists():
            raise FileNotFoundError(
                f"Neither the source model roster ({source_path}) nor the checked-in S47 table ({existing_path}) exists"
            )
        # A clone without the external source-project tree can still reproduce
        # the checked-in roster with the current field set.
        snapshot = pd.read_csv(existing_path)
        snapshot = apply_checkpoint_parameter_counts(snapshot)
        snapshot = apply_native_causal_interfaces(snapshot).sort_values("model").reset_index(drop=True)
        return normalize_model_roster_metadata(snapshot)

    info = pd.read_csv(source_path, sep="\t")
    info = info.rename(
        columns={
            "模型名": "model",
            "架构": "architecture",
            "参数量": "parameter_count",
            "物种": "training_scope",
            "发表/预印日期": "publication_or_preprint_date",
            "期刊": "venue",
            "文献URL": "literature_url",
        }
    )
    config = yaml.safe_load((GITLAB_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    model_cfg = pd.DataFrame(config["Models"])
    model_cfg = model_cfg.rename(
        columns={
            "name": "model",
            "url": "configured_model_path",
            "type": "model_lm_type",
            "base": "training_objective",
            "future": "embedding_readout",
            "batch_size": "default_batch_size",
            "max_length": "default_max_length",
        }
    )
    for frame in (info, model_cfg):
        nonprimary = frame["model"].astype(str).str.startswith("hEvo-") & ~frame["model"].isin(HEVO_ROSTER_MODELS)
        frame.drop(frame.index[nonprimary], inplace=True)
    merged = info.merge(model_cfg, on="model", how="outer", validate="one_to_one")
    ordered_cols = [
        "model",
        "Tokenization",
        "architecture",
        "parameter_count",
        "training_scope",
        "publication_or_preprint_date",
        "venue",
        "literature_url",
        "model_lm_type",
        "training_objective",
        "embedding_readout",
        "default_batch_size",
        "default_max_length",
        "configured_model_path",
    ]
    roster = merged[ordered_cols].sort_values("model").reset_index(drop=True)

    # Preserve curated source and weight metadata by model name. Also retain
    # public-data hEvo releases and GPN-Star checkpoints if the external model
    # registry omits them.
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        if "model" in existing.columns:
            preserved = existing[
                existing["model"].isin(PRESERVED_ROSTER_MODELS)
                & ~existing["model"].isin(roster["model"])
            ].reindex(columns=ordered_cols)
            if not preserved.empty:
                roster = pd.concat([roster, preserved], ignore_index=True, sort=False)
            curated_cols = [
                col for col in existing.columns if col not in ordered_cols
            ]
            if curated_cols:
                curated = existing[["model", *curated_cols]].drop_duplicates("model")
                roster = roster.merge(curated, on="model", how="left", validate="one_to_one")
    roster = apply_checkpoint_parameter_counts(roster)
    roster = apply_native_causal_interfaces(roster).sort_values("model").reset_index(drop=True)
    return normalize_model_roster_metadata(roster)


def build_evaluation_parameters() -> pd.DataFrame:
    rows = [
        {
            "evaluation_framework": "Single-sequence probe classification",
            "benchmark_tables_or_tasks": "S17 NT Benchmark; S18 Genomic Benchmark; S19 URE binary classification; S20 URE multiclass classification",
            "input_unit": "single DNA sequence with class label",
            "input_files_and_columns": "single-sequence records with task-defined sequence and label columns",
            "train_valid_test_split": "predefined train/test partition; an internal split may be used when only one source file is supplied",
            "model_representation": "precomputed sequence embedding",
            "pooling_or_score": "pooling=mean",
            "sequence_length": "configured task sequence length; current tasks use 512- or 1,024-bp inputs where applicable",
            "training_or_cv": "linear-probe classification; classifier settings are part of the task configuration",
            "uncertainty_or_seed": "bootstrap=1000; seed=40",
            "metrics": "accuracy, macro/micro F1, MCC, AUROC when applicable",
            "configuration_note": "Same probe framework and effective wrapper settings are shared by NT, Genomic Benchmark, and URE sequence-classification tasks.",
        },
        {
            "evaluation_framework": "Zero-shot paired-sequence variant evaluation",
            "benchmark_tables_or_tasks": "S23-S30 variant AUPRC tasks; S31 paired DDD same-position other-alt top-1 task; S32 dynamic ClinVar event-centric enrichment uses zero-shot scores",
            "input_unit": "reference and alternate DNA sequence pair with binary label",
            "input_files_and_columns": "paired records with reference, alternate, label, and optional coordinate columns",
            "train_valid_test_split": "zero-shot evaluation; no probe training split",
            "model_representation": "standard masked/causal-compatible LLR inputs; native Mistral-DNA and GENERator variant scores; coordinate-aware GPN-Star MSA inputs",
            "pooling_or_score": "standard task-directed LLR; Mistral-DNA: L2 distance between max-pooled vocabulary logits; GENERator: log(P_ref/P_alt) from hg38 upstream 6-mer base-marginal probabilities; GPN-Star: raw, uncalibrated native-MSA VEP LLR with prespecified task direction; cLLR is a separate sensitivity analysis",
            "sequence_length": "paired variant contexts up to 1,024 bp; Mistral-DNA uses 1,023 bp inputs (512-token model limit); GENERator uses up to 1,024 upstream hg38 bases aligned to 6-mers; GPN-Star uses a 1,024 bp coordinate-aware MSA window",
            "training_or_cv": "no learned classifier; direct standard LLR or model-native zero-shot variant scoring",
            "uncertainty_or_seed": "bootstrap=1000; seed=42",
            "metrics": "AUROC, AUPRC, precision-recall/ROC curves and bootstrap confidence intervals where enabled",
            "configuration_note": "The shared evaluator routes Mistral-DNA and GENERator to their native paired-variant scorers and GPN-Star to its coordinate-aware native-MSA scorer. Other compatible models use the configured standard LLR path. Mistral-DNA reports an effect distance. GENERator stores REF-minus-ALT log probabilities; use that score directly for disruption ranking and its absolute value for MPRA. GPN-MSA uses its coordinate-aware native MSA evaluator. The adaptive allele-preference tasks retain their separate protocol.",
        },
        {
            "evaluation_framework": "DeepSEA multi-label LoRA evaluation",
            "benchmark_tables_or_tasks": "S21 DeepSEA LoRA task groups: DHS, HM, and TF",
            "input_unit": "DeepSEA sequence with multi-label functional annotations",
            "input_files_and_columns": "DeepSEA sequence records with multi-label functional annotations and explicit splits",
            "train_valid_test_split": "uses explicit train, valid, and test splits",
            "model_representation": "sequence model with a supervised multi-label DeepSEA prediction head and LoRA adaptation",
            "pooling_or_score": "model-specific sequence representation with supervised multi-label logits",
            "sequence_length": "run configuration is recorded per model; current source table contains 512-bp and 1,024-bp runs",
            "training_or_cv": "DeepSEA LoRA adaptation with explicit train/valid/test splits",
            "uncertainty_or_seed": "source table preserves run_id and seed fields where available",
            "metrics": "DHS, HM and TF category AUROC/AUPRC summaries",
            "configuration_note": "S21 retains run-level task metrics and protocol/source provenance; runtime outputs are external to the public release.",
        },
        {
            "evaluation_framework": "Adaptive allele-preference evaluation",
            "benchmark_tables_or_tasks": "S38 30-locus adaptive allele-preference model summary",
            "input_unit": "30 prespecified biallelic SNVs with literature-specified selected and other alleles",
            "input_files_and_columns": "paired selected/other sequences and locus metadata; S39 contains the locus scores",
            "train_valid_test_split": "no learned classifier or train/test split; prespecified locus roster",
            "model_representation": "masked conditional nucleotide log probabilities or causal sequence log likelihoods; native GPN-MSA allele scores",
            "pooling_or_score": "selected-allele score minus other-allele score; positive values indicate directional concordance",
            "sequence_length": "model-specific paired context recorded by window_bp, center_index and tokenized_length in S39",
            "training_or_cv": "no classifier training and no cross-validation; direct paired allele scoring",
            "uncertainty_or_seed": "95% Clopper-Pearson interval for positive loci / evaluated loci; not training-seed uncertainty",
            "metrics": "positive_loci, n_loci, directional preference proportion and exact binomial CI; descriptive score summaries",
            "configuration_note": "S38 summarizes the prespecified 30-locus allele-preference analysis; directional agreement does not establish fitness or selection coefficients",
        },
        {
            "evaluation_framework": "Adaptive allele-preference locus-score evaluation",
            "benchmark_tables_or_tasks": "S39 30-locus adaptive allele-preference locus scores",
            "input_unit": "30 prespecified biallelic SNVs with literature-specified selected and other alleles",
            "input_files_and_columns": "S39: variant_id, chrom, pos, selected_allele, other_allele, selected_logprob, other_logprob, locus_score, directional_correct, score_definition, benchmark_status",
            "train_valid_test_split": "no learned classifier or train/test split; prespecified locus roster",
            "model_representation": "masked conditional nucleotide log probabilities or causal sequence log likelihoods; native GPN-MSA allele scores",
            "pooling_or_score": "selected-allele score minus other-allele score; positive values indicate directional concordance",
            "sequence_length": "model-specific paired context recorded by window_bp, center_index and tokenized_length in S39",
            "training_or_cv": "no classifier training and no cross-validation; direct paired allele scoring",
            "uncertainty_or_seed": "no locus-level sampling interval; aggregate exact binomial intervals are in S38",
            "metrics": "per-locus selected/other scores, their difference, directional_correct, QA and benchmark status; supplementary CH uses the 30 selected and 30 other raw score values",
            "configuration_note": "S39 is direct paired scoring; no population-centroid distance or adaptive-population ranking is computed",
        },
        {
            "evaluation_framework": "Ancestry embedding analysis",
            "benchmark_tables_or_tasks": "S33 unsupervised ancestry metrics; S34 supervised ancestry classification; S35 few-shot ancestry regression",
            "input_unit": "Individual-level embeddings with five 1KGP superpopulation labels and ADMIXTURE proportions; GPN-MSA uses 1,500 individuals (300 per superpopulation) at 10,000 shared loci per individual.",
            "input_files_and_columns": "Per-individual embeddings with population labels and K=5 ADMIXTURE targets; GPN-MSA inputs use GT[0] haplotypes and a shared 10,000-locus roster.",
            "train_valid_test_split": "S33: unsupervised separation; S34: five-fold stratified CV; S35: 10–60 training individuals sampled per superpopulation; remaining individuals are held out for evaluation.",
            "model_representation": "Individual embeddings; GPN-MSA uses ALT-minus-REF deltas from GPN-MSA last hidden states at 10,000 shared loci, aggregated to a 2,308-dimensional vector.",
            "pooling_or_score": "GPN-MSA concatenates the mean of the top 500 site deltas by L2 norm, global mean, norm-weighted mean and four norm statistics; standard model embeddings follow their configured individual-level aggregation.",
            "sequence_length": "For GPN-MSA, each 511-bp GT[0] haplotype context is cropped to 128 bp and supplied with alignment features from 89 auxiliary species; each individual has 10,000 shared loci.",
            "training_or_cv": "S34: feature-wise StandardScaler fitted within each training fold, then L2 logistic regression (C=1, solver=lbfgs, max_iter=1,000, random_state=42), five-fold StratifiedKFold (shuffle=True, random_state=42). For the added GPN-MSA S35 row, the finalizer passes Ridge alpha=1, shots 10–60 per population, 10 repetitions and seeds 10–19; earlier S35 rows retain aggregate output values and do not specify these training settings.",
            "uncertainty_or_seed": "S33 t-SNE: perplexity=30, init=PCA, learning_rate=auto, metric=cosine, random_state=42; CH and silhouette on 2D coordinates. S34 CV random_state=42. GPN-MSA S35 runs use seeds 10–19; earlier S35 rows retain aggregate output values without seed or repeat-count fields.",
            "metrics": "S33: Calinski–Harabasz and silhouette; S34: accuracy, MCC and macro-F1 (fold summaries); S35: Pearson r across repetitions for K=5 ancestry proportions.",
            "configuration_note": "The GPN-MSA finalizer calls the tracked hEvo_benchmark/scripts/population_classify/Regression_with_admixture.py with --min-shots 10, --max-shots 60, --num-repeats 10, --seed-start 10 and --alpha 1. The script samples training individuals within each superpopulation, fits StandardScaler and Ridge on those samples, and evaluates the remaining individuals. S35 reports shot-level mean and standard deviation summaries rather than individual run records.",
        },
    ]
    frame = pd.DataFrame(rows)
    wide = frame.set_index("evaluation_framework").T
    wide.loc["evaluation_framework"] = wide.columns
    wide = wide.loc[["evaluation_framework", *[idx for idx in wide.index if idx != "evaluation_framework"]]]
    wide.index.name = "configuration_item"
    return wide.reset_index()


def autosize(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for idx, column in enumerate(df.columns, start=1):
        values = df[column].astype(str).head(200)
        width = min(max([len(str(column)), *(len(v) for v in values)]) + 2, 70)
        worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = max(width, 10)


def build_excel(tables: dict[str, tuple[str, pd.DataFrame]]) -> None:
    zh = pd.DataFrame(
        [
            {"附表编号": "S46", "文件名": "S46_architecture_and_hyperparameters_of_the_primary_hevo_models.csv", "简要内容": "hEvo 四个主要模型的架构和共享训练超参数。"},
            {"附表编号": "S47", "文件名": "S47_architecture_training_scope_and_provenance_of_evaluated_models.csv", "简要内容": f"{len(tables['S47'][1])} 个登记模型的架构、来源与接口；推理接口和训练目标元数据分列。"},
            {"附表编号": "S48", "文件名": "S48_parameters_used_across_model_evaluation_protocols.csv", "简要内容": "标准 LLR、模型原生变异评分、GPN-Star MSA 及其他统一评估参数。"},
        ]
    )
    en = pd.DataFrame(
        [
            {"Table ID": "S46", "File name": "S46_architecture_and_hyperparameters_of_the_primary_hevo_models.csv", "Brief description": "Architecture and shared training settings for the four primary hEvo models."},
            {"Table ID": "S47", "File name": "S47_architecture_training_scope_and_provenance_of_evaluated_models.csv", "Brief description": f"Architecture, provenance, and inference interface for {len(tables['S47'][1])} registered models; training-objective metadata is separate."},
            {"Table ID": "S48", "File name": "S48_parameters_used_across_model_evaluation_protocols.csv", "Brief description": "Standard LLR, native model variant scores, GPN-Star MSA scoring, and other evaluation parameters."},
        ]
    )
    with pd.ExcelWriter(TABLES_DIR / "hEvo_benchmark_model_supplementary_tables.xlsx", engine="openpyxl") as writer:
        zh.to_excel(writer, sheet_name="目录_中文", index=False)
        autosize(writer, "目录_中文", zh)
        en.to_excel(writer, sheet_name="Index_EN", index=False)
        autosize(writer, "Index_EN", en)
        for table_id, (filename, df) in tables.items():
            sheet_name = table_id
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            autosize(writer, sheet_name, df)


def update_readme(tables: dict[str, tuple[str, pd.DataFrame]]) -> None:
    readme = TABLES_DIR / "README.md"
    text = readme.read_text(encoding="utf-8")
    marker = "\n## Model and evaluation protocol details (S46–S49)\n"
    for old_marker in ["\n## Model and evaluation protocol details (S46–S48)\n", marker]:
        if old_marker in text:
            text = text.split(old_marker)[0].rstrip() + "\n"

    sections = [
        marker,
        "\nThese model-supplementary tables summarize hEvo model hyperparameters, the evaluated model universe, and the common evaluation-parameter settings used by the benchmark framework.\n",
    ]
    descriptions = {
                "S46": "Architecture and shared Stage I/II training settings for the four primary hEvo models. Trainer optimizer/scheduler fields may be overridden by DeepSpeed; checkpoint dtype is distinct from training precision. The companion S46_training_run_budgets.csv contains nine configuration entries, including independent budget fields for the two public-resource models. Nominal nucleotide positions equal sequence budget times 1,024. Use scripts/sync_hevo_training_tables.py to synchronize publication S30 and R19 while preserving entered public-resource budgets.",
        "S47": "S47 contains 34 model records. The companion model table lists 31 single-sequence models, with the three GPN-Star-V/P/M MSA models described separately. The six variant result tables S23 and S26–S30 share 33 models; DNABERT is registered in S47 but absent from those tables. `model_lm_type` records the inference interface and is distinct from `training_objective`; NA identifies fields with no value in the cited model, publication, or checkpoint metadata.",
        "S48": "Public evaluation definitions and key parameters distinguish the standard task-directed paired-variant LLR from native Mistral-DNA pooled-logit L2 and GENERator log(P_ref/P_alt) scoring, and GPN-Star coordinate-aware MSA scoring; adaptive allele-preference tasks retain their separate protocol.",
    }
    sources = {
        "S46": "`<source-project-root>/models/hEvo/*/config.json`",
        "S47": "`<source-project-root>/models/models.tsv` when present, together with `configs/models.yaml`, `configs/gpn_star.yaml`, and pinned checkpoint metadata; otherwise the checked-in S47 snapshot is reused",
        "S48": "Benchmark configuration and task-specific evaluation documentation",
    }
    for table_id, (filename, df) in tables.items():
        sections.extend(
            [
                f"\n### {filename}\n\n",
                f"{descriptions[table_id]}\n\n",
                f"- Source: {sources[table_id]}\n",
                f"- Rows: {len(df)} rows\n",
                f"- Columns: {', '.join(f'`{col}`' for col in df.columns)}\n",
            ]
        )
        if table_id == "S47":
            sections.append(
                "\nThe hEvo parameter counts are counts of unique stored checkpoint parameters; "
                "shared tensor aliases are counted once. The four hEvo checkpoints contain "
                "87,094,922 parameters for hEvo-base and hEvo-open-base, and 319,462,410 "
                "parameters for hEvo-large and hEvo-open-large. S43–S44 report total predictor "
                "parameters, adding the task-specific probe to the inference backbone. "
                "For GPN-Star this excludes the unused masked-LM output head; S47 retains "
                "the full 203,022,215-parameter checkpoint count.\n"
            )
    rank_path = TABLES_DIR / "S49_model_task_level_scores_and_ranks_for_fig5.csv"
    if rank_path.exists():
        ranks = pd.read_csv(rank_path)
        star_coverage = ranks[ranks.model.str.startswith("GPN-Star-")].groupby("model").task_unit.nunique()
        sections.extend(
            [
                "\n### S49_model_task_level_scores_and_ranks_for_fig5.csv\n\n",
                "Task-level score and rank evidence for the Fig. 5 Pareto vertical axis. Scores are ranked within task with minimum ranks for ties; higher scores rank better. URE multiclass averages class-specific F1 within dataset, ancestry regression averages the shot sweep, and HAR/HAQER average the five negative-control runs. Tasks without GPN-Star scores and missing model/task scores are not imputed.\n\n",
                "- Source: S17–S20, S23, S26–S37\n",
                "- File: `S49_model_task_level_scores_and_ranks_for_fig5.csv`\n",
                "- Builder: `scripts/build_fig5_cross_task_ranks.py`\n",
                f"- Rows: {len(ranks)} model/task rows across {ranks.task_unit.nunique()} task units\n",
                f"- GPN-Star task coverage: {', '.join(f'{model} {int(count)}/{ranks.task_unit.nunique()}' for model, count in sorted(star_coverage.items()))}\n",
            ]
        )
    readme.write_text(text.rstrip() + "\n" + "".join(sections), encoding="utf-8")


def main() -> None:
    tables = {
        "S46": ("S46_architecture_and_hyperparameters_of_the_primary_hevo_models.csv", build_s46()),
        "S47": ("S47_architecture_training_scope_and_provenance_of_evaluated_models.csv", build_model_roster()),
        "S48": ("S48_parameters_used_across_model_evaluation_protocols.csv", build_evaluation_parameters()),
    }
    for _, (filename, df) in tables.items():
        df.to_csv(TABLES_DIR / filename, index=False)
    build_excel(tables)
    update_readme(tables)
    for table_id, (filename, df) in tables.items():
        print(f"{table_id}: {filename}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
