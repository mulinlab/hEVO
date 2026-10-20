#!/usr/bin/env python3
"""Rebuild Fig. 1h family-best profiles from the current task-level tables.

For each axis, metrics are averaged across the labeled benchmark subsets and
the resulting model means are ranked across all complete-coverage models. The
best complete-coverage member is retained independently for each of the hEVO,
Evo 2, NT and GPN families. All combined metrics are bounded 0-1; the ancestry
axis uses three explicitly defined 0-1 module summaries. This also fixes the
stale adaptation ranking in the legacy S6 table.
"""

from __future__ import annotations

# [Role] Build data for Fig. 1h without rendering the final page; summarize model-level results as the best profile for each family.
# [Input] Raw Fig. 1h rank and benchmark tables, aggregated by model family, task, and ranking rules.
# [Output] Revised family-profile CSV, subset/rank/coverage summaries, and QA JSON for the radar plotting script.
# [Order] Run this before plot_fig1h_family_rank_radars_compact.py and the integrated Fig. 1 script.

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = PROJECT_ROOT / "tables"
OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig1" / "figure1_a4_integrated_v2" / "source_data"

GPN_FAMILY = "GPN family"
FAMILY_ORDER = ("hEVO family", "Evo 2 family", "NT family", GPN_FAMILY)


def family_of(model: str) -> str | None:
    key = model.strip().lower()
    if key.startswith("hevo"):
        return "hEVO family"
    if key.startswith("evo2"):
        return "Evo 2 family"
    if key.startswith("nt_") or key.startswith("nt-") or key.startswith("ntv3"):
        return "NT family"
    if key.startswith("gpn-") or key.startswith("gpn_"):
        return GPN_FAMILY
    return None


def normalized_subset_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["metric"] = pd.to_numeric(result["metric"], errors="raise")
    result["n_models"] = result.groupby(["axis", "subset"])["model"].transform("nunique")
    result["rank"] = result.groupby(["axis", "subset"])["metric"].rank(method="min", ascending=False)
    denominator = (result["n_models"] - 1).replace(0, np.nan)
    result["normalized_rank_percentile"] = (result["n_models"] - result["rank"]) / denominator
    result["normalized_rank_percentile"] = result["normalized_rank_percentile"].fillna(1.0)
    return result


def standard_rows(path: str, axis: str, subset_col: str, model_col: str, metric_col: str) -> pd.DataFrame:
    frame = pd.read_csv(TABLE_DIR / path)
    return pd.DataFrame(
        {
            "axis": axis,
            "subset": frame[subset_col].astype(str),
            "model": frame[model_col].astype(str),
            "metric": pd.to_numeric(frame[metric_col], errors="raise"),
            "source_table": path,
        }
    )


def single_table_rows(path: str, axis: str, model_col: str, metric_col: str, subset: str) -> pd.DataFrame:
    frame = pd.read_csv(TABLE_DIR / path)
    return pd.DataFrame(
        {
            "axis": axis,
            "subset": subset,
            "model": frame[model_col].astype(str),
            "metric": pd.to_numeric(frame[metric_col], errors="raise"),
            "source_table": path,
        }
    )


def adaptation_rows() -> pd.DataFrame:
    frame = pd.read_csv(TABLE_DIR / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv")
    frame = frame.loc[frame["status"].eq("PASS") & pd.to_numeric(frame["accuracy"], errors="coerce").notna()]
    return pd.DataFrame(
        {
            "axis": "Adaptation",
            "subset": "14 verified loci",
            "model": frame["model_id"].astype(str),
            "metric": pd.to_numeric(frame["accuracy"], errors="raise"),
            "source_table": "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv",
        }
    )


def ure_rows() -> pd.DataFrame:
    binary = standard_rows(
        "S19_binary_classification_across_five_ure_enhancer_control_sets.csv",
        "URE",
        "benchmark_raw",
        "model",
        "mcc_mean",
    )
    multi_raw = pd.read_csv(TABLE_DIR / "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv")
    multi = (
        multi_raw.groupby(["dataset", "model"], as_index=False)["per_class_f1"]
        .mean()
        .rename(columns={"dataset": "subset", "per_class_f1": "metric"})
    )
    multi["axis"] = "URE"
    multi["subset"] = "multiclass:" + multi["subset"].astype(str)
    multi["source_table"] = "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv"
    binary["subset"] = "binary:" + binary["subset"].astype(str)
    return pd.concat([binary, multi[["axis", "subset", "model", "metric", "source_table"]]], ignore_index=True)


def ancestry_rows() -> pd.DataFrame:
    """Return three bounded 0-1 ancestry modules for direct averaging."""
    unsupervised = pd.read_csv(TABLE_DIR / "S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv")
    unsup_module = pd.DataFrame(
        {
            "axis": "Ancestry",
            "subset": "unsupervised",
            "model": unsupervised["model"].astype(str),
            "metric": pd.to_numeric(unsupervised["rank_score"], errors="raise"),
            "source_table": "S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv",
        }
    )

    supervised = pd.read_csv(TABLE_DIR / "S34_supervised_classification_of_1000_genomes_ancestry_groups.csv")
    sup_module = pd.DataFrame(
        {
            "axis": "Ancestry",
            "subset": "supervised",
            "model": supervised["model"].astype(str),
            "metric": pd.to_numeric(supervised["classification_mcc"], errors="raise"),
            "source_table": "S34_supervised_classification_of_1000_genomes_ancestry_groups.csv",
        }
    )

    fewshot = pd.read_csv(TABLE_DIR / "S35_few_shot_regression_of_ancestry_proportions.csv")
    fewshot_module = (
        fewshot.groupby("model", as_index=False)["pearson_r_mean"]
        .mean()
        .rename(columns={"pearson_r_mean": "metric"})
    )
    fewshot_module["axis"] = "Ancestry"
    fewshot_module["subset"] = "few-shot regression"
    fewshot_module["source_table"] = "S35_few_shot_regression_of_ancestry_proportions.csv"
    return pd.concat(
        [
            unsup_module,
            sup_module,
            fewshot_module[["axis", "subset", "model", "metric", "source_table"]],
        ],
        ignore_index=True,
    )


def build_subset_table() -> pd.DataFrame:
    pieces = [
        standard_rows("S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv", "Genomic", "dataset", "model", "mcc_mean"),
        standard_rows("S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv", "NT", "dataset", "model", "mcc_mean"),
        ure_rows(),
        standard_rows("S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv", "DeepSEA", "task", "model", "auroc"),
        single_table_rows("S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv", "ClinVar", "model", "auprc_mean", "all"),
        single_table_rows("S24_zero_shot_classification_of_coding_clinvar_variants.csv", "ClinVar", "model", "auprc_mean", "coding"),
        single_table_rows("S25_zero_shot_classification_of_noncoding_clinvar_variants.csv", "ClinVar", "model", "auprc_mean", "noncoding"),
        single_table_rows("S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv", "De novo", "model", "auprc_mean", "vs ClinVar benign"),
        single_table_rows("S31_within_site_top_rank_discrimination_of_ddd_de_novo_alleles.csv", "De novo", "model", "paired_top1", "same-position alleles"),
        single_table_rows("S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv", "GWAS causal", "model", "auprc_mean", "causal variants"),
        single_table_rows("S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv", "GWAS causal", "model", "auprc_mean", "rare causal variants"),
        single_table_rows("S28_zero_shot_classification_of_uk_biobank_rare_variants.csv", "Rare variants", "model", "auprc_mean", "UKB rare variants"),
        single_table_rows("S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv", "MPRA", "model", "auprc_mean", "MPRA"),
        standard_rows("S37_classification_of_human_accelerated_regions_across_negative_controls.csv", "HAR", "benchmark_raw", "model", "mcc_mean"),
        standard_rows("S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv", "HAQER", "benchmark_raw", "model", "mcc_mean"),
        ancestry_rows(),
        adaptation_rows(),
    ]
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["axis", "subset", "model"]).any():
        duplicated = result.loc[result.duplicated(["axis", "subset", "model"], keep=False), ["axis", "subset", "model"]]
        raise ValueError(f"Duplicate axis-subset-model rows:\n{duplicated.head(20)}")
    return result


def family_best(subset_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average subset metrics, rank model means, then retain each family best."""
    axis_counts = subset_rows.groupby("axis")["subset"].nunique().to_dict()
    coverage = (
        subset_rows.groupby(["axis", "model"], as_index=False)
        .agg(
            mean_metric=("metric", "mean"),
            observed_subsets=("subset", "nunique"),
        )
    )
    coverage["expected_subsets"] = coverage["axis"].map(axis_counts)
    coverage["complete_coverage"] = coverage["observed_subsets"] == coverage["expected_subsets"]
    eligible = coverage.loc[coverage["complete_coverage"]].copy()
    eligible["n_models"] = eligible.groupby("axis")["model"].transform("nunique")
    eligible["axis_rank"] = eligible.groupby("axis")["mean_metric"].rank(method="min", ascending=False)
    eligible["normalized_axis_rank"] = (
        (eligible["n_models"] - eligible["axis_rank"])
        / (eligible["n_models"] - 1).replace(0, np.nan)
    ).fillna(1.0)
    coverage = coverage.merge(
        eligible[["axis", "model", "n_models", "axis_rank", "normalized_axis_rank"]],
        on=["axis", "model"],
        how="left",
    )
    coverage["family"] = coverage["model"].map(family_of)
    candidates = coverage.loc[
        coverage["family"].isin(FAMILY_ORDER) & coverage["complete_coverage"]
    ].copy()
    candidates = candidates.sort_values(
        ["axis", "family", "mean_metric", "axis_rank", "model"],
        ascending=[True, True, False, True, True],
    )
    best = candidates.groupby(["axis", "family"], as_index=False).first()

    best["family_member_count"] = 1

    axes = sorted(subset_rows["axis"].unique())
    full_index = pd.MultiIndex.from_product([axes, FAMILY_ORDER], names=["axis", "family"])
    best = best.set_index(["axis", "family"]).reindex(full_index).reset_index()
    best["expected_subsets"] = best["axis"].map(axis_counts)
    best["display_radius"] = best["normalized_axis_rank"]
    return best, coverage


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return strict-JSON records, converting pandas missing values to null."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subset_rows = build_subset_table()
    ranked = normalized_subset_ranks(subset_rows)
    best, coverage = family_best(subset_rows)

    category_map = {
        "Genomic": "Genomic elements",
        "NT": "Genomic elements",
        "URE": "Genomic elements",
        "DeepSEA": "Genomic elements",
        "ClinVar": "Variant scoring",
        "De novo": "Variant scoring",
        "GWAS causal": "Variant scoring",
        "Rare variants": "Variant scoring",
        "MPRA": "Variant scoring",
        "HAR": "Evolution & ancestry",
        "HAQER": "Evolution & ancestry",
        "Ancestry": "Evolution & ancestry",
        "Adaptation": "Evolution & ancestry",
    }
    label_count_override = {"Adaptation": 14}
    best["category"] = best["axis"].map(category_map)
    best["label_count"] = best.apply(
        lambda row: label_count_override.get(str(row["axis"]), int(row["expected_subsets"])),
        axis=1,
    )
    best["axis_label"] = best.apply(lambda row: f"{row['axis']} ({int(row['label_count'])})", axis=1)

    best_path = OUT_DIR / "Fig1h_revised_family_best_profiles.csv"
    ranked_path = OUT_DIR / "Fig1h_revised_subset_rank_summary.csv"
    coverage_path = OUT_DIR / "Fig1h_revised_model_coverage_summary.csv"
    best.to_csv(best_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    adaptation = best.loc[best["axis"] == "Adaptation", ["family", "model", "mean_metric", "axis_rank", "normalized_axis_rank"]]
    deepsea = best.loc[best["axis"] == "DeepSEA", ["family", "model", "mean_metric", "normalized_axis_rank"]]
    gpn_profile = best.loc[
        best["family"] == GPN_FAMILY,
        ["axis", "model", "family_member_count", "mean_metric", "normalized_axis_rank"],
    ]
    qa = {
        "method": "mean metric across labeled benchmark subsets, followed by across-model rank normalization; the family-best complete-coverage model is retained independently for hEVO, Evo 2, NT and GPN",
        "axes": best.groupby("category")["axis"].unique().apply(list).to_dict(),
        "expected_subsets": {key: int(value) for key, value in ranked.groupby("axis")["subset"].nunique().to_dict().items()},
        "adaptation_recomputed": json_records(adaptation),
        "deepsea_family_coverage": json_records(deepsea),
        "gpn_family_best_profile": json_records(gpn_profile),
        "gpn_members": sorted(
            coverage.loc[coverage["family"] == GPN_FAMILY, "model"].dropna().unique().tolist()
        ),
        "gpn_selection_boundary": "The best complete-coverage member of GPN-MSA and GPN-promoter is selected independently on each axis; GPN has no DeepSEA row in the registered source tables, so that point is NA.",
        "critical_correction": "The current S42 accuracy-ranked adaptation summary is used; no legacy S-table ranking is mixed into the current profile.",
        "deepsea_boundary": "The current DeepSEA source has no Evo 2 model; its radar point must be displayed as NA, not zero.",
        "outputs": [str(best_path), str(ranked_path), str(coverage_path)],
    }
    print(best[["category", "axis_label", "family", "model", "mean_metric", "axis_rank", "normalized_axis_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
