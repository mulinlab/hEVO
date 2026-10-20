#!/usr/bin/env python3
"""Rebuild Fig.1h family-best ranks from the active public source tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from deepsea_run_selection import active_deepsea_rows


FAMILIES = ("hEVO family", "Evo 2 family", "NT family", "GPN family")
CANONICAL_ROSTER_SIZE = 29
# Fig. 1h retains the original comparison roster. These models were added to
# the wider supplementary roster later and are shown in Fig. 5, not Fig. 1h.
FIG1H_EXCLUDED_MODELS = {
    "hEvo-open-base", "hEvo-open-large",
    "GPN-Star-M", "GPN-Star-P", "GPN-Star-V",
}


def family(model: str) -> str | None:
    lowered = model.lower()
    if lowered.startswith("hevo"):
        return "hEVO family"
    if lowered.startswith("evo2"):
        return "Evo 2 family"
    if lowered.startswith("nt"):
        return "NT family"
    if lowered.startswith("gpn"):
        return "GPN family"
    return None


def mean_by_model(frame: pd.DataFrame, value: str, model: str = "model") -> pd.Series:
    values = pd.to_numeric(frame[value], errors="coerce")
    return values.groupby(frame[model].astype(str)).mean()


def combine(series: list[pd.Series]) -> pd.Series:
    return pd.concat(series, axis=1).mean(axis=1, skipna=True)


def source_scores(root: Path) -> list[dict[str, object]]:
    read = lambda name: pd.read_csv(root / name, low_memory=False)
    s17 = read("S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv")
    s18 = read("S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv")
    s19 = read("S19_binary_classification_across_five_ure_enhancer_control_sets.csv")
    s20 = read("S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv")
    s21 = active_deepsea_rows(read("S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv"))
    s23 = read("S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv")
    s24 = read("S24_zero_shot_classification_of_coding_clinvar_variants.csv")
    s25 = read("S25_zero_shot_classification_of_noncoding_clinvar_variants.csv")
    s26 = read("S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv")
    if {"num_samples", "num_positive", "num_negative"}.issubset(s26.columns):
        cohort = ["num_samples", "num_positive", "num_negative"]
        matched = s26.groupby(cohort, dropna=False).size().idxmax()
        s26 = s26.set_index(cohort).loc[[matched]].reset_index()
    s28 = read("S28_zero_shot_classification_of_uk_biobank_rare_variants.csv")
    s29 = read("S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv")
    s30 = read("S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv")
    s31 = read("S31_within_site_top_rank_discrimination_of_ddd_de_novo_alleles.csv")
    s33 = read("S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv")
    s34 = read("S34_supervised_classification_of_1000_genomes_ancestry_groups.csv")
    s35 = read("S35_few_shot_regression_of_ancestry_proportions.csv")
    s36 = read("S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv")
    s37 = read("S37_classification_of_human_accelerated_regions_across_negative_controls.csv")
    s38 = read("S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv")
    s38 = s38.loc[s38["status"].eq("PASS")].copy()

    return [
        {"category": "Genomic elements", "axis": "Genomic", "task_count": int(s18["dataset"].nunique()), "scores": mean_by_model(s18, "mcc_mean"), "sources": "S18"},
        {"category": "Genomic elements", "axis": "NT", "task_count": int(s17["dataset"].nunique()), "scores": mean_by_model(s17, "mcc_mean"), "sources": "S17"},
        {"category": "Genomic elements", "axis": "URE", "task_count": int(s19["benchmark_raw"].nunique() + s20["class_id"].nunique()), "scores": combine([mean_by_model(s19, "mcc_mean"), mean_by_model(s20, "per_class_f1")]), "sources": "S19;S20"},
        {"category": "Genomic elements", "axis": "DeepSEA", "task_count": int(s21["task"].nunique()), "scores": mean_by_model(s21, "auroc"), "sources": "S21"},
        {"category": "Variant scoring", "axis": "ClinVar", "task_count": 3, "scores": combine([mean_by_model(s23, "auprc_mean"), mean_by_model(s24, "auprc_mean"), mean_by_model(s25, "auprc_mean")]), "sources": "S23;S24;S25"},
        {"category": "Variant scoring", "axis": "De novo", "task_count": 2, "scores": combine([mean_by_model(s30, "auprc_mean"), mean_by_model(s31.loc[s31["status"].eq("PASS")] if "status" in s31 else s31, "paired_top1_bootstrap_mean")]), "sources": "S30;S31"},
        {"category": "Variant scoring", "axis": "GWAS causal", "task_count": 1, "scores": mean_by_model(s26, "auprc_mean"), "sources": "S26"},
        {"category": "Variant scoring", "axis": "Rare variants", "task_count": 1, "scores": mean_by_model(s28, "auprc_mean"), "sources": "S28"},
        {"category": "Variant scoring", "axis": "MPRA", "task_count": 1, "scores": mean_by_model(s29, "auprc_mean"), "sources": "S29"},
        {"category": "Evolution & ancestry", "axis": "HAR", "task_count": int(s37.groupby("model").size().mode().iloc[0]), "scores": mean_by_model(s37, "mean_mcc_for_sort"), "sources": "S37"},
        {"category": "Evolution & ancestry", "axis": "HAQER", "task_count": int(s36.groupby("model").size().mode().iloc[0]), "scores": mean_by_model(s36, "mean_mcc_for_sort"), "sources": "S36"},
        {"category": "Evolution & ancestry", "axis": "Ancestry", "task_count": 3, "scores": combine([mean_by_model(s33, "rank_score"), mean_by_model(s34, "classification_mcc"), mean_by_model(s35, "pearson_r_mean")]), "sources": "S33;S34;S35"},
        {"category": "Evolution & ancestry", "axis": "Adaptation", "task_count": int(pd.to_numeric(s38["n_loci_expected"], errors="raise").mode().iloc[0]), "scores": mean_by_model(s38, "accuracy", model="model_id"), "sources": "S38"},
    ]


def build(root: Path) -> pd.DataFrame:
    roster = pd.read_csv(root / "S47_architecture_training_scope_and_provenance_of_evaluated_models.csv")
    canonical = [
        model for model in dict.fromkeys(roster["model"].astype(str))
        if model not in FIG1H_EXCLUDED_MODELS
    ]
    if len(canonical) != CANONICAL_ROSTER_SIZE:
        raise ValueError(
            f"Fig. 1h comparison roster must contain {CANONICAL_ROSTER_SIZE} models "
            f"after excluding later additions, got {len(canonical)}"
        )
    canonical_set = set(canonical)
    rows: list[dict[str, object]] = []
    for spec in source_scores(root):
        axis = str(spec["axis"])
        scores = pd.to_numeric(spec["scores"], errors="coerce").dropna()
        scores = scores.loc[scores.index.astype(str).isin(canonical_set)]
        scores = scores.groupby(level=0).mean()
        ranks = scores.rank(method="min", ascending=False)
        n_eligible = int(len(scores))
        if n_eligible == 0:
            raise ValueError(f"No eligible canonical scores for {axis}")
        for family_name in FAMILIES:
            members = [name for name in scores.index if family(str(name)) == family_name]
            if members:
                best_model = str(scores.loc[members].idxmax())
                rank = float(ranks.loc[best_model])
                radius = float((n_eligible - rank + 1.0) / n_eligible)
                metric = float(scores.loc[best_model])
                status = "PASS"
            else:
                best_model, rank, radius, metric, status = "", pd.NA, pd.NA, pd.NA, "NOT_AVAILABLE"
            rows.append({
                "category": spec["category"], "axis": axis,
                "axis_label": f"{axis} ({int(spec['task_count'])})",
                "family": family_name, "best_model": best_model,
                "metric": metric, "rank": rank, "n_eligible": n_eligible,
                "canonical_roster_size": CANONICAL_ROSTER_SIZE, "display_radius": radius,
                "source_tables": spec["sources"], "status": status,
            })
    result = pd.DataFrame(rows)
    if len(result) != 13 * len(FAMILIES):
        raise ValueError("Unexpected Fig.1h row count")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.tables_dir.resolve()
    output = args.output.resolve() if args.output else root / "S10_best_model_family_ranks_across_functional_variant_and_evolutionary_tasks.csv"
    build(root).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
