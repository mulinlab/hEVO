#!/usr/bin/env python3
"""Merge checkpoint-backed native GPN-MSA raw scores into S38 and S39."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allele_direction import orient_alleles
from scipy.stats import beta, binomtest, wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s38", type=Path, required=True)
    parser.add_argument("--s39", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-s38", type=Path, required=True)
    parser.add_argument("--output-s39", type=Path, required=True)
    args = parser.parse_args()

    s38 = pd.read_csv(args.s38, dtype=str, keep_default_na=False)
    s39 = orient_alleles(pd.read_csv(args.s39, dtype=str, keep_default_na=False))
    scores = pd.read_csv(args.scores, sep="\t")
    if len(scores) != 30 or scores["variant_id"].nunique() != 30:
        raise ValueError("GPN-MSA score file must contain 30 unique loci")
    mask = s39["model_id"].eq("GPN-MSA")
    target = s39.loc[mask].copy().set_index("variant_id")
    scores = orient_alleles(scores).set_index("variant_id")
    if set(target.index) != set(scores.index):
        raise ValueError("S39 and GPN-MSA raw-score loci differ")
    aligned = scores.reindex(target.index)
    for column in ["selected_logprob", "other_logprob", "locus_score"]:
        target[column] = aligned[column].map(lambda value: f"{float(value):.10g}")
    target["directional_correct"] = (
        aligned["directional_correct"].astype(int).astype(str)
    )
    target["score_definition"] = (
        "GPN-MSA bidirectional full-vocabulary log-softmax selected_logprob - other_logprob"
    )
    target["benchmark_status"] = "PASS"
    update_columns = [column for column in target.columns if column != "variant_id"]
    s39.loc[mask, update_columns] = target.reindex(s39.loc[mask, "variant_id"])[
        update_columns
    ].to_numpy()

    values = aligned["locus_score"].astype(float)
    positive = int((values > 0).sum())
    accuracy = positive / len(values)
    low = (
        0.0
        if positive == 0
        else float(beta.ppf(0.025, positive, len(values) - positive + 1))
    )
    high = (
        1.0
        if positive == len(values)
        else float(beta.ppf(0.975, positive + 1, len(values) - positive))
    )
    wilcoxon_result = wilcoxon(
        values.to_numpy(), alternative="greater", zero_method="wilcox", method="auto"
    )
    rng = np.random.default_rng(42)
    bootstrap_medians = np.asarray(
        [
            np.median(rng.choice(values.to_numpy(), size=len(values), replace=True))
            for _ in range(10_000)
        ]
    )
    row = s38["model_id"].eq("GPN-MSA")
    if int(row.sum()) != 1:
        raise ValueError("S38 must contain exactly one GPN-MSA row")
    updates = {
        "status": "PASS",
        "n_loci": "30",
        "positive_loci": str(positive),
        "accuracy": f"{accuracy:.10g}",
        "accuracy_ci_low": f"{low:.10g}",
        "accuracy_ci_high": f"{high:.10g}",
        "one_sided_exact_binom_p": f"{binomtest(positive, 30, 0.5, alternative='greater').pvalue:.10g}",
        "median_score": f"{values.median():.10g}",
        "q1_score": f"{values.quantile(0.25):.10g}",
        "q3_score": f"{values.quantile(0.75):.10g}",
        "iqr_score": f"{values.quantile(0.75) - values.quantile(0.25):.10g}",
        "wilcoxon_statistic": f"{float(wilcoxon_result.statistic):.10g}",
        "wilcoxon_one_sided_p": f"{float(wilcoxon_result.pvalue):.10g}",
        "median_score_bootstrap_ci_low": f"{np.quantile(bootstrap_medians, 0.025):.10g}",
        "median_score_bootstrap_ci_high": f"{np.quantile(bootstrap_medians, 0.975):.10g}",
        "scoring_definition": "GPN-MSA bidirectional full-vocabulary log-softmax selected_logprob - other_logprob",
        "window_bp": "128",
        "center_index": "64",
    }
    for column, value in updates.items():
        s38.loc[row, column] = value
    passed_accuracies = pd.to_numeric(s38.loc[s38["status"].eq("PASS"), "accuracy"])
    gpn_rank = int((passed_accuracies > accuracy).sum() + 1)
    s38.loc[row, "rank_by_accuracy"] = str(gpn_rank)
    args.output_s38.parent.mkdir(parents=True, exist_ok=True)
    s38.to_csv(args.output_s38, index=False)
    s39.to_csv(args.output_s39, index=False)
    print(f"merged GPN-MSA: positive={positive}/30 accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
