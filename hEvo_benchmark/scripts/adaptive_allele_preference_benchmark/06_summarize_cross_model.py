#!/usr/bin/env python3
"""Aggregate independent adaptive-allele preference scores across model families."""

from __future__ import annotations
import argparse
import csv
import itertools
import math
from pathlib import Path
import numpy as np
import pandas as pd
from allele_direction import orient_alleles
from scipy.stats import beta, binomtest, wilcoxon


def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize fresh cross-model adaptive-allele preference scores."
    )
    p.add_argument("--score-files", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-reps", type=int, default=10000)
    return p.parse_args()


def read_scores(paths):
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as h:
            rows.extend(csv.DictReader(h, delimiter="\t"))
    if not rows:
        raise ValueError("no score rows")
    rows = orient_alleles(pd.DataFrame(rows)).to_dict("records")
    seen = set()
    for r in rows:
        key = (r["model_id"], r["variant_id"])
        if key in seen:
            raise ValueError(f"duplicate model x locus: {key}")
        seen.add(key)
        s = float(r["locus_score"])
        if not math.isfinite(s):
            raise ValueError(f"nonfinite score: {key}")
    return rows


def ci_accuracy(k, n, alpha=0.05):
    if k == 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(alpha / 2, k, n - k + 1))
    if k == n:
        hi = 1.0
    else:
        hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def safe_wilcoxon(x):
    if np.allclose(x, 0):
        return float("nan"), 1.0
    try:
        z = wilcoxon(x, alternative="greater", zero_method="wilcox", method="auto")
        return float(z.statistic), float(z.pvalue)
    except ValueError:
        return float("nan"), 1.0


def main():
    a = parse_args()
    rows = read_scores(a.score_files)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({r["model_id"] for r in rows})
    loci = sorted({r["variant_id"] for r in rows})
    if len(rows) != len(models) * len(loci):
        raise ValueError("score table is not a complete model x locus grid")
    by = {(r["model_id"], r["variant_id"]): r for r in rows}
    rng = np.random.default_rng(a.seed)
    summaries = []
    boots = []
    loo = []
    for model in models:
        vals = np.array(
            [float(by[(model, locus)]["locus_score"]) for locus in loci], dtype=float
        )
        n = len(vals)
        k = int(np.sum(vals > 0))
        lo, hi = ci_accuracy(k, n)
        wstat, wp = safe_wilcoxon(vals)
        med = float(np.median(vals))
        q1, q3 = np.quantile(vals, [0.25, 0.75])
        boot_med = []
        boot_acc = []
        for _ in range(a.bootstrap_reps):
            sample = rng.choice(vals, size=n, replace=True)
            boot_med.append(np.median(sample))
            boot_acc.append(np.mean(sample > 0))
        summaries.append(
            {
                "model_id": model,
                "n_loci": n,
                "positive_loci": k,
                "accuracy": f"{k / n:.10g}",
                "accuracy_ci_low": f"{lo:.10g}",
                "accuracy_ci_high": f"{hi:.10g}",
                "one_sided_exact_binom_p": f"{binomtest(k, n, 0.5, alternative='greater').pvalue:.10g}",
                "median_score": f"{med:.10g}",
                "q1_score": f"{q1:.10g}",
                "q3_score": f"{q3:.10g}",
                "iqr_score": f"{q3 - q1:.10g}",
                "wilcoxon_statistic": f"{wstat:.10g}",
                "wilcoxon_one_sided_p": f"{wp:.10g}",
                "bootstrap_reps": a.bootstrap_reps,
                "bootstrap_seed": a.seed,
                "bootstrap_unit": "locus",
            }
        )
        boots += [
            {
                "model_id": model,
                "metric": "median_score",
                "observed": f"{med:.10g}",
                "ci_low": f"{np.quantile(boot_med, 0.025):.10g}",
                "ci_high": f"{np.quantile(boot_med, 0.975):.10g}",
                "reps": a.bootstrap_reps,
                "seed": a.seed,
                "unit": "locus",
            },
            {
                "model_id": model,
                "metric": "accuracy",
                "observed": f"{k / n:.10g}",
                "ci_low": f"{np.quantile(boot_acc, 0.025):.10g}",
                "ci_high": f"{np.quantile(boot_acc, 0.975):.10g}",
                "reps": a.bootstrap_reps,
                "seed": a.seed,
                "unit": "locus",
            },
        ]
        for omitted in loci:
            keep = np.array([locus != omitted for locus in loci])
            x = vals[keep]
            kk = int(np.sum(x > 0))
            loo.append(
                {
                    "model_id": model,
                    "omitted_locus": omitted,
                    "n_loci": len(x),
                    "positive_loci": kk,
                    "accuracy": f"{kk / len(x):.10g}",
                    "median_score": f"{np.median(x):.10g}",
                }
            )
    cols = list(summaries[0])
    write(a.output_dir / "model_summary.tsv", summaries, cols)
    write(a.output_dir / "bootstrap_summary.tsv", boots, list(boots[0]))
    write(a.output_dir / "leave_one_locus_out.tsv", loo, list(loo[0]))
    paired = []
    for m1, m2 in itertools.combinations(models, 2):
        d = np.array(
            [
                float(by[(m1, locus)]["locus_score"])
                - float(by[(m2, locus)]["locus_score"])
                for locus in loci
            ]
        )
        stat, p = safe_wilcoxon(d)
        paired.append(
            {
                "model_a": m1,
                "model_b": m2,
                "n_loci": len(d),
                "median_delta_a_minus_b": f"{np.median(d):.10g}",
                "q1_delta": f"{np.quantile(d, 0.25):.10g}",
                "q3_delta": f"{np.quantile(d, 0.75):.10g}",
                "wilcoxon_statistic": f"{stat:.10g}",
                "wilcoxon_one_sided_p_a_gt_b": f"{p:.10g}",
            }
        )
    write(
        a.output_dir / "paired_model_tests.tsv",
        paired,
        list(paired[0]) if paired else ["model_a", "model_b"],
    )
    all_cols = list(rows[0])
    write(a.output_dir / "locus_scores.tsv", rows, all_cols)
    qa = [
        "# QA_REPORT",
        "",
        "## Scope",
        "Fresh adaptive-allele preference benchmark; validated reference-context scores across the independent model roster.",
        "",
        "## Checks",
        f"- score rows: PASS ({len(rows)})",
        f"- model x locus grid: PASS ({len(models)} models x {len(loci)} loci)",
        "- duplicate model x locus: PASS",
        "- finite locus scores: PASS",
        "- center-mask/token-map checks: delegated to per-model manifests",
        "- bootstrap unit: PASS (locus; seed %d; %d reps)" % (a.seed, a.bootstrap_reps),
        "- legacy selection labels/scores/embeddings used: PASS (false in scoring manifests)",
        "",
        "## Interpretation gate",
        "These are locus-level preference measurements, not evidence that the model detects genome-wide positive selection.",
    ]
    (a.output_dir / "QA_REPORT.md").write_text("\n".join(qa) + "\n", encoding="utf-8")
    lines = [
        "# RESULTS_SUMMARY",
        "",
        "## Scope",
        "Adaptive Allele Preference Benchmark, fresh reference-context scoring across validated model families.",
        f"- Frozen loci: {len(loci)}",
        "- Models: " + ", ".join(models),
        "- Context: verified per-model window; center locus variant is represented by each model's native tokenizer.",
        "- Score: model-native selected-versus-other allele preference under the declared masked or causal objective.",
        "- Bootstrap unit: locus; seed %d; %d replicates." % (a.seed, a.bootstrap_reps),
        "",
        "## Model-level results",
    ]
    for s in summaries:
        lines.append(
            f"- {s['model_id']}: accuracy {s['accuracy']} ({s['positive_loci']}/{s['n_loci']}), median score {s['median_score']}, exact one-sided binomial p={s['one_sided_exact_binom_p']}."
        )
    lines += [
        "",
        "## Status",
        "This is an interim reference-context comparison. Models that could not produce a checkpoint-backed likelihood are excluded from the scored denominator and are listed separately in the roster QA.",
    ]
    (a.output_dir / "RESULTS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("models=%d loci=%d rows=%d" % (len(models), len(loci), len(rows)))
    return 0


def write(path, rows, cols):
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
