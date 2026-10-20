#!/usr/bin/env python3
"""Build the S38-S39 source tables for the adaptive allele preference benchmark.

The script intentionally preserves blocked models as explicit rows in S38 and
S39.  It never imputes scores for a model without a checkpoint-backed head.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from allele_direction import orient_alleles

import sys

FIGURE_SCRIPTS = Path(__file__).resolve().parents[2] / "figures" / "scripts"
if str(FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SCRIPTS))
from common.model_names import canonical_model_name  # noqa: E402 - local path/runtime configuration precedes this import

MODEL_ID_ALIASES = {
    # Historical result bundles used these display spellings. Keep their
    # scores, but map them back to the stable IDs in the registries and S47.
    "Generator-1.2B": "GENERator-1.2B",
    "Generator-3B": "GENERator-3B",
}


def canonicalize_model_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if "model_id" not in frame.columns:
        raise ValueError("adaptive result table is missing model_id")
    result = frame.copy()
    result["model_id"] = result["model_id"].replace(MODEL_ID_ALIASES)
    return result


def parse_registry(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    pattern = re.compile(
        r"^\s*-\s*\{model_id:\s*([^,]+),\s*family:\s*([^,]+),\s*"
        r"kind:\s*([^,]+),\s*path:\s*([^}]+)\}"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            model_id, family, kind, model_path = (
                item.strip() for item in match.groups()
            )
            rows.append(
                {
                    "model_id": model_id,
                    "family": family,
                    "model_type": kind,
                }
            )
    if not rows:
        raise ValueError(f"no roster entries found in {path}")
    return pd.DataFrame(rows).assign(roster_order=lambda frame: range(len(frame)))


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", dtype=str)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).with_name("model_registry_34.yaml"),
        help="model roster (defaults to the active full S38/S39 roster)",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)
    roster = parse_registry(args.registry.expanduser().resolve())
    summary = read_tsv(result_dir / "model_summary.tsv")
    scores = read_tsv(result_dir / "locus_scores.tsv")
    bootstrap = read_tsv(result_dir / "bootstrap_summary.tsv")
    blocked = read_tsv(result_dir / "blocked_checkpoint_summary.tsv")
    summary = canonicalize_model_ids(summary)
    scores = orient_alleles(canonicalize_model_ids(scores))
    bootstrap = canonicalize_model_ids(bootstrap)
    if summary["model_id"].duplicated().any():
        raise ValueError("model summary has duplicate canonical model IDs")
    if scores.duplicated(["model_id", "variant_id"]).any():
        raise ValueError("locus scores have duplicate canonical model x locus rows")
    if (
        bootstrap.loc[bootstrap["metric"].eq("median_score"), "model_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "median-score bootstrap summary has duplicate canonical model IDs"
        )

    summary = numeric(
        summary,
        [
            "n_loci",
            "positive_loci",
            "accuracy",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "one_sided_exact_binom_p",
            "median_score",
            "q1_score",
            "q3_score",
            "iqr_score",
            "wilcoxon_statistic",
            "wilcoxon_one_sided_p",
        ],
    )
    bootstrap = bootstrap.loc[bootstrap["metric"].eq("median_score")].copy()
    bootstrap = numeric(
        bootstrap,
        ["observed", "ci_low", "ci_high", "reps", "seed"],
    ).rename(
        columns={
            "observed": "bootstrap_median_score_observed",
            "ci_low": "median_score_bootstrap_ci_low",
            "ci_high": "median_score_bootstrap_ci_high",
        }
    )
    scores = numeric(
        scores,
        [
            "pos",
            "window_bp",
            "center_index",
            "tokenized_length",
            "masked_token_id",
            "selected_logprob",
            "other_logprob",
            "locus_score",
        ],
    )
    scores["directional_correct"] = (scores["locus_score"] > 0).astype("Int64")
    scores["score_definition"] = "selected_logprob - other_logprob"
    scores["benchmark_status"] = scores["status"].fillna("UNKNOWN")

    registry_ids = set(roster["model_id"].astype(str))
    for table_name, frame in (
        ("summary", summary),
        ("locus scores", scores),
        ("bootstrap", bootstrap),
    ):
        unknown = sorted(set(frame["model_id"].astype(str)) - registry_ids)
        if unknown:
            raise ValueError(
                f"{table_name} contains model IDs outside the registry: {unknown[:8]}"
            )
    merged = roster.merge(
        summary, on="model_id", how="left", suffixes=("", "_run")
    ).merge(
        bootstrap[
            [
                "model_id",
                "median_score_bootstrap_ci_low",
                "median_score_bootstrap_ci_high",
            ]
        ],
        on="model_id",
        how="left",
    )
    merged["display_model"] = merged["model_id"].map(canonical_model_name)
    merged["status"] = merged["n_loci"].notna().map({True: "PASS", False: "BLOCKED"})
    merged["scoring_definition"] = merged["status"].map(
        {
            "PASS": "selected_logprob - other_logprob",
            "BLOCKED": "",
        }
    )
    merged["window_bp"] = 1024
    merged["center_index"] = 512
    merged["n_loci_expected"] = 30
    passed = merged["status"].eq("PASS")
    merged.loc[passed, "rank_by_accuracy"] = (
        merged.loc[passed, "accuracy"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    merged.loc[~passed, "rank_by_accuracy"] = pd.NA
    s38_columns = [
        "rank_by_accuracy",
        "model_id",
        "display_model",
        "family",
        "model_type",
        "status",
        "n_loci",
        "n_loci_expected",
        "positive_loci",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "one_sided_exact_binom_p",
        "median_score",
        "q1_score",
        "q3_score",
        "iqr_score",
        "wilcoxon_statistic",
        "wilcoxon_one_sided_p",
        "median_score_bootstrap_ci_low",
        "median_score_bootstrap_ci_high",
        "scoring_definition",
        "window_bp",
        "center_index",
    ]
    s38 = merged[s38_columns + ["roster_order"]].sort_values(
        ["status", "rank_by_accuracy", "roster_order"],
        ascending=[True, True, True],
        na_position="last",
    )
    s38[s38_columns].to_csv(
        tables_dir
        / "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv",
        index=False,
    )

    score_columns = [
        "model_id",
        "family",
        "model_type",
        "variant_id",
        "chrom",
        "pos",
        "gene",
        "benchmark_tier",
        "selected_allele",
        "other_allele",
        "target_population",
        "qa_status",
        "qa_reason",
        "window_bp",
        "center_index",
        "tokenized_length",
        "changed_token_positions",
        "selected_logprob",
        "other_logprob",
        "locus_score",
        "directional_correct",
        "score_definition",
        "benchmark_status",
    ]
    score_table = scores.merge(
        roster[["model_id", "family", "model_type", "roster_order"]],
        on="model_id",
        how="left",
        suffixes=("", "_registry"),
    )
    if "family_registry" in score_table:
        score_table["family"] = score_table["family"].fillna(score_table["family_registry"])
    score_table.sort_values(
        ["roster_order", "chrom", "pos"], inplace=True, kind="stable"
    )
    if set(s38["model_id"].astype(str)) != registry_ids:
        raise ValueError("S38 model IDs do not match the canonical registry")
    if score_table["roster_order"].isna().any():
        raise ValueError("S39 contains model IDs outside the canonical registry")
    score_table[score_columns].to_csv(
        tables_dir
        / "S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv",
        index=False,
    )

    print(
        f"wrote S38-S39 to {tables_dir}; "
        f"models={len(roster)}, scored={int(summary['model_id'].nunique())}, "
        f"locus_rows={len(scores)}, blocked={len(blocked)}"
    )


if __name__ == "__main__":
    main()
