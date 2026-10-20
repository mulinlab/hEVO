#!/usr/bin/env python3
"""Build a model-blind Rare Variants arm with external benign controls.

The current R0/R1 arms are immutable.  This builder creates an explicitly
separate exploratory arm by retaining every positive in R0 and selecting one
external ClinVar-benign control per positive.  Controls are filtered before
any hEvo score is read and are selected deterministically within chromosome
and substitution-class strata using sequence GC matching.

The script deliberately does not inspect model scores.  A training-overlap
exclusion list may be supplied (for example, exact variants found in 1KG).
The manifest records any remaining provenance limitation; an arm with an
unverified UKB overlap check cannot be promoted to a formal no-leakage result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


WINDOW_LEN = 1023
CENTER = 511
SEED = 42
DNA = frozenset("ACGT")
OUTPUT_COLUMNS = ["SeqID", "refSeq", "altSeq", "label", "chrom", "pos"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chrom(value: object) -> str:
    text = str(value).strip()
    return text[3:] if text.startswith("chr") else text


def variant_key(
    row: pd.Series, ref_col: str, alt_col: str
) -> tuple[str, int, str, str]:
    return (
        chrom(row["chrom"]),
        int(row["pos"]),
        str(row[ref_col])[CENTER],
        str(row[alt_col])[CENTER],
    )


def locus_key(row: pd.Series) -> tuple[str, int]:
    return chrom(row["chrom"]), int(row["pos"])


def validate_sequences(
    frame: pd.DataFrame, ref_col: str, alt_col: str, name: str
) -> None:
    if frame.empty:
        raise ValueError(f"{name} is empty")
    for column in (ref_col, alt_col):
        lengths = frame[column].astype(str).str.len()
        if not (lengths == WINDOW_LEN).all():
            raise ValueError(f"{name}.{column} contains non-{WINDOW_LEN}-bp sequences")
        bad = frame[column].astype(str).map(lambda value: set(value.upper()) - DNA)
        if bad.map(bool).any():
            raise ValueError(f"{name}.{column} contains non-ACGT bases")
    differences = []
    for ref, alt in zip(frame[ref_col].astype(str), frame[alt_col].astype(str)):
        indices = [
            i
            for i, (left, right) in enumerate(zip(ref.upper(), alt.upper()))
            if left != right
        ]
        differences.append(indices)
    if any(len(indices) != 1 or indices[0] != CENTER for indices in differences):
        raise ValueError(f"{name} contains a non-central single-nucleotide change")
    if frame["SeqID"].astype(str).duplicated().any():
        raise ValueError(f"{name} contains duplicate SeqID values")


def read_training_exclusions(path: Path | None) -> set[tuple[str, int, str, str]]:
    if path is None:
        return set()
    exclusions: set[tuple[str, int, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number} requires chrom, pos, ref, alt")
            if (
                fields[0].strip().lower() in {"chrom", "chr"}
                and fields[1].strip().lower() == "pos"
            ):
                continue
            exclusions.add(
                (chrom(fields[0]), int(fields[1]), fields[2].upper(), fields[3].upper())
            )
    return exclusions


def stable_key(seq_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{seq_id}".encode("utf-8")).hexdigest()


def transition_class(ref: str, alt: str) -> str:
    return (
        "transition"
        if (ref, alt) in {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
        else "transversion"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an external benign-control Rare Variants arm without reading model scores."
    )
    parser.add_argument(
        "--positive-dataset",
        type=Path,
        required=True,
        help="Paired-sequence CSV providing label=1 rows",
    )
    parser.add_argument(
        "--negative-dataset",
        type=Path,
        required=True,
        help="External paired-sequence CSV with label=0 controls",
    )
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        help="Optional model-blind dataset whose exact variants/loci are excluded from the arm",
    )
    parser.add_argument(
        "--training-exclusions",
        type=Path,
        help="Optional TSV of exact training variants: chrom, pos, ref, alt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--ukb-overlap-status",
        choices=("NOT_CHECKED", "NOT_VERIFIABLE", "CHECKED_ZERO"),
        default="NOT_VERIFIABLE",
        help="Provenance status for UKB training-source overlap; not inferred by this builder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = pd.read_csv(args.positive_dataset)
    negative_source = pd.read_csv(args.negative_dataset)
    positive = positive.rename(columns={"refseq": "refSeq", "altseq": "altSeq"})
    negative_source = negative_source.rename(
        columns={"seqID": "SeqID", "refseq": "refSeq", "altseq": "altSeq"}
    )

    required = set(OUTPUT_COLUMNS)
    if required - set(positive.columns):
        raise ValueError(
            f"positive dataset missing columns: {sorted(required - set(positive.columns))}"
        )
    if {"SeqID", "refSeq", "altSeq", "label", "chrom", "pos"} - set(
        negative_source.columns
    ):
        raise ValueError(
            "negative dataset must contain SeqID/refseq/altseq/label/chrom/pos"
        )

    positive = positive[positive["label"].astype(int).eq(1)].copy()
    negative_source = negative_source[negative_source["label"].astype(int).eq(0)].copy()
    positive_input_count = len(positive)
    validate_sequences(positive, "refSeq", "altSeq", "positive")
    validate_sequences(negative_source, "refSeq", "altSeq", "negative_source")

    exclusion_path = args.exclude_dataset or args.positive_dataset
    exclusion_frame = pd.read_csv(exclusion_path).rename(
        columns={"refseq": "refSeq", "altseq": "altSeq"}
    )
    if set(OUTPUT_COLUMNS) - set(exclusion_frame.columns):
        raise ValueError(
            f"exclude dataset missing columns: {sorted(set(OUTPUT_COLUMNS) - set(exclusion_frame.columns))}"
        )
    current_keys = {
        variant_key(row, "refSeq", "altSeq") for _, row in exclusion_frame.iterrows()
    }
    current_loci = {locus_key(row) for _, row in exclusion_frame.iterrows()}
    counts = Counter()
    if args.exclude_dataset is not None:
        positive_keys = positive.apply(
            lambda row: variant_key(row, "refSeq", "altSeq"), axis=1
        )
        positive_loci = positive.apply(locus_key, axis=1)
        positive_overlap = positive_keys.isin(current_keys) | positive_loci.isin(
            current_loci
        )
        counts["excluded_positive_exact_or_locus"] = int(positive_overlap.sum())
        positive = positive.loc[~positive_overlap].copy()
        if positive.empty:
            raise ValueError(
                "No positive rows remain after exclude-dataset overlap filtering"
            )
    current_keys.update(
        variant_key(row, "refSeq", "altSeq") for _, row in positive.iterrows()
    )
    current_loci.update(locus_key(row) for _, row in positive.iterrows())
    negative_source["_vkey"] = [
        variant_key(row, "refSeq", "altSeq") for _, row in negative_source.iterrows()
    ]
    negative_source["_lkey"] = [locus_key(row) for _, row in negative_source.iterrows()]
    exclusions = read_training_exclusions(args.training_exclusions)

    eligible = []
    for _, row in negative_source.iterrows():
        key = row["_vkey"]
        locus = row["_lkey"]
        if key in current_keys:
            counts["excluded_current_exact"] += 1
            continue
        if locus in current_loci:
            counts["excluded_current_locus"] += 1
            continue
        if key in exclusions:
            counts["excluded_training_exact"] += 1
            continue
        eligible.append(row)
    eligible = pd.DataFrame(eligible)
    if eligible.empty:
        raise ValueError("No eligible external benign controls remain")
    eligible["gc"] = (
        eligible["refSeq"]
        .astype(str)
        .str.count("G")
        .add(eligible["refSeq"].astype(str).str.count("C"))
        / WINDOW_LEN
    )
    positive["gc"] = (
        positive["refSeq"]
        .astype(str)
        .str.count("G")
        .add(positive["refSeq"].astype(str).str.count("C"))
        / WINDOW_LEN
    )
    positive["mutation"] = (
        positive["refSeq"].str[CENTER] + ">" + positive["altSeq"].str[CENTER]
    )
    eligible["mutation"] = (
        eligible["refSeq"].str[CENTER] + ">" + eligible["altSeq"].str[CENTER]
    )
    positive["substitution_class"] = [
        transition_class(ref, alt)
        for ref, alt in zip(
            positive["refSeq"].str[CENTER], positive["altSeq"].str[CENTER]
        )
    ]
    eligible["substitution_class"] = [
        transition_class(ref, alt)
        for ref, alt in zip(
            eligible["refSeq"].str[CENTER], eligible["altSeq"].str[CENTER]
        )
    ]
    positive["chrom_norm"] = positive["chrom"].map(chrom)
    eligible["chrom_norm"] = eligible["chrom"].map(chrom)

    negative_rows = []
    match_records = []
    shortfall = []
    group_columns = ["chrom_norm", "substitution_class"]
    for group_key, pos_group in positive.groupby(group_columns, sort=True):
        neg_group = eligible[
            (eligible["chrom_norm"] == group_key[0])
            & (eligible["substitution_class"] == group_key[1])
        ].copy()
        if len(neg_group) < len(pos_group):
            shortfall.append(
                {
                    "group": list(group_key),
                    "positive": len(pos_group),
                    "eligible_negative": len(neg_group),
                }
            )
            continue
        # Match sequence GC by deterministic quantile selection.  This uses no
        # label-derived or model-derived quantity beyond the frozen strata.
        neg_group = neg_group.assign(
            _tie=neg_group["SeqID"].map(lambda value: stable_key(str(value), args.seed))
        )
        neg_group = neg_group.sort_values(["gc", "_tie"], kind="mergesort")
        positions = [
            min(len(neg_group) - 1, int((j + 0.5) * len(neg_group) / len(pos_group)))
            for j in range(len(pos_group))
        ]
        selected = neg_group.iloc[positions].copy()
        pos_sorted = pos_group.sort_values(
            ["gc", "SeqID"], kind="mergesort"
        ).reset_index(drop=True)
        selected = selected.reset_index(drop=True)
        for pos_row, neg_row in zip(
            pos_sorted.to_dict("records"), selected.to_dict("records")
        ):
            negative_rows.append(neg_row)
            match_records.append(
                {
                    "chrom": group_key[0],
                    "substitution_class": group_key[1],
                    "positive_seqid": str(pos_row["SeqID"]),
                    "negative_seqid": str(neg_row["SeqID"]),
                    "positive_gc": float(pos_row["gc"]),
                    "negative_gc": float(neg_row["gc"]),
                    "absolute_gc_difference": abs(
                        float(pos_row["gc"]) - float(neg_row["gc"])
                    ),
                }
            )

    if shortfall:
        raise ValueError(
            json.dumps(
                {"status": "BLOCKED_MATCH_CAPACITY", "shortfall": shortfall},
                sort_keys=True,
            )
        )
    if len(negative_rows) != len(positive):
        raise AssertionError("one-to-one matching did not retain every positive")

    output_rows = []
    for row in positive.to_dict("records"):
        output_rows.append({column: row[column] for column in OUTPUT_COLUMNS})
    for row in negative_rows:
        output_rows.append(
            {
                "SeqID": row["SeqID"],
                "refSeq": row["refSeq"],
                "altSeq": row["altSeq"],
                "label": 0,
                "chrom": row["chrom"],
                "pos": int(row["pos"]),
            }
        )
    output = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    validate_sequences(
        output[output["label"].astype(int).eq(1)], "refSeq", "altSeq", "output_positive"
    )
    validate_sequences(
        output[output["label"].astype(int).eq(0)], "refSeq", "altSeq", "output_negative"
    )
    output_keys = [variant_key(row, "refSeq", "altSeq") for _, row in output.iterrows()]
    if len(output_keys) != len(set(output_keys)):
        raise ValueError("output contains duplicate exact variants")
    output.to_csv(args.output, index=False)

    match_frame = pd.DataFrame(match_records)
    qa = {
        "schema": "hevo_rare_external_benign_arm_v1",
        "status": "EXPLORATORY_NOT_FORMAL_NO_LEAKAGE_PROOF"
        if args.ukb_overlap_status != "CHECKED_ZERO"
        else "CANDIDATE_FORMAL",
        "selection_is_model_blind": True,
        "matching_rule": "one external ClinVar-benign control per retained positive within chromosome+transition/transversion class; deterministic GC-ordered quantile selection; SHA256 tie-break",
        "positive": int(output["label"].sum()),
        "negative": int((output["label"] == 0).sum()),
        "total": len(output),
        "positive_input_rows": positive_input_count,
        "exclude_dataset": None
        if args.exclude_dataset is None
        else args.exclude_dataset.name,
        "exclude_dataset_rows": len(exclusion_frame),
        "eligible_negative_pool": len(eligible),
        "exclusion_counts": dict(counts),
        "training_exclusions_exact": len(exclusions),
        "ukb_overlap_status": args.ukb_overlap_status,
        "matching_groups": int(
            match_frame[["chrom", "substitution_class"]].drop_duplicates().shape[0]
        ),
        "gc_absolute_difference_mean": float(
            match_frame["absolute_gc_difference"].mean()
        ),
        "gc_absolute_difference_max": float(
            match_frame["absolute_gc_difference"].max()
        ),
        "chromosome_counts": {
            "positive": dict(
                sorted(
                    Counter(output.loc[output.label.eq(1), "chrom"].map(chrom)).items()
                )
            ),
            "negative": dict(
                sorted(
                    Counter(output.loc[output.label.eq(0), "chrom"].map(chrom)).items()
                )
            ),
        },
    }
    args.qa_report.parent.mkdir(parents=True, exist_ok=True)
    args.qa_report.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "hevo_rare_external_benign_arm_manifest_v1",
        "arm_id": "R2_DDD_CLINVAR_BENIGN_1TO1_CHROM_TV_GC",
        "status": qa["status"],
        "inputs": {
            "positive_dataset": {
                "name": args.positive_dataset.name,
                "sha256": sha256(args.positive_dataset),
            },
            "negative_dataset": {
                "name": args.negative_dataset.name,
                "sha256": sha256(args.negative_dataset),
            },
            "exclude_dataset": None
            if args.exclude_dataset is None
            else {
                "name": args.exclude_dataset.name,
                "sha256": sha256(args.exclude_dataset),
            },
            "training_exclusions": None
            if args.training_exclusions is None
            else {
                "name": args.training_exclusions.name,
                "sha256": sha256(args.training_exclusions),
            },
        },
        "frozen_protocol": {
            "seed": args.seed,
            "window_length": WINDOW_LEN,
            "positive_rule": (
                "retain every label=1 row from supplied positive dataset"
                if args.exclude_dataset is None
                else "retain every supplied label=1 row after excluding exact/locus overlaps with exclude-dataset"
            ),
            "negative_rule": "external DDD public-research ClinVar-benign label=0 SNVs; exclude exclude-dataset/positive exact variants/loci and supplied exact training overlaps",
            "matching_rule": qa["matching_rule"],
            "model_scores_read_during_build": False,
            "allowed_models": ["hEvo-base", "hEvo-large"],
            "score_rule": "benchmark_score=-raw_llr; score_mode=neg_llr",
        },
        "qa": qa,
        "output": {"name": args.output.name, "sha256": sha256(args.output)},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "manifest": str(args.manifest), "qa": qa},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
