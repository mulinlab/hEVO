#!/usr/bin/env python3
"""Summarize a fixed, model-blind Rare Variants stratum grid.

The grid is deliberately exhaustive for the declared features.  It is an
exploratory sensitivity analysis, not a replacement for R0/R1 and not a
license to report a favorable subset as the overall Rare Variants result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable

from sklearn.metrics import average_precision_score


FULL_ROSTER_SIZE = 32
TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
BASES = ("A", "C", "G", "T")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_delimited(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def feature_row(row: dict[str, str]) -> dict[str, str]:
    ref_seq = row["refSeq"].strip().upper()
    alt_seq = row["altSeq"].strip().upper()
    if len(ref_seq) != 1023 or len(alt_seq) != 1023:
        raise ValueError(f"{row['SeqID']}: expected 1023-bp sequences")
    differences = [
        i for i, (ref, alt) in enumerate(zip(ref_seq, alt_seq)) if ref != alt
    ]
    if differences != [511]:
        raise ValueError(f"{row['SeqID']}: sequences must differ only at center")
    ref, alt = ref_seq[511], alt_seq[511]
    gc_fraction = sum(base in "GC" for base in ref_seq) / len(ref_seq)
    cpg = (ref, alt) == ("C", "T") and ref_seq[510] == "G"
    cpg = cpg or ((ref, alt) == ("G", "A") and ref_seq[512] == "C")
    return {
        "mutation": f"{ref}>{alt}",
        "transition": "transition" if (ref, alt) in TRANSITIONS else "transversion",
        "ref_base": ref,
        "alt_base": alt,
        "cpg": "CpG_transition" if cpg else "non_CpG",
        "gc_fixed": "GC<0.40"
        if gc_fraction < 0.40
        else ("GC_0.40-0.60" if gc_fraction < 0.60 else "GC>=0.60"),
        "chrom": str(row["chrom"]),
    }


def level_sort(field: str, level: str) -> tuple[int, object]:
    if field == "chrom" and level.isdigit():
        return (0, int(level))
    return (1, level)


def declared_grid(
    rows: list[dict[str, str]],
) -> list[tuple[str, Callable[[dict[str, str]], bool]]]:
    features = {row["SeqID"]: feature_row(row) for row in rows}
    grid: list[tuple[str, Callable[[dict[str, str]], bool]]] = [("all", lambda _: True)]
    for field in (
        "transition",
        "mutation",
        "ref_base",
        "alt_base",
        "cpg",
        "gc_fixed",
        "chrom",
    ):
        observed = sorted(
            {values[field] for values in features.values()},
            key=lambda x: level_sort(field, x),
        )
        for level in observed:
            grid.append(
                (
                    f"{field}={level}",
                    lambda values, field=field, level=level: values[field] == level,
                )
            )
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    script_path = Path(__file__).resolve()

    dataset = read_delimited(args.dataset)
    if not dataset:
        raise ValueError("dataset is empty")
    required = {"SeqID", "refSeq", "altSeq", "label", "chrom", "pos"}
    if set(dataset[0]) != required:
        raise ValueError(f"unexpected dataset schema: {list(dataset[0])}")
    ids = [row["SeqID"] for row in dataset]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset SeqID values are not unique")
    labels = {row["SeqID"]: int(row["label"]) for row in dataset}
    features = {row["SeqID"]: feature_row(row) for row in dataset}

    score_paths = sorted(args.scores_dir.glob("*.tsv"))
    if len(score_paths) != FULL_ROSTER_SIZE:
        raise ValueError(
            f"expected {FULL_ROSTER_SIZE} score files, found {len(score_paths)}"
        )
    scores: dict[str, dict[str, float]] = {}
    score_hashes: dict[str, str] = {}
    for path in score_paths:
        score_rows = read_delimited(path)
        by_id = {row.get("variant_id", ""): row for row in score_rows}
        if len(by_id) != len(ids) or set(by_id) != set(ids):
            raise ValueError(f"{path}: score ID set does not match dataset")
        model_names = {row.get("model", "") for row in score_rows}
        if len(model_names) != 1 or "" in model_names:
            raise ValueError(f"{path}: score model field is not constant")
        model = next(iter(model_names))
        if model in scores:
            raise ValueError(f"duplicate model: {model}")
        for sid, row in by_id.items():
            if int(row["label"]) != labels[sid] or row.get("score_mode") != "neg_llr":
                raise ValueError(f"{path}: label or score_mode mismatch for {sid}")
            float(row["benchmark_score"])
        scores[model] = {
            sid: float(row["benchmark_score"]) for sid, row in by_id.items()
        }
        score_hashes[model] = sha256(path)
    if len(scores) != FULL_ROSTER_SIZE:
        raise ValueError(
            f"expected {FULL_ROSTER_SIZE} unique models, found {len(scores)}"
        )

    grid = declared_grid(dataset)
    output_rows: list[dict[str, object]] = []
    stratum_status: list[dict[str, object]] = []
    for stratum, predicate in grid:
        selected = [sid for sid in ids if predicate(features[sid])]
        counts = Counter(labels[sid] for sid in selected)
        status = "PASS" if counts[0] and counts[1] else "BLOCKED_SINGLE_CLASS"
        entry = {
            "stratum": stratum,
            "rows": len(selected),
            "positive": counts[1],
            "negative": counts[0],
            "baseline_auprc": counts[1] / len(selected) if selected else None,
            "status": status,
        }
        stratum_status.append(entry)
        if status != "PASS":
            continue
        ranking = sorted(
            (
                (
                    average_precision_score(
                        [labels[sid] for sid in selected],
                        [scores[model][sid] for sid in selected],
                    ),
                    model,
                )
                for model in scores
            ),
            key=lambda item: (-item[0], item[1]),
        )
        ranks = {model: rank for rank, (_, model) in enumerate(ranking, start=1)}
        for auprc, model in ranking:
            output_rows.append(
                {
                    "stratum": stratum,
                    "rows": len(selected),
                    "positive": counts[1],
                    "negative": counts[0],
                    "baseline_auprc": counts[1] / len(selected),
                    "model": model,
                    "auprc": auprc,
                    "rank": ranks[model],
                    "rank_status": "FULL_ROSTER",
                    "score_mode": "neg_llr",
                }
            )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as handle:
        fields = list(output_rows[0])
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "schema": "hevo_rare_fixed_strata_summary_v1",
        "status": "PASS",
        "analysis_status": "EXPLORATORY",
        "prospective_freeze_status": "NOT_PROSPECTIVE_AFTER_BASELINE",
        "interpretation_note": "All declared strata are reported; this does not replace R0/R1 or establish overall Rare Variants rank 1. The grid was generated after baseline scores existed, so favorable strata remain exploratory.",
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "rows": len(dataset),
        "positive": sum(labels.values()),
        "negative": len(dataset) - sum(labels.values()),
        "model_scores_used_for_stratum_definition": False,
        "roster_size": len(scores),
        "models": sorted(scores),
        "score_hashes": score_hashes,
        "score_mode": "neg_llr",
        "features": {
            "mutation": "center ref>alt substitution",
            "transition": "A<->G or C<->T; all other SNVs are transversion",
            "cpg": "C>T with upstream G or G>A with downstream C",
            "gc_fixed": "GC<0.40, 0.40<=GC<0.60, GC>=0.60",
            "chrom": "observed chromosome labels",
        },
        "strata": stratum_status,
        "evaluated_strata": sum(item["status"] == "PASS" for item in stratum_status),
        "blocked_strata": sum(item["status"] != "PASS" for item in stratum_status),
        "output_tsv": str(args.output_tsv.resolve()),
        "output_tsv_sha256": sha256(args.output_tsv),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "roster_size": len(scores),
                "evaluated_strata": manifest["evaluated_strata"],
                "blocked_strata": manifest["blocked_strata"],
                "output_tsv": str(args.output_tsv.resolve()),
                "manifest": str(args.manifest.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
