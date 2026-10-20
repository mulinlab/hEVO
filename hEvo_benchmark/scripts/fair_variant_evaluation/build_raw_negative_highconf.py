#!/usr/bin/env python3
"""Build a model-blind high-confidence CausalDB arm from the raw-negative pool.

This is an exploratory estimand and must not replace the full CausalDB arm.
All positives satisfying the declared PP/beta/MAF thresholds are retained;
negative controls are sampled without replacement within frozen MAF bins after
exact-variant and same-locus exclusion.  No model package or score file is
loaded, so model-directed selection is impossible by construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


MAF_BINS = (
    (0.0, 0.01, "[0,0.01)"),
    (0.01, 0.05, "[0.01,0.05)"),
    (0.05, 0.10, "[0.05,0.10)"),
    (0.10, 0.20, "[0.10,0.20)"),
    (0.20, 0.30, "[0.20,0.30)"),
    (0.30, 0.40, "[0.30,0.40)"),
    (0.40, 0.50, "[0.40,0.50]"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def chrom(row: dict[str, str]) -> str:
    value = (row.get("chrom_hg38") or row.get("chrom") or "").strip()
    return value[3:] if value.lower().startswith("chr") else value


def pos(row: dict[str, str]) -> str:
    return (row.get("pos_hg38") or row.get("pos") or "").strip()


def alleles(row: dict[str, str]) -> tuple[str, str]:
    ref = (row.get("canonical_ref") or row.get("ref") or "").strip().upper()
    alt = (row.get("canonical_alt") or row.get("alt") or "").strip().upper()
    if not ref or not alt:
        refseq = row.get("refseq", "").strip().upper()
        altseq = row.get("altseq", "").strip().upper()
        if len(refseq) != len(altseq) or not refseq or len(refseq) % 2 == 0:
            raise ValueError("row lacks explicit alleles and valid paired sequences")
        center = len(refseq) // 2
        ref, alt = refseq[center], altseq[center]
    if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
        raise ValueError(f"invalid alleles: {ref!r}, {alt!r}")
    return ref, alt


def variant_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    ref, alt = alleles(row)
    return chrom(row), pos(row), ref, alt


def locus_key(row: dict[str, str]) -> tuple[str, str]:
    return chrom(row), pos(row)


def maf_bin(row: dict[str, str]) -> str:
    value = float(row["maf"])
    if not 0.0 <= value <= 0.5:
        raise ValueError(f"MAF outside [0,0.5]: {value}")
    for index, (left, right, label) in enumerate(MAF_BINS):
        if (index < len(MAF_BINS) - 1 and left <= value < right) or (
            index == len(MAF_BINS) - 1 and left <= value <= right
        ):
            return label
    raise ValueError(f"MAF did not fall into a bin: {value}")


def eligible(row: dict[str, str], maf_min: float) -> bool:
    text = str(row.get("maf", "")).strip()
    if not text:
        return False
    try:
        value = float(text)
    except ValueError:
        return False
    return value >= maf_min


def build_arm(
    positives: list[dict[str, str]],
    negatives: list[dict[str, str]],
    pp_threshold: float,
    beta_threshold: float,
    maf_min: float,
    ratio: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    selected = [
        row
        for row in positives
        if float(row["avgPP"]) >= pp_threshold
        and abs(float(row["beta"])) >= beta_threshold
        and eligible(row, maf_min)
    ]
    positive_variants = {variant_key(row) for row in selected}
    positive_loci = {locus_key(row) for row in selected}
    pool = [
        row
        for row in negatives
        if eligible(row, maf_min)
        and variant_key(row) not in positive_variants
        and locus_key(row) not in positive_loci
    ]
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        groups[maf_bin(row)].append(row)
    targets: defaultdict[str, int] = defaultdict(int)
    for row in selected:
        targets[maf_bin(row)] += ratio
    deficits = {
        group: target - len(groups[group])
        for group, target in sorted(targets.items())
        if len(groups[group]) < target
    }
    if deficits:
        raise RuntimeError(
            f"BLOCKED_MATCH_CAPACITY:{json.dumps(deficits, sort_keys=True)}"
        )
    rng = random.Random(seed)
    controls: list[dict[str, str]] = []
    for group, target in sorted(targets.items()):
        candidates = sorted(
            groups[group], key=lambda row: (variant_key(row), row.get("seqID", ""))
        )
        controls.extend(rng.sample(candidates, target))
    output = selected + controls
    rng.shuffle(output)
    metadata = {
        "positive_source_rows": len(positives),
        "positive_selected_rows": len(selected),
        "positive_excluded_from_full": len(positives) - len(selected),
        "negative_pool_rows": len(pool),
        "negative_selected_rows": len(controls),
        "rows": len(output),
        "positive": len(selected),
        "negative": len(controls),
        "baseline_auprc": len(selected) / len(output),
        "matching": "MAF_bin_without_replacement",
        "matching_ratio": f"1:{ratio}",
        "exact_variant_excluded": True,
        "same_locus_excluded": True,
        "seed": seed,
        "selection_model_scores_used": False,
        "status": "PASS",
    }
    return output, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-canonical", type=Path, required=True)
    parser.add_argument("--negative-canonical-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pp-threshold", type=float, default=0.99)
    parser.add_argument("--beta-threshold", type=float, default=1.0)
    parser.add_argument("--maf-min", type=float, default=0.01)
    parser.add_argument("--ratio", type=int, choices=(1, 5), default=1)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if (
        not 0 <= args.pp_threshold <= 1
        or args.beta_threshold < 0
        or not 0 <= args.maf_min <= 0.5
    ):
        raise ValueError("invalid PP, beta, or MAF threshold")
    positives = [
        row for row in read_csv(args.positive_canonical) if row.get("label") == "1"
    ]
    negatives = [
        row for row in read_csv(args.negative_canonical_pool) if row.get("label") == "0"
    ]
    output, metadata = build_arm(
        positives,
        negatives,
        args.pp_threshold,
        args.beta_threshold,
        args.maf_min,
        args.ratio,
        args.seed,
    )
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    arm = f"P_raw_pp{args.pp_threshold:g}_beta{args.beta_threshold:g}_common_maf_1to{args.ratio}"
    dataset = root / f"{arm}.csv"
    fields = list(output[0])
    for row in output:
        for field in row:
            if field not in fields:
                fields.append(field)
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    keys = [variant_key(row) for row in output]
    loci = [locus_key(row) for row in output]
    manifest = {
        "schema": "hevo_causaldb_prospective_raw_negative_v1",
        "status": "PASS",
        "arm_id": arm,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "positive_source": str(args.positive_canonical.resolve()),
        "positive_source_sha256": sha256(args.positive_canonical),
        "negative_canonical_pool": str(args.negative_canonical_pool.resolve()),
        "negative_canonical_pool_sha256": sha256(args.negative_canonical_pool),
        "pp_threshold": args.pp_threshold,
        "abs_beta_threshold": args.beta_threshold,
        "maf_min": args.maf_min,
        "model_scores_used_for_selection": False,
        "unique_variant": len(set(keys)),
        "unique_locus": len(set(loci)),
        **metadata,
    }
    manifest_path = root / f"{arm}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": arm,
                "dataset": str(dataset),
                "manifest": str(manifest_path),
                **metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
