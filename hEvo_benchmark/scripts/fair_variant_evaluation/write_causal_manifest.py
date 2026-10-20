#!/usr/bin/env python3
"""Write a model-blind manifest for the canonical CausalDB analysis arms.

The manifest is deliberately independent of score files.  It records the
complete denominator, label prevalence, variant/locus duplication, and the
pre-registered 29-model versus 32-model roster used by the final ranking.
Large datasets and the manifest itself should be written outside the Git
checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


CANONICAL_MODELS = [
    "AIDO.DNA-300M",
    "Caduceus-PH",
    "Caduceus-PS",
    "DNABERT",
    "DNABERT2",
    "GENA-LM-base",
    "GENA-LM-large",
    "GENERator-1.2B",
    "GENERator-3B",
    "GPN-MSA",
    "GPN-promoter",
    "GROVER",
    "Genos-1.2B",
    "HyenaDNA_tiny_1k",
    "Mistral-DNA-138M",
    "Mistral-DNA-17M",
    "Mistral-DNA-422M",
    "NT_2.5b_1kG",
    "NT_2.5b_MS",
    "NT_500m_1kG",
    "NT_500m_MS",
    "NTv3-100M-pre",
    "NTv3-650M-pre",
    "NTv3-8M-pre",
    "SpeciesLM",
    "evo2_1b_base",
    "evo2_7b_base",
    "hEvo-base",
    "hEvo-large",
]
EXTENSION_MODELS = ["GPN-Star-M", "GPN-Star-P", "GPN-Star-V"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chrom(row: dict[str, str]) -> str:
    value = row.get("chrom_hg38") or row.get("chrom") or ""
    value = value.strip()
    return value[3:] if value.lower().startswith("chr") else value


def pos(row: dict[str, str]) -> str:
    return (row.get("pos_hg38") or row.get("pos") or "").strip()


def alleles(row: dict[str, str]) -> tuple[str, str]:
    ref = (row.get("ref") or "").strip().upper()
    alt = (row.get("alt") or "").strip().upper()
    if not ref or not alt:
        refseq = (row.get("refseq") or row.get("refSeq") or "").strip().upper()
        altseq = (row.get("altseq") or row.get("altSeq") or "").strip().upper()
        if len(refseq) != len(altseq) or not refseq or len(refseq) % 2 == 0:
            raise ValueError(
                "dataset row lacks explicit alleles and valid paired sequences"
            )
        center = len(refseq) // 2
        ref, alt = refseq[center], altseq[center]
    return ref, alt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def summary(path: Path, arm_id: str) -> dict[str, object]:
    rows = read_rows(path)
    id_column = (
        "seqID"
        if "seqID" in rows[0]
        else "SeqID"
        if "SeqID" in rows[0]
        else "variant_id"
    )
    ids = [row.get(id_column, "") for row in rows]
    variant_keys = []
    loci = []
    for row in rows:
        c, p = chrom(row), pos(row)
        ref, alt = alleles(row)
        if not c or not p or not ref or not alt or ref == alt:
            raise ValueError(f"invalid hg38 identity in {path}: {row}")
        variant_keys.append((c, p, ref, alt))
        loci.append((c, p))
    labels = [int(row["label"]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError(f"{path} must contain both labels")
    positive = sum(label == 1 for label in labels)
    negative = len(labels) - positive
    positive_keys = {key for key, label in zip(variant_keys, labels) if label == 1}
    negative_keys = {key for key, label in zip(variant_keys, labels) if label == 0}
    positive_loci = {key for key, label in zip(loci, labels) if label == 1}
    negative_loci = {key for key, label in zip(loci, labels) if label == 0}
    return {
        "arm_id": arm_id,
        "dataset": str(path.resolve()),
        "dataset_sha256": sha256(path),
        "rows": len(rows),
        "positive": positive,
        "negative": negative,
        "baseline_auprc": positive / len(rows),
        "unique_id": len(set(ids)),
        "unique_variant": len(set(variant_keys)),
        "unique_locus": len(set(loci)),
        "duplicate_id_rows": len(rows) - len(set(ids)),
        "duplicate_variant_rows": len(rows) - len(set(variant_keys)),
        "duplicate_locus_rows": len(rows) - len(set(loci)),
        "positive_negative_exact_variant_overlap": len(positive_keys & negative_keys),
        "positive_negative_locus_overlap": len(positive_loci & negative_loci),
        "selection_model_scores_used": False,
    }


def parse_arm(value: str) -> tuple[str, Path]:
    arm_id, separator, path = value.partition(":")
    if not separator or not arm_id or not path:
        raise ValueError("--arm must be ARM_ID:DATASET_CSV")
    return arm_id, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", type=Path, required=True, help="canonical full CausalDB CSV"
    )
    parser.add_argument(
        "--arm", action="append", default=[], help="additional ARM_ID:CSV; repeatable"
    )
    parser.add_argument(
        "--source", type=Path, default=None, help="pre-canonical source CSV"
    )
    parser.add_argument("--canonicalization-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = [summary(args.full, "C0_full")]
    full_positive = arms[0]["positive"]
    full_negative = arms[0]["negative"]
    for value in args.arm:
        arm_id, path = parse_arm(value)
        item = summary(path, arm_id)
        item["positive_source_rows"] = full_positive
        item["positive_excluded_from_C0"] = int(full_positive) - int(item["positive"])
        item["negative_pool_rows"] = full_negative
        item["negative_selected_rows"] = item["negative"]
        arms.append(item)

    blocked = []
    for arm_id, ratio in (
        ("C1_strict_chr_maf_1to1", "1:1"),
        ("C2_strict_chr_maf_1to5", "1:5"),
    ):
        blocked.append(
            {
                "arm_id": arm_id,
                "status": "BLOCKED_MATCH_CAPACITY",
                "ratio": ratio,
                "reason": "strict chromosome+MAF matching capacity was insufficient; no relaxed substitute",
                "selection_model_scores_used": False,
            }
        )

    manifest: dict[str, object] = {
        "schema": "hevo_causaldb_fair_roster_manifest_v1",
        "status": "PASS",
        "assembly": "hg38",
        "sequence_length_bp": 1023,
        "score_selection_used": False,
        "matching": {
            "primary": "C0 full source cohort",
            "sensitivity": "pre-registered model-blind MAF-matched common-variant controls",
            "same_locus_excluded": True,
            "strict_arms": blocked,
        },
        "source": {
            "path": str((args.source or args.full).resolve()),
            "sha256": sha256(args.source or args.full),
        },
        "canonicalization_manifest": str(args.canonicalization_manifest.resolve())
        if args.canonicalization_manifest
        else None,
        "rosters": {
            "canonical_29": CANONICAL_MODELS,
            "extension_3": EXTENSION_MODELS,
            "extension_32": CANONICAL_MODELS + EXTENSION_MODELS,
        },
        "arms": arms,
        "blocked_arms": blocked,
        "ranking_rule": "point-estimate AUPRC within each arm; paired bootstrap CI for uncertainty",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "arms": len(arms),
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
