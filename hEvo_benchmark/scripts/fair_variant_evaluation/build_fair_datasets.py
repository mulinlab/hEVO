#!/usr/bin/env python3
"""Build the preregistered Causal/Rare fairness-analysis arms.

This script is deliberately model-blind: no score, ranking, checkpoint, or
model-derived annotation is read.  Outputs belong in an external run root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260913
WINDOW = 1023
LOW_FREQUENCY_MAX = 0.01
MAF_BIN_LABELS = (
    "missing",
    "[0,0.01]",
    "(0.01,0.05]",
    "(0.05,0.10]",
    "(0.10,0.20]",
    "(0.20,0.30]",
    "(0.30,0.40]",
    "(0.40,0.50]",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_chrom(value: str) -> str:
    value = str(value).strip()
    return value[3:] if value.lower().startswith("chr") else value


def variant_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    ref = row.get("ref") or row.get("refseq") or row.get("refSeq", "")[511:512]
    alt = row.get("alt") or row.get("altseq") or row.get("altSeq", "")[511:512]
    return (
        normalize_chrom(row["chrom"]),
        str(row["pos"]),
        ref.upper(),
        alt.upper(),
    )


def locus_key(row: dict[str, str]) -> tuple[str, str]:
    return normalize_chrom(row["chrom"]), str(row["pos"])


def validate_rows(rows: list[dict[str, str]], task: str) -> dict[str, object]:
    required = {"label", "chrom", "pos"}
    if task == "causal":
        required |= {"seqID", "refseq", "altseq", "ref", "alt", "maf"}
        id_col = "seqID"
        ref_col, alt_col = "refseq", "altseq"
    else:
        required |= {"SeqID", "refSeq", "altSeq"}
        id_col = "SeqID"
        ref_col, alt_col = "refSeq", "altSeq"
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"{task}: missing columns: {sorted(missing)}")
    ids = [row[id_col] for row in rows]
    keys = [variant_key(row) for row in rows]
    bad_length = []
    bad_pair = []
    bad_label = []
    for row in rows:
        refseq, altseq = row[ref_col].upper(), row[alt_col].upper()
        if len(refseq) != WINDOW or len(altseq) != WINDOW:
            bad_length.append(row[id_col])
        if len(refseq) == WINDOW and len(altseq) == WINDOW:
            if sum(a != b for a, b in zip(refseq, altseq)) != 1:
                bad_pair.append(row[id_col])
        if row["label"] not in {"0", "1"}:
            bad_label.append(row[id_col])
    if bad_length or bad_pair or bad_label:
        raise ValueError(
            f"{task}: invalid rows length={len(bad_length)} pair={len(bad_pair)} "
            f"label={len(bad_label)}"
        )
    return {
        "rows": len(rows),
        "positive": sum(row["label"] == "1" for row in rows),
        "negative": sum(row["label"] == "0" for row in rows),
        "unique_id": len(set(ids)),
        "unique_variant": len(set(keys)),
        "unique_locus": len({key[:2] for key in keys}),
        "duplicate_id_rows": len(rows) - len(set(ids)),
        "duplicate_variant_rows": len(rows) - len(set(keys)),
        "duplicate_locus_rows": len(rows) - len({key[:2] for key in keys}),
    }


def maf_bin(value: str) -> str:
    text = str(value).strip()
    if text in {"", "NULL", "None", "nan", "NaN"}:
        return "missing"
    maf = float(text)
    if maf < 0 or maf > 0.5:
        raise ValueError(f"MAF outside [0,0.5]: {value}")
    if maf <= 0.01:
        return "[0,0.01]"
    if maf <= 0.05:
        return "(0.01,0.05]"
    if maf <= 0.10:
        return "(0.05,0.10]"
    if maf <= 0.20:
        return "(0.10,0.20]"
    if maf <= 0.30:
        return "(0.20,0.30]"
    if maf <= 0.40:
        return "(0.30,0.40]"
    return "(0.40,0.50]"


def select_matched_controls(
    positives: list[dict[str, str]],
    negatives: list[dict[str, str]],
    ratio: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    positive_keys = {variant_key(row) for row in positives}
    positive_loci = {locus_key(row) for row in positives}
    eligible = [
        row
        for row in negatives
        if variant_key(row) not in positive_keys and locus_key(row) not in positive_loci
    ]
    candidates: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    targets: Counter[tuple[str, str]] = Counter()
    for row in positives:
        group = (normalize_chrom(row["chrom"]), maf_bin(row["maf"]))
        targets[group] += ratio
    for row in eligible:
        group = (normalize_chrom(row["chrom"]), maf_bin(row["maf"]))
        candidates[group].append(row)
    deficits = {
        f"{chrom}:{bin_name}": target - len(candidates.get((chrom, bin_name), []))
        for (chrom, bin_name), target in sorted(targets.items())
        if len(candidates.get((chrom, bin_name), [])) < target
    }
    if deficits:
        return [], {
            "status": "BLOCKED_MATCH_CAPACITY",
            "ratio": ratio,
            "target_negative_count": sum(targets.values()),
            "eligible_negative_count": len(eligible),
            "deficits": deficits,
            "selection_model_scores_used": False,
        }
    rng = random.Random(seed + ratio)
    selected: list[dict[str, str]] = []
    selected_by_group: dict[str, int] = {}
    for group in sorted(targets):
        pool = sorted(
            candidates[group],
            key=lambda row: variant_key(row) + (row.get("seqID", ""),),
        )
        count = targets[group]
        chosen = rng.sample(pool, count)
        selected.extend(chosen)
        selected_by_group[f"{group[0]}:{group[1]}"] = count
    if len({variant_key(row) for row in selected}) != len(selected):
        raise AssertionError("matched controls contain duplicate variant keys")
    rng.shuffle(selected)
    return selected, {
        "status": "PASS",
        "ratio": ratio,
        "target_negative_count": len(selected),
        "eligible_negative_count": len(eligible),
        "selected_by_chrom_maf": selected_by_group,
        "maf_bins": list(MAF_BIN_LABELS),
        "selection_model_scores_used": False,
    }


def summary(rows: list[dict[str, str]], task: str, arm: str) -> dict[str, object]:
    stats = validate_rows(rows, task)
    stats.update(
        {
            "arm": arm,
            "task": task,
            "baseline_auprc": stats["positive"] / stats["rows"],
            "label_values": sorted({row["label"] for row in rows}),
        }
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument(
        "--causal-control-pool",
        type=Path,
        default=None,
        help="Optional expanded canonical Causal control pool; must be model-blind and label=0.",
    )
    parser.add_argument("--rare", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not args.causal.is_file() or not args.rare.is_file():
        raise FileNotFoundError("both --causal and --rare must be existing files")

    causal = read_rows(args.causal)
    rare = read_rows(args.rare)
    validate_rows(causal, "causal")
    validate_rows(rare, "rare")
    if sum(row["label"] == "1" for row in causal) != 1117:
        raise ValueError("Causal positive count is not 1117")
    if sum(row["label"] == "1" for row in rare) != 1314:
        raise ValueError("Rare positive count is not 1314")
    control_source = args.causal
    control_rows = [row for row in causal if row["label"] == "0"]
    if args.causal_control_pool is not None:
        if not args.causal_control_pool.is_file():
            raise FileNotFoundError(args.causal_control_pool)
        expanded = read_rows(args.causal_control_pool)
        validate_rows(expanded, "causal")
        control_rows = [row for row in expanded if row["label"] == "0"]
        if not control_rows:
            raise ValueError("--causal-control-pool contains no label=0 rows")
        control_source = args.causal_control_pool

    root = args.output_root.resolve()
    dataset_dir = root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    causal_fields = list(causal[0])
    rare_fields = list(rare[0])
    arms: list[dict[str, object]] = []

    def add_arm(
        arm_id: str, task: str, rows: list[dict[str, str]], source: Path
    ) -> None:
        fields = causal_fields if task == "causal" else rare_fields
        output = dataset_dir / f"{arm_id}.csv"
        write_rows(output, rows, fields)
        arms.append(
            {
                "arm_id": arm_id,
                "task": task,
                "dataset": str(output),
                "dataset_sha256": sha256(output),
                "source": str(source.resolve()),
                "source_sha256": sha256(source),
                "summary": summary(rows, task, arm_id),
                "status": "PASS",
            }
        )

    # Keep the preregistered short estimand IDs stable.  Descriptive names
    # belong in task/source fields, not in the denominator identity.
    add_arm("C0", "causal", causal, args.causal)
    add_arm("R0", "rare", rare, args.rare)

    positives = [row for row in causal if row["label"] == "1"]
    negatives = control_rows
    for ratio, arm_id in ((1, "C1"), (5, "C2")):
        selected, match = select_matched_controls(
            positives, negatives, ratio, args.seed
        )
        if match["status"] != "PASS":
            arms.append(
                {
                    "arm_id": arm_id,
                    "task": "causal",
                    "status": match["status"],
                    "matching": match,
                    "source": str(control_source.resolve()),
                    "source_sha256": sha256(control_source),
                }
            )
            continue
        rng = random.Random(args.seed + ratio * 100)
        rows = [*positives, *selected]
        rng.shuffle(rows)
        add_arm(arm_id, "causal", rows, args.causal)
        arms[-1]["matching"] = match
        arms[-1]["control_source"] = str(control_source.resolve())
        arms[-1]["control_source_sha256"] = sha256(control_source)

    # R1 uses R0 scores and is evaluated at locus level; no second sequence file
    # is needed.  A deterministic representative is supplied for tools that
    # require a row-level registry, while the official R1 metric is aggregated.
    by_locus: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rare:
        by_locus[locus_key(row)].append(row)
    representatives = [
        sorted(rows, key=lambda row: row["SeqID"])[0] for rows in by_locus.values()
    ]
    arms.append(
        {
            "arm_id": "R1",
            "task": "rare",
            "status": "PASS",
            "evaluation_unit": "locus",
            "source_arm": "R0",
            "representative_rows": len(representatives),
            "summary": {
                "rows": len(representatives),
                "positive": sum(
                    any(row["label"] == "1" for row in group)
                    for group in by_locus.values()
                ),
                "negative": sum(
                    all(row["label"] == "0" for row in group)
                    for group in by_locus.values()
                ),
                "baseline_auprc": 1311 / 1989,
                "unique_locus": len(by_locus),
            },
            "aggregation": "label=max; score=mean|first|maximum, all reported",
        }
    )

    manifest = {
        "schema": "hevo_fair_variant_evaluation_datasets_v1",
        "assembly": "hg38",
        "sequence_length_bp": WINDOW,
        "seed": args.seed,
        "maf_matching": {
            "low_frequency_bin": "[0,0.01]",
            "bins": list(MAF_BIN_LABELS),
            "features": ["chrom", "maf_bin"],
            "model_scores_used": False,
        },
        "arms": arms,
    }
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
