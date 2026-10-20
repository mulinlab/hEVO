#!/usr/bin/env python3
"""Build strict chromosome/MAF-matched CausalDB control arms.

The selected controls come from the complete raw negative CausalDB export,
not from model outputs.  Coordinates are lifted to hg38, canonical
reference-to-alternate sequence pairs are constructed from the registered
hg38 reference, and controls are sampled
without replacement within pre-defined chromosome/MAF bins.  Positive loci,
exact variants, malformed records, and duplicate canonical variants are
excluded and counted in the manifest.

This script intentionally does not import any evaluator or model code.  Its
outputs are suitable for an independent sensitivity analysis; changing the
control source changes the estimand and must be reported explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pysam
from pyliftover import LiftOver


MAF_BINS = (
    (0.0, 0.01, "[0,0.01]"),
    (0.01, 0.05, "(0.01,0.05]"),
    (0.05, 0.10, "(0.05,0.10]"),
    (0.10, 0.20, "(0.10,0.20]"),
    (0.20, 0.30, "(0.20,0.30]"),
    (0.30, 0.40, "(0.30,0.40]"),
    (0.40, 0.50, "(0.40,0.50]"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_chrom(value: str) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def maf_bin(value: str) -> str:
    text = str(value).strip()
    if text in {"", "NULL", "None", "nan", "NaN"}:
        return "missing"
    maf = float(text)
    if not math.isfinite(maf) or maf < 0 or maf > 0.5:
        raise ValueError(f"invalid MAF: {value}")
    for index, (left, right, label) in enumerate(MAF_BINS):
        if index == 0 and left <= maf <= right:
            return label
        if index > 0 and left < maf <= right:
            return label
    raise ValueError(f"MAF did not fall into a bin: {value}")


def canonical_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        normalize_chrom(row["chrom_hg38"]),
        str(row["pos_hg38"]),
        row["canonical_ref"].upper(),
        row["canonical_alt"].upper(),
    )


def locus_key(row: dict[str, str]) -> tuple[str, str]:
    return normalize_chrom(row["chrom_hg38"]), str(row["pos_hg38"])


def choose_effect_pair(ea: str, nea: str, beta: float) -> tuple[str, str]:
    if beta < 0:
        return nea, ea
    return ea, nea


def lift(lo: LiftOver, chrom: str, pos: int) -> tuple[str, int] | None:
    source = normalize_chrom(chrom)
    converted = lo.convert_coordinate(source, pos - 1)
    if not converted:
        return None
    target, pos0, _, _ = converted[0]
    return target, int(pos0) + 1


def normalize_positive(row: dict[str, str]) -> dict[str, str]:
    required = {
        "seqID",
        "rsID",
        "refseq",
        "altseq",
        "label",
        "chrom_hg19",
        "pos_hg19",
        "chrom_hg38",
        "pos_hg38",
        "ea",
        "nea",
        "maf",
        "beta",
        "avgPP",
        "canonical_ref",
        "canonical_alt",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"positive canonical dataset is missing: {sorted(missing)}")
    if row["label"] != "1":
        raise ValueError("positive canonical dataset contains a non-positive row")
    result = dict(row)
    result["ref"] = row["canonical_ref"]
    result["alt"] = row["canonical_alt"]
    return result


def convert_negative(
    row: dict[str, str],
    lo: LiftOver,
    fasta: pysam.FastaFile,
    seq_len: int,
) -> tuple[dict[str, str] | None, str]:
    try:
        if row.get("label", "0") != "0":
            return None, "non_negative_label"
        ea = row["ea"].strip().upper()
        nea = row["nea"].strip().upper()
        if (
            len(ea) != 1
            or len(nea) != 1
            or ea not in "ACGT"
            or nea not in "ACGT"
            or ea == nea
        ):
            return None, "bad_allele"
        maf = row.get("maf", "").strip()
        maf_bin(maf)
        beta = float(row["beta"])
        lifted = lift(lo, row["chrom"], int(row["pos"]))
        if lifted is None:
            return None, "liftover_failed"
        chrom, pos = lifted
        if chrom not in fasta.references:
            return None, "chromosome_missing_from_fasta"
        half = seq_len // 2
        start = pos - 1 - half
        if start < 0:
            return None, "sequence_boundary"
        refseq = fasta.fetch(chrom, start, start + seq_len).upper()
        if len(refseq) != seq_len:
            return None, "sequence_length"
        genome_ref = refseq[half]
        if genome_ref not in {ea, nea}:
            return None, "reference_allele_mismatch"
        effect_ref, effect_alt = choose_effect_pair(ea, nea, beta)
        if effect_ref == effect_alt:
            return None, "identical_effect_alleles"
        canonical_alt = nea if genome_ref == ea else ea
        source_orientation = "unchanged" if effect_ref == genome_ref else "swapped"
        # All evaluators receive the same assembly-native pair: hg38 FASTA
        # reference at the center and the corresponding alternate allele.
        # The signed-effect orientation is retained only as provenance; it
        # must not change the sequence semantics used by any model.
        canonical_refseq = refseq
        canonical_altseq = refseq[:half] + canonical_alt + refseq[half + 1 :]
        result = {
            # Include the canonical allele identity because the raw export can
            # contain more than one record for the same rsID.
            "seqID": f"{row['rsID']}_0_{chrom}_{pos}_{genome_ref}{canonical_alt}",
            "rsID": row["rsID"],
            "refseq": canonical_refseq,
            "altseq": canonical_altseq,
            "label": "0",
            "chrom_hg19": row["chrom"],
            "pos_hg19": row["pos"],
            "chrom_hg38": chrom,
            "pos_hg38": str(pos),
            "ea": ea,
            "nea": nea,
            "maf": maf,
            "beta": row["beta"],
            "avgPP": row["avgPP"],
            "canonical_ref": genome_ref,
            "canonical_alt": canonical_alt,
            "source_orientation": source_orientation,
            "ref": genome_ref,
            "alt": canonical_alt,
        }
        if sum(a != b for a, b in zip(result["refseq"], result["altseq"])) != 1:
            return None, "constructed_pair_not_single_snv"
        return result, "PASS"
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "parse_or_numeric_error"


def summary(rows: list[dict[str, str]], arm: str) -> dict[str, object]:
    keys = [canonical_key(row) for row in rows]
    loci = [key[:2] for key in keys]
    positives = sum(row["label"] == "1" for row in rows)
    negatives = sum(row["label"] == "0" for row in rows)
    return {
        "arm_id": arm,
        "rows": len(rows),
        "positive": positives,
        "negative": negatives,
        "baseline_auprc": positives / len(rows) if rows else None,
        "unique_id": len({row["seqID"] for row in rows}),
        "unique_variant": len(set(keys)),
        "unique_locus": len(set(loci)),
        "duplicate_id_rows": len(rows) - len({row["seqID"] for row in rows}),
        "duplicate_variant_rows": len(rows) - len(set(keys)),
        "duplicate_locus_rows": len(rows) - len(set(loci)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-canonical", type=Path, required=True)
    parser.add_argument("--negative-raw", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--sequence-length", type=int, default=1023)
    args = parser.parse_args()
    if args.sequence_length % 2 != 1:
        raise ValueError("--sequence-length must be odd")
    for path in (args.positive_canonical, args.negative_raw, args.fasta):
        if not path.exists():
            raise FileNotFoundError(path)

    canonical_rows = read_csv(args.positive_canonical)
    positives = [
        normalize_positive(row) for row in canonical_rows if row.get("label") == "1"
    ]
    if len(positives) != 1117:
        raise ValueError(f"expected 1117 positive rows, got {len(positives)}")
    positive_keys = {canonical_key(row) for row in positives}
    positive_loci = {locus_key(row) for row in positives}
    if len(positive_keys) != len(positives):
        raise ValueError("positive canonical dataset contains duplicate variants")

    lo = LiftOver("hg19", "hg38")
    fasta = pysam.FastaFile(str(args.fasta))
    converted: list[dict[str, str]] = []
    exclusions: Counter[str] = Counter()
    raw_rows = read_tsv(args.negative_raw)
    for raw in raw_rows:
        record, status = convert_negative(raw, lo, fasta, args.sequence_length)
        if record is None:
            exclusions[status] += 1
        else:
            converted.append(record)
    fasta.close()

    # Keep a deterministic representative for duplicate canonical variants.
    by_variant: dict[tuple[str, str, str, str], dict[str, str]] = {}
    duplicate_variant_rows = 0
    for record in sorted(
        converted, key=lambda row: (canonical_key(row), row["rsID"], row["seqID"])
    ):
        key = canonical_key(record)
        if key in by_variant:
            duplicate_variant_rows += 1
            continue
        by_variant[key] = record
    deduplicated = list(by_variant.values())
    eligible = [
        row
        for row in deduplicated
        if canonical_key(row) not in positive_keys
        and locus_key(row) not in positive_loci
    ]
    exclusions["duplicate_canonical_variant"] = duplicate_variant_rows
    exclusions["positive_exact_variant_or_locus"] = len(deduplicated) - len(eligible)

    output_root = args.output_root.resolve()
    dataset_dir = output_root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_root / "negative_canonical_pool.csv"
    fields = (
        list(eligible[0])
        if eligible
        else [
            "seqID",
            "rsID",
            "refseq",
            "altseq",
            "label",
            "chrom_hg19",
            "pos_hg19",
            "chrom_hg38",
            "pos_hg38",
            "ea",
            "nea",
            "maf",
            "beta",
            "avgPP",
            "canonical_ref",
            "canonical_alt",
            "source_orientation",
            "ref",
            "alt",
        ]
    )
    with pool_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            sorted(eligible, key=lambda row: (canonical_key(row), row["seqID"]))
        )

    targets: Counter[tuple[str, str]] = Counter(
        (normalize_chrom(row["chrom_hg38"]), maf_bin(row["maf"])) for row in positives
    )
    pools: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        pools[(normalize_chrom(row["chrom_hg38"]), maf_bin(row["maf"]))].append(row)
    arms: list[dict[str, object]] = []
    for ratio, arm_id in ((1, "C1_strict_chr_maf_1to1"), (5, "C2_strict_chr_maf_1to5")):
        requested = {
            f"{chrom}:{bin_name}": ratio * count
            for (chrom, bin_name), count in sorted(targets.items())
        }
        deficits = {
            group: count - len(pools.get(tuple(group.split(":", 1)), []))
            for group, count in requested.items()
            if len(pools.get(tuple(group.split(":", 1)), [])) < count
        }
        matching = {
            "features": ["chrom_hg38", "maf_bin"],
            "ratio": f"1:{ratio}",
            "sampling_without_replacement": True,
            "same_locus_excluded": True,
            "exact_variant_excluded": True,
            "selection_model_scores_used": False,
            "requested_negative_count": sum(requested.values()),
            "eligible_negative_count": len(eligible),
            "deficits": deficits,
        }
        if deficits:
            arms.append(
                {
                    "arm_id": arm_id,
                    "status": "BLOCKED_MATCH_CAPACITY",
                    "matching": matching,
                    "pool": str(pool_path),
                    "pool_sha256": sha256(pool_path),
                }
            )
            continue
        rng = random.Random(args.seed + ratio)
        selected: list[dict[str, str]] = []
        for group in sorted(requested):
            chrom, bin_name = group.split(":", 1)
            pool = sorted(
                pools[(chrom, bin_name)],
                key=lambda row: (canonical_key(row), row["seqID"]),
            )
            selected.extend(rng.sample(pool, requested[group]))
        rows = [*positives, *selected]
        rng.shuffle(rows)
        output = dataset_dir / f"{arm_id}.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        matching["selected_negative_count"] = len(selected)
        matching["selected_by_chrom_maf"] = requested
        arms.append(
            {
                "arm_id": arm_id,
                "status": "PASS",
                "dataset": str(output),
                "dataset_sha256": sha256(output),
                "summary": summary(rows, arm_id),
                "matching": matching,
            }
        )

    # A second, explicitly distinct estimand avoids over-stratifying the
    # common-variant subset into sparse low-frequency cells.  It is not a
    # replacement for C1/C2: it matches chromosome and the pre-declared
    # common-MAF class (0.01 < MAF <= 0.50), and therefore must be reported as
    # a separate sensitivity arm.
    common_positives = [
        row
        for row in positives
        if row.get("maf", "").strip() not in {"", "NULL", "None", "nan", "NaN"}
        and float(row["maf"]) > 0.01
    ]
    common_controls = [
        row
        for row in eligible
        if row.get("maf", "").strip() not in {"", "NULL", "None", "nan", "NaN"}
        and float(row["maf"]) > 0.01
    ]
    common_pools: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in common_controls:
        common_pools[normalize_chrom(row["chrom_hg38"])].append(row)
    common_targets: Counter[str] = Counter(
        normalize_chrom(row["chrom_hg38"]) for row in common_positives
    )
    for ratio, arm_id in ((1, "C6_common_chr_maf_1to1"), (5, "C7_common_chr_maf_1to5")):
        requested = {
            chrom: ratio * count for chrom, count in sorted(common_targets.items())
        }
        deficits = {
            chrom: count - len(common_pools.get(chrom, []))
            for chrom, count in requested.items()
            if len(common_pools.get(chrom, [])) < count
        }
        matching = {
            "features": ["chrom_hg38", "common_maf_class"],
            "common_maf_definition": "0.01 < MAF <= 0.50; non-missing",
            "ratio": f"1:{ratio}",
            "sampling_without_replacement": True,
            "same_locus_excluded": True,
            "exact_variant_excluded": True,
            "selection_model_scores_used": False,
            "requested_positive_count": len(common_positives),
            "requested_negative_count": sum(requested.values()),
            "eligible_negative_count": len(common_controls),
            "deficits": deficits,
        }
        if deficits:
            arms.append(
                {
                    "arm_id": arm_id,
                    "status": "BLOCKED_MATCH_CAPACITY",
                    "matching": matching,
                    "pool": str(pool_path),
                    "pool_sha256": sha256(pool_path),
                }
            )
            continue
        rng = random.Random(args.seed + 1000 + ratio)
        selected: list[dict[str, str]] = []
        for chrom in sorted(requested):
            pool = sorted(
                common_pools[chrom], key=lambda row: (canonical_key(row), row["seqID"])
            )
            selected.extend(rng.sample(pool, requested[chrom]))
        rows = [*common_positives, *selected]
        rng.shuffle(rows)
        output = dataset_dir / f"{arm_id}.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        matching["selected_negative_count"] = len(selected)
        matching["selected_by_chrom"] = requested
        arms.append(
            {
                "arm_id": arm_id,
                "status": "PASS",
                "dataset": str(output),
                "dataset_sha256": sha256(output),
                "summary": summary(rows, arm_id),
                "matching": matching,
            }
        )

    # High-confidence causal labels are a separate scientific sensitivity
    # definition, not a post-hoc score filter: posterior support is at least
    # 0.99 and the reported absolute effect size is at least 0.5.  Controls
    # remain drawn from the complete negative pool and are matched only by
    # chromosome/common-MAF class.
    high_conf_positives = [
        row
        for row in common_positives
        if float(row["avgPP"]) >= 0.99 and abs(float(row["beta"])) >= 0.5
    ]
    high_conf_targets: Counter[str] = Counter(
        normalize_chrom(row["chrom_hg38"]) for row in high_conf_positives
    )
    for ratio, arm_id in (
        (1, "C8_highconf_common_chr_1to1"),
        (5, "C9_highconf_common_chr_1to5"),
    ):
        requested = {
            chrom: ratio * count for chrom, count in sorted(high_conf_targets.items())
        }
        deficits = {
            chrom: count - len(common_pools.get(chrom, []))
            for chrom, count in requested.items()
            if len(common_pools.get(chrom, [])) < count
        }
        matching = {
            "features": ["chrom_hg38", "common_maf_class"],
            "positive_definition": "avgPP >= 0.99 and abs(beta) >= 0.5; 0.01 < MAF <= 0.50; non-missing",
            "ratio": f"1:{ratio}",
            "sampling_without_replacement": True,
            "same_locus_excluded": True,
            "exact_variant_excluded": True,
            "selection_model_scores_used": False,
            "requested_positive_count": len(high_conf_positives),
            "requested_negative_count": sum(requested.values()),
            "eligible_negative_count": len(common_controls),
            "deficits": deficits,
        }
        if deficits:
            arms.append(
                {
                    "arm_id": arm_id,
                    "status": "BLOCKED_MATCH_CAPACITY",
                    "matching": matching,
                    "pool": str(pool_path),
                    "pool_sha256": sha256(pool_path),
                }
            )
            continue
        rng = random.Random(args.seed + 2000 + ratio)
        selected: list[dict[str, str]] = []
        for chrom in sorted(requested):
            pool = sorted(
                common_pools[chrom], key=lambda row: (canonical_key(row), row["seqID"])
            )
            selected.extend(rng.sample(pool, requested[chrom]))
        rows = [*high_conf_positives, *selected]
        rng.shuffle(rows)
        output = dataset_dir / f"{arm_id}.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        matching["selected_negative_count"] = len(selected)
        matching["selected_by_chrom"] = requested
        arms.append(
            {
                "arm_id": arm_id,
                "status": "PASS",
                "dataset": str(output),
                "dataset_sha256": sha256(output),
                "summary": summary(rows, arm_id),
                "matching": matching,
            }
        )

    manifest = {
        "schema": "hevo_strict_causal_controls_v1",
        "status": "PASS"
        if all(arm["status"] == "PASS" for arm in arms)
        else "BLOCKED_MATCH_CAPACITY",
        "assembly": "hg38",
        "sequence_length_bp": args.sequence_length,
        "seed": args.seed,
        "source": {
            "positive_canonical": str(args.positive_canonical.resolve()),
            "positive_canonical_sha256": sha256(args.positive_canonical),
            "negative_raw": str(args.negative_raw.resolve()),
            "negative_raw_sha256": sha256(args.negative_raw),
            "fasta": str(args.fasta.resolve()),
            "fasta_sha256": sha256(args.fasta),
            "liftover": "pyliftover hg19->hg38 first returned mapping",
        },
        "model_scores_used_for_selection": False,
        "canonical_input_rows": len(canonical_rows),
        "canonical_input_positive_rows": len(positives),
        "raw_negative_rows": len(raw_rows),
        "converted_negative_rows": len(converted),
        "deduplicated_negative_rows": len(deduplicated),
        "eligible_negative_rows": len(eligible),
        "exclusions": dict(sorted(exclusions.items())),
        "arms": arms,
        "negative_canonical_pool": str(pool_path),
        "negative_canonical_pool_sha256": sha256(pool_path),
    }
    manifest_path = output_root / "strict_controls_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
