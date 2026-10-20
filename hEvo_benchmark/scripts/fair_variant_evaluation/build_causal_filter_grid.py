#!/usr/bin/env python3
"""Build a complete, model-blind CausalDB PP/beta/MAF sensitivity grid.

Every requested threshold combination is represented in the manifest.  A
combination is either written as a deterministic 1:1 chromosome+MAF matched
dataset or recorded as BLOCKED_MATCH_CAPACITY; it is never silently relaxed.
This is a sensitivity generator, not a licence to replace the full C0 arm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def chrom(row: dict[str, str]) -> str:
    value = (row.get("chrom_hg38") or row.get("chrom") or "").strip()
    return value[3:] if value.lower().startswith("chr") else value


def pos(row: dict[str, str]) -> str:
    return (row.get("pos_hg38") or row.get("pos") or "").strip()


def allele_pair(row: dict[str, str]) -> tuple[str, str]:
    ref = (row.get("canonical_ref") or row.get("ref") or "").strip().upper()
    alt = (row.get("canonical_alt") or row.get("alt") or "").strip().upper()
    if not ref or not alt:
        refseq = (row.get("refseq") or "").strip().upper()
        altseq = (row.get("altseq") or "").strip().upper()
        if len(refseq) != len(altseq) or not refseq or len(refseq) % 2 == 0:
            raise ValueError("missing valid allele fields")
        center = len(refseq) // 2
        ref, alt = refseq[center], altseq[center]
    if ref == alt:
        raise ValueError("REF and ALT are identical")
    return ref, alt


def variant_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    ref, alt = allele_pair(row)
    return chrom(row), pos(row), ref, alt


def locus_key(row: dict[str, str]) -> tuple[str, str]:
    return chrom(row), pos(row)


def maf_bin(row: dict[str, str]) -> str:
    text = row.get("maf", "").strip()
    if not text:
        return "missing"
    value = float(text)
    if value < 0 or value > 0.5:
        raise ValueError(f"MAF outside [0,0.5]: {value}")
    if value < 0.01:
        return "[0,0.01)"
    if value < 0.05:
        return "[0.01,0.05)"
    if value < 0.10:
        return "[0.05,0.10)"
    if value < 0.20:
        return "[0.10,0.20)"
    if value < 0.30:
        return "[0.20,0.30)"
    if value < 0.40:
        return "[0.30,0.40)"
    return "[0.40,0.50]"


def maf_allowed(row: dict[str, str], policy: str) -> bool:
    text = row.get("maf", "").strip()
    if policy == "all_maf":
        return True
    if not text:
        return False
    value = float(text)
    return policy == "observed_maf" or value >= 0.01


def validate(rows: list[dict[str, str]]) -> None:
    required = {"label", "avgPP", "beta", "maf"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for row in rows:
        if row["label"] not in {"0", "1"}:
            raise ValueError("labels must be 0/1")
        variant_key(row)


def build_arm(
    positives: list[dict[str, str]],
    negatives: list[dict[str, str]],
    pp: float,
    beta: float,
    policy: str,
    match_features: str,
    seed: int,
) -> tuple[list[dict[str, str]] | None, dict[str, object]]:
    selected_pos = [
        row
        for row in positives
        if float(row["avgPP"]) >= pp
        and abs(float(row["beta"])) >= beta
        and maf_allowed(row, policy)
    ]
    eligible_neg = [row for row in negatives if maf_allowed(row, policy)]
    pos_keys = {variant_key(row) for row in selected_pos}
    pos_loci = {locus_key(row) for row in selected_pos}
    eligible_neg = [
        row
        for row in eligible_neg
        if variant_key(row) not in pos_keys and locus_key(row) not in pos_loci
    ]
    groups: defaultdict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)

    def match_key(row: dict[str, str]) -> tuple[str, ...]:
        if match_features == "chrom_maf":
            return chrom(row), maf_bin(row)
        return (maf_bin(row),)

    for row in eligible_neg:
        groups[match_key(row)].append(row)
    target = Counter(match_key(row) for row in selected_pos)
    deficits = {
        ":".join(group): n - len(groups.get(group, []))
        for group, n in sorted(target.items())
        if len(groups.get(group, [])) < n
    }
    metadata: dict[str, object] = {
        "status": "PASS" if not deficits else "BLOCKED_MATCH_CAPACITY",
        "positive_source_rows": len(positives),
        "positive_selected_rows": len(selected_pos),
        "positive_excluded_from_full": len(positives) - len(selected_pos),
        "negative_pool_rows": len(eligible_neg),
        "target_negative_rows": len(selected_pos),
        "maf_policy": policy,
        "matching": f"{match_features}; exact variant and locus excluded",
        "seed": seed,
        "selection_model_scores_used": False,
    }
    if deficits:
        metadata["deficits"] = deficits
        return None, metadata
    rng = random.Random(seed)
    chosen: list[dict[str, str]] = []
    for group, count in sorted(target.items()):
        pool = sorted(
            groups[group], key=lambda row: (variant_key(row), row.get("seqID", ""))
        )
        chosen.extend(rng.sample(pool, count))
    if len({variant_key(row) for row in chosen}) != len(chosen):
        raise AssertionError("duplicate matched negative variant")
    output = selected_pos + chosen
    rng.shuffle(output)
    metadata["negative_selected_rows"] = len(chosen)
    metadata["positive"] = len(selected_pos)
    metadata["negative"] = len(chosen)
    metadata["rows"] = len(output)
    metadata["baseline_auprc"] = len(selected_pos) / len(output) if output else None
    return output, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="canonical full CausalDB CSV"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--pp-thresholds", default="0.8,0.9,0.95,0.99")
    parser.add_argument("--beta-thresholds", default="0.5,1.0,2.0")
    parser.add_argument("--maf-policies", default="all_maf,observed_maf,common_maf")
    parser.add_argument(
        "--match-features",
        choices=("maf_bin_only", "chrom_maf"),
        default="maf_bin_only",
    )
    args = parser.parse_args()
    rows = read(args.input)
    validate(rows)
    positives = [row for row in rows if row["label"] == "1"]
    negatives = [row for row in rows if row["label"] == "0"]
    if not positives or not negatives:
        raise ValueError("input must contain both classes")
    pp_values = [float(value) for value in args.pp_thresholds.split(",")]
    beta_values = [float(value) for value in args.beta_thresholds.split(",")]
    policies = [
        value.strip() for value in args.maf_policies.split(",") if value.strip()
    ]
    if any(value < 0 or value > 1 for value in pp_values) or any(
        value < 0 for value in beta_values
    ):
        raise ValueError("invalid threshold")
    if set(policies) - {"all_maf", "observed_maf", "common_maf"}:
        raise ValueError("unknown MAF policy")
    root = args.output_root.resolve()
    dataset_dir = root / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_arms: list[dict[str, object]] = []
    fields = list(rows[0])
    for pp in pp_values:
        for beta in beta_values:
            for policy in policies:
                arm_id = f"E_pp{pp:g}_beta{beta:g}_{policy}"
                seed = args.seed + len(manifest_arms)
                output, metadata = build_arm(
                    positives, negatives, pp, beta, policy, args.match_features, seed
                )
                item: dict[str, object] = {
                    "arm_id": arm_id,
                    "task": "causal",
                    "pp_threshold": pp,
                    "abs_beta_threshold": beta,
                    "maf_policy": policy,
                    "source": str(args.input.resolve()),
                    "source_sha256": sha256(args.input),
                    **metadata,
                }
                if output is not None:
                    path = dataset_dir / f"{arm_id}.csv"
                    with path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=fields, lineterminator="\n"
                        )
                        writer.writeheader()
                        writer.writerows(output)
                    item["dataset"] = str(path)
                    item["dataset_sha256"] = sha256(path)
                    item["unique_variant"] = len({variant_key(row) for row in output})
                    item["unique_locus"] = len({locus_key(row) for row in output})
                manifest_arms.append(item)
    manifest = {
        "schema": "hevo_causaldb_model_blind_filter_grid_v1",
        "status": "PASS",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "seed": args.seed,
        "grid": {
            "pp_thresholds": pp_values,
            "abs_beta_thresholds": beta_values,
            "maf_policies": policies,
            "match_features": args.match_features,
        },
        "positive_source_rows": len(positives),
        "negative_source_rows": len(negatives),
        "model_scores_used_for_selection": False,
        "arms": manifest_arms,
    }
    manifest_path = root / "filter_grid_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "arms": len(manifest_arms),
                "output": str(manifest_path),
                "output_sha256": sha256(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
