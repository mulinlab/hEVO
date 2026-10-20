#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from workflow_config import hevo_data_root, project_root
import pysam

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_window_haplotype_diversity import (  # noqa: E402
    AUTOSOMES,
    choose_contig,
    chrom_length,
    find_vcf_for_chrom,
    is_snv_record,
    mix_uint64,
    should_keep_window,
    stable_uint64,
)


PROJECT = project_root()
UKB_FULL_DIR = hevo_data_root() / "Phasing_VCFs/UKB_full"
UKB_SUBSET_DIR = hevo_data_root() / "Phasing_VCFs/UKBB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Empirically compare selected UKB subset with random same-size UKB "
            "downsamples using sample IDs and SNV haplotypes from the same UKB_full VCF."
        )
    )
    parser.add_argument("--chrom", required=True, choices=AUTOSOMES)
    parser.add_argument("--window-size", type=int, default=100_000)
    parser.add_argument("--window-subsample-rate", type=float, default=0.02)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--reference-sample-size", type=int, default=5000)
    parser.add_argument(
        "--target-sample-size",
        type=int,
        default=None,
        help="Default uses all selected UKB subset IDs present on this chromosome.",
    )
    parser.add_argument("--random-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--ukb-full-dir", type=Path, default=UKB_FULL_DIR)
    parser.add_argument("--ukb-subset-dir", type=Path, default=UKB_SUBSET_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results/ukb_downsampling_vcf_empirical",
    )
    return parser.parse_args()


def selected_ids_from_subset_vcf(vcf_path: Path) -> list[str]:
    with pysam.VariantFile(str(vcf_path)) as vcf:
        return list(vcf.header.samples)


def sample_ids(ids: list[str], n: int | None, rng: np.random.Generator) -> list[str]:
    if n is None or len(ids) <= n:
        return ids
    idx = np.sort(rng.choice(len(ids), size=n, replace=False))
    return [ids[i] for i in idx]


def make_sample_sets(
    full_ids: list[str],
    selected_ids: list[str],
    reference_n: int,
    target_n: int | None,
    random_repeats: int,
    rng: np.random.Generator,
) -> tuple[dict[str, list[str]], list[str]]:
    full_set = set(full_ids)
    selected = [sample for sample in selected_ids if sample in full_set]
    selected = sample_ids(selected, target_n, rng)
    random_n = len(selected) if target_n is None else min(target_n, len(full_ids))

    sets: dict[str, list[str]] = {
        "reference": sample_ids(full_ids, min(reference_n, len(full_ids)), rng),
        "selected": selected,
    }
    for repeat in range(random_repeats):
        sets[f"random_{repeat + 1}"] = sample_ids(full_ids, random_n, rng)

    union = sorted(set().union(*sets.values()))
    return sets, union


def haplotype_hashes_for_union(
    vcf: pysam.VariantFile,
    contig: str,
    chrom: str,
    start0: int,
    end0: int,
    union_samples: list[str],
) -> tuple[np.ndarray, int]:
    hashes = np.zeros(len(union_samples) * 2, dtype=np.uint64)
    n_snv = 0
    for record in vcf.fetch(contig, start0, end0):
        if not is_snv_record(record):
            continue
        n_snv += 1
        alt_tokens = [
            mix_uint64(
                stable_uint64(f"{chrom}:{record.pos}:{record.ref}>{alt_idx}:{alt}")
            )
            for alt_idx, alt in enumerate(record.alts, start=1)
        ]
        samples = record.samples
        for sample_i, sample in enumerate(union_samples):
            gt = samples[sample].get("GT")
            if gt is None or len(gt) < 2 or gt[0] is None or gt[1] is None:
                continue
            for hap_offset, allele in enumerate((gt[0], gt[1])):
                if allele and allele > 0 and allele <= len(alt_tokens):
                    hashes[sample_i * 2 + hap_offset] ^= alt_tokens[allele - 1]
    return hashes, n_snv


def take_hashes(
    hashes: np.ndarray, union_index: dict[str, int], sample_ids: list[str]
) -> np.ndarray:
    idx = np.empty(len(sample_ids) * 2, dtype=np.int64)
    for i, sample in enumerate(sample_ids):
        base = union_index[sample] * 2
        idx[i * 2] = base
        idx[i * 2 + 1] = base + 1
    return hashes[idx]


def rows_for_window(
    chrom: str,
    start0: int,
    end0: int,
    hashes: np.ndarray,
    sample_sets: dict[str, list[str]],
    union_index: dict[str, int],
    n_snv: int,
) -> list[dict[str, object]]:
    ref_hashes = take_hashes(hashes, union_index, sample_sets["reference"])
    ref_unique, ref_counts = np.unique(ref_hashes, return_counts=True)
    selected_present = set(
        np.unique(take_hashes(hashes, union_index, sample_sets["selected"])).tolist()
    )
    random_sets = [
        set(np.unique(take_hashes(hashes, union_index, ids)).tolist())
        for key, ids in sample_sets.items()
        if key.startswith("random_")
    ]
    random_sample_sizes = [
        len(ids) for key, ids in sample_sets.items() if key.startswith("random_")
    ]

    rows = []
    n_ref_haplotypes = len(ref_hashes)
    for hap_hash, count in zip(ref_unique, ref_counts):
        hap_int = int(hap_hash)
        random_cover = sum(1 for present in random_sets if hap_int in present)
        rows.append(
            {
                "chrom": chrom,
                "window_start": start0 + 1,
                "window_end": end0,
                "reference_haplotype_hash": f"{hap_int:016x}",
                "reference_haplotype_count": int(count),
                "reference_haplotype_frequency": float(count) / n_ref_haplotypes,
                "covered_by_selected_ukb_subset": int(hap_int in selected_present),
                "random_cover_count": int(random_cover),
                "random_repeats": len(random_sets),
                "n_reference_samples": len(sample_sets["reference"]),
                "n_selected_samples": len(sample_sets["selected"]),
                "n_random_samples_per_repeat": int(np.median(random_sample_sizes))
                if random_sample_sizes
                else 0,
                "n_snv_records": n_snv,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.n_shards < 1:
        raise ValueError("--n-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.n_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < n-shards")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_vcf_path = find_vcf_for_chrom(args.ukb_full_dir, args.chrom)
    subset_vcf_path = find_vcf_for_chrom(args.ukb_subset_dir, args.chrom)
    selected_ids = selected_ids_from_subset_vcf(subset_vcf_path)

    seed = int(args.seed + int(args.chrom) * 10_000 + args.shard_index)
    rng = np.random.default_rng(seed)

    with pysam.VariantFile(str(full_vcf_path)) as header_vcf:
        full_ids = list(header_vcf.header.samples)
        contig = choose_contig(header_vcf, args.chrom)
        length = chrom_length(header_vcf, contig, args.chrom)

    sample_sets, union_samples = make_sample_sets(
        full_ids=full_ids,
        selected_ids=selected_ids,
        reference_n=args.reference_sample_size,
        target_n=args.target_sample_size,
        random_repeats=args.random_repeats,
        rng=rng,
    )
    union_index = {sample: i for i, sample in enumerate(union_samples)}

    vcf = pysam.VariantFile(str(full_vcf_path))
    try:
        vcf.subset_samples(union_samples)
        contig = choose_contig(vcf, args.chrom)
        suffix = (
            ""
            if args.n_shards == 1
            else f".shard{args.shard_index:03d}-of-{args.n_shards:03d}"
        )
        out_path = (
            args.out_dir / f"ukb_downsampling_empirical.chr{args.chrom}{suffix}.tsv.gz"
        )
        # Save exact sample sets per task so coverage results can be audited.
        manifest_path = out_path.with_suffix("").with_suffix(".samples.json")
        manifest_path.write_text(json.dumps({
            "full_vcf": str(full_vcf_path), "subset_vcf": str(subset_vcf_path),
            "seed": seed, "sample_sets": sample_sets,
        }, indent=2) + "\n")
        fieldnames = [
            "chrom",
            "window_start",
            "window_end",
            "reference_haplotype_hash",
            "reference_haplotype_count",
            "reference_haplotype_frequency",
            "covered_by_selected_ukb_subset",
            "random_cover_count",
            "random_repeats",
            "n_reference_samples",
            "n_selected_samples",
            "n_random_samples_per_repeat",
            "n_snv_records",
        ]
        windows_written = 0
        windows_sampled = 0
        with gzip.open(out_path, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
            for window_index, start0 in enumerate(range(0, length, args.window_size)):
                if window_index % args.n_shards != args.shard_index:
                    continue
                if not should_keep_window(
                    args.chrom, start0, args.window_subsample_rate, args.seed
                ):
                    continue
                windows_sampled += 1
                if args.max_windows is not None and windows_written >= args.max_windows:
                    break
                end0 = min(start0 + args.window_size, length)
                hashes, n_snv = haplotype_hashes_for_union(
                    vcf, contig, args.chrom, start0, end0, union_samples
                )
                if n_snv == 0:
                    continue
                for row in rows_for_window(
                    args.chrom, start0, end0, hashes, sample_sets, union_index, n_snv
                ):
                    writer.writerow(row)
                windows_written += 1
                if windows_written % 10 == 0:
                    print(
                        f"chr{args.chrom}: wrote {windows_written} windows", flush=True
                    )
    finally:
        vcf.close()

    print(
        f"Wrote {out_path} from {windows_written} variant-bearing windows; sampled {windows_sampled} windows; "
        f"union_samples={len(union_samples)} selected={len(sample_sets['selected'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
