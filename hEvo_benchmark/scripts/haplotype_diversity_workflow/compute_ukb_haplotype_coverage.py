#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

from workflow_config import hevo_data_root, project_root

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_window_haplotype_diversity import (  # noqa: E402
    AUTOSOMES,
    GRCH38_AUTOSOME_LENGTHS,
    choose_contig,
    chrom_length,
    find_vcf_for_chrom,
    open_cohort_reader,
    should_keep_window,
    stable_uint64,
    window_haplotype_hashes,
)


PROJECT = project_root()
UKB_FULL_DIR = hevo_data_root() / "Phasing_VCFs/UKB_full"
UKB_SUBSET_DIR = hevo_data_root() / "Phasing_VCFs/UKBB"
FINAL_MERGED_DIR = hevo_data_root() / "Phasing_VCFs/final_merged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how many UKB-full phased SNV haplotypes are observed in "
            "the UKB subset and in the final merged cohort."
        )
    )
    parser.add_argument("--chrom", required=True, choices=AUTOSOMES)
    parser.add_argument("--window-size", type=int, default=100_000)
    parser.add_argument("--window-subsample-rate", type=float, default=0.02)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-samples-ukb-full", type=int, default=2000)
    parser.add_argument("--max-samples-target", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--ukb-full-dir", type=Path, default=UKB_FULL_DIR)
    parser.add_argument("--ukb-subset-dir", type=Path, default=UKB_SUBSET_DIR)
    parser.add_argument("--final-merged-dir", type=Path, default=FINAL_MERGED_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results/ukb_haplotype_coverage_vcf",
    )
    return parser.parse_args()


def sample_seed(base_seed: int, label: str) -> int:
    return int(base_seed + stable_uint64(label).item() % 1_000_000)


def coverage_rows_for_window(
    chrom: str,
    start0: int,
    end0: int,
    ukb_full_hashes: np.ndarray,
    ukb_subset_hashes: np.ndarray,
    final_hashes: np.ndarray,
) -> list[dict[str, object]]:
    ref_unique, ref_counts = np.unique(ukb_full_hashes, return_counts=True)
    subset_present = set(np.unique(ukb_subset_hashes).tolist())
    # The final cohort contains the UKB subset. With sample-capped target scans,
    # keep the sampled UKB subset contribution explicit instead of letting a
    # random sample from final_merged dilute it away.
    final_present = subset_present | set(np.unique(final_hashes).tolist())
    n_ref_haplotypes = int(len(ukb_full_hashes))

    rows = []
    for hap_hash, count in zip(ref_unique, ref_counts):
        hap_int = int(hap_hash)
        rows.append(
            {
                "chrom": chrom,
                "window_start": start0 + 1,
                "window_end": end0,
                "ukb_full_haplotype_hash": f"{hap_int:016x}",
                "ukb_full_haplotype_count": int(count),
                "ukb_full_haplotype_frequency": float(count) / n_ref_haplotypes,
                "covered_by_ukb_subset": int(hap_int in subset_present),
                "covered_by_final_cohort": int(hap_int in final_present),
                "n_ukb_full_haplotypes_in_window": n_ref_haplotypes,
                "n_ukb_subset_haplotypes_in_window": int(len(ukb_subset_hashes)),
                "n_final_haplotypes_in_window": int(len(final_hashes)),
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
    vcf_paths = {
        "ukb_full": find_vcf_for_chrom(args.ukb_full_dir, args.chrom),
        "ukb_subset": find_vcf_for_chrom(args.ukb_subset_dir, args.chrom),
        "final_cohort": find_vcf_for_chrom(args.final_merged_dir, args.chrom),
    }

    import pysam

    with pysam.VariantFile(str(vcf_paths["ukb_full"])) as vcf:
        contig = choose_contig(vcf, args.chrom)
        length = chrom_length(vcf, contig, args.chrom)
    if length is None:
        length = GRCH38_AUTOSOME_LENGTHS[args.chrom]

    readers = {
        "ukb_full": open_cohort_reader(
            name="ukb_full",
            vcf_path=vcf_paths["ukb_full"],
            start0=0,
            chrom=args.chrom,
            max_samples=args.max_samples_ukb_full,
            seed=sample_seed(args.seed, "ukb_full"),
        ),
        "ukb_subset": open_cohort_reader(
            name="ukb_subset",
            vcf_path=vcf_paths["ukb_subset"],
            start0=0,
            chrom=args.chrom,
            max_samples=args.max_samples_target,
            seed=sample_seed(args.seed, "ukb_subset"),
        ),
        "final_cohort": open_cohort_reader(
            name="final_cohort",
            vcf_path=vcf_paths["final_cohort"],
            start0=0,
            chrom=args.chrom,
            max_samples=args.max_samples_target,
            seed=sample_seed(args.seed, "final_cohort"),
        ),
    }

    suffix = (
        ""
        if args.n_shards == 1
        else f".shard{args.shard_index:03d}-of-{args.n_shards:03d}"
    )
    out_path = args.out_dir / f"ukb_haplotype_coverage.chr{args.chrom}{suffix}.tsv.gz"
    fieldnames = [
        "chrom",
        "window_start",
        "window_end",
        "ukb_full_haplotype_hash",
        "ukb_full_haplotype_count",
        "ukb_full_haplotype_frequency",
        "covered_by_ukb_subset",
        "covered_by_final_cohort",
        "n_ukb_full_haplotypes_in_window",
        "n_ukb_subset_haplotypes_in_window",
        "n_final_haplotypes_in_window",
        "n_ukb_full_snv_records",
        "n_ukb_subset_snv_records",
        "n_final_snv_records",
    ]

    windows_written = 0
    windows_sampled = 0
    try:
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

                ukb_full_hashes, ukb_full_snv, _, _ = window_haplotype_hashes(
                    readers["ukb_full"], args.chrom, start0, end0
                )
                ukb_subset_hashes, ukb_subset_snv, _, _ = window_haplotype_hashes(
                    readers["ukb_subset"], args.chrom, start0, end0
                )
                final_hashes, final_snv, _, _ = window_haplotype_hashes(
                    readers["final_cohort"], args.chrom, start0, end0
                )
                if ukb_full_snv == 0:
                    continue

                rows = coverage_rows_for_window(
                    chrom=args.chrom,
                    start0=start0,
                    end0=end0,
                    ukb_full_hashes=ukb_full_hashes,
                    ukb_subset_hashes=ukb_subset_hashes,
                    final_hashes=final_hashes,
                )
                for row in rows:
                    row["n_ukb_full_snv_records"] = ukb_full_snv
                    row["n_ukb_subset_snv_records"] = ukb_subset_snv
                    row["n_final_snv_records"] = final_snv
                    writer.writerow(row)
                windows_written += 1
                if windows_written % 20 == 0:
                    print(
                        f"chr{args.chrom}: wrote {windows_written} windows", flush=True
                    )
    finally:
        for reader in readers.values():
            reader.close()

    print(
        f"Wrote {out_path} from {windows_written} UKB-full variant-bearing windows; "
        f"sampled {windows_sampled} windows",
        flush=True,
    )


if __name__ == "__main__":
    main()
