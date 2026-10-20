#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam

from workflow_config import results_root


AUTOSOMES = [str(i) for i in range(1, 23)]
GRCH38_AUTOSOME_LENGTHS = {
    "1": 248_956_422,
    "2": 242_193_529,
    "3": 198_295_559,
    "4": 190_214_555,
    "5": 181_538_259,
    "6": 170_805_979,
    "7": 159_345_973,
    "8": 145_138_636,
    "9": 138_394_717,
    "10": 133_797_422,
    "11": 135_086_622,
    "12": 133_275_309,
    "13": 114_364_328,
    "14": 107_043_718,
    "15": 101_991_189,
    "16": 90_338_345,
    "17": 83_257_441,
    "18": 80_373_285,
    "19": 58_617_616,
    "20": 64_444_167,
    "21": 46_709_983,
    "22": 50_818_468,
}

SCENARIOS: list[tuple[str, list[str], str]] = [
    ("UKB_reference", ["UKB_full"], "UKB full reference"),
    ("S1_UKB", ["UKB"], "UKB subset"),
    ("S2_UK10K", ["UKB", "ALSPAC740", "TWINSUK741"], "+ UK10K"),
    ("S3_EAS2K", ["UKB", "ALSPAC740", "TWINSUK741", "EAS2K"], "+ EAS2K"),
    ("S4_SG10K", ["UKB", "ALSPAC740", "TWINSUK741", "EAS2K", "SG10K"], "+ SG10K"),
    (
        "S5_AFR1K",
        ["UKB", "ALSPAC740", "TWINSUK741", "EAS2K", "SG10K", "AFR4K"],
        "+ AFR1K",
    ),
    (
        "S6_TOPMed",
        ["UKB", "ALSPAC740", "TWINSUK741", "EAS2K", "SG10K", "AFR4K", "TOPMed"],
        "+ TOPMed",
    ),
    (
        "S7_HGDP",
        ["UKB", "ALSPAC740", "TWINSUK741", "EAS2K", "SG10K", "AFR4K", "TOPMed", "HGDP"],
        "+ HGDP",
    ),
    (
        "S8_1KGP",
        [
            "UKB",
            "ALSPAC740",
            "TWINSUK741",
            "EAS2K",
            "SG10K",
            "AFR4K",
            "TOPMed",
            "HGDP",
            "1KGP",
        ],
        "+ 1KGP",
    ),
]

BALANCED_UNITS = {
    "UKB_reference": [["UKB_full"]],
    "S1_UKB": [["UKB"]],
    "S2_UK10K": [["UKB"], ["ALSPAC740", "TWINSUK741"]],
    "S3_EAS2K": [["UKB"], ["ALSPAC740", "TWINSUK741"], ["EAS2K"]],
    "S4_SG10K": [["UKB"], ["ALSPAC740", "TWINSUK741"], ["EAS2K"], ["SG10K"]],
    "S5_AFR1K": [["UKB"], ["ALSPAC740", "TWINSUK741"], ["EAS2K"], ["SG10K"], ["AFR4K"]],
    "S6_TOPMed": [
        ["UKB"],
        ["ALSPAC740", "TWINSUK741"],
        ["EAS2K"],
        ["SG10K"],
        ["AFR4K"],
        ["TOPMed"],
    ],
    "S7_HGDP": [
        ["UKB"],
        ["ALSPAC740", "TWINSUK741"],
        ["EAS2K"],
        ["SG10K"],
        ["AFR4K"],
        ["TOPMed"],
        ["HGDP"],
    ],
    "S8_1KGP": [
        ["UKB"],
        ["ALSPAC740", "TWINSUK741"],
        ["EAS2K"],
        ["SG10K"],
        ["AFR4K"],
        ["TOPMed"],
        ["HGDP"],
        ["1KGP"],
    ],
}


@dataclass(frozen=True)
class Cohort:
    cohort: str
    display_label: str
    group: str
    vcf_dir: Path


@dataclass
class CohortReader:
    name: str
    vcf_path: Path
    vcf: pysam.VariantFile
    contig: str
    samples: list[str]

    def close(self) -> None:
        self.vcf.close()


def stable_uint64(text: str) -> np.uint64:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little", signed=False))


def mix_uint64(value: np.uint64) -> np.uint64:
    mask = (1 << 64) - 1
    x = int(value)
    x = (x ^ (x >> 30)) & mask
    x = (x * 0xBF58476D1CE4E5B9) & mask
    x = (x ^ (x >> 27)) & mask
    x = (x * 0x94D049BB133111EB) & mask
    x = (x ^ (x >> 31)) & mask
    return np.uint64(x)


def read_manifest(path: Path) -> dict[str, Cohort]:
    rows: dict[str, Cohort] = {}
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["cohort"]] = Cohort(
                cohort=row["cohort"],
                display_label=row["display_label"],
                group=row["group"],
                vcf_dir=Path(row["vcf_dir"]),
            )
    return rows


def find_vcf_for_chrom(vcf_dir: Path, chrom: str) -> Path:
    pattern = re.compile(rf"(?:^|[._-])chr{re.escape(chrom)}(?:[._-])")
    matches = sorted(
        p
        for p in vcf_dir.glob("*.vcf.gz")
        if pattern.search(p.name) and not p.name.endswith((".tbi", ".csi"))
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one chr{chrom} VCF in {vcf_dir}, found {len(matches)}: {matches[:5]}"
        )
    return matches[0]


def choose_contig(vcf: pysam.VariantFile, chrom: str) -> str:
    for name in (f"chr{chrom}", chrom):
        if name in vcf.header.contigs:
            return name
    if len(vcf.header.contigs) == 1:
        return next(iter(vcf.header.contigs))
    raise ValueError(
        f"Cannot find contig for chr{chrom}; header has examples {list(vcf.header.contigs)[:5]}"
    )


def chrom_length(vcf: pysam.VariantFile, contig: str, chrom: str) -> int:
    length = vcf.header.contigs[contig].length
    if length is None and chrom in GRCH38_AUTOSOME_LENGTHS:
        return GRCH38_AUTOSOME_LENGTHS[chrom]
    if length is None:
        raise ValueError(
            f"Contig {contig} has no length in VCF header and no fallback is available"
        )
    return int(length)


def is_snv_record(record: pysam.VariantRecord) -> bool:
    if (
        record.ref is None
        or len(record.ref) != 1
        or record.ref.upper() not in {"A", "C", "G", "T"}
    ):
        return False
    if not record.alts:
        return False
    return all(
        len(alt) == 1 and alt.upper() in {"A", "C", "G", "T"} for alt in record.alts
    )


def selected_samples(
    vcf: pysam.VariantFile, max_samples: int | None, seed: int
) -> list[str]:
    samples = list(vcf.header.samples)
    if max_samples is None or len(samples) <= max_samples:
        return samples
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(samples), size=max_samples, replace=False))
    return [samples[i] for i in idx]


def open_cohort_reader(
    name: str,
    vcf_path: Path,
    start0: int,
    chrom: str,
    max_samples: int | None,
    seed: int,
) -> CohortReader:
    vcf = pysam.VariantFile(str(vcf_path))
    contig = choose_contig(vcf, chrom)
    samples = selected_samples(vcf, max_samples=max_samples, seed=seed)
    if len(samples) < len(vcf.header.samples):
        vcf.subset_samples(samples)
    return CohortReader(
        name=name, vcf_path=vcf_path, vcf=vcf, contig=contig, samples=samples
    )


def window_haplotype_hashes(
    reader: CohortReader,
    chrom: str,
    start0: int,
    end0: int,
) -> tuple[np.ndarray, int, int, int]:
    hashes = np.zeros(len(reader.samples) * 2, dtype=np.uint64)
    n_snv = 0
    n_records = 0
    n_called_diploid = 0

    for record in reader.vcf.fetch(reader.contig, start0, end0):
        n_records += 1
        if not is_snv_record(record):
            continue
        n_snv += 1
        alt_tokens = [
            mix_uint64(
                stable_uint64(f"{chrom}:{record.pos}:{record.ref}>{alt_idx}:{alt}")
            )
            for alt_idx, alt in enumerate(record.alts, start=1)
        ]
        for sample_i, sample in enumerate(reader.samples):
            gt = record.samples[sample].get("GT")
            if gt is None or len(gt) < 2 or gt[0] is None or gt[1] is None:
                continue
            n_called_diploid += 1
            for hap_offset, allele in enumerate((gt[0], gt[1])):
                if allele and allele > 0 and allele <= len(alt_tokens):
                    hashes[sample_i * 2 + hap_offset] ^= alt_tokens[allele - 1]
    return hashes, n_snv, n_records, n_called_diploid


def rarefy_unique_counts(
    hashes: np.ndarray, target: int, repeats: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    if len(hashes) == 0:
        return np.nan, np.nan, np.nan
    if target <= 0:
        return np.nan, np.nan, np.nan
    if len(hashes) <= target:
        value = float(len(np.unique(hashes)))
        return value, value, value
    counts = np.empty(repeats, dtype=float)
    for i in range(repeats):
        idx = rng.choice(len(hashes), size=target, replace=False)
        counts[i] = len(np.unique(hashes[idx]))
    return (
        float(counts.mean()),
        float(np.quantile(counts, 0.025)),
        float(np.quantile(counts, 0.975)),
    )


def balanced_rarefy_unique_counts(
    cohort_hashes: list[np.ndarray],
    target_per_cohort: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if not cohort_hashes or target_per_cohort <= 0:
        return np.nan, np.nan, np.nan
    counts = np.empty(repeats, dtype=float)
    for i in range(repeats):
        sampled = []
        for hashes in cohort_hashes:
            if len(hashes) <= target_per_cohort:
                sampled.append(hashes)
            else:
                idx = rng.choice(len(hashes), size=target_per_cohort, replace=False)
                sampled.append(hashes[idx])
        counts[i] = len(np.unique(np.concatenate(sampled)))
    return (
        float(counts.mean()),
        float(np.quantile(counts, 0.025)),
        float(np.quantile(counts, 0.975)),
    )


def should_keep_window(chrom: str, start0: int, rate: float, seed: int) -> bool:
    if rate >= 1.0:
        return True
    key = stable_uint64(f"{seed}:{chrom}:{start0}")
    return (int(key) / 2**64) < rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SNV-based window haplotype richness from phased VCFs."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("cohort_manifest.tsv")
    )
    parser.add_argument("--chrom", required=True, choices=AUTOSOMES)
    parser.add_argument("--window-size", type=int, default=10_000)
    parser.add_argument("--window-subsample-rate", type=float, default=1.0)
    parser.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="Split each chromosome's windows into this many shards.",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0, help="Zero-based shard index to run."
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-samples-per-cohort", type=int, default=None)
    parser.add_argument("--rarefy-haplotypes", type=int, default=None)
    parser.add_argument("--balanced-rarefy-haplotypes-per-unit", type=int, default=None)
    parser.add_argument("--rarefaction-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--out-dir", type=Path, default=results_root() / "haplotype_diversity"
    )
    parser.add_argument("--include-empty-windows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_shards < 1:
        raise ValueError("--n-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.n_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < n-shards")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohorts = read_manifest(args.manifest)
    vcf_paths = {
        name: find_vcf_for_chrom(cohort.vcf_dir, args.chrom)
        for name, cohort in cohorts.items()
    }

    with pysam.VariantFile(str(vcf_paths["UKB_full"])) as vcf:
        contig = choose_contig(vcf, args.chrom)
        length = chrom_length(vcf, contig, args.chrom)

    readers = {
        name: open_cohort_reader(
            name=name,
            vcf_path=vcf_path,
            start0=0,
            chrom=args.chrom,
            max_samples=args.max_samples_per_cohort,
            seed=args.seed + stable_uint64(name).item() % 1_000_000,
        )
        for name, vcf_path in vcf_paths.items()
    }

    shard_suffix = (
        ""
        if args.n_shards == 1
        else f".shard{args.shard_index:03d}-of-{args.n_shards:03d}"
    )
    out_path = (
        args.out_dir
        / f"window_haplotype_diversity.chr{args.chrom}{shard_suffix}.tsv.gz"
    )
    rng = np.random.default_rng(args.seed + int(args.chrom))
    windows_written = 0
    windows_seen = 0

    try:
        with gzip.open(out_path, "wt", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                delimiter="\t",
                fieldnames=[
                    "chrom",
                    "window_start",
                    "window_end",
                    "scenario",
                    "scenario_label",
                    "cohorts",
                    "n_haplotypes",
                    "n_unique_haplotypes",
                    "n_novel_vs_ukb_full",
                    "rarefy_target_haplotypes",
                    "rarefied_unique_mean",
                    "rarefied_unique_ci_low",
                    "rarefied_unique_ci_high",
                    "balanced_rarefy_target_per_unit",
                    "balanced_rarefied_unique_mean",
                    "balanced_rarefied_unique_ci_low",
                    "balanced_rarefied_unique_ci_high",
                    "n_snv_records",
                    "n_raw_records",
                    "n_called_diploid_genotypes",
                ],
            )
            writer.writeheader()

            for window_index, start0 in enumerate(range(0, length, args.window_size)):
                if window_index % args.n_shards != args.shard_index:
                    continue
                if not should_keep_window(
                    args.chrom, start0, args.window_subsample_rate, args.seed
                ):
                    continue
                windows_seen += 1
                if args.max_windows is not None and windows_written >= args.max_windows:
                    break
                end0 = min(start0 + args.window_size, length)

                cohort_hashes: dict[str, np.ndarray] = {}
                cohort_snv: dict[str, int] = {}
                cohort_raw: dict[str, int] = {}
                cohort_called: dict[str, int] = {}
                total_snv = 0
                for name, reader in readers.items():
                    hashes, n_snv, n_raw, n_called = window_haplotype_hashes(
                        reader=reader,
                        chrom=args.chrom,
                        start0=start0,
                        end0=end0,
                    )
                    cohort_hashes[name] = hashes
                    cohort_snv[name] = n_snv
                    cohort_raw[name] = n_raw
                    cohort_called[name] = n_called
                    total_snv += n_snv

                if total_snv == 0 and not args.include_empty_windows:
                    continue

                ukb_full_unique = set(np.unique(cohort_hashes["UKB_full"]).tolist())
                scenario_arrays = {
                    scenario: np.concatenate(
                        [cohort_hashes[c] for c in scenario_cohorts]
                    )
                    for scenario, scenario_cohorts, _ in SCENARIOS
                }
                positive_lengths = [
                    len(v) for v in scenario_arrays.values() if len(v) > 0
                ]
                auto_target = min(positive_lengths) if positive_lengths else 0
                rarefy_target = (
                    args.rarefy_haplotypes
                    if args.rarefy_haplotypes is not None
                    else auto_target
                )
                positive_cohort_lengths = [
                    len(v) for v in cohort_hashes.values() if len(v) > 0
                ]
                auto_balanced_target = (
                    min(positive_cohort_lengths) if positive_cohort_lengths else 0
                )
                balanced_target = (
                    args.balanced_rarefy_haplotypes_per_unit
                    if args.balanced_rarefy_haplotypes_per_unit is not None
                    else auto_balanced_target
                )

                for scenario, scenario_cohorts, label in SCENARIOS:
                    hashes = scenario_arrays[scenario]
                    unique_hashes = set(np.unique(hashes).tolist())
                    r_mean, r_low, r_high = rarefy_unique_counts(
                        hashes=hashes,
                        target=min(rarefy_target, len(hashes)),
                        repeats=args.rarefaction_repeats,
                        rng=rng,
                    )
                    balanced_unit_hashes = [
                        np.concatenate([cohort_hashes[c] for c in unit_cohorts])
                        for unit_cohorts in BALANCED_UNITS[scenario]
                    ]
                    unit_target = min(
                        balanced_target, min(len(h) for h in balanced_unit_hashes)
                    )
                    b_mean, b_low, b_high = balanced_rarefy_unique_counts(
                        cohort_hashes=balanced_unit_hashes,
                        target_per_cohort=unit_target,
                        repeats=args.rarefaction_repeats,
                        rng=rng,
                    )
                    writer.writerow(
                        {
                            "chrom": args.chrom,
                            "window_start": start0 + 1,
                            "window_end": end0,
                            "scenario": scenario,
                            "scenario_label": label,
                            "cohorts": ",".join(scenario_cohorts),
                            "n_haplotypes": len(hashes),
                            "n_unique_haplotypes": len(unique_hashes),
                            "n_novel_vs_ukb_full": len(unique_hashes - ukb_full_unique),
                            "rarefy_target_haplotypes": min(rarefy_target, len(hashes)),
                            "rarefied_unique_mean": r_mean,
                            "rarefied_unique_ci_low": r_low,
                            "rarefied_unique_ci_high": r_high,
                            "balanced_rarefy_target_per_unit": unit_target,
                            "balanced_rarefied_unique_mean": b_mean,
                            "balanced_rarefied_unique_ci_low": b_low,
                            "balanced_rarefied_unique_ci_high": b_high,
                            "n_snv_records": sum(
                                cohort_snv[c] for c in scenario_cohorts
                            ),
                            "n_raw_records": sum(
                                cohort_raw[c] for c in scenario_cohorts
                            ),
                            "n_called_diploid_genotypes": sum(
                                cohort_called[c] for c in scenario_cohorts
                            ),
                        }
                    )
                windows_written += 1
                if windows_written % 100 == 0:
                    print(
                        f"chr{args.chrom}: wrote {windows_written} windows", flush=True
                    )
    finally:
        for reader in readers.values():
            reader.close()

    print(
        f"Wrote {out_path} from {windows_written} variant-bearing windows; sampled {windows_seen} windows",
        flush=True,
    )


if __name__ == "__main__":
    main()
