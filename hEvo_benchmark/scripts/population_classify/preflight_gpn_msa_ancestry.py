#!/usr/bin/env python3
"""Validate the GPN-MSA 1KG inputs and cache the shared 89-species windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


INPUT_WINDOW = 511
INPUT_CENTER = 255
MODEL_WINDOW = 128
FORWARD_CENTER = MODEL_WINDOW // 2
REVERSE_CENTER = FORWARD_CENTER - 1
EXPECTED_SAMPLES = 1500
EXPECTED_ROWS_PER_SAMPLE = 10000
EXPECTED_POPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")
EXPECTED_TOTAL_SPECIES = 90
EXPECTED_AUX_FEATURES = 445


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def canonical_chrom(chrom: str) -> str:
    value = str(chrom)
    return value[3:] if value.startswith("chr") else value


def read_input_manifest(
    input_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(input_dir.glob("*.jsonl"))
    if len(files) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} JSONL files, found {len(files)} in {input_dir}"
        )

    samples: list[dict[str, Any]] = []
    canonical_roster: list[dict[str, Any]] | None = None
    canonical_roster_sha = ""
    identities: set[tuple[str, str]] = set()
    population_counts: Counter[str] = Counter()

    for file_index, path in enumerate(files):
        digest = hashlib.sha256()
        roster_digest = hashlib.sha256()
        row_count = 0
        sample_id = population = None
        first_roster: list[dict[str, Any]] = []
        unique_loci: set[tuple[str, int]] = set()

        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc

                identity = (str(row["sample"]), str(row["sample_popu"]))
                if sample_id is None:
                    sample_id, population = identity
                    if path.stem != f"{population}_{sample_id}":
                        raise ValueError(
                            f"File/sample identity mismatch: {path} vs {identity}"
                        )
                elif identity != (sample_id, population):
                    raise ValueError(f"Mixed sample identity in {path}:{line_number}")

                sequence = str(row["seq"]).upper()
                reference = str(row["ref_base"]).upper()
                alternate = str(row["alt_base"]).upper()
                if len(sequence) != INPUT_WINDOW or sequence[INPUT_CENTER] != alternate:
                    raise ValueError(
                        f"Invalid {INPUT_WINDOW}-bp sample haplotype at {path}:{line_number}"
                    )
                if reference not in "ACGT" or alternate not in "ACGT":
                    raise ValueError(
                        f"Invalid allele at {path}:{line_number}: {reference}>{alternate}"
                    )
                chrom = str(row["chr"])
                position = int(row["center_pos"])
                if position <= FORWARD_CENTER:
                    raise ValueError(
                        f"Invalid 1-based coordinate at {path}:{line_number}: {chrom}:{position}"
                    )

                roster_digest.update(
                    f"{chrom}\t{position}\t{reference}\n".encode("ascii")
                )
                unique_loci.add((chrom, position))
                if file_index == 0:
                    first_roster.append(
                        {"chrom": chrom, "position": position, "reference": reference}
                    )
                row_count += 1

        if sample_id is None:
            raise ValueError(f"Empty JSONL input: {path}")
        if row_count != EXPECTED_ROWS_PER_SAMPLE:
            raise ValueError(
                f"{path}: expected {EXPECTED_ROWS_PER_SAMPLE} rows, found {row_count}"
            )
        if len(unique_loci) != EXPECTED_ROWS_PER_SAMPLE:
            raise ValueError(
                f"{path}: expected unique coordinates, found {len(unique_loci)}"
            )
        if (sample_id, population) in identities:
            raise ValueError(f"Duplicate sample identity: {identity}")
        identities.add((sample_id, population))
        population_counts[population] += 1

        current_roster_sha = roster_digest.hexdigest()
        if file_index == 0:
            canonical_roster = first_roster
            canonical_roster_sha = current_roster_sha
        elif current_roster_sha != canonical_roster_sha:
            raise ValueError(
                f"Coordinate/reference roster differs from the first input: {path}"
            )

        samples.append(
            {
                "input_file": str(path.resolve()),
                "file_name": path.name,
                "sample_id": sample_id,
                "population": population,
                "input_rows": row_count,
                "input_sha256": digest.hexdigest(),
                "locus_roster_sha256": current_roster_sha,
            }
        )

    expected_counts = {population: 300 for population in EXPECTED_POPULATIONS}
    if dict(population_counts) != expected_counts:
        raise ValueError(
            f"Expected 300 samples per super-population, found {dict(population_counts)}"
        )
    if canonical_roster is None or len(canonical_roster) != EXPECTED_ROWS_PER_SAMPLE:
        raise RuntimeError("Could not construct the canonical 10,000-locus roster")
    return samples, canonical_roster


class ZipGenomeMSA:
    """Use GPN's official GenomeMSA methods on the published ZIP-backed Zarr."""

    def __init__(self, msa_path: Path):
        import zarr
        from gpn.data import GenomeMSA, ReverseComplementer, Tokenizer

        self._base_class = GenomeMSA
        if msa_path.is_file() and msa_path.suffix == ".zip":
            self.store = zarr.storage.ZipStore(str(msa_path), mode="r")
            self.root = zarr.open_group(store=self.store, mode="r", zarr_format=2)
        else:
            self.store = None
            self.root = zarr.open_group(str(msa_path), mode="r", zarr_format=2)

        self.genome_msa = object.__new__(GenomeMSA)
        self.genome_msa.reverse_complementer = ReverseComplementer()
        self.genome_msa.tokenizer = Tokenizer()
        self.genome_msa.f = self.root
        self.genome_msa.data = {chrom: self.root[chrom] for chrom in self.root.keys()}

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()


def get_msa_identity(msa: ZipGenomeMSA, msa_path: Path) -> dict[str, Any]:
    chroms = sorted(msa.genome_msa.data, key=lambda value: (not value.isdigit(), value))
    shape_by_chrom = {chrom: list(msa.genome_msa.data[chrom].shape) for chrom in chroms}
    wrong = {
        chrom: shape
        for chrom, shape in shape_by_chrom.items()
        if len(shape) != 2 or shape[1] != EXPECTED_TOTAL_SPECIES
    }
    if wrong:
        raise ValueError(
            f"MSA must contain human + 89 auxiliary species; invalid shapes: {wrong}"
        )
    if len(chroms) < 24:
        raise ValueError(
            f"Expected a whole-genome hg38 MSA, found only {len(chroms)} chromosomes"
        )

    stat = msa_path.stat()
    return {
        "path": str(msa_path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "zarr_format": 2,
        "chromosome_count": len(chroms),
        "chromosomes": chroms,
        "shape_by_chrom": shape_by_chrom,
        "total_species_including_human": EXPECTED_TOTAL_SPECIES,
        "auxiliary_species": EXPECTED_TOTAL_SPECIES - 1,
        "zarr_20_array_metadata_sha256": canonical_json_sha256(
            {
                "20": shape_by_chrom.get("20"),
                "chunks": list(msa.genome_msa.data["20"].chunks),
                "dtype": str(msa.genome_msa.data["20"].dtype),
            }
        ),
    }


def build_window_cache(
    msa: ZipGenomeMSA,
    roster: list[dict[str, Any]],
    cache_dir: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    from gpn.data import Tokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    auxiliary_count = EXPECTED_TOTAL_SPECIES - 1
    n_rows = len(roster)
    forward_path = cache_dir / "msa_aux_forward.npy"
    reverse_path = cache_dir / "msa_aux_reverse.npy"
    partial_forward = cache_dir / "msa_aux_forward.npy.partial"
    partial_reverse = cache_dir / "msa_aux_reverse.npy.partial"
    forward = np.lib.format.open_memmap(
        partial_forward,
        mode="w+",
        dtype=np.uint8,
        shape=(n_rows, MODEL_WINDOW, auxiliary_count),
    )
    reverse = np.lib.format.open_memmap(
        partial_reverse,
        mode="w+",
        dtype=np.uint8,
        shape=(n_rows, MODEL_WINDOW, auxiliary_count),
    )

    chroms = []
    starts = np.empty(n_rows, dtype=np.int64)
    ends = np.empty(n_rows, dtype=np.int64)
    expected_refs = np.empty(n_rows, dtype="S1")
    for index, row in enumerate(roster):
        chrom = canonical_chrom(row["chrom"])
        if chrom not in msa.genome_msa.data:
            raise ValueError(f"Chromosome {row['chrom']!r} is absent from the MSA")
        position0 = int(row["position"]) - 1
        start = position0 - FORWARD_CENTER
        end = start + MODEL_WINDOW
        if start < 0 or end > msa.genome_msa.data[chrom].shape[0]:
            raise ValueError(f"Out-of-range model window: {chrom}:{row['position']}")
        chroms.append(chrom)
        starts[index] = start
        ends[index] = end
        expected_refs[index] = row["reference"].encode("ascii")

    tokenizer = Tokenizer()
    for offset in range(0, n_rows, batch_size):
        end_offset = min(offset + batch_size, n_rows)
        msa_forward, msa_reverse = msa.genome_msa.get_msa_batch_fwd_rev(
            np.asarray(chroms[offset:end_offset]),
            starts[offset:end_offset],
            ends[offset:end_offset],
            tokenize=True,
        )
        if msa_forward.shape[1:] != (MODEL_WINDOW, EXPECTED_TOTAL_SPECIES):
            raise RuntimeError(
                f"Unexpected forward MSA batch shape: {msa_forward.shape}"
            )
        target_at_variant = msa_forward[:, FORWARD_CENTER, 0]
        expected_tokens = tokenizer(expected_refs[offset:end_offset])
        if not np.array_equal(target_at_variant, expected_tokens):
            mismatches = np.flatnonzero(target_at_variant != expected_tokens)
            first = offset + int(mismatches[0])
            row = roster[first]
            raise ValueError(
                f"Reference mismatch at {row['chrom']}:{row['position']}: "
                f"input={row['reference']} MSA={bytes([int(target_at_variant[mismatches[0]])])}"
            )
        forward[offset:end_offset] = msa_forward[:, :, 1:]
        reverse[offset:end_offset] = msa_reverse[:, :, 1:]
        print(f"[msa cache] {end_offset}/{n_rows} loci", flush=True)

    forward.flush()
    reverse.flush()
    del forward, reverse
    os.replace(partial_forward, forward_path)
    os.replace(partial_reverse, reverse_path)

    return {
        "forward_path": str(forward_path.resolve()),
        "reverse_path": str(reverse_path.resolve()),
        "forward_shape": [n_rows, MODEL_WINDOW, auxiliary_count],
        "reverse_shape": [n_rows, MODEL_WINDOW, auxiliary_count],
        "dtype": "uint8",
        "forward_sha256": sha256_file(forward_path),
        "reverse_sha256": sha256_file(reverse_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--msa-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--hash-msa",
        action="store_true",
        help="Compute a full SHA-256 over the 34-GB MSA archive",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    msa_path = args.msa_path.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config_path = model_path / "config.json"
    weight_path = model_path / "pytorch_model.bin"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint config/weights missing under {model_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("n_aux_features", -1)) != EXPECTED_AUX_FEATURES:
        raise ValueError(
            f"Checkpoint expects n_aux_features={config.get('n_aux_features')}; "
            f"expected {EXPECTED_AUX_FEATURES} for the published 89.zarr input"
        )
    if int(config.get("hidden_size", -1)) != 768:
        raise ValueError(f"Expected hidden_size=768, found {config.get('hidden_size')}")

    print("[inputs] validating all sample files and shared locus roster", flush=True)
    samples, roster = read_input_manifest(input_dir)
    roster_sha = samples[0]["locus_roster_sha256"]
    run_root.mkdir(parents=True, exist_ok=True)
    config_dir = run_root / "config"
    cache_dir = run_root / "cache"
    config_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(config_dir / "sample_manifest.json", samples)
    roster_path = config_dir / "locus_roster.jsonl"
    temporary_roster = roster_path.with_suffix(".jsonl.partial")
    with temporary_roster.open("w", encoding="utf-8") as handle:
        for row in roster:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary_roster, roster_path)
    input_list = config_dir / "input_files.txt"
    temporary_list = input_list.with_suffix(".txt.partial")
    temporary_list.write_text(
        "\n".join(item["input_file"] for item in samples) + "\n", encoding="utf-8"
    )
    os.replace(temporary_list, input_list)

    print(
        "[msa] opening the official 89.zarr and checking reference alleles", flush=True
    )
    msa = ZipGenomeMSA(msa_path)
    try:
        msa_identity = get_msa_identity(msa, msa_path)
        cache = build_window_cache(msa, roster, cache_dir, batch_size=args.batch_size)
    finally:
        msa.close()

    model_version = importlib.metadata.version("gpn")
    model_identity = {
        "path": str(model_path),
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weight_path),
        "model_type": config.get("model_type"),
        "hidden_size": int(config["hidden_size"]),
        "n_aux_features": int(config["n_aux_features"]),
        "aux_features_vocab_size": int(config.get("aux_features_vocab_size", -1)),
        "transformers_version": importlib.metadata.version("transformers"),
        "gpn_version": model_version,
    }
    if args.hash_msa:
        print("[msa] hashing the full MSA archive", flush=True)
        msa_identity["sha256"] = sha256_file(msa_path)
    else:
        msa_identity["sha256"] = None

    report = {
        "status": "preflight_complete",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "input_sample_count": len(samples),
        "rows_per_sample": EXPECTED_ROWS_PER_SAMPLE,
        "population_counts": {population: 300 for population in EXPECTED_POPULATIONS},
        "locus_count": len(roster),
        "locus_roster_sha256": roster_sha,
        "input_window_bp": INPUT_WINDOW,
        "model_window_bp": MODEL_WINDOW,
        "forward_center_index": FORWARD_CENTER,
        "reverse_center_index": REVERSE_CENTER,
        "center_pool_size": 5,
        "total_species_including_human": EXPECTED_TOTAL_SPECIES,
        "auxiliary_species": EXPECTED_TOTAL_SPECIES - 1,
        "checkpoint": model_identity,
        "msa": msa_identity,
        "msa_window_cache": cache,
        "aggregation": "topk500_mean_global_mean_norm_weighted_mean_four_scalar_stats",
        "embedding_width": 3 * int(config["hidden_size"]) + 4,
        "sample_haplotype": "GT[0] as stored in the existing sample-specific JSONL",
    }
    atomic_json(config_dir / "preflight.json", report)
    print(
        f"[preflight complete] {len(samples)} samples x {len(roster)} loci; "
        f"MSA species={msa_identity['total_species_including_human']}; "
        f"embedding width={report['embedding_width']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
