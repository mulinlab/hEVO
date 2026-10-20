#!/usr/bin/env python3
"""Validate 1KG ancestry loci against target-bearing GPN-Star MSA stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


MODEL_STORES = {
    "GPN-Star-M": [(Path("hg38_cactus447way/243/all.zarr"), 243)],
    "GPN-Star-P": [(Path("hg38_cactus447way/243/all.zarr"), 243)],
    "GPN-Star-V": [(Path("multiz100way-pigz/100/all.zarr"), 100)],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_msa_store(store: Path) -> dict:
    """Fingerprint Zarr metadata and the local file inventory, not chunk contents."""
    if not store.is_dir():
        raise FileNotFoundError(f"MSA Zarr store is not a directory: {store}")

    inventory = []
    metadata = []
    for directory, dirnames, filenames in os.walk(store, followlinks=False):
        dirnames.sort()
        filenames.sort()
        current = Path(directory)
        for name in dirnames:
            path = current / name
            if path.is_symlink():
                inventory.append(
                    {
                        "path": path.relative_to(store).as_posix(),
                        "kind": "symlink-directory",
                        "target": os.readlink(path),
                    }
                )
        for name in filenames:
            path = current / name
            relative = path.relative_to(store).as_posix()
            stat = path.stat()
            inventory.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            if name in {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}:
                metadata.append({"path": relative, "sha256": sha256_file(path)})

    inventory.sort(key=lambda entry: entry["path"])
    metadata.sort(key=lambda entry: entry["path"])
    return {
        "file_count": sum(entry["kind"] == "file" for entry in inventory),
        "metadata_file_count": len(metadata),
        "file_manifest_sha256": canonical_sha256(inventory),
        "metadata_sha256": canonical_sha256(metadata),
    }


def resolve_chrom(chrom: str, msa) -> str:
    no_prefix = chrom.removeprefix("chr")
    for candidate in dict.fromkeys((chrom, no_prefix, "chr" + no_prefix)):
        if candidate in msa.data:
            return candidate
    raise ValueError(f"Chromosome {chrom!r} missing from MSA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--msa-root",
        type=Path,
        default=Path(os.environ["GPN_STAR_MSA_ROOT"])
        if os.environ.get("GPN_STAR_MSA_ROOT")
        else None,
        help="Directory containing hg38_cactus447way/ and multiz100way-pigz/ (or GPN_STAR_MSA_ROOT)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.msa_root is None:
        parser.error("--msa-root or GPN_STAR_MSA_ROOT is required")
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    msa_root = args.msa_root.expanduser().resolve()
    input_file = args.input_file.expanduser().resolve()

    rows = []
    with input_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            position = int(row["center_pos"])
            rows.append(
                {
                    "line": line_number,
                    "chrom": str(row["chr"]),
                    "pos": position,
                    "start": position - 1,
                    "end": position,
                    "ref": str(row["ref_base"]).upper(),
                    "alt": str(row["alt_base"]).upper(),
                }
            )
    if not rows:
        raise ValueError(f"No loci in {input_file}")

    duplicate_keys = len(rows) - len({(r["chrom"], r["pos"]) for r in rows})
    roster_sha256 = canonical_sha256(
        [
            {key: row[key] for key in ("line", "chrom", "pos", "ref", "alt")}
            for row in rows
        ]
    )
    input_sha256 = sha256_file(input_file)
    script_sha256 = sha256_file(Path(__file__).resolve())
    report = {
        "input_file": input_file.name,
        "input_sha256": input_sha256,
        "roster_sha256": roster_sha256,
        "script_sha256": script_sha256,
        "cache_schema_version": 2,
        "loci": len(rows),
        "unique_coordinate_count": len({(r["chrom"], r["pos"]) for r in rows}),
        "duplicate_coordinate_rows": duplicate_keys,
        "models": {},
    }

    if args.output_json.exists():
        previous = json.loads(args.output_json.read_text(encoding="utf-8"))
        report["models"].update(previous.get("models", {}))

    from gpn.star.data import GenomeMSA

    results_by_target_store = {}
    fingerprints_by_target_store = {}
    for model_name, configured_stores in MODEL_STORES.items():
        # Only the target-bearing stores are checked against human reference
        # alleles. The GPN-Star-M 204-way store is source-only and is omitted.
        relative_store, target_species = configured_stores[0]
        target_store = msa_root / relative_store
        store_key = relative_store.as_posix()
        if store_key not in fingerprints_by_target_store:
            fingerprints_by_target_store[store_key] = fingerprint_msa_store(
                target_store
            )
        store_fingerprint = fingerprints_by_target_store[store_key]
        cache_identity = {
            "target_store": store_key,
            "target_species_count": target_species,
            "input_sha256": input_sha256,
            "roster_sha256": roster_sha256,
            "msa_store_fingerprint": store_fingerprint,
            "script_sha256": script_sha256,
            "cache_schema_version": report["cache_schema_version"],
        }
        if store_key in results_by_target_store:
            result = dict(results_by_target_store[store_key])
            result["reused_target_store_check"] = True
            report["models"][model_name] = result
            continue
        existing = report["models"].get(model_name)
        if (
            existing
            and all(existing.get(key) == value for key, value in cache_identity.items())
            and existing.get("status") == "PASS"
            and existing.get("queried_loci") == len(rows)
        ):
            results_by_target_store[store_key] = existing
            continue
        msa = GenomeMSA(str(target_store), n_species=target_species, in_memory=False)
        mismatches = []
        queried = 0
        for offset in range(0, len(rows), args.batch_size):
            block = rows[offset : offset + args.batch_size]
            chroms = [resolve_chrom(row["chrom"], msa) for row in block]
            starts = np.asarray([row["start"] for row in block], dtype=np.int64)
            ends = starts + 1
            observed = msa.get_msa_batch(
                chroms,
                starts,
                ends,
                strands=["+"] * len(block),
                tokenize=True,
            )[:, 0, 0]
            expected = msa.tokenizer(
                np.asarray([row["ref"].encode("ascii") for row in block], dtype="S1")
            )
            mismatch_indices = np.flatnonzero(observed != expected)
            for index in mismatch_indices:
                row = block[int(index)]
                mismatches.append(
                    {
                        "line": row["line"],
                        "chrom": row["chrom"],
                        "pos": row["pos"],
                        "ref": row["ref"],
                        "msa_target_token": int(observed[int(index)]),
                    }
                )
            queried += len(block)
        result = {
            **cache_identity,
            "queried_loci": queried,
            "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:50],
            "status": "PASS" if not mismatches else "FAIL",
        }
        report["models"][model_name] = result
        results_by_target_store[store_key] = result
        print(f"{model_name}: queried={queried} mismatches={len(mismatches)}")
        if mismatches:
            print(json.dumps(mismatches[:10], indent=2))
        store = getattr(getattr(msa, "f", None), "store", None)
        close = getattr(store, "close", None)
        if callable(close):
            close()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    report["status"] = (
        "PASS"
        if all(item["status"] == "PASS" for item in report["models"].values())
        else "FAIL"
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output_json}; overall={report['status']}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
