#!/usr/bin/env python3
"""Prepare immutable, traceable inputs for the non-ClinVar GPN-Star refresh.

The source CSV/HDF5 files are never edited.  This command writes filtered CSV
views and, for the Rare Causal cache, filtered HDF5 copies to an external run
directory together with a manifest containing row counts, removed IDs and
SHA256 digests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


MPRA_BLOCKED_IDS = {
    "chr7:142197163:G:A",
    "chr7:142163068:T:A",
}

TASKS = {
    "Causal": {
        "source": "hEvo_benchmark/datasets/Causal_Variants/causal_variants_hg38_1023.csv",
        "output": "causal_variants_hg38_1023_clean.csv",
        "expected": (11087, 1117, 9970),
    },
    "Rare_Causal": {
        "source": "hEvo_benchmark/datasets/Rare_Causal_Variants/rare_causal/rare_causal_ref_alt_hg38_1023.csv",
        "output": "rare_causal_ref_alt_hg38_1023_clean.csv",
        "expected": (1101, 426, 675),
    },
    "Rare_Variants": {
        "source": "hEvo_benchmark/datasets/Rare_Variants/rare_variants_ref_alt_dataset_w1023.csv",
        "output": "rare_variants_ref_alt_dataset_w1023_clean.csv",
        "expected": (1992, 1314, 678),
    },
    "MPRA": {
        "source": "hEvo_benchmark/datasets/MPRA/mpra_emvar_hg38_1023.csv",
        "output": "mpra_emvar_hg38_1023_clean.csv",
        "expected": (9998, 2640, 7358),
    },
    "DDD": {
        "source": "hEvo_benchmark/datasets/Denovo_Variants/DDD_LoF_splice_vs_ClinVar_benign/ddd_public_research_vs_clinvar_benign_plof_splice_snv.csv",
        "output": "ddd_public_research_vs_clinvar_benign_plof_splice_snv_clean.csv",
        "expected": (1836, 306, 1530),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_id(row: dict[str, str]) -> str:
    return row.get("seqID") or row.get("SeqID") or ""


def counts(rows: Iterable[dict[str, str]]) -> tuple[int, int, int]:
    values = [int(row["label"]) for row in rows]
    return len(values), sum(values), len(values) - sum(values)


def filter_csv(source: Path, destination: Path, task: str) -> dict[str, object]:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "label" not in fieldnames:
            raise ValueError(f"{source} has no label column")
        source_rows = list(reader)
    ids = [row_id(row) for row in source_rows]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{source} has missing or duplicate variant IDs")
    removed = []
    retained = []
    for row in source_rows:
        sid = row_id(row)
        if task == "MPRA" and sid in MPRA_BLOCKED_IDS:
            removed.append({"variant_id": sid, **row})
        else:
            retained.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(retained)
    observed = counts(retained)
    expected = TASKS[task]["expected"]
    if observed != expected:
        raise ValueError(f"{task} counts {observed} do not match expected {expected}")
    return {
        "task": task,
        "input": str(source.resolve()),
        "input_sha256": sha256(source),
        "output": str(destination.resolve()),
        "output_sha256": sha256(destination),
        "input_rows": len(source_rows),
        "output_rows": observed[0],
        "num_positive": observed[1],
        "num_negative": observed[2],
        "removed_rows": len(removed),
        "removed_ids": [row["variant_id"] for row in removed],
        "removed_records": removed,
        "operation": "remove only declared MPRA MSA-blocked IDs"
        if task == "MPRA"
        else "immutable copy of current clean cohort",
    }


def filter_h5(
    source: Path, destination: Path, allowed_ids: set[str]
) -> dict[str, object]:
    with h5py.File(source, "r") as source_h5:
        if (
            "details" not in source_h5
            or "labels" not in source_h5
            or "llrs" not in source_h5
        ):
            raise ValueError(f"{source} lacks required details/labels/llrs datasets")
        raw_details = source_h5["details"][...]
        details = [
            json.loads(value.decode() if isinstance(value, bytes) else str(value))
            for value in raw_details
        ]
        ids = [str(detail.get("idx", "")) for detail in details]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{source} has missing or duplicate HDF5 IDs")
        keep = np.array([sid in allowed_ids for sid in ids], dtype=bool)
        if set(ids) < allowed_ids:
            raise ValueError(f"{source} does not cover all requested IDs")
        with h5py.File(destination, "w") as output_h5:
            for name, dataset in source_h5.items():
                if name == "skipped_details":
                    continue
                values = dataset[...]
                if len(values) == len(keep):
                    values = values[keep]
                output_h5.create_dataset(name, data=values)
            output_h5.attrs["source_hdf5"] = str(source.resolve())
            output_h5.attrs["filter_operation"] = (
                "project onto current Rare Causal ID set"
            )
    return {
        "input": str(source.resolve()),
        "input_sha256": sha256(source),
        "output": str(destination.resolve()),
        "output_sha256": sha256(destination),
        "input_rows": len(ids),
        "output_rows": int(keep.sum()),
        "removed_rows": int((~keep).sum()),
        "removed_ids": [sid for sid, keep_row in zip(ids, keep) if not keep_row],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rare-causal-h5-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    input_dir = output_root / "inputs"
    h5_dir = output_root / "filtered_h5" / "Rare_Causal"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "gpn_star_nonclinvar_cleaning_v1",
        "status": "RUNNING",
        "created_from": str(repo_root),
        "tasks": {},
        "hdf5_filters": [],
    }
    for task, spec in TASKS.items():
        source = repo_root / spec["source"]
        destination = input_dir / spec["output"]
        manifest["tasks"][task] = filter_csv(source, destination, task)
    with (input_dir / TASKS["Rare_Causal"]["output"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rare_ids = {row_id(row) for row in csv.DictReader(handle)}
    h5_files = sorted(args.rare_causal_h5_root.resolve().glob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"no HDF5 files under {args.rare_causal_h5_root}")
    for source in h5_files:
        destination = h5_dir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest["hdf5_filters"].append(filter_h5(source, destination, rare_ids))
    manifest["status"] = "PASS"
    manifest["manifest_output"] = str(
        (output_root / "cleaning_manifest.json").resolve()
    )
    manifest_path = output_root / "cleaning_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
