#!/usr/bin/env python3
"""Merge HDF5 score shards with strict ID, label, and finite-score checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_shard(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        required = {"details", "llrs", "labels"}
        if not required.issubset(handle):
            raise ValueError(f"{path}: missing one of {sorted(required)}")
        details = np.asarray(handle["details"][...])
        llrs = np.asarray(handle["llrs"][...], dtype=float)
        labels = np.asarray(handle["labels"][...], dtype=int)
    if not (len(details) == len(llrs) == len(labels)):
        raise ValueError(f"{path}: details/llrs/labels lengths differ")
    ids = []
    for value in details:
        text = value.decode() if isinstance(value, bytes) else str(value)
        detail = json.loads(text)
        variant_id = str(detail.get("idx", ""))
        if not variant_id:
            raise ValueError(f"{path}: empty detail idx")
        ids.append(variant_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate IDs")
    if not np.isfinite(llrs).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path}: non-finite scores or non-binary labels")
    return details, llrs, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if not args.input:
        raise ValueError("at least one --input is required")
    parts = [read_shard(path) for path in args.input]
    details = np.concatenate([part[0] for part in parts])
    llrs = np.concatenate([part[1] for part in parts])
    labels = np.concatenate([part[2] for part in parts])
    ids = []
    for value in details:
        text = value.decode() if isinstance(value, bytes) else str(value)
        ids.append(str(json.loads(text)["idx"]))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate variant IDs across HDF5 shards")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("details", data=details)
        handle.create_dataset("llrs", data=llrs)
        handle.create_dataset("labels", data=labels)
        handle.attrs["task_name"] = args.task_name
        handle.attrs["merge_operation"] = "concatenate validated score shards"
    manifest = {
        "schema": "gpn_star_hdf5_score_shards_v1",
        "task_name": args.task_name,
        "input_shards": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": int(len(part[1])),
            }
            for path, part in zip(args.input, parts)
        ],
        "rows": int(len(ids)),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "finite_llrs": int(np.isfinite(llrs).sum()),
        "unique_ids": len(set(ids)),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "status": "PASS",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
