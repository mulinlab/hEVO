#!/usr/bin/env python3
"""Run the frozen all-30 UMAP protocol for one model and three seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

SEEDS = (42, 43, 44)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_order_sha256(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['seqID']}\t{row['label']}\t{row['seq']}\n".encode("utf-8"))
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".partial.npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    roster = json.loads((root / "config" / "model_roster.json").read_text(encoding="utf-8"))
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    model = roster["models"][index]
    dataset_id = roster["dataset_id"]
    spec = roster["model_specs"][model]
    with Path(roster["dataset_csv"]).open(newline="", encoding="utf-8") as stream:
        rows = [{"seqID": x["seqID"], "seq": x["seq"], "label": x["label"]} for x in csv.DictReader(stream)]
    ids = np.asarray([row["seqID"] for row in rows], dtype=str)
    labels = np.asarray([row["label"] for row in rows], dtype=str)
    h5_path = root / "embeddings" / dataset_id / f"{model}.h5"
    h5_manifest_path = h5_path.with_suffix(".manifest.json")
    h5_manifest = json.loads(h5_manifest_path.read_text(encoding="utf-8"))
    if h5_manifest.get("input_sha256") != roster["dataset_sha256"] or h5_manifest.get("row_order_sha256") != row_order_sha256(rows):
        raise ValueError(f"embedding input provenance mismatch: {model}")
    if h5_manifest.get("model_weight_sha256") != spec["model_weight_sha256_expected"]:
        raise ValueError(f"model weight fingerprint mismatch: {model}")
    with h5py.File(h5_path, "r") as handle:
        matrix = np.asarray(handle["embeddings"][:], dtype=np.float32)
        h5_ids, h5_labels = np.asarray(handle["seqID"][:]).astype(str), np.asarray(handle["label"][:]).astype(str)
    if matrix.shape[0] != 5000 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"invalid embedding matrix for {model}: {matrix.shape}")
    if not np.array_equal(h5_ids, ids) or not np.array_equal(h5_labels, labels):
        raise ValueError(f"embedding row order differs from dataset for {model}")
    scaled = StandardScaler().fit_transform(matrix).astype(np.float32)

    output_dir = root / "reductions" / dataset_id / model
    for seed in SEEDS:
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
            init="spectral", n_epochs=500, learning_rate=1.0, spread=1.0,
            random_state=seed, transform_seed=seed, n_jobs=1, low_memory=True,
        )
        coords = np.asarray(reducer.fit_transform(scaled), dtype=np.float32)
        if coords.shape != (5000, 2) or not np.isfinite(coords).all():
            raise ValueError(f"invalid UMAP output for {model}, seed {seed}: {coords.shape}")
        output = output_dir / f"umap_seed{seed}.npz"
        atomic_npz(output, seqID=ids, label=labels, umap=coords)
        metadata = {
            "run_id": roster["run_id"], "model_name": model, "dataset_id": dataset_id,
            "dataset_sha256": roster["dataset_sha256"], "embedding_h5": str(h5_path),
            "embedding_h5_sha256": sha256_file(h5_path), "embedding_manifest_sha256": sha256_file(h5_manifest_path),
            "protocol_id": roster["protocol_id"], "n_rows": 5000, "n_classes": 5,
            "embedding_dim": int(matrix.shape[1]), "standardized": True,
            "umap": {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "init": "spectral", "n_epochs": 500, "learning_rate": 1.0, "spread": 1.0, "random_state": seed, "transform_seed": seed, "n_jobs": 1, "low_memory": True},
            "seed": seed, "coordinate_file": str(output), "coordinate_sha256": sha256_file(output),
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "model": model, "seeds": list(SEEDS)}, flush=True))


if __name__ == "__main__":
    main()
