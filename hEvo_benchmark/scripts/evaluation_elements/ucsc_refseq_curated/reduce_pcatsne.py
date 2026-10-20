#!/usr/bin/env python3
"""Run standardized PCA and t-SNE for one model and three fixed seeds."""

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    csv_path = Path(roster["dataset_csv"])
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [{"seqID": x["seqID"], "seq": x["seq"], "label": x["label"]} for x in csv.DictReader(stream)]
    ids = np.asarray([row["seqID"] for row in rows], dtype=str)
    labels = np.asarray([row["label"] for row in rows], dtype=str)
    expected_row_hash = row_order_sha256(rows)
    h5_path = root / "embeddings" / dataset_id / f"{model}.h5"
    h5_manifest_path = h5_path.with_suffix(".manifest.json")
    h5_manifest = json.loads(h5_manifest_path.read_text(encoding="utf-8"))
    if h5_manifest.get("input_sha256") != roster["dataset_sha256"] or h5_manifest.get("row_order_sha256") != expected_row_hash:
        raise ValueError(f"embedding input provenance mismatch: {model}")
    if h5_manifest.get("model_weight_sha256") != spec["model_weight_sha256_expected"]:
        raise ValueError(f"model weight fingerprint mismatch: {model}")
    if h5_manifest.get("pooling") != "mean" or h5_manifest.get("truncation") is not False:
        raise ValueError(f"embedding pooling/truncation mismatch: {model}")
    with h5py.File(h5_path, "r") as handle:
        embeddings = np.asarray(handle["embeddings"][:], dtype=np.float32)
        h5_ids = np.asarray(handle["seqID"][:]).astype(str)
        h5_labels = np.asarray(handle["label"][:]).astype(str)
        attrs = handle.attrs
        for key, value in (("model_name", model), ("dataset_id", dataset_id), ("run_id", roster["run_id"]), ("protocol_id", roster["protocol_id"])):
            if str(attrs.get(key, "")) != value:
                raise ValueError(f"embedding H5 provenance mismatch {key}: {model}")
    if embeddings.shape[0] != 5000 or embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError(f"invalid embedding matrix for {model}: {embeddings.shape}")
    if not np.array_equal(h5_ids, ids) or not np.array_equal(h5_labels, labels):
        raise ValueError(f"embedding row order differs from dataset for {model}")

    scaled = StandardScaler().fit_transform(embeddings).astype(np.float32)
    output_dir = root / "reductions" / dataset_id / model
    for seed in SEEDS:
        pca = PCA(n_components=2, random_state=seed).fit_transform(scaled).astype(np.float32)
        tsne = TSNE(
            n_components=2,
            perplexity=100.0,
            learning_rate="auto",
            init="pca",
            random_state=seed,
            max_iter=5000,
        ).fit_transform(scaled).astype(np.float32)
        output = output_dir / f"seed{seed}.npz"
        atomic_npz(output, seqID=ids, label=labels, pca=pca, tsne=tsne)
        metadata = {
            "run_id": roster["run_id"], "model_name": model, "dataset_id": dataset_id,
            "dataset_sha256": roster["dataset_sha256"], "embedding_h5": str(h5_path),
            "embedding_h5_sha256": sha256_file(h5_path), "embedding_manifest_sha256": sha256_file(h5_manifest_path),
            "protocol_id": roster["protocol_id"], "n_rows": len(rows), "n_classes": len(set(labels.tolist())),
            "embedding_dim": int(embeddings.shape[1]), "standardized": True,
            "pca": {"n_components": 2, "random_state": seed},
            "tsne": {"n_components": 2, "perplexity": 100.0, "learning_rate": "auto", "init": "pca", "max_iter": 5000, "random_state": seed},
            "seed": seed, "coordinate_file": str(output), "coordinate_sha256": sha256_file(output),
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "model": model, "seeds": list(SEEDS)}, flush=True))


if __name__ == "__main__":
    main()
