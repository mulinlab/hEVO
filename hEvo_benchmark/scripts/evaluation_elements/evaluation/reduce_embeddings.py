#!/usr/bin/env python3
"""Reduce one fresh embedding H5 with the frozen Fig.2b protocol.

The input H5 must be produced for this run.  Its provenance attributes are
required so that historical H5 files cannot silently enter a new reduction.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
DATASET_ROWS = {
    "encode_ccre_v4_regulatory": (8000, 8),
    "gencode_v45_gene_structure": (5000, 5),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def required_attr(attrs: Any, key: str) -> str:
    if key not in attrs or attr_text(attrs[key]).strip() == "":
        raise ValueError(f"input H5 is missing required provenance attribute: {key}")
    return attr_text(attrs[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", choices=sorted(DATASET_ROWS))
    parser.add_argument("--run-id")
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--protocol-id")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--perplexity", type=float, default=100.0)
    parser.add_argument("--tsne-iterations", type=int, default=5000)
    return parser.parse_args()


def read_input(args: argparse.Namespace):
    import h5py
    import numpy as np

    if not args.input_h5.is_file():
        raise FileNotFoundError(args.input_h5)
    with h5py.File(args.input_h5, "r") as handle:
        for key in ("embeddings", "label", "seqID"):
            if key not in handle:
                raise ValueError(f"input H5 is missing dataset: {key}")
        attrs = handle.attrs
        truncation = attrs.get("truncation", None)
        if isinstance(truncation, str):
            truncation = truncation.strip().lower() == "true"
        elif truncation is not None:
            truncation = bool(truncation)
        metadata = {
            "model_name": required_attr(attrs, "model_name"),
            "model_weight_sha256": required_attr(attrs, "model_weight_sha256"),
            "dataset_id": required_attr(attrs, "dataset_id"),
            "dataset_sha256": required_attr(attrs, "dataset_sha256"),
            "input_sha256": required_attr(attrs, "input_sha256"),
            "row_order_sha256": required_attr(attrs, "row_order_sha256"),
            "run_id": required_attr(attrs, "run_id"),
            "protocol_id": required_attr(attrs, "protocol_id"),
            "pooling": attr_text(attrs.get("pooling", "")),
            "embedding_dtype": str(handle["embeddings"].dtype),
            "torch_seed": int(attrs.get("torch_seed", -1)),
            "truncation": truncation,
        }
        embeddings = handle["embeddings"][:]
        labels = np.asarray(
            [attr_text(value) for value in handle["label"][:]], dtype=object
        )
        seq_ids = np.asarray(
            [attr_text(value) for value in handle["seqID"][:]], dtype=object
        )
    if args.dataset_id is None:
        args.dataset_id = metadata["dataset_id"]
    if args.run_id is None:
        args.run_id = metadata["run_id"]
    if args.dataset_sha256 is None:
        args.dataset_sha256 = metadata["dataset_sha256"]
    if args.protocol_id is None:
        args.protocol_id = metadata["protocol_id"]
    expected_rows, expected_classes = DATASET_ROWS[args.dataset_id]
    if metadata["dataset_id"] != args.dataset_id:
        raise ValueError("H5 dataset_id does not match --dataset-id")
    if metadata["dataset_sha256"] != args.dataset_sha256:
        raise ValueError("H5 dataset_sha256 does not match the gated input hash")
    if metadata["run_id"] != args.run_id:
        raise ValueError("H5 run_id does not match; refusing historical H5 input")
    if metadata["protocol_id"] != args.protocol_id:
        raise ValueError("H5 protocol_id does not match the gated dataset protocol")
    if metadata["pooling"] != "mean" or metadata["embedding_dtype"] != "float32":
        raise ValueError("H5 must be mean-pooled float32 embeddings")
    if metadata["torch_seed"] != 42 or metadata["truncation"] is not False:
        raise ValueError("H5 must record torch_seed=42 and truncation=false")
    if metadata["input_sha256"] != metadata["dataset_sha256"]:
        raise ValueError("H5 input_sha256 and dataset_sha256 are inconsistent")
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(f"expected {expected_rows} embedding rows, got {embeddings.shape}")
    if len(labels) != expected_rows or len(seq_ids) != expected_rows:
        raise ValueError("H5 embeddings, labels and seqID have inconsistent row counts")
    if len(set(labels.tolist())) != expected_classes:
        raise ValueError("H5 label count does not match the formal dataset contract")
    if len(set(seq_ids.tolist())) != expected_rows:
        raise ValueError("H5 seqID values are not unique")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("H5 embeddings contain non-finite values")
    metadata.update(
        {
            "n_rows": expected_rows,
            "n_classes": expected_classes,
            "sequence_length": 512,
            "model_fingerprint": metadata["model_weight_sha256"],
        }
    )
    return embeddings, labels, seq_ids, metadata


def main() -> None:
    args = parse_args()
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"the formal protocol requires seeds exactly {list(SEEDS)}")
    if args.perplexity <= 0 or args.tsne_iterations < 1:
        raise ValueError("perplexity and tsne-iterations must be positive")

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    embeddings, labels, seq_ids, metadata = read_input(args)
    if args.perplexity >= len(labels):
        raise ValueError("t-SNE perplexity must be smaller than the number of rows")
    scaled = StandardScaler().fit_transform(embeddings).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("standardized embeddings contain non-finite values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_hash = sha256_file(Path(__file__).resolve())
    for seed in SEEDS:
        pca = PCA(n_components=2, random_state=seed).fit_transform(scaled).astype(np.float32)
        tsne_kwargs = {
            "n_components": 2,
            "perplexity": args.perplexity,
            "learning_rate": "auto",
            "init": "pca",
            "random_state": seed,
        }
        if "max_iter" in inspect.signature(TSNE).parameters:
            tsne_kwargs["max_iter"] = args.tsne_iterations
        else:  # compatibility with older sklearn in a frozen HPC environment
            tsne_kwargs["n_iter"] = args.tsne_iterations
        tsne = TSNE(**tsne_kwargs).fit_transform(scaled).astype(np.float32)
        output = args.output_dir / f"seed{seed}.npz"
        meta_path = args.output_dir / f"seed{seed}.json"
        # Store plain Unicode arrays, not object arrays, so aggregation can
        # use allow_pickle=False and cannot execute data-embedded Python.
        np.savez_compressed(
            output,
            seqID=np.asarray(seq_ids, dtype=str),
            label=np.asarray(labels, dtype=str),
            pca=pca,
            tsne=tsne,
        )
        reduction_metadata = {
            **metadata,
            "input_h5": str(args.input_h5.resolve()),
            "input_h5_sha256": sha256_file(args.input_h5),
            "output_npz": str(output.resolve()),
            "output_npz_sha256": sha256_file(output),
            "reduction_script": str(Path(__file__).resolve()),
            "reduction_script_sha256": script_hash,
            "standardized": True,
            "standardizer": "sklearn.preprocessing.StandardScaler",
            "pca_components": 2,
            "tsne_components": 2,
            "tsne_perplexity": args.perplexity,
            "tsne_learning_rate": "auto",
            "tsne_init": "pca",
            "tsne_iterations": args.tsne_iterations,
            "seed": seed,
            "methods": ["PCA", "tSNE"],
            "model_weight_sha256": metadata["model_weight_sha256"],
            "row_order_sha256": metadata["row_order_sha256"],
            "torch_seed": metadata["torch_seed"],
            "truncation": metadata["truncation"],
        }
        meta_path.write_text(
            json.dumps(reduction_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(output)


if __name__ == "__main__":
    main()
