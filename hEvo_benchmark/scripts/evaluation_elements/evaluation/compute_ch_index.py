#!/usr/bin/env python3
"""Aggregate strict 29-model x 3-seed x 2-method CH source tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
MODELS = (
    "AIDO.DNA-300M",
    "Caduceus-PH",
    "Caduceus-PS",
    "DNABERT",
    "DNABERT2",
    "GENA-LM-base",
    "GENA-LM-large",
    "GENERator-1.2B",
    "GENERator-3B",
    "GPN-promoter",
    "GROVER",
    "Genos-1.2B",
    "HyenaDNA_tiny_1k",
    "Mistral-DNA-138M",
    "Mistral-DNA-17M",
    "Mistral-DNA-422M",
    "NT_2.5b_1kG",
    "NT_2.5b_MS",
    "NT_500m_1kG",
    "NT_500m_MS",
    "NTv3-100M-pre",
    "NTv3-650M-pre",
    "NTv3-8M-pre",
    "SpeciesLM",
    "evo2_1b_base",
    "evo2_7b_base",
    "hEvo-base",
    "hEvo-large",
    "hEvo-open-base",
)
DATASET_ROWS = {
    "encode_ccre_v4_regulatory": (8000, 8),
    "gencode_v45_gene_structure": (5000, 5),
}
METHODS = ("PCA", "tSNE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reductions-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", choices=sorted(DATASET_ROWS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_meta(npz_path: Path) -> dict[str, Any]:
    meta_path = npz_path.with_suffix(".json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing reduction metadata beside {npz_path}")
    value = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid reduction metadata: {meta_path}")
    return value


def find_reductions(args: argparse.Namespace) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
    found: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(args.reductions_dir.rglob("seed*.npz")):
        meta = load_meta(path)
        if meta.get("dataset_id") != args.dataset_id:
            continue
        if meta.get("run_id") != args.run_id:
            continue
        model = text(meta.get("model_name", ""))
        seed = int(meta.get("seed", -1))
        if model not in MODELS or seed not in SEEDS:
            continue
        key = (model, seed)
        if key in found:
            raise ValueError(f"duplicate reduction for {model} seed {seed}")
        found[key] = (path, meta)
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    missing = sorted(expected - set(found))
    if missing:
        raise FileNotFoundError(
            "strict reduction completeness failed; missing "
            + ", ".join(f"{model}/seed{seed}" for model, seed in missing)
        )
    return found


def validate_meta(meta: dict[str, Any], args: argparse.Namespace, model: str, seed: int) -> None:
    expected_rows, expected_classes = DATASET_ROWS[args.dataset_id]
    required = {
        "model_name": model,
        "dataset_id": args.dataset_id,
        "run_id": args.run_id,
        "dataset_sha256": args.dataset_sha256,
        "protocol_id": args.protocol_id,
        "n_rows": expected_rows,
        "n_classes": expected_classes,
        "sequence_length": 512,
        "pooling": "mean",
        "embedding_dtype": "float32",
        "standardized": True,
        "pca_components": 2,
        "tsne_components": 2,
        "tsne_perplexity": 100.0,
        "tsne_learning_rate": "auto",
        "tsne_init": "pca",
        "tsne_iterations": 5000,
        "seed": seed,
        "torch_seed": 42,
        "truncation": False,
    }
    for key, expected in required.items():
        actual = meta.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(float(actual), expected):
                raise ValueError(f"{model}/seed{seed}: metadata mismatch for {key}")
        elif actual != expected:
            raise ValueError(f"{model}/seed{seed}: metadata mismatch for {key}")
    for key in (
        "model_fingerprint",
        "model_weight_sha256",
        "input_sha256",
        "row_order_sha256",
    ):
        if not text(meta.get(key, "")).strip():
            raise ValueError(f"{model}/seed{seed}: missing metadata {key}")


def read_coords(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        for key in ("label", "seqID", "pca", "tsne"):
            if key not in data:
                raise ValueError(f"reduction missing {key}: {path}")
        labels = np.asarray(data["label"], dtype=object)
        seq_ids = np.asarray(data["seqID"], dtype=object)
        coords = {method: np.asarray(data[key], dtype=np.float64) for method, key in (("PCA", "pca"), ("tSNE", "tsne"))}
    if len(labels) != len(seq_ids) or len(set(seq_ids.tolist())) != len(seq_ids):
        raise ValueError(f"invalid labels/seqID in {path}")
    for method, value in coords.items():
        if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] != len(labels):
            raise ValueError(f"invalid {method} coordinates in {path}: {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite {method} coordinates in {path}")
    return labels, coords


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(MODELS) != 29:
        raise AssertionError("formal roster must contain exactly 29 models")
    found = find_reductions(args)
    from sklearn.metrics import calinski_harabasz_score
    import numpy as np

    source_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for model in MODELS:
        model_fingerprints: set[str] = set()
        weight_hashes: set[str] = set()
        input_hashes: set[str] = set()
        row_orders: set[str] = set()
        expected_rows, expected_classes = DATASET_ROWS[args.dataset_id]
        for seed in SEEDS:
            path, meta = found[(model, seed)]
            validate_meta(meta, args, model, seed)
            model_fingerprints.add(text(meta["model_fingerprint"]))
            weight_hashes.add(text(meta["model_weight_sha256"]))
            input_hashes.add(text(meta["input_sha256"]))
            row_orders.add(text(meta["row_order_sha256"]))
            labels, coords_by_method = read_coords(path)
            if len(labels) != expected_rows:
                raise ValueError(
                    f"{model}/seed{seed}: reduction has {len(labels)} rows; "
                    f"expected {expected_rows}"
                )
            if len(set(labels.tolist())) != expected_classes:
                raise ValueError(
                    f"{model}/seed{seed}: reduction has {len(set(labels.tolist()))} "
                    f"classes; expected {expected_classes}"
                )
            label_ids = {label: index for index, label in enumerate(sorted(set(labels.tolist())))}
            y = np.asarray([label_ids[label] for label in labels], dtype=np.int64)
            for method in METHODS:
                ch = float(calinski_harabasz_score(coords_by_method[method], y))
                row = {
                    "model_name": model,
                    "seed": seed,
                    "method": method,
                    "n_rows": len(labels),
                    "n_classes": len(label_ids),
                    "ch_index": ch,
                    "silhouette": float("nan"),
                    "coordinate_space": "2D projected coordinates",
                    "metric": "euclidean",
                    "dataset_id": args.dataset_id,
                    "run_id": args.run_id,
                    "dataset_sha256": args.dataset_sha256,
                    "input_sha256": text(meta["input_sha256"]),
                    "model_fingerprint": text(meta["model_fingerprint"]),
                    "model_weight_sha256": text(meta["model_weight_sha256"]),
                    "row_order_sha256": text(meta["row_order_sha256"]),
                    "torch_seed": meta["torch_seed"],
                    "truncation": meta["truncation"],
                    "protocol_id": args.protocol_id,
                }
                source_rows.append(row)
                grouped[(model, method)].append(row)
        if (
            len(model_fingerprints) != 1
            or len(weight_hashes) != 1
            or len(input_hashes) != 1
            or len(row_orders) != 1
        ):
            raise ValueError(f"metadata fingerprint/hash changed across seeds for {model}")

    if len(source_rows) != 29 * 3 * 2:
        raise AssertionError(f"expected 174 source rows, got {len(source_rows)}")
    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for method in METHODS:
            rows = grouped[(model, method)]
            if len(rows) != 3:
                raise AssertionError(f"expected three seeds for {model}/{method}")
            ch = np.asarray([float(row["ch_index"]) for row in rows], dtype=np.float64)
            summary_rows.append(
                {
                    "model_name": model,
                    "method": method,
                    "n_seeds": 3,
                    "n_rows": rows[0]["n_rows"],
                    "n_classes": rows[0]["n_classes"],
                    "ch_index_mean": float(ch.mean()),
                    "ch_index_sd": float(ch.std(ddof=1)),
                    "silhouette_mean": float("nan"),
                    "silhouette_sd": float("nan"),
                    "dataset_id": args.dataset_id,
                    "run_id": args.run_id,
                    "dataset_sha256": args.dataset_sha256,
                    "input_sha256": rows[0]["input_sha256"],
                    "model_fingerprint": rows[0]["model_fingerprint"],
                    "model_weight_sha256": rows[0]["model_weight_sha256"],
                    "row_order_sha256": rows[0]["row_order_sha256"],
                    "torch_seed": rows[0]["torch_seed"],
                    "truncation": rows[0]["truncation"],
                    "protocol_id": args.protocol_id,
                }
            )
    if len(summary_rows) != 29 * 2:
        raise AssertionError(f"expected 58 summary rows, got {len(summary_rows)}")
    write_csv(args.source_csv, source_rows)
    write_csv(args.summary_csv, summary_rows)
    print(json.dumps({"dataset_id": args.dataset_id, "source_rows": 174, "summary_rows": 58}, indent=2))


if __name__ == "__main__":
    main()
