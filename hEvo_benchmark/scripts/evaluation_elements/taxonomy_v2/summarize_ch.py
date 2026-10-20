"""Calculate taxonomy-v2 CH from saved PCA/t-SNE coordinates."""

import csv
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import calinski_harabasz_score


TASKS = {
    "encode_ccre_v4_regulatory": (5000, 5, "S11"),
    "gencode_v45_gene_structure": (3000, 3, "S13"),
}
SEEDS = (42, 43, 44)
METHODS = ("PCA", "tSNE")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="External taxonomy-v2 run directory containing manifests and reductions.")
    parser.add_argument("--tables-dir", type=Path,
                        help="Output directory for CH tables; defaults to <run-dir>/tables.")
    args = parser.parse_args()
    root = args.run_dir.expanduser().resolve()
    tables = (args.tables_dir.expanduser().resolve() if args.tables_dir else root / "tables")
    gate = json.loads((root / "dataset_gate.json").read_text())
    tasks = json.loads((root / "reduction_tasks.json").read_text())
    models = sorted({task["model_name"] for task in tasks})
    if len(models) != 30 or len(tasks) != 60:
        raise ValueError("Expected 30 models and 60 model-dataset reduction tasks")
    tables.mkdir(parents=True, exist_ok=True)
    rank_rows = []
    manifest = {"status": "PASS", "protocol_id": gate["protocol_id"], "run_id": gate["run_id"], "tables": {}}
    for dataset, (expected_rows, expected_classes, table_id) in TASKS.items():
        dataset_hash = gate["dataset_sha256"][dataset]
        source_rows, grouped = [], defaultdict(list)
        for model in models:
            for seed in SEEDS:
                npz_path = root / "reductions" / dataset / model / f"seed{seed}.npz"
                meta_path = npz_path.with_suffix(".json")
                if not npz_path.is_file() or not meta_path.is_file():
                    raise FileNotFoundError(f"Missing reduction output: {npz_path}")
                meta = json.loads(meta_path.read_text())
                expected_meta = {
                    "model_name": model, "seed": seed, "dataset_id": dataset,
                    "run_id": gate["run_id"], "dataset_sha256": dataset_hash,
                    "protocol_id": gate["protocol_id"], "n_rows": expected_rows,
                    "n_classes": expected_classes, "sequence_length": 512,
                    "pooling": "mean", "embedding_dtype": "float32", "standardized": True,
                    "pca_components": 2, "tsne_components": 2, "tsne_perplexity": 100.0,
                    "tsne_learning_rate": "auto", "tsne_init": "pca", "tsne_iterations": 5000,
                    "torch_seed": 42, "truncation": False,
                }
                for key, expected in expected_meta.items():
                    if meta.get(key) != expected:
                        raise ValueError(f"Metadata mismatch at {npz_path}: {key}={meta.get(key)!r}; expected {expected!r}")
                if not meta.get("model_fingerprint") or not meta.get("model_weight_sha256"):
                    raise ValueError(f"Missing model provenance at {npz_path}")
                if hash_file(npz_path) != meta["output_npz_sha256"]:
                    raise ValueError(f"Reduction output checksum mismatch: {npz_path}")
                with np.load(npz_path, allow_pickle=False) as data:
                    labels = data["label"].astype(str)
                    seq_ids = data["seqID"].astype(str)
                    coords = {"PCA": data["pca"].astype(np.float64), "tSNE": data["tsne"].astype(np.float64)}
                if len(labels) != expected_rows or len(set(labels)) != expected_classes or len(set(seq_ids)) != expected_rows:
                    raise ValueError(f"Label/row count mismatch in {npz_path}")
                label_map = {label: idx for idx, label in enumerate(sorted(set(labels.tolist())))}
                y = np.asarray([label_map[label] for label in labels], dtype=np.int64)
                for method, points in coords.items():
                    if points.shape != (expected_rows, 2) or not np.isfinite(points).all():
                        raise ValueError(f"Invalid {method} coordinates in {npz_path}")
                    row = {
                        "model_name": model, "seed": seed, "method": method,
                        "n_rows": expected_rows, "n_classes": expected_classes,
                        "ch_index": float(calinski_harabasz_score(points, y)),
                        "silhouette": float("nan"), "coordinate_space": "2D projected coordinates",
                        "metric": "euclidean", "dataset_id": dataset, "run_id": gate["run_id"],
                        "dataset_sha256": dataset_hash, "input_sha256": meta["input_sha256"],
                        "model_fingerprint": meta["model_fingerprint"],
                        "model_weight_sha256": meta["model_weight_sha256"],
                        "row_order_sha256": meta["row_order_sha256"],
                        "torch_seed": meta["torch_seed"], "truncation": meta["truncation"],
                        "protocol_id": gate["protocol_id"],
                    }
                    source_rows.append(row)
                    grouped[(model, method)].append(row)
        if len(source_rows) != 30 * 3 * 2:
            raise ValueError(f"Expected 180 source rows for {dataset}; got {len(source_rows)}")
        summary_rows = []
        for model in models:
            for method in METHODS:
                rows = grouped[(model, method)]
                if len(rows) != 3:
                    raise ValueError(f"Expected three seeds for {dataset}/{model}/{method}")
                values = np.asarray([row["ch_index"] for row in rows], dtype=np.float64)
                first = rows[0]
                summary_rows.append({
                    "model_name": model, "method": method, "n_seeds": 3,
                    "n_rows": expected_rows, "n_classes": expected_classes,
                    "ch_index_mean": float(values.mean()), "ch_index_sd": float(values.std(ddof=1)),
                    "silhouette_mean": float("nan"), "silhouette_sd": float("nan"),
                    **{key: first[key] for key in ("dataset_id", "run_id", "dataset_sha256",
                       "input_sha256", "model_fingerprint", "model_weight_sha256",
                       "row_order_sha256", "torch_seed", "truncation", "protocol_id")},
                })
        for method in METHODS:
            ranked = sorted((r for r in summary_rows if r["method"] == method),
                            key=lambda r: r["ch_index_mean"], reverse=True)
            for rank, row in enumerate(ranked, 1):
                rank_rows.append({"dataset_id": dataset, "method": method,
                                  "model_name": row["model_name"], "ch_index_mean": row["ch_index_mean"],
                                  "ch_index_sd": row["ch_index_sd"], "rank": rank, "model_count": 30})
        source_path = tables / f"{table_id}_{dataset}_taxonomy_v2_ch_index_source.csv"
        summary_path = tables / f"{table_id}_{dataset}_taxonomy_v2_ch_index_summary.csv"
        write_csv(source_path, source_rows)
        write_csv(summary_path, summary_rows)
        manifest["tables"][dataset] = {
            "source": {"path": str(source_path), "rows": len(source_rows), "sha256": hash_file(source_path)},
            "summary": {"path": str(summary_path), "rows": len(summary_rows), "sha256": hash_file(summary_path)},
        }
        print(dataset, "source_rows=180", "summary_rows=60", flush=True)
    write_csv(tables / "family_rankings_taxonomy_v2.csv", rank_rows)
    manifest["family_rankings"] = {"path": str(tables / "family_rankings_taxonomy_v2.csv"),
                                  "rows": len(rank_rows), "sha256": hash_file(tables / "family_rankings_taxonomy_v2.csv")}
    (root / "ch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "source_rows_per_dataset": 180,
                      "summary_rows_per_dataset": 60, "family_rank_rows": len(rank_rows)}, indent=2))


if __name__ == "__main__":
    main()
