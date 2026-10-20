#!/usr/bin/env python3
"""Recompute CH from every saved 2D coordinate and summarize rank stability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import calinski_harabasz_score

SEEDS = (42, 43, 44)
METHODS = ("PCA", "tSNE", "UMAP")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_spearman(a, b) -> float:
    score = float(spearmanr(a, b).statistic)
    if not np.isfinite(score):
        raise ValueError("non-finite Spearman rank correlation")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    roster = json.loads((root / "config" / "model_roster.json").read_text(encoding="utf-8"))
    if roster.get("status") != "PASS" or len(roster.get("models", [])) != 30:
        raise ValueError("frozen model roster is invalid")
    dataset_id, run_id = roster["dataset_id"], roster["run_id"]
    with Path(roster["dataset_csv"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.asarray([row["label"] for row in rows], dtype=str)
    seq_ids = np.asarray([row["seqID"] for row in rows], dtype=str)
    label_map = {name: i for i, name in enumerate(sorted(set(labels.tolist())))}
    y = np.asarray([label_map[value] for value in labels], dtype=np.int64)
    if len(rows) != 5000 or len(label_map) != 5:
        raise ValueError("the frozen UCSC dataset is not 5,000 rows × 5 classes")

    source_rows: list[dict] = []
    scores: dict[tuple[str, str, int], float] = {}
    coordinate_meta: dict[tuple[str, str, int], tuple[Path, str, str]] = {}
    for model in roster["models"]:
        h5_path = root / "embeddings" / dataset_id / f"{model}.h5"
        h5_meta_path = h5_path.with_suffix(".manifest.json")
        if not h5_path.is_file() or not h5_meta_path.is_file():
            raise FileNotFoundError(f"missing model embedding: {model}")
        h5_manifest = json.loads(h5_meta_path.read_text(encoding="utf-8"))
        if h5_manifest.get("input_sha256") != roster["dataset_sha256"]:
            raise ValueError(f"embedding dataset hash mismatch for {model}")
        h5_hash = sha256_file(h5_path)
        for seed in SEEDS:
            candidates = {
                "PCA": root / "reductions" / dataset_id / model / f"seed{seed}.npz",
                "tSNE": root / "reductions" / dataset_id / model / f"seed{seed}.npz",
                "UMAP": root / "reductions" / dataset_id / model / f"umap_seed{seed}.npz",
            }
            cache = {}
            for method, coordinate_path in candidates.items():
                if not coordinate_path.is_file():
                    raise FileNotFoundError(f"missing {method} coordinates: {model} seed {seed}")
                with np.load(coordinate_path, allow_pickle=False) as data:
                    if not np.array_equal(data["seqID"].astype(str), seq_ids) or not np.array_equal(data["label"].astype(str), labels):
                        raise ValueError(f"coordinate row order/labels mismatch: {model}/{method}/{seed}")
                    coord_key = method.lower()
                    coords = np.asarray(data[coord_key], dtype=np.float64)
                if coords.shape != (5000, 2) or not np.isfinite(coords).all():
                    raise ValueError(f"invalid coordinates: {model}/{method}/{seed}")
                score = float(calinski_harabasz_score(coords, y))
                if not np.isfinite(score):
                    raise ValueError(f"non-finite CH score: {model}/{method}/{seed}")
                key = (model, method, seed)
                scores[key] = score
                coordinate_meta[key] = (coordinate_path, sha256_file(coordinate_path), h5_hash)
                source_rows.append({
                    "model_name": model,
                    "dataset_id": dataset_id,
                    "method": method,
                    "seed": seed,
                    "n_rows": len(rows),
                    "n_classes": len(label_map),
                    "ch_index": score,
                    "coordinate_file": str(coordinate_path),
                    "coordinate_sha256": coordinate_meta[key][1],
                    "embedding_file": str(h5_path),
                    "embedding_sha256": h5_hash,
                    "dataset_sha256": roster["dataset_sha256"],
                    "protocol_id": roster["protocol_id"],
                    "run_id": run_id,
                })
    expected_results = len(roster["models"]) * len(METHODS) * len(SEEDS)
    if len(source_rows) != expected_results or len(scores) != expected_results:
        raise ValueError(f"expected {expected_results} model×method×seed results, found {len(source_rows)}")

    seed_ranks = {}
    summary_rows = []
    for method in METHODS:
        for seed in SEEDS:
            ordered = sorted(roster["models"], key=lambda m: (-scores[(m, method, seed)], m))
            for rank, model in enumerate(ordered, 1):
                seed_ranks[(method, seed, model)] = rank
    for method in METHODS:
        ch_means = {model: float(np.mean([scores[(model, method, seed)] for seed in SEEDS])) for model in roster["models"]}
        rank_by_mean = {model: int(rank) for model, rank in zip(sorted(roster["models"], key=lambda m: (-ch_means[m], m)), range(1, 31))}
        top5_by_seed = {seed: set(sorted(roster["models"], key=lambda m: (-scores[(m, method, seed)], m))[:5]) for seed in SEEDS}
        pair_spearman = [safe_spearman([seed_ranks[(method, a, m)] for m in roster["models"]], [seed_ranks[(method, b, m)] for m in roster["models"]]) for a, b in combinations(SEEDS, 2)]
        pair_top5 = [len(top5_by_seed[a] & top5_by_seed[b]) for a, b in combinations(SEEDS, 2)]
        for model in roster["models"]:
            values = np.asarray([scores[(model, method, seed)] for seed in SEEDS], dtype=np.float64)
            seed_mean_rank = float(np.mean([seed_ranks[(method, seed, model)] for seed in SEEDS]))
            summary_rows.append({
                "model_name": model,
                "dataset_id": dataset_id,
                "method": method,
                "n_seeds": len(SEEDS),
                "n_rows": 5000,
                "n_classes": 5,
                "ch_index_mean": float(values.mean()),
                "ch_index_sd": float(values.std(ddof=1)),
                "mean_rank_over_seeds": seed_mean_rank,
                "rank_by_mean_ch": rank_by_mean[model],
                "seed_rank_spearman_mean": float(np.mean(pair_spearman)),
                "seed_top5_overlap_count_mean": float(np.mean(pair_top5)),
                "seed_top5_overlap_fraction_mean": float(np.mean(pair_top5) / 5.0),
                "dataset_sha256": roster["dataset_sha256"],
                "protocol_id": roster["protocol_id"],
                "run_id": run_id,
            })

    stability_rows = []
    for method in METHODS:
        for first, second in combinations(SEEDS, 2):
            rank_a = [seed_ranks[(method, first, model)] for model in roster["models"]]
            rank_b = [seed_ranks[(method, second, model)] for model in roster["models"]]
            top_a = {model for model in roster["models"] if seed_ranks[(method, first, model)] <= 5}
            top_b = {model for model in roster["models"] if seed_ranks[(method, second, model)] <= 5}
            stability_rows.append({
                "comparison_type": "within_method_seed_pair", "method_a": method, "method_b": method,
                "seed_a": first, "seed_b": second, "spearman_rank_correlation": safe_spearman(rank_a, rank_b),
                "top5_overlap_count": len(top_a & top_b), "top5_overlap_fraction": len(top_a & top_b) / 5.0,
                "dataset_id": dataset_id, "dataset_sha256": roster["dataset_sha256"], "run_id": run_id,
            })
    for method_a, method_b in combinations(METHODS, 2):
        rank_a = [next(row["rank_by_mean_ch"] for row in summary_rows if row["model_name"] == model and row["method"] == method_a) for model in roster["models"]]
        rank_b = [next(row["rank_by_mean_ch"] for row in summary_rows if row["model_name"] == model and row["method"] == method_b) for model in roster["models"]]
        top_a = {model for model in roster["models"] if next(row["rank_by_mean_ch"] for row in summary_rows if row["model_name"] == model and row["method"] == method_a) <= 5}
        top_b = {model for model in roster["models"] if next(row["rank_by_mean_ch"] for row in summary_rows if row["model_name"] == model and row["method"] == method_b) <= 5}
        stability_rows.append({
            "comparison_type": "between_method_mean_rank", "method_a": method_a, "method_b": method_b,
            "seed_a": "mean", "seed_b": "mean", "spearman_rank_correlation": safe_spearman(rank_a, rank_b),
            "top5_overlap_count": len(top_a & top_b), "top5_overlap_fraction": len(top_a & top_b) / 5.0,
            "dataset_id": dataset_id, "dataset_sha256": roster["dataset_sha256"], "run_id": run_id,
        })

    table_dir = root / "tables_candidate"
    source_path = table_dir / "ucsc_refseq_curated_hg38_ch_index_source.csv"
    summary_path = table_dir / "ucsc_refseq_curated_hg38_ch_index_summary.csv"
    stability_path = table_dir / "ucsc_refseq_curated_hg38_rank_stability.csv"
    write_csv(source_path, source_rows, list(source_rows[0]))
    write_csv(summary_path, summary_rows, list(summary_rows[0]))
    write_csv(stability_path, stability_rows, list(stability_rows[0]))

    final = {
        "status": "PASS",
        "run_id": run_id,
        "dataset_id": dataset_id,
        "dataset_sha256": roster["dataset_sha256"],
        "protocol_id": roster["protocol_id"],
        "models": len(roster["models"]),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "expected_coordinate_results": expected_results,
        "source_rows": len(source_rows),
        "summary_rows": len(summary_rows),
        "rank_stability_rows": len(stability_rows),
        "source_csv": str(source_path),
        "source_csv_sha256": sha256_file(source_path),
        "summary_csv": str(summary_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "rank_stability_csv": str(stability_path),
        "rank_stability_csv_sha256": sha256_file(stability_path),
        "ch_recomputed_from_saved_coordinates": True,
        "tsne_parameters": {"perplexity": 100, "learning_rate": "auto", "init": "pca", "max_iter": 5000},
        "umap_parameters": {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "init": "spectral", "n_epochs": 500, "learning_rate": 1, "spread": 1, "n_jobs": 1},
        "raw_ch_cross_method_interpretation": "Do not compare raw CH values across reducers; use method-specific ranks, rank correlation, and top-5 overlap.",
    }
    result_path = root / "qa" / "evaluation_qa.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
