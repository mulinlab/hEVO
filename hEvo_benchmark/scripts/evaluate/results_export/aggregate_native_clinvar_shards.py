#!/usr/bin/env python3
"""Validate and aggregate native ClinVar score shards into full-set metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm

try:
    from .common import sha256 as _sha256
except ImportError:  # direct script execution
    from common import sha256 as _sha256


def sha256_file(path: Path) -> str:
    return _sha256(path)


def bootstrap_metrics(
    labels: np.ndarray, scores: np.ndarray, n_bootstrap: int, seed: int
):
    if n_bootstrap == 0:
        return {}
    rng = np.random.RandomState(seed)
    ap_values: list[float] = []
    auc_values: list[float] = []
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap metrics"):
        indices = rng.randint(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        sampled_scores = scores[indices]
        if sampled_labels.min() == sampled_labels.max():
            continue
        ap_values.append(float(average_precision_score(sampled_labels, sampled_scores)))
        auc_values.append(float(roc_auc_score(sampled_labels, sampled_scores)))

    output = {}
    for name, values in (("auprc", ap_values), ("auroc", auc_values)):
        array = np.asarray(values, dtype=float)
        output[f"{name}_bootstrap_mean"] = float(array.mean())
        output[f"{name}_ci_lower"] = float(np.percentile(array, 2.5))
        output[f"{name}_ci_upper"] = float(np.percentile(array, 97.5))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data", required=True, help="Original unsplit ClinVar CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    args = parser.parse_args()
    if args.bootstrap < 0 or len(args.scores) != len(args.manifests):
        parser.error(
            "bootstrap must be nonnegative and each score shard needs one manifest"
        )

    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{args.model_name}.scores.tsv"
    metrics_path = output_dir / f"{args.model_name}.metrics.json"
    manifest_path = output_dir / f"{args.model_name}.manifest.json"
    if any(path.exists() for path in (combined_path, metrics_path, manifest_path)):
        raise FileExistsError(f"Aggregate output already exists for {args.model_name}")

    source = pd.read_csv(
        data_path, usecols=["SeqID", "label"], dtype={"SeqID": "string"}
    )
    source_ids = source["SeqID"].astype(str).tolist()
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Source ClinVar CSV has duplicate SeqID values")
    source_labels = dict(zip(source_ids, source["label"].astype(int).tolist()))
    source_order = {seq_id: index for index, seq_id in enumerate(source_ids)}

    shard_manifests = [Path(value).resolve() for value in args.manifests]
    manifest_objects = [
        json.loads(path.read_text(encoding="utf-8")) for path in shard_manifests
    ]
    for manifest in manifest_objects:
        if manifest.get("family") != "generator" or manifest.get("context_bp") != 1024:
            raise ValueError(
                "All shards must be GENERATOR runs with a 1,024 bp context"
            )
        if Path(manifest["data_path"]).resolve().is_file() is False:
            raise FileNotFoundError(manifest["data_path"])
        if manifest.get("model_path") != manifest_objects[0].get("model_path"):
            raise ValueError("Shard manifests use different model paths")

    rows_by_id: dict[str, dict[str, str]] = {}
    score_paths = [Path(value).resolve() for value in args.scores]
    for score_path in score_paths:
        with score_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "SeqID" not in reader.fieldnames:
                raise ValueError(f"Invalid score shard header: {score_path}")
            fields = reader.fieldnames
            for row in reader:
                seq_id = row["SeqID"]
                if seq_id in rows_by_id:
                    raise ValueError(f"Duplicate SeqID across score shards: {seq_id}")
                if seq_id not in source_labels:
                    raise ValueError(f"Unexpected SeqID in score shard: {seq_id}")
                if int(row["label"]) != source_labels[seq_id]:
                    raise ValueError(f"Label mismatch for SeqID {seq_id}")
                context_bp = int(row["context_bp"])
                score = float(row["score"])
                p_ref, p_alt = float(row["p_ref"]), float(row["p_alt"])
                if not 0 < context_bp <= 1024 or context_bp % 6:
                    raise ValueError(
                        f"Out-of-protocol context length for SeqID {seq_id}"
                    )
                if not all(math.isfinite(value) for value in (score, p_ref, p_alt)):
                    raise ValueError(f"Non-finite score/probability for SeqID {seq_id}")
                if p_ref <= 0 or p_alt < 0:
                    raise ValueError(f"Invalid base probability for SeqID {seq_id}")
                expected_score = math.log(p_ref) - math.log(p_alt + 1e-10)
                if not math.isclose(score, expected_score, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(
                        f"Score does not match log(P(ref)/P(alt)) for SeqID {seq_id}"
                    )
                rows_by_id[seq_id] = row

    missing_ids = set(source_ids) - set(rows_by_id)
    if missing_ids:
        raise ValueError(f"Score shards omit {len(missing_ids)} source variants")
    rows = sorted(rows_by_id.values(), key=lambda row: source_order[row["SeqID"]])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)

    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    precision, recall, _ = precision_recall_curve(labels, scores)
    metrics = {
        "model_name": args.model_name,
        "family": "generator",
        "num_samples": int(len(labels)),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "skip_counts": {},
        "rows_read": int(len(source_ids)),
        "score_file": str(combined_path),
    }
    metrics.update(bootstrap_metrics(labels, scores, args.bootstrap, args.seed))
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "model_path": manifest_objects[0]["model_path"],
        "model_name": args.model_name,
        "family": "generator",
        "dtype": manifest_objects[0].get("dtype"),
        "benchmark_max_input_bp": 1024,
        "context_bp": 1024,
        "scoring_protocol": "left_context_next_token_base_marginal_log_p_ref_over_p_alt",
        "aggregation": "source_ordered_exact_cover_of_all_score_shards",
        "shards": [
            {
                "scores_file": str(score_path),
                "scores_sha256": sha256_file(score_path),
                "shard_manifest": str(shard_manifest),
                "shard_manifest_sha256": sha256_file(shard_manifest),
                "shard_data_path": shard_manifest_obj["data_path"],
                "shard_data_sha256": shard_manifest_obj["data_sha256"],
            }
            for score_path, shard_manifest, shard_manifest_obj in zip(
                score_paths, shard_manifests, manifest_objects
            )
        ],
        "num_samples_scored": int(len(rows)),
        "num_samples_positive": int(labels.sum()),
        "num_samples_negative": int(len(labels) - labels.sum()),
        "metrics_file": str(metrics_path),
        "scores_file": str(combined_path),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Scores: {combined_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
