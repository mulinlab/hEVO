#!/usr/bin/env python3
"""Append event-centric Fig.3 i--k results to the canonical S32 source table.

The input layout is the public event-centric benchmark layout: one task TSV per
cutoff year and seed, plus one ``seedXX.scores.tsv`` file per model.  The
bootstrap follows the S32 convention used by the benchmark release: resample
five seeds and variants within each selected seed (20 replicates, seed
20260728), then report the observed five-seed mean and percentile intervals.
Existing model rows are retained unless their labels are explicitly selected
with ``--replace``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = {
    "blb_stable_vs_upgraded": ("label", -1.0),
    "plp_stable_vs_downgraded": ("inverse", 1.0),
    "vus_resolved_path_vs_benign": ("label", -1.0),
}
YEARS = tuple(range(2017, 2026))
TABLE_COLUMNS = [
    "model_label", "task_name", "cutoff_year", "top_fraction", "n_boot",
    "seed_count", "precision_at_k_mean", "precision_at_k_ci_low",
    "precision_at_k_ci_high", "enrichment_vs_baseline_mean",
    "enrichment_vs_baseline_ci_low", "enrichment_vs_baseline_ci_high",
]


def _metrics(labels: np.ndarray, scores: np.ndarray, top_fraction: float) -> tuple[float, float]:
    event_rate = float(labels.mean())
    top_n = max(1, int(math.ceil(len(labels) * top_fraction)))
    precision = float(labels[np.argsort(-scores, kind="stable")[:top_n]].mean())
    enrichment = precision / event_rate if event_rate else float("nan")
    return precision, enrichment


def _bootstrap(
    seed_arrays: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_boot: int,
    rng_seed: int,
    top_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    precision: list[float] = []
    enrichment: list[float] = []
    n_seeds = len(seed_arrays)
    for _ in range(n_boot):
        labels_parts: list[np.ndarray] = []
        scores_parts: list[np.ndarray] = []
        for seed_index in rng.integers(0, n_seeds, size=n_seeds):
            labels, scores = seed_arrays[int(seed_index)]
            indices = rng.integers(0, len(labels), size=len(labels))
            labels_parts.append(labels[indices])
            scores_parts.append(scores[indices])
        p, e = _metrics(np.concatenate(labels_parts), np.concatenate(scores_parts), top_fraction)
        precision.append(p)
        enrichment.append(e)
    return np.asarray(precision), np.asarray(enrichment)


def _load_arrays(
    task_root: Path,
    scores_root: Path,
    model_id: str,
    seed: int,
    year: int,
    task: str,
    score_cache: dict[tuple[str, int], pd.DataFrame],
):
    task_path = task_root / f"seed{seed:02d}" / "tasks" / f"{year}_{task}.tsv"
    score_path = scores_root / model_id / f"seed{seed:02d}.scores.tsv"
    task_frame = pd.read_csv(task_path, sep="\t", usecols=["variant_id", "label"])
    cache_key = (model_id, seed)
    score_frame = score_cache.get(cache_key)
    if score_frame is None:
        score_frame = pd.read_csv(score_path, sep="\t", usecols=["variant_id", "raw_llr"])
        score_cache[cache_key] = score_frame
    if score_frame["variant_id"].duplicated().any():
        raise ValueError(f"duplicate variant_id in {score_path}")
    merged = task_frame.merge(score_frame, on="variant_id", how="left", validate="one_to_one")
    if merged["raw_llr"].isna().any():
        raise ValueError(f"missing scores for {task_path}: {int(merged['raw_llr'].isna().sum())}")
    mode, sign = TASKS[task]
    labels = merged["label"].to_numpy(dtype=np.int8)
    if mode == "inverse":
        labels = 1 - labels
    scores = merged["raw_llr"].to_numpy(dtype=float) * sign
    return labels, scores


def build_rows(
    task_root: Path,
    scores_root: Path,
    model_id: str,
    *,
    top_fraction: float,
    n_boot: int,
    rng_seed: int,
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    score_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for task in TASKS:
        for year in YEARS:
            arrays = [_load_arrays(task_root, scores_root, model_id, seed, year, task, score_cache) for seed in seeds]
            observed = [_metrics(labels, scores, top_fraction) for labels, scores in arrays]
            boot_precision, boot_enrichment = _bootstrap(
                arrays,
                n_boot=n_boot,
                rng_seed=rng_seed + year + sum(ord(char) for char in f"{model_id}|{task}"),
                top_fraction=top_fraction,
            )
            rows.append({
                "model_label": model_id,
                "task_name": task,
                "cutoff_year": year,
                "top_fraction": top_fraction,
                "n_boot": n_boot,
                "seed_count": len(arrays),
                "precision_at_k_mean": float(np.mean([item[0] for item in observed])),
                "precision_at_k_ci_low": float(np.quantile(boot_precision, 0.025)),
                "precision_at_k_ci_high": float(np.quantile(boot_precision, 0.975)),
                "enrichment_vs_baseline_mean": float(np.mean([item[1] for item in observed])),
                "enrichment_vs_baseline_ci_low": float(np.quantile(boot_enrichment, 0.025)),
                "enrichment_vs_baseline_ci_high": float(np.quantile(boot_enrichment, 0.975)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--model-id", action="append", required=True,
                        help="Model directory/label; may be repeated.")
    parser.add_argument("--replace", action="store_true",
                        help="Replace rows for selected model labels instead of refusing duplicates.")
    parser.add_argument("--n-boot", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=20260728)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    if not 0 < args.top_fraction < 1 or args.n_boot < 2:
        raise ValueError("top fraction must be in (0,1) and n_boot must be >= 2")
    table = pd.read_csv(args.table)
    missing = set(TABLE_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{args.table} is missing columns: {sorted(missing)}")
    selected = set(args.model_id)
    present = selected.intersection(set(table["model_label"].astype(str)))
    if present and not args.replace:
        raise ValueError(f"rows already exist for {sorted(present)}; use --replace to update them")
    if present:
        table = table.loc[~table["model_label"].isin(present)].copy()
    new_rows: list[dict[str, object]] = []
    for model_id in args.model_id:
        new_rows.extend(build_rows(
            args.task_root, args.scores_root, model_id,
            top_fraction=args.top_fraction, n_boot=args.n_boot,
            rng_seed=args.rng_seed, seeds=tuple(args.seeds),
        ))
    output = pd.concat([table[TABLE_COLUMNS], pd.DataFrame(new_rows, columns=TABLE_COLUMNS)], ignore_index=True)
    if output.duplicated(["model_label", "task_name", "cutoff_year"]).any():
        raise ValueError("duplicate model/task/year keys after update")
    output = output.sort_values(["task_name", "cutoff_year", "model_label"], kind="stable").reset_index(drop=True)
    output.to_csv(args.table, index=False, float_format="%.15g")
    print(f"wrote {len(output)} rows to {args.table}")


if __name__ == "__main__":
    main()
