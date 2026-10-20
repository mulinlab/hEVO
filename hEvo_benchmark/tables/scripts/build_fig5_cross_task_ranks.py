#!/usr/bin/env python3
"""Build the auditable task-level score and rank source used by Fig. 5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TABLES_DIR = Path(__file__).resolve().parents[1]
OUTPUT_NAME = "S49_model_task_level_scores_and_ranks_for_fig5.csv"
GPN_STAR_MODELS = {"GPN-Star-M", "GPN-Star-P", "GPN-Star-V"}
EXPECTED_TASK_COUNT = 45


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(tables_dir: Path, table_id: str) -> tuple[pd.DataFrame, Path]:
    matches = sorted(tables_dir.glob(f"{table_id}_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {table_id} table under {tables_dir}; found {len(matches)}")
    return pd.read_csv(matches[0], float_precision="round_trip"), matches[0]


def task_rows(
    tables_dir: Path,
    *,
    table_id: str,
    task_group: str,
    metric: str,
    score_column: str,
    task_column: str | None = None,
    task_name: str | None = None,
    aggregation: str = "none",
) -> list[dict]:
    frame, source_path = read_table(tables_dir, table_id)
    if table_id == "S26" and {"num_samples", "num_positive", "num_negative"}.issubset(frame.columns):
        cohort = ["num_samples", "num_positive", "num_negative"]
        matched = frame.groupby(cohort, dropna=False).size().idxmax()
        frame = frame.set_index(cohort).loc[[matched]].reset_index()
    if score_column not in frame or "model" not in frame:
        raise ValueError(f"{source_path.name}: missing model or score column {score_column}")

    if task_name is not None:
        frame = frame.copy()
        frame["_task_dataset"] = task_name
    elif task_column is not None:
        if task_column not in frame:
            raise ValueError(f"{source_path.name}: missing task column {task_column}")
        frame = frame.copy()
        frame["_task_dataset"] = frame[task_column].astype(str)
    else:
        raise ValueError("Provide task_column or task_name")

    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame = frame[np.isfinite(frame[score_column])].copy()
    grouped = frame.groupby(["model", "_task_dataset"], sort=True)[score_column].agg(
        score="mean", source_result_row_count="count"
    ).reset_index()
    star_tasks = set(
        grouped.loc[grouped.model.isin(GPN_STAR_MODELS), "_task_dataset"].astype(str)
    )
    grouped = grouped[grouped["_task_dataset"].astype(str).isin(star_tasks)]
    rows = []
    for row in grouped.to_dict("records"):
        dataset = str(row["_task_dataset"])
        rows.append(
            {
                "table_id": "S49",
                "task_group": task_group,
                "task_unit": f"{table_id}:{dataset}",
                "source_table": source_path.name,
                "source_sha256": sha256(source_path),
                "source_dataset": dataset,
                "score_metric": metric,
                "score_aggregation": aggregation,
                "rank_direction": "higher_score_better",
                "model": str(row["model"]),
                "score": float(row["score"]),
                "source_result_row_count": int(row["source_result_row_count"]),
            }
        )
    return rows


def build(tables_dir: Path, output_path: Path | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    # Each S17/S18 row is one model's score on one named benchmark task.
    rows.extend(task_rows(tables_dir, table_id="S17", task_group="NT", metric="MCC", score_column="mcc_mean", task_column="dataset"))
    rows.extend(task_rows(tables_dir, table_id="S18", task_group="Genomic Benchmark", metric="MCC", score_column="mcc_mean", task_column="dataset"))
    rows.extend(task_rows(tables_dir, table_id="S19", task_group="URE binary", metric="MCC", score_column="mcc_mean", task_column="benchmark_raw"))
    rows.extend(task_rows(tables_dir, table_id="S20", task_group="URE multiclass", metric="macro F1", score_column="per_class_f1", task_column="dataset", aggregation="arithmetic mean across class-specific F1 rows"))

    for table_id, task_group in [("S23", "Variant"), ("S26", "Variant"), ("S27", "Variant"), ("S28", "Variant"), ("S29", "Variant"), ("S30", "Variant")]:
        rows.extend(task_rows(tables_dir, table_id=table_id, task_group=task_group, metric="AUPRC", score_column="auprc_mean", task_column="dataset"))
    rows.extend(task_rows(tables_dir, table_id="S31", task_group="Variant", metric="paired top-1 accuracy", score_column="paired_top1", task_column="dataset"))

    rows.extend(task_rows(tables_dir, table_id="S33", task_group="Ancestry", metric="rank_score", score_column="rank_score", task_name="1000G ancestry unsupervised"))
    rows.extend(task_rows(tables_dir, table_id="S34", task_group="Ancestry", metric="MCC", score_column="classification_mcc", task_name="1000G ancestry supervised"))
    rows.extend(task_rows(tables_dir, table_id="S35", task_group="Ancestry", metric="Pearson r", score_column="pearson_r_mean", task_name="1000G ancestry proportion regression", aggregation="arithmetic mean across shot levels"))
    rows.extend(task_rows(tables_dir, table_id="S36", task_group="HAQER", metric="MCC", score_column="mcc_mean", task_column="benchmark", aggregation="arithmetic mean across five negative-control runs"))
    rows.extend(task_rows(tables_dir, table_id="S37", task_group="HAR", metric="MCC", score_column="mcc_mean", task_column="benchmark", aggregation="arithmetic mean across five negative-control runs"))

    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["task_unit", "model"]).any():
        raise ValueError("S49 has no score rows or duplicate model/task cells")
    task_count = frame.task_unit.nunique()
    if task_count != EXPECTED_TASK_COUNT:
        counts = frame.groupby("source_table").task_unit.nunique().to_dict()
        raise ValueError(f"Expected {EXPECTED_TASK_COUNT} normalized task units, found {task_count}: {counts}")

    frame["task_rank"] = frame.groupby("task_unit", sort=False)["score"].rank(method="min", ascending=False)
    frame["n_models_ranked"] = frame.groupby("task_unit", sort=False)["model"].transform("size").astype(int)
    frame["rank_method"] = "minimum rank for ties; higher score ranks better"
    frame = frame.sort_values(["task_group", "task_unit", "task_rank", "model"], kind="stable").reset_index(drop=True)

    star = frame[frame.model.isin(GPN_STAR_MODELS)].groupby("model").task_unit.nunique()
    missing_star = {model: int(star.get(model, 0)) for model in sorted(GPN_STAR_MODELS) if int(star.get(model, 0)) != EXPECTED_TASK_COUNT}
    if missing_star:
        raise ValueError(f"GPN-Star rank coverage must be {EXPECTED_TASK_COUNT} tasks each: {missing_star}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False, na_rep="NA")
    report = {
        "status": "PASS",
        "table_id": "S49",
        "output": str(output_path) if output_path else None,
        "task_units": int(task_count),
        "task_rows": int(len(frame)),
        "models": int(frame.model.nunique()),
        "rank_coverage_by_model": frame.groupby("model").task_unit.nunique().sort_values().to_dict(),
        "gpn_star_coverage": {model: int(star[model]) for model in sorted(GPN_STAR_MODELS)},
        "missing_scores_are_not_imputed": True,
        "aggregation_notes": frame.groupby("task_group").score_aggregation.apply(lambda values: "; ".join(sorted(set(values)))).to_dict(),
    }
    print(json.dumps(report, indent=2))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    build(args.tables_dir, args.output or args.tables_dir / OUTPUT_NAME)


if __name__ == "__main__":
    main()
