#!/usr/bin/env python3
"""Extract ClinVar coding/noncoding metrics from an existing all-cache.

The ClinVar ``all`` cache was generated from coding rows followed by
noncoding rows and then shuffled.  The generated ``all_N`` identifiers retain
the pre-shuffle row number, so this script can select the two S29/S30 subsets
without running the model again.  It validates every input shard before
computing point estimates and the benchmark's 1,000-replicate row bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL_NAME = "hEvo-large"
DISPLAY_MODEL = "hEvo-large"
TOTAL_SHARDS = 12
DEFAULT_BOOTSTRAP = 1000
DEFAULT_BOOTSTRAP_SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def provenance_path(path: Path, repo_root: Path) -> str:
    """Keep tracked provenance portable and free of machine-specific paths."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return f"<external-run-root>/{path.name}"


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def parse_seq_id(value: str) -> int:
    prefix, number = value.split("_", 1)
    if prefix != "all":
        raise ValueError(f"unexpected all-cache SeqID: {value}")
    return int(number)


def load_all_cache(
    all_source_dir: Path,
    h5_root: Path,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    llrs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    seq_numbers: list[np.ndarray] = []
    shard_records: list[dict[str, Any]] = []

    for shard in range(TOTAL_SHARDS):
        source = all_source_dir / f"clinvar_all_1023bp.shard-{shard:03d}.csv"
        h5_path = (
            h5_root / f"ClinVar260615_all_shard_{shard:03d}" / f"{model_name}.hdf5"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)

        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        source_ids = [parse_seq_id(row["SeqID"]) for row in rows]
        source_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)

        with h5py.File(h5_path, "r") as h5:
            required = {"details", "labels", "llrs"}
            missing = required.difference(h5.keys())
            if missing:
                raise ValueError(f"{h5_path}: missing datasets {sorted(missing)}")
            if h5.attrs.get("model_name") != model_name:
                raise ValueError(f"{h5_path}: wrong model_name")
            if h5.attrs.get("score_mode") != "neg_llr":
                raise ValueError(f"{h5_path}: expected score_mode=neg_llr")
            h5_labels = np.asarray(h5["labels"][:], dtype=np.int32)
            h5_llrs = np.asarray(h5["llrs"][:], dtype=np.float64)
            detail_ids = np.asarray(
                [
                    parse_seq_id(json.loads(item.decode("utf-8"))["idx"])
                    for item in h5["details"][:]
                ],
                dtype=np.int64,
            )
            detail_labels = np.asarray(
                [
                    int(json.loads(item.decode("utf-8"))["label"])
                    for item in h5["details"][:]
                ],
                dtype=np.int32,
            )

        if len(rows) != len(h5_llrs):
            raise ValueError(f"{source}: {len(rows)} rows != {len(h5_llrs)} H5 scores")
        if not np.array_equal(detail_ids, source_ids):
            raise ValueError(f"{h5_path}: details IDs do not match source shard order")
        if not np.array_equal(h5_labels, source_labels):
            raise ValueError(f"{h5_path}: H5 labels do not match source shard")
        if not np.array_equal(detail_labels, h5_labels):
            raise ValueError(f"{h5_path}: details labels do not match H5 labels")
        if not np.isfinite(h5_llrs).all():
            raise ValueError(f"{h5_path}: non-finite LLR")

        llrs.append(h5_llrs)
        labels.append(h5_labels)
        seq_numbers.append(detail_ids)
        shard_records.append(
            {
                "shard": shard,
                "source_path": str(source),
                "source_sha256": sha256(source),
                "h5_path": str(h5_path),
                "h5_sha256": sha256(h5_path),
                "rows": len(rows),
            }
        )

    return (
        np.concatenate(llrs),
        np.concatenate(labels),
        np.concatenate(seq_numbers),
        shard_records,
    )


def bootstrap_ci(
    llrs: np.ndarray,
    labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    """Match zeroshot_paried_seq_input.py's row-IID percentile bootstrap."""
    rng = np.random.RandomState(seed)
    values: list[float] = []
    scores = -llrs
    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(labels), size=len(labels))
        try:
            values.append(float(metric_fn(labels[indices], scores[indices])))
        except ValueError:
            continue
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("bootstrap produced no finite metric values")
    return (
        float(array.mean()),
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    )


def metric_record(
    name: str,
    llrs: np.ndarray,
    labels: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    scores = -llrs
    ap = float(average_precision_score(labels, scores))
    auroc = float(roc_auc_score(labels, scores))
    auroc_boot = bootstrap_ci(llrs, labels, roc_auc_score, bootstrap, seed)
    auprc_boot = bootstrap_ci(llrs, labels, average_precision_score, bootstrap, seed)
    return {
        "table": "S29" if name == "coding" else "S30",
        "dataset": name,
        "model": DISPLAY_MODEL,
        "source_model": MODEL_NAME,
        "score_mode": "neg_llr",
        "num_samples": int(labels.size),
        "num_positive": int(labels.sum()),
        "num_negative": int(labels.size - labels.sum()),
        "baseline": float(labels.mean()),
        "auprc_mean": ap,
        "auprc_bootstrap_mean": auprc_boot[0],
        "auprc_ci_lower": auprc_boot[1],
        "auprc_ci_upper": auprc_boot[2],
        "auroc_mean": auroc,
        "auroc_bootstrap_mean": auroc_boot[0],
        "auroc_ci_lower": auroc_boot[1],
        "auroc_ci_upper": auroc_boot[2],
        "bootstrap_n": bootstrap,
        "bootstrap_seed": seed,
        "bootstrap_method": "nonparametric_percentile",
        "bootstrap_resampling_unit": "row_iid_with_replacement",
    }


def replace_table_row(path: Path, record: dict[str, Any]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing table header: {path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    matches = [row for row in rows if row.get("model") == DISPLAY_MODEL]
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected exactly one {DISPLAY_MODEL} row, found {len(matches)}"
        )
    target = matches[0]
    expected_counts = {
        "num_samples": str(record["num_samples"]),
        "num_positive": str(record["num_positive"]),
        "num_negative": str(record["num_negative"]),
    }
    for field, value in expected_counts.items():
        if target[field] != value:
            raise ValueError(f"{path}: {field} changed from the source contract")
    target.update(
        {
            "auprc_mean": repr(record["auprc_mean"]),
            "auprc_ci_lower": repr(record["auprc_ci_lower"]),
            "auprc_ci_upper": repr(record["auprc_ci_upper"]),
            "auroc_mean": repr(record["auroc_mean"]),
            "auroc_ci_lower": repr(record["auroc_ci_lower"]),
            "auroc_ci_upper": repr(record["auroc_ci_upper"]),
        }
    )
    _output: list[str] = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def update_provenance(
    path: Path,
    metrics_path: Path,
    records: list[dict[str, Any]],
    shard_records: list[dict[str, Any]],
    coding_count: int,
    noncoding_count: int,
    all_source: Path,
    all_summary: Path,
    repo_root: Path,
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    old_results = payload.get("source_results", [])
    payload["source_results"] = [
        item for item in old_results if item.get("table") not in {"S29", "S30"}
    ]
    now = datetime.now(timezone.utc).isoformat()
    common = {
        "dataset_name": "ClinVar260615_all_1023bp",
        "finished_at_utc": now,
        "manifest_path": "<runs-root>/hEvo-large/ClinVar260615/summary_results.json",
        "manifest_sha256": sha256(all_summary),
        "manifest_status": "SUMMARY_COMPLETED",
        "result_path": provenance_path(metrics_path, repo_root),
        "result_sha256": sha256(metrics_path),
        "run_status": "COMPLETED",
        "task": "ClinVar260615_all_subselected",
        "protocol": {
            "score_mode": "neg_llr",
            "bootstrap_replicates": records[0]["bootstrap_n"],
            "bootstrap_seed": records[0]["bootstrap_seed"],
            "bootstrap_method": records[0]["bootstrap_method"],
            "bootstrap_resampling_unit": records[0]["bootstrap_resampling_unit"],
            "source": "existing all-cache; no model re-evaluation",
        },
        "source_all": {
            "csv": provenance_path(all_source, repo_root),
            "csv_sha256": sha256(all_source),
            "summary_json": provenance_path(all_summary, repo_root),
            "summary_json_sha256": sha256(all_summary),
            "h5_shards": [
                {
                    **item,
                    "source_path": provenance_path(
                        Path(item["source_path"]), repo_root
                    ),
                    "h5_path": provenance_path(Path(item["h5_path"]), repo_root),
                }
                for item in shard_records
            ],
        },
    }
    for record in records:
        _cutoff = coding_count if record["dataset"] == "coding" else coding_count
        entry = dict(common)
        entry.update(
            {
                "table": record["table"],
                "source_model": MODEL_NAME,
                "result_json_metadata": {
                    "subset_metrics_path": provenance_path(metrics_path, repo_root),
                    "subset_metrics_sha256": sha256(metrics_path),
                    "subset": record["dataset"],
                    "seqid_numeric_rule": (
                        f"all_1..all_{coding_count}"
                        if record["dataset"] == "coding"
                        else f"all_{coding_count + 1}..all_{coding_count + noncoding_count}"
                    ),
                    "validated_counts": {
                        "coding_rows": coding_count,
                        "noncoding_rows": noncoding_count,
                    },
                    "metrics": record,
                },
                "split": {
                    "mode": "all_cache_seqid_subselection",
                    "criterion": f"all_1..all_{coding_count}"
                    if record["dataset"] == "coding"
                    else f"all_{coding_count + 1}..all_{coding_count + noncoding_count}",
                    "rows": record["num_samples"],
                },
            }
        )
        payload["source_results"].append(entry)
    payload["generated_utc"] = now
    atomic_write_json(path, payload)


def main() -> None:
    # Keep the historical hEvo-large CLI intact; explicit multi-model inputs
    # use the identity-validated reconstruction workflow.
    import sys

    if any(
        arg == "--input-manifest" or arg.startswith("--input-manifest=")
        for arg in sys.argv[1:]
    ):
        from rebuild_gpn_subsets import main as rebuild_main

        rebuild_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-source-dir", type=Path, required=True)
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument("--all-summary", type=Path, required=True)
    parser.add_argument("--coding-csv", type=Path, required=True)
    parser.add_argument("--noncoding-csv", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--apply", action="store_true", help="replace the two hEvo-large table rows"
    )
    args = parser.parse_args()

    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    coding_count = count_csv_rows(args.coding_csv)
    noncoding_count = count_csv_rows(args.noncoding_csv)
    llrs, labels, seq_numbers, shard_records = load_all_cache(
        args.all_source_dir, args.h5_root, MODEL_NAME
    )
    total = coding_count + noncoding_count
    if len(llrs) != total:
        raise ValueError(f"all-cache rows {len(llrs)} != coding+noncoding {total}")
    expected_numbers = np.arange(1, total + 1, dtype=np.int64)
    if not np.array_equal(np.sort(seq_numbers), expected_numbers):
        raise ValueError("all-cache SeqIDs are not a complete 1..N permutation")

    records = []
    for name, mask in (
        ("coding", seq_numbers <= coding_count),
        ("noncoding", seq_numbers > coding_count),
    ):
        record = metric_record(
            name, llrs[mask], labels[mask], args.bootstrap, args.bootstrap_seed
        )
        records.append(record)

    metrics_payload = {
        "schema": "clinvar_all_subselected_metrics_v1",
        "model_name": MODEL_NAME,
        "display_model": DISPLAY_MODEL,
        "source": {
            "all_summary": str(args.all_summary),
            "all_summary_sha256": sha256(args.all_summary),
            "all_source_dir": str(args.all_source_dir),
            "coding_csv": str(args.coding_csv),
            "coding_csv_sha256": sha256(args.coding_csv),
            "noncoding_csv": str(args.noncoding_csv),
            "noncoding_csv_sha256": sha256(args.noncoding_csv),
            "h5_shards": shard_records,
            "selection_rule": f"all_1..all_{coding_count}=coding; remaining all IDs=noncoding",
        },
        "records": records,
    }
    atomic_write_json(args.metrics_json, metrics_payload)
    if args.apply:
        replace_table_row(
            args.tables_dir
            / "S24_zero_shot_classification_of_coding_clinvar_variants.csv",
            records[0],
        )
        replace_table_row(
            args.tables_dir
            / "S25_zero_shot_classification_of_noncoding_clinvar_variants.csv",
            records[1],
        )
        update_provenance(
            args.provenance,
            args.metrics_json,
            records,
            shard_records,
            coding_count,
            noncoding_count,
            args.all_source_dir / "clinvar_all_1023bp.csv",
            args.all_summary,
            Path(__file__).resolve().parents[3],
        )
    print(
        json.dumps(
            {
                "metrics_json": str(args.metrics_json),
                "records": records,
                "applied": args.apply,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
