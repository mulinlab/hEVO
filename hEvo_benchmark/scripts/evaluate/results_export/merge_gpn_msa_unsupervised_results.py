#!/usr/bin/env python3
"""Merge validated GPN-MSA element-clustering results into S11–S16."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

try:
    from .common import read_csv as _read_csv, write_csv as _write_csv
except ImportError:  # direct script execution
    from common import read_csv as _read_csv, write_csv as _write_csv


TASKS = {
    "ENCODE": {
        "rows": 5000,
        "classes": 5,
        "source": "S11_seed_level_representation_separation_for_encode_candidate_cis_regulatory_elements.csv",
        "summary": "S12_summary_of_representation_separation_for_encode_candidate_cis_regulatory_elements.csv",
    },
    "GENCODEv45": {
        "rows": 3000,
        "classes": 3,
        "source": "S13_seed_level_representation_separation_for_gencode_gene_structure_classes.csv",
        "summary": "S14_summary_of_representation_separation_for_gencode_gene_structure_classes.csv",
    },
    "UCSC": {
        "rows": 5000,
        "classes": 5,
        "source": "S15_seed_level_representation_separation_for_curated_refseq_gene_structure_classes.csv",
        "summary": "S16_summary_of_representation_separation_for_curated_refseq_gene_structure_classes.csv",
    },
}
SOURCE_COLUMNS = (
    "model_name",
    "seed",
    "method",
    "n_rows",
    "n_classes",
    "ch_index",
    "silhouette",
    "coordinate_space",
    "metric",
)
SUMMARY_COLUMNS = (
    "model_name",
    "method",
    "n_seeds",
    "n_rows",
    "ch_index_mean",
    "ch_index_sd",
    "silhouette_mean",
    "silhouette_sd",
)
METHODS = ("PCA", "tSNE")
SEEDS = (42, 43, 44)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_order_sha256(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['seqID']}\t{row['label']}\t{row['seq']}\n".encode("utf-8"))
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    return _read_csv(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    _write_csv(path, fieldnames, rows, atomic=True, atomic_style="suffix")


def finite(value: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def validate_candidate(
    run_root: Path, task: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    spec = TASKS[task]
    task_root = run_root / "unsupervised" / task
    run_manifest = json.loads(
        (run_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    task_manifest = run_manifest["tasks"][task]
    input_manifest = json.loads(
        (task_root / "input_manifest.json").read_text(encoding="utf-8")
    )
    input_path = Path(task_manifest["input_csv"])
    if sha256_file(input_path) != task_manifest["input_sha256"]:
        raise ValueError(f"{task}: input CSV checksum changed")
    if input_manifest.get("input_sha256") != task_manifest["input_sha256"]:
        raise ValueError(
            f"{task}: run and input manifests disagree on the input checksum"
        )
    with input_path.open(newline="", encoding="utf-8") as handle:
        input_rows = list(csv.DictReader(handle))
    if (
        len(input_rows) != spec["rows"]
        or len({r["label"] for r in input_rows}) != spec["classes"]
    ):
        raise ValueError(f"{task}: unexpected input row or class count")
    if row_order_sha256(input_rows) != input_manifest.get("row_order_sha256"):
        raise ValueError(f"{task}: input row order checksum changed")

    h5_manifest_path = task_root / "embeddings" / "GPN-MSA.manifest.json"
    h5_manifest = json.loads(h5_manifest_path.read_text(encoding="utf-8"))
    if h5_manifest.get("embedding_backend") != "gpn.msa.embedding.EmbeddingInference":
        raise ValueError(f"{task}: HDF5 was not produced by official GPN-MSA inference")
    if (
        h5_manifest.get("n_rows") != spec["rows"]
        or h5_manifest.get("hidden_size") != 768
    ):
        raise ValueError(f"{task}: unexpected GPN-MSA embedding dimensions")
    if h5_manifest.get("input_sha256") != task_manifest["input_sha256"]:
        raise ValueError(f"{task}: embedding input checksum does not match this run")
    if h5_manifest.get("row_order_sha256") != input_manifest.get("row_order_sha256"):
        raise ValueError(f"{task}: embedding row order does not match this run")

    source_path = task_root / "tables_candidate" / "source.csv"
    summary_path = task_root / "tables_candidate" / "summary.csv"
    _, source = read_csv(source_path)
    _, summary = read_csv(summary_path)
    if len(source) != 6 or {(r["method"], int(r["seed"])) for r in source} != {
        (method, seed) for method in METHODS for seed in SEEDS
    }:
        raise ValueError(f"{task}: expected two methods × three source seeds")
    for row in source:
        if row["model_name"] != "GPN-MSA":
            raise ValueError(f"{task}: unexpected source model {row['model_name']}")
        if (
            int(row["n_rows"]) != spec["rows"]
            or int(row["n_classes"]) != spec["classes"]
        ):
            raise ValueError(f"{task}: unexpected source dimensions")
        finite(row["ch_index"], f"{task} CH index")
        if row["silhouette"].lower() not in {"nan", "", "none"}:
            finite(row["silhouette"], f"{task} silhouette")
    if len(summary) != 2 or {(r["model_name"], r["method"]) for r in summary} != {
        ("GPN-MSA", method) for method in METHODS
    }:
        raise ValueError(f"{task}: expected two GPN-MSA summary rows")
    by_method = {
        method: [r for r in source if r["method"] == method] for method in METHODS
    }
    for row in summary:
        method = row["method"]
        values = [
            finite(r["ch_index"], f"{task} source CH index") for r in by_method[method]
        ]
        if int(row["n_seeds"]) != 3 or int(row["n_rows"]) != spec["rows"]:
            raise ValueError(f"{task}: unexpected summary dimensions")
        if not math.isclose(
            float(row["ch_index_mean"]), mean(values), rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(f"{task}: summary mean disagrees with source rows")
        if not math.isclose(
            float(row["ch_index_sd"]), stdev(values), rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(f"{task}: summary SD disagrees with source rows")
    return source, summary


def merge_one(
    path: Path, incoming: list[dict[str, str]], required: tuple[str, ...]
) -> str:
    fieldnames, old_rows = read_csv(path)
    if tuple(fieldnames) != required:
        raise ValueError(f"unexpected schema in {path}: {fieldnames}")
    rows = [row for row in old_rows if row["model_name"] != "GPN-MSA"]
    for row in incoming:
        clean = {
            key: ("" if value.lower() in {"nan", "none"} else value)
            for key, value in row.items()
        }
        rows.append(clean)
    expected_per_model = 6 if "seed" in required else 2
    if sum(row["model_name"] == "GPN-MSA" for row in rows) != expected_per_model:
        raise ValueError(f"{path}: duplicate or incomplete GPN-MSA rows")
    expected_models = 34
    if len({row["model_name"] for row in rows}) != expected_models:
        raise ValueError(f"{path}: expected {expected_models} models after merge")
    rows.sort(
        key=lambda row: (
            row["model_name"],
            int(row["seed"]) if "seed" in required else 0,
            METHODS.index(row["method"]),
        )
    )
    write_csv(path, fieldnames, rows)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    tables = args.benchmark_root / "tables"
    merge_manifest = {
        "run_manifest": str(args.run_root / "run_manifest.json"),
        "tasks": {},
    }
    for task, spec in TASKS.items():
        source, summary = validate_candidate(args.run_root, task)
        merge_manifest["tasks"][task] = {
            "source_candidate_sha256": sha256_file(
                args.run_root
                / "unsupervised"
                / task
                / "tables_candidate"
                / "source.csv"
            ),
            "summary_candidate_sha256": sha256_file(
                args.run_root
                / "unsupervised"
                / task
                / "tables_candidate"
                / "summary.csv"
            ),
            "source_table_sha256": merge_one(
                tables / spec["source"], source, SOURCE_COLUMNS
            ),
            "summary_table_sha256": merge_one(
                tables / spec["summary"], summary, SUMMARY_COLUMNS
            ),
            "merged_source_rows": len(source),
            "merged_summary_rows": len(summary),
        }
    output = args.run_root / "table_merge_manifest.json"
    output.write_text(json.dumps(merge_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(merge_manifest["tasks"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
