#!/usr/bin/env python3
"""Validate and aggregate per-seed GPN-Star-M DeepSEA category metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


CATEGORIES = ("DHS", "HM", "TF")
METRICS = ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc", "bce_loss")
DNAMODELS_ROOT = Path(
    os.environ.get("DNAMODELS_BENCHMARK_ROOT", str(Path.home() / "DNAModelsBenchmark"))
)
DEFAULT_ROOT = (
    DNAMODELS_ROOT
    / "runs/deepsea_gpn_star_m_v1/training/deepsea_1epoch_full_40gpu_evo2params"
)
DEFAULT_SEEDS = (42,)


def load_results(root: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_test_samples: int | None = None
    for seed in seeds:
        run_dir = root / f"seed_{seed}"
        category_path = run_dir / "category_metrics.json"
        manifest_path = run_dir / "run_manifest.json"
        if not category_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Seed {seed} is missing completed metrics or manifest: {run_dir}"
            )
        payload = json.loads(category_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"Seed {seed} metrics are not marked completed")
        if manifest.get("seed") != seed or manifest.get("epochs") != 1:
            raise ValueError(f"Seed {seed} manifest does not match the requested run")
        if (
            manifest.get("train_samples") != 4_400_000
            or manifest.get("train_row_stride") != 1
        ):
            raise ValueError(
                f"Seed {seed} manifest has an unexpected training sample definition"
            )
        if manifest.get("lr_schedule") != "constant":
            raise ValueError(
                f"Seed {seed} did not use the Evo2-1B constant-after-warmup schedule"
            )
        if set(payload.get("categories", {})) != set(CATEGORIES):
            raise ValueError(f"Seed {seed} does not contain exactly DHS, HM and TF")
        mapping_path = Path(manifest["gpn_star"]["coordinate_mapping_summary"])
        mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        train_mapping = mapping_payload["splits"]["train"]
        test_mapping = mapping_payload["splits"]["test"]
        test_samples = int(payload["test_samples"])
        if test_samples != int(test_mapping["valid"]):
            raise ValueError(
                f"Seed {seed} evaluated {test_samples} mapped test rows, expected "
                f"{test_mapping['valid']} from {mapping_path}"
            )
        if expected_test_samples is None:
            expected_test_samples = test_samples
        elif expected_test_samples != test_samples:
            raise ValueError("Test sample counts differ across seeds")
        for category in CATEGORIES:
            metrics = payload["categories"][category]
            row: dict[str, Any] = {
                "model_name": payload["model_name"],
                "seed": seed,
                "category": category,
                "run_dir": str(run_dir),
                "test_samples": test_samples,
                "num_labels": int(
                    payload["category_definition"]["categories"][category]["count"]
                ),
                "train_rows": int(train_mapping["rows"]),
                "train_mapped_rows": int(train_mapping["valid"]),
                "train_mapping_coverage": float(train_mapping["valid_rate"]),
                "test_rows": int(test_mapping["rows"]),
                "test_mapped_rows": int(test_mapping["valid"]),
                "test_mapping_coverage": float(test_mapping["valid_rate"]),
            }
            for metric in METRICS:
                value = float(metrics[metric])
                if not math.isfinite(value):
                    raise ValueError(
                        f"Nonfinite {metric} for seed {seed}, category {category}"
                    )
                row[metric] = value
            row["valid_label_count"] = int(metrics["valid_label_count"])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "summary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results(root, tuple(args.seeds))

    fields = [
        "model_name",
        "seed",
        "category",
        "run_dir",
        "test_samples",
        "num_labels",
        "train_rows",
        "train_mapped_rows",
        "train_mapping_coverage",
        "test_rows",
        "test_mapped_rows",
        "test_mapping_coverage",
        *METRICS,
        "valid_label_count",
    ]
    with (output_dir / "category_metrics_by_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, Any]] = []
    for category in CATEGORIES:
        selected = [row for row in rows if row["category"] == category]
        record: dict[str, Any] = {
            "category": category,
            "num_labels": selected[0]["num_labels"],
            "test_samples": selected[0]["test_samples"],
            "num_seeds": len(selected),
            "seed": selected[0]["seed"],
            "train_rows": selected[0]["train_rows"],
            "train_mapped_rows": selected[0]["train_mapped_rows"],
            "train_mapping_coverage": selected[0]["train_mapping_coverage"],
            "test_rows": selected[0]["test_rows"],
            "test_mapped_rows": selected[0]["test_mapped_rows"],
            "test_mapping_coverage": selected[0]["test_mapping_coverage"],
        }
        for metric in METRICS:
            record[metric] = selected[0][metric]
        summary.append(record)

    summary_fields = list(summary[0])
    with (output_dir / "category_metrics_seed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    result = {
        "status": "completed",
        "model_name": rows[0]["model_name"],
        "run_root": str(root),
        "seeds": list(args.seeds),
        "category_order": list(CATEGORIES),
        "metrics": list(METRICS),
        "coordinate_mapping": {
            "train_rows": rows[0]["train_rows"],
            "train_mapped_rows": rows[0]["train_mapped_rows"],
            "train_mapping_coverage": rows[0]["train_mapping_coverage"],
            "test_rows": rows[0]["test_rows"],
            "test_mapped_rows": rows[0]["test_mapped_rows"],
            "test_mapping_coverage": rows[0]["test_mapping_coverage"],
        },
        "by_seed": rows,
        "summary": summary,
    }
    (output_dir / "category_metrics_seed_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": "completed", "output_dir": str(output_dir), "rows": len(rows)}
        )
    )


if __name__ == "__main__":
    main()
