from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def read_record(path: Path) -> dict[str, Any]:
    manifest_path = path / "run_manifest.json"
    metrics_path = path / "metrics.json"
    summary_path = path / "model_summary.json"
    record: dict[str, Any] = {"run_dir": str(path), "status": "incomplete"}
    if manifest_path.exists():
        record["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary_path.exists():
        record.update(
            {
                f"model_{key}": value
                for key, value in json.loads(
                    summary_path.read_text(encoding="utf-8")
                ).items()
            }
        )
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record["status"] = metrics.get("status", "completed")
        record["best_epoch"] = metrics.get("best_epoch")
        record["best_valid_macro_auprc"] = metrics.get("best_valid_macro_auprc")
        record["elapsed_seconds"] = metrics.get("elapsed_seconds")
        record["peak_gpu_memory_bytes"] = metrics.get("peak_gpu_memory_bytes")
        record.update(
            {f"test_{key}": value for key, value in metrics.get("test", {}).items()}
        )
    manifest = record.pop("manifest", {})
    model = manifest.get("model", {})
    record.setdefault("model_name", model.get("name"))
    record.setdefault("model_path", model.get("url"))
    record["max_length"] = manifest.get("max_length")
    record["train_samples"] = manifest.get("train_samples")
    record["seed"] = manifest.get("seed")
    record["lora_r"] = manifest.get("lora", {}).get("r")
    record["lora_alpha"] = manifest.get("lora", {}).get("alpha")
    record["lora_dropout"] = manifest.get("lora", {}).get("dropout")
    record["target_modules"] = ",".join(
        manifest.get("lora", {}).get("target_modules", [])
    )
    record.pop("model_model_name", None)
    record.pop("model_model_path", None)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize DeepSEA LoRA run directories"
    )
    parser.add_argument(
        "--results_root",
        default=str(
            Path(
                os.environ.get(
                    "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
                )
            )
            / "results/DeepSEA_lora"
        ),
    )
    parser.add_argument("--output_csv")
    parser.add_argument("--output_json")
    args = parser.parse_args()
    root = Path(args.results_root)
    rows = [
        read_record(path)
        for path in sorted(root.iterdir())
        if path.is_dir()
        and not path.name.startswith(".")
        and not path.name.startswith("smoke_")
    ]
    rows.sort(key=lambda row: (str(row.get("model_name")), str(row.get("run_dir"))))
    output_csv = Path(args.output_csv or root / "summary.csv")
    output_json = Path(args.output_json or root / "summary.json")
    keys = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {"rows": len(rows), "csv": str(output_csv), "json": str(output_json)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
