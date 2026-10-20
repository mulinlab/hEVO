from __future__ import annotations

import argparse
import os

import csv
import json
from pathlib import Path


RESULT_ROOT = Path(
    os.environ.get(
        "DEEPSEA_RESULTS_ROOT",
        str(Path(__file__).resolve().parents[2] / "results/DeepSEA_lora"),
    )
)
CATEGORIES = ("DHS", "TF", "HM")
FIELDS = [
    "model_name",
    "category",
    "run_dir",
    "test_samples",
    "num_labels",
    "macro_auroc",
    "macro_auprc",
    "micro_auroc",
    "micro_auprc",
    "bce_loss",
    "valid_label_count",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize existing DeepSEA category metrics; no inference."
    )
    parser.add_argument("--results-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result_root = args.results_root
    output_csv = args.output_csv or result_root / "category_summary.csv"
    output_json = args.output_json or result_root / "category_summary.json"
    if not result_root.is_dir():
        parser.error(f"results directory does not exist: {result_root}")
    rows = []
    for path in sorted(result_root.glob("*/category_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for category in CATEGORIES:
            item = payload["categories"][category]
            rows.append(
                {
                    "model_name": payload["model_name"],
                    "category": category,
                    "run_dir": payload["run_dir"],
                    "test_samples": payload["test_samples"],
                    "num_labels": payload["category_definition"]["categories"][
                        category
                    ]["count"],
                    **{key: item.get(key) for key in FIELDS[5:]},
                }
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(
        json.dumps(
            {"categories": list(CATEGORIES), "rows": rows}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
