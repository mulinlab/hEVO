#!/usr/bin/env python3
"""Build candidate Fig.2 CH tables for the retained ENCODE and GENCODE tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DATASETS = (
    ("encode_ccre_v4_regulatory", "S11_encode_ccre_v4_hg38_ch_index_source.csv", "S12_encode_ccre_v4_hg38_ch_index_summary.csv"),
    ("gencode_v45_gene_structure", "S13_gencode_v45_gene_structure_hg38_ch_index_source.csv", "S14_gencode_v45_gene_structure_hg38_ch_index_summary.csv"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "datasets/dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("dataset manifest is not PASS")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reducer = Path(__file__).resolve().parent / "compute_ch_index.py"
    for dataset_id, source_name, summary_name in DATASETS:
        command = [
            sys.executable, str(reducer),
            "--reductions-dir", str(run / "reductions" / dataset_id),
            "--dataset-id", dataset_id,
            "--run-id", manifest["run_id"],
            "--dataset-sha256", manifest["dataset_sha256"][dataset_id],
            "--protocol-id", manifest["protocol_id"],
            "--source-csv", str(args.output_dir / source_name),
            "--summary-csv", str(args.output_dir / summary_name),
        ]
        subprocess.run(command, check=True)
    print(json.dumps({"status": "PASS", "datasets": len(DATASETS), "tables": 6}, indent=2))


if __name__ == "__main__":
    main()
