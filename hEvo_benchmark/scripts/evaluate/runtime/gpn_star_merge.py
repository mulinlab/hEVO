#!/usr/bin/env python3
"""Merge independent GPN-Star score TSV shards with provenance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from ..results_export.common import sha256 as _sha256
except ImportError:  # direct script execution
    evaluate_dir = Path(__file__).resolve().parents[1]
    if str(evaluate_dir) not in sys.path:
        sys.path.insert(0, str(evaluate_dir))
    from results_export.common import sha256 as _sha256


def sha256(path: Path) -> str:
    return _sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional external merge manifest path.",
    )
    args = parser.parse_args()
    if not args.input:
        raise ValueError("at least one --input is required")
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)

    frames = [
        pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        for path in args.input
    ]
    columns = list(frames[0].columns)
    if any(list(frame.columns) != columns for frame in frames[1:]):
        raise ValueError("score shard schemas do not match")
    merged = pd.concat(frames, ignore_index=True)
    if "variant_id" not in merged or "source_row" not in merged:
        raise ValueError("score shards must contain variant_id and source_row")
    if merged.variant_id.duplicated().any():
        duplicates = (
            merged.loc[merged.variant_id.duplicated(), "variant_id"].head(5).tolist()
        )
        raise ValueError(f"duplicate variant_id across score shards: {duplicates}")
    merged["task_name"] = args.task_name
    merged["source_row_numeric"] = pd.to_numeric(merged.source_row, errors="raise")
    merged = merged.sort_values("source_row_numeric", kind="stable").drop(
        columns=["source_row_numeric"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, sep="\t", index=False)

    raw = pd.to_numeric(merged.raw_llr, errors="coerce")
    manifest = {
        "schema": "gpn_star_variant_scores",
        "task_name": args.task_name,
        "input_shards": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": int(len(frame)),
            }
            for path, frame in zip(args.input, frames)
        ],
        "rows": int(len(merged)),
        "status_counts": merged.status.value_counts(dropna=False).to_dict(),
        "finite_raw_llr": int(np.isfinite(raw).sum()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {args.output} rows={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
