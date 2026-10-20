#!/usr/bin/env python3
"""Split a normalized GPN-Star TSV into balanced, header-preserving shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="shard")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional external shard manifest path.",
    )
    args = parser.parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        args.output_dir / f"{args.prefix}_{index:02d}.tsv"
        for index in range(args.shards)
    ]
    counts = [0] * args.shards
    handles = [path.open("w", encoding="utf-8", newline="") for path in paths]
    try:
        with args.input.open("r", encoding="utf-8", newline="") as source:
            header = source.readline()
            if not header:
                raise ValueError("normalized input is empty")
            for handle in handles:
                handle.write(header)
            for row_index, line in enumerate(source):
                shard = row_index % args.shards
                handles[shard].write(line)
                counts[shard] += 1
    finally:
        for handle in handles:
            handle.close()

    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "shards": [str(path.resolve()) for path in paths],
        "shard_count": args.shards,
        "rows": sum(counts),
        "rows_per_shard": counts,
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(f"wrote {sum(counts)} rows to {args.shards} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
