#!/usr/bin/env python3
"""Split a variant CSV into deterministic, header-preserving input shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=12)
    parser.add_argument("--prefix", default="variant")
    args = parser.parse_args()
    if args.parts < 1:
        raise ValueError("--parts must be positive")
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{args.input}: expected a CSV label column")
        rows = list(reader)
        fieldnames = reader.fieldnames
    ids = [row.get("seqID") or row.get("SeqID") or "" for row in rows]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{args.input}: missing or duplicate variant IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "gpn_star_variant_csv_shards_v1",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "input_rows": len(rows),
        "parts": args.parts,
        "shards": [],
    }
    for shard_index in range(args.parts):
        start = (len(rows) * shard_index) // args.parts
        end = (len(rows) * (shard_index + 1)) // args.parts
        path = args.output_dir / f"{args.prefix}.shard{shard_index:02d}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[start:end])
        manifest["shards"].append(
            {
                "index": shard_index,
                "start_row": start,
                "end_row_exclusive": end,
                "rows": end - start,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "first_variant_id": ids[start] if start < end else "",
                "last_variant_id": ids[end - 1] if start < end else "",
            }
        )
    manifest["status"] = "PASS"
    manifest_path = args.output_dir / "shard_manifest.json"
    manifest["manifest"] = str(manifest_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
