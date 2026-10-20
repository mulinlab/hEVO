#!/usr/bin/env python3
"""Average two fixed-orientation paired-sequence score tables."""

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


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty score table: {path}")
    return rows


def normalized_key(row: dict[str, str], id_column: str) -> tuple[str, str, str, str]:
    return (
        row[id_column],
        row["chrom"].replace("chr", ""),
        str(int(row["pos"])),
        row["label"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse-complement", type=Path, required=True)
    parser.add_argument(
        "--model", required=True, help="Output hEvo strand-average model name"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--id-column", choices=("variant_id", "locus_id"), default="variant_id"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forward, reverse = read(args.forward), read(args.reverse_complement)
    id_column = args.id_column
    if any(id_column not in row for row in forward + reverse):
        raise ValueError(f"both score tables must contain {id_column}")
    f_by_id = {row[id_column]: row for row in forward}
    r_by_id = {row[id_column]: row for row in reverse}
    if len(f_by_id) != len(forward) or len(r_by_id) != len(reverse):
        raise ValueError("score tables contain duplicate variant_id values")
    if set(f_by_id) != set(r_by_id):
        raise ValueError("forward and reverse score tables have different variant IDs")
    rows: list[dict[str, object]] = []
    for row in forward:
        sid = row[id_column]
        other = r_by_id[sid]
        if normalized_key(row, id_column) != normalized_key(other, id_column):
            raise ValueError(f"forward/reverse coordinate or label mismatch for {sid}")
        if (
            row.get("score_mode", "neg_llr") != "neg_llr"
            or other.get("score_mode", "neg_llr") != "neg_llr"
        ):
            raise ValueError(f"unexpected score mode for {sid}")
        rows.append(
            {
                "model": args.model,
                id_column: sid,
                "chrom": row["chrom"],
                "pos": row["pos"],
                "label": row["label"],
                "benchmark_score": (
                    float(row["benchmark_score"]) + float(other["benchmark_score"])
                )
                / 2.0,
                "score_mode": "neg_llr",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        output_columns = [
            "model",
            id_column,
            "chrom",
            "pos",
            "label",
            "benchmark_score",
            "score_mode",
        ]
        writer = csv.DictWriter(
            handle, fieldnames=output_columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": args.model,
                "rows": len(rows),
                "forward_sha256": sha256(args.forward),
                "reverse_complement_sha256": sha256(args.reverse_complement),
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
