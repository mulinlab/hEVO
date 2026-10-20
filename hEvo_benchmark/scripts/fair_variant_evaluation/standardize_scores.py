#!/usr/bin/env python3
"""Convert evaluator outputs to the common fair-summary score schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(
            csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
        )


def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def _dataset_fields(
    row: dict[str, str], variant_index: int | None = None
) -> tuple[str, str, str, str]:
    """Return the registered coordinate and allele fields for either dataset schema."""
    chrom = _first(row, "chrom", "chrom_hg38", "chromosome_hg38")
    pos = _first(row, "pos", "pos_hg38", "position_hg38")
    ref = _first(row, "ref", "vcf_ref", "refbase")
    alt = _first(row, "alt", "vcf_alt", "altbase")
    if not ref or not alt:
        ref_seq = _first(row, "refseq", "refSeq", "ref_sequence")
        alt_seq = _first(row, "altseq", "altSeq", "alt_sequence")
        if len(ref_seq) != len(alt_seq) or not ref_seq or len(ref_seq) % 2 == 0:
            raise ValueError(
                "dataset must contain explicit ref/alt or an odd-length paired SNV sequence"
            )
        center = len(ref_seq) // 2 if variant_index is None else variant_index
        if not 0 <= center < len(ref_seq):
            raise ValueError(f"variant index out of range: {center}")
        if ref_seq[center] == alt_seq[center]:
            raise ValueError("paired dataset sequences do not differ at their center")
        ref, alt = ref_seq[center], alt_seq[center]
    if not chrom or not pos or not ref or not alt:
        raise ValueError(f"dataset row is missing canonical hg38 fields: {row}")
    return chrom, pos, ref.upper(), alt.upper()


def read_h5(path: Path) -> dict[str, tuple[float, int]]:
    # HDF5 details are JSON strings produced by zeroshot_paried_seq_input.py.
    with h5py.File(path, "r") as handle:
        details = [json.loads(value) for value in handle["details"][...].astype(str)]
        llrs = handle["llrs"][...]
        labels = handle["labels"][...]
    result: dict[str, tuple[float, int]] = {}
    for detail, llr, label in zip(details, llrs, labels):
        key = str(detail.get("idx", ""))
        if not key or key in result:
            raise ValueError(f"invalid or duplicate HDF5 variant id: {key!r}")
        result[key] = (-float(llr), int(label))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset CSV")
    parser.add_argument(
        "--input", type=Path, required=True, help="GPN-Star TSV or hEvo HDF5"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant-index",
        type=int,
        default=None,
        help="SNV index for non-centered paired sequences",
    )
    args = parser.parse_args()
    dataset = read_csv(args.dataset)
    ids = {row.get("seqID") or row.get("SeqID") or "" for row in dataset}
    if "" in ids or len(ids) != len(dataset):
        raise ValueError("dataset variant IDs are not unique")
    labels = {
        row["seqID"] if "seqID" in row else row["SeqID"]: int(row["label"])
        for row in dataset
    }
    fields = {
        row.get("seqID") or row.get("SeqID") or "": _dataset_fields(
            row, args.variant_index
        )
        for row in dataset
    }
    rows: list[dict[str, object]] = []
    if args.input.suffix.lower() in {".h5", ".hdf5"}:
        scores = read_h5(args.input)
        if set(scores) != ids:
            raise ValueError(
                f"HDF5 ID set mismatch: dataset={len(ids)} scores={len(scores)}"
            )
        for row in dataset:
            sid = row.get("seqID") or row.get("SeqID") or ""
            score, score_label = scores[sid]
            if score_label != labels[sid]:
                raise ValueError(f"label mismatch for {sid}")
            chrom, pos, _, _ = fields[sid]
            rows.append(
                {
                    "model": args.model,
                    "variant_id": sid,
                    "chrom": chrom,
                    "pos": pos,
                    "label": labels[sid],
                    "benchmark_score": score,
                    "score_mode": "neg_llr",
                }
            )
    else:
        source = read_csv(args.input)
        passed = [row for row in source if row.get("status", "PASS") == "PASS"]
        by_id = {row.get("variant_id", ""): row for row in passed}
        if set(by_id) != ids:
            raise ValueError(
                f"TSV ID set mismatch: dataset={len(ids)} scores={len(by_id)}"
            )
        for row in dataset:
            sid = row.get("seqID") or row.get("SeqID") or ""
            score_row = by_id[sid]
            if score_row.get("score_mode", "neg_llr") != "neg_llr":
                raise ValueError(
                    f"score direction is not the preregistered neg_llr mode for {sid}: "
                    f"{score_row.get('score_mode')}"
                )
            for field in ("chrom", "pos", "label"):
                expected = (
                    fields[sid][0 if field == "chrom" else 1]
                    if field != "label"
                    else labels[sid]
                )
                observed = _first(
                    score_row, field, "chrom_hg38" if field == "chrom" else "pos_hg38"
                )
                if str(observed).replace("chr", "") != str(expected).replace("chr", ""):
                    raise ValueError(f"{field} mismatch for {sid}")
            chrom, pos, expected_ref, expected_alt = fields[sid]
            for field in ("ref", "alt"):
                expected = expected_ref if field == "ref" else expected_alt
                observed = _first(
                    score_row, field, "vcf_ref" if field == "ref" else "vcf_alt"
                )
                if observed.upper() != expected.upper():
                    raise ValueError(f"{field} mismatch for {sid}")
            benchmark_score = _first(score_row, "benchmark_score", "locus_score")
            if not benchmark_score:
                raise ValueError(f"missing benchmark_score/locus_score for {sid}")
            rows.append(
                {
                    "model": args.model,
                    "variant_id": sid,
                    "chrom": chrom,
                    "pos": pos,
                    "label": labels[sid],
                    "benchmark_score": float(benchmark_score),
                    "score_mode": _first(score_row, "score_mode") or "neg_llr",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": args.model,
                "rows": len(rows),
                "input_sha256": sha256(args.input),
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
