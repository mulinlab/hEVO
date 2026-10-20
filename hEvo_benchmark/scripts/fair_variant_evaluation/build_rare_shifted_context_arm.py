#!/usr/bin/env python3
"""Build a fixed-position hg38 context arm for the Rare benchmark.

The variant set and alleles are unchanged.  Only the 1023-bp window start is
shifted by a fixed genomic offset, so this is a model-blind context-position
sensitivity arm rather than a sample-selection procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pysam


WINDOW_LEN = 1023
DEFAULT_CENTER = WINDOW_LEN // 2
OUTPUT_COLUMNS = ["SeqID", "refSeq", "altSeq", "label", "chrom", "pos"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chromosome_name(fasta: pysam.FastaFile, value: object) -> str:
    raw = str(value)
    candidates = [raw, f"chr{raw}"] if not raw.startswith("chr") else [raw, raw[3:]]
    for candidate in candidates:
        if candidate in fasta.references:
            return candidate
    raise ValueError(f"chromosome_not_in_fasta: {raw}")


def validate_source(frame: pd.DataFrame) -> None:
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)}")
    if frame["SeqID"].duplicated().any():
        raise ValueError("dataset contains duplicate SeqID values")
    labels = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if set(labels) != {0, 1}:
        raise ValueError("dataset must contain both label=0 and label=1")
    if frame[["chrom", "pos"]].isna().any().any():
        raise ValueError("dataset contains missing chrom/pos values")
    for column in ("refSeq", "altSeq"):
        lengths = frame[column].astype(str).str.len()
        if not (lengths == WINDOW_LEN).all():
            raise ValueError(f"{column} is not {WINDOW_LEN} bp for every row")
        bad = (
            frame[column]
            .astype(str)
            .str.upper()
            .map(lambda value: set(value) - set("ACGT"))
        )
        if bad.map(bool).any():
            raise ValueError(f"{column} contains non-ACGT bases")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument(
        "--offset",
        type=int,
        required=True,
        help="Genomic window-start offset from the centered window",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.dataset)
    validate_source(source)
    if (
        not args.fasta.exists()
        or not args.fasta.with_suffix(args.fasta.suffix + ".fai").exists()
    ):
        raise FileNotFoundError("FASTA and matching .fai index are required")
    if args.offset not in {-128, -64, 64, 128}:
        raise ValueError("offset must be one of -128, -64, 64, 128")

    target_index = DEFAULT_CENTER - args.offset
    fasta = pysam.FastaFile(str(args.fasta))
    rows = []
    mismatches = []
    for row in source[OUTPUT_COLUMNS].itertuples(index=False):
        chrom = chromosome_name(fasta, row.chrom)
        pos = int(row.pos)
        start0 = pos - 1 - target_index
        ref_seq = fasta.fetch(chrom, start0, start0 + WINDOW_LEN).upper()
        if (
            len(ref_seq) != WINDOW_LEN
            or ref_seq[target_index] != str(row.refSeq)[DEFAULT_CENTER].upper()
        ):
            mismatches.append(str(row.SeqID))
            continue
        alt_seq = (
            ref_seq[:target_index]
            + str(row.altSeq)[DEFAULT_CENTER].upper()
            + ref_seq[target_index + 1 :]
        )
        rows.append(
            {
                "SeqID": row.SeqID,
                "refSeq": ref_seq,
                "altSeq": alt_seq,
                "label": int(row.label),
                "chrom": row.chrom,
                "pos": pos,
            }
        )
    if mismatches:
        raise ValueError(
            f"reference_mismatches={len(mismatches)} examples={mismatches[:5]}"
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    qa = {
        "schema": "hevo_rare_shifted_context_arm_v1",
        "arm_id": args.arm_id,
        "status": "EXPLORATORY_NOT_PROSPECTIVE_AFTER_BASELINE",
        "rows": len(output),
        "positive": int(output["label"].sum()),
        "negative": int((output["label"] == 0).sum()),
        "sequence_length": WINDOW_LEN,
        "original_variant_index": DEFAULT_CENTER,
        "variant_index": target_index,
        "window_start_offset_bp": args.offset,
        "source_coordinate_assembly": "hg38",
        "model_scores_read_during_build": False,
        "allowed_models": ["hEvo-base", "hEvo-large"],
        "score_rule": "benchmark_score=-raw_llr; score_mode=neg_llr",
        "source_sha256": sha256(args.dataset),
        "output_sha256": sha256(args.output),
    }
    args.qa_report.parent.mkdir(parents=True, exist_ok=True)
    args.qa_report.write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "hevo_rare_shifted_context_arm_manifest_v1",
        "arm_id": args.arm_id,
        "status": qa["status"],
        "input": {"name": args.dataset.name, "sha256": sha256(args.dataset)},
        "output": {"name": args.output.name, "sha256": sha256(args.output)},
        "frozen_protocol": {
            "evaluation_unit": "variant",
            "window_length": WINDOW_LEN,
            "source_coordinate_assembly": "hg38",
            "window_start_offset_bp": args.offset,
            "variant_index": target_index,
            "model_scores_read_during_build": False,
            "allowed_models": ["hEvo-base", "hEvo-large"],
            "score_rule": "benchmark_score=-raw_llr; score_mode=neg_llr",
        },
        "qa": qa,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"manifest": str(args.manifest), "output": str(args.output), "qa": qa},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
