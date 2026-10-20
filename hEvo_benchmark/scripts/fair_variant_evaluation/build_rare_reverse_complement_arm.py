#!/usr/bin/env python3
"""Build a model-blind reverse-complement context arm for Rare Variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


WINDOW_LEN = 1023
CENTER = WINDOW_LEN // 2
DNA = str.maketrans("ACGT", "TGCA")
OUTPUT_COLUMNS = ["SeqID", "refSeq", "altSeq", "label", "chrom", "pos"]
TRAINING_OVERLAP_STATUS = "NOT_VERIFIABLE"
TRAINING_OVERLAP_EVIDENCE = (
    "No complete hEvo-60k training-variant manifest was found in the accessible "
    "model package or registry; exact overlap with the hEvo training corpus cannot "
    "be established from the available evidence."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(DNA)[::-1]


def validate(frame: pd.DataFrame, center_index: int = CENTER) -> None:
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
        if (
            frame[column]
            .astype(str)
            .map(lambda value: set(value.upper()) - set("ACGT"))
            .map(bool)
            .any()
        ):
            raise ValueError(f"{column} contains non-ACGT bases")
    for ref, alt in zip(frame["refSeq"].astype(str), frame["altSeq"].astype(str)):
        changed = [
            index
            for index, pair in enumerate(zip(ref.upper(), alt.upper()))
            if pair[0] != pair[1]
        ]
        if changed != [center_index]:
            raise ValueError(f"each row must contain one SNV at index {center_index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Frozen Rare paired-sequence CSV"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    parser.add_argument("--arm-id", default="R4_R0_REVERSE_COMPLEMENT")
    parser.add_argument("--center-index", type=int, default=CENTER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.dataset)
    if not 0 <= args.center_index < WINDOW_LEN:
        raise ValueError(f"center-index must be in [0, {WINDOW_LEN - 1}]")
    validate(source, args.center_index)
    output = source[OUTPUT_COLUMNS].copy()
    output["refSeq"] = output["refSeq"].map(reverse_complement)
    output["altSeq"] = output["altSeq"].map(reverse_complement)
    output_center_index = WINDOW_LEN - 1 - args.center_index
    validate(output, output_center_index)
    output.to_csv(args.output, index=False)
    qa = {
        "schema": "hevo_rare_reverse_complement_arm_v1",
        "arm_id": args.arm_id,
        "status": "EXPLORATORY_NOT_PROSPECTIVE_AFTER_BASELINE",
        "training_overlap_status": TRAINING_OVERLAP_STATUS,
        "training_overlap_evidence": TRAINING_OVERLAP_EVIDENCE,
        "rows": len(output),
        "positive": int(output["label"].astype(int).sum()),
        "negative": int((output["label"].astype(int) == 0).sum()),
        "sequence_length": WINDOW_LEN,
        "input_variant_index": args.center_index,
        "output_variant_index": output_center_index,
        "transform": "reverse-complement refSeq and altSeq for every row; retain SeqID, chrom, pos, label",
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
        "schema": "hevo_rare_reverse_complement_arm_manifest_v1",
        "arm_id": args.arm_id,
        "status": qa["status"],
        "input": {"name": args.dataset.name, "sha256": sha256(args.dataset)},
        "output": {"name": args.output.name, "sha256": sha256(args.output)},
        "frozen_protocol": {
            "evaluation_unit": "variant",
            "input_variant_index": args.center_index,
            "output_variant_index": output_center_index,
            "model_scores_read_during_build": False,
            "allowed_models": ["hEvo-base", "hEvo-large"],
            "score_rule": "benchmark_score=-raw_llr; score_mode=neg_llr",
            "aggregation_rule": "equal arithmetic mean of forward and reverse-complement benchmark_score per model",
            "training_overlap_status": TRAINING_OVERLAP_STATUS,
            "training_overlap_evidence": TRAINING_OVERLAP_EVIDENCE,
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
