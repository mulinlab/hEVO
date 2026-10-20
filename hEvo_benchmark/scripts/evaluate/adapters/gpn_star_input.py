#!/usr/bin/env python3
"""Normalize paired variant datasets for native GPN-Star VEP inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import pandas as pd


BASES = frozenset("ACGT")
OUTPUT_COLUMNS = [
    "task_name",
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "selected_allele",
    "other_allele",
    "score_mode",
    "source_row",
    "source_file",
    "input_status",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--score-mode",
        choices=["neg_llr", "raw_llr", "abs_llr", "selected_other"],
        default="neg_llr",
    )
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_csv(
        path, sep=None, engine="python", dtype=str, keep_default_na=False
    )


def _normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in frame.columns:
        key = re.sub(r"[^a-z0-9]", "", str(column).lower())
        result.setdefault(key, str(column))
    return result


def _pick(columns: dict[str, str], *names: str) -> str | None:
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in columns:
            return columns[key]
    return None


def _value(row: pd.Series, column: str | None) -> str:
    return "" if column is None else str(row.get(column, "")).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_pair(row: pd.Series, columns: dict[str, str]) -> tuple[str, str]:
    ref_col = _pick(
        columns,
        "refseq",
        "refSeq",
        "ref_sequence",
        "reference_sequence",
        "ref_seq",
    )
    alt_col = _pick(
        columns,
        "altseq",
        "altSeq",
        "alt_sequence",
        "alternate_sequence",
        "alt_seq",
    )
    return _value(row, ref_col).upper(), _value(row, alt_col).upper()


def _coordinates(row: pd.Series, columns: dict[str, str]) -> tuple[str, str]:
    chrom_col = _pick(
        columns,
        "chrom_hg38",
        "chrom38",
        "chromosome_hg38",
        "chrom_grch38",
        "chr_grch38",
        "chr",
        "chrom",
    )
    pos_col = _pick(
        columns,
        "pos_hg38",
        "position_hg38",
        "pos_grch38",
        "position_grch38",
        "pos38",
        "pos",
    )
    chrom = _value(row, chrom_col)
    pos = _value(row, pos_col)
    if chrom.lower().startswith("chr"):
        chrom = "chr" + chrom[3:]
    elif chrom:
        chrom = "chr" + chrom
    return chrom, pos


def _explicit_alleles(row: pd.Series, columns: dict[str, str]) -> tuple[str, str]:
    ref_col = _pick(columns, "ref", "vcf_ref", "refbase", "reference")
    alt_col = _pick(columns, "alt", "vcf_alt", "altbase", "alternate")
    return _value(row, ref_col).upper(), _value(row, alt_col).upper()


def _selected_other(row: pd.Series, columns: dict[str, str]) -> tuple[str, str]:
    selected_col = _pick(columns, "selected_allele", "selected", "favored_allele")
    other_col = _pick(columns, "other_allele", "other", "alternative_allele")
    selected = _value(row, selected_col).upper()
    other = _value(row, other_col).upper()
    if selected and not other:
        ref, alt = _explicit_alleles(row, columns)
        if selected == ref:
            other = alt
        elif selected == alt:
            other = ref
    return selected, other


def _variant_id(row: pd.Series, columns: dict[str, str], source_row: int) -> str:
    column = _pick(columns, "variant_id", "variant", "seqid", "id", "rsid")
    value = _value(row, column)
    return value or f"source_row_{source_row}"


def _label(row: pd.Series, columns: dict[str, str]) -> str:
    column = _pick(columns, "label", "target", "class", "y")
    return _value(row, column)


def _derive_alleles(
    row: pd.Series,
    columns: dict[str, str],
) -> tuple[str, str, str | None]:
    ref, alt = _explicit_alleles(row, columns)
    sequence_ref, sequence_alt = _sequence_pair(row, columns)
    if not ref or not alt:
        if sequence_ref and sequence_alt:
            if len(sequence_ref) != len(sequence_alt):
                return "", "", "sequence_length_mismatch"
            differences = [
                index
                for index, (left, right) in enumerate(zip(sequence_ref, sequence_alt))
                if left != right
            ]
            if len(differences) != 1:
                return "", "", "sequence_pair_is_not_single_snv"
            index = differences[0]
            ref, alt = sequence_ref[index], sequence_alt[index]
        else:
            return "", "", "missing_ref_alt_or_paired_sequences"
    if sequence_ref and sequence_alt:
        if len(sequence_ref) != len(sequence_alt):
            return "", "", "sequence_length_mismatch"
        differences = [
            index
            for index, (left, right) in enumerate(zip(sequence_ref, sequence_alt))
            if left != right
        ]
        if len(differences) != 1:
            return "", "", "sequence_pair_is_not_single_snv"
        index = differences[0]
        if (ref, alt) != (sequence_ref[index], sequence_alt[index]):
            return "", "", "explicit_alleles_disagree_with_sequence_center"
    return ref, alt, None


def normalize(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = _read_table(args.input)
    if args.max_records:
        frame = frame.iloc[: args.max_records].copy()
    columns = _normalized_columns(frame)
    output: list[dict[str, str]] = []

    for source_row, (_, row) in enumerate(frame.iterrows(), start=2):
        variant_id = _variant_id(row, columns, source_row)
        chrom, pos_text = _coordinates(row, columns)
        ref, alt, error = _derive_alleles(row, columns)
        sequence_ref, sequence_alt = _sequence_pair(row, columns)
        selected, other = _selected_other(row, columns)
        status = "PASS"
        reason = ""

        try:
            pos = int(pos_text)
            if pos <= 0:
                raise ValueError("position_must_be_positive")
        except ValueError as exc:
            pos = ""
            status = "BLOCKED_NO_NATIVE_MSA_INPUT"
            reason = f"invalid_or_missing_hg38_position:{exc}"

        if not chrom:
            status = "BLOCKED_NO_NATIVE_MSA_INPUT"
            reason = "missing_hg38_chromosome"
        if error:
            status = "BLOCKED_INVALID_VARIANT"
            reason = error
        if not error and sequence_ref and sequence_alt:
            difference = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(sequence_ref, sequence_alt)
                    )
                    if left != right
                ),
                None,
            )
            allowed_centers = {len(sequence_ref) // 2}
            if len(sequence_ref) % 2 == 0:
                allowed_centers.add(len(sequence_ref) // 2 - 1)
            if difference not in allowed_centers:
                status = "BLOCKED_INVALID_VARIANT"
                reason = "paired_snv_is_not_at_sequence_center"
        if ref not in BASES or alt not in BASES or ref == alt:
            status = "BLOCKED_INVALID_VARIANT"
            reason = "ref_alt_must_be_distinct_single_acgt_bases"

        if selected or other:
            if selected not in BASES or other not in BASES or selected == other:
                status = "BLOCKED_INVALID_VARIANT"
                reason = "selected_other_must_be_distinct_single_acgt_bases"
            elif {selected, other} != {ref, alt}:
                status = "BLOCKED_INVALID_VARIANT"
                reason = "selected_other_do_not_match_ref_alt"
        if args.score_mode == "selected_other" and not (selected and other):
            status = "BLOCKED_INVALID_VARIANT"
            reason = "selected_other_score_mode_requires_selected_and_other_alleles"

        output.append(
            {
                "task_name": args.task_name,
                "variant_id": variant_id,
                "chrom": chrom,
                "pos": str(pos),
                "ref": ref,
                "alt": alt,
                "label": _label(row, columns),
                "selected_allele": selected,
                "other_allele": other,
                "score_mode": args.score_mode,
                "source_row": str(source_row),
                "source_file": str(args.input.resolve()),
                "input_status": status,
                "failure_reason": reason,
            }
        )

    normalized = pd.DataFrame(output, columns=OUTPUT_COLUMNS)
    if normalized.empty:
        raise ValueError("input contains no rows")
    duplicate_ids = normalized.variant_id.duplicated(keep=False)
    duplicate_id_count = int(duplicate_ids.sum())
    if duplicate_id_count:
        normalized.loc[duplicate_ids, "variant_id"] = (
            normalized.loc[duplicate_ids, "variant_id"]
            + "__source_row_"
            + normalized.loc[duplicate_ids, "source_row"]
        )
        if normalized.variant_id.duplicated().any():
            raise ValueError(
                "variant_id uniqueness could not be restored deterministically"
            )
    valid = normalized[normalized.input_status.eq("PASS")]
    duplicate_keys = valid.duplicated(["chrom", "pos", "ref", "alt"], keep=False)
    duplicate_count = int(duplicate_keys.sum())
    manifest = {
        "schema": "gpn_star_variant_input_v1",
        "task_name": args.task_name,
        "input": str(args.input.resolve()),
        "input_sha256": _sha256_file(args.input),
        "rows": int(len(normalized)),
        "valid_rows": int(len(valid)),
        "blocked_rows": int(len(normalized) - len(valid)),
        "duplicate_variant_rows": duplicate_count,
        "duplicate_variant_id_rows": duplicate_id_count,
        "score_mode": args.score_mode,
        "assembly": "hg38",
        "coordinate_convention": "chr_prefixed_hg38_chromosome; one_based_pos",
        "sequence_only_rows": int(
            normalized.input_status.eq("BLOCKED_NO_NATIVE_MSA_INPUT").sum()
        ),
    }
    return normalized, manifest


def main() -> int:
    args = parse_args()
    normalized, manifest = normalize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {args.output} rows={len(normalized)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
