#!/usr/bin/env python3
"""Canonicalize CausalDB paired alleles to the hg38 reference strand.

The source CausalDB export orders ``refseq``/``altseq`` by the signed effect
allele (beta), so ``refseq`` is not necessarily the GRCh38 reference allele.
Coordinate-aware models such as GPN-MSA require the latter.  This script keeps
the complete source cohort and labels unchanged, and only swaps the paired
alleles when the hg38 FASTA proves that the source ALT is the reference base.
No model output is read or used for selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pysam


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_chrom(value: str) -> str:
    value = str(value).strip()
    return value if value.startswith("chr") else f"chr{value}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CausalDB CSV")
    parser.add_argument(
        "--fasta", type=Path, required=True, help="hg38 FASTA with .fai"
    )
    parser.add_argument("--output", type=Path, required=True, help="Canonicalized CSV")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="JSON summary manifest"
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if (
        not args.fasta.is_file()
        or not args.fasta.with_suffix(args.fasta.suffix + ".fai").is_file()
    ):
        raise FileNotFoundError(f"FASTA and .fai are required: {args.fasta}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(args.fasta))
    total = 0
    flipped = 0
    unchanged = 0
    invalid = 0
    rows = []

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("input has no header")
        required = {"seqID", "refseq", "altseq", "label", "chrom_hg38", "pos_hg38"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"input missing required columns: {sorted(missing)}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            total += 1
            sid = row["seqID"]
            refseq = row["refseq"].strip().upper()
            altseq = row["altseq"].strip().upper()
            if len(refseq) == 0 or len(refseq) != len(altseq) or len(refseq) % 2 != 1:
                invalid += 1
                raise ValueError(f"invalid paired sequence for {sid}")
            differences = [
                (i, a, b) for i, (a, b) in enumerate(zip(refseq, altseq)) if a != b
            ]
            if len(differences) != 1 or differences[0][0] != len(refseq) // 2:
                invalid += 1
                raise ValueError(f"expected one center SNV for {sid}")
            center, source_ref, source_alt = differences[0]
            chrom = canonical_chrom(row["chrom_hg38"])
            pos = int(row["pos_hg38"])
            genome_ref = fasta.fetch(chrom, pos - 1, pos).upper()
            if genome_ref == source_ref:
                orientation = "unchanged"
            elif genome_ref == source_alt:
                orientation = "swapped"
                refseq, altseq = altseq, refseq
                flipped += 1
            else:
                invalid += 1
                raise ValueError(
                    f"neither paired allele matches hg38 at {chrom}:{pos} for {sid}: "
                    f"genome={genome_ref} source={source_ref}>{source_alt}"
                )
            if orientation == "unchanged":
                unchanged += 1
            row["refseq"] = refseq
            row["altseq"] = altseq
            row["canonical_ref"] = genome_ref
            row["canonical_alt"] = altseq[center]
            row["source_orientation"] = orientation
            rows.append(row)

    if total != len(rows) or invalid:
        raise ValueError(
            f"canonicalization failed: total={total} rows={len(rows)} invalid={invalid}"
        )
    output_fields = fieldnames + [
        "canonical_ref",
        "canonical_alt",
        "source_orientation",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "status": "PASS",
        "purpose": "model_blind_hg38_reference_allele_canonicalization",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "fasta": str(args.fasta.resolve()),
        "fasta_sha256": sha256(args.fasta),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "assembly": "hg38",
        "sequence_length_bp": len(rows[0]["refseq"]) if rows else None,
        "rows": len(rows),
        "source_orientation_unchanged": unchanged,
        "source_orientation_swapped": flipped,
        "positive": sum(int(row["label"]) == 1 for row in rows),
        "negative": sum(int(row["label"]) == 0 for row in rows),
        "model_scores_used_for_selection": False,
    }
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
