#!/usr/bin/env python3
"""Post-process native GPN-Star raw LLRs without changing primary scores.

Use the checkpoint-specific published neutral mean for the forward-reference
pentanucleotide and ALT: cLLR = raw LLR - llr_neutral_mean. Coordinates are
one-based hg38 positions. No strand canonicalization or calibration-table
averaging is introduced. See the official GPN-Star mutation-rate demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Reference:
    """Small indexed reads; does not load the whole reference into RAM."""

    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.index = {}
        for line in Path(str(path) + ".fai").read_text().splitlines():
            name, length, offset, bases, width, *_ = line.split("\t")
            self.index[name] = tuple(map(int, (length, offset, bases, width)))

    def context(self, chrom: str, pos: int) -> str:
        candidates = (chrom, chrom[3:] if chrom.startswith("chr") else "chr" + chrom)
        name = next((s for s in candidates if s in self.index), None)
        if name is None:
            raise ValueError("unknown_reference_contig")
        length, offset, bases, width = self.index[name]
        start, end = pos - 3, pos + 2
        if start < 0 or end > length:
            raise ValueError("pentanucleotide_crosses_contig_boundary")
        left = offset + start // bases * width + start % bases
        right = offset + (end - 1) // bases * width + (end - 1) % bases + 1
        return (
            self.data[left:right]
            .replace(b"\n", b"")
            .replace(b"\r", b"")
            .decode("ascii")
            .upper()
        )

    def close(self):
        self.data.close()
        self.file.close()


def orient(values: pd.Series, frame: pd.DataFrame) -> pd.Series:
    modes = frame["score_mode"].astype(str)
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    out.loc[modes.eq("neg_llr")] = -values
    out.loc[modes.eq("raw_llr")] = values
    out.loc[modes.eq("abs_llr")] = values.abs()
    selected = modes.eq("selected_other")
    if selected.any():
        alt_selected = frame.selected_allele.eq(frame.alt) & frame.other_allele.eq(
            frame.ref
        )
        ref_selected = frame.selected_allele.eq(frame.ref) & frame.other_allele.eq(
            frame.alt
        )
        if (selected & ~(alt_selected | ref_selected)).any():
            raise ValueError("selected_other alleles disagree with REF/ALT")
        out.loc[selected & alt_selected] = values
        out.loc[selected & ref_selected] = -values
    if not modes.isin(["neg_llr", "raw_llr", "abs_llr", "selected_other"]).all():
        raise ValueError("unsupported score_mode")
    return out


def calibrate(
    frame: pd.DataFrame,
    calibration: pd.DataFrame,
    reference: Reference,
    model_id: str,
    model_revision: str,
) -> pd.DataFrame:
    required = {"variant_id", "chrom", "pos", "ref", "alt", "raw_llr", "score_mode"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"missing input columns: {sorted(required - set(frame.columns))}"
        )
    if frame.variant_id.duplicated().any():
        raise ValueError("duplicate variant_id; combine/validate shards first")
    for key, expected in [("model_id", model_id), ("model_revision", model_revision)]:
        if key not in frame or set(frame[key].dropna().astype(str)) != {expected}:
            raise ValueError(f"input {key} does not match the supplied checkpoint")
    if not {"pentanuc_mut", "llr_neutral_mean"}.issubset(calibration):
        raise ValueError("not an official GPN-Star LLR calibration table")
    if calibration.pentanuc_mut.duplicated().any():
        raise ValueError("duplicate calibration lookup keys")
    lookup = calibration.set_index("pentanuc_mut").llr_neutral_mean
    if not np.isfinite(pd.to_numeric(lookup, errors="coerce")).all():
        raise ValueError("nonfinite calibration values")
    out = frame.copy()
    out["raw_llr"] = pd.to_numeric(out.raw_llr, errors="coerce")
    contexts, neutral, statuses = [], [], []
    for row in out.itertuples(index=False):
        context, value, status = "", np.nan, "PASS"
        if str(getattr(row, "status", "PASS")) != "PASS" or not np.isfinite(
            row.raw_llr
        ):
            status = "RAW_SCORE_UNAVAILABLE"
        else:
            try:
                if (
                    row.ref not in "ACGT"
                    or row.alt not in "ACGT"
                    or len(str(row.ref)) != 1
                    or len(str(row.alt)) != 1
                    or row.ref == row.alt
                ):
                    raise ValueError("not_biallelic_acgt_snv")
                context = reference.context(str(row.chrom), int(row.pos))
                if len(context) != 5 or context[2] != row.ref:
                    raise ValueError("reference_mismatch")
                key = context + "_" + row.alt
                if key not in lookup:
                    raise ValueError("calibration_context_unavailable")
                value = float(lookup[key])
            except (ValueError, KeyError, TypeError) as exc:
                status = str(exc)
        contexts.append(context)
        neutral.append(value)
        statuses.append(status)
    out["pentanucleotide"] = contexts
    out["llr_neutral_mean"] = neutral
    out["calibration_status"] = statuses
    out["calibrated_llr"] = out.raw_llr - out.llr_neutral_mean
    out["raw_benchmark_score"] = orient(out.raw_llr, out)
    out["calibrated_benchmark_score"] = orient(out.calibrated_llr, out)
    return out


def evaluate(frame: pd.DataFrame) -> dict:
    eligible = frame.calibration_status.eq("PASS")
    result = {
        "input_rows": len(frame),
        "raw_available": int(np.isfinite(frame.raw_llr).sum()),
        "calibrated_rows": int(eligible.sum()),
        "coverage": float(eligible.mean()),
        "calibration_status_counts": frame.calibration_status.value_counts().to_dict(),
        "comparison": "raw and cLLR on the identical calibrated subset; descriptive point estimates",
    }
    if "label" not in frame:
        return result
    labelled = pd.to_numeric(frame.label, errors="coerce")
    use = eligible & labelled.isin([0, 1])
    y = labelled.loc[use].to_numpy(int)
    result.update(
        num_samples=len(y), num_positive=int(y.sum()), num_negative=int((y == 0).sum())
    )
    if len(np.unique(y)) == 2:
        for name, column in [
            ("raw", "raw_benchmark_score"),
            ("cllr", "calibrated_benchmark_score"),
        ]:
            values = frame.loc[use, column].to_numpy(float)
            result[name + "_auprc"] = float(average_precision_score(y, values))
            result[name + "_auroc"] = float(roc_auc_score(y, values))
        result["delta_auprc"] = result["cllr_auprc"] - result["raw_auprc"]
        result["delta_auroc"] = result["cllr_auroc"] - result["raw_auroc"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration-table", type=Path, required=True)
    parser.add_argument(
        "--calibration-sha256",
        required=True,
        help="Hash independently verified against the pinned official artifact",
    )
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("refusing to overwrite primary raw scores")
    if sha256(args.calibration_table) != args.calibration_sha256:
        raise ValueError("calibration artifact hash mismatch")
    raw = pd.read_csv(
        args.input, sep="\t", dtype={"chrom": str, "ref": str, "alt": str}
    )
    reference = Reference(args.reference_fasta)
    try:
        output = calibrate(
            raw,
            pd.read_parquet(args.calibration_table),
            reference,
            args.model_id,
            args.model_revision,
        )
    finally:
        reference.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    manifest = evaluate(output)
    manifest.update(
        model=args.model_id,
        revision=args.model_revision,
        input=str(args.input.resolve()),
        input_sha256=sha256(args.input),
        calibration_table=str(args.calibration_table.resolve()),
        calibration_sha256=args.calibration_sha256,
        reference_fasta=str(args.reference_fasta.resolve()),
        reference_fai_sha256=sha256(Path(str(args.reference_fasta) + ".fai")),
        output_sha256=sha256(args.output),
        method_url="https://gpn.readthedocs.io/en/latest/_notebooks/gpn_star_demo/",
    )
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
