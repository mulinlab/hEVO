#!/usr/bin/env python3
"""Validate the GitLab HAR/HAQER source bundle without inventing tracks."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

REQUIRED = {
    "g_score": "g/HAR_NPAS3_optimized_with_tracks.smoothed_scores.npz",
    "g_peaks": "g/HAR_NPAS3_optimized_with_tracks.peaks.tsv",
    "g_overlap": "g/HAR_NPAS3_optimized_with_tracks.overlap_summary.json",
    "g_known_har": "g/HARs.CSV",
    "h_score": "h/val_region_29_normal.smoothed_scores.npz",
    "h_peaks": "h/val_region_29_normal.peaks.tsv",
    "h_overlap": "h/HAQER_top1_chr19_0_1000000_normal_with_ucsc.overlap_summary.json",
    "h_known_haqer": "h/haqer.hg38.bed",
    "g_ape": "g/g_ape_divergence_hg38.tsv",
    "h_ape": "h/h_ape_divergence_hg38.tsv",
    "g_gnocchi": "g/g_gnocchi_GRCh38_1kb.tsv",
    "h_gnocchi": "h/h_gnocchi_GRCh38_1kb.tsv",
    "g_gerp": "g/g_gerp_release101_GRCh38.tsv",
    "h_gerp": "h/h_gerp_release101_GRCh38.tsv",
    "g_linsight": "g/g_linsight_hg19_to_hg38.tsv",
    "h_linsight": "h/h_linsight_hg19_to_hg38.tsv",
    "g_cadd": "g/HAR_NPAS3_CADD_GRCh38_v1.7_binned_track.tsv",
    "h_cadd": "h/HAQER_chr19_CADD_GRCh38_v1.7_binned_track.tsv",
}
OPTIONAL = {
    "h3k27ac": "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig",
    "phastcons": "external_tracks/ucsc/hg38.phastCons100way.bw",
    "ccre_cage": "external_tracks/ucsc/encodeCcreCombined.hg38.bb",
    "refseq": "reference/hg38_refGene.txt.gz",
}
MISSING_TRACKS = ("ATAC-seq", "Human-ape delta", "CAGE", "Human-branch phyloP")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def validate(bundle: Path) -> dict:
    missing = [rel for rel in REQUIRED.values() if not (bundle / rel).is_file()]
    empty = [
        rel
        for rel in REQUIRED.values()
        if (bundle / rel).is_file() and (bundle / rel).stat().st_size == 0
    ]
    malformed = []
    for key in ("g_overlap", "h_overlap"):
        rel = REQUIRED[key]
        p = bundle / rel
        if p.is_file():
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                malformed.append(f"{rel}: {exc}")
    for key in ("g_score", "h_score"):
        rel = REQUIRED[key]
        p = bundle / rel
        if p.is_file():
            try:
                with np.load(p, allow_pickle=False) as archive:
                    if not archive.files:
                        malformed.append(f"{rel}: no arrays")
            except Exception as exc:
                malformed.append(f"{rel}: {exc}")
    return {
        "schema_version": "har_haqer_bundle_validation_v1",
        "bundle": str(bundle.resolve()),
        "status": "SOURCE_BUNDLE_VALID_INCOMPLETE_PREVIEW"
        if not (missing or empty or malformed)
        else "BUNDLE_ERROR",
        "required_file_count": len(REQUIRED),
        "required_files_present": len(REQUIRED) - len(missing),
        "missing_required": missing,
        "empty_required": empty,
        "malformed_required": malformed,
        "required_sha256": {
            key: sha256(bundle / rel)
            for key, rel in REQUIRED.items()
            if (bundle / rel).is_file()
        },
        "known_missing_external_tracks": list(MISSING_TRACKS),
        "external_dependencies": {
            key: {
                "path": rel,
                "present": (bundle / rel).is_file(),
                "sha256": sha256(bundle / rel) if (bundle / rel).is_file() else None,
            }
            for key, rel in OPTIONAL.items()
        },
        "external_dependencies_not_copied": {
            key: rel for key, rel in OPTIONAL.items() if not (bundle / rel).is_file()
        },
        "notes": [
            "No values are zero-filled or synthesized.",
            "ATAC and human-ape delta remain blocked until real quantitative tracks are produced.",
            "This validates figure source inputs; it does not establish five-replicate training provenance or accession-level merge history.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tables/har_haqer_tracks",
    )
    p.add_argument("--json-output", type=Path)
    args = p.parse_args()
    result = validate(args.bundle_dir.resolve())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["status"] == "BUNDLE_ERROR" else 0)


if __name__ == "__main__":
    main()
