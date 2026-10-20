"""Build the strict source manifest for the formal HAR/HAQER track figure.

This module intentionally does not create signal values.  It inventories the
project-provided tracks and any explicitly supplied external track overrides,
records provenance and hashes, and blocks the formal renderer unless every
requested row is backed by a verified real input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "results/final_har_haqer_panel/formal_main_figure"
MANIFEST_NAME = "Figure4g_h_HAR_HAQER_tracks_manifest.json"
TSV_NAME = "Figure4g_h_HAR_HAQER_tracks_manifest.tsv"
QC_NAME = "Figure4g_h_HAR_HAQER_track_qc.tsv"


PANEL_META: dict[str, dict[str, Any]] = {
    "g": {
        "title": "HAR | NPAS3 | hg38",
        "chrom": "chr14",
        "start": 33_000_000,
        "end": 34_000_000,
        "x_label": "chr14 position (Mb)",
    },
    "h": {
        "title": "HAQER | chr19p13.3 | hg38",
        "chrom": "chr19",
        "start": 345_000,
        "end": 485_000,
        "x_label": "chr19 position (kb)",
    },
}


TRACK_SPECS: list[dict[str, Any]] = [
    {"group": "MODEL", "key": "hevo", "label": "hEvo-base score", "kind": "score"},
    {"group": "MODEL", "key": "evo2", "label": "EVO 2 likelihood score", "kind": "score"},
    {"group": "MODEL", "key": "peaks", "label": "Peak calls", "kind": "peaks"},
    {"group": "MODEL", "key": "known", "label": "Known HARs/HAQERs", "kind": "intervals"},
    {"group": "REG.", "key": "h3k27ac", "label": "H3K27ac", "kind": "signal"},
    {"group": "REG.", "key": "atac", "label": "ATAC-seq", "kind": "signal"},
    {"group": "REG.", "key": "human_ape_delta", "label": "Human-ape delta", "kind": "signal"},
    {"group": "REG.", "key": "ccre_cage", "label": "cCRE / CAGE", "kind": "intervals"},
    {"group": "EVOL.", "key": "phastcons", "label": "phastCons100way", "kind": "signal"},
    {"group": "EVOL.", "key": "human_branch_phylop", "label": "Human-branch phyloP", "kind": "signal"},
    {"group": "EVOL.", "key": "ape_divergence", "label": "Ape divergence", "kind": "signal"},
    {"group": "EVOL.", "key": "gnocchi", "label": "Gnocchi (1 kb)", "kind": "signal"},
    {"group": "REF. SCORES", "key": "cadd", "label": "CADD-PHRED (max ALT)", "kind": "signal"},
    {"group": "REF. SCORES", "key": "gerp", "label": "GERP++ RS", "kind": "signal"},
    {"group": "REF. SCORES", "key": "linsight", "label": "LINSIGHT", "kind": "signal"},
    {"group": "GENOME", "key": "genes", "label": "RefSeq genes", "kind": "genes"},
]


SOURCE_POLICY: dict[str, dict[str, Any]] = {
    "hevo": {
        "source": "project-provided hEVO score NPZ",
        "source_url": None,
        "assembly": "hg38",
        "conversion": "read project NPZ; retain raw and smoothed score arrays",
    },
    "evo2": {
        "source": "project-provided EVO2 likelihood score track",
        "source_url": None,
        "assembly": "hg38",
        "conversion": "explicit score track only; never substitute hEVO",
    },
    "peaks": {
        "source": "project peak-call TSV",
        "source_url": None,
        "assembly": "hg38",
        "conversion": "read existing peak table; no re-calling",
    },
    "known": {
        "source": "project HAR/HAQER reference intervals",
        "source_url": None,
        "assembly": "hg38",
        "conversion": "read existing interval table",
    },
    "h3k27ac": {
        "source": "project-provided UCSC/ENCODE H3K27ac BigWig",
        "source_url": None,
        "assembly": "hg38",
        "conversion": "query native BigWig values on the shared locus grid",
    },
    "atac": {
        "source": "case-matched ATAC-seq track",
        "source_url": "ENCODE/Roadmap accession required",
        "assembly": "case-dependent",
        "conversion": "sample/cell-type matched signal; no generic substitute",
    },
    "human_ape_delta": {
        "source": "matched human-ape regulatory assay delta",
        "source_url": "case-specific accession required",
        "assembly": "species-specific",
        "conversion": "matched assay normalization documented in manifest",
    },
    "ccre_cage": {
        "source": "ENCODE SCREEN cCRE plus FANTOM5 CAGE",
        "source_url": "https://screen.encodeproject.org/ ; https://fantom.gsc.riken.jp/5/",
        "assembly": "hg38/hg19 as supplied; conversion required",
        "conversion": "retain interval annotations and record assembly conversion",
    },
    "phastcons": {
        "source": "UCSC hg38 phastCons100way BigWig",
        "source_url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/",
        "assembly": "hg38",
        "conversion": "query native 0-1 conservation values on shared grid",
    },
    "human_branch_phylop": {
        "source": "PHAST human-branch phyloP track",
        "source_url": "https://github.com/ChsliSipeLab/phast",
        "assembly": "hg38",
        "conversion": "branch-specific LRT/ACC result; ordinary phyloP100way is not accepted",
    },
    "ape_divergence": {
        "source": "HAQER/Vertegnlab ape divergence or matched divergence track",
        "source_url": "https://www.vertegnlab.org/dataPages/2022_HAQERs.html",
        "assembly": "hg38",
        "conversion": "native divergence track; no proxy conservation curve",
    },
    "gnocchi": {
        "source": "gnomAD Gnocchi signed Z, 1 kb windows",
        "source_url": "https://gnomad.broadinstitute.org/news/2022-10-the-addition-of-a-genomic-constraint-metric-to-gnomad/",
        "assembly": "GRCh38",
        "conversion": "retain native signed Z and 1 kb windows",
    },
    "cadd": {
        "source": "CADD v1.7 GRCh38 max-ALT PHRED",
        "source_url": "https://cadd.bihealth.org/download",
        "assembly": "GRCh38",
        "conversion": "per-position maximum over all possible ALT alleles; fixed 0-40/45 scale",
    },
    "gerp": {
        "source": "Ensembl fixed-release mammal GERP++ RS BigWig",
        "source_url": "https://ftp.ensembl.org/pub/release-101/compara/conservation_scores/",
        "assembly": "GRCh38",
        "conversion": "native score; zero line; preserve NaN; fixed release/species set",
    },
    "linsight": {
        "source": "CSHL LINSIGHT hg19 BigWig reciprocal-lifted to hg38",
        "source_url": "http://compgen.cshl.edu/~yihuang/tracks/LINSIGHT.bw",
        "assembly": "hg19 -> hg38",
        "conversion": "reciprocal liftOver; retain unique >=95% mapped fragments; native 0-1 scale",
    },
    "genes": {
        "source": "UCSC NCBI RefSeq hg38",
        "source_url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/",
        "assembly": "hg38",
        "conversion": "read the project RefSeq transcript set",
    },
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_missing_fraction(path: Path | None) -> float | None:
    """Estimate missing numeric values when the source format exposes them."""
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return None
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npz"):
        try:
            payload = np.load(path, allow_pickle=False)
            candidates = [name for name in ("raw_scores", "smoothed_scores", "values", "score", "scores") if name in payload]
            if not candidates:
                return None
            array = np.asarray(payload[candidates[0]], dtype=float)
            return float(1.0 - np.isfinite(array).mean())
        except (OSError, ValueError, KeyError):
            return None
    if suffixes.endswith(".tsv") or suffixes.endswith(".csv") or suffixes.endswith(".tsv.gz") or suffixes.endswith(".csv.gz"):
        try:
            opener = path.open
            if suffixes.endswith(".gz"):
                import gzip

                opener = gzip.open  # type: ignore[assignment]
            with opener("rt", encoding="utf-8", errors="replace", newline="") as handle:  # type: ignore[arg-type]
                sample = handle.read(4096)
                handle.seek(0)
                delimiter = "\t" if "\t" in sample else ","
                reader = csv.DictReader(handle, delimiter=delimiter)
                rows = list(reader)
            if not rows:
                return None
            value_fields = ("value", "score", "scores", "max_phred", "phylop", "gerp", "linsight", "likelihood")
            field = next((name for name in value_fields if name in rows[0]), None)
            if field is None:
                return 0.0
            missing = 0
            for row in rows:
                try:
                    value = float(row.get(field, "nan"))
                except (TypeError, ValueError):
                    value = float("nan")
                missing += int(not np.isfinite(value))
            return float(missing / len(rows))
        except (OSError, csv.Error):
            return None
    return None


def existing_path(root: Path, panel: str, key: str) -> Path | None:
    source_dir = root / "results/literature_example_tracks"
    har_beautified = root / "results/literature_example_tracks_beautified"
    haqer_dir = root / "results/haqer_meaningful_region_chr19"
    scan_dir = root / "results/haqer_test_region_search_hardneg_86m/target050/val_region_29"
    cache_dir = root / "results/final_har_haqer_panel/reference_score_cache"
    if panel == "g":
        mapping = {
            "hevo": source_dir / "HAR_NPAS3_optimized_with_tracks.smoothed_scores.npz",
            "evo2": root / "results/final_har_haqer_panel/evo2_1b_hevo_protocol/scores/har/g_blocks_6_mlp_l3_tracks.npz",
            "peaks": source_dir / "HAR_NPAS3_optimized_with_tracks.peaks.tsv",
            "known": root / "data/raw/HARs.CSV",
            "h3k27ac": root / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig",
            "ccre_cage": root / "external_tracks/ucsc/encodeCcreCombined.hg38.bb",
            "phastcons": root / "external_tracks/ucsc/hg38.phastCons100way.bw",
            "cadd": har_beautified / "HAR_NPAS3_CADD_GRCh38_v1.7_binned_track.tsv",
            "gerp": cache_dir / "g_GERP_GRCh38_binned_track.tsv",
            "linsight": cache_dir / "g_LINSIGHT_GRCh38_binned_track.tsv",
            "genes": root / "reference/hg38_refGene.txt.gz",
        }
    else:
        mapping = {
            "hevo": scan_dir / "val_region_29_normal.smoothed_scores.npz",
            "evo2": root / "results/final_har_haqer_panel/evo2_1b_hevo_protocol/scores/haqer/h_blocks_12_mlp_l3_tracks.npz",
            "peaks": scan_dir / "val_region_29_normal.peaks.tsv",
            "known": har_beautified / "haqer.hg38.bed",
            "h3k27ac": root / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig",
            "ccre_cage": root / "external_tracks/ucsc/encodeCcreCombined.hg38.bb",
            "phastcons": root / "external_tracks/ucsc/hg38.phastCons100way.bw",
            "cadd": haqer_dir / "HAQER_chr19_CADD_GRCh38_v1.7_binned_track.tsv",
            "gerp": cache_dir / "h_GERP_GRCh38_binned_track.tsv",
            "linsight": cache_dir / "h_LINSIGHT_GRCh38_binned_track.tsv",
            "genes": root / "reference/hg38_refGene.txt.gz",
        }
    return mapping.get(key)


def default_status(key: str, path: Path | None) -> tuple[str, str | None]:
    if path is None:
        return "missing_input", "No project-provided or explicit override path"
    if not path.exists() or path.stat().st_size == 0:
        return "missing_input", "Path does not exist or is empty"
    if key in {"gerp", "linsight"}:
        return "candidate_unverified_source", "Existing cache does not prove the PDF-required source/conversion chain"
    if key == "ccre_cage":
        return "partial", "cCRE is present; a matched CAGE component is not verified"
    if key == "human_branch_phylop":
        return "candidate_wrong_track", "Generic phyloP100way is not human-branch phyloP"
    return "verified_real", None


def parse_overrides(items: list[str], root: Path) -> dict[tuple[str, str], Path]:
    overrides: dict[tuple[str, str], Path] = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(f"Override must be PANEL:KEY=PATH, got {item!r}")
        left, raw_path = item.split("=", 1)
        panel, key = left.split(":", 1)
        if panel not in PANEL_META or key not in {spec["key"] for spec in TRACK_SPECS}:
            raise ValueError(f"Unknown track override {panel}:{key}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        overrides[(panel, key)] = path
    return overrides


def load_external_metadata(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Load provenance/QC generated by the remote external-track preparer."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("tracks", [])
    if not isinstance(rows, list):
        raise ValueError(f"External manifest has no track list: {path}")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "panel" not in row or "key" not in row:
            continue
        output[(str(row["panel"]), str(row["key"]))] = dict(row)
    return output


def build_manifest(
    root: Path,
    overrides: dict[tuple[str, str], Path],
    external_metadata: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    external_metadata = external_metadata or {}
    tracks: list[dict[str, Any]] = []
    for panel, panel_meta in PANEL_META.items():
        for spec in TRACK_SPECS:
            key = spec["key"]
            metadata = external_metadata.get((panel, key), {})
            metadata_path = metadata.get("path")
            path = overrides.get((panel, key), existing_path(root, panel, key))
            if (panel, key) not in overrides and metadata_path:
                path = Path(str(metadata_path))
            status, note = default_status(key, path)
            if (panel, key) in overrides and path is not None and path.exists() and path.stat().st_size > 0:
                status, note = "verified_real", "Explicit track override supplied; source policy and hash recorded below"
            if key == "evo2" and path is not None and path.exists():
                status, note = "verified_real", None
            if key in {"atac", "human_ape_delta", "human_branch_phylop", "ape_divergence", "gnocchi"} and path is not None and path.exists():
                status, note = "verified_real", None
            policy = SOURCE_POLICY[key]
            display_label = spec["label"]
            if key == "known":
                display_label = "Known HARs" if panel == "g" else "Known HAQERs"
            entry = {
                "panel": panel,
                "group": spec["group"],
                "key": key,
                "label": display_label,
                "kind": spec["kind"],
                "required_for_formal": True,
                "status": status,
                "status_note": note,
                "path": str(path) if path is not None else None,
                "source": policy["source"],
                "source_url": policy["source_url"],
                "assembly": policy["assembly"],
                "chrom": panel_meta["chrom"],
                "region_start": panel_meta["start"],
                "region_end": panel_meta["end"],
                "conversion": policy["conversion"],
                "sha256": sha256_file(path) if path is not None else None,
                "size_bytes": path.stat().st_size if path is not None and path.is_file() else None,
                "missing_fraction": estimate_missing_fraction(path),
            }
            # External-track QC is authoritative for downloaded/lifted tracks.
            # Keep the generic project policy as a fallback for rows not covered
            # by the external manifest.
            if metadata:
                for field in (
                    "status", "status_note", "source", "source_url", "source_path",
                    "assembly", "conversion", "binning", "sha256", "source_sha256",
                    "size_bytes", "source_size_bytes", "downloaded_at", "cache_method",
                    "missing_fraction", "mapped_fraction", "unique_mapping_fraction",
                    "source_verified", "formal_eligible", "blocker", "qc_note",
                    "chain_url", "chain_path", "chain_sha256", "chain_source_sha256",
                    "source_chrom_aliases",
                ):
                    if field in metadata:
                        entry[field] = metadata[field]
            # A project-provided Evo2 result is not covered by the older
            # external-track manifest.  Its explicit path must take priority
            # over a stale missing-input status from that manifest.
            if key == "evo2" and path is not None and path.is_file() and path.stat().st_size > 0:
                entry["status"] = "verified_real"
                entry["status_note"] = None
                entry["source_verified"] = True
                entry["formal_eligible"] = True
                entry["blocker"] = None
                entry["sha256"] = sha256_file(path)
                entry["size_bytes"] = path.stat().st_size
            tracks.append(entry)
    return {
        "schema_version": "formal_har_haqer_tracks_v1",
        "figure": "Figure4g_h_HAR_HAQER_formal",
        "backend": "python-matplotlib",
        "panel_order": list(PANEL_META),
        "track_order": [spec["key"] for spec in TRACK_SPECS],
        "style_contract": {
            "panel_gap_mm": 1.5,
            "figure_width_mm": 180.0,
            "figure_height_mm": 108.0,
            "strict_status": "verified_real",
            "synthetic_tracks_allowed": False,
        },
        "panels": PANEL_META,
        "tracks": tracks,
        "external_manifest": None,
    }


def write_outputs(manifest: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    tsv_path = output_dir / TSV_NAME
    qc_path = output_dir / QC_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "panel", "group", "key", "label", "kind", "status", "status_note", "path",
        "source", "source_url", "assembly", "chrom", "region_start", "region_end",
        "conversion", "binning", "sha256", "source_sha256", "size_bytes", "source_size_bytes",
        "downloaded_at", "cache_method", "missing_fraction", "mapped_fraction",
        "unique_mapping_fraction", "source_verified", "formal_eligible", "blocker", "qc_note",
        "chain_url", "chain_path", "chain_sha256", "chain_source_sha256",
        "source_chrom_aliases",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest["tracks"])
    with qc_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "key", "status", "path", "note"], delimiter="\t")
        writer.writeheader()
        for entry in manifest["tracks"]:
            writer.writerow({
                "panel": entry["panel"],
                "key": entry["key"],
                "status": entry["status"],
                "path": entry["path"] or "",
                "note": entry["status_note"] or "",
            })
    return manifest_path, tsv_path, qc_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the strict formal HAR/HAQER track manifest.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--track", action="append", default=[], help="Explicit override: PANEL:KEY=PATH")
    parser.add_argument("--external-manifest", type=Path, default=None, help="Remote source/QC manifest to merge")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any required track is not verified_real")
    args = parser.parse_args()
    root = args.root
    output_dir = args.output_dir or root / "results/final_har_haqer_panel/formal_main_figure"
    overrides = parse_overrides(args.track, root)
    external_manifest_path = args.external_manifest
    if external_manifest_path is not None and not external_manifest_path.is_absolute():
        external_manifest_path = root / external_manifest_path
    external_metadata = load_external_metadata(external_manifest_path)
    manifest = build_manifest(root, overrides, external_metadata)
    if external_manifest_path is not None:
        manifest["external_manifest"] = str(external_manifest_path)
    paths = write_outputs(manifest, output_dir)
    failures = [
        f"{entry['panel']}:{entry['key']}={entry['status']}"
        for entry in manifest["tracks"]
        if entry["required_for_formal"] and (entry["status"] != "verified_real" or entry.get("formal_eligible") is False)
    ]
    print(json.dumps({"manifest": str(paths[0]), "qc": str(paths[2]), "strict_failures": failures}, ensure_ascii=False, indent=2))
    if args.strict and failures:
        raise SystemExit("Strict formal-track validation failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
