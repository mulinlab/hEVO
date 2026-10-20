"""Download and QC the real external tracks used by the formal figure.

This script is intended to run in the documented ``llmbm`` environment.
It deliberately keeps source downloads, derived locus excerpts, provenance,
and missing-input decisions separate from the plotting code.  BigWig servers
in the current environment do not support pyBigWig HTTP-range opening, so the
script records that fact and falls back to complete, hashable source files.

No representative ATAC, CAGE, human--ape delta, or ordinary phyloP track is
created.  Those rows remain ``missing_input`` until a matching accession or a
branch-specific PHAST input is supplied explicitly.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests


DEFAULT_ROOT = Path("<source-project-root>")
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "results/final_har_haqer_panel/formal_main_figure"
CACHE_NAME = "external_track_cache"
MANIFEST_NAME = "Figure4g_h_HAR_HAQER_external_tracks_manifest.json"
TSV_NAME = "Figure4g_h_HAR_HAQER_external_tracks_manifest.tsv"
QC_NAME = "Figure4g_h_HAR_HAQER_external_track_qc.tsv"
MISSING_JSON_NAME = "Figure4g_h_HAR_HAQER_missing_tracks.json"
MISSING_TSV_NAME = "Figure4g_h_HAR_HAQER_missing_tracks.tsv"


REGIONS: dict[str, dict[str, Any]] = {
    "g": {"chrom": "chr14", "start": 33_000_000, "end": 34_000_000},
    "h": {"chrom": "chr19", "start": 345_000, "end": 485_000},
}

HG38_CHROM_SIZES = {"chr14": 107_043_718, "chr19": 58_617_616}
HG19_CHROM_SIZES = {"chr14": 107_349_540, "chr19": 59_128_983}


SOURCES: dict[str, dict[str, Any]] = {
    "ape_divergence": {
        "url": "https://vertgenlab.cloud.duke.edu/publications/haqer/haqer.divergenceDensity.bw",
        "filename": "haqer.divergenceDensity.bw",
        "assembly": "hg38",
        "source": "Vertegnlab official HAQER divergence-density BigWig",
        "conversion": "native hg38 BigWig; fixed-bin mean query; no proxy conservation track",
    },
    "gnocchi": {
        "url": "https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1/secondary_analyses/genomic_constraint/constraint_z_genome_1kb.qc.download.txt.gz",
        "filename": "constraint_z_genome_1kb.qc.download.txt.gz",
        "assembly": "GRCh38",
        "source": "gnomAD v3.1.2 Gnocchi signed-Z 1 kb download",
        "conversion": "native 1 kb windows; z column retained; missing values remain NaN",
    },
    "gerp": {
        "url": "https://ftp.ensembl.org/pub/release-101/compara/conservation_scores/111_mammals.gerp_conservation_score/gerp_conservation_scores.homo_sapiens.GRCh38.bw",
        "filename": "gerp_conservation_scores.homo_sapiens.GRCh38.bw",
        "assembly": "GRCh38",
        "source": "Ensembl release 101 111-mammals GERP++ conservation score BigWig",
        "conversion": "native GERP++ RS; fixed-bin mean query; NaN preserved",
    },
    "linsight": {
        "url": "http://hgdownload.soe.ucsc.edu/gbdb/hg19/linsight/LINSIGHT.bw",
        "filename": "LINSIGHT.hg19.bw",
        "assembly": "hg19",
        "source": "UCSC mirror of the official CSHL LINSIGHT hg19 BigWig",
        "conversion": "hg19 source values mapped to hg38 with UCSC chain blocks; reciprocal round-trip QC; NaN preserved",
    },
}

# These are auxiliary, case-matched source candidates.  They are downloaded
# and hashed when available, but are not promoted to formal tracks unless the
# source contains the quantitative assay required by the PDF contract.
AUXILIARY_SOURCES: dict[str, dict[str, Any]] = {
    "gse222113_series_metadata": {
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE222113&targ=self&form=text&view=quick",
        "filename": "GSE222113_series_metadata.txt",
        "assembly": "mixed; see GEO metadata",
        "source": "NCBI GEO GSE222113 series metadata",
        "role": "case-matched HAR assay discovery and provenance",
    },
    "gse221830_human_atac_peaks": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE221nnn/GSE221830/suppl/GSE221830_Human_idr.optimal_peak.narrowPeak.gz",
        "filename": "GSE221830_Human_idr.optimal_peak.narrowPeak.gz",
        "assembly": "hg38",
        "source": "NCBI GEO GSE221830 human ATAC-seq optimal peaks",
        "role": "case-matched NPAS3/HAR ATAC candidate; peak-only, not quantitative signal",
    },
    "gse221830_chimp_atac_peaks": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE221nnn/GSE221830/suppl/GSE221830_Chimp_idr.optimal_peak.narrowPeak.gz",
        "filename": "GSE221830_Chimp_idr.optimal_peak.narrowPeak.gz",
        "assembly": "panTro5",
        "source": "NCBI GEO GSE221830 chimpanzee ATAC-seq optimal peaks",
        "role": "matched human-ape ATAC candidate; requires orthology/count workflow for delta",
    },
    "fantom5_datafiles_index": {
        "url": "https://fantom.gsc.riken.jp/5/datafiles/",
        "filename": "FANTOM5_datafiles_index.html",
        "assembly": "mixed; source index",
        "source": "FANTOM5 official datafiles index",
        "role": "CAGE source discovery; no guessed file URL is accepted",
    },
}

CHAIN_URL = (
    "https://hgdownload.soe.ucsc.edu/gbdb/hg19/liftOver/hg19ToHg38.over.chain.gz"
)
CHAIN_FILENAME = "hg19ToHg38.over.chain.gz"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_metadata(url: str) -> dict[str, Any]:
    try:
        response = requests.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        # A few large mirrors do not answer HEAD reliably.  A one-byte range
        # still gives the content metadata without downloading the source.
        response = requests.get(
            url,
            headers={"Range": "bytes=0-0"},
            allow_redirects=True,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()
    content_range = response.headers.get("content-range", "")
    range_size = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
    length = (
        range_size if range_size.isdigit() else response.headers.get("content-length")
    )
    return {
        "http_status": response.status_code,
        "final_url": str(response.url),
        "source_size_bytes": int(length) if length and length.isdigit() else None,
        "etag": response.headers.get("etag"),
        "last_modified": response.headers.get("last-modified"),
        "accept_ranges": response.headers.get("accept-ranges"),
    }


def parallel_range_download(
    url: str,
    destination: Path,
    expected: int,
    workers: int = 128,
    chunk_size: int = 4 * 1024 * 1024,
) -> str:
    """Fetch a complete source through validated HTTP Range requests."""
    part_dir = destination.with_name(destination.name + ".ranges")
    part_dir.mkdir(parents=True, exist_ok=True)
    part_count = int(math.ceil(expected / chunk_size))

    def fetch(index: int) -> Path:
        left = index * chunk_size
        right = min(expected - 1, left + chunk_size - 1)
        part_path = part_dir / f"{index:06d}.part"
        expected_part = right - left + 1
        if part_path.is_file() and part_path.stat().st_size == expected_part:
            return part_path
        if part_path.exists():
            part_path.unlink()
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={left}-{right}"},
                    allow_redirects=True,
                    stream=True,
                    timeout=(60, 900),
                )
                if response.status_code != 206:
                    response.close()
                    raise RuntimeError(
                        f"Range request {left}-{right} returned HTTP {response.status_code}, not 206"
                    )
                with part_path.open("wb") as handle:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if block:
                            handle.write(block)
                response.close()
                if part_path.stat().st_size != expected_part:
                    raise RuntimeError(
                        f"Range part {index} size mismatch: {part_path.stat().st_size} != {expected_part}"
                    )
                return part_path
            except (OSError, requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if part_path.exists():
                    part_path.unlink()
                time.sleep(min(30, 2**attempt))
        raise RuntimeError(f"Range part {index} failed after retries: {last_error}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, index) for index in range(part_count)]
        for future in as_completed(futures):
            future.result()

    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as output:
        for index in range(part_count):
            part_path = part_dir / f"{index:06d}.part"
            with part_path.open("rb") as handle:
                shutil.copyfileobj(handle, output, length=4 * 1024 * 1024)
    if temporary.stat().st_size != expected:
        raise RuntimeError(
            f"Combined range download size mismatch: {temporary.stat().st_size} != {expected}"
        )
    os.replace(temporary, destination)
    shutil.rmtree(part_dir)
    return "parallel_http_range_download_with_complete_source"


def download(url: str, destination: Path, force: bool = False) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        remote = head_metadata(url)
    except requests.RequestException as exc:
        if destination.is_file() and destination.stat().st_size > 0 and not force:
            return {
                "http_status": None,
                "final_url": url,
                "source_size_bytes": destination.stat().st_size,
                "etag": None,
                "last_modified": None,
                "accept_ranges": None,
                "head_error": f"{type(exc).__name__}: {exc}",
                "path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "downloaded_at": utc_now(),
                "cache_method": "reused_complete_file_head_timeout",
            }
        raise
    expected = remote.get("source_size_bytes")
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        if expected is None or destination.stat().st_size == expected:
            return {
                **remote,
                "path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "downloaded_at": utc_now(),
                "cache_method": "reused_complete_file",
            }
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    if (
        expected
        and expected > 32 * 1024 * 1024
        and str(remote.get("accept_ranges", "")).lower() == "bytes"
    ):
        range_workers = 16 if "hgdownload.soe.ucsc.edu" in url else 128
        cache_method = parallel_range_download(
            url, destination, int(expected), workers=range_workers
        )
        return {
            **remote,
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "downloaded_at": utc_now(),
            "cache_method": cache_method,
        }
    command = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--connect-timeout",
        "60",
        "--output",
        str(temporary),
        url,
    ]
    subprocess.run(command, check=True)
    if expected is not None and temporary.stat().st_size != expected:
        raise RuntimeError(
            f"Downloaded size mismatch for {url}: {temporary.stat().st_size} != {expected}"
        )
    os.replace(temporary, destination)
    return {
        **remote,
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "downloaded_at": utc_now(),
        "cache_method": "complete_download_after_HTTP_range_open_failed",
    }


def try_remote_bigwig(url: str) -> tuple[bool, str | None]:
    """Probe the documented range path without treating it as a formal source."""
    try:
        import pyBigWig

        handle = pyBigWig.open(url)
        chroms = handle.chroms()
        handle.close()
        if chroms:
            return True, None
        return False, "Remote BigWig opened without chromosome metadata"
    except Exception as exc:  # pyBigWig emits C-level diagnostics on failure.
        return False, f"HTTP-range BigWig open failed: {type(exc).__name__}: {exc}"


def bin_edges(start: int, end: int, n_bins: int) -> list[tuple[int, int]]:
    edges = np.linspace(start, end, n_bins + 1, dtype=np.int64)
    pairs: list[tuple[int, int]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        left_i, right_i = int(left), int(right)
        if right_i > left_i:
            pairs.append((left_i, right_i))
    return pairs


def resolve_chrom_name(chroms: dict[str, int], requested: str) -> str | None:
    candidates = [
        requested,
        requested.removeprefix("chr"),
        "chr" + requested.removeprefix("chr"),
    ]
    for candidate in candidates:
        if candidate in chroms:
            return candidate
    return None


def write_rows(
    path: Path, rows: Iterable[dict[str, Any]], fields: list[str]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    missing = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            count += 1
            value = row.get("value")
            try:
                finite = np.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            missing += int(not finite)
            writer.writerow(row)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": count,
        "missing_fraction": (missing / count) if count else None,
    }


def extract_bigwig(
    source_path: Path, panel: str, output_path: Path, n_bins: int = 1200
) -> dict[str, Any]:
    import pyBigWig

    region = REGIONS[panel]
    edges = bin_edges(int(region["start"]), int(region["end"]), n_bins)
    handle = pyBigWig.open(str(source_path))
    try:
        chroms = handle.chroms()
        source_chrom = resolve_chrom_name(chroms, str(region["chrom"]))
        if source_chrom is None:
            raise RuntimeError(
                f"{source_path} has no {region['chrom']} or its unprefixed alias"
            )
        if int(chroms[source_chrom]) != HG38_CHROM_SIZES[region["chrom"]]:
            raise RuntimeError(
                f"{source_path} chromosome length is not hg38 for {region['chrom']}: {chroms[source_chrom]}"
            )
        values = handle.stats(
            source_chrom,
            int(region["start"]),
            int(region["end"]),
            nBins=len(edges),
            exact=True,
        )
    finally:
        handle.close()
    rows = (
        {
            "chrom": region["chrom"],
            "start": left,
            "end": right,
            "value": (float(value) if value is not None else float("nan")),
        }
        for (left, right), value in zip(edges, values)
    )
    result = write_rows(output_path, rows, ["chrom", "start", "end", "value"])
    result.update(
        {
            "assembly": "hg38",
            "region": dict(region),
            "binning": f"{len(edges)} equal-width bins; exact=True",
        }
    )
    return result


def extract_gnocchi(source_path: Path, panel: str, output_path: Path) -> dict[str, Any]:
    region = REGIONS[panel]

    def rows() -> Iterable[dict[str, Any]]:
        with gzip.open(
            source_path, "rt", encoding="utf-8", errors="replace", newline=""
        ) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "z" not in reader.fieldnames:
                raise RuntimeError(f"Unexpected Gnocchi columns: {reader.fieldnames}")
            for row in reader:
                if row.get("chrom") != region["chrom"]:
                    continue
                try:
                    left = int(row["start"])
                    right = int(row["end"])
                    value = float(row["z"])
                except (KeyError, TypeError, ValueError):
                    value = float("nan")
                    try:
                        left, right = int(row["start"]), int(row["end"])
                    except (KeyError, TypeError, ValueError):
                        continue
                if right <= int(region["start"]) or left >= int(region["end"]):
                    continue
                yield {
                    "chrom": region["chrom"],
                    "start": left,
                    "end": right,
                    "value": value,
                }

    result = write_rows(output_path, rows(), ["chrom", "start", "end", "value"])
    result.update(
        {
            "assembly": "GRCh38",
            "region": dict(region),
            "binning": "native gnomAD Gnocchi 1 kb windows",
        }
    )
    return result


@dataclass(frozen=True)
class ChainBlock:
    t_chrom: str
    t_start: int
    t_end: int
    q_chrom: str
    q_size: int
    q_plus_start: int
    q_plus_end: int
    q_strand: str


def parse_chain(path: Path, target_chrom: str) -> list[ChainBlock]:
    blocks: list[ChainBlock] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        current: tuple[str, int, int, str, int, str] | None = None
        source_pos = 0
        target_pos = 0
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("chain "):
                parts = line.split()
                if len(parts) != 13:
                    raise RuntimeError(f"Unexpected chain header: {line}")
                source_chrom = parts[2]
                source_size = int(parts[3])
                source_start = int(parts[5])
                target_name = parts[7]
                target_size = int(parts[8])
                target_strand = parts[9]
                target_start = int(parts[10])
                source_aliases = {
                    target_chrom,
                    target_chrom.removeprefix("chr"),
                    "chr" + target_chrom.removeprefix("chr"),
                }
                if target_name == target_chrom and source_chrom in source_aliases:
                    if target_size != HG38_CHROM_SIZES[target_chrom]:
                        raise RuntimeError(
                            f"Chain q chromosome length is not hg38 for {target_chrom}: {target_size}"
                        )
                    if source_size != HG19_CHROM_SIZES[target_chrom]:
                        raise RuntimeError(
                            f"Chain t chromosome length is not hg19 for {target_chrom}: {source_size}"
                        )
                    current = (
                        source_chrom,
                        source_size,
                        source_start,
                        target_name,
                        target_size,
                        target_strand,
                    )
                else:
                    current = None
                source_pos = source_start
                target_pos = target_start
                continue
            if current is None:
                continue
            parts = line.split()
            size = int(parts[0])
            dt = int(parts[1]) if len(parts) > 1 else 0
            dq = int(parts[2]) if len(parts) > 2 else 0
            (
                source_chrom,
                source_size,
                _source_start,
                target_name,
                target_size,
                target_strand,
            ) = current
            if target_name == target_chrom:
                if target_strand == "+":
                    target_plus_start, target_plus_end = target_pos, target_pos + size
                else:
                    target_plus_start, target_plus_end = (
                        target_size - (target_pos + size),
                        target_size - target_pos,
                    )
                blocks.append(
                    ChainBlock(
                        target_name,
                        target_plus_start,
                        target_plus_end,
                        source_chrom,
                        source_size,
                        source_pos,
                        source_pos + size,
                        target_strand,
                    )
                )
            source_pos += size + dt
            target_pos += size + dq
    return blocks


def target_to_query(block: ChainBlock, left: int, right: int) -> tuple[int, int] | None:
    overlap_left = max(left, block.t_start)
    overlap_right = min(right, block.t_end)
    if overlap_right <= overlap_left:
        return None
    if block.q_strand == "+":
        q_left = block.q_plus_start + (overlap_left - block.t_start)
        q_right = block.q_plus_start + (overlap_right - block.t_start)
    else:
        q_left = block.q_plus_end - (overlap_right - block.t_start)
        q_right = block.q_plus_end - (overlap_left - block.t_start)
    return q_left, q_right


def query_to_target(block: ChainBlock, left: int, right: int) -> tuple[int, int] | None:
    overlap_left = max(left, block.q_plus_start)
    overlap_right = min(right, block.q_plus_end)
    if overlap_right <= overlap_left:
        return None
    if block.q_strand == "+":
        t_left = block.t_start + (overlap_left - block.q_plus_start)
        t_right = block.t_start + (overlap_right - block.q_plus_start)
    else:
        t_left = block.t_start + (block.q_plus_end - overlap_right)
        t_right = block.t_start + (block.q_plus_end - overlap_left)
    return t_left, t_right


def extract_linsight(
    source_path: Path,
    chain_path: Path,
    panel: str,
    output_path: Path,
    n_bins: int = 1400,
) -> dict[str, Any]:
    import pyBigWig

    region = REGIONS[panel]
    blocks = [
        block
        for block in parse_chain(chain_path, str(region["chrom"]))
        if block.t_end > int(region["start"]) and block.t_start < int(region["end"])
    ]
    if not blocks:
        raise RuntimeError(f"No hg19ToHg38 chain blocks for {region['chrom']}")
    edges = bin_edges(int(region["start"]), int(region["end"]), n_bins)
    handle = pyBigWig.open(str(source_path))
    mapped_bins = 0
    covered_bp = 0
    duplicate_bins = 0
    rows: list[dict[str, Any]] = []
    try:
        source_chroms = handle.chroms()
        source_aliases = {}
        for block in blocks:
            alias = resolve_chrom_name(source_chroms, block.q_chrom)
            if alias is None:
                raise RuntimeError(
                    f"{source_path} has no hg19 source chromosome alias for {block.q_chrom}"
                )
            if int(source_chroms[alias]) != HG19_CHROM_SIZES[region["chrom"]]:
                raise RuntimeError(
                    f"{source_path} chromosome length is not hg19 for {block.q_chrom}: {source_chroms[alias]}"
                )
            source_aliases[block.q_chrom] = alias
        for left, right in edges:
            matches = [
                block
                for block in blocks
                if block.t_end > left and block.t_start < right
            ]
            mapped = []
            for block in matches:
                query_interval = target_to_query(block, left, right)
                if query_interval is not None:
                    mapped.append((block, query_interval))
            target_covered = sum(
                min(right, block.t_end) - max(left, block.t_start)
                for block, _ in mapped
            )
            covered_bp += max(0, target_covered)
            if len(mapped) != 1:
                duplicate_bins += int(len(mapped) > 1)
                rows.append(
                    {
                        "chrom": region["chrom"],
                        "start": left,
                        "end": right,
                        "value": float("nan"),
                        "source_start": "",
                        "source_end": "",
                        "roundtrip_fraction": 0.0,
                    }
                )
                continue
            block, (q_left, q_right) = mapped[0]
            source_chrom = source_aliases[block.q_chrom]
            target_fraction = target_covered / max(1, right - left)
            roundtrip = query_to_target(block, q_left, q_right)
            roundtrip_fraction = 0.0
            if roundtrip is not None:
                rt_left, rt_right = roundtrip
                roundtrip_fraction = max(
                    0, min(right, rt_right) - max(left, rt_left)
                ) / max(1, right - left)
            if target_fraction < 0.95 or roundtrip_fraction < 0.95:
                value = float("nan")
            else:
                stats = handle.stats(source_chrom, q_left, q_right, nBins=1, exact=True)
                value = (
                    float(stats[0]) if stats and stats[0] is not None else float("nan")
                )
                mapped_bins += 1
            rows.append(
                {
                    "chrom": region["chrom"],
                    "start": left,
                    "end": right,
                    "value": value,
                    "source_start": q_left,
                    "source_end": q_right,
                    "roundtrip_fraction": roundtrip_fraction,
                }
            )
    finally:
        handle.close()
    result = write_rows(
        output_path,
        rows,
        [
            "chrom",
            "start",
            "end",
            "value",
            "source_start",
            "source_end",
            "roundtrip_fraction",
        ],
    )
    total_bp = int(region["end"]) - int(region["start"])
    result.update(
        {
            "assembly": "hg19 -> hg38",
            "region": dict(region),
            "binning": f"{len(edges)} target hg38 bins; unique >=95% target coverage and reciprocal round-trip",
            "mapped_fraction": covered_bp / max(1, total_bp),
            "unique_mapping_fraction": mapped_bins / max(1, len(edges)),
            "duplicate_bin_count": duplicate_bins,
            "chain_block_count": len(blocks),
            "source_chrom_aliases": source_aliases,
        }
    )
    return result


def source_base(
    key: str, download_info: dict[str, Any], derived: dict[str, Any], panel: str
) -> dict[str, Any]:
    source = SOURCES[key]
    eligible = (
        bool(derived.get("row_count", 0))
        and bool(download_info.get("sha256"))
        and bool(derived.get("sha256"))
    )
    return {
        "panel": panel,
        "key": key,
        "path": derived["path"],
        "source_path": download_info["path"],
        "source": source["source"],
        "source_url": download_info.get("final_url", source["url"]),
        "assembly": derived.get("assembly", source["assembly"]),
        "region_start": REGIONS[panel]["start"],
        "region_end": REGIONS[panel]["end"],
        "conversion": source["conversion"],
        "binning": derived.get("binning"),
        "sha256": derived["sha256"],
        "source_sha256": download_info["sha256"],
        "size_bytes": derived["size_bytes"],
        "source_size_bytes": download_info["size_bytes"],
        "downloaded_at": download_info["downloaded_at"],
        "cache_method": download_info["cache_method"],
        "missing_fraction": derived.get("missing_fraction"),
        "mapped_fraction": derived.get("mapped_fraction"),
        "unique_mapping_fraction": derived.get("unique_mapping_fraction"),
        "source_chrom_aliases": derived.get("source_chrom_aliases"),
        "source_verified": True,
        "formal_eligible": eligible,
        "status": "verified_real" if eligible else "no_coverage",
        "status_note": None
        if eligible
        else "The source opened but produced no locus rows",
        "blocker": None if eligible else "No locus rows were extracted",
        "qc_note": None,
    }


def missing_entry(
    panel: str, key: str, blocker: str, source_url: str | None = None
) -> dict[str, Any]:
    return {
        "panel": panel,
        "key": key,
        "path": None,
        "source_path": None,
        "source": "Required source not supplied",
        "source_url": source_url,
        "assembly": REGIONS[panel].get("assembly", "hg38"),
        "region_start": REGIONS[panel]["start"],
        "region_end": REGIONS[panel]["end"],
        "conversion": None,
        "binning": None,
        "sha256": None,
        "source_sha256": None,
        "size_bytes": None,
        "source_size_bytes": None,
        "downloaded_at": utc_now(),
        "cache_method": None,
        "missing_fraction": None,
        "mapped_fraction": None,
        "unique_mapping_fraction": None,
        "source_verified": False,
        "formal_eligible": False,
        "status": "missing_input",
        "status_note": blocker,
        "blocker": blocker,
        "qc_note": None,
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "panel",
        "key",
        "status",
        "status_note",
        "path",
        "source_path",
        "source",
        "source_url",
        "assembly",
        "region_start",
        "region_end",
        "conversion",
        "binning",
        "sha256",
        "source_sha256",
        "size_bytes",
        "source_size_bytes",
        "downloaded_at",
        "cache_method",
        "missing_fraction",
        "mapped_fraction",
        "unique_mapping_fraction",
        "source_verified",
        "formal_eligible",
        "blocker",
        "qc_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and QC strict real external tracks."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--force", action="store_true", help="Redownload complete sources"
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=None,
        help="Reuse an existing complete-source cache while writing derived files to --output-dir",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any required external row is missing",
    )
    args = parser.parse_args()
    root = args.root
    output_dir = (
        args.output_dir or root / "results/final_har_haqer_panel/formal_main_figure"
    )
    cache = output_dir / CACHE_NAME
    cache.mkdir(parents=True, exist_ok=True)
    source_cache = args.source_cache
    if source_cache is not None and not source_cache.is_absolute():
        source_cache = root / source_cache
    tracks: list[dict[str, Any]] = []
    download_records: dict[str, Any] = {}

    # Download and hash the official source files.  The complete files are
    # retained because the current pyBigWig/libcurl build cannot open these
    # URLs using HTTP ranges.
    for key, source in SOURCES.items():
        source_path = (source_cache or cache) / source["filename"]
        range_ok, range_error = (
            try_remote_bigwig(source["url"])
            if key in {"ape_divergence", "gerp", "linsight"}
            else (False, "not a BigWig source")
        )
        info = download(source["url"], source_path, force=args.force)
        info["range_probe_ok"] = range_ok
        info["range_probe_note"] = range_error
        download_records[key] = info

    chain_path = (
        (source_cache / CHAIN_FILENAME)
        if source_cache is not None
        else cache / CHAIN_FILENAME
    )
    chain_info = download(CHAIN_URL, chain_path, force=args.force)
    download_records["liftover_chain"] = chain_info

    # Download only the small/metadata candidates that can be found from
    # authoritative public records.  These are provenance inputs; the peak
    # files are deliberately not treated as quantitative ATAC signal.
    auxiliary_records: dict[str, Any] = {}
    for key, source in AUXILIARY_SOURCES.items():
        destination = cache / source["filename"]
        try:
            info = download(source["url"], destination, force=args.force)
            auxiliary_records[key] = {
                **info,
                "source": source["source"],
                "role": source["role"],
                "assembly": source["assembly"],
                "source_verified": True,
                "formal_eligible": False,
                "status": "candidate_only",
                "status_note": "Downloaded for source discovery; not promoted without quantitative, locus-matched assay validation",
            }
        except Exception as exc:  # keep the missing-source report complete
            auxiliary_records[key] = {
                "url": source["url"],
                "source": source["source"],
                "role": source["role"],
                "assembly": source["assembly"],
                "source_verified": False,
                "formal_eligible": False,
                "status": "download_failed",
                "status_note": f"{type(exc).__name__}: {exc}",
            }

    # Extract locus-sized, hashable derived files.
    for panel in REGIONS:
        ape = extract_bigwig(
            download_records["ape_divergence"]["path"],
            panel,
            cache / f"{panel}_ape_divergence_hg38.tsv",
            n_bins=1200,
        )
        tracks.append(
            source_base(
                "ape_divergence", download_records["ape_divergence"], ape, panel
            )
        )
        gnocchi = extract_gnocchi(
            Path(download_records["gnocchi"]["path"]),
            panel,
            cache / f"{panel}_gnocchi_GRCh38_1kb.tsv",
        )
        tracks.append(
            source_base("gnocchi", download_records["gnocchi"], gnocchi, panel)
        )
        gerp = extract_bigwig(
            Path(download_records["gerp"]["path"]),
            panel,
            cache / f"{panel}_gerp_release101_GRCh38.tsv",
            n_bins=1200,
        )
        tracks.append(source_base("gerp", download_records["gerp"], gerp, panel))
        linsight = extract_linsight(
            Path(download_records["linsight"]["path"]),
            chain_path,
            panel,
            cache / f"{panel}_linsight_hg19_to_hg38.tsv",
            n_bins=1400,
        )
        linsight_entry = source_base(
            "linsight", download_records["linsight"], linsight, panel
        )
        linsight_entry["source_url"] = download_records["linsight"].get(
            "final_url", SOURCES["linsight"]["url"]
        )
        linsight_entry["source_path"] = download_records["linsight"]["path"]
        linsight_entry["chain_url"] = CHAIN_URL
        linsight_entry["chain_path"] = str(chain_path)
        linsight_entry["chain_sha256"] = chain_info["sha256"]
        linsight_entry["chain_source_sha256"] = chain_info["sha256"]
        linsight_entry["conversion"] = (
            "hg19 LINSIGHT -> hg38 via hg19ToHg38.over.chain.gz; reciprocal round-trip QC"
        )
        if (
            float(linsight.get("unique_mapping_fraction") or 0.0) < 0.95
            or int(linsight.get("duplicate_bin_count") or 0) > 0
        ):
            linsight_entry["formal_eligible"] = False
            linsight_entry["status"] = "qc_failed"
            linsight_entry["status_note"] = (
                "Unique reciprocal mapping fraction is below 95% or duplicate bins were observed"
            )
            linsight_entry["blocker"] = linsight_entry["status_note"]
        tracks.append(linsight_entry)

    # Deliberately do not substitute unrelated public tracks for these rows.
    missing_reasons = {
        "atac": "No case-matched ATAC-seq accession, sample, species, and cell type were supplied",
        "human_ape_delta": "No matched human-ape regulatory assay accession and normalization were supplied",
        "ccre_cage": "Project cCRE exists, but no matched CAGE component and assembly-conversion record were supplied",
        "human_branch_phylop": "No branch-specific PHAST alignment/model/result was supplied; ordinary phyloP100way is not accepted",
    }
    for panel in REGIONS:
        for key, reason in missing_reasons.items():
            tracks.append(missing_entry(panel, key, reason))

    # Add run-level provenance to the JSON manifest, while the rows remain
    # directly consumable by downstream track-table assembly.
    required_failures = [
        f"{row['panel']}:{row['key']}={row['status']}"
        for row in tracks
        if not row.get("formal_eligible", False)
    ]
    payload = {
        "schema_version": "formal_har_haqer_external_tracks_v1",
        "figure": "Figure4g_h_HAR_HAQER_formal",
        "created_at": utc_now(),
        "root": str(root),
        "cache_dir": str(cache),
        "regions": REGIONS,
        "source_policy": "Only source-verified real tracks are eligible; no missing-to-zero or representative assay substitution",
        "download_records": download_records,
        "auxiliary_source_records": auxiliary_records,
        "liftover_chain": {"url": CHAIN_URL, **chain_info},
        "tracks": tracks,
        "strict_formal_eligible": not required_failures,
        "missing_or_failed": required_failures,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_table(output_dir / TSV_NAME, tracks)
    write_table(output_dir / QC_NAME, tracks)
    missing_rows = [row for row in tracks if not row.get("formal_eligible", False)]
    (output_dir / MISSING_JSON_NAME).write_text(
        json.dumps({"missing_or_failed": missing_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / MISSING_TSV_NAME).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["panel", "key", "status", "blocker"], delimiter="\t"
        )
        writer.writeheader()
        for row in missing_rows:
            writer.writerow(
                {
                    "panel": row["panel"],
                    "key": row["key"],
                    "status": row["status"],
                    "blocker": row.get("blocker") or row.get("status_note") or "",
                }
            )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "cache": str(cache),
                "strict_formal_eligible": not required_failures,
                "missing_or_failed": required_failures,
                "source_files": {
                    key: {
                        "path": value["path"],
                        "size_bytes": value["size_bytes"],
                        "sha256": value["sha256"],
                    }
                    for key, value in download_records.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.strict and required_failures:
        raise SystemExit(
            "Strict external-track validation failed: " + ", ".join(required_failures)
        )


if __name__ == "__main__":
    main()
