"""Build quantitative GSE221830 ATAC tracks for the HAR/NPAS3 panel.

This module is an execution-oriented, resumable orchestrator.  It does not use
the downloaded narrowPeak files as signal.  Instead it downloads the public SRA
runs, aligns human reads to hg38 and chimpanzee reads to panTro5, computes
paired-end fragment coverage, projects chimp coverage to hg38, and writes a
fixed 25-bp grid plus provenance/QC metadata.

The script itself is run with the project's llmbm Python.  External command
line tools are invoked from the separate remote har_atac_tools environment (or
the existing llmbm bedtools/liftOver installation).
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
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MICROMAMBA = os.environ.get("MICROMAMBA", shutil.which("micromamba") or "micromamba")
TOOLS_ENV = "har_atac_tools"
BEDTOOLS_ENV = "llmbm"
REGION = {"chrom": "chr14", "start": 33_000_000, "end": 34_000_000}
BIN_SIZE = 25

SAMPLES: list[dict[str, Any]] = [
    {
        "sample": "GSM6900347",
        "species": "human",
        "assembly": "hg38",
        "cell_type": "excitatory neurons",
        "time": "2 week",
        "cell_line": "WTC11",
        "srx": "SRX18859777",
        "runs": ["SRR22902192", "SRR22902193"],
    },
    {
        "sample": "GSM6900348",
        "species": "human",
        "assembly": "hg38",
        "cell_type": "excitatory neurons",
        "time": "2 week",
        "cell_line": "WTC11",
        "srx": "SRX18859778",
        "runs": ["SRR22902191"],
    },
    {
        "sample": "GSM6900349",
        "species": "chimpanzee",
        "assembly": "panTro5",
        "cell_type": "excitatory neurons",
        "time": "2 week",
        "cell_line": "C3649",
        "srx": "SRX18859775",
        "runs": ["SRR22902195"],
    },
    {
        "sample": "GSM6900350",
        "species": "chimpanzee",
        "assembly": "panTro5",
        "cell_type": "excitatory neurons",
        "time": "2 week",
        "cell_line": "C3649",
        "srx": "SRX18859776",
        "runs": ["SRR22902194"],
    },
]

REFERENCE_URLS = {
    "hg38": {
        "fasta_gz": "https://ftp.ensembl.org/pub/release-101/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
        "sizes": None,
    },
    "panTro5": {
        # The UCSC mirror exposes the panTro5 assembly used by the chain
        # files and is substantially faster from the configured mirror than Ensembl FTP.
        "fasta_gz": "https://hgdownload3.gi.ucsc.edu/goldenPath/panTro5/bigZips/panTro5.fa.gz",
        "sizes": None,
    },
}
CHAIN_URLS = {
    # UCSC's mirror is reachable from the configured environment while the primary
    # hgdownload.soe.ucsc.edu endpoint is not.  These are the same UCSC
    # chain resources, not a substitute coordinate conversion.
    "hg38_to_panTro5": "https://hgdownload3.gi.ucsc.edu/goldenPath/hg38/liftOver/hg38ToPanTro5.over.chain.gz",
    "panTro5_to_hg38": "https://hgdownload3.gi.ucsc.edu/goldenPath/panTro5/liftOver/panTro5ToHg38.over.chain.gz",
}
SRA_OBJECT_URL = (
    "https://sra-pub-run-odp.s3.amazonaws.com/sra/{run_accession}/{run_accession}"
)
ENA_FILEREPORT_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession={run_accession}&result=read_run&fields=run_accession,fastq_ftp,fastq_bytes"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q(value: Path | str) -> str:
    return shlex.quote(str(value))


def env_command(env: str, command: str) -> list[str]:
    return [MICROMAMBA, "run", "-n", env, "bash", "-lc", command]


def run(
    command: str, env: str, log_path: Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {env}$ {command}\n")
        result = subprocess.run(
            env_command(env, command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if check and result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}); see {log_path}: {command}"
        )
    return result


def download(url: str, destination: Path, log_path: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    # Keep a resumable partial file: the remote route to public archives
    # is bandwidth-limited and an SSH interruption must not restart a GB file.
    run(
        f"curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 60 -C - -o {q(temporary)} {q(url)} && mv {q(temporary)} {q(destination)}",
        TOOLS_ENV,
        log_path,
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {destination}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def download_references(
    root: Path, out: Path, log_path: Path, reuse_fasta: dict[str, Path] | None = None
) -> dict[str, Any]:
    ref_dir = out / "references"
    reuse_fasta = reuse_fasta or {}
    records: dict[str, Any] = {}
    for assembly, urls in REFERENCE_URLS.items():
        gz = ref_dir / f"{assembly}.fa.gz"
        fasta = ref_dir / f"{assembly}.fa"
        sizes = ref_dir / f"{assembly}.chrom.sizes"
        reused = reuse_fasta.get(assembly)
        if reused is not None:
            if not reused.is_file() or reused.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Requested reusable {assembly} FASTA is missing: {reused}"
                )
            if not fasta.is_file():
                run(f"ln -s {q(reused)} {q(fasta)}", TOOLS_ENV, log_path)
        else:
            download(urls["fasta_gz"], gz, log_path)
        if not fasta.is_file() or fasta.stat().st_size == 0:
            temporary = fasta.with_name(fasta.name + ".part")
            run(
                f'gzip -dc {q(gz)} | awk \'BEGIN{{OFS=""}} /^>/{{split($0,a," " ); h=a[1]; sub(/^>/,"",h); if(h=="MT") h="chrM"; else if(h !~ /^chr/) h="chr" h; print ">" h substr($0,length(a[1])+1); next}} {{print}}\' > {q(temporary)} && mv {q(temporary)} {q(fasta)}',
                TOOLS_ENV,
                log_path,
            )
        if not (fasta.with_suffix(fasta.suffix + ".fai")).is_file():
            run(f"samtools faidx {q(fasta)}", TOOLS_ENV, log_path)
        if urls.get("sizes"):
            download(str(urls["sizes"]), sizes, log_path)
        elif not sizes.is_file():
            fai = fasta.with_suffix(fasta.suffix + ".fai")
            temporary = sizes.with_name(sizes.name + ".part")
            run(
                f"cut -f1,2 {q(fai)} > {q(temporary)} && mv {q(temporary)} {q(sizes)}",
                TOOLS_ENV,
                log_path,
            )
        records[assembly] = {
            "fasta": str(fasta),
            "fasta_gz": None if reused is not None else str(gz),
            "fasta_sha256": sha256_file(fasta),
            "fasta_gz_sha256": None if reused is not None else sha256_file(gz),
            "sizes": str(sizes),
            "sizes_sha256": sha256_file(sizes),
            "assembly": assembly,
        }
    chains = out / "chains"
    for key, url in CHAIN_URLS.items():
        destination = chains / f"{key}.over.chain.gz"
        download(url, destination, log_path)
        records[key] = {
            "path": str(destination),
            "url": url,
            "sha256": sha256_file(destination),
        }
    write_json(out / "reference_download_manifest.json", records)
    return records


def sra_path(raw_dir: Path, run_accession: str) -> Path | None:
    candidates = [
        raw_dir / "sra" / f"{run_accession}.sra",
        raw_dir / "sra" / run_accession / f"{run_accession}.sra",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    for candidate in (raw_dir / "sra").glob(f"**/{run_accession}.sra"):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def ena_fastq_metadata(run_accession: str) -> list[dict[str, Any]]:
    url = ENA_FILEREPORT_URL.format(run_accession=run_accession)
    result = subprocess.run(
        env_command(
            TOOLS_ENV, f"curl -L --fail --silent --show-error --max-time 60 {q(url)}"
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(
            f"ENA did not return paired FASTQ metadata for {run_accession}"
        )
    header = lines[0].split("\t")
    values = lines[1].split("\t")
    record = dict(zip(header, values))
    urls = str(record.get("fastq_ftp", "")).split(";")
    sizes = str(record.get("fastq_bytes", "")).split(";")
    if len(urls) < 2 or len(sizes) < 2:
        raise RuntimeError(
            f"ENA run is not paired-end or has no sizes: {run_accession}"
        )
    return [
        {
            "url": f"https://{item}" if not item.startswith("http") else item,
            "size_bytes": int(size),
        }
        for item, size in zip(urls[:2], sizes[:2])
    ]


def remote_content_length(url: str) -> int:
    header = subprocess.run(
        env_command(
            TOOLS_ENV, f"curl -L --silent --show-error --head --max-time 60 {q(url)}"
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    lengths = re.findall(r"(?im)^content-length:\s*(\d+)\s*$", header.stdout)
    if not lengths:
        raise RuntimeError(f"Could not determine remote object size: {url}")
    return int(lengths[-1])


def download_fastq_file(
    url: str, destination: Path, expected_size: int, log_path: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    if destination.is_file() and destination.stat().st_size != expected_size:
        destination.rename(
            destination.with_name(
                destination.name + f".invalid_size_{destination.stat().st_size}"
            )
        )
    if temporary.is_file() and temporary.stat().st_size > expected_size:
        temporary.rename(
            temporary.with_name(
                temporary.name + f".invalid_size_{temporary.stat().st_size}"
            )
        )
    segment = temporary.with_name(temporary.name + ".segment")
    # Use explicit byte ranges and append only the received segment.  This is
    # more reliable on the remote route than curl -C - for multi-GB files:
    # a timed-out response can be committed and the next request starts at the
    # exact next byte, without the -r/-C incompatibility.
    command = (
        f'failures=0; while [ "$(stat -c%s {q(temporary)} 2>/dev/null || echo 0)" -lt {expected_size} ]; do '
        f'offset="$(stat -c%s {q(temporary)} 2>/dev/null || echo 0)"; end=$(({expected_size}-1)); rm -f {q(segment)}; '
        f'curl -L --fail --retry 8 --retry-delay 5 --connect-timeout 60 --max-time 900 --silent --show-error -r "$offset-$end" -o {q(segment)} {q(url)} || true; '
        f'got="$(stat -c%s {q(segment)} 2>/dev/null || echo 0)"; remaining=$(({expected_size}-$offset)); '
        f'if [ "$got" -gt 0 ] && [ "$got" -le "$remaining" ]; then cat {q(segment)} >> {q(temporary)}; rm -f {q(segment)}; failures=0; '
        f'else failures=$((failures+1)); rm -f {q(segment)}; echo NO_PROGRESS_$failures >&2; if [ "$failures" -ge 20 ]; then exit 42; fi; sleep 5; fi; '
        f'done; test "$(stat -c%s {q(temporary)})" -eq {expected_size}; mv {q(temporary)} {q(destination)}'
    )
    result = run(command, TOOLS_ENV, log_path, check=False)
    if result.returncode:
        raise RuntimeError(f"FASTQ download failed ({result.returncode}): {url}")
    if not destination.is_file() or destination.stat().st_size != expected_size:
        raise RuntimeError(f"FASTQ size mismatch after download: {destination}")


def download_sra(raw_dir: Path, log_path: Path) -> dict[str, Any]:
    """Download paired FASTQ objects for the GEO/SRA runs from the ENA mirror."""
    records: dict[str, Any] = {}
    fastq_source_dir = raw_dir / "fastq_source"
    fastq_source_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for sample in SAMPLES:
        for run_accession in sample["runs"]:
            for mate, metadata in enumerate(ena_fastq_metadata(run_accession), start=1):
                tasks.append(
                    {
                        "sample": sample["sample"],
                        "run_accession": run_accession,
                        "mate": mate,
                        "url": metadata["url"],
                        "expected_size": int(metadata["size_bytes"]),
                        "destination": fastq_source_dir
                        / f"{run_accession}_{mate}.fastq.gz",
                    }
                )

    def download_one(task: dict[str, Any]) -> dict[str, Any]:
        task_log = log_path.with_name(
            f"{log_path.stem}_{task['run_accession']}_{task['mate']}{log_path.suffix}"
        )
        download_fastq_file(
            task["url"], task["destination"], task["expected_size"], task_log
        )
        return {
            "sample": task["sample"],
            "run_accession": task["run_accession"],
            "mate": str(task["mate"]),
            "url": task["url"],
            "size_bytes": task["expected_size"],
            "sha256": sha256_file(task["destination"]),
        }

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as executor:
        futures = [executor.submit(download_one, task) for task in tasks]
        for future in as_completed(futures):
            completed.append(future.result())
    for sample in SAMPLES:
        for run_accession in sample["runs"]:
            mate_records = {}
            for item in completed:
                if item["run_accession"] != run_accession:
                    continue
                mate_records[item["mate"]] = {
                    "path": str(
                        fastq_source_dir / f"{run_accession}_{item['mate']}.fastq.gz"
                    ),
                    "url": item["url"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
            if set(mate_records) != {"1", "2"}:
                raise RuntimeError(
                    f"Missing paired FASTQ download records for {run_accession}"
                )
            records[run_accession] = {
                "sample": sample["sample"],
                "sra_accession": run_accession,
                "download_method": "ENA mirrored FASTQ for NCBI SRA run; paired-end source files",
                "mates": mate_records,
            }
    write_json(
        raw_dir / "sra_manifest.json",
        {"created_at": utc_now(), "samples": SAMPLES, "runs": records},
    )
    return records


def fastq_paths(raw_dir: Path, run_accession: str) -> tuple[Path, Path]:
    fastq_dir = raw_dir / "fastq"
    return (
        fastq_dir / f"{run_accession}_1.fastq",
        fastq_dir / f"{run_accession}_2.fastq",
    )


def materialise_fastq(
    raw_dir: Path, run_accession: str, log_path: Path, threads: int
) -> tuple[Path, Path]:
    source_first = raw_dir / "fastq_source" / f"{run_accession}_1.fastq.gz"
    source_second = raw_dir / "fastq_source" / f"{run_accession}_2.fastq.gz"
    if (
        source_first.is_file()
        and source_second.is_file()
        and source_first.stat().st_size
        and source_second.stat().st_size
    ):
        # bowtie2 reads gzip-compressed FASTQ directly; avoid a second copy of
        # the multi-GB source files on the shared project volume.
        return source_first, source_second
    first, second = fastq_paths(raw_dir, run_accession)
    if not (
        first.is_file()
        and second.is_file()
        and first.stat().st_size
        and second.stat().st_size
    ):
        raw_dir.joinpath("fastq").mkdir(parents=True, exist_ok=True)
        temp = raw_dir / "fastq_tmp" / run_accession
        temp.mkdir(parents=True, exist_ok=True)
        sra = sra_path(raw_dir, run_accession)
        if sra is None:
            raise RuntimeError(f"Missing SRA file for {run_accession}")
        run(
            f"fasterq-dump --split-files --threads {threads} --temp {q(temp)} --outdir {q(raw_dir / 'fastq')} {q(sra)}",
            TOOLS_ENV,
            log_path,
        )
    if not first.is_file() or not second.is_file():
        raise RuntimeError(f"Paired FASTQ files missing for {run_accession}")
    return first, second


def index_prefix(out: Path, assembly: str) -> Path:
    return out / "references" / assembly / assembly


def build_indexes(out: Path, reference_records: dict[str, Any], log_path: Path) -> None:
    for assembly, record in reference_records.items():
        if assembly not in REFERENCE_URLS:
            continue
        prefix = index_prefix(out, assembly)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if (
            not (prefix.parent / f"{assembly}.1.bt2").is_file()
            and not (prefix.parent / f"{assembly}.1.bt2l").is_file()
        ):
            run(
                f"bowtie2-build --threads 16 {q(record['fasta'])} {q(prefix)}",
                TOOLS_ENV,
                log_path,
            )


def bam_paths(out: Path, run_accession: str) -> dict[str, Path]:
    directory = out / "alignments"
    return {
        "name_sorted": directory / f"{run_accession}.name_sorted.bam",
        "fixmate": directory / f"{run_accession}.fixmate.bam",
        "coordinate_sorted": directory / f"{run_accession}.coordinate_sorted.bam",
        "dedup": directory / f"{run_accession}.dedup.bam",
        "filtered": directory / f"{run_accession}.filtered.bam",
    }


def align_run(
    out: Path,
    raw_dir: Path,
    sample: dict[str, Any],
    run_accession: str,
    log_path: Path,
    threads: int,
) -> dict[str, Any]:
    paths = bam_paths(out, run_accession)
    paths["filtered"].parent.mkdir(parents=True, exist_ok=True)
    first, second = materialise_fastq(raw_dir, run_accession, log_path, threads)
    prefix = index_prefix(out, sample["assembly"])
    if not paths["filtered"].is_file():
        if not paths["name_sorted"].is_file():
            cmd = (
                f"bowtie2 --very-sensitive-local --no-unal --threads {threads} -x {q(prefix)} "
                f"-1 {q(first)} -2 {q(second)} | samtools sort -n -@ {threads} -o {q(paths['name_sorted'])} -"
            )
            run(cmd, TOOLS_ENV, log_path)
        if not paths["fixmate"].is_file():
            run(
                f"samtools fixmate -m -@ {threads} {q(paths['name_sorted'])} {q(paths['fixmate'])}",
                TOOLS_ENV,
                log_path,
            )
        if not paths["coordinate_sorted"].is_file():
            run(
                f"samtools sort -@ {threads} -o {q(paths['coordinate_sorted'])} {q(paths['fixmate'])}",
                TOOLS_ENV,
                log_path,
            )
        if not paths["dedup"].is_file():
            run(
                f"samtools markdup -r -@ {threads} {q(paths['coordinate_sorted'])} {q(paths['dedup'])}",
                TOOLS_ENV,
                log_path,
            )
        run(
            f"samtools view -b -q 30 -F 0xF0C -o {q(paths['filtered'])} {q(paths['dedup'])}",
            TOOLS_ENV,
            log_path,
        )
    index = Path(str(paths["filtered"]) + ".bai")
    if not index.is_file():
        run(f"samtools index -@ {threads} {q(paths['filtered'])}", TOOLS_ENV, log_path)
    count_cmd = f"samtools view -c -f 0x42 -q 30 -F 0xF0C {q(paths['filtered'])}"
    result = subprocess.run(
        env_command(TOOLS_ENV, count_cmd), capture_output=True, text=True, check=True
    )
    fragments = int(result.stdout.strip().splitlines()[-1])
    return {
        "run": run_accession,
        "sample": sample["sample"],
        "assembly": sample["assembly"],
        "filtered_bam": str(paths["filtered"]),
        "filtered_bam_sha256": sha256_file(paths["filtered"]),
        "valid_fragments": fragments,
    }


def merge_sample_bams(
    out: Path,
    sample: dict[str, Any],
    run_records: list[dict[str, Any]],
    log_path: Path,
    threads: int,
) -> dict[str, Any]:
    sample_dir = out / "sample_bams"
    sample_dir.mkdir(parents=True, exist_ok=True)
    merged = sample_dir / f"{sample['sample']}.filtered.bam"
    if not merged.is_file():
        inputs = " ".join(q(record["filtered_bam"]) for record in run_records)
        if len(run_records) == 1:
            run(f"cp {inputs} {q(merged)}", TOOLS_ENV, log_path)
        else:
            run(
                f"samtools merge -f -@ {threads} {q(merged)} {inputs}",
                TOOLS_ENV,
                log_path,
            )
    if not Path(str(merged) + ".bai").is_file():
        run(f"samtools index -@ {threads} {q(merged)}", TOOLS_ENV, log_path)
    count_cmd = f"samtools view -c -f 0x42 -q 30 -F 0xF0C {q(merged)}"
    result = subprocess.run(
        env_command(TOOLS_ENV, count_cmd), capture_output=True, text=True, check=True
    )
    return {
        "sample": sample["sample"],
        "species": sample["species"],
        "assembly": sample["assembly"],
        "bam": str(merged),
        "bam_sha256": sha256_file(merged),
        "valid_fragments": int(result.stdout.strip().splitlines()[-1]),
        "run_records": run_records,
    }


def write_target_bed(out: Path, name: str, region: dict[str, int]) -> Path:
    path = out / "mapping" / f"{name}.bed"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{region['chrom']}\t{region['start']}\t{region['end']}\t{name}\n",
        encoding="utf-8",
    )
    return path


def read_bed(path: Path) -> list[tuple[str, int, int, list[str]]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            rows.append((fields[0], int(fields[1]), int(fields[2]), fields[3:]))
    return rows


def map_target_to_chimp(
    out: Path, reference_records: dict[str, Any], log_path: Path
) -> list[tuple[str, int, int, list[str]]]:
    target = write_target_bed(out, "g_hg38_target", REGION)
    mapped = out / "mapping" / "g_hg38_target.panTro5.bed"
    unmapped = out / "mapping" / "g_hg38_target.panTro5.unmapped.bed"
    if not mapped.is_file():
        run(
            f"liftOver -minMatch=0.95 {q(target)} {q(reference_records['hg38_to_panTro5']['path'])} {q(mapped)} {q(unmapped)}",
            BEDTOOLS_ENV,
            log_path,
        )
    rows = read_bed(mapped)
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one unique hg38->panTro5 target interval; got {len(rows)}"
        )
    chrom, start, end, _ = rows[0]
    if end <= start:
        raise RuntimeError("Mapped panTro5 target interval is empty")
    return rows


def make_bins(region: dict[str, int], size: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(region["start"], region["end"], size, dtype=np.int64)
    ends = np.minimum(starts + size, region["end"])
    return starts, ends


def bedgraph_coverage(
    bam: Path,
    assembly: str,
    source_region: tuple[str, int, int],
    total_fragments: int,
    out: Path,
    log_path: Path,
    *,
    project_chain: Path | None = None,
) -> dict[str, Any]:
    chrom, start, end = source_region
    subset = out.with_suffix(".subset.bam")
    if not subset.is_file():
        # samtools coordinates are 1-based inclusive; BED/source coordinates
        # are 0-based half-open.
        run(
            f"samtools view -b -o {q(subset)} {q(bam)} {q(f'{chrom}:{start + 1}-{end}')}",
            TOOLS_ENV,
            log_path,
        )
    native_bg = out.with_suffix(".native.bedGraph")
    if not native_bg.is_file():
        run(
            f"bedtools genomecov -bg -pc -ibam {q(subset)} > {q(native_bg)}",
            BEDTOOLS_ENV,
            log_path,
        )
    input_bg = native_bg
    mapped_bg = None
    if project_chain is not None:
        mapped_bg = out.with_suffix(".mapped_hg38.bedGraph")
        unmapped = out.with_suffix(".mapped_hg38.unmapped.bedGraph")
        if not mapped_bg.is_file():
            run(
                f"liftOver -minMatch=0.95 {q(native_bg)} {q(project_chain)} {q(mapped_bg)} {q(unmapped)}",
                BEDTOOLS_ENV,
                log_path,
            )
        input_bg = mapped_bg

    if total_fragments <= 0:
        raise RuntimeError(f"No valid fragments for {bam}")

    def read_rows(path: Path) -> list[tuple[str, int, int, float]]:
        result: list[tuple[str, int, int, float]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    continue
                try:
                    left, right, depth = (
                        int(fields[1]),
                        int(fields[2]),
                        float(fields[3]),
                    )
                except ValueError:
                    continue
                if right > left and math.isfinite(depth):
                    result.append((fields[0], left, right, depth))
        return result

    native_rows = read_rows(native_bg)
    rows = native_rows if input_bg == native_bg else read_rows(input_bg)
    return {
        "rows": rows,
        "native_rows": native_rows,
        "native_bedgraph": str(native_bg),
        "mapped_bedgraph": str(mapped_bg) if mapped_bg else None,
        "source_region": [chrom, start, end],
        "valid_fragments": total_fragments,
        "assembly": assembly,
    }


def values_from_rows(
    rows: Iterable[tuple[str, int, int, float]],
    starts: np.ndarray,
    ends: np.ndarray,
    chrom: str,
    total_fragments: int,
    *,
    allow_missing: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    values = np.full(starts.shape, np.nan if allow_missing else 0.0, dtype=float)
    if not allow_missing:
        values.fill(0.0)
    overlap_depth = np.zeros(starts.shape, dtype=float)
    covered = np.zeros(starts.shape, dtype=np.int64)
    contributors = np.zeros(starts.shape, dtype=np.int64)
    for row_chrom, left, right, depth in rows:
        if row_chrom != chrom:
            continue
        first = max(0, int(np.searchsorted(ends, left, side="right")))
        last = min(len(starts), int(np.searchsorted(starts, right, side="left")))
        for index in range(first, last):
            overlap_left = max(left, int(starts[index]))
            overlap_right = min(right, int(ends[index]))
            if overlap_right <= overlap_left:
                continue
            overlap_depth[index] += depth * (overlap_right - overlap_left)
            covered[index] += overlap_right - overlap_left
            contributors[index] += 1
    # Values from bedtools are zero where no fragment covers a bin.  For the
    # projected chimp track, bins without a mapped block stay NaN.
    duplicate = contributors > 1
    for index in range(len(values)):
        if covered[index] > 0:
            values[index] = (
                overlap_depth[index]
                / (int(ends[index]) - int(starts[index]))
                / total_fragments
                * 1_000_000.0
            )
        elif not allow_missing:
            values[index] = 0.0
    if allow_missing:
        values[duplicate] = np.nan
    return values, {
        "covered_bins": int(np.count_nonzero(covered)),
        "duplicate_bins": int(np.count_nonzero(duplicate)),
    }


def write_grid(
    path: Path, chrom: str, starts: np.ndarray, ends: np.ndarray, values: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["chrom", "start", "end", "value"])
        for left, right, value in zip(starts, ends, values):
            writer.writerow(
                [
                    chrom,
                    int(left),
                    int(right),
                    "nan" if not np.isfinite(value) else f"{float(value):.10g}",
                ]
            )


def write_bigwig(path: Path, tsv: Path, sizes: Path, log_path: Path) -> None:
    bedgraph = path.with_suffix(".finite.bedGraph")
    with (
        tsv.open(encoding="utf-8") as source,
        bedgraph.open("w", encoding="utf-8") as target,
    ):
        next(source)
        for line in source:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 4 and fields[3].lower() != "nan":
                target.write("\t".join(fields) + "\n")
    run(
        f"sort -k1,1 -k2,2n {q(bedgraph)} -o {q(bedgraph)} && bedGraphToBigWig {q(bedgraph)} {q(sizes)} {q(path)}",
        TOOLS_ENV,
        log_path,
    )


def read_narrowpeak(
    narrowpeak: Path, chrom: str, start: int, end: int
) -> list[dict[str, Any]]:
    peaks: list[dict[str, Any]] = []
    with gzip.open(narrowpeak, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[0] != chrom:
                continue
            left, right = int(fields[1]), int(fields[2])
            if right > start and left < end:
                peaks.append(
                    {
                        "id": fields[3]
                        if len(fields) > 3 and fields[3]
                        else f"peak_{index}",
                        "chrom": chrom,
                        "start": max(start, left),
                        "end": min(end, right),
                        "source_start": left,
                        "source_end": right,
                    }
                )
    return peaks


def peak_overlap_intervals(
    peaks: list[dict[str, Any]], values: np.ndarray, starts: np.ndarray
) -> dict[str, Any]:
    covered = 0
    for peak in peaks:
        left, right = int(peak["start"]), int(peak["end"])
        first = max(0, int(np.searchsorted(starts, left, side="right") - 1))
        last = min(len(starts), int(np.searchsorted(starts, right, side="left") + 1))
        if np.any(np.isfinite(values[first:last]) & (values[first:last] > 0)):
            covered += 1
    return {
        "narrowpeak_count_in_locus": len(peaks),
        "narrowpeak_with_nonzero_coverage": covered,
    }


def map_chimp_narrowpeaks_to_hg38(
    narrowpeak: Path,
    source_region: tuple[str, int, int],
    chain: Path,
    out: Path,
    log_path: Path,
) -> dict[str, Any]:
    source_peaks = read_narrowpeak(
        narrowpeak, source_region[0], source_region[1], source_region[2]
    )
    source_bed = out / "g_chimp_narrowPeak_panTro5_source.bed"
    mapped_bed = out / "g_chimp_narrowPeak_mapped_hg38.bed"
    unmapped_bed = out / "g_chimp_narrowPeak_unmapped.bed"
    with source_bed.open("w", encoding="utf-8") as handle:
        for peak in source_peaks:
            handle.write(
                f"{peak['chrom']}\t{peak['source_start']}\t{peak['source_end']}\t{peak['id']}\n"
            )
    if source_peaks:
        run(
            f"liftOver -minMatch=0.95 -multiple {q(source_bed)} {q(chain)} {q(mapped_bed)} {q(unmapped_bed)}",
            BEDTOOLS_ENV,
            log_path,
        )
    else:
        mapped_bed.write_text("", encoding="utf-8")
        unmapped_bed.write_text("", encoding="utf-8")
    mapped_by_id: dict[str, list[tuple[str, int, int]]] = {}
    if mapped_bed.is_file():
        for line in mapped_bed.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            mapped_by_id.setdefault(fields[3], []).append(
                (fields[0], int(fields[1]), int(fields[2]))
            )
    unique = {
        key: value[0]
        for key, value in mapped_by_id.items()
        if len(value) == 1 and value[0][0] == REGION["chrom"]
    }
    return {
        "source_peak_count": len(source_peaks),
        "mapped_record_count": int(sum(len(value) for value in mapped_by_id.values())),
        "unique_same_chromosome_count": len(unique),
        "duplicate_source_peak_count": int(
            sum(len(value) > 1 for value in mapped_by_id.values())
        ),
        "unmapped_source_peak_count": max(0, len(source_peaks) - len(mapped_by_id)),
        "unique_mapping_fraction": len(unique) / max(1, len(source_peaks)),
        "source_bed": str(source_bed),
        "mapped_bed": str(mapped_bed),
        "unmapped_bed": str(unmapped_bed),
        "unique_intervals": [
            {"id": key, "chrom": value[0], "start": value[1], "end": value[2]}
            for key, value in unique.items()
        ],
    }


def process_tracks(
    root: Path,
    out: Path,
    reference_records: dict[str, Any],
    sample_records: list[dict[str, Any]],
    log_path: Path,
    threads: int,
) -> dict[str, Any]:
    starts, ends = make_bins(REGION, BIN_SIZE)
    mapped_target = map_target_to_chimp(out, reference_records, log_path)
    chimp_source = mapped_target[0]
    chimp_source_region = (chimp_source[0], chimp_source[1], chimp_source[2])
    human_sample_values: list[np.ndarray] = []
    chimp_sample_values: list[np.ndarray] = []
    native_chimp_values: list[np.ndarray] = []
    sample_qc: list[dict[str, Any]] = []
    for sample_record in sample_records:
        sample = next(
            item for item in SAMPLES if item["sample"] == sample_record["sample"]
        )
        sample_dir = out / "coverage" / sample["sample"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        bam = Path(sample_record["bam"])
        if sample["species"] == "human":
            coverage = bedgraph_coverage(
                bam,
                "hg38",
                (REGION["chrom"], REGION["start"], REGION["end"]),
                sample_record["valid_fragments"],
                sample_dir / "human",
                log_path,
            )
            values, stats = values_from_rows(
                coverage["rows"],
                starts,
                ends,
                REGION["chrom"],
                sample_record["valid_fragments"],
                allow_missing=False,
            )
            human_sample_values.append(values)
        else:
            coverage = bedgraph_coverage(
                bam,
                "panTro5",
                chimp_source_region,
                sample_record["valid_fragments"],
                sample_dir / "chimp",
                log_path,
                project_chain=Path(reference_records["panTro5_to_hg38"]["path"]),
            )
            projected, stats = values_from_rows(
                coverage["rows"],
                starts,
                ends,
                REGION["chrom"],
                sample_record["valid_fragments"],
                allow_missing=True,
            )
            chimp_sample_values.append(projected)
            native_starts = np.arange(
                chimp_source_region[1], chimp_source_region[2], BIN_SIZE, dtype=np.int64
            )
            native_ends = np.minimum(native_starts + BIN_SIZE, chimp_source_region[2])
            native, _ = values_from_rows(
                coverage["native_rows"],
                native_starts,
                native_ends,
                chimp_source_region[0],
                sample_record["valid_fragments"],
                allow_missing=False,
            )
            native_chimp_values.append(native)
        sample_qc.append(
            {
                "sample": sample["sample"],
                "species": sample["species"],
                "valid_fragments": sample_record["valid_fragments"],
                **stats,
            }
        )

    if len(human_sample_values) != 2 or len(chimp_sample_values) != 2:
        raise RuntimeError(
            f"Expected two human and two chimp replicates; got {len(human_sample_values)} and {len(chimp_sample_values)}"
        )
    human = np.mean(np.stack(human_sample_values), axis=0)
    chimp_projected = np.nanmean(np.stack(chimp_sample_values), axis=0)
    native_chimp = np.mean(np.stack(native_chimp_values), axis=0)
    delta = np.full_like(human, np.nan)
    finite = np.isfinite(human) & np.isfinite(chimp_projected)
    delta[finite] = np.log2((human[finite] + 1.0) / (chimp_projected[finite] + 1.0))

    cache = out
    human_tsv = cache / "g_human_atac_hg38_2wk_excitatory_neuron_CPM.tsv"
    chimp_tsv = cache / "g_chimp_atac_panTro5_2wk_excitatory_neuron_CPM.tsv"
    projected_tsv = cache / "g_chimp_atac_panTro5_to_hg38_2wk_excitatory_neuron_CPM.tsv"
    delta_tsv = cache / "g_human_ape_delta_ATAC_log2ratio.tsv"
    write_grid(human_tsv, REGION["chrom"], starts, ends, human)
    write_grid(delta_tsv, REGION["chrom"], starts, ends, delta)
    native_starts = np.arange(
        chimp_source_region[1], chimp_source_region[2], BIN_SIZE, dtype=np.int64
    )
    native_ends = np.minimum(native_starts + BIN_SIZE, chimp_source_region[2])
    write_grid(
        chimp_tsv, chimp_source_region[0], native_starts, native_ends, native_chimp
    )
    write_grid(projected_tsv, REGION["chrom"], starts, ends, chimp_projected)
    write_bigwig(
        cache / "g_human_atac_hg38_2wk_excitatory_neuron_CPM.bw",
        human_tsv,
        Path(reference_records["hg38"]["sizes"]),
        log_path,
    )
    write_bigwig(
        cache / "g_chimp_atac_panTro5_2wk_excitatory_neuron_CPM.bw",
        chimp_tsv,
        Path(reference_records["panTro5"]["sizes"]),
        log_path,
    )
    write_bigwig(
        cache / "g_chimp_atac_panTro5_to_hg38_2wk_excitatory_neuron_CPM.bw",
        projected_tsv,
        Path(reference_records["hg38"]["sizes"]),
        log_path,
    )
    write_bigwig(
        cache / "g_human_ape_delta_ATAC_log2ratio.bw",
        delta_tsv,
        Path(reference_records["hg38"]["sizes"]),
        log_path,
    )

    human_peak = out.parent / "GSE221830_Human_idr.optimal_peak.narrowPeak.gz"
    chimp_peak = out.parent / "GSE221830_Chimp_idr.optimal_peak.narrowPeak.gz"
    peak_qc: dict[str, Any] = {}
    if human_peak.is_file():
        human_peaks = read_narrowpeak(
            human_peak, REGION["chrom"], REGION["start"], REGION["end"]
        )
        peak_qc["human"] = peak_overlap_intervals(human_peaks, human, starts)
        peak_qc["human"]["source_path"] = str(human_peak)
        peak_qc["human"]["source_sha256"] = sha256_file(human_peak)
    if chimp_peak.is_file():
        chimp_peaks = read_narrowpeak(
            chimp_peak,
            chimp_source_region[0],
            chimp_source_region[1],
            chimp_source_region[2],
        )
        peak_qc["chimp_native_source"] = {
            **peak_overlap_intervals(chimp_peaks, native_chimp, native_starts),
            "path": str(chimp_peak),
            "sha256": sha256_file(chimp_peak),
            "assembly": "panTro5",
        }
        mapped_peak_qc = map_chimp_narrowpeaks_to_hg38(
            chimp_peak,
            chimp_source_region,
            Path(reference_records["panTro5_to_hg38"]["path"]),
            out,
            log_path,
        )
        mapped_intervals = mapped_peak_qc.pop("unique_intervals")
        mapped_peak_qc["coverage_overlap"] = peak_overlap_intervals(
            mapped_intervals, chimp_projected, starts
        )
        peak_qc["chimp_mapped_hg38"] = mapped_peak_qc

    target_length = REGION["end"] - REGION["start"]
    mapped_length = chimp_source_region[2] - chimp_source_region[1]
    mapping_qc = {
        "source_assembly": "panTro5",
        "target_assembly": "hg38",
        "target_region": dict(REGION),
        "mapped_source_interval": list(chimp_source_region),
        "mapped_fraction": mapped_length / max(1, target_length),
        "unique_mapping_fraction": 1.0 if len(mapped_target) == 1 else 0.0,
        "duplicate_projected_bins": int(
            sum(
                row.get("duplicate_bins", 0)
                for row in sample_qc
                if row["species"] == "chimpanzee"
            )
        ),
    }

    payload = {
        "schema_version": "g_atac_quantitative_v1",
        "created_at": utc_now(),
        "panel": "g",
        "region": {**REGION, "assembly": "hg38"},
        "source": {
            "accession": "GSE221830",
            "title": "Comparative characterization of human accelerated regions in neurons [ATAC-seq]",
            "human": ["GSM6900347", "GSM6900348"],
            "chimpanzee": ["GSM6900349", "GSM6900350"],
        },
        "samples": SAMPLES,
        "reference_records": reference_records,
        "processing": {
            "aligner": "bowtie2 --very-sensitive-local",
            "filter": "MAPQ>=30; proper paired fragments; secondary/supplementary/QC-fail/unmapped/duplicate excluded",
            "coverage": "bedtools genomecov -bg -pc; 25 bp bins; sample-level CPM then equal-weight replicate mean",
            "delta": "log2((human_CPM+1)/(chimp_CPM+1)); chimp projected panTro5->hg38; unmapped bins NaN",
            "bin_size_bp": BIN_SIZE,
            "no_missing_to_zero": True,
        },
        "sample_qc": sample_qc,
        "mapping_qc": mapping_qc,
        "peak_qc": peak_qc,
        "outputs": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for key, path in {
                "human_tsv": human_tsv,
                "human_bw": cache / "g_human_atac_hg38_2wk_excitatory_neuron_CPM.bw",
                "chimp_tsv": chimp_tsv,
                "chimp_bw": cache / "g_chimp_atac_panTro5_2wk_excitatory_neuron_CPM.bw",
                "projected_chimp_tsv": projected_tsv,
                "projected_chimp_bw": cache
                / "g_chimp_atac_panTro5_to_hg38_2wk_excitatory_neuron_CPM.bw",
                "delta_tsv": delta_tsv,
                "delta_bw": cache / "g_human_ape_delta_ATAC_log2ratio.bw",
            }.items()
        },
        "formal_eligible": True,
    }
    write_json(cache / "g_atac_processing_manifest.json", payload)
    with (cache / "g_atac_qc.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "sample",
            "species",
            "valid_fragments",
            "covered_bins",
            "duplicate_bins",
        ]
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(sample_qc)
    return payload


def update_external_manifest(out: Path, processing: dict[str, Any]) -> Path:
    manifest_path = out.parent / "Figure4g_h_HAR_HAQER_external_tracks_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"External formal manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_map = processing["outputs"]
    rows = {
        ("g", "atac"): output_map["human_tsv"],
        ("g", "human_ape_delta"): output_map["delta_tsv"],
    }
    for entry in payload.get("tracks", []):
        key = (str(entry.get("panel")), str(entry.get("key")))
        if key not in rows:
            continue
        path = Path(rows[key]["path"])
        entry.update(
            {
                "path": str(path),
                "source_path": str(out / "g_atac_processing_manifest.json"),
                "source": "NCBI GEO GSE221830 matched 2-week human/chimp excitatory-neuron ATAC-seq",
                "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221830&targ=self&form=text&view=quick",
                "assembly": "hg38",
                "conversion": "sample-level CPM fragment coverage on 25 bp grid; chimp panTro5 coverage projected to hg38 for delta",
                "binning": "25 bp bins; bedtools -pc paired-end fragment coverage",
                "sha256": processing["outputs"][
                    "human_tsv" if key[1] == "atac" else "delta_tsv"
                ]["sha256"],
                "size_bytes": processing["outputs"][
                    "human_tsv" if key[1] == "atac" else "delta_tsv"
                ]["size_bytes"],
                "source_sha256": sha256_file(out / "g_atac_processing_manifest.json"),
                "missing_fraction": None,
                "mapped_fraction": None
                if key[1] == "atac"
                else processing.get("mapping_qc", {}).get("mapped_fraction"),
                "unique_mapping_fraction": processing.get("mapping_qc", {}).get(
                    "unique_mapping_fraction"
                ),
                "source_verified": True,
                "formal_eligible": True,
                "status": "verified_real",
                "status_note": None,
                "blocker": None,
                "qc_note": "Raw SRA processed; narrowPeak retained only for independent QC",
                "sample_accessions": [
                    item["sample"] for item in SAMPLES if item["species"] == "human"
                ]
                if key[1] == "atac"
                else [item["sample"] for item in SAMPLES],
            }
        )
    payload["updated_at"] = utc_now()
    payload["quantitative_g_atac_processing"] = str(
        out / "g_atac_processing_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    global MICROMAMBA
    parser = argparse.ArgumentParser(
        description="Build GSE221830 quantitative ATAC tracks for HAR/NPAS3."
    )
    parser.add_argument("--root", type=Path, default=Path("<source-project-root>"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="External track cache; defaults to formal_main_figure_split/external_track_cache",
    )
    parser.add_argument(
        "--micromamba", default=MICROMAMBA, help="Micromamba executable (or MICROMAMBA)"
    )
    parser.add_argument(
        "--stage", choices=("all", "download", "align", "tracks"), default="all"
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--reuse-hg38-fasta",
        type=Path,
        default=None,
        help="Reuse an existing remote hg38 FASTA instead of downloading it",
    )
    parser.add_argument(
        "--reuse-pantro5-fasta",
        dest="reuse_pantro5_fasta",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    MICROMAMBA = args.micromamba
    root = args.root
    output_dir = (
        args.output_dir
        or root
        / "results/final_har_haqer_panel/formal_main_figure_split/external_track_cache"
    )
    raw_dir = output_dir / "gse221830_raw"
    log_path = output_dir.parent / "g_atac_processing.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    reuse_fasta = {}
    if args.reuse_hg38_fasta is not None:
        reuse_fasta["hg38"] = args.reuse_hg38_fasta
    if args.reuse_pantro5_fasta is not None:
        reuse_fasta["panTro5"] = args.reuse_pantro5_fasta
    references = download_references(root, output_dir, log_path, reuse_fasta)
    if args.stage in {"all", "download"}:
        download_sra(raw_dir, log_path)
    if args.stage == "download":
        print(
            json.dumps(
                {
                    "stage": "download",
                    "references": references,
                    "raw_dir": str(raw_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.stage in {"all", "align"}:
        if not (raw_dir / "sra_manifest.json").is_file():
            download_sra(raw_dir, log_path)
        run_records: dict[str, list[dict[str, Any]]] = {}
        for sample in SAMPLES:
            for run_accession in sample["runs"]:
                record = align_run(
                    output_dir, raw_dir, sample, run_accession, log_path, args.threads
                )
                run_records.setdefault(sample["sample"], []).append(record)
        sample_records = []
        for sample in SAMPLES:
            sample_records.append(
                merge_sample_bams(
                    output_dir,
                    sample,
                    run_records[sample["sample"]],
                    log_path,
                    args.threads,
                )
            )
        write_json(
            output_dir / "g_atac_alignment_manifest.json",
            {"created_at": utc_now(), "samples": sample_records},
        )
    if args.stage == "align":
        print(
            json.dumps(
                {
                    "stage": "align",
                    "manifest": str(output_dir / "g_atac_alignment_manifest.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    alignment_manifest = output_dir / "g_atac_alignment_manifest.json"
    if not alignment_manifest.is_file():
        raise FileNotFoundError(
            f"Missing alignment manifest: {alignment_manifest}; run --stage align"
        )
    alignment = json.loads(alignment_manifest.read_text(encoding="utf-8"))
    processing = process_tracks(
        root, output_dir, references, alignment["samples"], log_path, args.threads
    )
    update_external_manifest(output_dir, processing)
    print(
        json.dumps(
            {
                "stage": "tracks",
                "processing_manifest": str(
                    output_dir / "g_atac_processing_manifest.json"
                ),
                "formal_eligible": processing["formal_eligible"],
                "outputs": processing["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Keep a machine-readable failure record for interrupted downloads,
        # unavailable public endpoints, or tool failures.  No quantitative
        # track is promoted when this path is reached.
        try:
            root_arg = Path("<source-project-root>")
            if "--root" in sys.argv:
                root_arg = Path(sys.argv[sys.argv.index("--root") + 1])
            output_arg = None
            if "--output-dir" in sys.argv:
                output_arg = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            blocked_path = (
                output_arg
                or root_arg
                / "results/final_har_haqer_panel/formal_main_figure_split/external_track_cache"
            )
            write_json(
                blocked_path / "g_atac_blocked_report.json",
                {
                    "created_at": utc_now(),
                    "panel": "g",
                    "stage": sys.argv[sys.argv.index("--stage") + 1]
                    if "--stage" in sys.argv
                    else "all",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "formal_eligible": False,
                    "no_synthetic_signal": True,
                    "note": "No quantitative ATAC or human-ape delta track was promoted after this failure.",
                },
            )
        except Exception:
            pass
        raise
