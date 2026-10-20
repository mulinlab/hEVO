#!/usr/bin/env python3
"""Extract all available annotation tracks for the fixed Fig4f loci.

Run this on AiStation in the llmbm environment because its local cache contains
the source BigWigs, BigBed, chain file, and Gnocchi release used by the project.
Only compact interval snippets are exported back to the repository.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyBigWig
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    segments = json.loads(args.segments.read_text())["segments"]
    project = args.project_root
    cache = (
        project
        / "results/final_har_haqer_panel/formal_main_figure/external_track_cache"
    )
    source_paths = {
        "h3k27ac": project
        / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig",
        "phastcons": project / "external_tracks/ucsc/hg38.phastCons100way.bw",
        "ape_divergence": cache / "haqer.divergenceDensity.bw",
        "gerp": cache / "gerp_conservation_scores.homo_sapiens.GRCh38.bw",
    }
    annotations: dict[str, dict] = {}
    signal_rows: list[dict] = []

    # The helper contains strict reciprocal chain mapping. Extend its small
    # historical chr14/chr19 size table from the exact input BigWigs.
    workflow = project / "hEvo_benchmark/scripts/har_haqer_workflow"
    if not workflow.is_dir():
        workflow = Path(__file__).resolve().parent
    sys.path.insert(0, str(workflow))
    import prepare_external_tracks_remote as prep  # noqa: E402

    hg38_handle = pyBigWig.open(str(source_paths["phastcons"]))
    hg19_handle = pyBigWig.open(str(cache / "LINSIGHT.hg19.bw"))
    for segment in segments:
        chrom = segment["chrom"]
        prep.REGIONS[segment["id"]] = segment
        hg38_name = (
            chrom if chrom in hg38_handle.chroms() else chrom.removeprefix("chr")
        )
        hg19_name = (
            chrom if chrom in hg19_handle.chroms() else chrom.removeprefix("chr")
        )
        prep.HG38_CHROM_SIZES[chrom] = int(hg38_handle.chroms()[hg38_name])
        prep.HG19_CHROM_SIZES[chrom] = int(hg19_handle.chroms()[hg19_name])
    hg38_handle.close()
    hg19_handle.close()

    for key, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with pyBigWig.open(str(path)) as bigwig:
            chrom_sizes = bigwig.chroms()
            for index, segment in enumerate(segments):
                chrom = segment["chrom"]
                source_chrom = (
                    chrom if chrom in chrom_sizes else chrom.removeprefix("chr")
                )
                if source_chrom not in chrom_sizes:
                    continue
                n_bins = math.ceil((segment["end"] - segment["start"]) / 32)
                edges = np.linspace(
                    segment["start"], segment["end"], n_bins + 1
                ).astype(int)
                values = bigwig.stats(
                    source_chrom,
                    segment["start"],
                    segment["end"],
                    nBins=n_bins,
                    exact=True,
                )
                for left, right, value in zip(edges[:-1], edges[1:], values):
                    signal_rows.append(
                        {
                            "track": key,
                            "segment": index,
                            "chrom": chrom,
                            "start": int(left),
                            "end": int(right),
                            "position": (int(left) + int(right)) / 2,
                            "value": value,
                        }
                    )
        annotations[key] = {
            "source": str(path),
            "assembly": "hg38",
            "method": "exact BigWig mean over 32 bp target bins; missing values retained",
        }
        print("extracted", key, flush=True)

    linsight_path = cache / "LINSIGHT.hg19.bw"
    chain_path = cache / "hg19ToHg38.over.chain.gz"
    for index, segment in enumerate(segments):
        result = prep.extract_linsight(
            linsight_path,
            chain_path,
            segment["id"],
            args.output / f"segment_{index}_linsight.tsv",
            n_bins=math.ceil((segment["end"] - segment["start"]) / 32),
        )
        with (args.output / f"segment_{index}_linsight.tsv").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                value = float(row["value"])
                left, right = int(row["start"]), int(row["end"])
                signal_rows.append(
                    {
                        "track": "linsight",
                        "segment": index,
                        "chrom": segment["chrom"],
                        "start": left,
                        "end": right,
                        "position": (left + right) / 2,
                        "value": value if math.isfinite(value) else None,
                    }
                )
        annotations.setdefault(
            "linsight",
            {
                "source": str(linsight_path),
                "assembly": "hg19 -> hg38",
                "method": "unique mapping and reciprocal round-trip >=95% per bin; missing bins retained",
                "segments": [],
            },
        )["segments"].append(result)
    print("extracted linsight", flush=True)

    # Gnocchi uses the release's native 1 kb GRCh38 intervals.
    gnocchi_path = cache / "constraint_z_genome_1kb.qc.download.txt.gz"
    selected_chroms = {segment["chrom"] for segment in segments}
    segment_by_chrom = {}
    for index, segment in enumerate(segments):
        segment_by_chrom.setdefault(segment["chrom"], []).append((index, segment))
    with gzip.open(gnocchi_path, "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            chrom = row.get("chrom", "")
            if chrom not in selected_chroms:
                continue
            left, right = int(row["start"]), int(row["end"])
            for index, segment in segment_by_chrom[chrom]:
                if left < segment["end"] and right > segment["start"]:
                    signal_rows.append(
                        {
                            "track": "gnocchi",
                            "segment": index,
                            "chrom": chrom,
                            "start": left,
                            "end": right,
                            "position": (left + right) / 2,
                            "value": float(row["z"]),
                        }
                    )
    annotations["gnocchi"] = {
        "source": str(gnocchi_path),
        "assembly": "GRCh38",
        "method": "native 1 kb windows; no interpolation",
    }
    print("extracted gnocchi", flush=True)

    # cCREs are interval features; preserve their official labels.
    ccre_path = project / "external_tracks/ucsc/encodeCcreCombined.hg38.bb"
    ccre_rows = []
    with pyBigWig.open(str(ccre_path)) as bigbed:
        chrom_sizes = bigbed.chroms()
        for index, segment in enumerate(segments):
            chrom = (
                segment["chrom"]
                if segment["chrom"] in chrom_sizes
                else segment["chrom"].removeprefix("chr")
            )
            if chrom not in chrom_sizes:
                continue
            for left, right, annotation in (
                bigbed.entries(chrom, segment["start"], segment["end"]) or []
            ):
                label = str(annotation or "")
                ccre_rows.append(
                    {
                        "segment": index,
                        "chrom": segment["chrom"],
                        "start": int(left),
                        "end": int(right),
                        "name": label.split("\t")[0],
                        "annotation": label,
                    }
                )
    (args.output / "ccres.json").write_text(json.dumps(ccre_rows, indent=2) + "\n")
    annotations["ccres"] = {
        "source": str(ccre_path),
        "assembly": "hg38",
        "method": "ENCODE cCRE intervals; no CAGE signal is implied",
    }

    # CADD v1.7: per-base maximum across A/C/G/T, then max within each 32 bp bin.
    cache_dir = args.output / "cadd_api"
    cache_dir.mkdir(exist_ok=True)

    def fetch(task):
        index, allele = task
        segment = segments[index]
        track = f"cadd1_7_{allele}"
        cache_file = cache_dir / f"{index}_{allele}.json"
        url = (
            "https://api.genome.ucsc.edu/getData/track?genome=hg38;"
            f"track={track};chrom={segment['chrom']};start={segment['start']};end={segment['end']}"
        )
        if cache_file.is_file():
            data = json.loads(cache_file.read_text())
        else:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            data = response.json()
            if track not in data or data.get("maxItemsLimit"):
                raise RuntimeError(f"Incomplete CADD response: {url}")
            cache_file.write_text(json.dumps(data))
        values = np.full(segment["end"] - segment["start"], np.nan)
        for row in data[track]:
            left, right = (
                max(int(row["start"]), segment["start"]),
                min(int(row["end"]), segment["end"]),
            )
            if left < right:
                values[left - segment["start"] : right - segment["start"]] = float(
                    row["value"]
                )
        return index, allele, values, url

    with ThreadPoolExecutor(max_workers=4) as pool:
        cadd_values = list(
            pool.map(
                fetch, [(i, allele) for i in range(len(segments)) for allele in "ACGT"]
            )
        )
    cadd_rows = []
    for index, segment in enumerate(segments):
        arrays = np.stack(
            [values for i, allele, values, url in cadd_values if i == index]
        )
        complete = np.isfinite(arrays).all(axis=0)
        maximum = np.max(arrays, axis=0)
        maximum[~complete] = np.nan
        for left in range(segment["start"], segment["end"], 32):
            right = min(left + 32, segment["end"])
            chunk = maximum[left - segment["start"] : right - segment["start"]]
            value = float(np.nanmax(chunk)) if np.isfinite(chunk).any() else None
            cadd_rows.append(
                {
                    "track": "cadd",
                    "segment": index,
                    "chrom": segment["chrom"],
                    "start": left,
                    "end": right,
                    "position": (left + right) / 2,
                    "value": value,
                }
            )
    annotations["cadd"] = {
        "source": "UCSC hg38 CADD v1.7 A/C/G/T BigWig API",
        "assembly": "hg38",
        "method": "per-base max ALT PHRED, then 32 bp-bin maximum",
        "urls": [url for i, allele, values, url in cadd_values],
    }
    with (args.output / "signals.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "track",
                "segment",
                "chrom",
                "start",
                "end",
                "position",
                "value",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(signal_rows + cadd_rows)
    (args.output / "sources.json").write_text(json.dumps(annotations, indent=2) + "\n")
    print(
        json.dumps(
            {
                "signal_rows": len(signal_rows),
                "cadd_rows": len(cadd_rows),
                "ccres": len(ccre_rows),
                "segments": len(segments),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
