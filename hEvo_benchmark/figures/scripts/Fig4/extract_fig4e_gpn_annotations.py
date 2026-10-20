#!/usr/bin/env python3
"""Extract Fig. 4e annotations for selected hg38 loci on the track-data host."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pyBigWig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    segments = json.loads(args.segments.read_text())["segments"]
    tracks = {
        "h3k27ac": project / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig",
        "phastcons": project / "external_tracks/ucsc/hg38.phastCons100way.bw",
        "ape_divergence": project / "results/final_har_haqer_panel/formal_main_figure/external_track_cache/haqer.divergenceDensity.bw",
    }
    for track, path in tracks.items():
        if not path.is_file():
            raise FileNotFoundError(path)

    signal_rows = []
    for track, path in tracks.items():
        with pyBigWig.open(str(path)) as bigwig:
            chrom_sizes = bigwig.chroms()
            for segment in segments:
                chrom = segment["chrom"]
                source_chrom = chrom if chrom in chrom_sizes else chrom.removeprefix("chr")
                if source_chrom not in chrom_sizes:
                    continue
                count = math.ceil((segment["end"] - segment["start"]) / 32)
                edges = [
                    round(segment["start"] + (segment["end"] - segment["start"]) * i / count)
                    for i in range(count + 1)
                ]
                values = bigwig.stats(
                    source_chrom, segment["start"], segment["end"],
                    nBins=count, exact=True,
                )
                for left, right, value in zip(edges[:-1], edges[1:], values):
                    signal_rows.append({
                        "track": track,
                        "segment": segment["index"],
                        "chrom": chrom,
                        "start": left,
                        "end": right,
                        "position": (left + right) / 2,
                        "value": value,
                    })

    ccre_path = project / "external_tracks/ucsc/encodeCcreCombined.hg38.bb"
    if not ccre_path.is_file():
        raise FileNotFoundError(ccre_path)
    ccre_rows = []
    with pyBigWig.open(str(ccre_path)) as bigbed:
        chrom_sizes = bigbed.chroms()
        for segment in segments:
            chrom = segment["chrom"]
            source_chrom = chrom if chrom in chrom_sizes else chrom.removeprefix("chr")
            if source_chrom not in chrom_sizes:
                continue
            entries = bigbed.entries(source_chrom, segment["start"], segment["end"]) or []
            for start, end, annotation in entries:
                annotation = str(annotation or "")
                ccre_rows.append({
                    "segment": segment["index"],
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "name": annotation.split("\t", 1)[0],
                    "annotation": annotation,
                })

    with (output / "signals.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["track", "segment", "chrom", "start", "end", "position", "value"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(signal_rows)
    (output / "ccres.json").write_text(json.dumps(ccre_rows, indent=2) + "\n")
    sources = {
        track: {"source": str(path), "assembly": "hg38",
                "method": "exact BigWig mean over 32 bp target bins; missing values retained"}
        for track, path in tracks.items()
    }
    sources["ccres"] = {
        "source": str(ccre_path), "assembly": "hg38",
        "method": "ENCODE cCRE intervals; no CAGE signal is implied",
    }
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps({
        "segments": len(segments),
        "signal_rows": len(signal_rows),
        "ccre_rows": len(ccre_rows),
        "output_dir": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
