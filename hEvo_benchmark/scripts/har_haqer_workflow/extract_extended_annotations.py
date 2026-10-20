#!/usr/bin/env python3
"""Extract source-backed annotations for the selected HAQER display segments."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np
import pyBigWig
import prepare_external_tracks_remote as prep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    segments = json.loads(a.segments.read_text())["segments"]
    cache = (
        a.project_root
        / "results/final_har_haqer_panel/formal_main_figure/external_track_cache"
    )
    sources = {
        "h3k27ac": a.project_root
        / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig",
        "phastcons": a.project_root / "external_tracks/ucsc/hg38.phastCons100way.bw",
        "ape_divergence": cache / "haqer.divergenceDensity.bw",
        "gerp": cache / "gerp_conservation_scores.homo_sapiens.GRCh38.bw",
    }
    rows = []
    metadata = {}
    ccres = []
    for key, source in sources.items():
        with pyBigWig.open(str(source)) as bw:
            for i, r in enumerate(segments):
                chrom = (
                    r["chrom"]
                    if r["chrom"] in bw.chroms()
                    else r["chrom"].removeprefix("chr")
                )
                assert bw.chroms()[chrom] == 58617616
                n = math.ceil((r["end"] - r["start"]) / 32)
                edges = np.linspace(r["start"], r["end"], n + 1).astype(int)
                values = bw.stats(chrom, r["start"], r["end"], nBins=n, exact=True)
                for s, e, v in zip(edges[:-1], edges[1:], values):
                    rows.append(
                        dict(
                            track=key,
                            segment=i,
                            chrom=r["chrom"],
                            start=int(s),
                            end=int(e),
                            position=(s + e) / 2,
                            value=v,
                        )
                    )
        metadata[key] = {
            "source": str(source),
            "assembly": "hg38",
            "method": "32 bp target bins; exact mean; missing preserved",
        }
        print("extracted", key, flush=True)
    for i, r in enumerate(segments):
        key = f"seg{i}"
        prep.REGIONS[key] = r
        result = prep.extract_linsight(
            cache / "LINSIGHT.hg19.bw",
            cache / "hg19ToHg38.over.chain.gz",
            key,
            a.output / f"{key}_linsight.tsv",
            n_bins=math.ceil((r["end"] - r["start"]) / 32),
        )
        with (a.output / f"{key}_linsight.tsv").open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                s, e = int(row["start"]), int(row["end"])
                v = float(row["value"])
                rows.append(
                    dict(
                        track="linsight",
                        segment=i,
                        chrom=r["chrom"],
                        start=s,
                        end=e,
                        position=(s + e) / 2,
                        value=v if np.isfinite(v) else None,
                    )
                )
        metadata.setdefault(
            "linsight",
            {
                "source": str(cache / "LINSIGHT.hg19.bw"),
                "assembly": "hg19 -> hg38",
                "method": "unique mapping and reciprocal round-trip >=95% per bin; missing bins retained",
                "segments": [],
            },
        )["segments"].append(result)
    print("extracted linsight", flush=True)
    source = cache / "constraint_z_genome_1kb.qc.download.txt.gz"
    with gzip.open(source, "rt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["chrom"] != "chr19":
                continue
            s, e = int(row["start"]), int(row["end"])
            for i, r in enumerate(segments):
                if s < r["end"] and e > r["start"]:
                    try:
                        v = float(row["z"])
                    except ValueError:
                        v = None
                    rows.append(
                        dict(
                            track="gnocchi",
                            segment=i,
                            chrom="chr19",
                            start=s,
                            end=e,
                            position=(s + e) / 2,
                            value=v,
                        )
                    )
    metadata["gnocchi"] = {
        "source": str(source),
        "assembly": "GRCh38",
        "method": "native 1 kb z windows; no interpolation",
    }
    with pyBigWig.open(
        str(a.project_root / "external_tracks/ucsc/encodeCcreCombined.hg38.bb")
    ) as bb:
        for i, r in enumerate(segments):
            for s, e, annotation in bb.entries(r["chrom"], r["start"], r["end"]) or []:
                ccres.append(
                    dict(
                        segment=i,
                        chrom=r["chrom"],
                        start=s,
                        end=e,
                        annotation=annotation,
                        name=annotation.split("\t")[0],
                    )
                )
    metadata["ccres"] = {
        "source": str(
            a.project_root / "external_tracks/ucsc/encodeCcreCombined.hg38.bb"
        ),
        "assembly": "hg38",
        "method": "ENCODE cCRE intervals; no CAGE claim",
    }
    with (a.output / "signals.tsv").open("w") as f:
        writer = csv.DictWriter(
            f,
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
        writer.writerows(rows)
    (a.output / "ccres.json").write_text(json.dumps(ccres, indent=2) + "\n")
    (a.output / "sources.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("complete", len(rows), len(ccres), flush=True)


if __name__ == "__main__":
    main()
