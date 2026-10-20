#!/usr/bin/env python3
"""Query UCSC CADD v1.7 hg38 tracks for all retained display intervals."""

import argparse
import json
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    segments = json.loads(a.segments.read_text())["segments"]
    cache = a.output / "cadd_api"
    cache.mkdir(exist_ok=True)

    def fetch(task):
        i, allele = task
        r = segments[i]
        track = f"cadd1_7_{allele}"
        file = cache / f"{i}_{allele}.json"
        url = f"https://api.genome.ucsc.edu/getData/track?genome=hg38;track={track};chrom={r['chrom']};start={r['start']};end={r['end']}"
        if file.exists():
            j = json.loads(file.read_text())
        else:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            j = response.json()
            assert track in j, j
            file.write_text(json.dumps(j))
        assert j["chromSize"] == 58617616 and not j.get("maxItemsLimit")
        values = np.full(r["end"] - r["start"], np.nan)
        for row in j[track]:
            values[
                max(row["start"], r["start"]) - r["start"] : min(row["end"], r["end"])
                - r["start"]
            ] = row["value"]
        return i, allele, values, url

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(
            pool.map(fetch, [(i, a) for i in range(len(segments)) for a in "ACGT"])
        )
    rows = []
    for i, r in enumerate(segments):
        x = np.stack([v for j, a, v, u in result if j == i])
        valid = np.isfinite(x).all(axis=0)
        v = np.max(x, axis=0)
        v[~valid] = np.nan
        for s in range(r["start"], r["end"], 32):
            e = min(s + 32, r["end"])
            chunk = v[s - r["start"] : e - r["start"]]
            value = float(np.nanmax(chunk)) if np.isfinite(chunk).any() else None
            rows.append(
                dict(
                    track="cadd",
                    segment=i,
                    chrom=r["chrom"],
                    start=s,
                    end=e,
                    position=(s + e) / 2,
                    value=value,
                )
            )
    with (a.output / "cadd.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    (a.output / "cadd_sources.json").write_text(
        json.dumps(
            {
                "source": "UCSC hg38 CADD v1.7 A/C/G/T BigWig API",
                "method": "per-base maximum across four allele tracks; maximum within each 32 bp display bin; incomplete allele coverage excluded",
                "urls": [u for i, a, v, u in result],
            },
            indent=2,
        )
        + "\n"
    )
    print("CADD complete", len(rows), flush=True)


if __name__ == "__main__":
    main()
