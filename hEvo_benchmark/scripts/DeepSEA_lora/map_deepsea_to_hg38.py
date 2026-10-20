#!/usr/bin/env python3
"""Map DeepSEA sequence windows to hg38 for native GPN-Star-M MSA lookup.

The DeepSEA CSVs contain sequences but no coordinates. This utility builds a
minimap2 index through mappy, prepares sparse byte-offset checkpoints and
memory-mapped coordinate arrays, then maps disjoint row ranges in Slurm array
tasks. Only confident primary-chromosome mappings are marked valid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


HPC_HOME = Path(os.environ.get("HPC_HOME", str(Path.home())))
DNAMODELS_ROOT = Path(
    os.environ.get("DNAMODELS_BENCHMARK_ROOT", str(HPC_HOME / "DNAModelsBenchmark"))
)
DEFAULT_DATASET = Path(
    os.environ.get("DEEPSEA_DATA_ROOT", str(HPC_HOME / "hEvo/dataset/deepsea"))
)
DEFAULT_REFERENCE = Path(
    os.environ.get("HG38_FASTA", str(HPC_HOME / "hEvo/dataset/hg38.fa"))
)
DEFAULT_OUTPUT = Path(
    os.environ.get(
        "GPN_DEEPSEA_COORDINATE_ROOT",
        str(DNAMODELS_ROOT / "runs/deepsea_gpn_star_m_v1/coordinates"),
    )
)
SPLITS = {
    "train": ("train_out.csv", "train_labels.npy"),
    "valid": ("valid_out.csv", None),
    "test": ("test_out.csv", "test_labels.npy"),
}
CHROMS = [*(f"chr{i}" for i in range(1, 23)), "chrX", "chrY", "chrM"]
CHROM_TO_ID = {chrom: i + 1 for i, chrom in enumerate(CHROMS)}
DTYPE = np.dtype(
    [
        ("chrom_id", "<u2"),
        ("start", "<i8"),
        ("strand", "i1"),
        ("mapq", "u1"),
        ("qcov", "<f4"),
        ("identity", "<f4"),
        ("valid", "u1"),
    ]
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build_index", "prepare", "map", "summarize"])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--offset-stride", type=int, default=10000)
    parser.add_argument("--min-query-coverage", type=float, default=0.50)
    parser.add_argument("--min-identity", type=float, default=0.90)
    parser.add_argument("--minimum-valid-rate", type=float, default=0.80)
    return parser.parse_args()


def count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def build_index(reference: Path, index: Path) -> None:
    import mappy

    index.parent.mkdir(parents=True, exist_ok=True)
    if index.exists() and index.stat().st_size > 0:
        print(
            json.dumps(
                {"index": str(index), "status": "exists", "bytes": index.stat().st_size}
            ),
            flush=True,
        )
        return
    started = time.time()
    aligner = mappy.Aligner(
        str(reference),
        preset="sr",
        best_n=10,
        n_threads=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))),
        fn_idx_out=str(index),
    )
    if not aligner:
        raise RuntimeError(f"Failed to build minimap2 index from {reference}")
    print(
        json.dumps(
            {
                "index": str(index),
                "status": "built",
                "bytes": index.stat().st_size,
                "elapsed_seconds": time.time() - started,
            }
        ),
        flush=True,
    )


def prepare(dataset_root: Path, output_dir: Path, stride: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, (csv_name, labels_name) in SPLITS.items():
        csv_path = dataset_root / csv_name
        if labels_name:
            labels = np.load(dataset_root / labels_name, mmap_mode="r")
            count = int(labels.shape[0])
        else:
            count = count_rows(csv_path)
        offsets = []
        with csv_path.open("rb") as handle:
            row = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if row % stride == 0:
                    offsets.append(offset)
                row += 1
        if row != count:
            raise ValueError(f"{split}: CSV has {row} rows, expected {count}")
        (output_dir / f"{split}.offsets.json").write_text(
            json.dumps(
                {
                    "row_count": count,
                    "stride": stride,
                    "offsets": offsets,
                }
            )
            + "\n"
        )
        coord_path = output_dir / f"{split}.coordinates.npy"
        coords = np.lib.format.open_memmap(
            coord_path, mode="w+", dtype=DTYPE, shape=(count,)
        )
        coords["chrom_id"] = 0
        coords["start"] = -1
        coords["strand"] = 0
        coords["mapq"] = 0
        coords["qcov"] = 0
        coords["identity"] = 0
        coords["valid"] = 0
        coords.flush()
        splits[split] = {
            "csv": str(csv_path),
            "rows": count,
            "coordinates": str(coord_path),
            "offset_checkpoints": len(offsets),
        }
        print(json.dumps({"split": split, **splits[split]}), flush=True)
    manifest = {
        "schema": "deepsea_gpn_star_hg38_coordinates_v1",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "chromosome_id_map": {str(i + 1): name for i, name in enumerate(CHROMS)},
        "coordinate_dtype": DTYPE.descr,
        "splits": splits,
        "offset_stride": stride,
        "mapping_filters": {
            "primary_chromosomes_only": True,
            "minimum_query_coverage": 0.50,
            "minimum_identity": 0.90,
            "minimum_mapq": 1,
        },
    }
    (output_dir / "coordinate_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def map_rows(cli: argparse.Namespace) -> None:
    import mappy

    index = cli.index or (cli.output_dir / "hg38.sr.mmi")
    aligner = mappy.Aligner(str(index), preset="sr", best_n=10)
    if not aligner:
        raise RuntimeError(f"Could not load minimap2 index: {index}")
    if cli.shard_count < 1 or not 0 <= cli.shard_index < cli.shard_count:
        raise ValueError("Invalid shard index/count")
    started = time.time()
    for split, (csv_name, _) in SPLITS.items():
        checkpoint_path = cli.output_dir / f"{split}.offsets.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Run prepare before map: {checkpoint_path}")
        checkpoints = json.loads(checkpoint_path.read_text())
        total = int(checkpoints["row_count"])
        first = total * cli.shard_index // cli.shard_count
        last = total * (cli.shard_index + 1) // cli.shard_count
        if first == last:
            continue
        stride = int(checkpoints["stride"])
        checkpoint_row = (first // stride) * stride
        offset = checkpoints["offsets"][checkpoint_row // stride]
        coords = np.load(cli.output_dir / f"{split}.coordinates.npy", mmap_mode="r+")
        counts = {
            "rows": 0,
            "valid": 0,
            "invalid": 0,
            "no_primary_hit": 0,
            "low_coverage": 0,
            "low_identity": 0,
            "low_mapq": 0,
        }
        with (cli.dataset_root / csv_name).open("rb") as handle:
            handle.seek(offset)
            row_idx = checkpoint_row
            while row_idx < last:
                line = handle.readline()
                if not line:
                    break
                if row_idx >= first:
                    sequence = (
                        line.split(maxsplit=1)[0]
                        .decode("ascii", errors="ignore")
                        .upper()
                    )
                    hits = [
                        hit for hit in aligner.map(sequence) if hit.ctg in CHROM_TO_ID
                    ]
                    counts["rows"] += 1
                    if not hits:
                        counts["no_primary_hit"] += 1
                    else:
                        hits.sort(
                            key=lambda hit: (
                                int(hit.mapq),
                                int(hit.mlen),
                                int(hit.q_en - hit.q_st),
                            ),
                            reverse=True,
                        )
                        hit = hits[0]
                        qcov = (hit.q_en - hit.q_st) / max(1, len(sequence))
                        identity = hit.mlen / max(1, hit.blen)
                        if qcov < cli.min_query_coverage:
                            counts["low_coverage"] += 1
                        elif identity < cli.min_identity:
                            counts["low_identity"] += 1
                        elif hit.mapq < 1:
                            counts["low_mapq"] += 1
                        else:
                            # mappy reports query coordinates in input orientation.
                            # Extend aligned coordinates to the full fixed-width input window.
                            start = (
                                hit.r_st - hit.q_st
                                if hit.strand == 1
                                else hit.r_st - (len(sequence) - hit.q_en)
                            )
                            if start >= 0:
                                coords[row_idx] = (
                                    CHROM_TO_ID[hit.ctg],
                                    start,
                                    hit.strand,
                                    hit.mapq,
                                    qcov,
                                    identity,
                                    1,
                                )
                                counts["valid"] += 1
                            else:
                                counts["invalid"] += 1
                row_idx += 1
        coords.flush()
        counts["invalid"] += counts["rows"] - sum(
            counts[k]
            for k in (
                "valid",
                "no_primary_hit",
                "low_coverage",
                "low_identity",
                "low_mapq",
                "invalid",
            )
        )
        report = {
            "split": split,
            "shard_index": cli.shard_index,
            "shard_count": cli.shard_count,
            "row_start": first,
            "row_end_exclusive": last,
            **counts,
        }
        report_path = (
            cli.output_dir
            / f"shards_{cli.shard_count}"
            / f"{split}.{cli.shard_index:03d}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    print(
        json.dumps(
            {"shard": cli.shard_index, "elapsed_seconds": time.time() - started}
        ),
        flush=True,
    )


def summarize(cli: argparse.Namespace) -> None:
    summaries = {}
    for split in SPLITS:
        reports = sorted(
            (cli.output_dir / f"shards_{cli.shard_count}").glob(f"{split}.*.json")
        )
        if len(reports) != cli.shard_count:
            raise RuntimeError(
                f"{split}: found {len(reports)} shard reports, expected {cli.shard_count}"
            )
        values = [json.loads(path.read_text()) for path in reports]
        totals = {
            key: sum(int(item.get(key, 0)) for item in values)
            for key in (
                "rows",
                "valid",
                "invalid",
                "no_primary_hit",
                "low_coverage",
                "low_identity",
                "low_mapq",
            )
        }
        if totals["rows"] != int(
            json.loads((cli.output_dir / f"{split}.offsets.json").read_text())[
                "row_count"
            ]
        ):
            raise RuntimeError(
                f"{split}: mapping shard row totals do not match CSV count"
            )
        totals["valid_rate"] = totals["valid"] / max(1, totals["rows"])
        if totals["valid_rate"] < cli.minimum_valid_rate:
            raise RuntimeError(
                f"{split}: confident hg38 mapping rate {totals['valid_rate']:.3%} is below {cli.minimum_valid_rate:.1%}"
            )
        coord = np.load(cli.output_dir / f"{split}.coordinates.npy", mmap_mode="r")
        valid_count = int(np.count_nonzero(coord["valid"]))
        if valid_count != totals["valid"]:
            raise RuntimeError(
                f"{split}: coordinate map has {valid_count} valid rows, reports say {totals['valid']}"
            )
        summaries[split] = totals
        print(json.dumps({"split": split, **totals}), flush=True)
    payload = {
        "schema": "deepsea_gpn_star_hg38_mapping_summary_v1",
        "created_at_epoch": time.time(),
        "dataset_root": str(cli.dataset_root),
        "reference": str(cli.reference),
        "index": str(cli.index or (cli.output_dir / "hg38.sr.mmi")),
        "coordinate_root": str(cli.output_dir),
        "shard_count": cli.shard_count,
        "splits": summaries,
    }
    (cli.output_dir / "mapping_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def main() -> None:
    cli = args()
    if cli.mode == "build_index":
        build_index(cli.reference, cli.index or (cli.output_dir / "hg38.sr.mmi"))
    elif cli.mode == "prepare":
        prepare(cli.dataset_root, cli.output_dir, cli.offset_stride)
    elif cli.mode == "map":
        map_rows(cli)
    else:
        summarize(cli)


if __name__ == "__main__":
    main()
