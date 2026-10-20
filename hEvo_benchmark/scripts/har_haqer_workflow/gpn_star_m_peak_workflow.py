#!/usr/bin/env python3
"""Train and evaluate one frozen GPN-Star-M HAQER scoring head on HPC."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
HPC_HOME = Path(os.environ.get("HPC_HOME", str(Path.home())))
DNAMODELS_ROOT = Path(
    os.environ.get("DNAMODELS_BENCHMARK_ROOT", str(HPC_HOME / "DNAModelsBenchmark"))
)
GPN_ASSETS_ROOT = Path(
    os.environ.get("GPN_STAR_ASSETS_ROOT", str(HPC_HOME / "gpn_star_assets"))
)
RUN_ROOT = Path(
    os.environ.get(
        "GPN_STAR_PEAK_RUN",
        str(DNAMODELS_ROOT / "runs/gpn_star_m_har_haqer_v1"),
    )
)
DATA_CACHE = Path(
    os.environ.get(
        "HAQER_DATA_CACHE",
        str(HPC_HOME / "hEvo_runtime/HAQER_golden_windwos"),
    )
)
PREPARED = DATA_CACHE / "prepared"
FASTA = Path(
    os.environ.get(
        "HAQER_FASTA",
        str(DNAMODELS_ROOT / "datasets/hg38.fa"),
    )
)
HAQER_BED = BENCHMARK_ROOT / "datasets/HAQER_golden_windwos/haqer.hg38.bed"
HAR_BED = BENCHMARK_ROOT / "datasets/Evolution/HAR/HARs_combined.bed"
MODEL_DIR = Path(
    os.environ.get(
        "GPN_STAR_MODEL_DIR",
        str(GPN_ASSETS_ROOT / "models/GPN-Star-M"),
    )
)
MSA_243 = Path(
    os.environ.get(
        "GPN_STAR_MSA_243",
        str(GPN_ASSETS_ROOT / "msa/hg38_cactus447way/243/all.zarr"),
    )
)
MSA_204 = Path(
    os.environ.get(
        "GPN_STAR_MSA_204",
        str(GPN_ASSETS_ROOT / "msa/hg38_cactus447way/204/all.zarr"),
    )
)
MODEL_REVISION = "0d8f5164fba8e8108200c160370b86652bfb6244"
MSA_REVISION = "f523f0467bc143b62fb5b3b9a91c510e8dc2a7f6"
MSA_DATASET = "songlab/hg38_cactus447way"
FEATURE_ROOT = RUN_ROOT / "features/gpn_star_m"
SEEDS = (20260515, 20260516, 20260517)
SPLITS = ("train", "val", "test")
CHROMS = {
    "train": {f"chr{i}" for i in range(1, 13)},
    "val": {f"chr{i}" for i in range(13, 18)},
    "test": {f"chr{i}" for i in range(18, 23)},
}
CALLERS = {
    "normal": (256, 8, 0.70, 0.15, 1024, 256),
    "strict": (512, 8, 0.75, 0.25, 3072, 512),
    "looser_1": (384, 8, 0.72, 0.18, 2048, 384),
    "balanced_2": (448, 10, 0.74, 0.24, 3072, 448),
}
WINDOW_BP = 512
WINDOW_STEP = 64
MODEL_CONTEXT_BP = 1024
MODEL_POOL_BP = 512
FLANK_BP = 15000
SMOOTH_BUFFER_BP = 8192
WINDOW_SHARDS = 8
SCAN_SHARDS = 48
EMBED_BATCH_SIZE = 8
PROGRESS_EVERY_BATCHES = 10


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    temp.replace(path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fasta_lengths() -> dict[str, int]:
    import pysam

    with pysam.FastaFile(str(FASTA)) as fasta:
        return dict(zip(fasta.references, fasta.lengths))


def read_bed(path: Path, prefix: str | None = None) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                fields = line.split()
            chrom, start, end = fields[:3]
            start, end = int(start), int(end)
            if end <= start:
                raise ValueError(f"{path}:{line_number}: invalid half-open interval")
            row_id = (
                fields[3]
                if len(fields) > 3
                else f"{prefix or path.stem}_{line_number:06d}"
            )
            rows.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "id": row_id,
                    "line": line_number,
                }
            )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate interval IDs in {path}")
    return rows


def split_for_chrom(chrom: str) -> str | None:
    for split, chromosomes in CHROMS.items():
        if chrom in chromosomes:
            return split
    return None


def fasta_chromosome_name(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def split_stem(task: str, split: str) -> str:
    return f"14_{task.upper()}_peakshape_{split}_512.lmdb"


def data_dir(task: str, split: str) -> Path:
    return FEATURE_ROOT / f"{task}_peakshape_{split}"


def load_array(directory: Path, stem: str, suffix: str) -> np.ndarray:
    return np.load(directory / f"{stem}.{suffix}.npy", allow_pickle=True)


def read_har_eval_records(fasta) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    har = read_bed(HAR_BED, "HAR")
    haqer = read_bed(HAQER_BED, "HAQER")
    all_known = defaultdict(list)
    for row in har + haqer:
        all_known[row["chrom"]].append((row["start"], row["end"]))
    overlap_index = {}
    for chrom, intervals in all_known.items():
        intervals.sort()
        starts = [start for start, _ in intervals]
        prefix_max_end = []
        maximum = -1
        for _, end in intervals:
            maximum = max(maximum, end)
            prefix_max_end.append(maximum)
        overlap_index[chrom] = (starts, prefix_max_end)

    def overlaps_known(chrom: str, start: int, end: int) -> bool:
        if chrom not in overlap_index:
            return False
        starts, max_ends = overlap_index[chrom]
        idx = bisect.bisect_left(starts, end) - 1
        return idx >= 0 and max_ends[idx] > start

    lengths = dict(zip(fasta.references, fasta.lengths))
    rows_by_split = {"val": [], "test": []}
    offsets = (-4096, -2048, -1024, -512, 512, 1024, 2048, 4096)
    counts = defaultdict(int)
    for gold in har:
        split = split_for_chrom(gold["chrom"])
        if split not in rows_by_split:
            continue
        chrom, center_start = gold["chrom"], gold["start"]
        chromosome_length = lengths.get(chrom)
        if chromosome_length is None or center_start + WINDOW_BP > chromosome_length:
            continue
        group_id = f"{gold['id']}@{chrom}:{center_start}-{center_start + WINDOW_BP}"
        sequence = fasta.fetch(chrom, center_start, center_start + WINDOW_BP).upper()
        if len(sequence) != WINDOW_BP or set(sequence) - set("ACGT"):
            continue
        rows_by_split[split].append(
            {
                "chromosome": chrom,
                "start": center_start,
                "end": center_start + WINDOW_BP,
                "size_bp": WINDOW_BP,
                "group_id": group_id,
                "role": "center",
                "offset_bp": 0,
                "sample_id": gold["id"],
                "source": "HAR_gold",
                "split": split,
                "region_type": "center",
                "label_name": "positive",
                "label": 1,
                "sequence": sequence,
            }
        )
        counts[f"{split}_centers"] += 1
        for offset in offsets:
            flank_start = center_start + offset
            flank_end = flank_start + WINDOW_BP
            if flank_start < 0 or flank_end > chromosome_length:
                continue
            if overlaps_known(chrom, flank_start, flank_end):
                continue
            flank_sequence = fasta.fetch(chrom, flank_start, flank_end).upper()
            if len(flank_sequence) != WINDOW_BP or set(flank_sequence) - set("ACGT"):
                continue
            role = "flank" if abs(offset) <= 1024 else "background"
            rows_by_split[split].append(
                {
                    "chromosome": chrom,
                    "start": flank_start,
                    "end": flank_end,
                    "size_bp": WINDOW_BP,
                    "group_id": group_id,
                    "role": role,
                    "offset_bp": offset,
                    "sample_id": f"{gold['id']}_offset_{offset}",
                    "source": "HAR_local_negative",
                    "split": split,
                    "region_type": role,
                    "label_name": role,
                    "label": 0,
                    "sequence": flank_sequence,
                }
            )
            counts[f"{split}_{role}"] += 1

    output = {}
    for split, rows in rows_by_split.items():
        fields = (
            "chromosome",
            "start",
            "end",
            "size_bp",
            "group_id",
            "role",
            "offset_bp",
            "sample_id",
            "source",
            "split",
            "region_type",
            "label_name",
            "label",
            "sequence",
        )
        arrays = {}
        for name in fields:
            values = [row[name] for row in rows]
            if name == "label":
                arrays["labels"] = np.asarray(values, dtype=np.int64)
            elif name == "sequence":
                arrays[name] = np.asarray(values, dtype=f"U{WINDOW_BP}")
            elif name in {"start", "end", "size_bp", "offset_bp"}:
                arrays[name] = np.asarray(values, dtype=np.int64)
            else:
                arrays[name] = np.asarray(values, dtype=object)
        arrays["dataset_index"] = np.arange(len(rows), dtype=np.int64)
        output[split] = arrays
    return output, {
        "bed": str(HAR_BED),
        "bed_sha256": sha256_file(HAR_BED),
        "rows": dict(counts),
    }


def validate_existing_haqer_data() -> dict:
    audit_path = PREPARED / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Prepared HAQER audit missing: {audit_path}")
    audit = read_json(audit_path)
    if not audit.get("gold_sha256") or audit["gold_sha256"] != sha256_file(HAQER_BED):
        raise ValueError("Prepared HAQER data were built from a different gold BED")
    pairwise = audit.get("pairwise", {})
    if any(any(value for value in overlap.values()) for overlap in pairwise.values()):
        raise ValueError(
            f"HAQER chromosome split leakage reported by audit: {pairwise}"
        )
    split_windows = {}
    split_groups = {}
    for split, chroms in CHROMS.items():
        source = PREPARED / f"haqer_peakshape_{split}"
        stem = split_stem("haqer", split)
        required = (
            "sequence",
            "chromosome",
            "start",
            "end",
            "labels",
            "group_id",
            "role",
        )
        arrays = {field: load_array(source, stem, field) for field in required}
        n = len(arrays["labels"])
        if any(len(value) != n for value in arrays.values()):
            raise ValueError(f"Misaligned HAQER arrays in {split}")
        observed = set(np.asarray(arrays["chromosome"]).astype(str).tolist())
        if not observed or not observed <= chroms:
            raise ValueError(
                f"Unexpected chromosome assignment in {split}: {sorted(observed)}"
            )
        if arrays["sequence"].shape != (n,) or any(
            len(str(seq)) != WINDOW_BP for seq in arrays["sequence"]
        ):
            raise ValueError(f"Non-512bp HAQER sequences in {split}")
        starts = np.asarray(arrays["start"], dtype=np.int64)
        ends = np.asarray(arrays["end"], dtype=np.int64)
        if np.any(ends - starts != WINDOW_BP):
            raise ValueError(f"Non-512bp HAQER coordinates in {split}")
        chromosomes = np.asarray(arrays["chromosome"]).astype(str)
        split_windows[split] = set(
            zip(chromosomes.tolist(), starts.tolist(), ends.tolist())
        )
        split_groups[split] = set(np.asarray(arrays["group_id"]).astype(str).tolist())
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        window_overlap = split_windows[left] & split_windows[right]
        group_overlap = split_groups[left] & split_groups[right]
        if window_overlap or group_overlap:
            raise ValueError(
                f"HAQER split leakage between {left} and {right}: "
                f"{len(window_overlap)} windows, {len(group_overlap)} groups"
            )
    gold_rows = read_bed(HAQER_BED, "HAQER")
    known_ids_by_split = {
        split: {
            row["id"] for row in gold_rows if split_for_chrom(row["chrom"]) == split
        }
        for split in SPLITS
    }
    out_of_scope = [row for row in gold_rows if split_for_chrom(row["chrom"]) is None]
    if any(row["chrom"] not in {"chrX", "chrY"} for row in out_of_scope):
        raise ValueError(
            f"Unexpected HAQER gold chromosomes outside chr1-22: {out_of_scope[:5]}"
        )
    if any(
        known_ids_by_split[left] & known_ids_by_split[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Known HAQER IDs overlap across chromosome splits")
    audit["source_gold_ids_by_split"] = {
        split: len(ids) for split, ids in known_ids_by_split.items()
    }
    audit["source_gold_ids_excluded_chrX_chrY"] = len(out_of_scope)
    return audit


def prepare_data() -> None:
    audit = validate_existing_haqer_data()
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        source = PREPARED / f"haqer_peakshape_{split}"
        destination = data_dir("haqer", split)
        destination.mkdir(parents=True, exist_ok=True)
        stem = split_stem("haqer", split)
        for path in source.glob(f"{stem}.*.npy"):
            target = destination / path.name
            if not target.exists() or target.stat().st_size != path.stat().st_size:
                shutil.copy2(path, target)
    import pysam

    with pysam.FastaFile(str(FASTA)) as fasta:
        har_arrays, har_meta = read_har_eval_records(fasta)
    for split, arrays in har_arrays.items():
        destination = data_dir("har", split)
        destination.mkdir(parents=True, exist_ok=True)
        stem = split_stem("har", split)
        for name, value in arrays.items():
            np.save(destination / f"{stem}.{name}.npy", value, allow_pickle=True)
        n = len(arrays["labels"])
        if n == 0 or not np.isin(arrays["labels"], [0, 1]).all():
            raise ValueError(f"Empty or invalid HAR evaluation windows in {split}")
    manifest = {
        "schema": "gpn_star_m_peak_data_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "haqer_source": str(PREPARED),
        "haqer_audit": audit,
        "har_window_evaluation": har_meta,
        "har_negative_definition": (
            "Non-overlapping local flanks at +/-512, 1024, 2048, and 4096 bp; "
            "window metrics describe center-vs-local-background transfer."
        ),
        "split_chromosomes": {name: sorted(chroms) for name, chroms in CHROMS.items()},
    }
    dump_json(RUN_ROOT / "data/prepared_manifest.json", manifest)
    print(
        json.dumps(
            {
                "haqer_rows": {
                    split: audit["splits"][split]["n_rows"] for split in SPLITS
                },
                "har_windows": har_meta["rows"],
                "feature_root": str(FEATURE_ROOT),
            },
            indent=2,
        ),
        flush=True,
    )


class GPNStarEmbedder:
    """Frozen native MSA embeddings, pooled over the central 512 target bases."""

    def __init__(self, device: str = "cuda"):
        import torch
        from gpn.star.data import GenomeMSA
        from gpn.star.embedding import EmbeddingInference

        self.torch = torch
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.msa_paths = [MSA_243, MSA_204]
        self.species = [243, 204]
        self.msas = [
            GenomeMSA(str(path), n_species=count, in_memory=False)
            for path, count in zip(self.msa_paths, self.species)
        ]
        self.inference = EmbeddingInference(
            str(MODEL_DIR),
            self.msas,
            MODEL_CONTEXT_BP,
            center_window_size=MODEL_POOL_BP,
        )
        self.encoder = self.inference.model.to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.encoder.model.config.hidden_size)
        if self.hidden_size != 1024:
            raise ValueError(
                f"GPN-Star-M hidden size expected 1024, got {self.hidden_size}"
            )
        self.tokenizer = self.msas[0].tokenizer
        self.chromosome_names = sorted(
            set.intersection(*(set(msa.data) for msa in self.msas))
        )

    def resolve_chrom(self, chrom: str) -> str:
        without = chrom.removeprefix("chr")
        for candidate in dict.fromkeys((chrom, without, "chr" + without)):
            if all(candidate in msa.data for msa in self.msas):
                return candidate
        raise ValueError(f"Chromosome {chrom} is absent from a GPN-Star-M MSA store")

    def embed(
        self,
        chroms: list[str],
        window_starts: np.ndarray,
        sequences: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        starts = np.asarray(window_starts, dtype=np.int64)
        if len(chroms) != len(starts) or len(sequences) != len(starts):
            raise ValueError("Coordinate, sequence, and chromosome counts differ")
        if np.any(starts < 256):
            raise ValueError(
                "A 1024bp context requires at least 256bp before the 512bp window"
            )
        chroms = [self.resolve_chrom(chrom) for chrom in chroms]
        context_starts = starts - 256
        context_ends = context_starts + MODEL_CONTEXT_BP
        encoded = self.inference.tokenize_function(
            {
                "chrom": chroms,
                "start": context_starts,
                "end": context_ends,
            }
        )
        sequence_tokens = np.asarray(
            self.tokenizer(
                np.stack(
                    [
                        np.frombuffer(sequence.upper().encode("ascii"), dtype="S1")
                        for sequence in sequences
                    ]
                )
            ),
            dtype=np.int64,
        )
        target_tokens = np.asarray(encoded["input_ids_fwd"])[:, 256:768, 0]
        ref_match = np.all(target_tokens == sequence_tokens, axis=1)
        tensors = {
            name: self.torch.as_tensor(value, device=self.device)
            for name, value in encoded.items()
        }
        with self.torch.inference_mode():
            feature_tensor = self.encoder(**tensors).float()
            if feature_tensor.ndim == 3 and feature_tensor.shape[1] == 1:
                feature_tensor = feature_tensor[:, 0, :]
            features = feature_tensor.cpu().numpy()
        if features.shape != (len(starts), self.hidden_size):
            raise RuntimeError(f"Unexpected GPN-Star embedding shape: {features.shape}")
        finite = np.isfinite(features).all(axis=1)
        msa_axis_valid = np.ones(len(starts), dtype=bool)
        for msa in self.msas:
            msa_axis_valid &= np.asarray(
                [
                    int(context_ends[index]) <= int(msa.data[chrom].shape[0])
                    for index, chrom in enumerate(chroms)
                ],
                dtype=bool,
            )
        return features.astype(np.float32, copy=False), (
            ref_match & finite & msa_axis_valid
        )

    def close(self) -> None:
        for msa in self.msas:
            store = getattr(getattr(msa, "f", None), "store", None)
            close = getattr(store, "close", None)
            if callable(close):
                close()


def preflight() -> None:
    import pysam

    if not MODEL_DIR.is_dir() or not (MODEL_DIR / "model.safetensors").is_file():
        raise FileNotFoundError(f"GPN-Star-M model weights missing under {MODEL_DIR}")
    if not MSA_243.is_dir() or not MSA_204.is_dir():
        raise FileNotFoundError("The 243-way and 204-way MSA stores must both exist")
    model_config = json.loads((MODEL_DIR / "config.json").read_text())
    configured_hidden = int(model_config.get("hidden_size", 0))
    if configured_hidden != 1024:
        raise ValueError(
            f"Checkpoint config hidden_size is {configured_hidden}, expected 1024"
        )
    audit = validate_existing_haqer_data()
    with pysam.FastaFile(str(FASTA)) as fasta:
        src = PREPARED / "haqer_peakshape_train"
        stem = split_stem("haqer", "train")
        starts = load_array(src, stem, "start")
        chroms = load_array(src, stem, "chromosome").astype(str)
        sequences = load_array(src, stem, "sequence").astype(str)
        lengths = dict(zip(fasta.references, fasta.lengths))
        rows = []
        for index in np.linspace(
            0, len(starts) - 1, num=min(8, len(starts)), dtype=int
        ):
            chrom, start = chroms[index], int(starts[index])
            sequence = sequences[index]
            if start < 256 or start + 768 > lengths[chrom]:
                raise ValueError(
                    f"Training interval lacks 1024bp context: {chrom}:{start}"
                )
            if fasta.fetch(chrom, start, start + WINDOW_BP).upper() != sequence:
                raise ValueError(
                    f"Prepared sequence differs from hg38 at {chrom}:{start}"
                )
            rows.append((chrom, start, sequence))
    embedder = GPNStarEmbedder()
    expected_chroms = {f"chr{i}" for i in range(1, 23)}
    resolved = {embedder.resolve_chrom(chrom) for chrom in expected_chroms}
    if resolved != expected_chroms and resolved != {str(i) for i in range(1, 23)}:
        raise ValueError(f"Unexpected GPN-Star chromosome keys: {sorted(resolved)}")
    shapes = {}
    msa_lengths = {}
    for msa, species in zip(embedder.msas, (243, 204)):
        lengths_for_store = {}
        for chrom in expected_chroms:
            key = embedder.resolve_chrom(chrom)
            array = msa.data[key]
            if int(array.shape[1]) != species:
                raise ValueError(
                    f"{key} has {array.shape[1]} species; expected {species}"
                )
            fasta_key = chrom if chrom in lengths else chrom.removeprefix("chr")
            if fasta_key not in lengths:
                raise ValueError(f"hg38 FASTA is missing {fasta_key}")
            if species == 243 and int(array.shape[0]) != int(lengths[fasta_key]):
                raise ValueError(
                    f"243-way target MSA {key} length {array.shape[0]} does not match "
                    f"hg38 reference length for {fasta_key}: {lengths[fasta_key]}"
                )
            if species == 204 and int(array.shape[0]) <= 1024:
                raise ValueError(
                    f"204-way source-only MSA {key} is too short: {array.shape[0]}"
                )
            lengths_for_store[chrom] = int(array.shape[0])
        msa_lengths[str(species)] = lengths_for_store
        shapes[str(species)] = {
            chrom: list(msa.data[embedder.resolve_chrom(chrom)].shape)
            for chrom in ("chr1", "chr22")
        }
    source_lengths = msa_lengths["204"]
    source_contexts_in_bounds = 0
    source_contexts_total = 0
    for split in SPLITS:
        source = PREPARED / f"haqer_peakshape_{split}"
        stem = split_stem("haqer", split)
        split_starts = load_array(source, stem, "start").astype(np.int64)
        split_chroms = load_array(source, stem, "chromosome").astype(str)
        in_bounds = np.asarray(
            [
                int(start) + 768 <= source_lengths[fasta_chromosome_name(chrom)]
                for chrom, start in zip(split_chroms, split_starts)
            ],
            dtype=bool,
        )
        source_contexts_total += len(in_bounds)
        source_contexts_in_bounds += int(in_bounds.sum())
        if split == "train" and not in_bounds.all():
            raise ValueError(
                f"{int((~in_bounds).sum())} HAQER training contexts exceed the "
                "204-way source-only MSA coordinate extent"
            )
    scan_source_windows_in_bounds = 0
    scan_source_windows_total = 0
    scan_regions_without_source_coverage = []
    scan_regions = []
    for task, bed_path in (("haqer", HAQER_BED), ("har", HAR_BED)):
        gold_rows = read_bed(bed_path, task.upper())
        for split in ("val", "test"):
            scan_regions.extend(_merge_region_records(task, split, gold_rows, lengths))
    for region in scan_regions:
        source_length = source_lengths[fasta_chromosome_name(region["chrom"])]
        maximum_start = min(int(region["scan_end"]), source_length - 768)
        available = max(
            0, (maximum_start - int(region["scan_start"])) // WINDOW_STEP + 1
        )
        scan_source_windows_in_bounds += available
        scan_source_windows_total += int(region["window_count"])
        if available == 0:
            scan_regions_without_source_coverage.append(region["id"])
    if scan_regions_without_source_coverage:
        raise ValueError(
            "No source-only MSA contexts are available for scan regions: "
            f"{scan_regions_without_source_coverage[:10]}"
        )
    probe_rows = rows[:4]
    values, ref_match = embedder.embed(
        [row[0] for row in probe_rows],
        np.asarray([row[1] for row in probe_rows]),
        [row[2] for row in probe_rows],
    )
    embedder.close()
    if not ref_match.all():
        raise ValueError(
            f"MSA target sequence differs from hg38 for preflight rows {np.flatnonzero(~ref_match).tolist()}"
        )
    if not np.isfinite(values).all() or values.shape != (len(probe_rows), 1024):
        raise ValueError(f"Preflight feature validation failed: shape={values.shape}")
    frozen_count = sum(
        p.numel() for p in embedder.encoder.parameters() if p.requires_grad
    )
    if frozen_count != 0:
        raise ValueError(f"GPN-Star-M backbone has {frozen_count} trainable parameters")
    report = {
        "schema": "gpn_star_m_peak_preflight_v1",
        "status": "PASS",
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": embedder.torch.cuda.get_device_name(embedder.device)
        if embedder.device.type == "cuda"
        else "cpu",
        "torch_version": embedder.torch.__version__,
        "gpn_package_version": __import__(
            "importlib.metadata", fromlist=["version"]
        ).version("gpn"),
        "workflow_sha256": sha256_file(Path(__file__).resolve()),
        "git_revision": subprocess.check_output(
            ["git", "-C", str(BENCHMARK_ROOT.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "model_dir": str(MODEL_DIR),
        "model_revision": MODEL_REVISION,
        "model_config_sha256": sha256_file(MODEL_DIR / "config.json"),
        "model_weights_sha256": sha256_file(MODEL_DIR / "model.safetensors"),
        "msa_dataset": MSA_DATASET,
        "msa_revision": MSA_REVISION,
        "msa_stores": [str(MSA_243), str(MSA_204)],
        "msa_species": [243, 204],
        "msa_concat_species": 447,
        "msa_channel_order": "243-way target alignment, then 204-way source-only alignment",
        "coordinate_mode": "official GenomeMSA direct start/end slicing in both stores",
        "reference_validation_policy": (
            "The 243-way first channel is the hg38 target and is compared base-by-base to FASTA. "
            "The 204-way source-only store has no target row; contexts beyond its stored axis "
            "are marked invalid rather than counted as a score."
        ),
        "msa_root_metadata_sha256": [
            sha256_file(path / ".zgroup") for path in (MSA_243, MSA_204)
        ],
        "sampled_chromosome_shapes": shapes,
        "msa_chromosome_lengths": msa_lengths,
        "source_only_204_length_delta_vs_hg38": {
            chrom: int(source_lengths[chrom] - lengths[chrom])
            for chrom in sorted(expected_chroms)
        },
        "haqer_contexts_within_both_msa_extents": {
            "in_bounds": source_contexts_in_bounds,
            "total": source_contexts_total,
        },
        "heldout_scan_contexts_within_both_msa_extents": {
            "in_bounds": scan_source_windows_in_bounds,
            "total": scan_source_windows_total,
            "regions_without_source_coverage": 0,
        },
        "feature_dimension": int(values.shape[1]),
        "context_bp": MODEL_CONTEXT_BP,
        "pool_bp": MODEL_POOL_BP,
        "frozen_trainable_parameters": frozen_count,
        "reference_rows_checked": len(ref_match),
        "reference_rows_matched": int(ref_match.sum()),
        "features_finite": bool(np.isfinite(values).all()),
        "haqer_source_audit": audit.get("gold_sha256"),
    }
    dump_json(RUN_ROOT / "preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)


def prepare() -> None:
    if not (RUN_ROOT / "preflight.json").is_file():
        raise FileNotFoundError("Run the Slurm preflight stage first")
    prepare_data()


def extract_task_names() -> list[tuple[str, str]]:
    return [
        ("haqer", "train"),
        ("haqer", "val"),
        ("haqer", "test"),
        ("har", "val"),
        ("har", "test"),
    ]


def extract_window_shard(task_index: int) -> None:
    dataset_index = task_index // WINDOW_SHARDS
    shard = task_index % WINDOW_SHARDS
    if dataset_index >= len(extract_task_names()):
        raise ValueError(f"Window extraction task index out of range: {task_index}")
    task, split = extract_task_names()[dataset_index]
    directory = data_dir(task, split)
    stem = split_stem(task, split)
    starts = load_array(directory, stem, "start").astype(np.int64)
    chroms = load_array(directory, stem, "chromosome").astype(str)
    sequences = load_array(directory, stem, "sequence").astype(str)
    all_indices = np.arange(len(starts), dtype=np.int64)
    indices = np.array_split(all_indices, WINDOW_SHARDS)[shard]
    shard_path = RUN_ROOT / "feature_shards" / task / split / f"part_{shard:02d}.npz"
    done_path = shard_path.with_suffix(".json")
    if shard_path.is_file() and done_path.is_file():
        prior = read_json(done_path)
        if (
            prior.get("indices") == len(indices)
            and prior.get("model_revision") == MODEL_REVISION
            and prior.get("msa_revision") == MSA_REVISION
            and prior.get("all_reference_matched") is True
        ):
            print(f"Already complete: {shard_path}", flush=True)
            return
    if len(indices) == 0:
        raise ValueError(f"Empty extraction shard {dataset_index}/{shard}")
    embedder = GPNStarEmbedder()
    values = np.empty((len(indices), embedder.hidden_size), dtype=np.float16)
    ref_matches = np.zeros(len(indices), dtype=bool)
    for offset in range(0, len(indices), EMBED_BATCH_SIZE):
        local = np.arange(offset, min(offset + EMBED_BATCH_SIZE, len(indices)))
        ix = indices[local]
        batch_values, batch_valid = embedder.embed(
            chroms[ix].tolist(), starts[ix], sequences[ix].tolist()
        )
        ref_matches[local] = batch_valid
        if not batch_valid.all():
            raise ValueError(
                f"MSA target/reference mismatch in {task}/{split}, rows "
                f"{ix[np.flatnonzero(~batch_valid)[:10]].tolist()}"
            )
        values[local] = batch_values.astype(np.float16)
        if offset % (EMBED_BATCH_SIZE * PROGRESS_EVERY_BATCHES) == 0:
            print(f"{task}/{split} shard {shard}: {offset}/{len(indices)}", flush=True)
    embedder.close()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite feature values in window extraction")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temp = shard_path.with_name(shard_path.stem + ".partial.npz")
    np.savez(temp, indices=indices, features=values, ref_match=ref_matches)
    temp.replace(shard_path)
    dump_json(
        done_path,
        {
            "task": task,
            "split": split,
            "shard": shard,
            "indices": len(indices),
            "model_revision": MODEL_REVISION,
            "msa_revision": MSA_REVISION,
            "feature_dimension": int(values.shape[1]),
            "all_reference_matched": bool(ref_matches.all()),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )


def merge_window_shards() -> None:
    preflight_path = RUN_ROOT / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("Preflight manifest is missing")
    feature_dim = int(read_json(preflight_path)["feature_dimension"])
    for task, split in extract_task_names():
        directory = data_dir(task, split)
        stem = split_stem(task, split)
        labels = load_array(directory, stem, "labels")
        total = len(labels)
        final_path = directory / f"{stem}.embeddings.npy"
        merged = np.lib.format.open_memmap(
            final_path, mode="w+", dtype=np.float16, shape=(total, feature_dim)
        )
        seen = np.zeros(total, dtype=np.uint8)
        for shard in range(WINDOW_SHARDS):
            part_path = (
                RUN_ROOT / "feature_shards" / task / split / f"part_{shard:02d}.npz"
            )
            if not part_path.is_file():
                raise FileNotFoundError(f"Missing feature shard {part_path}")
            with np.load(part_path, allow_pickle=False) as part:
                indices = part["indices"].astype(np.int64)
                features = part["features"]
                valid = part["ref_match"].astype(bool)
            if features.shape != (len(indices), feature_dim) or not valid.all():
                raise ValueError(f"Invalid shard shape/reference mask: {part_path}")
            if np.any(seen[indices]):
                raise ValueError(f"Overlapping feature shard indices: {part_path}")
            merged[indices] = features
            seen[indices] = 1
        merged.flush()
        if not np.all(seen == 1) or not np.isfinite(np.asarray(merged)).all():
            raise ValueError(
                f"Incomplete or non-finite merged embeddings for {task}/{split}"
            )
        dump_json(
            directory / f"{stem}.embedding_manifest.json",
            {
                "task": task,
                "split": split,
                "rows": total,
                "feature_dimension": feature_dim,
                "dtype": "float16",
                "model_revision": MODEL_REVISION,
                "msa_revision": MSA_REVISION,
                "context_bp": MODEL_CONTEXT_BP,
                "central_pool_bp": MODEL_POOL_BP,
                "window_bp": WINDOW_BP,
                "reference_rows_matched": total,
            },
        )
        del merged
        print(f"Merged {task}/{split}: {total} x {feature_dim}", flush=True)
    dump_json(
        RUN_ROOT / "window_embeddings_complete.json",
        {
            "status": "complete",
            "model_revision": MODEL_REVISION,
            "msa_revision": MSA_REVISION,
            "feature_dimension": feature_dim,
        },
    )


def trainer_script() -> Path:
    return (
        BENCHMARK_ROOT / "scripts/har_haqer_workflow/core/16_train_peakshape_heads.py"
    )


def hardneg_script() -> Path:
    return (
        BENCHMARK_ROOT
        / "scripts/har_haqer_workflow/core/46_bootstrap_peakshape_hardneg_from_embeddings.py"
    )


def call_script() -> Path:
    return BENCHMARK_ROOT / "scripts/har_haqer_workflow/core/11_call_smoothed_peaks.py"


def head_output(head: str, variant: str, seed: int) -> Path:
    return RUN_ROOT / "heads/gpn_star_m/last" / head / variant / str(seed)


def train_head(head: str, variant: str, seed: int) -> None:
    if not (RUN_ROOT / "window_embeddings_complete.json").is_file():
        raise FileNotFoundError("Merged feature arrays are missing")
    destination = head_output(head, variant, seed)
    if (destination / "haqer/metrics.json").is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    embeddings_root = (
        RUN_ROOT / "hardneg/gpn_star_m" if variant == "hardneg" else FEATURE_ROOT
    )
    command = [
        sys.executable,
        str(trainer_script()),
        "--embeddings-root",
        str(embeddings_root),
        "--output-dir",
        str(destination),
        "--tasks",
        "haqer",
        "--head-type",
        head,
        "--epochs",
        "20",
        "--batch-size",
        "2048",
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--hidden-dim",
        "256",
        "--dropout",
        ".1",
        "--patience",
        "5",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--tf32",
        "--margin-center-flank",
        ".5",
        "--margin-center-background",
        ".5",
        "--lambda-bce",
        "1",
        "--lambda-center-flank",
        "1",
        "--lambda-center-background",
        ".5",
        "--selection-metric",
        "window_auprc",
        "--embedding-store-device",
        "cuda",
        "--embedding-store-max-gb",
        "40",
        "--posthoc-negative-calibration",
        "quantile",
        "--posthoc-negative-quantile",
        ".95",
        "--posthoc-negative-target-prob",
        ".50",
    ]
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def head_artifact(head: str, variant: str, seed: int) -> dict:
    task_dir = head_output(head, variant, seed) / "haqer"
    metrics = read_json(task_dir / "metrics.json")
    checkpoint = task_dir / "best_model.pt"
    score = float(metrics["val"]["window_auprc"])
    if not np.isfinite(score) or not checkpoint.is_file():
        raise ValueError(f"Invalid head artifact: {task_dir}")
    return {
        "model": "gpn_star_m",
        "layer": "last",
        "head": head,
        "variant": variant,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "val_auprc": score,
        "metrics": str(task_dir / "metrics.json"),
    }


def _selection_key(item: dict) -> tuple:
    head_order = {"linear": 0, "mlp1": 1, "mlp2": 2}
    return (-item["val_auprc"], head_order[item["head"]])


def train_all() -> None:
    base = []
    for head in ("linear", "mlp1", "mlp2"):
        train_head(head, "base", SEEDS[0])
        base.append(head_artifact(head, "base", SEEDS[0]))
    base.sort(key=_selection_key)
    selected = base[0]
    dump_json(RUN_ROOT / "reports/base_head_leaderboard.json", base)
    dump_json(RUN_ROOT / "reports/base_head_selection.json", selected)

    hard_root = RUN_ROOT / "hardneg/gpn_star_m"
    hard_summary = hard_root / "haqer_bootstrap_summary.json"
    if not hard_summary.is_file():
        command = [
            sys.executable,
            str(hardneg_script()),
            "--source-root",
            str(FEATURE_ROOT),
            "--output-root",
            str(hard_root),
            "--task",
            "haqer",
            "--checkpoint",
            selected["checkpoint"],
            "--stats-dir",
            str(Path(selected["checkpoint"]).parent),
            "--pool-top-k",
            "2000",
            "--pool-min-probability",
            ".5",
            "--hard-negatives-per-group",
            "2",
            "--seed",
            str(SEEDS[0]),
            "--device",
            "cuda",
        ]
        subprocess.run(command, check=True)
    train_head(selected["head"], "hardneg", SEEDS[0])
    hard = head_artifact(selected["head"], "hardneg", SEEDS[0])
    final_primary = hard if hard["val_auprc"] > selected["val_auprc"] else selected
    final_primary["hardneg_val_auprc"] = hard["val_auprc"]
    final_primary["base_val_auprc"] = selected["val_auprc"]
    final_primary["hardneg_retained"] = bool(hard["val_auprc"] > selected["val_auprc"])
    final = [final_primary]
    for seed in SEEDS[1:]:
        train_head(selected["head"], final_primary["variant"], seed)
        final.append(head_artifact(selected["head"], final_primary["variant"], seed))
    dump_json(RUN_ROOT / "selected_head.json", final_primary)
    dump_json(RUN_ROOT / "all_final_heads.json", final)
    dump_json(
        RUN_ROOT / "training_complete.json",
        {
            "status": "complete",
            "selected_head": selected["head"],
            "selected_variant": final_primary["variant"],
            "hardneg_retained": final_primary["hardneg_retained"],
            "seeds": list(SEEDS),
            "selection_metric": "validation window AUPRC",
        },
    )


def _load_scoring_heads(device: str = "cuda"):
    import torch
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gpn_peak_head_trainer", trainer_script()
    )
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    heads = []
    for artifact in read_json(RUN_ROOT / "all_final_heads.json"):
        checkpoint_path = Path(artifact["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = (
            trainer.MLPScoringHead(
                int(checkpoint["input_dim"]),
                int(checkpoint["hidden_dim"]),
                float(checkpoint["dropout"]),
                str(checkpoint["head_type"]),
                int(checkpoint.get("conv_kernel_size", 7)),
            )
            .to(device)
            .eval()
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        stats_dir = checkpoint_path.parent
        mean = np.load(stats_dir / "train_mean.npy").astype(np.float32)
        std = np.load(stats_dir / "train_std.npy").astype(np.float32)
        calibration = checkpoint.get("posthoc_negative_calibration") or {}
        shift = float(calibration.get("inference_logit_shift", 0.0))
        heads.append(
            {
                "seed": int(artifact["seed"]),
                "model": model,
                "mean": mean,
                "std": std,
                "shift": shift,
                "checkpoint": str(checkpoint_path),
            }
        )
    return torch, heads


def _merge_region_records(
    task: str, split: str, records: list[dict], lengths: dict[str, int]
) -> list[dict]:
    grouped = defaultdict(list)
    for row in records:
        if split_for_chrom(row["chrom"]) == split:
            grouped[row["chrom"]].append(row)
    result = []
    for chrom in sorted(grouped, key=lambda x: int(x.removeprefix("chr"))):
        chrom_rows = sorted(
            grouped[chrom], key=lambda row: (row["start"], row["end"], row["id"])
        )
        chromosome_length = int(lengths[chrom])
        spans = []
        for row in chrom_rows:
            left = max(0, row["start"] - FLANK_BP)
            right = min(chromosome_length, row["end"] + FLANK_BP)
            if spans and left <= spans[-1]["end"]:
                spans[-1]["end"] = max(spans[-1]["end"], right)
                spans[-1]["known"].append(row)
            else:
                spans.append({"start": left, "end": right, "known": [row]})
        for index, span in enumerate(spans):
            low = max(256, span["start"] - SMOOTH_BUFFER_BP)
            high = min(chromosome_length - 768, span["end"] + SMOOTH_BUFFER_BP)
            first = ((low + WINDOW_STEP - 1) // WINDOW_STEP) * WINDOW_STEP
            last = (high // WINDOW_STEP) * WINDOW_STEP
            if last < first:
                raise ValueError(
                    f"No valid scan windows around {chrom}:{span['start']}-{span['end']}"
                )
            result.append(
                {
                    "task": task,
                    "split": split,
                    "id": f"{task}_{split}_{chrom}_{index:05d}",
                    "chrom": chrom,
                    "core_start": span["start"],
                    "core_end": span["end"],
                    "scan_start": first,
                    "scan_end": last,
                    "known": [
                        {
                            "chrom": row["chrom"],
                            "start": row["start"],
                            "end": row["end"],
                            "id": row["id"],
                        }
                        for row in span["known"]
                    ],
                    "window_count": (last - first) // WINDOW_STEP + 1,
                }
            )
    return result


def prepare_scan_shards() -> None:
    if not (RUN_ROOT / "training_complete.json").is_file():
        raise FileNotFoundError("Training must complete before preparing scans")
    import pysam

    with pysam.FastaFile(str(FASTA)) as fasta:
        lengths = dict(zip(fasta.references, fasta.lengths))
    haqer = read_bed(HAQER_BED, "HAQER")
    har = read_bed(HAR_BED, "HAR")
    regions = []
    for task, gold in (("haqer", haqer), ("har", har)):
        for split in ("val", "test"):
            regions.extend(_merge_region_records(task, split, gold, lengths))
    if not regions:
        raise ValueError("No validation/test scan regions were generated")
    dump_json(RUN_ROOT / "regions.json", regions)

    bins = [[] for _ in range(SCAN_SHARDS)]
    loads = [0] * SCAN_SHARDS
    for region in sorted(regions, key=lambda row: row["window_count"], reverse=True):
        target = min(range(SCAN_SHARDS), key=lambda i: loads[i])
        bins[target].append(region)
        loads[target] += region["window_count"]
    shard_root = RUN_ROOT / "scan_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for index, rows in enumerate(bins):
        dump_json(shard_root / f"part_{index:02d}.json", rows)
    dump_json(
        RUN_ROOT / "scan_shard_manifest.json",
        {
            "schema": "gpn_star_m_scan_shards_v1",
            "shards": SCAN_SHARDS,
            "regions": len(regions),
            "scan_windows": int(sum(loads)),
            "windows_per_shard": loads,
            "tasks": {
                task: {
                    split: sum(
                        row["task"] == task and row["split"] == split for row in regions
                    )
                    for split in ("val", "test")
                }
                for task in ("haqer", "har")
            },
            "stride_bp": WINDOW_STEP,
            "window_bp": WINDOW_BP,
            "context_bp": MODEL_CONTEXT_BP,
            "pool_bp": MODEL_POOL_BP,
            "edge_smoothing_buffer_bp": SMOOTH_BUFFER_BP,
        },
    )
    print(
        json.dumps(read_json(RUN_ROOT / "scan_shard_manifest.json"), indent=2),
        flush=True,
    )


def scan_starts(region: dict) -> np.ndarray:
    return np.arange(
        int(region["scan_start"]),
        int(region["scan_end"]) + 1,
        WINDOW_STEP,
        dtype=np.int64,
    )


def scan_shard(shard_index: int) -> None:
    if not (RUN_ROOT / "scan_shard_manifest.json").is_file():
        raise FileNotFoundError("Scan shards have not been prepared")
    import pysam
    import torch

    shard_path = RUN_ROOT / "scan_shards" / f"part_{shard_index:02d}.json"
    regions = read_json(shard_path)
    if not regions:
        dump_json(
            RUN_ROOT / "scan_shards" / f"done_{shard_index:02d}.json",
            {"status": "empty"},
        )
        return
    torch.set_num_threads(4)
    embedder = GPNStarEmbedder()
    torch_backend, heads = _load_scoring_heads("cuda")
    fasta = pysam.FastaFile(str(FASTA))
    for region in regions:
        out_path = (
            RUN_ROOT
            / "scans"
            / region["task"]
            / region["split"]
            / f"{region['id']}.npz"
        )
        done_path = out_path.with_suffix(".json")
        if out_path.is_file() and done_path.is_file():
            previous = read_json(done_path)
            if (
                previous.get("region") == region["id"]
                and previous.get("model_revision") == MODEL_REVISION
                and previous.get("msa_revision") == MSA_REVISION
                and previous.get("n_windows") == int(region["window_count"])
            ):
                continue
        starts = scan_starts(region)
        centers = starts + WINDOW_BP // 2
        valid = np.zeros(len(starts), dtype=bool)
        probabilities = np.full((len(heads), len(starts)), np.nan, dtype=np.float32)
        for offset in range(0, len(starts), EMBED_BATCH_SIZE):
            ix_all = np.arange(offset, min(offset + EMBED_BATCH_SIZE, len(starts)))
            coords = starts[ix_all]
            seqs = [
                fasta.fetch(region["chrom"], int(start), int(start) + WINDOW_BP).upper()
                for start in coords
            ]
            canonical = np.asarray(
                [
                    len(seq) == WINDOW_BP and not (set(seq) - set("ACGT"))
                    for seq in seqs
                ],
                dtype=bool,
            )
            if not canonical.any():
                continue
            ix = ix_all[canonical]
            batch_features, reference_valid = embedder.embed(
                [region["chrom"]] * len(ix),
                starts[ix],
                [seqs[j] for j in np.flatnonzero(canonical)],
            )
            valid_ix = ix[reference_valid]
            if not len(valid_ix):
                continue
            valid[valid_ix] = True
            x = torch.as_tensor(
                batch_features[reference_valid], dtype=torch.float32, device="cuda"
            )
            for head_index, item in enumerate(heads):
                z = (
                    x - torch.as_tensor(item["mean"], device="cuda")
                ) / torch.as_tensor(item["std"], device="cuda")
                with torch.inference_mode():
                    probs = (
                        torch.sigmoid(item["model"](z) + item["shift"])
                        .float()
                        .cpu()
                        .numpy()
                    )
                if not np.isfinite(probs).all():
                    raise ValueError(f"Non-finite score in {region['id']}")
                probabilities[head_index, valid_ix] = probs
            if offset % (EMBED_BATCH_SIZE * PROGRESS_EVERY_BATCHES) == 0:
                print(
                    f"shard={shard_index} {region['id']} "
                    f"{offset}/{len(starts)} valid={int(valid.sum())}",
                    flush=True,
                )
        if not valid.any():
            raise ValueError(f"No valid GPN-Star scan windows in {region['id']}")
        tmp = out_path.with_name(out_path.stem + ".partial.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tmp,
            chromosome=np.full(len(starts), region["chrom"]),
            start=starts,
            end=starts + WINDOW_BP,
            center=centers,
            valid_mask=valid,
            probabilities=probabilities,
            seeds=np.asarray([item["seed"] for item in heads], dtype=np.int64),
            stride=np.int64(WINDOW_STEP),
        )
        tmp.replace(out_path)
        dump_json(
            done_path,
            {
                "region": region["id"],
                "task": region["task"],
                "split": region["split"],
                "n_windows": len(starts),
                "n_valid": int(valid.sum()),
                "reference_invalid": int((~valid).sum()),
                "model_revision": MODEL_REVISION,
                "msa_revision": MSA_REVISION,
                "shard": shard_index,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
    fasta.close()
    embedder.close()
    dump_json(
        RUN_ROOT / "scan_shards" / f"done_{shard_index:02d}.json",
        {
            "status": "complete",
            "regions": len(regions),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )


def _caller_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gpn_star_peak_caller", call_script())
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def call_one_region(region: dict, seed: int, mode: str) -> list[dict]:
    caller = _caller_module()
    scan_path = (
        RUN_ROOT / "scans" / region["task"] / region["split"] / f"{region['id']}.npz"
    )
    with np.load(scan_path, allow_pickle=False) as source:
        seeds = source["seeds"].astype(int).tolist()
        score_index = seeds.index(seed)
        starts = source["start"].astype(np.int64)
        ends = source["end"].astype(np.int64)
        centers = source["center"].astype(np.int64)
        valid = source["valid_mask"].astype(bool)
        raw = source["probabilities"][score_index].astype(np.float32)
    sigma, truncate, score_threshold, min_prominence, min_distance, min_width = CALLERS[
        mode
    ]
    smoothed = caller.smooth_signal(
        raw,
        valid,
        caller.gaussian_kernel(sigma / WINDOW_STEP, truncate),
    )
    candidates = caller.find_local_maxima(smoothed, valid)
    candidates = candidates[
        np.asarray(
            [smoothed[index] >= score_threshold for index in candidates], dtype=bool
        )
    ]
    candidates = caller.suppress_nearby_peaks(
        candidates, smoothed, centers, min_distance
    )
    result = []
    for index in candidates.tolist():
        prominence, _, _ = caller.compute_prominence(smoothed, index)
        if prominence < min_prominence:
            continue
        width, left, right = caller.compute_width(
            smoothed, index, prominence, WINDOW_STEP, centers
        )
        if width < min_width:
            continue
        peak = {
            "task": region["task"],
            "split": region["split"],
            "region_id": region["id"],
            "chrom": region["chrom"],
            "peak_start": int(starts[left]),
            "peak_end": int(ends[right]),
            "summit_center": int(centers[index]),
            "raw_score": float(raw[index]),
            "smoothed_score": float(smoothed[index]),
            "prominence": float(prominence),
            "width_bp": float(width),
        }
        if region["core_start"] <= peak["summit_center"] < region["core_end"]:
            result.append(peak)
    return result


def _region_for_gold(regions: list[dict]) -> dict[str, dict]:
    mapping = {}
    for region in regions:
        for gold in region["known"]:
            if gold["id"] in mapping:
                raise ValueError(f"Gold interval appears in two regions: {gold['id']}")
            mapping[gold["id"]] = region
    return mapping


def _score_coverage(
    gold: dict, region: dict, seed: int
) -> tuple[bool, np.ndarray, np.ndarray]:
    scan_path = (
        RUN_ROOT / "scans" / region["task"] / region["split"] / f"{region['id']}.npz"
    )
    with np.load(scan_path, allow_pickle=False) as source:
        seeds = source["seeds"].astype(int).tolist()
        score_index = seeds.index(seed)
        centers = source["center"].astype(np.int64)
        valid = source["valid_mask"].astype(bool)
        probabilities = source["probabilities"][score_index].astype(np.float32)
    mask = (centers >= gold["start"]) & (centers < gold["end"])
    evaluable = bool(mask.any() and valid[mask].any())
    return evaluable, centers[mask], probabilities[mask]


def _task_gold(task: str, split: str) -> list[dict]:
    source = (
        read_bed(HAQER_BED, "HAQER") if task == "haqer" else read_bed(HAR_BED, "HAR")
    )
    return [row for row in source if split_for_chrom(row["chrom"]) == split]


def _peak_metrics(
    task: str, split: str, seed: int, mode: str, regions: list[dict]
) -> tuple[dict, list[dict], dict]:
    selected_regions = [
        region
        for region in regions
        if region["task"] == task and region["split"] == split
    ]
    peaks = []
    for region in selected_regions:
        peaks.extend(call_one_region(region, seed, mode))
    gold = _task_gold(task, split)
    region_for_id = _region_for_gold(selected_regions)
    coverage = {}
    usable = []
    excluded = []
    for row in gold:
        region = region_for_id.get(row["id"])
        if region is None:
            raise ValueError(f"No scan region contains {row['id']}")
        evaluable, _, _ = _score_coverage(row, region, seed)
        coverage[row["id"]] = evaluable
        (usable if evaluable else excluded).append(row)
    hits = {
        row["id"]
        for row in usable
        if any(
            peak["chrom"] == row["chrom"]
            and row["start"] <= peak["summit_center"] < row["end"]
            for peak in peaks
        )
    }
    overlap_hits = {
        row["id"]
        for row in usable
        if any(
            peak["chrom"] == row["chrom"]
            and peak["peak_start"] < row["end"]
            and row["start"] < peak["peak_end"]
            for peak in peaks
        )
    }
    _usable_ids = {row["id"] for row in usable}
    true_peaks = sum(
        any(
            peak["chrom"] == row["chrom"]
            and row["start"] <= peak["summit_center"] < row["end"]
            for row in usable
        )
        for peak in peaks
    )
    false_exclusion = [
        peak
        for peak in peaks
        if any(
            peak["chrom"] == row["chrom"]
            and row["start"] <= peak["summit_center"] < row["end"]
            for row in excluded
        )
    ]
    peak_eval = [peak for peak in peaks if peak not in false_exclusion]
    precision = true_peaks / len(peak_eval) if peak_eval else 0.0
    recall = len(hits) / len(usable) if usable else 0.0
    overlap_false_exclusion = [
        peak
        for peak in peaks
        if any(
            peak["chrom"] == row["chrom"]
            and peak["peak_start"] < row["end"]
            and row["start"] < peak["peak_end"]
            for row in excluded
        )
    ]
    overlap_peak_eval = [peak for peak in peaks if peak not in overlap_false_exclusion]
    overlap_true_peaks = sum(
        any(
            peak["chrom"] == row["chrom"]
            and peak["peak_start"] < row["end"]
            and row["start"] < peak["peak_end"]
            for row in usable
        )
        for peak in overlap_peak_eval
    )
    overlap_precision = (
        overlap_true_peaks / len(overlap_peak_eval) if overlap_peak_eval else 0.0
    )
    overlap_recall = len(overlap_hits) / len(usable) if usable else 0.0
    metrics = {
        "task": task,
        "split": split,
        "seed": seed,
        "caller": mode,
        "n_peaks": len(peak_eval),
        "n_known": len(usable),
        "n_known_excluded": len(excluded),
        "n_known_hits": len(hits),
        "n_true_peaks": true_peaks,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "n_known_interval_overlap_hits": len(overlap_hits),
        "n_interval_overlap_true_peaks": overlap_true_peaks,
        "interval_overlap_precision": overlap_precision,
        "interval_overlap_recall": overlap_recall,
        "interval_overlap_f1": (
            2
            * overlap_precision
            * overlap_recall
            / (overlap_precision + overlap_recall)
            if overlap_precision + overlap_recall
            else 0.0
        ),
    }
    return metrics, peaks, coverage


def _write_tsv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _score_window_set(task: str, split: str, head: dict) -> tuple[dict, np.ndarray]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    import torch

    directory = data_dir(task, split)
    stem = split_stem(task, split)
    path = directory / f"{stem}.embeddings.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    features = np.load(path, mmap_mode="r")
    labels = load_array(directory, stem, "labels").astype(np.int64)
    roles = load_array(directory, stem, "role").astype(str)
    groups = load_array(directory, stem, "group_id").astype(str)
    if len(features) != len(labels):
        raise ValueError(f"Window labels and features differ for {task}/{split}")
    batch_probs = []
    model = head["model"]
    device = next(model.parameters()).device
    mean = torch.as_tensor(head["mean"], device=device)
    std = torch.as_tensor(head["std"], device=device)
    for offset in range(0, len(labels), 16384):
        batch = torch.as_tensor(
            np.asarray(features[offset : offset + 16384], dtype=np.float32),
            device=device,
        )
        with torch.inference_mode():
            probs = (
                torch.sigmoid(model((batch - mean) / std) + head["shift"])
                .float()
                .cpu()
                .numpy()
            )
        batch_probs.append(probs)
    probabilities = np.concatenate(batch_probs)
    if not np.isfinite(probabilities).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Invalid HAR/HAQER transfer scores for {task}/{split}")
    top1 = []
    for group_id in dict.fromkeys(groups.tolist()):
        indices = np.flatnonzero(groups == group_id)
        center = indices[roles[indices] == "center"]
        negative = indices[roles[indices] != "center"]
        if len(center) and len(negative):
            top1.append(
                float(np.max(probabilities[center]) > np.max(probabilities[negative]))
            )
    metrics = {
        "task": task,
        "split": split,
        "seed": int(head["seed"]),
        "window_auprc": float(average_precision_score(labels, probabilities)),
        "window_auroc": float(roc_auc_score(labels, probabilities)),
        "center_top1_rate": float(np.mean(top1)) if top1 else 0.0,
        "n_positive_centers": int(np.sum(roles == "center")),
        "n_negative_windows": int(np.sum(roles != "center")),
        "n_center_groups_with_negative": int(len(top1)),
        "negative_definition": (
            "HAQER peakshape flank/backgrounds"
            if task == "haqer"
            else "non-overlapping local HAR flanks/backgrounds"
        ),
    }
    return metrics, probabilities


def _evaluate_window_metrics(heads: list[dict]) -> list[dict]:
    rows = []
    scores_root = RUN_ROOT / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)
    for head in heads:
        for task, splits in (("haqer", ("val", "test")), ("har", ("val", "test"))):
            for split in splits:
                metrics, probabilities = _score_window_set(task, split, head)
                rows.append(metrics)
                if split == "test":
                    directory = data_dir(task, split)
                    stem = split_stem(task, split)
                    np.savez_compressed(
                        scores_root
                        / f"{task}_{split}_{head['seed']}_window_scores.npz",
                        probabilities=probabilities,
                        labels=load_array(directory, stem, "labels").astype(np.int8),
                        roles=load_array(directory, stem, "role").astype(str),
                        group_ids=load_array(directory, stem, "group_id").astype(str),
                    )
    return rows


def evaluate() -> None:
    if not (RUN_ROOT / "scan_shard_manifest.json").is_file():
        raise FileNotFoundError("Scan manifest is missing")
    regions = read_json(RUN_ROOT / "regions.json")
    for shard in range(SCAN_SHARDS):
        done = RUN_ROOT / "scan_shards" / f"done_{shard:02d}.json"
        if not done.is_file():
            raise FileNotFoundError(f"Scan shard {shard} is incomplete")
    _, heads = _load_scoring_heads("cpu")
    reports = RUN_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    validation_leaderboard = []
    for task in ("haqer", "har"):
        for mode in CALLERS:
            metrics, _, _ = _peak_metrics(task, "val", SEEDS[0], mode, regions)
            validation_leaderboard.append(metrics)
    _write_tsv(reports / "validation_caller_leaderboard.tsv", validation_leaderboard)
    locked = {}
    for task in ("haqer", "har"):
        candidates = [row for row in validation_leaderboard if row["task"] == task]
        locked[task] = sorted(
            candidates,
            key=lambda row: (
                -row["f1"],
                -row["recall"],
                row["n_peaks"],
                list(CALLERS).index(row["caller"]),
            ),
        )[0]
    dump_json(
        RUN_ROOT / "locked_callers.json",
        {
            "selected": locked,
            "selection_split": "validation chromosomes 13-17",
            "selection_seed": SEEDS[0],
            "caller_parameters": CALLERS,
            "tie_break": ["recall", "fewer_peaks", "configured_order"],
        },
    )

    test_metrics = []
    calls_by_key = {}
    for task in ("haqer", "har"):
        mode = locked[task]["caller"]
        for seed in SEEDS:
            metrics, peaks, coverage = _peak_metrics(task, "test", seed, mode, regions)
            test_metrics.append(metrics)
            calls_by_key[(task, seed)] = peaks
            _write_tsv(
                RUN_ROOT / "calls" / task / "test" / str(seed) / f"{mode}.tsv",
                peaks,
            )
    _write_tsv(reports / "test_peak_call_metrics.tsv", test_metrics)

    region_for_gold = {
        task: _region_for_gold(
            [row for row in regions if row["task"] == task and row["split"] == "test"]
        )
        for task in ("haqer", "har")
    }
    hit_rows = []
    for task in ("haqer", "har"):
        mode = locked[task]["caller"]
        for gold in _task_gold(task, "test"):
            region = region_for_gold[task][gold["id"]]
            usable, centers, scores = _score_coverage(gold, region, SEEDS[0])
            row = {
                **gold,
                "region": region["id"],
                "evaluable": usable,
                "n_windows": int(len(centers)),
                "max_calibrated_score": float(np.nanmax(scores))
                if len(scores)
                else None,
            }
            for seed in SEEDS:
                peaks = [
                    peak
                    for peak in calls_by_key[(task, seed)]
                    if peak["region_id"] == region["id"]
                ]
                summits = [
                    peak
                    for peak in peaks
                    if gold["start"] <= peak["summit_center"] < gold["end"]
                ]
                overlaps = [
                    peak
                    for peak in peaks
                    if peak["peak_start"] < gold["end"]
                    and gold["start"] < peak["peak_end"]
                ]
                row[f"hit_{seed}"] = bool(summits) if usable else None
                row[f"overlap_{seed}"] = bool(overlaps)
                row[f"summits_{seed}"] = ",".join(
                    str(peak["summit_center"]) for peak in summits
                )
            primary_peaks = [
                peak
                for peak in calls_by_key[(task, SEEDS[0])]
                if peak["region_id"] == region["id"]
            ]
            if primary_peaks:
                nearest = min(
                    primary_peaks,
                    key=lambda peak: max(
                        gold["start"] - peak["summit_center"],
                        peak["summit_center"] - (gold["end"] - 1),
                        0,
                    ),
                )
                row["nearest_summit"] = nearest["summit_center"]
                row["nearest_summit_distance_bp"] = max(
                    gold["start"] - nearest["summit_center"],
                    nearest["summit_center"] - (gold["end"] - 1),
                    0,
                )
            else:
                row["nearest_summit"] = None
                row["nearest_summit_distance_bp"] = None
            row["caller"] = mode
            hit_rows.append(row)
    _write_tsv(
        reports / "test_known_HAQER_hits.tsv",
        [
            row
            for row in hit_rows
            if row["id"] in {g["id"] for g in _task_gold("haqer", "test")}
        ],
    )
    _write_tsv(
        reports / "test_known_HAR_hits.tsv",
        [
            row
            for row in hit_rows
            if row["id"] in {g["id"] for g in _task_gold("har", "test")}
        ],
    )

    window_rows = _evaluate_window_metrics(heads)
    _write_tsv(reports / "window_metrics.tsv", window_rows)
    summary = {
        "schema": "gpn_star_m_haqer_har_peak_results_v1",
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "msa_dataset": MSA_DATASET,
        "msa_revision": MSA_REVISION,
        "msa_stores": [str(MSA_243), str(MSA_204)],
        "pooling": "official GPN-Star forward/reverse mean; central 512bp mean pooling from 1024bp context",
        "training": {
            "frozen_backbone": True,
            "candidate_heads": ["linear", "mlp1", "mlp2"],
            "selection_metric": "validation window AUPRC",
            "epochs": 20,
            "learning_rate": 1e-3,
            "early_stopping_patience": 5,
            "hard_negative_rounds": 1,
            "seeds": list(SEEDS),
            "primary_seed": SEEDS[0],
            "normalization_fit_split": "train",
            "probability_calibration_split": "validation",
        },
        "scan": {"window_bp": WINDOW_BP, "step_bp": WINDOW_STEP},
        "seeds": list(SEEDS),
        "locked_callers": locked,
        "peak_call_metrics": test_metrics,
        "window_metrics": window_rows,
        "har_window_metric_scope": "held-out HAR centers versus non-overlapping local flanks/background windows",
        "outputs": {
            "validation_callers": str(reports / "validation_caller_leaderboard.tsv"),
            "test_peak_metrics": str(reports / "test_peak_call_metrics.tsv"),
            "haqer_hits": str(reports / "test_known_HAQER_hits.tsv"),
            "har_hits": str(reports / "test_known_HAR_hits.tsv"),
            "window_metrics": str(reports / "window_metrics.tsv"),
        },
    }
    dump_json(reports / "summary.json", summary)
    dump_json(
        RUN_ROOT / "status.json",
        {
            "stage": "complete",
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "preflight",
            "prepare",
            "extract-window",
            "merge-window",
            "train",
            "prepare-scan",
            "scan",
            "evaluate",
        ],
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--shard", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        preflight()
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "extract-window":
        index = args.task_index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        extract_window_shard(index)
    elif args.stage == "merge-window":
        merge_window_shards()
    elif args.stage == "train":
        train_all()
    elif args.stage == "prepare-scan":
        prepare_scan_shards()
    elif args.stage == "scan":
        index = args.shard
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        scan_shard(index)
    elif args.stage == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()
