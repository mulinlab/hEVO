#!/usr/bin/env python3
"""Measure GPN-Star NT test-set embedding efficiency for Fig. 5 S44/S43."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch


GPN_STAR_MODELS = {"GPN-Star-M", "GPN-Star-P", "GPN-Star-V"}
EXPECTED_BACKBONE_PARAMS = 202_228_865
EXPECTED_EMBEDDING_DIM = 1024
DEFAULT_S17 = (
    Path(__file__).resolve().parents[3]
    / "tables"
    / "S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv"
)
DEFAULT_TEST = (
    Path(__file__).resolve().parents[3]
    / "datasets"
    / "NT_benchmark_v2"
    / "_combined"
    / "test.csv"
)
DEFAULT_TRAIN = (
    Path(__file__).resolve().parents[3]
    / "datasets"
    / "NT_benchmark_v2"
    / "_combined"
    / "train.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "results" / "Probe_NT_Pareto"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_s17(path: Path, model: str) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") != model:
                continue
            task = str(row["dataset"])
            if task in scores:
                raise ValueError(f"Duplicate S17 score for {model}/{task}")
            scores[task] = {
                "mean": float(row["mcc_mean"]),
                "lower": float(row["mcc_lower"]),
                "upper": float(row["mcc_upper"]),
            }
    if len(scores) != 18:
        raise ValueError(f"Expected 18 S17 results for {model}, found {len(scores)}")
    return scores


def read_train_class_counts(path: Path) -> dict[str, set[int]]:
    labels: dict[str, set[int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            task = row.get("SeqID", "").split("|", 1)[0]
            if not task:
                raise ValueError(f"Missing task prefix in {path}")
            labels.setdefault(task, set()).add(int(row["label"]))
    return labels


def read_test_records(
    path: Path, tasks: set[str]
) -> dict[str, list[dict[str, object]]]:
    records = {task: [] for task in tasks}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seq_id = str(row.get("SeqID", ""))
            task = seq_id.split("|", 1)[0]
            if task not in records:
                raise ValueError(f"Unexpected NT test task {task!r} in {path}")
            sequence = str(row["seq"]).strip().upper()
            start, end = int(row["start"]), int(row["end"])
            if not sequence or end - start != len(sequence):
                raise ValueError(
                    f"Invalid sequence/interval at {seq_id}: {start}-{end}, len={len(sequence)}"
                )
            records[task].append(
                {
                    "SeqID": seq_id,
                    "seq": sequence,
                    "label": int(row["label"]),
                    "chrom": str(row["chrom"]).strip().removeprefix("chr"),
                    "start": start,
                    "end": end,
                }
            )
    empty = sorted(task for task, group in records.items() if not group)
    if empty:
        raise ValueError(f"NT test input has no records for tasks: {empty}")
    return records


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def profile_task(
    extractor, records: list[dict[str, object]], batch_size: int, device: torch.device
) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_memory = int(torch.cuda.memory_allocated(device))
    started = time.perf_counter()
    embedding_dim = None
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        embedding = extractor.get_embedding_for_intervals(batch, verify_sequences=True)
        if embedding.shape[0] != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings, got shape {tuple(embedding.shape)}"
            )
        observed_dim = int(embedding.shape[1])
        if embedding_dim is None:
            embedding_dim = observed_dim
        elif embedding_dim != observed_dim:
            raise RuntimeError(
                f"Embedding dimension changed within a task: {embedding_dim} to {observed_dim}"
            )
        del embedding
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if elapsed <= 0 or embedding_dim is None:
        raise RuntimeError("Invalid GPN-Star efficiency measurement")
    peak_increment_mb = (
        int(torch.cuda.max_memory_allocated(device)) - start_memory
    ) / (1024**2)
    fps = len(records) / elapsed
    return {
        "backbone_throughput_fps": fps,
        "backbone_throughput_bps": fps * 1024,
        "backbone_latency_ms": elapsed * 1000 / len(records),
        "backbone_peak_vram_mb": peak_increment_mb,
        "embedding_dim": float(embedding_dim),
        "sequence_length": 1024.0,
        "projection_dim": float(embedding_dim),
        "backbone_flops_G": None,
        "elapsed_seconds": elapsed,
        "n_samples": float(len(records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=sorted(GPN_STAR_MODELS))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--msa-store", type=Path, action="append", required=True)
    parser.add_argument("--msa-species", type=int, action="append", required=True)
    parser.add_argument("--msa-dataset", required=True)
    parser.add_argument("--msa-revision", required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--s17-table", type=Path, default=DEFAULT_S17)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--center-window-size", type=int, default=1000)
    args = parser.parse_args()

    if len(args.msa_store) != len(args.msa_species) or not args.msa_store:
        raise ValueError("Each --msa-store requires a matching --msa-species value")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("GPN-Star NT profiling requires CUDA")
    for path in [args.model_dir, *args.msa_store]:
        if not path.exists():
            raise FileNotFoundError(path)

    with args.s17_table.open("r", encoding="utf-8-sig", newline="") as handle:
        expected_tasks = {
            str(row["dataset"])
            for row in csv.DictReader(handle)
            if row.get("model") == args.model_id
        }
    if len(expected_tasks) != 18:
        raise ValueError(
            f"S17 must contain all 18 NT tasks for {args.model_id}; got {len(expected_tasks)}"
        )
    s17_scores = read_s17(args.s17_table, args.model_id)
    classes = read_train_class_counts(args.train)
    task_records = read_test_records(args.test, expected_tasks)
    if set(classes) != expected_tasks:
        raise ValueError(
            f"Training task set does not match S17: missing={sorted(expected_tasks - set(classes))}, extra={sorted(set(classes) - expected_tasks)}"
        )
    if any(len(classes[task]) < 2 for task in expected_tasks):
        raise ValueError("Every NT task must have at least two training labels")

    import sys

    evaluate_dir = Path(__file__).resolve().parents[1]
    if str(evaluate_dir) not in sys.path:
        sys.path.insert(0, str(evaluate_dir))
    from adapters.extract_embeddings_classes import ExtractEmbeddingForGPNStar

    device = torch.device("cuda:0")
    extractor = ExtractEmbeddingForGPNStar(
        model_path=str(args.model_dir),
        model_name=args.model_id,
        msa_paths=[str(path) for path in args.msa_store],
        msa_species=args.msa_species,
        window_size=args.window_size,
        center_window_size=args.center_window_size,
    )
    if extractor.device.type != device.type:
        raise RuntimeError(
            f"Expected single-GPU device {device}, got {extractor.device}"
        )
    backbone_params = sum(
        parameter.numel() for parameter in extractor.model.parameters()
    )
    if backbone_params != EXPECTED_BACKBONE_PARAMS:
        raise ValueError(f"Unexpected GPN-Star parameter count: {backbone_params}")

    model_config_path = args.model_dir / "config.json"
    weight_path = args.model_dir / "model.safetensors"
    code_path = Path(__file__).resolve()
    code_sha = sha256(code_path)
    input_hashes = {
        str(path): sha256(path) for path in [args.train, args.test, args.s17_table]
    }
    model_hashes = {
        "config_sha256": sha256(model_config_path),
        "weights_sha256": sha256(weight_path),
    }
    metadata = {
        "profile": "GPN-Star frozen coordinate-aware embedding inference on the NT test split",
        "protocol": {
            "batch_size": args.batch_size,
            "window_size": args.window_size,
            "center_window_size": args.center_window_size,
            "max_length_for_bps": 1024,
            "measurement": "synchronized wall time over all NT test rows; peak allocated CUDA memory above loaded-model baseline",
            "embedding_readout": "mean over centered sequence positions, averaging forward/reverse embeddings",
        },
        "model_id": args.model_id,
        "model_dir": str(args.model_dir),
        "model_revision": args.model_revision,
        "model_hashes": model_hashes,
        "backbone_parameters": backbone_params,
        "msa_dataset": args.msa_dataset,
        "msa_revision": args.msa_revision,
        "msa_stores": [str(path) for path in args.msa_store],
        "msa_species_counts": args.msa_species,
        "inputs_sha256": input_hashes,
        "s17_performance_reused": True,
        "profiler_script": str(code_path),
        "profiler_sha256": code_sha,
    }

    summary = {}
    for task in sorted(expected_tasks):
        records = task_records[task]
        efficiency = profile_task(extractor, records, args.batch_size, device)
        if int(efficiency["embedding_dim"]) != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected embedding dimension for {args.model_id}/{task}: {efficiency['embedding_dim']}"
            )
        n_classes = len(classes[task])
        probe_params = EXPECTED_EMBEDDING_DIM * n_classes + n_classes
        result = {
            "performance": {
                "mcc": [
                    s17_scores[task]["mean"],
                    s17_scores[task]["lower"],
                    s17_scores[task]["upper"],
                ]
            },
            "efficiency": {
                "backbone_params_M": backbone_params / 1_000_000,
                "probe_params_M": probe_params / 1_000_000,
                "total_params_M": (backbone_params + probe_params) / 1_000_000,
                "backbone_flops_G": None,
                "backbone_throughput_fps": efficiency["backbone_throughput_fps"],
                "backbone_throughput_bps": efficiency["backbone_throughput_bps"],
                "backbone_latency_ms": efficiency["backbone_latency_ms"],
                "backbone_peak_vram_mb": efficiency["backbone_peak_vram_mb"],
                "embedding_dim": efficiency["embedding_dim"],
                "projection_dim": efficiency["projection_dim"],
                "sequence_length": efficiency["sequence_length"],
            },
        }
        output = (
            args.output_root
            / task
            / args.model_id
            / f"results_{args.model_id}_{task}.json"
        )
        atomic_json(
            output,
            {
                args.model_id: {task: {"no_projection": result}},
                "_run_metadata": metadata,
            },
        )
        summary[task] = {
            "n_samples": int(efficiency["n_samples"]),
            "elapsed_seconds": efficiency["elapsed_seconds"],
            "throughput_fps": efficiency["backbone_throughput_fps"],
            "throughput_bps": efficiency["backbone_throughput_bps"],
            "latency_ms": efficiency["backbone_latency_ms"],
            "peak_vram_mb": efficiency["backbone_peak_vram_mb"],
            "output": str(output),
        }
        print(
            json.dumps(
                {"model": args.model_id, "task": task, **summary[task]}, sort_keys=True
            ),
            flush=True,
        )

    metadata["task_measurements"] = summary
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.manifest, metadata)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model": args.model_id,
                "task_count": len(summary),
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
