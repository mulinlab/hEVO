#!/usr/bin/env python3
"""Extract sample-specific GPN-MSA ancestry delta embeddings from cached MSA windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


INPUT_WINDOW = 511
INPUT_CENTER = 255
MODEL_WINDOW = 128
FORWARD_CENTER = MODEL_WINDOW // 2
REVERSE_CENTER = FORWARD_CENTER - 1
POOL_SIZE = 5
HIDDEN_SIZE = 768
EXPECTED_AUX_FEATURES = 445
EXPECTED_ROWS_PER_SAMPLE = 10000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def aggregate_feature_matrix(matrix: np.ndarray, topk: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"Expected [n,{HIDDEN_SIZE}] site embeddings, found {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Site embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    valid = np.flatnonzero(np.isfinite(norms) & (norms > 0))
    if not len(valid):
        return np.zeros(3 * HIDDEN_SIZE + 4, dtype=np.float32)

    selected = valid[np.argsort(norms[valid])[-min(topk, len(valid)) :]]
    topk_mean = matrix[selected].mean(axis=0, dtype=np.float64).astype(np.float32)
    valid_matrix = matrix[valid]
    valid_norms = norms[valid]
    global_mean = valid_matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    norm_sum = float(valid_norms.sum())
    weighted_mean = (
        np.average(valid_matrix, axis=0, weights=valid_norms).astype(np.float32)
        if norm_sum > 0
        else global_mean
    )
    scalar_stats = np.asarray(
        [
            len(valid) / max(len(matrix), 1),
            float(valid_norms.mean()),
            float(valid_norms.std()),
            float(valid_norms.max()),
        ],
        dtype=np.float32,
    )
    result = np.concatenate([topk_mean, global_mean, weighted_mean, scalar_stats])
    if result.shape != (3 * HIDDEN_SIZE + 4,) or not np.isfinite(result).all():
        raise FloatingPointError("Aggregated sample vector is invalid")
    return result


def read_sample(
    path: Path, *, max_records: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a sample while hashing and validating its full input file."""
    digest = hashlib.sha256()
    roster_digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    sample_id = population = None
    total_rows = 0
    total_active = 0

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            identity = (str(row["sample"]), str(row["sample_popu"]))
            if sample_id is None:
                sample_id, population = identity
            elif identity != (sample_id, population):
                raise ValueError(f"Mixed sample identity in {path}:{line_number}")

            sequence = str(row["seq"]).upper()
            ref = str(row["ref_base"]).upper()
            alt = str(row["alt_base"]).upper()
            chrom = str(row["chr"])
            position = int(row["center_pos"])
            if len(sequence) != INPUT_WINDOW or sequence[INPUT_CENTER] != alt:
                raise ValueError(f"Invalid sample sequence at {path}:{line_number}")
            if ref not in "ACGT" or alt not in "ACGT" or position <= FORWARD_CENTER:
                raise ValueError(
                    f"Invalid marker at {path}:{line_number}: {chrom}:{position}"
                )
            roster_digest.update(f"{chrom}\t{position}\t{ref}\n".encode("ascii"))
            total_active += int(ref != alt)
            total_rows += 1

            if not max_records or len(records) < max_records:
                start = INPUT_CENTER - FORWARD_CENTER
                seq_alt = sequence[start : start + MODEL_WINDOW]
                if len(seq_alt) != MODEL_WINDOW or seq_alt[FORWARD_CENTER] != alt:
                    raise ValueError(
                        f"Cannot center-crop {MODEL_WINDOW} bp at {path}:{line_number}"
                    )
                seq_ref = seq_alt[:FORWARD_CENTER] + ref + seq_alt[FORWARD_CENTER + 1 :]
                records.append(
                    {
                        "chrom": chrom,
                        "position": position,
                        "ref": ref,
                        "alt": alt,
                        "seq_ref": seq_ref,
                        "seq_alt": seq_alt,
                        "row_index": total_rows - 1,
                    }
                )

    if sample_id is None:
        raise ValueError(f"Empty input file: {path}")
    if total_rows != EXPECTED_ROWS_PER_SAMPLE:
        raise ValueError(
            f"{path}: expected {EXPECTED_ROWS_PER_SAMPLE} rows, found {total_rows}"
        )
    return records, {
        "sample_id": sample_id,
        "population": population,
        "input_rows": total_rows,
        "active_variant_rows": total_active,
        "input_sha256": digest.hexdigest(),
        "locus_roster_sha256": roster_digest.hexdigest(),
        "processed_rows": len(records),
    }


class GPNMSAEncoder:
    """Use the official GPN AutoModel and last-hidden-state center pooling."""

    def __init__(self, model_path: Path):
        from gpn import register_auto_classes

        register_auto_classes("gpn")
        from transformers import AutoModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("GPN-MSA extraction workers must run on a GPU node")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = AutoModel.from_pretrained(str(model_path)).to(self.device).eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self.n_aux_features = int(self.model.config.n_aux_features)
        if self.hidden_size != HIDDEN_SIZE:
            raise ValueError(
                f"Expected hidden size {HIDDEN_SIZE}, found {self.hidden_size}"
            )
        if self.n_aux_features != EXPECTED_AUX_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_AUX_FEATURES} auxiliary features, found {self.n_aux_features}"
            )
        from gpn.data import ReverseComplementer, Tokenizer

        self.tokenizer = Tokenizer()
        self.reverse_complementer = ReverseComplementer()

    @staticmethod
    def _sequence_bytes(sequences: list[str]) -> np.ndarray:
        return np.stack(
            [
                np.frombuffer(sequence.encode("ascii"), dtype="S1")
                for sequence in sequences
            ]
        )

    def embed_batch(
        self,
        records: list[dict[str, Any]],
        aux_forward: np.ndarray,
        aux_reverse: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        result = np.zeros((len(records), HIDDEN_SIZE), dtype=np.float32)
        active_positions = [
            index for index, row in enumerate(records) if row["ref"] != row["alt"]
        ]
        for offset in range(0, len(active_positions), batch_size):
            local_positions = active_positions[offset : offset + batch_size]
            batch = [records[index] for index in local_positions]
            row_indices = np.asarray(
                [row["row_index"] for row in batch], dtype=np.int64
            )
            seq_ref = [row["seq_ref"] for row in batch]
            seq_alt = [row["seq_alt"] for row in batch]

            ref_bytes = self._sequence_bytes(seq_ref)
            alt_bytes = self._sequence_bytes(seq_alt)
            ref_ids_forward = self.tokenizer(ref_bytes).astype(np.int64)
            alt_ids_forward = self.tokenizer(alt_bytes).astype(np.int64)
            ref_ids_reverse = self.tokenizer(
                self.reverse_complementer(ref_bytes, position_axis=1)
            ).astype(np.int64)
            alt_ids_reverse = self.tokenizer(
                self.reverse_complementer(alt_bytes, position_axis=1)
            ).astype(np.int64)

            aux_fwd = np.asarray(aux_forward[row_indices], dtype=np.uint8)
            aux_rev = np.asarray(aux_reverse[row_indices], dtype=np.uint8)
            n = len(batch)
            input_ids = torch.as_tensor(
                np.concatenate(
                    [
                        ref_ids_forward,
                        alt_ids_forward,
                        ref_ids_reverse,
                        alt_ids_reverse,
                    ],
                    axis=0,
                ),
                dtype=torch.long,
                device=self.device,
            )
            aux_features = torch.as_tensor(
                np.concatenate([aux_fwd, aux_fwd, aux_rev, aux_rev], axis=0),
                dtype=torch.long,
                device=self.device,
            )

            with torch.inference_mode():
                hidden = self.model(
                    input_ids=input_ids, aux_features=aux_features
                ).last_hidden_state
            if hidden.ndim != 3 or hidden.shape != (4 * n, MODEL_WINDOW, HIDDEN_SIZE):
                raise RuntimeError(
                    f"Unexpected hidden-state shape: {tuple(hidden.shape)}"
                )

            half = POOL_SIZE // 2
            fwd_left = FORWARD_CENTER - half
            rev_left = REVERSE_CENTER - half
            pooled_forward = hidden[:, fwd_left : fwd_left + POOL_SIZE].mean(dim=1)
            pooled_reverse = hidden[:, rev_left : rev_left + POOL_SIZE].mean(dim=1)
            ref_forward, alt_forward, _, _ = pooled_forward.split(n, dim=0)
            _, _, ref_reverse, alt_reverse = pooled_reverse.split(n, dim=0)
            deltas = ((alt_forward - ref_forward) + (alt_reverse - ref_reverse)) * 0.5
            deltas = deltas.float().cpu().numpy()
            if not np.isfinite(deltas).all():
                raise FloatingPointError("GPN-MSA returned non-finite site embeddings")
            result[np.asarray(local_positions)] = deltas

        return result


def valid_existing_output(
    output_path: Path,
    manifest_path: Path,
    *,
    sample: dict[str, Any],
    preflight: dict[str, Any],
    topk: int,
) -> bool:
    if not output_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        vector = np.load(output_path, allow_pickle=False)
    except Exception:
        return False
    return (
        manifest.get("status") == "complete"
        and manifest.get("sample_id") == sample["sample_id"]
        and manifest.get("population") == sample["population"]
        and manifest.get("input_sha256") == sample["input_sha256"]
        and manifest.get("locus_roster_sha256") == preflight["locus_roster_sha256"]
        and manifest.get("checkpoint_weights_sha256")
        == preflight["checkpoint"]["weights_sha256"]
        and manifest.get("msa_cache_forward_sha256")
        == preflight["msa_window_cache"]["forward_sha256"]
        and manifest.get("msa_cache_reverse_sha256")
        == preflight["msa_window_cache"]["reverse_sha256"]
        and manifest.get("topk") == topk
        and manifest.get("embedding_shape") == [3 * HIDDEN_SIZE + 4]
        and list(vector.shape) == [3 * HIDDEN_SIZE + 4]
        and np.isfinite(vector).all()
        and manifest.get("output_sha256") == sha256_file(output_path)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Process only the first N records per sample (GPU smoke run)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    args = parser.parse_args()

    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        parser.error("worker index must be within a positive worker count")
    if (
        args.batch_size < 1
        or args.topk < 1
        or args.max_samples < 0
        or args.max_records < 0
    ):
        parser.error("batch size/topk must be positive and limits must be nonnegative")
    run_root = args.run_root.expanduser().resolve()
    config_dir = run_root / "config"
    preflight_path = config_dir / "preflight.json"
    sample_manifest_path = config_dir / "sample_manifest.json"
    if not preflight_path.is_file() or not sample_manifest_path.is_file():
        raise FileNotFoundError("Run preflight_gpn_msa_ancestry.py before extraction")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "preflight_complete":
        raise RuntimeError("Preflight report is not complete")
    samples = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    input_files = [
        Path(line)
        for line in (config_dir / "input_files.txt").read_text().splitlines()
        if line
    ]
    if len(input_files) != len(samples):
        raise RuntimeError("Sample manifest and input_files.txt disagree")

    output_dir = (
        (args.output_dir or run_root / "results" / "GPN-MSA_popu_emb")
        .expanduser()
        .resolve()
    )
    manifest_dir = (
        (args.manifest_dir or run_root / "results" / "manifests").expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cache = preflight["msa_window_cache"]
    aux_forward = np.load(cache["forward_path"], mmap_mode="r", allow_pickle=False)
    aux_reverse = np.load(cache["reverse_path"], mmap_mode="r", allow_pickle=False)
    expected_cache_shape = (
        preflight["locus_count"],
        MODEL_WINDOW,
        EXPECTED_AUX_FEATURES // 5,
    )
    if (
        aux_forward.shape != expected_cache_shape
        or aux_reverse.shape != expected_cache_shape
    ):
        raise ValueError(
            f"MSA window cache shape mismatch: {aux_forward.shape}, {aux_reverse.shape}; "
            f"expected {expected_cache_shape}"
        )
    if (
        sha256_file(Path(cache["forward_path"])) != cache["forward_sha256"]
        or sha256_file(Path(cache["reverse_path"])) != cache["reverse_sha256"]
    ):
        raise ValueError("MSA window-cache checksum mismatch")

    model_path = Path(preflight["checkpoint"]["path"])
    encoder = GPNMSAEncoder(model_path)
    assigned = list(enumerate(zip(input_files, samples)))[
        args.worker_index :: args.worker_count
    ]
    if args.max_samples:
        assigned = assigned[: args.max_samples]
    if not assigned:
        raise ValueError(f"Worker {args.worker_index} has no assigned samples")

    print(
        f"[worker {args.worker_index}/{args.worker_count}] {len(assigned)} samples on "
        f"{torch.cuda.get_device_name(0)}; batch_size={args.batch_size}; "
        f"smoke_records={args.max_records or 'all'}",
        flush=True,
    )
    for global_index, (input_path, sample_record) in assigned:
        output_path = output_dir / f"{input_path.stem}.npy"
        manifest_path = manifest_dir / f"{input_path.stem}.json"
        if not args.max_records and valid_existing_output(
            output_path,
            manifest_path,
            sample=sample_record,
            preflight=preflight,
            topk=args.topk,
        ):
            print(f"[skip] {input_path.name}: validated existing output", flush=True)
            continue

        start_time = time.time()
        records, observed = read_sample(input_path, max_records=args.max_records)
        expected = {
            "sample_id": sample_record["sample_id"],
            "population": sample_record["population"],
            "input_rows": sample_record["input_rows"],
            "input_sha256": sample_record["input_sha256"],
            "locus_roster_sha256": sample_record["locus_roster_sha256"],
        }
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Input changed since preflight: {input_path}")
        if observed["locus_roster_sha256"] != preflight["locus_roster_sha256"]:
            raise ValueError(
                f"Sample locus roster does not match the cached MSA windows: {input_path}"
            )

        site_matrix = encoder.embed_batch(
            records,
            aux_forward,
            aux_reverse,
            batch_size=args.batch_size,
        )
        vector = aggregate_feature_matrix(site_matrix, args.topk)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(output_path.suffix + ".partial")
        with temporary_output.open("wb") as handle:
            np.save(handle, vector, allow_pickle=False)
        os.replace(temporary_output, output_path)

        manifest = {
            "status": "smoke_complete" if args.max_records else "complete",
            "sample_index": global_index,
            "input_file": str(input_path.resolve()),
            "input_sha256": observed["input_sha256"],
            "sample_id": observed["sample_id"],
            "population": observed["population"],
            "input_rows": observed["input_rows"],
            "active_variant_rows": observed["active_variant_rows"],
            "processed_rows": observed["processed_rows"],
            "locus_roster_sha256": observed["locus_roster_sha256"],
            "model": "GPN-MSA",
            "model_path": str(model_path),
            "checkpoint_config_sha256": preflight["checkpoint"]["config_sha256"],
            "checkpoint_weights_sha256": preflight["checkpoint"]["weights_sha256"],
            "gpn_version": importlib.metadata.version("gpn"),
            "transformers_version": importlib.metadata.version("transformers"),
            "msa_path": preflight["msa"]["path"],
            "msa_sha256": preflight["msa"].get("sha256"),
            "msa_total_species_including_human": preflight["msa"][
                "total_species_including_human"
            ],
            "msa_auxiliary_species": preflight["msa"]["auxiliary_species"],
            "msa_cache_forward_sha256": cache["forward_sha256"],
            "msa_cache_reverse_sha256": cache["reverse_sha256"],
            "sample_sequence": "GT[0] haplotype; focal reference vs sample allele",
            "window_bp": MODEL_WINDOW,
            "forward_center_index": FORWARD_CENTER,
            "reverse_center_index": REVERSE_CENTER,
            "center_pool_size": POOL_SIZE,
            "pooling": "mean last_hidden_state over five variant-centered positions per strand",
            "strand_aggregation": "mean of forward and reverse-complement ref/alt deltas",
            "site_embedding": "alternate_minus_reference",
            "aggregation": "topk_mean_global_mean_norm_weighted_mean_four_scalar_stats",
            "topk": args.topk,
            "embedding_shape": list(vector.shape),
            "embedding_dtype": str(vector.dtype),
            "output_sha256": sha256_file(output_path),
            "batch_size": args.batch_size,
            "device": torch.cuda.get_device_name(0),
            "host": socket.gethostname(),
            "elapsed_seconds": time.time() - start_time,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        atomic_json(manifest_path, manifest)
        print(
            f"[done {global_index + 1}/1500] {input_path.name}: "
            f"active={observed['active_variant_rows']} processed={observed['processed_rows']} "
            f"shape={vector.shape} norm={np.linalg.norm(vector):.6g} "
            f"elapsed={manifest['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
