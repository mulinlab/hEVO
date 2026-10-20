#!/usr/bin/env python3
"""Extract coordinate-aware GPN-Star ancestry delta embeddings for one sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

MODEL_NAMES = {"GPN-Star-M", "GPN-Star-P", "GPN-Star-V"}
WINDOW_LENGTH = 511
CENTER_OFFSET = WINDOW_LENGTH // 2
POOL_RADIUS = 2


def aggregate_feature_matrix(final_matrix: np.ndarray, topk: int) -> np.ndarray:
    """Match population_classify.extract_single_sample_emb aggregation exactly."""
    norms = np.linalg.norm(final_matrix, axis=1)
    valid_indices = np.where(np.isfinite(norms) & (norms > 0))[0]
    if len(valid_indices) == 0:
        return np.zeros(final_matrix.shape[1] * 3 + 4, dtype=np.float32)

    actual_k = min(topk, len(valid_indices))
    top_k_idx = valid_indices[np.argsort(norms[valid_indices])[-actual_k:]]
    topk_mean = np.mean(final_matrix[top_k_idx], axis=0).astype(np.float32)
    valid_matrix = final_matrix[valid_indices]
    valid_norms = norms[valid_indices]
    global_mean = np.mean(valid_matrix, axis=0).astype(np.float32)
    weighted_mean = np.average(valid_matrix, axis=0, weights=valid_norms).astype(
        np.float32
    )
    scalar_stats = np.array(
        [
            len(valid_indices) / max(len(final_matrix), 1),
            float(np.mean(valid_norms)),
            float(np.std(valid_norms)),
            float(np.max(valid_norms)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([topk_mean, global_mean, weighted_mean, scalar_stats])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True, choices=sorted(MODEL_NAMES))
    parser.add_argument("--msa-store", type=Path, action="append", required=True)
    parser.add_argument("--msa-species", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def read_sample(path: Path, limit: int = 0) -> tuple[list[dict[str, Any]], str, str]:
    records: list[dict[str, Any]] = []
    sample_id = ""
    population = ""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit and len(records) >= limit:
                break
            row = json.loads(line)
            if not sample_id:
                sample_id = str(row["sample"])
                population = str(row["sample_popu"])
            sequence = str(row["seq"]).upper()
            reference = str(row["ref_base"]).upper()
            allele = str(row["alt_base"]).upper()
            if len(sequence) != WINDOW_LENGTH:
                raise ValueError(
                    f"{path}:{line_number}: expected {WINDOW_LENGTH} bp, found {len(sequence)}"
                )
            if sequence[CENTER_OFFSET] != allele:
                raise ValueError(
                    f"{path}:{line_number}: center base {sequence[CENTER_OFFSET]} "
                    f"does not match sample allele {allele}"
                )
            if reference not in "ACGT" or allele not in "ACGT":
                raise ValueError(
                    f"{path}:{line_number}: unsupported alleles ref={reference}, alt={allele}"
                )
            chrom = str(row["chr"])
            position = int(row["center_pos"])
            start = position - 1 - CENTER_OFFSET
            end = start + WINDOW_LENGTH
            if position <= 0 or start < 0:
                raise ValueError(
                    f"{path}:{line_number}: invalid 1-based interval {chrom}:{position}"
                )
            records.append(
                {
                    "chrom": chrom,
                    "pos": position,
                    "start": start,
                    "end": end,
                    "ref": reference,
                    "alt": allele,
                    "seq_alt": sequence,
                    "seq_ref": sequence[:CENTER_OFFSET]
                    + reference
                    + sequence[CENTER_OFFSET + 1 :],
                }
            )
    if not records:
        raise ValueError(f"No records found in {path}")
    return records, sample_id, population


class GPNStarAncestryExtractor:
    """Use official coordinate MSA inputs and paired target-sequence embeddings."""

    class _HiddenStateEncoder(torch.nn.Module):
        """Return a tensor so torch DataParallel can gather model outputs."""

        def __init__(self, encoder: torch.nn.Module):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, source_ids, target_species):
            return self.encoder(
                input_ids=input_ids,
                source_ids=source_ids,
                target_species=target_species,
            ).last_hidden_state

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        msa_stores: list[Path],
        msa_species: list[int],
    ) -> None:
        from gpn.star.data import GenomeMSA
        from gpn.star.embedding import EmbeddingInference

        if len(msa_stores) != len(msa_species) or not msa_stores:
            raise ValueError(
                "MSA store and species lists must be nonempty and equal length"
            )
        self.model_name = model_name
        self.msa_stores = [str(path.resolve()) for path in msa_stores]
        self.msa_species = [int(value) for value in msa_species]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.genome_msa_list = [
            GenomeMSA(str(path), n_species=count, in_memory=False)
            for path, count in zip(msa_stores, self.msa_species)
        ]
        self.inference = EmbeddingInference(
            model_path=str(model_path),
            genome_msa_list=self.genome_msa_list,
            window_size=WINDOW_LENGTH,
            center_window_size=POOL_RADIUS * 2 + 1,
        )
        base_encoder = self.inference.model.model.to(self.device).eval()
        hidden_state_encoder = self._HiddenStateEncoder(base_encoder).eval()
        self.gpu_count = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(
                hidden_state_encoder, device_ids=list(range(self.gpu_count))
            )
            print(
                f"GPN-Star ancestry extraction uses DataParallel across {self.gpu_count} GPUs"
            )
        else:
            self.encoder = hidden_state_encoder
        # EmbeddingInference delegates sequence encoding to GenomeMSA; the
        # installed official package does not expose tokenizer attributes on
        # the inference wrapper itself.
        self.tokenizer = self.genome_msa_list[0].tokenizer
        self.reverse_complementer = self.genome_msa_list[0].reverse_complementer
        self.hidden_size = int(base_encoder.config.hidden_size)
        if self.hidden_size != 1024:
            raise ValueError(
                f"{model_name} expected hidden_size=1024, got {self.hidden_size}"
            )

    def resolve_chrom(self, chrom: str) -> str:
        candidates = [chrom]
        without_prefix = chrom.removeprefix("chr")
        candidates.extend([without_prefix, "chr" + without_prefix])
        for candidate in dict.fromkeys(candidates):
            if all(candidate in msa.data for msa in self.genome_msa_list):
                return candidate
        raise ValueError(f"Chromosome {chrom!r} missing from one or more MSA stores")

    def _run_paired(self, ids_ref, ids_alt, source_ids, target_species):
        batch_size = ids_ref.shape[0]
        doubled = torch.as_tensor(
            np.concatenate([ids_ref, ids_alt], axis=0),
            dtype=torch.long,
            device=self.device,
        )
        source = torch.as_tensor(
            np.concatenate([source_ids, source_ids], axis=0), device=self.device
        )
        target = torch.as_tensor(
            np.concatenate([target_species, target_species], axis=0),
            device=self.device,
        )
        with torch.inference_mode():
            hidden = self.encoder(
                input_ids=doubled,
                source_ids=source,
                target_species=target,
            )
        if hidden.ndim == 4:
            if hidden.shape[2] != 1:
                raise RuntimeError(f"Unexpected target axis: {tuple(hidden.shape)}")
            hidden = hidden.squeeze(2)
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(f"Unexpected hidden-state shape: {tuple(hidden.shape)}")
        center = hidden.shape[1] // 2
        pooled = hidden[:, center - POOL_RADIUS : center + POOL_RADIUS + 1].mean(dim=1)
        reference, alternate = pooled[:batch_size], pooled[batch_size:]
        return reference, alternate

    def embed_batch(self, records: list[dict[str, Any]]) -> torch.Tensor:
        chroms = [self.resolve_chrom(row["chrom"]) for row in records]
        starts = np.asarray([row["start"] for row in records], dtype=np.int64)
        ends = np.asarray([row["end"] for row in records], dtype=np.int64)
        coordinates = self.inference.tokenize_function(
            {"chrom": chroms, "start": starts, "end": ends}
        )
        reference_tokens = np.asarray(
            self.tokenizer(
                np.stack(
                    [
                        np.frombuffer(row["seq_ref"].encode("ascii"), dtype="S1")
                        for row in records
                    ]
                )
            ),
            dtype=np.int64,
        )
        alternate_tokens = np.asarray(
            self.tokenizer(
                np.stack(
                    [
                        np.frombuffer(row["seq_alt"].encode("ascii"), dtype="S1")
                        for row in records
                    ]
                )
            ),
            dtype=np.int64,
        )
        reverse_reference_tokens = np.asarray(
            self.tokenizer(
                self.reverse_complementer(
                    np.stack(
                        [
                            np.frombuffer(row["seq_ref"].encode("ascii"), dtype="S1")
                            for row in records
                        ]
                    ),
                    position_axis=1,
                )
            ),
            dtype=np.int64,
        )
        reverse_alternate_tokens = np.asarray(
            self.tokenizer(
                self.reverse_complementer(
                    np.stack(
                        [
                            np.frombuffer(row["seq_alt"].encode("ascii"), dtype="S1")
                            for row in records
                        ]
                    ),
                    position_axis=1,
                )
            ),
            dtype=np.int64,
        )

        target_reference = coordinates["input_ids_fwd"][:, CENTER_OFFSET, 0].astype(
            np.int64
        )
        expected_reference = self.tokenizer(
            np.asarray([row["ref"].encode("ascii") for row in records], dtype="S1")
        ).reshape(-1)
        mismatches = np.flatnonzero(target_reference != expected_reference)
        if len(mismatches):
            bad = records[int(mismatches[0])]
            raise ValueError(
                "MSA reference mismatch at "
                f"{bad['chrom']}:{bad['pos']} (input={bad['ref']}, "
                f"MSA token={int(target_reference[int(mismatches[0])])})"
            )

        source_fwd = coordinates["source_ids_fwd"].copy()
        source_rev = coordinates["source_ids_rev"].copy()
        # Match GPN-Star's VEP preprocessing: mask the target-bearing human
        # species at the variant base while retaining the aligned source taxa.
        mask_id = self.tokenizer.mask_token_id()
        source_fwd[:, CENTER_OFFSET, 0] = mask_id
        source_rev[:, CENTER_OFFSET, 0] = mask_id

        ids_ref_fwd = reference_tokens[:, :, None]
        ids_alt_fwd = alternate_tokens[:, :, None]
        ids_ref_rev = reverse_reference_tokens[:, :, None]
        ids_alt_rev = reverse_alternate_tokens[:, :, None]
        target_species = coordinates["target_species"]

        ref_fwd, alt_fwd = self._run_paired(
            ids_ref_fwd, ids_alt_fwd, source_fwd, target_species
        )
        ref_rev, alt_rev = self._run_paired(
            ids_ref_rev, ids_alt_rev, source_rev, target_species
        )
        return ((alt_fwd + alt_rev) - (ref_fwd + ref_rev)) * 0.5

    def close(self) -> None:
        for genome_msa in self.genome_msa_list:
            store = getattr(getattr(genome_msa, "f", None), "store", None)
            close = getattr(store, "close", None)
            if callable(close):
                close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_complete_artifact(
    output_path: Path,
    manifest_path: Path,
    *,
    model_name: str,
    model_path: Path,
    msa_stores: list[Path],
    msa_species: list[int],
    input_sha256: str,
    sample_id: str,
    population: str,
    input_rows: int,
    active_variant_rows: int,
    topk: int,
) -> tuple[bool, str]:
    """Return whether an existing full-run artifact matches this invocation."""
    if not output_path.is_file() or not manifest_path.is_file():
        return False, "embedding or manifest is missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embedding = np.load(output_path, allow_pickle=False)
    except Exception as exc:
        return False, f"cannot read existing artifact: {exc}"

    expected = {
        "status": "complete",
        "model": model_name,
        "model_path": str(model_path.resolve()),
        "msa_stores": [str(path.resolve()) for path in msa_stores],
        "msa_species": [int(value) for value in msa_species],
        "input_sha256": input_sha256,
        "sample_id": sample_id,
        "population": population,
        "input_rows": input_rows,
        "active_variant_rows": active_variant_rows,
        "window_bp": WINDOW_LENGTH,
        "center_offset": CENTER_OFFSET,
        "pooling": "mean_hidden_states_center_plus_minus_2bp",
        "aggregation": (
            f"multi_stats_topk_{topk}_global_mean_norm_weighted_mean_scalar_stats"
        ),
        "embedding_shape": [3 * 1024 + 4],
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if list(embedding.shape) != [3 * 1024 + 4] or not np.isfinite(embedding).all():
        mismatches.append("embedding_shape_or_values")
    recorded_output_sha = manifest.get("output_sha256")
    if recorded_output_sha and recorded_output_sha != sha256_file(output_path):
        mismatches.append("output_sha256")
    if mismatches:
        return False, "manifest/artifact mismatch: " + ", ".join(mismatches)
    return True, "validated complete artifact"


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.topk < 1:
        raise ValueError("batch size and topk must be positive")
    if args.max_records < 0:
        raise ValueError(
            "max-records must be zero (full run) or a positive pilot limit"
        )
    if len(args.msa_store) != len(args.msa_species):
        raise ValueError("Each --msa-store must have one corresponding --msa-species")

    records, sample_id, population = read_sample(args.input_file, args.max_records)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    slurm_array_task_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    index_offset_text = os.environ.get("INDEX_OFFSET")
    sample_index = None
    if any((slurm_array_job_id, slurm_array_task_text, index_offset_text)):
        if not all((slurm_array_job_id, slurm_array_task_text, index_offset_text)):
            raise RuntimeError(
                "Array extraction requires SLURM_ARRAY_JOB_ID, "
                "SLURM_ARRAY_TASK_ID, and INDEX_OFFSET together"
            )
        try:
            slurm_array_task_id = int(slurm_array_task_text)
            sample_index = int(index_offset_text) + slurm_array_task_id
        except ValueError as exc:
            raise RuntimeError("Invalid Slurm array task/index offset") from exc
    else:
        slurm_array_task_id = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = f"{population}_{sample_id}"
    if args.max_records:
        artifact_stem += ".pilot"
    output_path = args.output_dir / f"{artifact_stem}.npy"
    manifest_path = args.manifest_dir / f"{artifact_stem}.json"

    # Reference-allele rows are exact zero deltas under the established paired
    # sequence protocol; omit model inference for them but retain them in the
    # full matrix so the existing aggregation denominator remains unchanged.
    active_indices = [
        index for index, row in enumerate(records) if row["ref"] != row["alt"]
    ]
    input_sha256 = sha256_file(args.input_file)
    if not args.max_records and output_path.exists() and not args.recompute:
        is_valid, reason = valid_complete_artifact(
            output_path,
            manifest_path,
            model_name=args.model_name,
            model_path=args.model_path,
            msa_stores=args.msa_store,
            msa_species=args.msa_species,
            input_sha256=input_sha256,
            sample_id=sample_id,
            population=population,
            input_rows=len(records),
            active_variant_rows=len(active_indices),
            topk=args.topk,
        )
        if is_valid:
            print(f"Already complete and validated: {output_path}")
            return
        print(f"Recomputing existing artifact ({reason}): {output_path}", flush=True)
    delta_matrix = np.zeros((len(records), 1024), dtype=np.float32)
    extractor = GPNStarAncestryExtractor(
        args.model_path, args.model_name, args.msa_store, args.msa_species
    )
    try:
        for offset in range(0, len(active_indices), args.batch_size):
            indices = active_indices[offset : offset + args.batch_size]
            batch = [records[index] for index in indices]
            deltas = (
                extractor.embed_batch(batch).detach().cpu().numpy().astype(np.float32)
            )
            if deltas.shape != (len(indices), extractor.hidden_size):
                raise RuntimeError(f"Unexpected delta matrix shape: {deltas.shape}")
            if not np.isfinite(deltas).all():
                raise RuntimeError(f"Non-finite deltas for {args.input_file}")
            delta_matrix[indices] = deltas
    finally:
        extractor.close()

    aggregate = aggregate_feature_matrix(delta_matrix, args.topk)
    expected_width = 3 * 1024 + 4
    if aggregate.shape != (expected_width,) or not np.isfinite(aggregate).all():
        raise RuntimeError(
            f"Expected finite ({expected_width},) vector, got {aggregate.shape}"
        )
    temp_path = output_path.with_suffix(output_path.suffix + ".partial")
    with temp_path.open("wb") as handle:
        np.save(handle, aggregate.astype(np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)

    manifest = {
        "status": "complete" if not args.max_records else "pilot_partial",
        "model": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "msa_stores": [str(path.resolve()) for path in args.msa_store],
        "msa_species": args.msa_species,
        "input_file": str(args.input_file.resolve()),
        "input_sha256": input_sha256,
        "sample_id": sample_id,
        "population": population,
        "input_rows": len(records),
        "active_variant_rows": len(active_indices),
        "zero_delta_rows": len(records) - len(active_indices),
        "window_bp": WINDOW_LENGTH,
        "center_offset": CENTER_OFFSET,
        "pooling": "mean_hidden_states_center_plus_minus_2bp",
        "aggregation": (
            f"multi_stats_topk_{args.topk}_global_mean_norm_weighted_mean_scalar_stats"
        ),
        "embedding_shape": [expected_width],
        "batch_size": args.batch_size,
        "visible_gpu_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "slurm_job_id": slurm_job_id,
        "slurm_array_job_id": slurm_array_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "sample_index": sample_index,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "output_sha256": sha256_file(output_path),
    }
    manifest_temp = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    manifest_temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_temp, manifest_path)
    print(
        f"{'Pilot' if args.max_records else 'Complete'} {args.model_name} {sample_id}: "
        f"rows={len(records)} active={len(active_indices)} shape={aggregate.shape}"
    )


if __name__ == "__main__":
    main()
