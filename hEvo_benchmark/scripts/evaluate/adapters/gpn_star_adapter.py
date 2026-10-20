#!/usr/bin/env python3
"""Run the official GPN-Star VEP protocol on normalized hg38 variants.

The implementation deliberately uses ``gpn.star.vep.VEPInference`` and its
paired forward/reverse MSA inputs. It does not fall back to a single-sequence
model.
It is called by ``zeroshot_paried_seq_input.py`` and can also be used directly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "task_name",
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "selected_allele",
    "other_allele",
    "score_mode",
    "source_row",
    "input_status",
    "failure_reason",
}
OUTPUT_COLUMNS = [
    "model_id",
    "model_revision",
    "task_name",
    "variant_id",
    "chrom",
    "pos",
    "ref",
    "alt",
    "raw_llr",
    "selected_other_score",
    "benchmark_score",
    "label",
    "score_mode",
    "selected_allele",
    "other_allele",
    "source_row",
    "status",
    "failure_reason",
    "window_bp",
    "msa_species",
    "msa_dataset",
    "msa_revision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--msa-store", action="append", required=True)
    parser.add_argument("--msa-species", nargs="+", type=int, required=True)
    parser.add_argument("--msa-dataset", default="")
    parser.add_argument("--msa-revision", default="")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-bp", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional external run manifest path.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance() -> dict[str, str]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "CUDA_VISIBLE_DEVICES",
    )
    return {
        "hostname": socket.gethostname(),
        **{key.lower(): os.environ[key] for key in keys if os.environ.get(key)},
    }


def _read_input(path: Path, max_records: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"normalized input is missing columns: {sorted(missing)}")
    if max_records:
        frame = frame.iloc[:max_records].copy()
    if frame.empty:
        raise ValueError("normalized input contains no rows")
    return frame


def _resolve_store(path: str) -> Path:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"MSA store does not exist: {root}")
    if root.is_dir() and (root / "all.zarr").is_dir():
        return root / "all.zarr"
    if root.is_dir() and root.name.endswith(".zarr"):
        return root
    raise ValueError(
        f"MSA store must be an extracted Zarr directory or a directory containing all.zarr: {root}"
    )


def _load_msa(stores: list[str], species: list[int], chroms: list[str]) -> list[Any]:
    from gpn.star.data import GenomeMSA

    if len(stores) != len(species):
        raise ValueError(
            f"--msa-store count ({len(stores)}) must equal --msa-species count ({len(species)})"
        )
    result = []
    requested_chroms = set(chroms)
    requested_chroms.update(chrom[3:] for chrom in chroms if chrom.startswith("chr"))
    for store, n_species in zip(stores, species):
        msa = GenomeMSA(
            str(_resolve_store(store)),
            n_species=n_species,
            subset_chroms=requested_chroms,
            in_memory=False,
        )
        if not msa.data:
            raise ValueError(f"MSA store has no requested chromosomes: {store}")
        if any(array.shape[1] != n_species for array in msa.data.values()):
            raise ValueError(f"MSA species count does not match --msa-species: {store}")
        result.append(msa)
    return result


def _msa_chrom(msa: Any, chrom: str) -> str | None:
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append("chr" + chrom)
    return next((candidate for candidate in candidates if candidate in msa.data), None)


def _shared_msa_chrom(msa_list: list[Any], chrom: str) -> str | None:
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append("chr" + chrom)
    return next(
        (
            candidate
            for candidate in candidates
            if all(candidate in msa.data for msa in msa_list)
        ),
        None,
    )


def _msa_base(msa: Any, chrom: str, pos: int) -> str | None:
    msa_chrom = _msa_chrom(msa, chrom)
    if msa_chrom is None:
        return None
    array = msa.data[msa_chrom]
    if pos <= 0 or pos > array.shape[0]:
        return None
    values = np.asarray(array[pos - 1]).reshape(-1)
    if len(values) == 0:
        return None
    value = values[0]
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace").upper()
    return str(value).upper()


def _reference_status(frame: pd.DataFrame, msa_list: list[Any]) -> pd.Series:
    """Validate against the target-bearing alignment, not source-only stores.

    In the official multi-store GPN-Star layout, the first store is the
    target-bearing alignment and subsequent stores contribute source species.
    For hg38 GPN-Star-M this is the 243-way store followed by the 204-way
    source-only store.  Only the first column of the first store is therefore
    the human reference sequence; requiring every store's first column to
    match the human reference incorrectly rejects valid MSA rows.
    """
    target_msa = msa_list[0]
    statuses: list[str] = []
    for row in frame.itertuples(index=False):
        chrom = str(row.chrom)
        pos = int(row.pos)
        base = _msa_base(target_msa, chrom, pos)
        if base is None:
            statuses.append("missing_chromosome_or_position")
        elif base != str(row.ref).upper():
            statuses.append("msa_reference_does_not_match_ref")
        else:
            statuses.append("")
    return pd.Series(statuses, index=frame.index, dtype=str)


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_inference(
    model_dir: Path,
    msa_list: list[Any],
    window_bp: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Load with explicit HF key diagnostics, then construct official VEPInference."""

    import torch
    from transformers import AutoModelForMaskedLM

    # gpn.star.vep registers the custom ``star`` auto classes on import.
    from gpn.star.vep import VEPInference

    _, loading_info = AutoModelForMaskedLM.from_pretrained(
        str(model_dir), output_loading_info=True
    )
    missing_keys = list(loading_info.get("missing_keys", []))
    unexpected_keys = list(loading_info.get("unexpected_keys", []))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint key summary failed: "
            f"missing={missing_keys[:5]} unexpected={unexpected_keys[:5]}"
        )
    del loading_info
    gc.collect()
    inference = VEPInference(str(model_dir), msa_list, window_bp)
    inference.model.to(device)
    inference.model.eval()
    architecture = {
        "model_class": type(inference.model.model).__name__,
        "config_class": type(inference.model.model.config).__name__,
        "model_type": getattr(inference.model.model.config, "model_type", ""),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "parameter_count": sum(
            parameter.numel() for parameter in inference.model.parameters()
        ),
        "torch_version": torch.__version__,
    }
    return inference, architecture


def _benchmark_score(raw_llr: float, row: Any) -> tuple[float, float]:
    selected = str(row.selected_allele).upper()
    other = str(row.other_allele).upper()
    ref = str(row.ref).upper()
    alt = str(row.alt).upper()
    selected_other = np.nan
    if selected and other:
        if selected == alt and other == ref:
            selected_other = raw_llr
        elif selected == ref and other == alt:
            selected_other = -raw_llr
    mode = str(row.score_mode)
    if mode == "neg_llr":
        benchmark = -raw_llr
    elif mode == "raw_llr":
        benchmark = raw_llr
    elif mode == "abs_llr":
        benchmark = abs(raw_llr)
    elif mode == "selected_other":
        if not np.isfinite(selected_other):
            raise ValueError(
                "selected_other mode requires selected/other matching ref/alt"
            )
        benchmark = selected_other
    else:
        raise ValueError(f"unknown score mode: {mode}")
    return float(selected_other), float(benchmark)


def _score_unique(
    inference: Any,
    frame: pd.DataFrame,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    values: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        encoded = inference.tokenize_function(
            {
                "chrom": batch["chrom"].tolist(),
                "pos": [int(value) for value in batch["pos"]],
                "ref": batch["ref"].tolist(),
                "alt": batch["alt"].tolist(),
            }
        )
        tensors = {
            key: torch.as_tensor(value, device=device) for key, value in encoded.items()
        }
        with torch.inference_mode():
            scores = inference.model(**tensors).detach().float().cpu().numpy()
        values.append(scores)
    return np.concatenate(values).astype(float, copy=False)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.window_bp <= 0 or args.window_bp % 2:
        raise ValueError("--window-bp must be a positive even number")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model_dir}")
    expected_species = {
        "GPN-Star-V": 100,
        "GPN-Star-M": 447,
        "GPN-Star-P": 243,
    }.get(args.model_id)
    if expected_species is not None and sum(args.msa_species) != expected_species:
        raise ValueError(
            f"{args.model_id} requires {expected_species} MSA species, got {sum(args.msa_species)}"
        )
    if len(args.msa_store) != len(args.msa_species):
        raise ValueError(
            f"--msa-store count ({len(args.msa_store)}) must equal "
            f"--msa-species count ({len(args.msa_species)})"
        )
    if len(args.msa_species) > 1:
        # Keep each store paired with its species count while enforcing the
        # official descending order.  This prevents a reversed shell/YAML
        # argument from silently treating the source-only 204-way store as
        # the human target sequence.
        pairs = sorted(
            zip(args.msa_species, args.msa_store),
            key=lambda pair: pair[0],
            reverse=True,
        )
        args.msa_species = [species for species, _ in pairs]
        args.msa_store = [store for _, store in pairs]

    frame = _read_input(args.input, args.max_records)
    result = pd.DataFrame(
        {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "task_name": frame.task_name,
            "variant_id": frame.variant_id,
            "chrom": frame.chrom,
            "pos": frame.pos,
            "ref": frame.ref,
            "alt": frame.alt,
            "raw_llr": np.nan,
            "selected_other_score": np.nan,
            "benchmark_score": np.nan,
            "label": frame.label,
            "score_mode": frame.score_mode,
            "selected_allele": frame.selected_allele,
            "other_allele": frame.other_allele,
            "source_row": frame.source_row,
            "status": frame.input_status,
            "failure_reason": frame.failure_reason,
            "window_bp": args.window_bp,
            "msa_species": ",".join(str(value) for value in args.msa_species),
            "msa_dataset": args.msa_dataset,
            "msa_revision": args.msa_revision,
        },
        columns=OUTPUT_COLUMNS,
    )
    valid_mask = frame.input_status.eq("PASS")
    if not valid_mask.any():
        return result, {
            "architecture": None,
            "device": "not_loaded",
            "unique_scored": 0,
        }

    valid = frame.loc[valid_mask].copy()
    msa_list = _load_msa(args.msa_store, args.msa_species, sorted(valid.chrom.unique()))
    mismatch = _reference_status(valid, msa_list)
    valid = valid.loc[mismatch.eq("")].copy()
    blocked_reference = mismatch.index[mismatch.ne("")]
    result.loc[blocked_reference, "status"] = "BLOCKED_REFERENCE_MISMATCH"
    result.loc[blocked_reference, "failure_reason"] = mismatch[mismatch.ne("")]
    if valid.empty:
        return result, {
            "architecture": None,
            "device": "not_loaded",
            "unique_scored": 0,
        }

    device = _select_device(args.device)
    inference, architecture = _load_inference(
        args.model_dir, msa_list, args.window_bp, device
    )
    scoring = valid.copy()
    scoring["chrom"] = scoring["chrom"].map(
        lambda chrom: _shared_msa_chrom(msa_list, chrom)
    )
    if scoring.chrom.isna().any():
        raise RuntimeError(
            "reference validation passed but no MSA chromosome key was resolved"
        )
    unique = scoring.drop_duplicates(["chrom", "pos", "ref", "alt"]).copy()
    raw_scores = _score_unique(inference, unique, args.batch_size, device)
    raw_by_key = {
        tuple(row[key] for key in ["chrom", "pos", "ref", "alt"]): float(score)
        for (_, row), score in zip(unique.iterrows(), raw_scores)
    }
    for index, row in valid.iterrows():
        key = (_shared_msa_chrom(msa_list, row.chrom), row.pos, row.ref, row.alt)
        raw = raw_by_key[key]
        if not np.isfinite(raw):
            result.loc[index, "status"] = "BLOCKED_NONFINITE_SCORE"
            result.loc[index, "failure_reason"] = "raw_llr_is_not_finite"
            continue
        selected_other, benchmark = _benchmark_score(raw, row)
        result.loc[index, "raw_llr"] = raw
        result.loc[index, "selected_other_score"] = selected_other
        result.loc[index, "benchmark_score"] = benchmark
        result.loc[index, "status"] = "PASS"
        result.loc[index, "failure_reason"] = ""
    return result, {
        "architecture": architecture,
        "device": device,
        "unique_scored": int(len(unique)),
        "msa_chromosomes": [sorted(msa.data.keys()) for msa in msa_list],
        "reference_validation_store": str(Path(args.msa_store[0]).resolve()),
    }


def write_manifest(
    args: argparse.Namespace,
    result: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> None:
    """Write an explicitly requested runtime manifest outside the repository."""
    manifest_path = getattr(args, "manifest", None)
    if manifest_path is None:
        return
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "gpn_star_variant_scores",
        "model_id": args.model_id,
        "model_dir": str(args.model_dir.resolve()),
        "model_revision": args.model_revision,
        "msa_stores": [str(Path(value).resolve()) for value in args.msa_store],
        "msa_species": args.msa_species,
        "msa_dataset": args.msa_dataset,
        "msa_revision": args.msa_revision,
        "assembly": "hg38",
        "window_bp": args.window_bp,
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "output": str(args.output.resolve()),
        "rows": int(len(result)),
        "status_counts": result.status.value_counts(dropna=False).to_dict(),
        "finite_raw_llr": int(
            np.isfinite(pd.to_numeric(result.raw_llr, errors="coerce")).sum()
        ),
        "gpn_version": importlib.metadata.version("gpn"),
        "diagnostics": diagnostics,
        "provenance": _provenance(),
    }
    manifest["output_sha256"] = _sha256(args.output)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"manifest={manifest_path}")


def main() -> int:
    args = parse_args()
    result, diagnostics = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    write_manifest(args, result, diagnostics)
    print(f"wrote {args.output} rows={len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
