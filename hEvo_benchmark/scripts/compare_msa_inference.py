#!/usr/bin/env python3
"""Run real baseline/current embedding and native-scoring comparisons for GPN.

This is a focused, offline regression utility. It AST-loads only the selected
extractor classes from each source tree, avoiding imports for unrelated model
families in ``extract_embeddings_classes.py``. CPU is the default; CUDA must
be requested explicitly.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from compare_evaluator_caches import evaluator_source

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pysam
import torch


WORKSPACE = Path(__file__).resolve().parents[2]
EVALUATE_DIR = WORKSPACE / "hEvo_benchmark/scripts/evaluate"
DEFAULT_BASELINE = Path(
    "/tmp/hevo-eval-baseline-9405e9d/hEvo_benchmark/scripts/evaluate"
)
DEFAULT_OUTPUT = Path("/tmp/hevo-eval-regression-20261017/msa")
DEFAULT_VARIANTS = (
    WORKSPACE / "hEvo_benchmark/datasets/Causal_Variants/causal_variants_hg38_1023.csv"
)
DEFAULT_FASTA = Path("/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/datasets/hg38.fa")
DEFAULT_GPN_MSA_MODEL = Path(
    "/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/models/"
    "gpn-msa-sapiens-official-4a7d4f75449cb2abd560b2af024d76f99233c6db"
)
DEFAULT_GPN_MSA_STORE = Path(
    "/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/datasets/GPN_Datasets/"
    "multiz100way/89.zarr.zip"
)
DEFAULT_GPN_STAR_MODEL = Path("/mnt/hpc/home/xuqinyi/gpn_star_assets/models/GPN-Star-V")
DEFAULT_GPN_STAR_STORE = Path(
    "/mnt/hpc/home/xuqinyi/gpn_star_assets/msa/multiz100way-pigz/100/all.zarr"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_classes(source: Path, names: set[str]) -> dict[str, type]:
    """Load selected class definitions without importing the full module."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    }
    missing = names.difference(selected)
    if missing:
        raise RuntimeError(f"Missing classes in {source}: {sorted(missing)}")

    namespace: dict[str, Any] = {
        "os": os,
        "np": np,
        "torch": torch,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Union": Union,
    }
    if "ExtractEmbeddingForDNABert" in names:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        namespace["AutoModelForSequenceClassification"] = (
            AutoModelForSequenceClassification
        )
        namespace["AutoTokenizer"] = AutoTokenizer
        base = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ExtractEmbeddingBase"
        )
        exec(
            compile(ast.Module(body=[base], type_ignores=[]), str(source), "exec"),
            namespace,
        )
    nodes = [selected[name] for name in names]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    return {name: namespace[name] for name in names}


def load_records(csv_path: Path, fasta_path: Path, max_rows: int, interval_length: int):
    records = []
    fasta = pysam.FastaFile(str(fasta_path))
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                if row_index >= max_rows or len(records) >= 4:
                    break
                try:
                    chrom = str(row["chrom_hg38"]).strip()
                    pos1 = int(row["pos_hg38"])
                    expected_variant_ref = str(row["ref"]).upper()
                    variant_id = str(row["seqID"])
                    if not chrom or pos1 < 1 or expected_variant_ref not in "ACGT":
                        continue
                    center0 = pos1 - 1
                    start = center0 - interval_length // 2
                    end = start + interval_length
                    if start < 0 or end > fasta.get_reference_length(chrom):
                        continue
                    sequence = fasta.fetch(chrom, start, end).upper()
                    source_sequence = str(row.get("refseq", "")).upper()
                    if len(sequence) != interval_length or set(sequence) - set("ACGTN"):
                        continue
                    if sequence != source_sequence:
                        continue
                    if sequence[center0 - start] != expected_variant_ref:
                        continue
                    records.append(
                        {
                            "SeqID": variant_id,
                            "chrom": chrom,
                            "start": start,
                            "end": end,
                            "seq": sequence,
                            "ref": expected_variant_ref,
                            "alt": str(row.get("alt", "")).upper(),
                            "csv_row": row_index + 2,
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    finally:
        fasta.close()
    if len(records) != 4:
        raise RuntimeError(
            f"Found {len(records)} valid reference-matched variants in the first "
            f"{max_rows} CSV rows; expected 4"
        )
    return records


def array_summary(old_values: torch.Tensor, new_values: torch.Tensor) -> dict[str, Any]:
    old = old_values.detach().cpu().to(torch.float32).numpy()
    new = new_values.detach().cpu().to(torch.float32).numpy()
    if old.shape != new.shape:
        return {
            "old_shape": list(old.shape),
            "new_shape": list(new.shape),
            "shape_match": False,
        }
    delta = np.abs(old - new)
    return {
        "shape": list(old.shape),
        "shape_match": True,
        "old_sha256": hashlib.sha256(old.tobytes()).hexdigest(),
        "new_sha256": hashlib.sha256(new.tobytes()).hexdigest(),
        "exact_equal": bool(np.array_equal(old, new)),
        "allclose_atol_1e-6_rtol_1e-5": bool(
            np.allclose(old, new, atol=1e-6, rtol=1e-5)
        ),
        "max_abs_diff": float(delta.max(initial=0.0)),
        "mean_abs_diff": float(delta.mean()) if delta.size else 0.0,
    }


def run_interval_protocol(
    name: str,
    cls_names: set[str],
    baseline_source: Path,
    current_source: Path,
    records: list[dict[str, Any]],
    kwargs_by_class: dict[str, dict[str, Any]],
    expected_device: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "running", "records": []}
    embeddings: dict[str, torch.Tensor] = {}
    for version, source in (("baseline", baseline_source), ("current", current_source)):
        try:
            cls = selected_classes(source, cls_names)[next(iter(cls_names))]
            started = time.perf_counter()
            extractor = cls(**kwargs_by_class[next(iter(cls_names))])
            if extractor.device.type != expected_device:
                raise RuntimeError(
                    f"Expected {expected_device} inference, got {extractor.device}"
                )
            output = extractor.get_embedding_for_intervals(
                records, verify_sequences=True
            )
            embeddings[version] = output
            result[version] = {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "device": str(extractor.device),
                "elapsed_seconds": time.perf_counter() - started,
                "sequence_mismatch_rows": list(
                    getattr(extractor, "sequence_mismatch_rows", [])
                ),
            }
            close = getattr(extractor, "close", None)
            if callable(close):
                close()
            del extractor
            gc.collect()
        except Exception as exc:
            result[version] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            gc.collect()
    if len(embeddings) == 2:
        result["comparison"] = array_summary(
            embeddings["baseline"], embeddings["current"]
        )
        result["status"] = (
            "passed"
            if result["comparison"].get("allclose_atol_1e-6_rtol_1e-5")
            else "failed"
        )
    else:
        result["status"] = "failed"
    result["records"] = [
        {key: value for key, value in record.items() if key != "seq"}
        for record in records
    ]
    return result


def run_gpn_star_variant_regression(
    baseline_adapter: Path,
    current_adapter: Path,
    model_path: Path,
    msa_store: Path,
    records: list[dict[str, Any]],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    import pandas as pd

    input_path = output_dir / "native_variant_input.tsv"
    rows = [
        {
            "task_name": "Causal_Variants",
            "variant_id": record["SeqID"],
            "chrom": record["chrom"],
            "pos": record["start"] + len(record["seq"]) // 2 + 1,
            "ref": record["ref"],
            "alt": record["alt"],
            "label": 0,
            "selected_allele": "",
            "other_allele": "",
            "score_mode": "raw_llr",
            "source_row": record["csv_row"],
            "input_status": "PASS",
            "failure_reason": "",
        }
        for record in records
    ]
    pd.DataFrame(rows).to_csv(input_path, sep="\t", index=False)
    results: dict[str, Any] = {
        "input_path": str(input_path),
        "records": [row["variant_id"] for row in rows],
    }
    frames = {}
    for version, source in (
        ("baseline", baseline_adapter),
        ("current", current_adapter),
    ):
        try:
            module_name = f"_gpn_star_adapter_{version}"
            spec = importlib.util.spec_from_file_location(module_name, source)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import adapter source: {source}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            args = argparse.Namespace(
                model_id="GPN-Star-V",
                model_dir=model_path,
                model_revision="local-checkpoint",
                msa_store=[str(msa_store)],
                msa_species=[100],
                msa_dataset="multiz100way-pigz-100",
                msa_revision="local-store",
                input=input_path,
                output=output_dir / f"native_variant_{version}.tsv",
                window_bp=1024,
                batch_size=1,
                device=device,
                max_records=0,
                manifest=None,
            )
            started = time.perf_counter()
            frame, diagnostics = module.run(args)
            frame.to_csv(args.output, sep="\t", index=False)
            frames[version] = frame
            results[version] = {
                "elapsed_seconds": time.perf_counter() - started,
                "diagnostics": diagnostics,
                "status_counts": {
                    str(key): int(value)
                    for key, value in frame["status"].value_counts(dropna=False).items()
                },
                "scores": frame[
                    [
                        "variant_id",
                        "raw_llr",
                        "benchmark_score",
                        "status",
                        "failure_reason",
                    ]
                ].to_dict(orient="records"),
            }
        except Exception as exc:
            results[version] = {"error_type": type(exc).__name__, "error": str(exc)}

    if len(frames) != 2:
        results["status"] = "failed"
        return results
    old = frames["baseline"].set_index("variant_id").sort_index()
    new = frames["current"].set_index("variant_id").sort_index()
    old_scores = old[["raw_llr", "benchmark_score"]].to_numpy(dtype=float)
    new_scores = new[["raw_llr", "benchmark_score"]].to_numpy(dtype=float)
    same_status = old["status"].tolist() == new["status"].tolist()
    same_ids = old.index.tolist() == new.index.tolist()
    same_scores = bool(
        np.array_equal(np.isfinite(old_scores), np.isfinite(new_scores))
        and np.allclose(
            old_scores,
            new_scores,
            atol=1e-6,
            rtol=1e-5,
            equal_nan=True,
        )
    )
    results["comparison"] = {
        "same_variant_ids": same_ids,
        "same_filter_status": same_status,
        "scores_allclose_atol_1e-6_rtol_1e-5": same_scores,
        "max_abs_score_diff": float(np.nanmax(np.abs(old_scores - new_scores))),
        "n_pass_current": int(new["status"].eq("PASS").sum()),
    }
    results["status"] = (
        "passed"
        if same_ids
        and same_status
        and same_scores
        and results["comparison"]["n_pass_current"] > 0
        else "failed"
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current-dir", type=Path, default=EVALUATE_DIR)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--gpn-msa-model", type=Path, default=DEFAULT_GPN_MSA_MODEL)
    parser.add_argument("--gpn-msa-store", type=Path, default=DEFAULT_GPN_MSA_STORE)
    parser.add_argument("--gpn-star-model", type=Path, default=DEFAULT_GPN_STAR_MODEL)
    parser.add_argument("--gpn-star-store", type=Path, default=DEFAULT_GPN_STAR_STORE)
    parser.add_argument(
        "--baseline-adapter",
        type=Path,
        default=DEFAULT_BASELINE / "gpn_star_adapter.py",
    )
    parser.add_argument(
        "--current-adapter",
        type=Path,
        default=EVALUATE_DIR / "adapters/gpn_star_adapter.py",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scan-rows", type=int, default=32)
    parser.add_argument("--interval-length", type=int, default=1023)
    parser.add_argument("--star-species", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--native-variants-only", action="store_true")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_source = evaluator_source(
        args.baseline_dir, "extract_embeddings_classes.py"
    )
    current_source = evaluator_source(args.current_dir, "extract_embeddings_classes.py")
    evidence: dict[str, Any] = {
        "status": "running",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_policy": f"Explicit {args.device} inference",
        "sources": {
            "baseline": str(baseline_source),
            "current": str(current_source),
            "baseline_sha256": sha256_file(baseline_source),
            "current_sha256": sha256_file(current_source),
            "byte_identical": baseline_source.read_bytes()
            == current_source.read_bytes(),
        },
        "inputs": {
            "variant_csv": str(args.variants),
            "variant_csv_sha256": sha256_file(args.variants),
            "reference_fasta": str(args.fasta),
            "reference_fasta_sha256": sha256_file(args.fasta),
            "interval_length": args.interval_length,
            "scan_rows": args.scan_rows,
        },
        "protocols": {},
    }

    records = load_records(
        args.variants, args.fasta, args.scan_rows, args.interval_length
    )
    evidence["records"] = [
        {key: value for key, value in record.items() if key != "seq"}
        for record in records
    ]

    if args.native_variants_only:
        result = run_gpn_star_variant_regression(
            args.baseline_adapter,
            args.current_adapter,
            args.gpn_star_model,
            args.gpn_star_store,
            records,
            args.output_dir,
            args.device,
        )
        result["status"] = result.get("status", "failed")
        evidence["protocols"]["gpn_star_native_variants"] = result
        evidence["status"] = result["status"]
        output_path = args.output_dir / "gpn_star_native_variant_scores.json"
        output_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": evidence["status"],
                    "evidence": str(output_path),
                    "protocols": {
                        key: value["status"]
                        for key, value in evidence["protocols"].items()
                    },
                },
                indent=2,
            )
        )
        return 0 if evidence["status"] == "passed" else 1

    try:
        evidence["protocols"]["gpn_msa"] = run_interval_protocol(
            "gpn_msa",
            {"ExtractEmbeddingForGPNMSA"},
            baseline_source,
            current_source,
            records,
            {
                "ExtractEmbeddingForGPNMSA": {
                    "model_path": str(args.gpn_msa_model),
                    "model_name": "GPN-MSA",
                    "msa_path": str(args.gpn_msa_store),
                    "window_size": 128,
                    "center_window_size": 1000,
                }
            },
            expected_device=args.device,
        )
    except Exception as exc:
        evidence["protocols"]["gpn_msa"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    try:
        evidence["protocols"]["gpn_star"] = run_interval_protocol(
            "gpn_star",
            {"ExtractEmbeddingForGPNStar"},
            baseline_source,
            current_source,
            records,
            {
                "ExtractEmbeddingForGPNStar": {
                    "model_path": str(args.gpn_star_model),
                    "model_name": "GPN-Star-V",
                    "msa_paths": [str(args.gpn_star_store)],
                    "msa_species": [args.star_species],
                    "window_size": 1024,
                    "center_window_size": 1000,
                }
            },
            expected_device=args.device,
        )
    except Exception as exc:
        evidence["protocols"]["gpn_star"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    statuses = [item["status"] for item in evidence["protocols"].values()]
    evidence["status"] = (
        "passed" if statuses and all(x == "passed" for x in statuses) else "failed"
    )
    output_path = args.output_dir / "old_new_msa_inference.json"
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "evidence": str(output_path),
                "protocols": {
                    key: value["status"] for key, value in evidence["protocols"].items()
                },
            },
            indent=2,
        )
    )
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
