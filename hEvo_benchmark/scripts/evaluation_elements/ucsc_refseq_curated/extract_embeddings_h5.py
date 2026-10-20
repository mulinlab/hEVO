#!/usr/bin/env python3
"""Extract ordered sequence embeddings for the genomic-element benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import os
import sys
from pathlib import Path

EVALUATE_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "shared"
if str(EVALUATE_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATE_DIR))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_order_sha256(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (f"{row['seqID']}\t{row['label']}\t{row['seq']}\n").encode("utf-8")
        )
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--pooling", choices=["mean", "cls", "max"], default="mean")
    parser.add_argument("--seq-col", type=int, default=1)
    parser.add_argument("--label-col", type=int, default=2)
    parser.add_argument("--id-col", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--extractor-dir", type=Path, default=EVALUATE_DIR)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evo2-config-dir", type=Path, default=None)
    return parser.parse_args()


def read_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty CSV: {args.input_csv}")
        required = max(args.seq_col, args.label_col, args.id_col)
        if required >= len(header):
            raise ValueError(f"column index {required} outside header {header}")
        rows = []
        for line_number, values in enumerate(reader, 2):
            if len(values) <= required:
                raise ValueError(f"{args.input_csv}:{line_number}: too few columns")
            seq_id = values[args.id_col].strip()
            sequence = values[args.seq_col].strip().upper()
            label = values[args.label_col].strip()
            if not seq_id or not label or not sequence:
                raise ValueError(f"{args.input_csv}:{line_number}: empty id, label or sequence")
            if len(sequence) != 512:
                raise ValueError(
                    f"{args.input_csv}:{line_number}: expected 512 bp, got {len(sequence)}"
                )
            rows.append({"seqID": seq_id, "seq": sequence, "label": label})
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError(f"no rows read from {args.input_csv}")
    if len({row["seqID"] for row in rows}) != len(rows):
        raise ValueError("duplicate seqID values")
    if len({row["seq"] for row in rows}) != len(rows):
        raise ValueError("duplicate sequence values")
    return rows


def weight_sha256(model_path: Path) -> str:
    if model_path.is_file():
        return sha256_file(model_path)
    files = sorted(model_path.glob("*.safetensors")) or sorted(model_path.glob("pytorch_model*.bin")) or sorted(model_path.glob("*.ckpt"))
    if not files:
        raise FileNotFoundError(f"no model weight files in {model_path}")
    if len(files) == 1:
        return sha256_file(files[0])
    parts = {path.name: sha256_file(path) for path in files}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def configure_evo2_for_rtx3090(model_name: str, config_dir: Path | None = None) -> tuple[str, str | None]:
    """Use the explicit BF16 Evo2-1B config on non-Hopper GPUs.

    Evo2-1B ships with FP8 input projections enabled, but the cluster's
    RTX 3090 nodes do not support FP8.  The generated config is kept beside
    this frozen run's evaluator so the precision choice is reproducible and
    traceable; Evo2-7B already ships with FP8 disabled.
    """
    if model_name != "evo2_1b_base":
        return "", None
    import evo2.models as evo_models

    bundled = Path(evo_models.__file__).resolve().parent / "configs" / "evo2-1b-8k.yml"
    text = bundled.read_text(encoding="utf-8")
    old = "use_fp8_input_projections: True"
    if old not in text:
        raise ValueError(f"unexpected Evo2-1B config; missing {old}")
    config_text = text.replace(old, "use_fp8_input_projections: False", 1)
    if config_dir is None:
        raise ValueError("--evo2-config-dir is required for evo2_1b_base")
    config_path = config_dir / "evo2-1b-8k-bf16.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != config_text:
        raise ValueError(f"frozen Evo2 BF16 config differs: {config_path}")
    config_path.write_text(config_text, encoding="utf-8")
    evo_models.CONFIG_MAP[model_name] = str(config_path)
    return "bfloat16", sha256_file(config_path)


def main() -> None:
    args = parse_args()
    if args.pooling != "mean" or args.max_rows is not None:
        raise ValueError("formal element evaluation requires mean pooling and all input rows")
    gate = json.loads(args.dataset_manifest.read_text())
    dataset_id = args.input_csv.stem
    input_hash = sha256_file(args.input_csv)
    if gate.get("status") != "PASS" or gate.get("dataset_sha256", {}).get(dataset_id) != input_hash:
        raise ValueError("input CSV is not bound to a PASS dataset manifest")
    if gate["run_id"] != args.run_id:
        raise ValueError("run_id does not match the dataset manifest")
    sys.path.insert(0, str(args.extractor_dir.resolve()))
    # Keep --help lightweight on login nodes; the model/evaluator stack is
    # imported only after argument validation has selected a real run.
    import h5py
    import numpy as np
    import probe_single_seq_input as probe  # noqa: E402
    import torch
    from tqdm import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError("embedding extraction requires an allocated GPU")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive")
    if args.output_h5.exists() and not args.recompute:
        raise FileExistsError(f"output exists; pass --recompute to replace: {args.output_h5}")

    rows = read_rows(args)
    from collections import Counter
    if Counter(row["label"] for row in rows) != {label: 1000 for label in gate["tasks"][dataset_id]}:
        raise ValueError("input class roster/counts do not match the frozen protocol")
    sequences = [row["seq"] for row in rows]
    labels = [row["label"] for row in rows]
    extractor_cls = probe.model_extractor_map.get(args.model_name)
    if extractor_cls is None:
        raise KeyError(f"no embedding extractor registered for {args.model_name}")
    if args.model_name == "AIDO.DNA-300M":
        # The shared adapter otherwise ignores model_path and loads a registry
        # default. Bind the same backbone class to the requested local weights.
        from modelgenerator.backbones import aido_dna_300m
        aido_dna_300m.model_path = str(args.model_path.resolve())
    inference_precision, evo2_config_sha256 = configure_evo2_for_rtx3090(args.model_name, args.evo2_config_dir)
    if args.model_name.startswith("evo2_") and args.model_name != "evo2_1b_base":
        # The installed Evo2 local-checkpoint loader opens CONFIG_MAP paths
        # relative to cwd; resolve its bundled configuration explicitly.
        import evo2.models as evo_models
        config = Path(evo_models.CONFIG_MAP[args.model_name])
        if not config.is_absolute():
            config = Path(evo_models.__file__).resolve().parent / config
        if not config.is_file():
            raise FileNotFoundError(config)
        evo_models.CONFIG_MAP[args.model_name] = str(config)
    model_hash = weight_sha256(args.model_path)
    extractor = extractor_cls(str(args.model_path), args.model_name)
    if args.model_name == "AIDO.DNA-300M" and Path(extractor.model.backbone.model_path).resolve() != args.model_path.resolve():
        raise RuntimeError("AIDO did not load the declared model directory")
    for model_object in (extractor.model, getattr(extractor.model, "model", None)):
        if hasattr(model_object, "eval"):
            model_object.eval()

    embeddings = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(sequences), args.batch_size), desc="extract embeddings"):
            batch_sequences = sequences[start : start + args.batch_size]
            tokenizer = extractor.tokenizer if args.model_name.startswith("Caduceus-") else extractor.tokenize
            encoded = tokenizer(
                batch_sequences,
                max_length=args.max_length,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            # Check after untruncated tokenization; fail rather than dropping bases.
            if encoded["input_ids"].shape[-1] > args.max_length:
                raise ValueError("untruncated sequence exceeds max_length; increase the model-supported limit")
            input_ids = encoded["input_ids"].to(extractor.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(extractor.device)
            batch_embeddings = extractor.get_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pooling=args.pooling,
                # hEvo-open-base is evaluated in bfloat16 by the shared
                # extractor.  Convert explicitly on the torch side because
                # NumPy cannot consume a CUDA/CPU bfloat16 tensor directly.
                return_numpy=False,
            )
            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.float().cpu().numpy()
            batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
            if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != len(batch_sequences):
                raise RuntimeError(f"unexpected embedding shape: {batch_embeddings.shape}")
            if not np.isfinite(batch_embeddings).all():
                raise RuntimeError(f"non-finite embedding values in rows {start}:{start + len(batch_sequences)}")
            embeddings.append(batch_embeddings)

    embedding_array = np.concatenate(embeddings, axis=0)
    if embedding_array.shape[0] != len(rows):
        raise RuntimeError("embedding row count does not match input row count")

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    temporary_output = args.output_h5.with_suffix(".partial.h5")
    with h5py.File(temporary_output, "w") as handle:
        handle.create_dataset("embeddings", data=embedding_array, dtype="float32")
        handle.create_dataset("label", data=np.asarray(labels, dtype=object), dtype=string_dtype)
        handle.create_dataset("seqID", data=np.asarray([row["seqID"] for row in rows], dtype=object), dtype=string_dtype)
        handle.attrs["model_name"] = args.model_name
        handle.attrs["model_weight_sha256"] = model_hash
        handle.attrs["input_sha256"] = input_hash
        handle.attrs["dataset_sha256"] = input_hash
        handle.attrs["dataset_id"] = dataset_id
        handle.attrs["run_id"] = args.run_id
        handle.attrs["protocol_id"] = gate["protocol_id"]
        handle.attrs["torch_seed"] = 42
        handle.attrs["truncation"] = False
        handle.attrs["row_order_sha256"] = row_order_sha256(rows)
        handle.attrs["hidden_size"] = int(embedding_array.shape[1])
        handle.attrs["n_rows"] = int(embedding_array.shape[0])
        handle.attrs["max_length"] = int(args.max_length)
        handle.attrs["pooling"] = args.pooling
        handle.attrs["inference_precision"] = inference_precision or "model_default"
        handle.attrs["evo2_config_sha256"] = evo2_config_sha256 or ""

    manifest = {
        "input_csv": str(args.input_csv),
        "output_h5": str(args.output_h5),
        "model_name": args.model_name,
        "model_path": str(args.model_path),
        "model_weight_sha256": model_hash,
        "input_sha256": input_hash,
        "dataset_sha256": input_hash,
        "dataset_id": dataset_id,
        "run_id": args.run_id,
        "protocol_id": gate["protocol_id"],
        "torch_seed": 42,
        "truncation": False,
        "row_order_sha256": row_order_sha256(rows),
        "n_rows": int(embedding_array.shape[0]),
        "n_classes": len(set(labels)),
        "hidden_size": int(embedding_array.shape[1]),
        "max_length": args.max_length,
        "pooling": args.pooling,
        "inference_precision": inference_precision or "model_default",
        "evo2_config_sha256": evo2_config_sha256 or "",
        "batch_size": args.batch_size,
        "extractor": str(Path(__file__).resolve()),
        "extractor_sha256": sha256_file(Path(__file__).resolve()),
        "shared_adapter_sha256": sha256_file(args.extractor_dir / "extract_embeddings_classes.py"),
        "shared_mapping_sha256": sha256_file(args.extractor_dir / "probe_single_seq_input.py"),
    }
    manifest_path = args.output_h5.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_output, args.output_h5)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
