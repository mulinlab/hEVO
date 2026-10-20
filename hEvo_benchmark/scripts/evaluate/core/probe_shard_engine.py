#!/usr/bin/env python3
"""Sharded embedding precompute and single linear-probe merge runner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["precompute", "merge_probe"], required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_files", required=True, help="train.csv,test.csv")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--shard_count", type=int, default=1)
    p.add_argument("--merged_hdf5", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--log_dir", default="")
    p.add_argument("--sep", default=",")
    p.add_argument("--seq_col", type=int, default=1)
    p.add_argument("--label_col", type=int, default=2)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--pooling", choices=["mean", "cls", "max"], default="mean")
    p.add_argument("--precompute_batch_size", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--bootstrap", type=int, default=1000)
    return p


# Help uses the same parser as execution, before optional model imports.
if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()


import hashlib  # noqa: E402 - lightweight CLI help precedes runtime imports
import json  # noqa: E402 - lightweight CLI help precedes runtime imports

import h5py  # noqa: E402 - lightweight CLI help precedes runtime imports
import numpy as np  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch  # noqa: E402 - lightweight CLI help precedes runtime imports

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))
import probe_single_seq_input as probe  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = build_parser()
    return p.parse_args()


def read_split(path: str, args: argparse.Namespace):
    return probe.read_data_file(
        path, sep=args.sep, seq_col=args.seq_col, label_col=args.label_col
    )


def precompute(args: argparse.Namespace) -> None:
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid shard index/count")
    train_path, test_path = [x.strip() for x in args.dataset_files.split(",")]
    extractor = probe.model_extractor_map[args.model_name](
        args.model_path, args.model_name
    )
    shard_path = Path(args.shard_dir) / f"shard-{args.shard_index:03d}.h5"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if shard_path.exists():
        shard_path.unlink()

    with h5py.File(shard_path, "w") as h5:
        for split, source in (("train", train_path), ("test", test_path)):
            sequences, labels = read_split(source, args)
            indices = np.arange(len(sequences), dtype=np.int64)[
                args.shard_index :: args.shard_count
            ]
            seqs = [sequences[int(i)] for i in indices]
            lbs = [labels[int(i)] for i in indices]
            temporary = shard_path.with_suffix(f".{split}.tmp.h5")
            if temporary.exists():
                temporary.unlink()
            probe.precompute_and_cache_embeddings_hdf5(
                extractor,
                sequences=seqs,
                labels=lbs,
                hdf5_path=str(temporary),
                split_name=split,
                batch_size=args.precompute_batch_size,
                num_workers=0,
                max_length=args.max_length,
                pooling=args.pooling,
            )
            with h5py.File(temporary, "r") as src_h5:
                src_group = src_h5[split]
                dst_group = h5.create_group(split)
                dst_group.create_dataset("embeddings", data=src_group["embeddings"][:])
                dst_group.create_dataset("labels", data=src_group["labels"][:])
                dst_group.create_dataset("indices", data=indices)
                dst_group.attrs["source_file"] = source
                dst_group.attrs["source_sha256"] = sha256_file(source)
            temporary.unlink()
    manifest = {
        "mode": "precompute",
        "model_name": args.model_name,
        "model_path": args.model_path,
        "model_sha256": sha256_file(os.path.join(args.model_path, "pytorch_model.bin")),
        "dataset_name": args.dataset_name,
        "dataset_files": [train_path, test_path],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_file": str(shard_path),
        "max_length": args.max_length,
        "pooling": args.pooling,
    }
    shard_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


def merge_group(shard_paths: list[Path], split: str, expected_labels: list[int]):
    parts = []
    for path in shard_paths:
        with h5py.File(path, "r") as h5:
            group = h5[split]
            parts.append(
                (group["indices"][:], group["embeddings"][:], group["labels"][:])
            )
    indices = np.concatenate([x[0] for x in parts])
    embeddings = np.concatenate([x[1] for x in parts])
    labels = np.concatenate([x[2] for x in parts])
    order = np.argsort(indices, kind="stable")
    indices, embeddings, labels = indices[order], embeddings[order], labels[order]
    expected = np.arange(len(expected_labels), dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise RuntimeError(f"{split}: shard indices have duplicates or omissions")
    if not np.array_equal(
        labels.astype(np.int64), np.asarray(expected_labels, dtype=np.int64)
    ):
        raise RuntimeError(f"{split}: merged labels differ from source labels")
    return embeddings, labels


def merge_probe(args: argparse.Namespace) -> None:
    train_path, test_path = [x.strip() for x in args.dataset_files.split(",")]
    train_sequences, train_labels = read_split(train_path, args)
    test_sequences, test_labels = read_split(test_path, args)
    shard_paths = [
        Path(args.shard_dir) / f"shard-{i:03d}.h5" for i in range(args.shard_count)
    ]
    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing shard files: " + ", ".join(missing))
    train_embs, train_y = merge_group(shard_paths, "train", train_labels)
    test_embs, test_y = merge_group(shard_paths, "test", test_labels)
    Path(args.merged_hdf5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.merged_hdf5, "w") as h5:
        for split, embeddings, labels in (
            ("train", train_embs, train_y),
            ("test", test_embs, test_y),
        ):
            group = h5.create_group(split)
            group.create_dataset("embeddings", data=embeddings.astype(np.float32))
            group.create_dataset("labels", data=labels.astype(np.int64))
            group.attrs["merged_from_shards"] = args.shard_count
            group.attrs["source_file"] = train_path if split == "train" else test_path

    train_embs_t = torch.from_numpy(train_embs.astype(np.float32))
    test_embs_t = torch.from_numpy(test_embs.astype(np.float32))
    train_y_t = torch.from_numpy(train_y.astype(np.int64))
    test_y_t = torch.from_numpy(test_y.astype(np.int64))
    train_loader = torch.utils.data.DataLoader(
        probe.EmbeddingDataset(train_embs_t, train_y_t),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=probe.collate_emb_fn,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        probe.EmbeddingDataset(test_embs_t, test_y_t),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=probe.collate_emb_fn,
        num_workers=0,
    )
    num_classes = len(set(train_y.tolist()))
    linear_probe = probe.FlexibleProbe(
        train_embs.shape[1], num_classes, normalize_dim=None
    )
    trained = probe.train_probe_on_embeddings(
        linear_probe,
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=None,
        writer=None,
    )
    metrics, labels_all, preds_all, probs_all = probe.eval_probe_on_embeddings(
        trained, test_loader, device=None, return_samples=True
    )
    bootstrap = probe.bootstrap_ci(labels_all, preds_all, probs_all, args.bootstrap, 40)
    result = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "train_samples": len(train_y),
        "test_samples": len(test_y),
        "num_classes": num_classes,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "model_sha256": sha256_file(os.path.join(args.model_path, "pytorch_model.bin")),
        "dataset_sha256": [sha256_file(train_path), sha256_file(test_path)],
        "merged_hdf5": args.merged_hdf5,
        "shard_count": args.shard_count,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "evaluator_sha256": sha256_file(str(EVAL_DIR / "probe_single_seq_input.py")),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    output = Path(args.out_dir) / f"results_{args.model_name}_{args.dataset_name}.json"
    output.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result, indent=2, default=float))


def main() -> None:
    args = parse_args()
    if args.mode == "precompute":
        precompute(args)
    else:
        merge_probe(args)


if __name__ == "__main__":
    main()
