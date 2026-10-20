#!/usr/bin/env python3
"""Compatibility entry point for sharded standard-protocol evaluation."""

import argparse
from pathlib import Path
import sys

EVALUATE_DIR = Path(__file__).resolve().parents[1]
if str(EVALUATE_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATE_DIR))


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


if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()


def _engine():
    from core import probe_shard_engine

    return probe_shard_engine


def main():
    return _engine().main()


if __name__ != "__main__":
    from core.probe_shard_engine import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
