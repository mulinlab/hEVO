#!/usr/bin/env python3
"""Compatibility entry point for the legacy Pareto probe protocol."""

import argparse
from pathlib import Path
import sys

EVALUATE_DIR = Path(__file__).resolve().parents[1]
if str(EVALUATE_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATE_DIR))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Linear probe + precompute embeddings + HDF5 or pickle cache"
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help="逗号分隔的模型目录路径列表（本次运行只会跑一个模型也支持多个）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="模型名称，用来对应不同的emb Extractor",
    )
    parser.add_argument(
        "--dataset_files",
        type=str,
        required=True,
        help="逗号分隔的数据文件路径列表（若为1个则按 train_ratio 划分；若>=2则第一个为train，第二个为val）",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="数据集名称（仅一个）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="训练 probe 时的 batch size（基于 emb）",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--sep", type=str, default="space")
    parser.add_argument("--seq_col", type=int, default=0)
    parser.add_argument("--label_col", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="./probe_results")
    parser.add_argument("--log_dir", type=str, default="./runs/probe")
    parser.add_argument(
        "--hdf5_dir", type=str, default="./emb_cache_hdf5", help="HDF5 文件存放目录"
    )
    parser.add_argument(
        "--recompute", action="store_true", help="强制重新计算 embeddings 并覆盖 HDF5"
    )
    parser.add_argument(
        "--del_old_runs", action="store_true", help="是否删除旧的 TensorBoard 运行日志"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="用于 precompute 时 DataLoader 的 num_workers",
    )
    parser.add_argument(
        "--precompute_batch_size", type=int, default=64, help="precompute 时每批大小"
    )
    parser.add_argument(
        "--pooling", type=str, default="mean", choices=["mean", "cls", "max"]
    )
    parser.add_argument(
        "--save_pickle",
        action="store_true",
        help="是否将 embedding 保存为 pickle 文件而非直接写入 HDF5",
    )
    parser.add_argument(
        "--pickle_dir", type=str, default="./emb_pickle", help="保存 pickle 文件的目录"
    )
    parser.add_argument(
        "--normalize_dim",
        type=int,
        default=None,
        help="统一投影维度，None 表示不投影（纯线性模式）",
    )
    parser.add_argument(
        "--multi_dim_eval",
        action="store_true",
        help="是否测试多个投影维度 [128, 256, 512, 1024]",
    )
    parser.add_argument(
        "--use_nonlinearity",
        action="store_true",
        help="在投影层后是否使用 ReLU 激活（默认保持线性）",
    )
    return parser


# Help uses the same parser as execution, before optional model imports.
if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()


def _engine():
    from core import probe_pareto_engine

    return probe_pareto_engine


def main(argv=None):
    return _engine().main(argv)


# Preserve the historical module-level API for callers that import this file.
if __name__ != "__main__":
    from core.probe_pareto_engine import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
