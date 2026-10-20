#!/usr/bin/env python3
# run_probe_eval_with_hdf5.py
"""
线性探测器 + embedding 预计算 + HDF5 文件缓存

用法示例:
python run_probe_eval_with_hdf5.py \
  --model_paths /path/to/model \
  --dataset_files /path/to/data_train.txt,/path/to/data_val.txt \
  --dataset_name H3K4me1 \
  --hdf5_dir ./emb_cache \
  --batch_size 64 \
  --precompute_batch_size 64 \
  --num_workers 4

如果只有一个 dataset_file，则会按 --train_ratio 做内部划分（train/val）。
若 HDF5 中已有对应 split 的 embeddings 并且未指定 --recompute，则直接加载。
"""

import argparse
import os
import sys
from pathlib import Path

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


import random  # noqa: E402 - lightweight CLI help precedes runtime imports
import time  # noqa: E402 - lightweight CLI help precedes runtime imports
from typing import Dict, List, Optional, Tuple, Union  # noqa: E402 - lightweight CLI help precedes runtime imports

from core import probe_common as _probe_common  # noqa: E402 - lightweight help precedes runtime imports
import h5py  # Additional import.  # noqa: E402 - lightweight CLI help precedes runtime imports.
import numpy as np  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch  # noqa: E402 - lightweight CLI help precedes runtime imports
import tqdm  # noqa: E402 - lightweight CLI help precedes runtime imports
from adapters.extract_embeddings_classes import (  # noqa: E402 - lightweight CLI help precedes runtime imports
    ExtractEmbeddingForAIDO,
    ExtractEmbeddingForCaduceus,
    ExtractEmbeddingForDNABert,
    ExtractEmbeddingForEnformer,
    ExtractEmbeddingForEvo2_1B,
    ExtractEmbeddingForEvo2_7B,
    ExtractEmbeddingForGENALM,
    ExtractEmbeddingForGENERator,
    ExtractEmbeddingForGenos_1B,
    ExtractEmbeddingForGPN,
    ExtractEmbeddingForGROVER,
    ExtractEmbeddingForhEvo,
    ExtractEmbeddingForhEvo_rand,
    ExtractEmbeddingForhEvo_rare,
    ExtractEmbeddingForHyena,
    ExtractEmbeddingForMistralDNA,
    ExtractEmbeddingForNT,
    ExtractEmbeddingForNTv3,
    ExtractEmbeddingForSpeciesLM,
)
from filelock import FileLock  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch import nn  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch.utils.data import DataLoader, Dataset  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch.utils.tensorboard import SummaryWriter  # noqa: E402 - lightweight CLI help precedes runtime imports

# ========= Random seed setup ========= #
SEED = 40
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ================================= #


# ---------- Extractor ----------

# ---------- Extractor mapping ----------
model_extractor_map = {
    "DNABERT2": ExtractEmbeddingForDNABert,
    "DNABERT": ExtractEmbeddingForDNABert,
    "hEvo-1kG-86M": ExtractEmbeddingForhEvo,
    "hEvo-1kG-sliding128-86M": ExtractEmbeddingForhEvo_rare,
    "hEvo-open-base": ExtractEmbeddingForhEvo_rare,
    "hEvo-base": ExtractEmbeddingForhEvo_rare,
    "hEvo-large": ExtractEmbeddingForhEvo_rare,
    "hEvo-fullUKB-86M": ExtractEmbeddingForhEvo_rand,
    "hEvo-60k-86M": ExtractEmbeddingForhEvo_rare,
    "evo2_7b_base": ExtractEmbeddingForEvo2_7B,
    "evo2_1b_base": ExtractEmbeddingForEvo2_1B,
    "NT_2.5b_1kG": ExtractEmbeddingForNT,
    "NT_500m_1kG": ExtractEmbeddingForNT,
    "NT_500m_MS": ExtractEmbeddingForNT,
    "NT_2.5b_MS": ExtractEmbeddingForNT,
    "HyenaDNA_tiny_1k": ExtractEmbeddingForHyena,
    "GPN-MSA": ExtractEmbeddingForGPN,
    "GPN-promoter": ExtractEmbeddingForGPN,
    "SpeciesLM": ExtractEmbeddingForSpeciesLM,
    "Genos-1.2B": ExtractEmbeddingForGenos_1B,
    # "Genos-10B": ExtractEmbeddingForGenos_10B,
    "AIDO.DNA-300M": ExtractEmbeddingForAIDO,
    # "AIDO.DNA-7B": ExtractEmbeddingForAIDO,
    "GENA-LM-base": ExtractEmbeddingForGENALM,
    "GENA-LM-large": ExtractEmbeddingForGENALM,
    "GROVER": ExtractEmbeddingForGROVER,
    "Mistral-DNA-17M": ExtractEmbeddingForMistralDNA,
    "Mistral-DNA-138M": ExtractEmbeddingForMistralDNA,
    "Mistral-DNA-422M": ExtractEmbeddingForMistralDNA,
    "NTv3-8M-pre": ExtractEmbeddingForNTv3,
    "NTv3-100M-pre": ExtractEmbeddingForNTv3,
    "NTv3-650M-pre": ExtractEmbeddingForNTv3,
    "GENERator-1.2B": ExtractEmbeddingForGENERator,
    "GENERator-3B": ExtractEmbeddingForGENERator,
    "Enformer": ExtractEmbeddingForEnformer,
    "Caduceus-PH": ExtractEmbeddingForCaduceus,
    "Caduceus-PS": ExtractEmbeddingForCaduceus,
}
# ---------- I/O and datasets ----------


class EfficiencyTracker:
    """用于记录模型推理效率的工具类"""

    def __init__(self, device):
        self.device = device
        self.start_time = 0
        self.end_time = 0
        self.start_vram = 0
        self.peak_vram = 0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            self.start_vram = torch.cuda.memory_allocated(self.device)
        self.start_time = time.perf_counter()

    def stop(self, num_samples):
        self.end_time = time.perf_counter()
        total_time = self.end_time - self.start_time

        if torch.cuda.is_available():
            # Record peak GPU memory usage in MB.
            self.peak_vram = (
                (torch.cuda.max_memory_allocated(self.device) - self.start_vram)
                / 1024
                / 1024
            )

        metrics = {
            "total_inference_time_s": total_time,
            "latency_ms_per_sample": (total_time / num_samples) * 1000
            if num_samples > 0
            else 0,
            "throughput_samples_per_s": num_samples / total_time
            if total_time > 0
            else 0,
            "peak_vram_mb": self.peak_vram,
        }
        return metrics


## Utility functions ##


def count_parameters(model):
    """
    鲁棒的参数统计函数，兼容标准 nn.Module 和 Evo2/StripedHyena 包装器
    """
    # 1. Try the standard method.
    if hasattr(model, "parameters"):
        params = list(model.parameters())
        if len(params) > 0:
            return sum(p.numel() for p in params) / 1e6

    # 2. Check common wrapper attributes used by Evo2 and StripedHyena.
    # In many implementations, the underlying model is exposed as .model or .backbone.
    for attr_name in ["model", "backbone", "base_model", "module"]:
        if hasattr(model, attr_name):
            sub_obj = getattr(model, attr_name)
            if hasattr(sub_obj, "parameters"):
                return sum(p.numel() for p in sub_obj.parameters()) / 1e6

    # 3. Try accessing named parameters directly for special implementations.
    if hasattr(model, "named_parameters"):
        return sum(p.numel() for n, p in model.named_parameters()) / 1e6

    return 0.0


def calculate_efficiency_ratios(metrics_mean: float, flops_g: float, params_m: float):
    """
    计算性价比指标
    """
    # Performance per GFLOP (using MCC as an example).
    compute_efficiency = metrics_mean / flops_g if flops_g > 0 else 0
    # Performance per million parameters.
    param_efficiency = metrics_mean / params_m if params_m > 0 else 0
    return compute_efficiency, param_efficiency


## End of utility functions ##


class DNADataset(Dataset):
    def __init__(
        self,
        sequences: Optional[List[str]] = None,
        labels: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        pretokenized: Optional[Dict[str, torch.Tensor]] = None,
        tokenizer_callable=None,
        max_length: int = 1024,
    ):
        if pretokenized is None and sequences is None:
            raise ValueError("Either sequences or pretokenized must be provided.")
        self.sequences = sequences
        self.labels = (
            None
            if labels is None
            else (
                torch.tensor(labels, dtype=torch.long)
                if not isinstance(labels, torch.Tensor)
                else labels
            )
        )
        self.pretokenized = pretokenized
        self.tokenizer_callable = tokenizer_callable
        self.max_length = max_length

        if self.pretokenized is not None:
            assert (
                "input_ids" in self.pretokenized
                and "attention_mask" in self.pretokenized
            )
            if isinstance(self.pretokenized["input_ids"], np.ndarray):
                self.pretokenized["input_ids"] = torch.from_numpy(
                    self.pretokenized["input_ids"]
                )
            if isinstance(self.pretokenized["attention_mask"], np.ndarray):
                self.pretokenized["attention_mask"] = torch.from_numpy(
                    self.pretokenized["attention_mask"]
                )

    def __len__(self):
        return (
            len(self.labels)
            if self.labels is not None
            else (
                len(self.sequences)
                if self.sequences is not None
                else len(self.pretokenized["input_ids"])
            )
        )

    def __getitem__(self, idx: int):
        if self.pretokenized is not None:
            item = {
                "input_ids": self.pretokenized["input_ids"][idx],
                "attention_mask": self.pretokenized["attention_mask"][idx],
            }
        else:
            assert self.tokenizer_callable is not None, (
                "tokenizer_callable required for sequences input"
            )
            encoded = self.tokenizer_callable(
                [self.sequences[idx]],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if "attention_mask" not in encoded:
                item = {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": torch.ones_like(encoded["input_ids"].squeeze(0)),
                }
            else:
                item = {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    labels = (
        torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
        if "label" in batch[0]
        else None
    )
    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if labels is not None:
        out["labels"] = labels
    return out


# ---------- Embedding dataset for probe training ----------
EmbeddingDataset = _probe_common.EmbeddingDataset
collate_emb_fn = _probe_common.collate_emb_fn


# ---------- Probe Model ----------
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)
        return self.linear(x)


FlexibleProbe = _probe_common.FlexibleProbe

# ---------- HDF5 access helpers ----------


def precompute_and_cache_embeddings_hdf5(
    extractor,
    sequences: List[str],
    labels: List[int],
    hdf5_path: str,
    split_name: str,
    batch_size: int = 64,
    num_workers: int = 4,
    max_length: int = 512,
    pooling: str = "mean",
    lock_file: Optional[str] = None,
):
    """
    计算 sequences 的 embedding 并存入 HDF5。
    hdf5 结构:
      /{split_name}/embeddings   -> dataset (N, H)
      /{split_name}/labels       -> dataset (N,)
    """

    ds = DNADataset(
        sequences=sequences,
        labels=labels,
        tokenizer_callable=extractor.tokenize,
        max_length=max_length,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # pin_memory=True,
    )

    all_embs = []
    all_labels = []
    if (extractor.model_name == "evo2-7b") or (extractor.model_name == "evo2-1b"):
        pass
    else:
        extractor.model.eval()
    # --- Efficiency monitoring ---
    tracker = EfficiencyTracker(extractor.device)
    tracker.start()

    with torch.inference_mode():
        for batch in tqdm.tqdm(loader, desc=f"Precompute {split_name} emb"):
            input_ids = batch["input_ids"].to(extractor.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(extractor.device)
            emb = extractor.get_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pooling=pooling,
                return_numpy=False,
            )
            all_embs.append(emb.detach().cpu())
            if "labels" in batch:
                all_labels.append(batch["labels"])
    efficiency_stats = tracker.stop(len(sequences))

    embeddings = torch.cat(all_embs, dim=0)
    labels_t = (
        torch.cat(all_labels, dim=0)
        if len(all_labels) > 0
        else torch.tensor([], dtype=torch.long)
    )

    # Write to HDF5.
    def _write_hdf5(path, split, embs_np, lbls_np):
        with h5py.File(path, "a") as f:
            if split in f:
                del f[split]  # Remove the existing group before writing.
            grp = f.create_group(split)
            grp.create_dataset("embeddings", data=embs_np)
            grp.create_dataset("labels", data=lbls_np)
            # Optionally store metadata.
            grp.attrs["num"] = embs_np.shape[0]
            grp.attrs["hidden_size"] = embs_np.shape[1]
            grp.attrs["dtype"] = str(embs_np.dtype)

    embs_np = embeddings.to(torch.float32).numpy()
    lbls_np = labels_t.numpy()

    if lock_file:
        lock = FileLock(lock_file, timeout=300)
        with lock:
            _write_hdf5(hdf5_path, split_name, embs_np, lbls_np)
    else:
        _write_hdf5(hdf5_path, split_name, embs_np, lbls_np)

    # Optionally store efficiency metrics as HDF5 attributes for later access.
    with h5py.File(hdf5_path, "a") as f:
        if split_name in f:
            grp = f[split_name]
            for k, v in efficiency_stats.items():
                grp.attrs[k] = v

    return efficiency_stats


def load_embeddings_from_hdf5(
    hdf5_path: str, split_name: str
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    从 HDF5 文件加载指定 split 的 embeddings
    """
    if not os.path.exists(hdf5_path):
        return None

    try:
        with h5py.File(hdf5_path, "r") as f:
            if split_name not in f:
                return None
            grp = f[split_name]
            if "embeddings" not in grp or "labels" not in grp:
                return None

            # Read the data.
            emb_np = grp["embeddings"][:]
            labels_np = grp["labels"][:]

            emb = torch.from_numpy(emb_np)
            labels = torch.from_numpy(labels_np)
            return emb, labels
    except OSError:
        # The file may be corrupt or currently being written.
        return None


# ---------- Train and evaluate using precomputed embeddings ----------


def eval_probe_on_embeddings(
    probe: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    return_samples: bool = False,
):
    return _probe_common.eval_probe_on_embeddings(
        probe,
        data_loader,
        device=device,
        return_samples=return_samples,
        include_per_class_f1=False,
    )


# ---------- Bootstrap confidence intervals (backward compatible) ----------


# ---------- Run one model on one dataset (main workflow) ----------
def run_one_model_on_dataset(
    model_path: str,
    dataset_files: List[str],
    dataset_name: str,
    args,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    model_name = args.model_name
    print(f"=== Model: {model_name} on dataset_files {dataset_files} ===")

    extractor = model_extractor_map[args.model_name](model_path, model_name)

    # Try to determine the parameter count manually.
    try:
        backbone_params = sum(p.numel() for p in extractor.model.parameters()) / 1e6
    except AttributeError:
        # Special handling for Evo2.
        if hasattr(extractor.model, "model"):
            backbone_params = (
                sum(p.numel() for p in extractor.model.model.parameters()) / 1e6
            )
        elif hasattr(extractor.model, "backbone"):
            backbone_params = (
                sum(p.numel() for p in extractor.model.backbone.parameters()) / 1e6
            )
        else:
            backbone_params = 0.0

    # --- Calculate backbone FLOPs ---
    print(f"Calculating Backbone FLOPs for length {args.max_length}...")
    backbone_flops_g = 0
    # --------------------------------
    # Record efficiency during extraction; evaluating on the test/validation set provides a fairer comparison.
    backbone_efficiency = {}

    # In pickle mode, save a pickle file instead of writing to HDF5.
    if args.save_pickle:
        for split_name, seqs, lbs in [
            ("train", None, None),
            ("val", None, None),
        ]:
            # Read the data file.
            if len(dataset_files) == 0:
                raise ValueError("至少需要一个 dataset_file")
            elif len(dataset_files) == 1:
                sequences_all, labels_all = read_data_file(
                    dataset_files[0],
                    sep=args.sep,
                    seq_col=args.seq_col,
                    label_col=args.label_col,
                )
                if len(sequences_all) == 0:
                    raise RuntimeError(f"No sequences read from {dataset_files[0]}")
                split_idx = int(args.train_ratio * len(sequences_all))
                if split_name == "train":
                    seqs = sequences_all[:split_idx]
                    lbs = labels_all[:split_idx]
                else:
                    seqs = sequences_all[split_idx:]
                    lbs = labels_all[split_idx:]
            else:
                if split_name == "train":
                    seqs, lbs = read_data_file(
                        dataset_files[0],
                        sep=args.sep,
                        seq_col=args.seq_col,
                        label_col=args.label_col,
                    )
                else:
                    seqs, lbs = read_data_file(
                        dataset_files[1],
                        sep=args.sep,
                        seq_col=args.seq_col,
                        label_col=args.label_col,
                    )
                if len(seqs) == 0:
                    print(f"Warning: {split_name} set empty.")
            if seqs is None or len(seqs) == 0:
                continue
            print(
                f"Precomputing embeddings for {split_name} (n={len(seqs)}) and saving as pickle..."
            )
            precompute_and_save_embeddings(
                extractor,
                sequences=seqs,
                labels=lbs,
                output_dir=args.pickle_dir,
                split_name=split_name,
                batch_size=args.precompute_batch_size,
                num_workers=args.num_workers,
                max_length=args.max_length,
                pooling=args.pooling,
            )
        print(f"All embeddings saved as pickle files in {args.pickle_dir}.")
        return {}  # Do not train the probe again.

    # Otherwise, use HDF5 mode.
    hdf5_dir = args.hdf5_dir
    os.makedirs(hdf5_dir, exist_ok=True)
    # Use one file per dataset: {model_name}_{dataset_name}.h5.
    hdf5_filename = f"{model_name}_{dataset_name}.h5"
    hdf5_path = os.path.join(hdf5_dir, hdf5_filename)

    # Define the lock-file path.
    lock_file = os.path.join(hdf5_dir, f"{hdf5_filename}.lock")

    # Read the data files; multiple files are supported.
    if len(dataset_files) == 0:
        raise ValueError("至少需要一个 dataset_file")
    elif len(dataset_files) == 1:
        sequences_all, labels_all = read_data_file(
            dataset_files[0],
            sep=args.sep,
            seq_col=args.seq_col,
            label_col=args.label_col,
        )
        if len(sequences_all) == 0:
            raise RuntimeError(f"No sequences read from {dataset_files[0]}")
        split_idx = int(args.train_ratio * len(sequences_all))
        seq_train = sequences_all[:split_idx]
        lbl_train = labels_all[:split_idx]
        seq_val = sequences_all[split_idx:]
        lbl_val = labels_all[split_idx:]
    else:
        seq_train, lbl_train = read_data_file(
            dataset_files[0],
            sep=args.sep,
            seq_col=args.seq_col,
            label_col=args.label_col,
        )
        seq_val, lbl_val = read_data_file(
            dataset_files[1],
            sep=args.sep,
            seq_col=args.seq_col,
            label_col=args.label_col,
        )
        if len(seq_train) == 0:
            raise RuntimeError(f"No sequences read from {dataset_files[0]}")
        if len(seq_val) == 0:
            print("Warning: validation set empty.")

    # Check whether the HDF5 file contains embeddings for train and validation separately.
    for split_name, seqs, lbs in [
        ("train", seq_train, lbl_train),
        ("val", seq_val, lbl_val),
    ]:
        if seqs is None or len(seqs) == 0:
            continue

        # Check whether it already exists.
        loaded = load_embeddings_from_hdf5(hdf5_path, split_name)

        if (loaded is None) or args.recompute:
            print(f"Precomputing embeddings for {split_name} (n={len(seqs)}) ...")
            stats = precompute_and_cache_embeddings_hdf5(
                extractor,
                sequences=seqs,
                labels=lbs,
                hdf5_path=hdf5_path,
                split_name=split_name,
                batch_size=args.precompute_batch_size,
                num_workers=args.num_workers,
                max_length=args.max_length,
                pooling=args.pooling,
                lock_file=lock_file,
            )
            if split_name == "val":  # Efficiency is generally measured during inference (validation).
                backbone_efficiency = stats
        else:
            print(f"Found cached embeddings in HDF5 for {split_name}, loading...")

    train_loaded = load_embeddings_from_hdf5(hdf5_path, "train")
    val_loaded = load_embeddings_from_hdf5(hdf5_path, "val")

    if train_loaded is None:
        raise RuntimeError("训练集 embeddings 缺失，无法训练")
    train_embs, train_labels = train_loaded
    if val_loaded is None:
        print("Warning: val embeddings missing; evaluation will be skipped.")
        val_embs, val_labels = None, None
    else:
        val_embs, val_labels = val_loaded

    hidden_size = train_embs.shape[1]
    num_classes = len(set(train_labels.tolist()))

    # ========== Support evaluation at multiple dimensions ========== #
    if args.multi_dim_eval:
        dims_to_test = [128, 256, 512, 1024, None]  # None means no projection.
    else:
        dims_to_test = [args.normalize_dim]  # A single dimension or None.

    # Store results for each dimension.
    all_results = {}

    for test_dim in dims_to_test:
        dim_name = f"dim_{test_dim}" if test_dim is not None else "no_projection"

        print(f"\n{'=' * 60}")
        print(f"Testing with projection dim: {test_dim}")
        print(f"{'=' * 60}\n")

        # Create the probe.
        probe = FlexibleProbe(
            input_dim=hidden_size,
            num_classes=num_classes,
            normalize_dim=test_dim,
            use_nonlinearity=args.use_nonlinearity,
        )

        # Prepare the data loaders.
        train_dataset = EmbeddingDataset(train_embs, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_emb_fn,
            num_workers=0,
        )

        val_loader = None
        if val_embs is not None:
            val_dataset = EmbeddingDataset(val_embs, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_emb_fn,
                num_workers=0,
            )

        # Train.
        trained_probe = train_probe_on_embeddings(
            probe,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=None,
            writer=writer,
        )

        # Evaluate.
        if val_loader is not None:
            metrics, labels_all, preds_all, probs_all = eval_probe_on_embeddings(
                trained_probe, val_loader, device=None, return_samples=True
            )
            bs = bootstrap_ci(
                labels_all,
                preds_all,
                probs_all,
                n_bootstrap=args.bootstrap,
                random_state=SEED,
            )
        else:
            _metrics = {}
            bs = {}

        # Log to TensorBoard with the dimension as a tag.
        if writer:
            for m in ["accuracy", "auc", "mcc"]:
                if m in bs:
                    mean, lo, hi = bs[m]
                    writer.add_scalar(f"eval/{dataset_name}/{dim_name}/{m}", mean, 0)

        # Save the results for the current dimension.
        all_results[dim_name] = {}
        for k, v in bs.items():
            all_results[dim_name][k] = v

        # --- Prepare data for the Pareto plot ---
        probe_params = count_parameters(trained_probe)
        val_mcc_mean = bs.get("mcc", (0, 0, 0))[0]

        compute_eff, param_eff = calculate_efficiency_ratios(
            val_mcc_mean, backbone_flops_g, backbone_params
        )
        dim_results = {
            "performance": bs,
            "efficiency": {
                "backbone_params_M": backbone_params,
                "probe_params_M": probe_params,
                "total_params_M": backbone_params + probe_params,
                "backbone_flops_G": backbone_flops_g,  # Theoretical compute cost.
                "backbone_throughput_fps": backbone_efficiency.get(
                    "throughput_samples_per_s", 0
                ),
                # Base throughput (bp/s), an important metric for DNA models.
                "backbone_throughput_bps": backbone_efficiency.get(
                    "throughput_samples_per_s", 0
                )
                * args.max_length,
                "backbone_latency_ms": backbone_efficiency.get(
                    "latency_ms_per_sample", 0
                ),
                "backbone_peak_vram_mb": backbone_efficiency.get("peak_vram_mb", 0),
                # Cost-effectiveness metrics.
                "mcc_per_gflops": compute_eff,
                "mcc_per_mparams": param_eff,
                "embedding_dim": hidden_size,
                "projection_dim": test_dim if test_dim else hidden_size,
                "sequence_length": args.max_length,
            },
        }
        all_results[dim_name] = dim_results

    # ========== Return results for all dimensions ========== #
    return {dataset_name: all_results}


# ---------- Main program ----------
def main(argv=None):
    SEED = 40
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    parser = build_parser()

    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.del_old_runs and os.path.exists(args.log_dir):
        import shutil

        shutil.rmtree(args.log_dir, ignore_errors=True)

    model_paths = [p.strip() for p in args.model_paths.split(",") if p.strip()]
    dataset_files = [p.strip() for p in args.dataset_files.split(",") if p.strip()]

    writer = SummaryWriter(log_dir=args.log_dir)

    results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    for mpath in model_paths:
        model_name = os.path.basename(mpath.rstrip("/"))
        results[model_name] = {}
        out = run_one_model_on_dataset(
            mpath, dataset_files, args.dataset_name, args, writer=writer
        )
        results[model_name].update(out)

    metrics = ["accuracy", "auc", "mcc"]
    n_models = len(model_paths)
    n_datasets = 1
    model_names = [os.path.basename(p.rstrip("/")) for p in model_paths]
    dataset_names = [args.dataset_name]

    # ========== Support saving results for multiple dimensions ========== #
    metrics = ["accuracy", "auc", "mcc"]
    n_models = len(model_paths)
    n_datasets = 1
    model_names = [os.path.basename(p.rstrip("/")) for p in model_paths]
    dataset_names = [args.dataset_name]

    # Iterate over all dimensions when running a multidimensional evaluation.
    if args.multi_dim_eval:
        test_dims = ["dim_128", "dim_256", "dim_512", "dim_1024", "no_projection"]
    else:
        test_dims = [
            f"dim_{args.normalize_dim}"
            if args.normalize_dim is not None
            else "no_projection"
        ]

    for dim_name in test_dims:
        print(f"\n{'=' * 60}")
        print(f"Results summary for {dim_name}:")
        print(f"{'=' * 60}")

        for metric in metrics:
            means = np.zeros((n_models, n_datasets), dtype=float)
            lowers = np.zeros_like(means)
            uppers = np.zeros_like(means)

            for i, mname in enumerate(model_names):
                for j, dname in enumerate(dataset_names):
                    # Access the nested structure: results[model][dataset][dim][metric].
                    val = results[mname].get(dname, {}).get(dim_name, {})
                    if metric in val:
                        mean, lo, hi = val[metric]
                    else:
                        mean, lo, hi = float("nan"), float("nan"), float("nan")
                    means[i, j] = mean
                    lowers[i, j] = lo
                    uppers[i, j] = hi

                    # Log to TensorBoard.
                    writer.add_scalar(f"eval/{dname}/{dim_name}/{metric}", mean, i)

            # Print the metrics for this dimension.
            print(
                f"  {metric}: {means[0, 0]:.4f} [{lowers[0, 0]:.4f}, {uppers[0, 0]:.4f}]"
            )

    # Save the complete results to JSON.
    import json as _json

    output_filename = f"results_{args.model_name}_{args.dataset_name}"
    if args.multi_dim_eval:
        output_filename += "_multi_dim"
    elif args.normalize_dim is not None:
        output_filename += f"_dim{args.normalize_dim}"
    output_filename += ".json"

    with open(
        os.path.join(args.out_dir, output_filename),
        "w",
        encoding="utf-8",
    ) as f:
        _json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {os.path.join(args.out_dir, output_filename)}")
    print(f"{'=' * 60}\n")

    writer.close()
    print(
        f"All done. Results saved to {args.out_dir}, tensorboard logs in {args.log_dir}"
    )


def read_data_file(
    file_path: str,
    sep: str = "space",
    seq_col: Optional[int] = 0,
    label_col: Optional[int] = 3,
) -> Tuple[List[str], List[int]]:
    return _probe_common.read_data_file(
        file_path=file_path, sep=sep, seq_col=seq_col, label_col=label_col
    )


def save_embedding_as_pickle(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    split_name: str,
    batch_index: int,
):
    return _probe_common.save_embedding_as_pickle(
        embeddings=embeddings,
        labels=labels,
        output_dir=output_dir,
        split_name=split_name,
        batch_index=batch_index,
    )


def precompute_and_save_embeddings(
    extractor,
    sequences: List[str],
    labels: List[int],
    output_dir: str,
    split_name: str,
    batch_size: int = 64,
    num_workers: int = 4,
    max_length: int = 512,
    pooling: str = "mean",
):
    return _probe_common.precompute_and_save_embeddings(
        extractor=extractor,
        sequences=sequences,
        labels=labels,
        output_dir=output_dir,
        split_name=split_name,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        pooling=pooling,
        dataset_type=DNADataset,
        collate_batch=collate_fn,
    )


def train_probe_on_embeddings(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    writer: Optional[SummaryWriter] = None,
):
    return _probe_common.train_probe_on_embeddings(
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        writer=writer,
        evaluate=eval_probe_on_embeddings,
    )


def bootstrap_ci(
    labels: List[int],
    preds: List[int],
    probs: List[List[float]],
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> Dict[str, Tuple[float, float, float]]:
    return _probe_common.bootstrap_ci(
        labels=labels,
        preds=preds,
        probs=probs,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
