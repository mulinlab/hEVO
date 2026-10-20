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
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import pickle
import random
import socket
import sys
from typing import Dict, List, Optional, Tuple, Union

import h5py  # Additional import.
import numpy as np
import torch
import tqdm
from extract_embeddings_classes import (
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
from filelock import FileLock  # Keep the import available for file locking.
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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
    "hEvo-open-large": ExtractEmbeddingForhEvo_rare,
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


def read_data_file(
    file_path: str,
    sep: str = "space",
    seq_col: Optional[int] = 0,
    label_col: Optional[int] = 3,
) -> Tuple[List[str], List[int]]:
    sequences = []
    labels = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if sep == "space":
                parts = line.split(" ")
            elif sep == "tab":
                parts = line.split("\t")
            else:
                parts = line.split(sep)
            if (
                seq_col is not None
                and label_col is not None
                and len(parts) > max(seq_col, label_col)
            ):
                try:
                    labels.append(int(parts[label_col]))
                    sequences.append(parts[seq_col])
                except Exception as e:
                    print(f"跳过行，无法解析标签为整数: {line}. 错误: {e}")
                    continue

            else:
                continue
    return sequences, labels


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
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.emb = embeddings
        self.labels = labels

    def __len__(self):
        return self.emb.size(0)

    def __getitem__(self, idx):
        out = {"input_emb": self.emb[idx]}
        if self.labels is not None:
            out["labels"] = int(self.labels[idx].item())
        return out


def collate_emb_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    embs = torch.stack([b["input_emb"] for b in batch], dim=0)
    out = {"input_emb": embs}
    if "labels" in batch[0]:
        out["labels"] = torch.tensor(
            [int(b["labels"]) for b in batch], dtype=torch.long
        )
    return out


# ---------- Probe Model ----------
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)
        return self.linear(x)


class FlexibleProbe(nn.Module):
    """支持纯线性和维度归一化两种模式"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        normalize_dim: Optional[int] = None,
        use_nonlinearity: bool = False,
    ):
        super().__init__()
        if normalize_dim is None:
            # Linear-only mode.
            self.probe = nn.Linear(input_dim, num_classes)
        else:
            # Dimension-normalization mode.
            layers = [nn.Linear(input_dim, normalize_dim)]
            if use_nonlinearity:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(normalize_dim, num_classes))
            self.probe = nn.Sequential(*layers)

    def forward(self, x):
        return self.probe(x)


# ---------- HDF5 access helpers ----------


def filesystem_fingerprint(value: str) -> str:
    root = Path(value)
    candidates = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in candidates:
        stat = item.stat()
        relative = item.name if root.is_file() else str(item.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a content hash for a regular file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_dataset_file(
    path: str,
    sep: str,
    seq_col: Optional[int],
    label_col: Optional[int],
) -> dict[str, object]:
    """Capture the exact input file and the rows accepted by read_data_file."""
    file_path = Path(path).expanduser().resolve()
    record: dict[str, object] = {
        "path": str(file_path),
        "exists": file_path.is_file(),
        "status": "AVAILABLE" if file_path.is_file() else "NOT_AVAILABLE",
    }
    if not file_path.is_file():
        return record
    record["size_bytes"] = file_path.stat().st_size
    record["sha256"] = sha256_file(file_path)
    with file_path.open("r", encoding="utf-8") as handle:
        record["nonempty_lines"] = sum(1 for line in handle if line.strip())
    sequences, labels = read_data_file(
        str(file_path), sep=sep, seq_col=seq_col, label_col=label_col
    )
    record["valid_sequence_label_rows"] = len(sequences)
    record["label_values"] = sorted({int(label) for label in labels})
    order_digest = hashlib.sha256()
    for sequence, label in zip(sequences, labels):
        order_digest.update(str(sequence).encode("utf-8"))
        order_digest.update(b"\0")
        order_digest.update(str(int(label)).encode("ascii"))
        order_digest.update(b"\n")
    record["sequence_label_order_sha256"] = order_digest.hexdigest()
    return record


def build_run_manifest(
    args,
    model_paths: list[str],
    dataset_files: list[str],
    result_path: Path,
) -> dict[str, object]:
    """Build a machine-readable record of the resolved probe protocol."""
    dataset_path_text = " ".join(dataset_files).lower()
    if "nt_benchmark" in dataset_path_text or "nt_formatted" in dataset_path_text:
        benchmark_name = "NT_Pareto"
    elif "genomicbenchmark" in dataset_path_text:
        benchmark_name = "GenomicBenchmark"
    else:
        benchmark_name = "UNSPECIFIED"
    dataset_records = [
        describe_dataset_file(
            path,
            sep=args.sep,
            seq_col=args.seq_col,
            label_col=args.label_col,
        )
        for path in dataset_files
    ]
    if len(dataset_records) == 2:
        split = {
            "mode": "predefined_train_test_files",
            "train_file_index_range": "[0, n_train)",
            "validation_file_index_range": "[0, n_validation)",
            "train_ratio": None,
            "order_preserved": True,
        }
    elif len(dataset_records) == 1:
        split = {
            "mode": "single_file_sequential_80_20_fallback",
            "train_file_index_range": "[0, floor(0.8*n))",
            "validation_file_index_range": "[floor(0.8*n), n)",
            "train_ratio": float(args.train_ratio),
            "order_preserved": True,
        }
    else:
        split = {"mode": "INVALID_DATASET_FILE_COUNT", "order_preserved": False}

    if len(dataset_records) == 2:
        split["train_rows"] = dataset_records[0].get("valid_sequence_label_rows")
        split["validation_rows"] = dataset_records[1].get("valid_sequence_label_rows")
        split["train_order_sha256"] = dataset_records[0].get("sequence_label_order_sha256")
        split["validation_order_sha256"] = dataset_records[1].get("sequence_label_order_sha256")
    elif len(dataset_records) == 1:
        total_rows = dataset_records[0].get("valid_sequence_label_rows")
        if isinstance(total_rows, int):
            split_index = int(float(args.train_ratio) * total_rows)
            split["total_rows"] = total_rows
            split["train_rows"] = split_index
            split["validation_rows"] = total_rows - split_index
        split["single_file_order_sha256"] = dataset_records[0].get("sequence_label_order_sha256")

    model_records = []
    for path in model_paths:
        model_path = Path(path).expanduser().resolve()
        model_record: dict[str, object] = {
            "path": str(model_path),
            "exists": model_path.exists(),
            "status": "AVAILABLE" if model_path.exists() else "NOT_AVAILABLE",
        }
        if model_path.exists():
            model_record["filesystem_fingerprint"] = filesystem_fingerprint(str(model_path))
        model_records.append(model_record)

    result_path = result_path.resolve()
    manifest = {
        "schema_version": "probe-run-manifest-v1",
        "run_status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark_name,
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "cli_args": dict(vars(args)),
        "model_paths": model_records,
        "dataset_files": dataset_records,
        "split": split,
        "protocol": {
            "probe": "unprojected_linear_classifier" if args.normalize_dim is None else "projected_classifier",
            "optimizer": "AdamW",
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "precompute_batch_size": int(args.precompute_batch_size),
            "num_workers": int(args.num_workers),
            "max_length": int(args.max_length),
            "pooling": args.pooling,
            "bootstrap_replicates": int(args.bootstrap),
            "seed": 40,
            "normalize_dim": args.normalize_dim,
            "use_nonlinearity": bool(args.use_nonlinearity),
        },
        "code": {
            "entrypoint": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
        },
        "result_json": str(result_path),
    }
    return manifest


def write_json_atomic(path: str | Path, payload: dict[str, object]) -> None:
    """Atomically publish a JSON manifest beside the result artifact."""
    target = Path(path)
    partial = target.with_suffix(target.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    partial.replace(target)


def result_filename(args) -> str:
    filename = f"results_{args.model_name}_{args.dataset_name}"
    if args.multi_dim_eval:
        filename += "_multi_dim"
    elif args.normalize_dim is not None:
        filename += f"_dim{args.normalize_dim}"
    return filename + ".json"


def build_cache_metadata(model_path: str, dataset_files: List[str], dataset_name: str, args) -> dict[str, str | int]:
    files = []
    for value in dataset_files:
        stat = os.stat(value)
        files.append({
            "path": os.path.abspath(value),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    payload = {
        "schema": "embedding-cache-v2",
        "model_path": os.path.abspath(model_path),
        "model_fingerprint": filesystem_fingerprint(model_path),
        "dataset_name": dataset_name,
        "dataset_files": json.dumps(files, sort_keys=True),
        "max_length": int(args.max_length),
        "pooling": args.pooling,
        "train_ratio": float(args.train_ratio),
        "model_name": args.model_name,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


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
    cache_metadata: Optional[Dict[str, str | int]] = None,
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
            if cache_metadata:
                for key, value in cache_metadata.items():
                    grp.attrs[key] = value

    embs_np = embeddings.to(torch.float32).numpy()
    lbls_np = labels_t.numpy()

    if lock_file:
        lock = FileLock(lock_file, timeout=300)
        with lock:
            _write_hdf5(hdf5_path, split_name, embs_np, lbls_np)
    else:
        _write_hdf5(hdf5_path, split_name, embs_np, lbls_np)


def save_embedding_as_pickle(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    split_name: str,
    batch_index: int,
):
    """
    保存单个 batch 的 embedding 和 label 到 pickle 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split_name}_batch_{batch_index}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)
    print(f"Saved batch {batch_index} to {output_file}")


def precompute_and_save_embeddings(
    extractor,
    sequences: List[str],
    labels: List[int],
    output_dir: str,  # Directory for saved pickle files.
    split_name: str,
    batch_size: int = 64,
    num_workers: int = 4,
    max_length: int = 512,
    pooling: str = "mean",
):
    """
    计算 sequences 的 embedding 并将结果逐批保存为 pickle 文件。
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
    batch_index = 0
    with torch.inference_mode():
        for batch in tqdm.tqdm(loader, desc=f"Precompute {split_name} emb (pickle)"):
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
            emb_cpu = emb.detach().cpu()
            labels_batch = batch.get("labels")
            save_embedding_as_pickle(
                emb_cpu, labels_batch, output_dir, split_name, batch_index
            )
            batch_index += 1


def load_embeddings_from_hdf5(
    hdf5_path: str,
    split_name: str,
    expected_fingerprint: Optional[str] = None,
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
            if expected_fingerprint is not None and grp.attrs.get("fingerprint") != expected_fingerprint:
                return None
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
def train_probe_on_embeddings(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    writer: Optional[SummaryWriter] = None,
):
    if device is None:
        device = next(probe.parameters()).device
    probe.to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        probe.train()
        losses = []
        preds_all = []
        labels_all = []
        pbar = tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch in pbar:
            emb = batch["input_emb"].to(device)
            labels = batch["labels"].to(device)

            logits = probe(emb)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            batch_preds = torch.argmax(logits.detach().cpu(), dim=1).numpy().tolist()
            preds_all.extend(batch_preds)
            labels_all.extend(labels.detach().cpu().numpy().tolist())

            pbar.set_postfix({"loss": np.mean(losses)})
            writer_step = (epoch - 1) * len(train_loader) + pbar.n
            if writer is not None:
                writer.add_scalar("train/loss", np.mean(losses), writer_step)

        train_acc = accuracy_score(labels_all, preds_all)
        train_f1 = f1_score(labels_all, preds_all, average="macro")
        print(
            f"Epoch {epoch} train_loss={np.mean(losses):.4f} acc={train_acc:.4f} macro_f1={train_f1:.4f}"
        )
        writer_step = epoch * len(train_loader)
        if writer is not None:
            writer.add_scalar("train/epoch_loss", np.mean(losses), writer_step)
            writer.add_scalar("train/epoch_acc", train_acc, writer_step)
            writer.add_scalar("train/epoch_macro_f1", train_f1, writer_step)

        if val_loader is not None:
            val_metrics = eval_probe_on_embeddings(probe, val_loader, device=device)
            extra = f" val_auc={val_metrics['auc']:.4f}" if "auc" in val_metrics else ""
            print(
                f"Epoch {epoch} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} val_mcc={val_metrics['mcc']:.4f}"
                + extra
            )
            if writer is not None:
                writer.add_scalar("val/epoch_acc", val_metrics["accuracy"], writer_step)
                writer.add_scalar(
                    "val/epoch_macro_f1", val_metrics["macro_f1"], writer_step
                )
                writer.add_scalar("val/epoch_mcc", val_metrics["mcc"], writer_step)
                if "auc" in val_metrics:
                    writer.add_scalar("val/epoch_auc", val_metrics["auc"], writer_step)

    return probe


def eval_probe_on_embeddings(
    probe: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    return_samples: bool = False,
):
    probe.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device)

    preds_all = []
    labels_all = []
    probs_all = []

    with torch.inference_mode():
        for batch in tqdm.tqdm(data_loader, desc="Eval", leave=False):
            emb = batch["input_emb"].to(device)
            labels = batch["labels"].to(device)

            logits = probe(emb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            preds_all.extend(preds.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            probs_all.extend(probs.detach().cpu().numpy().tolist())

    # Basic metrics.
    acc = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    mcc = matthews_corrcoef(labels_all, preds_all)
    metrics = {"accuracy": acc, "macro_f1": macro_f1, "mcc": mcc}

    # Calculate AUC for binary and multiclass classification.
    if len(probs_all) > 0:
        probs_arr = np.array(probs_all)  # shape (N, C)
        labels_arr = np.array(labels_all)
        num_classes = probs_arr.shape[1]

        # Preserve the point-estimate F1 for every classifier class.  The
        # bootstrap summary below intentionally remains aggregate-only, but
        # the active multiclass probe result contract requires traceable
        # per-class F1 values.
        metrics["per_class_f1"] = [
            float(x)
            for x in f1_score(
                labels_arr,
                np.asarray(preds_all),
                labels=np.arange(num_classes),
                average=None,
                zero_division=0,
            )
        ]

        if num_classes == 2:
            # Binary classification: preserve the existing behavior by using the positive-class probability.
            try:
                probs_pos = probs_arr[:, 1]
                auc = roc_auc_score(labels_arr, probs_pos)
                metrics["auc"] = float(auc)
            except Exception:
                metrics["auc"] = float("nan")
        else:
            # Multiclass: calculate per-class, macro (ignoring NaN), and micro AUC.
            try:
                # Binarize labels to shape (N, C).
                classes = np.arange(num_classes)
                labels_bin = label_binarize(labels_arr, classes=classes)
            except Exception:
                # Use NaN for unexpected errors.
                metrics["auc_macro"] = float("nan")
                metrics["auc_micro"] = float("nan")
                metrics["auc_per_class"] = [float("nan")] * num_classes
            else:
                per_class = []
                for i in range(num_classes):
                    y_true = labels_bin[:, i]
                    if np.unique(y_true).size < 2:
                        # AUC is undefined when this class has no positive or no negative samples.
                        per_class.append(float("nan"))
                    else:
                        try:
                            a = roc_auc_score(y_true, probs_arr[:, i])
                            per_class.append(float(a))
                        except Exception:
                            per_class.append(float("nan"))

                # Macro: compute nanmean over per-class values, ignoring classes that cannot be evaluated.
                try:
                    if np.all(np.isnan(per_class)):
                        auc_macro = float("nan")
                    else:
                        auc_macro = float(np.nanmean(per_class))
                    # Micro: try the sklearn micro calculation; use NaN if it fails.
                    try:
                        auc_micro = float(
                            roc_auc_score(
                                labels_bin,
                                probs_arr,
                                average="micro",
                                multi_class="ovr",
                            )
                        )
                    except Exception:
                        auc_micro = float("nan")

                    metrics["auc_macro"] = auc_macro
                    metrics["auc_micro"] = auc_micro
                    metrics["auc_per_class"] = per_class
                except Exception:
                    metrics["auc_macro"] = float("nan")
                    metrics["auc_micro"] = float("nan")
                    metrics["auc_per_class"] = [float("nan")] * num_classes
    else:
        # No probability data is available.
        metrics["auc"] = float("nan")

    if return_samples:
        return metrics, labels_all, preds_all, probs_all
    return metrics


# ---------- Bootstrap confidence intervals (backward compatible) ----------
def bootstrap_ci(
    labels: List[int],
    preds: List[int],
    probs: List[List[float]],
    n_bootstrap: int = 1000,
    random_state: int = 0,
) -> Dict[str, Tuple[float, float, float]]:
    """
    通过 bootstrap 估计各指标的均值和 95% CI。
    - 二分类：返回 accuracy, mcc, auc（二分类 AUC）
    - 多分类：返回 accuracy, mcc, auc_macro, auc_micro
      同时为兼容旧代码，将 auc_macro 也赋值到 res["auc"]。
    """
    rng = np.random.RandomState(random_state)
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    n = len(labels)

    accs = []
    mccs = []

    # Determine the number of classes.
    if probs.ndim == 1:
        # Treat a single probability column as the positive-class probability for binary classification.
        num_classes = 2
    else:
        num_classes = probs.shape[1]

    # Prepare containers for AUC calculations.
    if num_classes == 2:
        aucs = []  # Binary AUC.
    else:
        auc_macros = []  # Multiclass macro AUC.
        auc_micros = []  # Multiclass micro AUC.
        # Binarize labels once so bootstrap iterations only need to select rows.
        classes = np.arange(num_classes)
        labels_bin_full = label_binarize(labels, classes=classes)

    for i in range(n_bootstrap):
        idxs = rng.randint(0, n, size=n)
        lab_b = labels[idxs]
        pred_b = preds[idxs]
        try:
            accs.append(accuracy_score(lab_b, pred_b))
        except Exception:
            accs.append(float("nan"))
        try:
            mccs.append(matthews_corrcoef(lab_b, pred_b))
        except Exception:
            mccs.append(float("nan"))
        try:
            if num_classes == 2:
                # Binary classification: use the positive-class probability.
                if probs.ndim == 1:
                    probs_pos = probs[idxs]
                else:
                    probs_pos = probs[idxs, 1]
                aucs.append(roc_auc_score(lab_b, probs_pos))
            else:
                # Multiclass: calculate macro and micro AUC.
                probs_b = probs[idxs]
                labels_bin_b = labels_bin_full[idxs]
                # macro
                try:
                    auc_macro = roc_auc_score(
                        labels_bin_b,
                        probs_b,
                        average="macro",
                        multi_class="ovr",
                    )
                except Exception:
                    auc_macro = float("nan")
                # micro
                try:
                    auc_micro = roc_auc_score(
                        labels_bin_b,
                        probs_b,
                        average="micro",
                        multi_class="ovr",
                    )
                except Exception:
                    auc_micro = float("nan")

                auc_macros.append(auc_macro)
                auc_micros.append(auc_micro)
        except Exception:
            # Record any remaining exception as NaN; specific failure cases are handled above.
            if num_classes == 2:
                aucs.append(float("nan"))
            else:
                auc_macros.append(float("nan"))
                auc_micros.append(float("nan"))

    def summarize(arr):
        arr = np.array(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        mean = float(np.mean(arr))
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        return (mean, lo, hi)

    res: Dict[str, Tuple[float, float, float]] = {}
    res["accuracy"] = summarize(accs)
    res["mcc"] = summarize(mccs)

    if num_classes == 2:
        res["auc"] = summarize(aucs)
    else:
        res["auc_macro"] = summarize(auc_macros)
        res["auc_micro"] = summarize(auc_micros)
        # For compatibility with code that reads res["auc"], set auc to the macro AUC.
        res["auc"] = res["auc_macro"]

    return res


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
    cache_meta = build_cache_metadata(model_path, dataset_files, dataset_name, args)

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
        loaded = load_embeddings_from_hdf5(hdf5_path, split_name, cache_meta["fingerprint"])

        if (loaded is None) or args.recompute:
            print(f"Precomputing embeddings for {split_name} (n={len(seqs)}) ...")
            precompute_and_cache_embeddings_hdf5(
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
                cache_metadata=cache_meta,
            )
        else:
            print(f"Found cached embeddings in HDF5 for {split_name}, loading...")

    train_loaded = load_embeddings_from_hdf5(hdf5_path, "train", cache_meta["fingerprint"])
    val_loaded = load_embeddings_from_hdf5(hdf5_path, "val", cache_meta["fingerprint"])

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
    print(f"[INFO] Dataset has `{num_classes}` classes.")

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
            metrics = {}
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
        if "per_class_f1" in metrics:
            all_results[dim_name]["per_class_f1"] = metrics["per_class_f1"]

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
    parser.add_argument("--lr", type=float, default=1e-3)
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
    # Add options to save embeddings as pickle files and specify their directory.
    parser.add_argument(
        "--save_pickle",
        action="store_true",
        help="是否将 embedding 保存为 pickle 文件而非直接写入 HDF5",
    )
    parser.add_argument(
        "--pickle_dir", type=str, default="./emb_pickle", help="保存 pickle 文件的目录"
    )
    # ========== Projection dimension options ========== #
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

    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.del_old_runs and os.path.exists(args.log_dir):
        import shutil

        shutil.rmtree(args.log_dir, ignore_errors=True)

    model_paths = [p.strip() for p in args.model_paths.split(",") if p.strip()]
    dataset_files = [p.strip() for p in args.dataset_files.split(",") if p.strip()]

    result_path = Path(args.out_dir).resolve() / result_filename(args)
    run_manifest_path = Path(args.out_dir).resolve() / "run_manifest.json"
    run_manifest = build_run_manifest(args, model_paths, dataset_files, result_path)
    write_json_atomic(run_manifest_path, run_manifest)
    print(f"Run manifest initialized at: {run_manifest_path}")

    writer = SummaryWriter(log_dir=args.log_dir)

    results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    try:
        for mpath in model_paths:
            model_name = os.path.basename(mpath.rstrip("/"))
            results[model_name] = {}
            out = run_one_model_on_dataset(
                mpath, dataset_files, args.dataset_name, args, writer=writer
            )
            results[model_name].update(out)
    except Exception as exc:
        run_manifest["run_status"] = "FAILED"
        run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json_atomic(run_manifest_path, run_manifest)
        writer.close()
        raise

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

    output_filename = result_filename(args)

    with open(
        os.path.join(args.out_dir, output_filename),
        "w",
        encoding="utf-8",
    ) as f:
        _json.dump(results, f, indent=2, ensure_ascii=False)

    run_manifest["run_status"] = "COMPLETED"
    run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["result_sha256"] = sha256_file(result_path)
    write_json_atomic(run_manifest_path, run_manifest)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {os.path.join(args.out_dir, output_filename)}")
    print(f"{'=' * 60}\n")

    writer.close()
    print(
        f"All done. Results saved to {args.out_dir}, tensorboard logs in {args.log_dir}"
    )


if __name__ == "__main__":
    main()
