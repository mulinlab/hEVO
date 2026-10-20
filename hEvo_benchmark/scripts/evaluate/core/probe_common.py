"""Shared probe operations; callers explicitly supply protocol-specific callbacks.

Standard and Pareto entry points retain their dataset/collation/evaluation
policies. Seeds, optimizers, bootstrap draws and output fields are unchanged.
"""

from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class EmbeddingDataset(torch.utils.data.Dataset):
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


class FlexibleProbe(nn.Module):
    """Supports the historical linear and normalized projection modes."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        normalize_dim: Optional[int] = None,
        use_nonlinearity: bool = False,
    ):
        super().__init__()
        if normalize_dim is None:
            self.probe = nn.Linear(input_dim, num_classes)
        else:
            layers = [nn.Linear(input_dim, normalize_dim)]
            if use_nonlinearity:
                layers.append(nn.ReLU())
            layers.append(nn.Linear(normalize_dim, num_classes))
            self.probe = nn.Sequential(*layers)

    def forward(self, x):
        return self.probe(x)


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
    *,
    dataset_type,
    collate_batch,
):
    """
    计算 sequences 的 embedding 并将结果逐批保存为 pickle 文件。
    """
    ds = dataset_type(
        sequences=sequences,
        labels=labels,
        tokenizer_callable=extractor.tokenize,
        max_length=max_length,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
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


def eval_probe_on_embeddings(
    probe: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    return_samples: bool = False,
    include_per_class_f1: bool = False,
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
        if include_per_class_f1:
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


def train_probe_on_embeddings(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    writer: Optional[SummaryWriter] = None,
    *,
    evaluate,
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
            val_metrics = evaluate(probe, val_loader, device=device)
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
