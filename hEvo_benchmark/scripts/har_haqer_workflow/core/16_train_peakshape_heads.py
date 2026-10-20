from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train peak-shape scoring heads that explicitly encourage center windows to score higher "
            "than nearby flanks and background windows."
        )
    )
    parser.add_argument(
        "--embeddings-root", type=Path, default=Path("output/peakshape_embeddings")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/peakshape_heads")
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["har", "haqer"], choices=["har", "haqer"]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Number of groups per batch."
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--head-type", choices=["linear", "mlp1", "mlp2", "cnn1d"], default="linear"
    )
    parser.add_argument(
        "--feature-dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--margin-center-flank", type=float, default=0.5)
    parser.add_argument("--margin-center-background", type=float, default=0.5)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-center-flank", type=float, default=1.0)
    parser.add_argument("--lambda-center-background", type=float, default=0.5)
    parser.add_argument(
        "--flank-weighting",
        choices=["none", "inverse_distance"],
        default="inverse_distance",
        help="How to weight flank ranking loss by flank distance.",
    )
    parser.add_argument(
        "--flank-weight-reference-bp",
        type=float,
        default=512.0,
        help="Reference bp used by inverse_distance weighting. Weight = reference / |offset_bp|.",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=7,
        help="Kernel size for cnn1d scoring head.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["center_top1_rate", "window_auprc", "window_auroc", "composite_hit"],
        default="center_top1_rate",
        help=(
            "Validation metric used to select the best checkpoint. "
            "composite_hit favors region retrieval more than pure ranking."
        ),
    )
    parser.add_argument(
        "--posthoc-negative-calibration",
        choices=["none", "quantile"],
        default="none",
        help=(
            "Optional posthoc logit-shift calibration computed on validation negatives. "
            "This preserves ranking but lowers absolute off-target probabilities."
        ),
    )
    parser.add_argument(
        "--posthoc-negative-quantile",
        type=float,
        default=0.95,
        help="Negative-logit quantile used when posthoc-negative-calibration=quantile.",
    )
    parser.add_argument(
        "--posthoc-negative-target-prob",
        type=float,
        default=0.35,
        help="Target probability assigned to the chosen negative-logit quantile after calibration.",
    )
    parser.add_argument(
        "--embedding-store-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Where to keep normalized embedding matrices after loading. "
            "cuda is much faster for small/medium peak-shape datasets."
        ),
    )
    parser.add_argument(
        "--embedding-store-max-gb",
        type=float,
        default=8.0,
        help="Maximum combined embedding size to auto-cache on GPU when embedding-store-device=auto.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    probs = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    negative = ~positive
    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[negative])
    probs[negative] = exp_logits / (1.0 + exp_logits)
    return probs.astype(np.float32, copy=False)


def stable_logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return float(np.log(clipped / (1.0 - clipped)))


def compute_feature_stats(
    features_path: Path, chunk_size: int = 65536
) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(features_path, mmap_mode="r")
    n_samples, feature_dim = features.shape
    running_sum = np.zeros(feature_dim, dtype=np.float64)
    running_sq_sum = np.zeros(feature_dim, dtype=np.float64)
    for start in tqdm(
        range(0, n_samples, chunk_size),
        desc=f"Stats {features_path.parent.name}",
        dynamic_ncols=True,
    ):
        end = min(start + chunk_size, n_samples)
        batch = np.asarray(features[start:end], dtype=np.float32)
        running_sum += batch.sum(axis=0, dtype=np.float64)
        running_sq_sum += np.square(batch, dtype=np.float64).sum(
            axis=0, dtype=np.float64
        )
    mean = running_sum / max(n_samples, 1)
    variance = running_sq_sum / max(n_samples, 1) - np.square(mean)
    variance = np.clip(variance, a_min=1e-6, a_max=None)
    std = np.sqrt(variance, dtype=np.float64)
    return mean.astype(np.float32), std.astype(np.float32)


class MLPScoringHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        head_type: str,
        conv_kernel_size: int = 7,
    ):
        super().__init__()
        if head_type == "linear":
            self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))
        elif head_type == "mlp1":
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        elif head_type == "mlp2":
            bottleneck_dim = max(hidden_dim // 4, 32)
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )
        elif head_type == "cnn1d":
            if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
                raise ValueError(
                    "conv_kernel_size must be a positive odd integer for cnn1d."
                )
            conv_hidden_dim = max(hidden_dim // 2, 32)
            padding = conv_kernel_size // 2
            self.norm = nn.LayerNorm(input_dim)
            self.net = nn.Sequential(
                nn.Conv1d(
                    1, conv_hidden_dim, kernel_size=conv_kernel_size, padding=padding
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    conv_hidden_dim,
                    hidden_dim,
                    kernel_size=conv_kernel_size,
                    padding=padding,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool1d(1),
            )
            self.out = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.head_type = head_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "cnn1d":
            x = self.norm(x).unsqueeze(1)
            x = self.net(x).squeeze(-1)
            return self.out(x).squeeze(-1)
        return self.net(x).squeeze(-1)


@dataclass
class SplitBundle:
    name: str
    embeddings_path: Path
    labels_path: Path
    role_path: Path
    group_id_path: Path
    offset_bp_path: Path
    chromosome_path: Path
    start_path: Path


def resolve_bundle(root: Path, task: str, split: str) -> SplitBundle:
    upper = task.upper()
    stem = f"14_{upper}_peakshape_{split}_512.lmdb"
    split_dir = root / f"{task}_peakshape_{split}"
    return SplitBundle(
        name=f"{task}_peakshape_{split}",
        embeddings_path=split_dir / f"{stem}.embeddings.npy",
        labels_path=split_dir / f"{stem}.labels.npy",
        role_path=split_dir / f"{stem}.role.npy",
        group_id_path=split_dir / f"{stem}.group_id.npy",
        offset_bp_path=split_dir / f"{stem}.offset_bp.npy",
        chromosome_path=split_dir / f"{stem}.chromosome.npy",
        start_path=split_dir / f"{stem}.start.npy",
    )


@dataclass
class GroupRecord:
    group_id: str
    center_index: int
    flank_indices: np.ndarray
    flank_offsets_bp: np.ndarray
    background_indices: np.ndarray
    chromosome: str
    start: int


class PeakShapeArrayStore:
    def __init__(
        self,
        bundle: SplitBundle,
        mean: np.ndarray,
        std: np.ndarray,
        feature_dtype: str,
        embedding_device: torch.device | None = None,
    ):
        np_dtype = np.float16 if feature_dtype == "float16" else np.float32
        embeddings = np.load(bundle.embeddings_path).astype(np_dtype, copy=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = (embeddings - mean) / std
        self.embeddings = torch.from_numpy(np.ascontiguousarray(embeddings))
        if embedding_device is not None:
            self.embeddings = self.embeddings.to(embedding_device, non_blocking=False)
        self.labels = np.load(bundle.labels_path).astype(np.int64, copy=False)
        self.roles = np.load(bundle.role_path, allow_pickle=True)
        self.group_ids = np.load(bundle.group_id_path, allow_pickle=True)
        self.offset_bp = np.load(bundle.offset_bp_path).astype(np.int64, copy=False)
        self.chromosomes = np.load(bundle.chromosome_path, allow_pickle=True)
        self.starts = np.load(bundle.start_path).astype(np.int64, copy=False)
        self.groups = self._build_groups()

    def _build_groups(self) -> list[GroupRecord]:
        grouped: dict[str, dict[str, list[int] | int | str]] = {}
        for idx, (group_id, role) in enumerate(
            zip(self.group_ids.tolist(), self.roles.tolist())
        ):
            if group_id is None or role is None:
                continue
            entry = grouped.setdefault(
                str(group_id),
                {
                    "center": None,
                    "flank": [],
                    "flank_offsets_bp": [],
                    "background": [],
                    "chromosome": str(self.chromosomes[idx]),
                    "start": int(self.starts[idx]),
                },
            )
            if role == "center":
                entry["center"] = idx
                entry["chromosome"] = str(self.chromosomes[idx])
                entry["start"] = int(self.starts[idx])
            elif role == "flank":
                entry["flank"].append(idx)
                entry["flank_offsets_bp"].append(int(self.offset_bp[idx]))
            elif role == "background":
                entry["background"].append(idx)

        groups: list[GroupRecord] = []
        for group_id, entry in grouped.items():
            center_index = entry["center"]
            flank_indices = np.asarray(entry["flank"], dtype=np.int64)
            flank_offsets_bp = np.asarray(entry["flank_offsets_bp"], dtype=np.int64)
            background_indices = np.asarray(entry["background"], dtype=np.int64)
            if (
                center_index is None
                or flank_indices.size == 0
                or background_indices.size == 0
            ):
                continue
            groups.append(
                GroupRecord(
                    group_id=group_id,
                    center_index=int(center_index),
                    flank_indices=flank_indices,
                    flank_offsets_bp=flank_offsets_bp,
                    background_indices=background_indices,
                    chromosome=str(entry["chromosome"]),
                    start=int(entry["start"]),
                )
            )
        groups.sort(
            key=lambda record: (record.chromosome, record.start, record.group_id)
        )
        return groups


class PeakShapeGroupDataset(Dataset):
    def __init__(self, store: PeakShapeArrayStore):
        self.store = store
        self.groups = store.groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> GroupRecord:
        return self.groups[index]


def collate_groups(
    records: list[GroupRecord], store: PeakShapeArrayStore
) -> dict[str, torch.Tensor | list[str]]:
    index_device = store.embeddings.device
    center_indices = torch.tensor(
        [record.center_index for record in records],
        dtype=torch.long,
        device=index_device,
    )
    center_embeddings = store.embeddings[center_indices]

    flank_group_index: list[int] = []
    flank_embedding_list: list[torch.Tensor] = []
    flank_offset_list: list[torch.Tensor] = []
    background_group_index: list[int] = []
    background_embedding_list: list[torch.Tensor] = []

    for group_idx, record in enumerate(records):
        if record.flank_indices.size > 0:
            flank_group_index.extend([group_idx] * int(record.flank_indices.size))
            flank_indices = torch.as_tensor(
                record.flank_indices, dtype=torch.long, device=index_device
            )
            flank_embedding_list.append(store.embeddings[flank_indices])
            flank_offset_list.append(
                torch.as_tensor(
                    record.flank_offsets_bp.astype(np.int64, copy=False),
                    dtype=torch.long,
                    device=index_device,
                )
            )
        if record.background_indices.size > 0:
            background_group_index.extend(
                [group_idx] * int(record.background_indices.size)
            )
            background_indices = torch.as_tensor(
                record.background_indices, dtype=torch.long, device=index_device
            )
            background_embedding_list.append(store.embeddings[background_indices])

    flank_embeddings = (
        torch.cat(flank_embedding_list, dim=0)
        if flank_embedding_list
        else torch.empty(
            (0, store.embeddings.shape[1]),
            device=index_device,
            dtype=store.embeddings.dtype,
        )
    )
    background_embeddings = (
        torch.cat(background_embedding_list, dim=0)
        if background_embedding_list
        else torch.empty(
            (0, store.embeddings.shape[1]),
            device=index_device,
            dtype=store.embeddings.dtype,
        )
    )
    flank_offsets_bp = (
        torch.cat(flank_offset_list, dim=0)
        if flank_offset_list
        else torch.empty((0,), dtype=torch.long, device=index_device)
    )

    return {
        "group_id": [record.group_id for record in records],
        "center_embeddings": center_embeddings,
        "flank_embeddings": flank_embeddings,
        "background_embeddings": background_embeddings,
        "flank_group_index": torch.tensor(
            flank_group_index, dtype=torch.long, device=index_device
        ),
        "flank_offsets_bp": flank_offsets_bp,
        "background_group_index": torch.tensor(
            background_group_index, dtype=torch.long, device=index_device
        ),
    }


def move_batch(
    batch: dict[str, torch.Tensor | list[str]], device: torch.device
) -> dict[str, torch.Tensor | list[str]]:
    moved: dict[str, torch.Tensor | list[str]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def compute_group_logits(
    model: nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
) -> dict[str, torch.Tensor]:
    center_logits = model(batch["center_embeddings"])  # type: ignore[arg-type]
    flank_logits = (
        model(batch["flank_embeddings"])
        if batch["flank_embeddings"].shape[0] > 0
        else torch.empty(  # type: ignore[arg-type]
            (0,), device=center_logits.device
        )
    )
    background_logits = (
        model(batch["background_embeddings"])
        if batch["background_embeddings"].shape[0] > 0
        else torch.empty((0,), device=center_logits.device)  # type: ignore[arg-type]
    )
    return {
        "center_logits": center_logits,
        "flank_logits": flank_logits,
        "background_logits": background_logits,
    }


def load_head_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MLPScoringHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        head_type=str(checkpoint["head_type"]),
        conv_kernel_size=int(checkpoint.get("conv_kernel_size", 7)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def compute_peakshape_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
    pos_weight: float,
    margin_center_flank: float,
    margin_center_background: float,
    lambda_bce: float,
    lambda_center_flank: float,
    lambda_center_background: float,
    flank_weighting: str,
    flank_weight_reference_bp: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    center_logits = logits["center_logits"]
    flank_logits = logits["flank_logits"]
    background_logits = logits["background_logits"]
    device = center_logits.device

    pos_targets = torch.ones_like(center_logits)
    neg_flank_targets = torch.zeros_like(flank_logits)
    neg_background_targets = torch.zeros_like(background_logits)

    pos_weight_tensor = torch.tensor(
        pos_weight, dtype=center_logits.dtype, device=device
    )
    bce = nn.functional.binary_cross_entropy_with_logits(
        center_logits, pos_targets, pos_weight=pos_weight_tensor
    )
    if flank_logits.numel() > 0:
        bce = bce + nn.functional.binary_cross_entropy_with_logits(
            flank_logits, neg_flank_targets
        )
    if background_logits.numel() > 0:
        bce = bce + nn.functional.binary_cross_entropy_with_logits(
            background_logits, neg_background_targets
        )

    cf_loss = torch.tensor(0.0, device=device)
    if flank_logits.numel() > 0:
        center_for_flanks = center_logits[batch["flank_group_index"]]  # type: ignore[index]
        raw_cf = torch.relu(margin_center_flank - (center_for_flanks - flank_logits))
        if flank_weighting == "inverse_distance":
            flank_offsets_bp = batch["flank_offsets_bp"].to(
                device=device, dtype=center_logits.dtype
            )  # type: ignore[union-attr]
            abs_offsets_bp = torch.clamp(torch.abs(flank_offsets_bp), min=1.0)
            flank_weights = torch.clamp(
                float(flank_weight_reference_bp) / abs_offsets_bp,
                min=0.0,
                max=1.0,
            )
            cf_loss = (raw_cf * flank_weights).sum() / torch.clamp(
                flank_weights.sum(), min=1e-9
            )
        else:
            cf_loss = raw_cf.mean()

    cb_loss = torch.tensor(0.0, device=device)
    if background_logits.numel() > 0:
        center_for_background = center_logits[batch["background_group_index"]]  # type: ignore[index]
        cb_loss = torch.relu(
            margin_center_background - (center_for_background - background_logits)
        ).mean()

    total = (
        lambda_bce * bce
        + lambda_center_flank * cf_loss
        + lambda_center_background * cb_loss
    )
    return total, {
        "bce": float(bce.detach().cpu()),
        "center_flank": float(cf_loss.detach().cpu()),
        "center_background": float(cb_loss.detach().cpu()),
    }


@torch.inference_mode()
def evaluate_split(
    model: nn.Module,
    store: PeakShapeArrayStore,
    dataset: PeakShapeGroupDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda records: collate_groups(records, store),
        drop_last=False,
    )

    all_center_logits: list[np.ndarray] = []
    all_window_logits: list[np.ndarray] = []
    all_window_labels: list[np.ndarray] = []
    group_top1: list[float] = []
    group_center_flank_margin: list[float] = []
    group_center_background_margin: list[float] = []

    for batch in tqdm(
        loader,
        desc=f"Eval {dataset.store.groups[0].group_id.split('_')[0] if dataset.store.groups else 'split'}",
        leave=False,
        dynamic_ncols=True,
    ):
        batch = move_batch(batch, device)
        logits = compute_group_logits(model, batch)
        center_logits = logits["center_logits"]
        flank_logits = logits["flank_logits"]
        background_logits = logits["background_logits"]

        all_center_logits.append(center_logits.detach().cpu().numpy())
        if center_logits.numel() > 0:
            all_window_logits.append(center_logits.detach().cpu().numpy())
            all_window_labels.append(np.ones(center_logits.shape[0], dtype=np.int64))
        if flank_logits.numel() > 0:
            all_window_logits.append(flank_logits.detach().cpu().numpy())
            all_window_labels.append(np.zeros(flank_logits.shape[0], dtype=np.int64))
        if background_logits.numel() > 0:
            all_window_logits.append(background_logits.detach().cpu().numpy())
            all_window_labels.append(
                np.zeros(background_logits.shape[0], dtype=np.int64)
            )

        center_np = center_logits.detach().cpu().numpy()
        flank_np = flank_logits.detach().cpu().numpy()
        background_np = background_logits.detach().cpu().numpy()
        flank_group_index = (
            batch["flank_group_index"].detach().cpu().numpy()
            if flank_logits.numel() > 0
            else np.empty((0,), dtype=np.int64)
        )  # type: ignore[union-attr]
        background_group_index = (
            batch["background_group_index"].detach().cpu().numpy()
            if background_logits.numel() > 0
            else np.empty((0,), dtype=np.int64)
        )  # type: ignore[union-attr]

        for group_idx in range(center_np.shape[0]):
            flank_group_scores = flank_np[flank_group_index == group_idx]
            background_group_scores = background_np[background_group_index == group_idx]
            max_flank = (
                float(flank_group_scores.max())
                if flank_group_scores.size > 0
                else -float("inf")
            )
            max_background = (
                float(background_group_scores.max())
                if background_group_scores.size > 0
                else -float("inf")
            )
            hardest_negative = max(max_flank, max_background)
            group_top1.append(float(center_np[group_idx] > hardest_negative))
            group_center_flank_margin.append(
                float(center_np[group_idx] - max_flank)
                if np.isfinite(max_flank)
                else 0.0
            )
            group_center_background_margin.append(
                float(center_np[group_idx] - max_background)
                if np.isfinite(max_background)
                else 0.0
            )

    window_logits = (
        np.concatenate(all_window_logits, axis=0)
        if all_window_logits
        else np.empty((0,), dtype=np.float32)
    )
    window_labels = (
        np.concatenate(all_window_labels, axis=0)
        if all_window_labels
        else np.empty((0,), dtype=np.int64)
    )
    window_probabilities = stable_sigmoid(window_logits)

    metrics = {
        "window_auprc": float(
            average_precision_score(window_labels, window_probabilities)
        )
        if window_labels.size > 0
        else 0.0,
        "window_auroc": float(roc_auc_score(window_labels, window_probabilities))
        if window_labels.size > 0
        else 0.0,
        "center_top1_rate": float(np.mean(group_top1)) if group_top1 else 0.0,
        "mean_center_flank_margin": float(np.mean(group_center_flank_margin))
        if group_center_flank_margin
        else 0.0,
        "mean_center_background_margin": float(np.mean(group_center_background_margin))
        if group_center_background_margin
        else 0.0,
        "num_groups": len(group_top1),
    }
    return metrics


@torch.inference_mode()
def collect_split_logits(
    model: nn.Module,
    store: PeakShapeArrayStore,
    dataset: PeakShapeGroupDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda records: collate_groups(records, store),
        drop_last=False,
    )

    center_logits_list: list[np.ndarray] = []
    flank_logits_list: list[np.ndarray] = []
    background_logits_list: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Collect logits", leave=False, dynamic_ncols=True):
        batch = move_batch(batch, device)
        logits = compute_group_logits(model, batch)
        center_logits = logits["center_logits"]
        flank_logits = logits["flank_logits"]
        background_logits = logits["background_logits"]

        if center_logits.numel() > 0:
            center_logits_list.append(center_logits.detach().cpu().numpy())
        if flank_logits.numel() > 0:
            flank_logits_list.append(flank_logits.detach().cpu().numpy())
        if background_logits.numel() > 0:
            background_logits_list.append(background_logits.detach().cpu().numpy())

    return {
        "center_logits": np.concatenate(center_logits_list, axis=0)
        if center_logits_list
        else np.empty((0,), dtype=np.float32),
        "flank_logits": np.concatenate(flank_logits_list, axis=0)
        if flank_logits_list
        else np.empty((0,), dtype=np.float32),
        "background_logits": np.concatenate(background_logits_list, axis=0)
        if background_logits_list
        else np.empty((0,), dtype=np.float32),
    }


def build_negative_calibration(
    negative_logits: np.ndarray,
    negative_quantile: float,
    target_probability: float,
) -> dict[str, float] | None:
    if negative_logits.size == 0:
        return None

    quantile = float(np.clip(negative_quantile, 0.0, 1.0))
    observed_logit = float(
        np.quantile(negative_logits.astype(np.float64, copy=False), quantile)
    )
    observed_probability = float(
        stable_sigmoid(np.asarray([observed_logit], dtype=np.float64))[0]
    )
    target_logit = stable_logit(target_probability)
    logit_shift = float(target_logit - observed_logit)
    return {
        "mode": "quantile",
        "negative_quantile": quantile,
        "target_probability": float(np.clip(target_probability, 1e-6, 1.0 - 1e-6)),
        "observed_negative_logit": observed_logit,
        "observed_negative_probability": observed_probability,
        "inference_logit_shift": logit_shift,
        "calibrated_negative_probability": float(
            stable_sigmoid(
                np.asarray([observed_logit + logit_shift], dtype=np.float64)
            )[0]
        ),
    }


def compute_pos_weight(store: PeakShapeArrayStore) -> float:
    positives = float((store.labels == 1).sum())
    negatives = float((store.labels == 0).sum())
    return max(1.0, min(negatives / max(positives, 1.0), 32.0))


def selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "center_top1_rate":
        return float(metrics["center_top1_rate"])
    if selection_metric == "window_auprc":
        return float(metrics["window_auprc"])
    if selection_metric == "window_auroc":
        return float(metrics["window_auroc"])
    if selection_metric == "composite_hit":
        # Emphasize center-winning behavior first, but keep calibration/separation in the loop.
        return float(
            0.55 * metrics["center_top1_rate"]
            + 0.30 * metrics["window_auprc"]
            + 0.10 * metrics["window_auroc"]
            + 0.05 * max(metrics["mean_center_flank_margin"], 0.0)
        )
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def resolve_embedding_store_device(
    args: argparse.Namespace,
    target_device: torch.device,
    bundles: list[SplitBundle],
) -> torch.device | None:
    if args.embedding_store_device == "cpu":
        return None
    if args.embedding_store_device == "cuda":
        if target_device.type != "cuda":
            raise RuntimeError(
                "--embedding-store-device=cuda requested but training device is not CUDA."
            )
        return target_device
    if target_device.type != "cuda":
        return None

    total_bytes = 0
    for bundle in bundles:
        mmap = np.load(bundle.embeddings_path, mmap_mode="r")
        total_bytes += int(mmap.shape[0]) * int(mmap.shape[1]) * 4
        del mmap
    max_bytes = int(float(args.embedding_store_max_gb) * (1024**3))
    if total_bytes <= max_bytes:
        return target_device
    return None


def train_task(args: argparse.Namespace, task: str) -> None:
    task_output_dir = args.output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    train_bundle = resolve_bundle(args.embeddings_root, task, "train")
    val_bundle = resolve_bundle(args.embeddings_root, task, "val")
    test_bundle = resolve_bundle(args.embeddings_root, task, "test")

    mean, std = compute_feature_stats(train_bundle.embeddings_path)
    np.save(task_output_dir / "train_mean.npy", mean)
    np.save(task_output_dir / "train_std.npy", std)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    embedding_store_device = resolve_embedding_store_device(
        args, device, [train_bundle, val_bundle, test_bundle]
    )

    train_store = PeakShapeArrayStore(
        train_bundle, mean, std, args.feature_dtype, embedding_store_device
    )
    val_store = PeakShapeArrayStore(
        val_bundle, mean, std, args.feature_dtype, embedding_store_device
    )
    test_store = PeakShapeArrayStore(
        test_bundle, mean, std, args.feature_dtype, embedding_store_device
    )

    train_dataset = PeakShapeGroupDataset(train_store)
    val_dataset = PeakShapeGroupDataset(val_store)
    test_dataset = PeakShapeGroupDataset(test_store)

    input_dim = int(train_store.embeddings.shape[1])
    print(
        json.dumps(
            {
                "task": task,
                "training_device": str(device),
                "embedding_store_device": str(train_store.embeddings.device),
                "train_embeddings_shape": list(train_store.embeddings.shape),
                "val_embeddings_shape": list(val_store.embeddings.shape),
                "test_embeddings_shape": list(test_store.embeddings.shape),
            },
            ensure_ascii=False,
        )
    )

    model = MLPScoringHead(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        head_type=args.head_type,
        conv_kernel_size=args.conv_kernel_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = compute_pos_weight(train_store)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda records: collate_groups(records, train_store),
        drop_last=False,
    )

    best_state: dict[str, object] | None = None
    best_epoch = -1
    best_val_metrics: dict[str, float] | None = None
    best_selection_score = -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        loss_components = {"bce": 0.0, "center_flank": 0.0, "center_background": 0.0}

        progress = tqdm(
            train_loader, desc=f"Train {task} epoch {epoch}", dynamic_ncols=True
        )
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = compute_group_logits(model, batch)
            loss, components = compute_peakshape_loss(
                logits=logits,
                batch=batch,
                pos_weight=pos_weight,
                margin_center_flank=args.margin_center_flank,
                margin_center_background=args.margin_center_background,
                lambda_bce=args.lambda_bce,
                lambda_center_flank=args.lambda_center_flank,
                lambda_center_background=args.lambda_center_background,
                flank_weighting=args.flank_weighting,
                flank_weight_reference_bp=args.flank_weight_reference_bp,
            )
            loss.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip_norm
                )
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            for key, value in components.items():
                loss_components[key] += value
            progress.set_postfix(loss=float(loss.detach().cpu()))

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate_split(
            model, val_store, val_dataset, args.batch_size, device
        )
        print(
            json.dumps(
                {
                    "task": task,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_center_top1_rate": val_metrics["center_top1_rate"],
                    "val_window_auprc": val_metrics["window_auprc"],
                    "val_window_auroc": val_metrics["window_auroc"],
                    "val_mean_center_flank_margin": val_metrics[
                        "mean_center_flank_margin"
                    ],
                    "val_mean_center_background_margin": val_metrics[
                        "mean_center_background_margin"
                    ],
                    "selection_metric": args.selection_metric,
                    "selection_score": selection_score(
                        val_metrics, args.selection_metric
                    ),
                },
                ensure_ascii=False,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_center_top1_rate": val_metrics["center_top1_rate"],
                "val_window_auprc": val_metrics["window_auprc"],
                "val_window_auroc": val_metrics["window_auroc"],
                "val_mean_center_flank_margin": val_metrics["mean_center_flank_margin"],
                "val_mean_center_background_margin": val_metrics[
                    "mean_center_background_margin"
                ],
                "selection_score": selection_score(val_metrics, args.selection_metric),
            }
        )

        current_selection_score = selection_score(val_metrics, args.selection_metric)
        if best_val_metrics is None or current_selection_score > best_selection_score:
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "epoch": epoch,
            }
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_selection_score = current_selection_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError(f"No valid model state produced for task {task}")

    checkpoint_path = task_output_dir / "best_model.pt"
    calibration: dict[str, float] | None = None
    model.load_state_dict(best_state["model_state_dict"])
    if args.posthoc_negative_calibration == "quantile":
        val_logits = collect_split_logits(
            model, val_store, val_dataset, args.batch_size, device
        )
        negative_parts = [
            arr
            for arr in [val_logits["flank_logits"], val_logits["background_logits"]]
            if arr.size > 0
        ]
        if negative_parts:
            negative_logits = np.concatenate(negative_parts, axis=0)
            calibration = build_negative_calibration(
                negative_logits=negative_logits,
                negative_quantile=args.posthoc_negative_quantile,
                target_probability=args.posthoc_negative_target_prob,
            )

    torch.save(
        {
            **best_state,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "head_type": args.head_type,
            "conv_kernel_size": args.conv_kernel_size,
            "best_threshold": 0.5,
            "training_objective": "peakshape_ranking",
            "margin_center_flank": args.margin_center_flank,
            "margin_center_background": args.margin_center_background,
            "lambda_bce": args.lambda_bce,
            "lambda_center_flank": args.lambda_center_flank,
            "lambda_center_background": args.lambda_center_background,
            "flank_weighting": args.flank_weighting,
            "flank_weight_reference_bp": args.flank_weight_reference_bp,
            "best_val_metrics": best_val_metrics,
            "selection_metric": args.selection_metric,
            "best_selection_score": best_selection_score,
            "pos_weight": pos_weight,
            "posthoc_negative_calibration": calibration,
        },
        checkpoint_path,
    )

    val_metrics = evaluate_split(model, val_store, val_dataset, args.batch_size, device)
    test_metrics = evaluate_split(
        model, test_store, test_dataset, args.batch_size, device
    )

    summary = {
        "task": task,
        "best_epoch": best_epoch,
        "head_type": args.head_type,
        "conv_kernel_size": args.conv_kernel_size,
        "objective": "peakshape_ranking",
        "selection_metric": args.selection_metric,
        "best_selection_score": best_selection_score,
        "posthoc_negative_calibration": calibration,
        "val": val_metrics,
        "test": test_metrics,
    }
    (task_output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (task_output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        train_task(args, task)


if __name__ == "__main__":
    main()
