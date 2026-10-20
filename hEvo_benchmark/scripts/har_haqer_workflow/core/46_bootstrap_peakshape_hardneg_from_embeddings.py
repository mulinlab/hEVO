from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


class MLPScoringHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, head_type: str):
        super().__init__()
        if head_type == "linear":
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, 1)
            )
        elif head_type == "mlp1":
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, 1),
            )
        elif head_type == "mlp2":
            bottleneck_dim = max(hidden_dim // 4, 32)
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, bottleneck_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(bottleneck_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap hard-negative background rows directly from peakshape embeddings."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=["har", "haqer"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--pool-top-k", type=int, default=2000)
    parser.add_argument("--pool-min-probability", type=float, default=0.5)
    parser.add_argument("--hard-negatives-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260515)
    return parser.parse_args()


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    probs = np.empty_like(logits, dtype=np.float64)
    pos = logits >= 0
    neg = ~pos
    probs[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_logits = np.exp(logits[neg])
    probs[neg] = exp_logits / (1.0 + exp_logits)
    return probs.astype(np.float32, copy=False)


def score_embeddings(
    model: torch.nn.Module, features: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = torch.from_numpy(features[start:end]).to(device, non_blocking=True)
            batch_logits = model(batch).detach().float().cpu().numpy()
            logits.append(batch_logits)
    return (
        np.concatenate(logits, axis=0) if logits else np.empty((0,), dtype=np.float32)
    )


def _infer_end(start: np.ndarray, size_bp: np.ndarray | None) -> np.ndarray:
    start_i64 = start.astype(np.int64, copy=False)
    if size_bp is not None:
        return (start_i64 + size_bp.astype(np.int64, copy=False)).astype(
            start.dtype, copy=False
        )
    return (start_i64 + 1).astype(start.dtype, copy=False)


def _default_object_array(values: list[str], length: int) -> np.ndarray:
    return np.asarray(values[:length], dtype=object)


def load_split_arrays(
    split_dir: Path, stem: str, split_name: str, task: str
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path in split_dir.glob(f"{stem}.*.npy"):
        suffix = path.name[len(stem) + 1 :]
        arrays[suffix] = np.load(path, allow_pickle=True)

    required = [
        "chromosome.npy",
        "embeddings.npy",
        "group_id.npy",
        "labels.npy",
        "offset_bp.npy",
        "role.npy",
        "start.npy",
    ]
    missing_required = [suffix for suffix in required if suffix not in arrays]
    if missing_required:
        raise FileNotFoundError(
            f"Missing required arrays in {split_dir}: {missing_required}"
        )

    n_rows = int(arrays["embeddings.npy"].shape[0])
    if "dataset_index.npy" not in arrays:
        arrays["dataset_index.npy"] = np.arange(n_rows, dtype=np.int64)
    if "size_bp.npy" not in arrays:
        arrays["size_bp.npy"] = np.zeros(n_rows, dtype=np.int64)
    if "end.npy" not in arrays:
        arrays["end.npy"] = _infer_end(arrays["start.npy"], arrays.get("size_bp.npy"))
    if "label_name.npy" not in arrays:
        labels = arrays["labels.npy"].astype(np.int64, copy=False)
        roles = arrays["role.npy"].astype(str)
        label_names = np.where(labels > 0, "positive", roles)
        arrays["label_name.npy"] = np.asarray(label_names, dtype=object)
    if "lmdb_key.npy" not in arrays:
        arrays["lmdb_key.npy"] = np.asarray(
            [f"{task}_{split_name}_{i:08d}" for i in range(n_rows)], dtype=object
        )
    if "region_type.npy" not in arrays:
        arrays["region_type.npy"] = arrays["role.npy"].astype(object, copy=False)
    if "sample_id.npy" not in arrays:
        arrays["sample_id.npy"] = np.asarray(
            [f"{task}_{split_name}_{i:08d}" for i in range(n_rows)], dtype=object
        )
    if "source.npy" not in arrays:
        arrays["source.npy"] = np.asarray(["original"] * n_rows, dtype=object)
    if "split.npy" not in arrays:
        arrays["split.npy"] = np.asarray([split_name] * n_rows, dtype=object)
    return arrays


def save_split_arrays(
    split_dir: Path, stem: str, arrays: dict[str, np.ndarray]
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for suffix, values in arrays.items():
        np.save(split_dir / f"{stem}.{suffix}", values)


def copy_meta_files(
    src_dir: Path,
    dst_dir: Path,
    src_stem: str,
    dst_stem: str,
    extra_meta: dict[str, object],
) -> None:
    for name in ("manifest.json", "meta.json"):
        src_path = src_dir / f"{src_stem}.{name}"
        payload = (
            json.loads(src_path.read_text(encoding="utf-8"))
            if src_path.exists()
            else {}
        )
        payload.update(extra_meta)
        payload["num_embeddings"] = int(extra_meta["num_embeddings"])
        (dst_dir / f"{dst_stem}.{name}").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    upper = args.task.upper()
    train_src_dir = args.source_root / f"{args.task}_peakshape_train"
    val_src_dir = args.source_root / f"{args.task}_peakshape_val"
    test_src_dir = args.source_root / f"{args.task}_peakshape_test"
    train_stem = f"14_{upper}_peakshape_train_512.lmdb"
    val_stem = f"14_{upper}_peakshape_val_512.lmdb"
    test_stem = f"14_{upper}_peakshape_test_512.lmdb"

    train_arrays = load_split_arrays(train_src_dir, train_stem, "train", args.task)
    val_arrays = load_split_arrays(val_src_dir, val_stem, "val", args.task)
    test_arrays = load_split_arrays(test_src_dir, test_stem, "test", args.task)

    mean = np.load(args.stats_dir / "train_mean.npy").astype(np.float32, copy=False)
    std = np.load(args.stats_dir / "train_std.npy").astype(np.float32, copy=False)
    features = (
        train_arrays["embeddings.npy"].astype(np.float32, copy=False) - mean
    ) / std

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MLPScoringHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        head_type=str(checkpoint["head_type"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logits = score_embeddings(model, features, args.score_batch_size, device)
    probs = stable_sigmoid(logits)

    roles = train_arrays["role.npy"].astype(str)
    labels = train_arrays["labels.npy"].astype(np.int64, copy=False)
    group_ids = train_arrays["group_id.npy"]
    background_mask = (roles == "background") & (labels == 0)
    pool_indices = np.flatnonzero(
        background_mask & (probs >= args.pool_min_probability)
    )
    if pool_indices.size == 0:
        raise RuntimeError(
            "No hard-negative candidates passed the probability threshold."
        )

    pool_indices = pool_indices[np.argsort(probs[pool_indices])[::-1]]
    pool_indices = pool_indices[: min(args.pool_top_k, pool_indices.size)]

    unique_group_ids = np.unique(group_ids)
    new_rows: dict[str, list[object]] = {suffix: [] for suffix in train_arrays.keys()}
    next_dataset_index = int(train_arrays["dataset_index.npy"].shape[0])
    duplicate_counter = 0

    for group_id in unique_group_ids:
        allowed = pool_indices[group_ids[pool_indices] != group_id]
        if allowed.size == 0:
            allowed = pool_indices
        take = min(args.hard_negatives_per_group, allowed.size)
        chosen = rng.choice(allowed, size=take, replace=False)
        for source_idx in np.atleast_1d(chosen).tolist():
            duplicate_counter += 1
            for suffix, values in train_arrays.items():
                if suffix == "dataset_index.npy":
                    new_rows[suffix].append(next_dataset_index)
                elif suffix == "group_id.npy":
                    new_rows[suffix].append(group_id)
                elif suffix == "role.npy":
                    new_rows[suffix].append("background")
                elif suffix == "labels.npy":
                    new_rows[suffix].append(0)
                elif suffix == "label_name.npy":
                    new_rows[suffix].append("background")
                elif suffix == "offset_bp.npy":
                    new_rows[suffix].append(0)
                elif suffix == "split.npy":
                    new_rows[suffix].append("train")
                elif suffix == "sample_id.npy":
                    new_rows[suffix].append(
                        f"{values[source_idx]}__hardnegdup_{duplicate_counter:08d}"
                    )
                elif suffix == "source.npy":
                    new_rows[suffix].append("mined_hard_negative")
                elif suffix == "lmdb_key.npy":
                    new_rows[suffix].append(f"hardnegdup_{duplicate_counter:08d}")
                else:
                    new_rows[suffix].append(values[source_idx])
            next_dataset_index += 1

    augmented_train = {}
    for suffix, values in train_arrays.items():
        appended = np.asarray(
            new_rows[suffix], dtype=values.dtype if values.dtype != object else object
        )
        augmented_train[suffix] = np.concatenate([values, appended], axis=0)

    output_train_dir = args.output_root / f"{args.task}_peakshape_train"
    output_val_dir = args.output_root / f"{args.task}_peakshape_val"
    output_test_dir = args.output_root / f"{args.task}_peakshape_test"

    save_split_arrays(output_train_dir, train_stem, augmented_train)
    save_split_arrays(output_val_dir, val_stem, val_arrays)
    save_split_arrays(output_test_dir, test_stem, test_arrays)

    meta = {
        "bootstrap_source_root": str(args.source_root),
        "bootstrap_checkpoint": str(args.checkpoint),
        "pool_top_k": int(args.pool_top_k),
        "pool_min_probability": float(args.pool_min_probability),
        "hard_negatives_per_group": int(args.hard_negatives_per_group),
        "num_embeddings": int(augmented_train["dataset_index.npy"].shape[0]),
        "num_hard_negative_duplicates": int(duplicate_counter),
    }
    copy_meta_files(train_src_dir, output_train_dir, train_stem, train_stem, meta)
    copy_meta_files(
        val_src_dir,
        output_val_dir,
        val_stem,
        val_stem,
        {"num_embeddings": int(val_arrays["dataset_index.npy"].shape[0])},
    )
    copy_meta_files(
        test_src_dir,
        output_test_dir,
        test_stem,
        test_stem,
        {"num_embeddings": int(test_arrays["dataset_index.npy"].shape[0])},
    )

    summary = {
        "task": args.task,
        "source_rows": int(train_arrays["dataset_index.npy"].shape[0]),
        "augmented_rows": int(augmented_train["dataset_index.npy"].shape[0]),
        "num_groups": int(unique_group_ids.shape[0]),
        "pool_candidates": int(pool_indices.shape[0]),
        "num_hard_negative_duplicates": int(duplicate_counter),
        "output_root": str(args.output_root),
    }
    (args.output_root / f"{args.task}_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
