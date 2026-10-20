from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import pysam  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent fallback
    pysam = None

try:
    from pyfaidx import Fasta as PyfaidxFasta
except ImportError:  # pragma: no cover - environment-dependent fallback
    PyfaidxFasta = None


DEFAULT_HEAD_CHECKPOINT = Path(
    "results/har_loss_ratio_grid/bce_ratio_64/har/best_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a genomic region with overlapping windows, extract last-layer pooled embeddings, "
            "score them with a trained HAR/HAQER head, and export raw score tracks."
        )
    )
    parser.add_argument(
        "--region",
        type=str,
        required=True,
        help='Genomic region like "chr1:100000-200000".',
    )
    parser.add_argument(
        "--fasta", type=Path, required=True, help="Reference FASTA path."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="HF-compatible sequence model path.",
    )
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        default=DEFAULT_HEAD_CHECKPOINT,
        help="Trained scoring-head checkpoint. Defaults to the current best HAR head.",
    )
    parser.add_argument(
        "--head-stats-dir",
        type=Path,
        default=None,
        help="Directory containing train_mean.npy and train_std.npy. Defaults to checkpoint parent.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix without extension.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Inference batch size."
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="DataLoader worker count."
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor."
    )
    parser.add_argument(
        "--window-size", type=int, default=512, help="Scanning window size."
    )
    parser.add_argument("--stride", type=int, default=64, help="Sliding stride in bp.")
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Tokenizer max_length."
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "cls", "max"],
        default="mean",
        help="Pooling applied to the model last hidden state.",
    )
    parser.add_argument(
        "--model-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Sequence model inference dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device for inference, e.g. "cuda" or "cpu".',
    )
    parser.add_argument(
        "--no-tf32", action="store_true", help="Disable TF32 acceleration on CUDA."
    )
    parser.add_argument(
        "--logit-shift",
        type=float,
        default=None,
        help=(
            "Optional additive shift applied to logits before sigmoid. "
            "If omitted, the script uses checkpoint posthoc_negative_calibration when available."
        ),
    )
    parser.add_argument(
        "--disable-checkpoint-calibration",
        action="store_true",
        help="Ignore any posthoc_negative_calibration stored inside the checkpoint.",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep windows containing non-ACGT bases and mark scores as NaN instead of dropping them.",
    )
    return parser.parse_args()


def parse_region(region: str) -> tuple[str, int, int]:
    if ":" not in region or "-" not in region:
        raise ValueError(f"Invalid region string: {region}")
    chrom, coords = region.split(":", 1)
    start_text, end_text = coords.replace(",", "").split("-", 1)
    start = int(start_text)
    end = int(end_text)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid region boundaries: {region}")
    return chrom, start, end


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    probs = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0
    negative_mask = ~positive_mask
    probs[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_logits = np.exp(logits[negative_mask])
    probs[negative_mask] = exp_logits / (1.0 + exp_logits)
    return probs.astype(np.float32, copy=False)


def ensure_attention_mask(
    encoded: dict[str, torch.Tensor], tokenizer: AutoTokenizer
) -> torch.Tensor:
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        return attention_mask
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError(
            "Tokenizer returned no attention_mask and has no pad_token_id."
        )
    return (encoded["input_ids"] != pad_token_id).long()


def resolve_backbone(model: torch.nn.Module) -> torch.nn.Module:
    backbone_names = [
        "bert",
        "roberta",
        "transformer",
        "backbone",
        "model",
        "base_model",
    ]
    for name in backbone_names:
        module = getattr(model, name, None)
        if module is not None and module is not model:
            return module
    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return base_model
    return model


def pool_last_hidden(
    outputs: Any, attention_mask: torch.Tensor, pooling: str
) -> torch.Tensor:
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Model output has neither last_hidden_state nor hidden_states."
            )
        last_hidden = hidden_states[-1]

    if pooling == "cls":
        pooler = getattr(outputs, "pooler_output", None)
        if pooler is not None:
            return pooler
        return last_hidden[:, 0, :]

    if pooling == "max":
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        masked = last_hidden.masked_fill(mask == 0, torch.finfo(last_hidden.dtype).min)
        return masked.max(dim=1).values

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = torch.sum(last_hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


class MLPScoringHead(torch.nn.Module):
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
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, 1),
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
        elif head_type == "cnn1d":
            if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
                raise ValueError(
                    "conv_kernel_size must be a positive odd integer for cnn1d."
                )
            conv_hidden_dim = max(hidden_dim // 2, 32)
            padding = conv_kernel_size // 2
            self.norm = torch.nn.LayerNorm(input_dim)
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(
                    1, conv_hidden_dim, kernel_size=conv_kernel_size, padding=padding
                ),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Conv1d(
                    conv_hidden_dim,
                    hidden_dim,
                    kernel_size=conv_kernel_size,
                    padding=padding,
                ),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.AdaptiveAvgPool1d(1),
            )
            self.out = torch.nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.head_type = head_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "cnn1d":
            x = self.norm(x).unsqueeze(1)
            x = self.net(x).squeeze(-1)
            return self.out(x).squeeze(-1)
        return self.net(x).squeeze(-1)


def generate_windows(
    region_start: int, region_end: int, window_size: int, stride: int
) -> list[tuple[int, int]]:
    span = region_end - region_start
    if span <= 0:
        return []
    if span <= window_size:
        return [(region_start, region_start + window_size)]

    windows: list[tuple[int, int]] = []
    current = region_start
    last_start = region_end - window_size
    while current <= last_start:
        windows.append((current, current + window_size))
        current += stride
    if windows[-1][0] != last_start:
        windows.append((last_start, region_end))
    return windows


class RegionWindowDataset(Dataset):
    def __init__(
        self,
        fasta_path: Path,
        chrom: str,
        region_start: int,
        region_end: int,
        window_size: int,
        stride: int,
        keep_invalid: bool,
    ):
        self.fasta_path = str(fasta_path)
        self.chrom = chrom
        self.region_start = region_start
        self.region_end = region_end
        self.window_size = window_size
        self.stride = stride
        self.keep_invalid = keep_invalid
        self.windows = generate_windows(region_start, region_end, window_size, stride)
        self.fasta: Any | None = None

    def _get_fasta(self) -> Any:
        if self.fasta is None:
            if pysam is not None:
                self.fasta = pysam.FastaFile(self.fasta_path)
            elif PyfaidxFasta is not None:
                self.fasta = PyfaidxFasta(
                    self.fasta_path,
                    rebuild=False,
                    as_raw=True,
                    sequence_always_upper=True,
                )
            else:
                raise RuntimeError(
                    "Neither pysam nor pyfaidx is available for FASTA reading."
                )
        return self.fasta

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start, end = self.windows[index]
        fasta = self._get_fasta()
        if pysam is not None and isinstance(fasta, pysam.FastaFile):
            sequence = fasta.fetch(self.chrom, start, end).upper()
        else:
            sequence = str(fasta[self.chrom][start:end]).upper()
        valid = len(sequence) == self.window_size and set(sequence).issubset(
            {"A", "C", "G", "T"}
        )
        return {
            "index": index,
            "chromosome": self.chrom,
            "start": start,
            "end": end,
            "center": start + self.window_size // 2,
            "sequence": sequence,
            "valid": valid,
            "keep": valid or self.keep_invalid,
        }


def collate_windows(records: list[dict[str, Any]]) -> dict[str, Any]:
    kept = [record for record in records if record["keep"]]
    return {
        "all_indices": np.asarray(
            [record["index"] for record in records], dtype=np.int64
        ),
        "indices": np.asarray([record["index"] for record in kept], dtype=np.int64),
        "chromosome": [record["chromosome"] for record in kept],
        "start": np.asarray([record["start"] for record in kept], dtype=np.int64),
        "end": np.asarray([record["end"] for record in kept], dtype=np.int64),
        "center": np.asarray([record["center"] for record in kept], dtype=np.int64),
        "sequence": [record["sequence"] for record in kept],
        "valid": np.asarray([record["valid"] for record in kept], dtype=bool),
    }


def build_loader(
    dataset: Dataset, batch_size: int, num_workers: int, prefetch_factor: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_windows,
        drop_last=False,
    )


@torch.inference_mode()
def extract_embeddings(
    loader: DataLoader,
    tokenizer: AutoTokenizer,
    backbone: torch.nn.Module,
    device: torch.device,
    model_dtype: torch.dtype,
    max_length: int,
    pooling: str,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = len(loader.dataset)
    embeddings = np.full((total, feature_dim), np.nan, dtype=np.float32)
    valid_mask = np.zeros(total, dtype=bool)
    centers = np.full(total, -1, dtype=np.int64)

    progress = tqdm(loader, desc="Scan windows", dynamic_ncols=True)
    autocast_enabled = device.type == "cuda" and model_dtype in {
        torch.float16,
        torch.bfloat16,
    }

    for batch in progress:
        if batch["indices"].size == 0:
            continue

        encoded = tokenizer(
            batch["sequence"],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = ensure_attention_mask(encoded, tokenizer).to(
            device, non_blocking=True
        )

        with torch.autocast(
            device_type=device.type, dtype=model_dtype, enabled=autocast_enabled
        ):
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=False,
            )
            pooled = pool_last_hidden(outputs, attention_mask, pooling)

        pooled_np = pooled.detach().float().cpu().numpy()
        embeddings[batch["indices"]] = pooled_np
        valid_mask[batch["indices"]] = batch["valid"]
        centers[batch["indices"]] = batch["center"]
        progress.set_postfix(valid_windows=int(valid_mask.sum()))

    return embeddings, valid_mask, centers


def load_head(
    checkpoint_path: Path, device: torch.device
) -> tuple[MLPScoringHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MLPScoringHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        head_type=str(checkpoint["head_type"]),
        conv_kernel_size=int(checkpoint.get("conv_kernel_size", 7)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def resolve_logit_shift(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> tuple[float, str]:
    if args.logit_shift is not None:
        return float(args.logit_shift), "cli"
    if args.disable_checkpoint_calibration:
        return 0.0, "disabled"
    calibration = checkpoint.get("posthoc_negative_calibration")
    if isinstance(calibration, dict):
        shift = calibration.get("inference_logit_shift")
        if shift is not None:
            return float(shift), "checkpoint"
    return 0.0, "none"


@torch.inference_mode()
def score_embeddings(
    head: MLPScoringHead,
    normalized_embeddings: np.ndarray,
    valid_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    total = normalized_embeddings.shape[0]
    logits = np.full(total, np.nan, dtype=np.float32)
    probabilities = np.full(total, np.nan, dtype=np.float32)

    valid_indices = np.flatnonzero(valid_mask)
    progress = tqdm(
        range(0, valid_indices.size, batch_size),
        desc="Score windows",
        dynamic_ncols=True,
    )
    for start in progress:
        end = min(start + batch_size, valid_indices.size)
        batch_indices = valid_indices[start:end]
        features = torch.from_numpy(normalized_embeddings[batch_indices]).to(
            device, non_blocking=True
        )
        batch_logits = head(features).detach().float().cpu().numpy()
        batch_probabilities = stable_sigmoid(batch_logits)
        logits[batch_indices] = batch_logits
        probabilities[batch_indices] = batch_probabilities
    return logits, probabilities


def write_bedgraph(
    path: Path, chrom: str, starts: np.ndarray, ends: np.ndarray, values: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for start, end, value in zip(starts.tolist(), ends.tolist(), values.tolist()):
            if np.isnan(value):
                continue
            handle.write(f"{chrom}\t{start}\t{end}\t{value:.8f}\n")


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    chrom, region_start, region_end = parse_region(args.region)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    stats_dir = args.head_stats_dir or args.head_checkpoint.parent
    mean = np.load(stats_dir / "train_mean.npy").astype(np.float32, copy=False)
    std = np.load(stats_dir / "train_std.npy").astype(np.float32, copy=False)

    head, checkpoint = load_head(args.head_checkpoint, device)
    feature_dim = int(checkpoint["input_dim"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    sequence_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    backbone = resolve_backbone(sequence_model).to(device)
    backbone.eval()

    dataset = RegionWindowDataset(
        fasta_path=args.fasta,
        chrom=chrom,
        region_start=region_start,
        region_end=region_end,
        window_size=args.window_size,
        stride=args.stride,
        keep_invalid=args.keep_invalid,
    )
    loader = build_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    embeddings, valid_mask, centers = extract_embeddings(
        loader=loader,
        tokenizer=tokenizer,
        backbone=backbone,
        device=device,
        model_dtype=dtype_from_name(args.model_dtype),
        max_length=args.max_length,
        pooling=args.pooling,
        feature_dim=feature_dim,
    )

    normalized_embeddings = np.zeros_like(embeddings, dtype=np.float32)
    valid_indices = np.flatnonzero(valid_mask)
    normalized_embeddings[valid_indices] = (embeddings[valid_indices] - mean) / std

    logits, probabilities = score_embeddings(
        head=head,
        normalized_embeddings=normalized_embeddings,
        valid_mask=valid_mask,
        batch_size=args.batch_size,
        device=device,
    )
    logit_shift, logit_shift_source = resolve_logit_shift(args, checkpoint)
    if logit_shift != 0.0:
        logits[valid_indices] = logits[valid_indices] + np.float32(logit_shift)
        probabilities[valid_indices] = stable_sigmoid(logits[valid_indices])

    starts = np.asarray([window[0] for window in dataset.windows], dtype=np.int64)
    ends = np.asarray([window[1] for window in dataset.windows], dtype=np.int64)

    np.savez_compressed(
        args.output_prefix.with_suffix(".raw_scores.npz"),
        chromosome=np.asarray([chrom] * len(dataset.windows)),
        start=starts,
        end=ends,
        center=centers,
        valid_mask=valid_mask,
        logits=logits,
        probabilities=probabilities,
        stride=np.int64(args.stride),
        window_size=np.int64(args.window_size),
        region_start=np.int64(region_start),
        region_end=np.int64(region_end),
    )

    write_bedgraph(
        args.output_prefix.with_suffix(".logits.bedgraph"), chrom, starts, ends, logits
    )
    write_bedgraph(
        args.output_prefix.with_suffix(".probabilities.bedgraph"),
        chrom,
        starts,
        ends,
        probabilities,
    )

    metadata = {
        "region": args.region,
        "fasta": str(args.fasta),
        "model_path": str(args.model_path),
        "head_checkpoint": str(args.head_checkpoint),
        "head_type": checkpoint["head_type"],
        "best_threshold": float(checkpoint.get("best_threshold", 0.5)),
        "pooling": args.pooling,
        "window_size": args.window_size,
        "stride": args.stride,
        "logit_shift": logit_shift,
        "logit_shift_source": logit_shift_source,
        "posthoc_negative_calibration": checkpoint.get("posthoc_negative_calibration"),
        "total_windows": len(dataset.windows),
        "valid_windows": int(valid_mask.sum()),
        "invalid_windows": int((~valid_mask).sum()),
    }
    args.output_prefix.with_suffix(".scan_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
