from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

DEFAULT_CONFIG = str(Path(__file__).resolve().parents[2] / "configs/models.yaml")
DEFAULT_DATASET = str(
    Path(
        os.environ.get(
            "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
        )
    )
    / "datasets/DeepSEA"
)
DEFAULT_RESULTS = str(
    Path(
        os.environ.get(
            "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
        )
    )
    / "results/DeepSEA_lora"
)
HPC_HOME = Path(os.environ.get("HPC_HOME", str(Path.home())))
DNAMODELS_ROOT = Path(
    os.environ.get("DNAMODELS_BENCHMARK_ROOT", str(HPC_HOME / "DNAModelsBenchmark"))
)
GPN_ASSETS_ROOT = Path(
    os.environ.get("GPN_STAR_ASSETS_ROOT", str(HPC_HOME / "gpn_star_assets"))
)


def build_parser():
    parser = argparse.ArgumentParser(description="Unified DeepSEA LoRA evaluation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model_name")
    parser.add_argument("--all", action="store_true", dest="all_models")
    parser.add_argument("--model_index", type=int)
    parser.add_argument(
        "--model_scope", choices=["standard", "all"], default="standard"
    )
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET)
    parser.add_argument(
        "--gpn_coordinate_root",
        default=str(DNAMODELS_ROOT / "runs/deepsea_gpn_star_m_v1/coordinates"),
    )
    parser.add_argument(
        "--gpn_msa_243",
        default=str(GPN_ASSETS_ROOT / "msa/hg38_cactus447way/243/all.zarr"),
    )
    parser.add_argument(
        "--gpn_msa_204",
        default=str(GPN_ASSETS_ROOT / "msa/hg38_cactus447way/204/all.zarr"),
    )
    parser.add_argument("--output_root", default=DEFAULT_RESULTS)
    parser.add_argument("--output_dir")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--train_samples", type=int, default=4400000)
    parser.add_argument(
        "--train_row_stride",
        type=int,
        default=1,
        help="Select every Nth row across the complete training split",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--stage", choices=["screen", "final", "custom"], default="screen"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr_schedule",
        choices=["linear", "constant"],
        default="linear",
        help="Learning-rate schedule after warmup; Evo2-1B uses constant.",
    )
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_every_steps", type=int, default=5000)
    parser.add_argument("--eval_subset_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--target_global_batch_size", type=int, default=128)
    parser.add_argument("--auto_batch", action="store_true")
    parser.add_argument("--batch_probe_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient_checkpointing", dest="gradient_checkpointing", action="store_true"
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument(
        "--attention_backend",
        choices=["eager", "flash_attention_2", "torch_flash_sdp"],
        default="eager",
        help="Attention backend for custom hEvo models; other model families ignore this setting.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="CUDA autocast dtype. BF16 matches hEvo pretraining and disables GradScaler.",
    )
    parser.add_argument("--smoke_test", action="store_true")
    return parser


# Help uses the same parser as execution, before optional model imports.
if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()


import csv  # noqa: E402 - lightweight CLI help precedes runtime imports
import json  # noqa: E402 - lightweight CLI help precedes runtime imports
import math  # noqa: E402 - lightweight CLI help precedes runtime imports
import random  # noqa: E402 - lightweight CLI help precedes runtime imports
import shutil  # noqa: E402 - lightweight CLI help precedes runtime imports
import time  # noqa: E402 - lightweight CLI help precedes runtime imports
from datetime import datetime, timedelta  # noqa: E402 - lightweight CLI help precedes runtime imports
from typing import Any  # noqa: E402 - lightweight CLI help precedes runtime imports

import numpy as np  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch.distributed as dist  # noqa: E402 - lightweight CLI help precedes runtime imports
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402 - lightweight CLI help precedes runtime imports
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch.utils.tensorboard import SummaryWriter  # noqa: E402 - lightweight CLI help precedes runtime imports

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from DeepSEA_lora.adapters import load_model_and_tokenizer, trainable_summary  # type: ignore
    from DeepSEA_lora.config import ModelEntry, select_model_entries  # type: ignore
    from DeepSEA_lora.data import (
        DeepSEAIterableDataset,
        GPNStarMSACollator,
        make_loader,
        validate_dataset,
    )  # type: ignore
else:
    from .adapters import load_model_and_tokenizer, trainable_summary
    from .config import ModelEntry, select_model_entries
    from .data import (
        DeepSEAIterableDataset,
        GPNStarMSACollator,
        make_loader,
        validate_dataset,
    )


NUM_LABELS = 919


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args()


def make_entry_loader(
    dataset,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    entry: ModelEntry,
    args: argparse.Namespace,
    **kwargs,
):
    """Build a normal sequence loader or the coordinate-backed native MSA loader."""
    collator = None
    if entry.name == "GPN-Star-M":
        collator = GPNStarMSACollator(
            dataset.split, args.gpn_coordinate_root, args.gpn_msa_243, args.gpn_msa_204
        )
        kwargs["num_workers"] = 0
        kwargs["prefetch_factor"] = 1
    return make_loader(
        dataset, tokenizer, max_length, batch_size, shuffle, collator=collator, **kwargs
    )


def row_offsets_path(
    entry: ModelEntry, args: argparse.Namespace, split: str
) -> Path | None:
    if entry.name != "GPN-Star-M":
        return None
    return Path(args.gpn_coordinate_root) / f"{split}.offsets.json"


def masked_bce_loss(
    logits: torch.Tensor, labels: torch.Tensor, sample_valid: torch.Tensor | None
) -> torch.Tensor:
    if sample_valid is None:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), labels.float()
        )
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), labels.float(), reduction="none"
    ).mean(dim=1)
    weights = sample_valid.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def amp_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.mixed_precision == "bf16":
        return torch.bfloat16
    if args.mixed_precision == "fp32":
        return torch.float32
    return torch.float16


def make_grad_scaler(device: torch.device, args: argparse.Namespace):
    return torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.mixed_precision == "fp16"
    )


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[int, int, torch.device]:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed DeepSEA LoRA requires CUDA")
        torch.cuda.set_device(local_rank)
        # Rank 0 computes 919 full-test AUROC/AUPRC values after all ranks
        # write their prediction shards. Other ranks wait in the following
        # broadcast; the default 10-minute watchdog killed the 32-GPU run.
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(hours=1)
        )
        return rank, world_size, torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    return rank, world_size, device


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(value: Any, rank: int) -> Any:
    if not dist.is_available() or not dist.is_initialized():
        return value
    objects = [value if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, default=json_default) + "\n"
        )


def update_progress(
    output_dir: Path,
    writer: SummaryWriter | None,
    rank: int,
    epoch: int,
    global_step: int,
    samples_processed: int,
    samples_total: int,
    start_time: float,
    previous_samples: int,
    previous_time: float,
    train_loss: float,
    learning_rate: float,
    device: torch.device,
) -> tuple[int, float, float]:
    """Write a rank-0 global sample progress record and a log-friendly bar."""
    if rank != 0:
        return previous_samples, previous_time, 0.0
    now = time.time()
    elapsed = max(now - start_time, 1e-6)
    window_seconds = max(now - previous_time, 1e-6)
    window_samples = max(samples_processed - previous_samples, 0)
    instant_sps = window_samples / window_seconds
    overall_sps = samples_processed / elapsed
    # Smooth the first noisy steps by preferring the overall rate until a
    # meaningful window has accumulated.
    throughput = (
        overall_sps
        if samples_processed < 10_000
        else 0.5 * overall_sps + 0.5 * instant_sps
    )
    remaining = max(samples_total - samples_processed, 0)
    eta_seconds = remaining / max(throughput, 1e-6)
    eta_wallclock = datetime.now() + timedelta(seconds=eta_seconds)
    fraction = min(max(samples_processed / max(samples_total, 1), 0.0), 1.0)
    width = 30
    filled = int(width * fraction)
    bar = "=" * filled + ">" + "." * max(0, width - filled - 1)
    record = {
        "kind": "progress",
        "epoch": epoch,
        "step": global_step,
        "samples_processed": int(samples_processed),
        "samples_total": int(samples_total),
        "progress_percent": 100.0 * fraction,
        "throughput_samples_per_sec": throughput,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta_seconds,
        "eta_wallclock": eta_wallclock.strftime("%Y-%m-%d %H:%M:%S"),
        "train_loss": train_loss,
        "learning_rate": learning_rate,
        "peak_gpu_memory_bytes": int(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    append_jsonl(output_dir / "progress.jsonl", record)
    if not (output_dir / "progress.csv").exists():
        with (output_dir / "progress.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv.DictWriter(handle, fieldnames=list(record)).writeheader()
    with (output_dir / "progress.csv").open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        csv.DictWriter(handle, fieldnames=list(record)).writerow(record)
    if writer:
        writer.add_scalar("train/samples_processed", samples_processed, global_step)
        writer.add_scalar("train/throughput_samples_per_sec", throughput, global_step)
        writer.add_scalar("train/eta_seconds", eta_seconds, global_step)
        writer.add_scalar(
            "train/peak_gpu_memory_gb",
            record["peak_gpu_memory_bytes"] / 1024**3,
            global_step,
        )
    print(
        f"[Progress] [{bar}] {fraction * 100:6.2f}% "
        f"samples={samples_processed:,}/{samples_total:,} "
        f"step={global_step:,} throughput={throughput:,.1f}/s "
        f"ETA={eta_seconds / 3600:.2f}h ({record['eta_wallclock']})",
        flush=True,
    )
    return samples_processed, now, throughput


def make_output_dir(args: argparse.Namespace, entry: ModelEntry, rank: int) -> Path:
    if args.output_dir:
        selected = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        max_length = args.max_length or entry.max_length
        name = f"{entry.name}__lora-r{args.lora_r}-a{args.lora_alpha}-d{args.lora_dropout:g}__len{max_length}__seed{args.seed}__{timestamp}"
        selected = Path(args.output_root) / name
    selected = Path(broadcast_object(str(selected), rank))
    if rank == 0:
        selected.mkdir(parents=True, exist_ok=True)
    barrier()
    return selected


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def multilabel_metrics(
    labels: np.ndarray, logits: np.ndarray
) -> dict[str, float | int]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    roc_values: list[float] = []
    pr_values: list[float] = []
    valid_labels = 0
    for column in range(labels.shape[1]):
        target = labels[:, column]
        if np.unique(target).size < 2:
            continue
        valid_labels += 1
        roc_values.append(float(roc_auc_score(target, probabilities[:, column])))
        pr_values.append(
            float(average_precision_score(target, probabilities[:, column]))
        )
    flat_labels = labels.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)
    result: dict[str, float | int] = {
        "macro_auroc": float(np.mean(roc_values)) if roc_values else float("nan"),
        "macro_auprc": float(np.mean(pr_values)) if pr_values else float("nan"),
        "valid_label_count": int(valid_labels),
        "micro_auroc": float(roc_auc_score(flat_labels, flat_probabilities))
        if np.unique(flat_labels).size > 1
        else float("nan"),
        "micro_auprc": float(average_precision_score(flat_labels, flat_probabilities)),
    }
    result["bce_loss"] = float(
        np.mean(
            np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
        )
    )
    return result


def evaluate(
    model: torch.nn.Module,
    loader,
    split: str,
    output_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    precision: torch.dtype = torch.float16,
) -> dict[str, float | int]:
    # Inference shards can differ by one example (for example 455,024 test
    # rows over 40 ranks). Bypass DDP's per-forward buffer broadcasts during
    # evaluation so those uneven final batches cannot stall peers.
    eval_model = model.module if isinstance(model, DDP) else model
    eval_model.eval()
    local_logits: list[np.ndarray] = []
    local_labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            sample_valid = batch.pop("sample_valid", None)
            valid_mask = (
                sample_valid.cpu().numpy().astype(bool)
                if sample_valid is not None
                else np.ones(len(labels), dtype=bool)
            )
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type="cuda",
                dtype=precision,
                enabled=device.type == "cuda" and precision != torch.float32,
            ):
                logits = eval_model(batch).float().cpu().numpy()
            local_logits.append(logits[valid_mask])
            local_labels.append(labels[valid_mask])
    logits = (
        np.concatenate(local_logits, axis=0)
        if local_logits
        else np.empty((0, NUM_LABELS), dtype=np.float32)
    )
    labels = (
        np.concatenate(local_labels, axis=0)
        if local_labels
        else np.empty((0, NUM_LABELS), dtype=np.float32)
    )
    part_path = output_dir / f".eval_{split}.rank{rank}.npz"
    np.savez(
        part_path, logits=logits.astype(np.float32), labels=labels.astype(np.float32)
    )
    barrier()
    metrics: dict[str, float | int] | None = None
    if rank == 0:
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for part_rank in range(world_size):
            part = np.load(output_dir / f".eval_{split}.rank{part_rank}.npz")
            all_logits.append(part["logits"])
            all_labels.append(part["labels"])
        metrics = multilabel_metrics(
            np.concatenate(all_labels), np.concatenate(all_logits)
        )
        write_json(output_dir / f"evaluation_{split}.json", metrics)
        if split == "test":
            all_y = np.concatenate(all_labels)
            all_z = np.concatenate(all_logits)
            category_ranges = {"DHS": (0, 125), "HM": (815, 919), "TF": (125, 815)}
            manifest_path = output_dir / "run_manifest.json"
            model_name = json.loads(manifest_path.read_text(encoding="utf-8"))["model"][
                "name"
            ]
            category_payload = {
                "status": "completed",
                "model_name": model_name,
                "run_dir": str(output_dir),
                "test_samples": int(all_y.shape[0]),
                "num_labels": int(all_y.shape[1]),
                "category_definition": {
                    "source": "DeepSEA published 919-feature row blocks",
                    "num_labels": NUM_LABELS,
                    "categories": {
                        name: {"start": start, "end": end, "count": end - start}
                        for name, (start, end) in category_ranges.items()
                    },
                },
                "overall": metrics,
                "categories": {
                    name: multilabel_metrics(all_y[:, start:end], all_z[:, start:end])
                    for name, (start, end) in category_ranges.items()
                },
            }
            write_json(output_dir / "category_metrics.json", category_payload)
        for part_rank in range(world_size):
            (output_dir / f".eval_{split}.rank{part_rank}.npz").unlink(missing_ok=True)
    metrics = broadcast_object(metrics, rank)
    barrier()
    return metrics


def enable_checkpointing(model: torch.nn.Module) -> bool:
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(core, "backbone", core)
    if backbone.__class__.__name__ == "GPNStarModel":
        # GPN-Star's encoder inherits RoFormerEncoder but the installed
        # Transformers release does not advertise checkpointing support on the
        # top-level custom model. Its encoder already implements checkpointed
        # layer calls, so enable that flag directly.
        encoders = [
            module
            for module in backbone.modules()
            if module.__class__.__name__ == "GPNStarEncoder"
        ]
        for encoder in encoders:
            encoder.gradient_checkpointing = True
        embedding = getattr(
            getattr(backbone, "target_embedding", None), "input_embed", None
        )
        if embedding is None:
            raise RuntimeError(
                "GPN-Star target input embedding was not found for checkpointing"
            )
        if not hasattr(backbone, "_deepsea_input_grad_hook"):

            def require_input_grads(_module, _inputs, output):
                output.requires_grad_(True)

            backbone._deepsea_input_grad_hook = embedding.register_forward_hook(
                require_input_grads
            )
        return bool(encoders)
    if hasattr(backbone, "gradient_checkpointing_enable"):
        try:
            backbone.gradient_checkpointing_enable()
        except TypeError:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
        except ValueError as exc:
            print(
                f"[checkpointing] disabled for {type(backbone).__name__}: {exc}",
                flush=True,
            )
            return False
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    config = getattr(backbone, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    return True


def save_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    named = dict(model.named_parameters())
    for name, value in state.items():
        if name in named:
            named[name].data.copy_(
                value.to(named[name].device, dtype=named[name].dtype)
            )


def configure_peft(
    model: torch.nn.Module, targets: list[str], args: argparse.Namespace
):
    # The cluster environment contains an old bitsandbytes build without a
    # CUDA 12.4 binary. PEFT 0.13 otherwise probes it even for ordinary
    # nn.Linear LoRA layers and aborts before the default dispatcher runs.
    # Quantized runs are explicitly disabled below until a compatible bnb
    # build is installed, so disabling these optional dispatchers is safe.
    import peft.tuners.lora.model as peft_lora_model

    peft_lora_model.is_bnb_available = lambda: False
    peft_lora_model.is_bnb_4bit_available = lambda: False
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
        modules_to_save=["classifier"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(model, config)


def choose_batch_and_accum(
    args: argparse.Namespace, entry: ModelEntry, world_size: int
) -> tuple[int, int]:
    """Return a safe initial batch/accumulation pair for the current node.

    The actual batch tuner can replace this pair after a short four-GPU probe.
    These conservative values keep the first smoke run from reserving an
    unnecessarily large gradient accumulation schedule.
    """
    defaults = {
        "hEvo-open-base": 16,
        "hEvo-base": 16,
        "hEvo-large": 4,
        "NT_500m_1kG": 4,
        "NT_2.5b_1kG": 2,
    }
    batch = defaults.get(entry.name, args.per_device_train_batch_size)
    batch = max(1, int(batch))
    accum = max(
        1, math.ceil(args.target_global_batch_size / max(1, world_size * batch))
    )
    return batch, accum


def probe_batch_sizes(
    model: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
    entry: ModelEntry,
    max_length: int,
    device: torch.device,
    rank: int,
    world_size: int,
    output_dir: Path,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Probe batch sizes independently on each GPU before DDP wrapping."""
    candidates = [1, 2, 4, 8, 16, 32]
    original_state = save_trainable_state(model)
    results: list[dict[str, Any]] = []
    for batch_size in candidates:
        grad_accum = max(
            1,
            math.ceil(args.target_global_batch_size / max(1, world_size * batch_size)),
        )
        dataset = DeepSEAIterableDataset(
            Path(args.dataset_root) / "train_out.csv",
            Path(args.dataset_root) / "train_labels.npy",
            "train",
            max_samples=world_size * batch_size * max(args.batch_probe_steps, 1),
            rank=rank,
            world_size=world_size,
            shuffle=False,
            offsets_path=row_offsets_path(entry, args, "train"),
        )
        loader = make_entry_loader(
            dataset,
            tokenizer,
            max_length,
            batch_size,
            False,
            entry,
            args,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
        )
        scaler = make_grad_scaler(device, args)
        ok = 1
        local_samples = 0
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                labels = batch.pop("labels").to(device, non_blocking=True)
                sample_valid = batch.pop("sample_valid", None)
                local_samples += int(labels.shape[0])
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype(args),
                    enabled=device.type == "cuda" and args.mixed_precision != "fp32",
                ):
                    logits = model(move_batch(batch, device))
                    loss = masked_bce_loss(logits, labels, sample_valid)
                if not torch.isfinite(loss):
                    ok = 0
                    break
                scaler.scale(loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                if step + 1 >= args.batch_probe_steps:
                    break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" in str(exc).lower() or isinstance(
                exc, torch.cuda.OutOfMemoryError
            ):
                ok = 0
            else:
                raise
        elapsed = max(time.time() - started, 1e-6)
        local_speed = local_samples / elapsed
        flag = torch.tensor(ok, dtype=torch.int32, device=device)
        speed_tensor = torch.tensor(local_speed, dtype=torch.float64, device=device)
        peak_tensor = torch.tensor(
            int(
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            dtype=torch.long,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            dist.all_reduce(speed_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        restore_trainable_state(model, original_state)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, loader, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        record = {
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "status": "ok" if int(flag.item()) else "oom_or_nonfinite",
            "throughput_samples_per_sec": float(speed_tensor.item()),
            "peak_gpu_memory_bytes": int(peak_tensor.item()),
        }
        results.append(record)
        if rank == 0:
            print(json.dumps({"kind": "batch_probe", **record}), flush=True)
    valid = [item for item in results if item["status"] == "ok"]
    if not valid:
        raise RuntimeError(f"No viable per-GPU batch size for {entry.name}: {results}")
    selected = max(valid, key=lambda item: item["throughput_samples_per_sec"])
    selected_batch = int(selected["batch_size"])
    selected_accum = int(selected["gradient_accumulation_steps"])
    selected_pair = torch.tensor(
        [selected_batch, selected_accum], dtype=torch.long, device=device
    )
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(selected_pair, src=0)
    selected_batch, selected_accum = map(int, selected_pair.tolist())
    if rank == 0:
        write_json(
            output_dir / "batch_tuning.json",
            {
                "selected_batch_size": selected_batch,
                "selected_gradient_accumulation_steps": selected_accum,
                "trials": results,
            },
        )
    return selected_batch, selected_accum, results


def run_smoke(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    rank: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5
    )
    scaler = make_grad_scaler(device, args)
    losses: list[float] = []
    core_model = model.module if isinstance(model, DDP) else model
    lora_b_grad_norms: list[float] = []
    for step, batch in enumerate(loader):
        labels = batch.pop("labels").to(device)
        sample_valid = batch.pop("sample_valid", None)
        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype(args),
            enabled=device.type == "cuda" and args.mixed_precision != "fp32",
        ):
            logits = model(move_batch(batch, device))
            loss = masked_bce_loss(logits, labels, sample_valid)
        scaler.scale(loss).backward()
        nonfinite_gradients = [
            name
            for name, parameter in core_model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]
        if nonfinite_gradients:
            raise RuntimeError(f"SMOKE_NONFINITE_GRADIENT: {nonfinite_gradients[:8]}")
        lora_b_grad_norms.append(
            float(
                torch.sqrt(
                    sum(
                        (
                            parameter.grad.detach().float().pow(2).sum()
                            for name, parameter in core_model.named_parameters()
                            if "lora_B" in name and parameter.grad is not None
                        ),
                    )
                ).item()
            )
            if any(
                "lora_B" in name and parameter.grad is not None
                for name, parameter in core_model.named_parameters()
            )
            else 0.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"SMOKE_NONFINITE_LOSS at step {step}: {loss_value}")
        losses.append(loss_value)
        if step >= 1:
            break
    barrier()
    if rank == 0:
        lora_b_param_norm = float(
            torch.sqrt(
                sum(
                    (
                        parameter.detach().float().pow(2).sum()
                        for name, parameter in core_model.named_parameters()
                        if "lora_B" in name
                    ),
                )
            ).item()
        )
        write_json(
            output_dir / "smoke_test.json",
            {
                "steps": len(losses),
                "losses": losses,
                "lora_b_grad_norms": lora_b_grad_norms,
                "lora_b_param_norm_after_step": lora_b_param_norm,
                "status": "PASS",
            },
        )


def run_one(
    args: argparse.Namespace,
    entry: ModelEntry,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    set_seed(args.seed + rank)
    output_dir = make_output_dir(args, entry, rank)
    dataset_info = validate_dataset(args.dataset_root) if rank == 0 else None
    dataset_info = broadcast_object(dataset_info, rank)
    max_length = int(args.max_length or entry.max_length or 1024)
    if args.smoke_test:
        train_samples = min(args.train_samples, 128)
        epochs = 1
    elif args.stage == "screen":
        train_samples = args.train_samples
        epochs = 1
    elif args.stage == "final":
        train_samples = args.train_samples
        epochs = 3
    else:
        train_samples = args.train_samples
        epochs = args.epochs

    if args.auto_batch:
        args.per_device_train_batch_size, args.gradient_accumulation_steps = (
            choose_batch_and_accum(args, entry, world_size)
        )

    print(
        f"[Run] model={entry.name} path={entry.url} output={output_dir} device={device} "
        f"world={world_size} stage={args.stage} train_batch={args.per_device_train_batch_size} "
        f"grad_accum={args.gradient_accumulation_steps}",
        flush=True,
    )
    model, tokenizer, config, family, targets = load_model_and_tokenizer(
        entry.name, entry.url, attention_backend=args.attention_backend
    )
    actual_pooling = model.pooling
    if args.quantization != "none":
        raise RuntimeError(
            "4-bit fallback is reserved for a separate quantized adapter implementation; use --quantization none for this suite"
        )
    model = configure_peft(model, targets, args)
    checkpointing_enabled = False
    # The bundled RNABert implementation does not reliably propagate input
    # gradients through its checkpointed blocks.  In the previous AIDO run
    # the classifier changed but every LoRA-B tensor stayed exactly zero and
    # validation remained at AUROC=0.5.  AIDO is small enough for 24 GB cards,
    # so prefer a non-checkpointed diagnostic/fair run until its custom
    # checkpoint wrapper has been independently validated.
    if args.gradient_checkpointing and family == "rnabert":
        print(
            "[checkpointing] disabled for rnabert/AIDO: custom checkpoint path is not LoRA-safe",
            flush=True,
        )
    elif args.gradient_checkpointing:
        checkpointing_enabled = enable_checkpointing(model)
    model = model.to(device)
    if device.type == "cuda":
        if args.mixed_precision == "bf16":
            # Match hEvo pretraining: backbone, LoRA and classifier use BF16.
            model = model.to(dtype=torch.bfloat16)
        elif args.mixed_precision == "fp16":
            # NTv3's custom stem keeps float32 activations even when its
            # checkpoint is cast to fp16; casting the whole module produces a
            # Float-vs-Half Conv1d mismatch.  Keep this architecture in FP32
            # and let autocast handle eligible CUDA kernels.
            if family == "ntv3":
                model = model.float()
            else:
                model = model.half()
            # Keep trainable parameters in FP32 for the FP16 path.
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        else:
            model = model.float()
    summary = trainable_summary(model)
    if summary["trainable_parameters"] <= 0:
        raise RuntimeError("LoRA injection produced zero trainable parameters")
    trainable_dtypes = sorted(
        {
            str(parameter.dtype)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
    )
    if (
        device.type == "cuda"
        and args.mixed_precision == "fp16"
        and trainable_dtypes != ["torch.float32"]
    ):
        raise RuntimeError(
            f"FP16 trainable parameters must be FP32, got {trainable_dtypes}"
        )
    if rank == 0:
        write_json(
            output_dir / "model_summary.json",
            {
                "model_name": entry.name,
                "model_path": entry.url,
                "adapter_family": family,
                "target_modules": targets,
                "pooling": actual_pooling,
                "declared_pooling": entry.pooling,
                "trainable_parameter_dtypes": trainable_dtypes,
                "config_model_type": getattr(config, "model_type", None),
                "config_architectures": getattr(config, "architectures", None),
                **summary,
            },
        )
        shutil.copy2(args.config, output_dir / "config_snapshot.yaml")
        write_json(
            output_dir / "run_manifest.json",
            {
                "model": entry.as_dict(),
                "dataset": dataset_info,
                "gpn_star": (
                    {
                        "model_revision": "0d8f5164fba8e8108200c160370b86652bfb6244",
                        "msa_dataset": "songlab/hg38_cactus447way",
                        "msa_revision": "f523f0467bc143b62fb5b3b9a91c510e8dc2a7f6",
                        "target_store_243_way": args.gpn_msa_243,
                        "source_only_store_204_way": args.gpn_msa_204,
                        "coordinate_map": str(
                            Path(args.gpn_coordinate_root) / "coordinate_manifest.json"
                        ),
                        "coordinate_mapping_summary": str(
                            Path(args.gpn_coordinate_root) / "mapping_summary.json"
                        ),
                    }
                    if entry.name == "GPN-Star-M"
                    else None
                ),
                "max_length": max_length,
                "train_samples": train_samples,
                "train_row_stride": args.train_row_stride,
                "epochs": epochs,
                "stage": args.stage,
                "seed": args.seed,
                "world_size": world_size,
                "device": str(device),
                "precision": args.mixed_precision if device.type == "cuda" else "fp32",
                "actual_pooling": actual_pooling,
                "trainable_parameter_dtypes": trainable_dtypes,
                "attention_backend": (
                    args.attention_backend
                    if "hevo" in entry.name.lower()
                    else "not_applicable"
                ),
                "tf32": bool(device.type == "cuda"),
                "gradient_checkpointing": bool(checkpointing_enabled),
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_schedule": args.lr_schedule,
                "gradient_clip_norm": 1.0,
                "effective_global_batch_size": world_size
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps,
                "eval_every_steps": args.eval_every_steps,
                "eval_subset_size": args.eval_subset_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "lora": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "bias": "none",
                    "target_modules": targets,
                    "modules_to_save": ["classifier"],
                },
            },
        )
    barrier()

    core_model = model
    if args.auto_batch:
        selected_batch, selected_accum, _ = probe_batch_sizes(
            core_model,
            tokenizer,
            args,
            entry,
            max_length,
            device,
            rank,
            world_size,
            output_dir,
        )
        args.per_device_train_batch_size = selected_batch
        args.gradient_accumulation_steps = selected_accum
        if rank == 0:
            write_json(
                output_dir / "selected_optimization.json",
                {
                    "per_device_train_batch_size": selected_batch,
                    "gradient_accumulation_steps": selected_accum,
                    "effective_global_batch_size": world_size
                    * selected_batch
                    * selected_accum,
                },
            )
            manifest_path = output_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "per_device_train_batch_size": selected_batch,
                    "gradient_accumulation_steps": selected_accum,
                    "effective_global_batch_size": world_size
                    * selected_batch
                    * selected_accum,
                }
            )
            write_json(manifest_path, manifest)
    wrapped_model: torch.nn.Module = (
        DDP(model, device_ids=[device.index], find_unused_parameters=False)
        if world_size > 1
        else model
    )
    if args.smoke_test:
        smoke_dataset = DeepSEAIterableDataset(
            Path(args.dataset_root) / "train_out.csv",
            Path(args.dataset_root) / "train_labels.npy",
            "train",
            max_samples=train_samples,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            offsets_path=row_offsets_path(entry, args, "train"),
        )
        loader = make_entry_loader(
            smoke_dataset,
            tokenizer,
            max_length,
            args.per_device_train_batch_size,
            False,
            entry,
            args,
        )
        run_smoke(wrapped_model, loader, device, rank, output_dir, args)
        return

    train_dataset = DeepSEAIterableDataset(
        Path(args.dataset_root) / "train_out.csv",
        Path(args.dataset_root) / "train_labels.npy",
        "train",
        max_samples=train_samples,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        offsets_path=row_offsets_path(entry, args, "train"),
        row_stride=args.train_row_stride,
    )
    valid_dataset = DeepSEAIterableDataset(
        Path(args.dataset_root) / "valid_out.csv",
        None,
        "valid",
        max_samples=args.eval_subset_size,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        offsets_path=row_offsets_path(entry, args, "valid"),
    )
    valid_full_dataset = DeepSEAIterableDataset(
        Path(args.dataset_root) / "valid_out.csv",
        None,
        "valid",
        rank=rank,
        world_size=world_size,
        shuffle=False,
        offsets_path=row_offsets_path(entry, args, "valid"),
    )
    test_dataset = DeepSEAIterableDataset(
        Path(args.dataset_root) / "test_out.csv",
        Path(args.dataset_root) / "test_labels.npy",
        "test",
        rank=rank,
        world_size=world_size,
        shuffle=False,
        offsets_path=row_offsets_path(entry, args, "test"),
    )
    train_loader = make_entry_loader(
        train_dataset,
        tokenizer,
        max_length,
        args.per_device_train_batch_size,
        True,
        entry,
        args,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    valid_loader = make_entry_loader(
        valid_dataset,
        tokenizer,
        max_length,
        args.per_device_eval_batch_size,
        False,
        entry,
        args,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    valid_full_loader = make_entry_loader(
        valid_full_dataset,
        tokenizer,
        max_length,
        args.per_device_eval_batch_size,
        False,
        entry,
        args,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = make_entry_loader(
        test_dataset,
        tokenizer,
        max_length,
        args.per_device_eval_batch_size,
        False,
        entry,
        args,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    local_rows = math.ceil(train_samples / world_size)
    updates_per_epoch = math.ceil(
        local_rows
        / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    total_steps = updates_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    optimizer_parameters = [
        parameter for parameter in wrapped_model.parameters() if parameter.requires_grad
    ]
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    try:
        optimizer = torch.optim.AdamW(
            optimizer_parameters, fused=True, **optimizer_kwargs
        )
        optimizer_impl = "AdamW_fused"
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(optimizer_parameters, **optimizer_kwargs)
        optimizer_impl = "AdamW"
    if args.lr_schedule == "constant":

        def lr_multiplier(step):
            return min((step + 1) / warmup_steps, 1.0)
    else:

        def lr_multiplier(step):
            return min((step + 1) / warmup_steps, 1.0) * max(
                0.0, (total_steps - step) / max(1, total_steps - warmup_steps)
            )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    writer = SummaryWriter(str(output_dir / "tensorboard")) if rank == 0 else None
    scaler = make_grad_scaler(device, args)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    start = time.time()
    total_samples = train_samples * epochs
    local_samples_total = 0
    last_logged_samples = 0
    last_logged_time = start
    last_throughput = 0.0
    if rank == 0:
        write_json(
            output_dir / "optimization_config.json",
            {
                "optimizer": optimizer_impl,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_schedule": args.lr_schedule,
                "gradient_clip_norm": 1.0,
                "effective_global_batch_size": world_size
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps,
                "target_global_batch_size": args.target_global_batch_size,
                "attention_backend": (
                    args.attention_backend
                    if "hevo" in entry.name.lower()
                    else "not_applicable"
                ),
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "progress_total_samples": total_samples,
            },
        )

    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        wrapped_model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        batch_count = 0
        for batch_index, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device, non_blocking=True)
            sample_valid = batch.pop("sample_valid", None)
            local_samples_total += int(labels.shape[0])
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype(args),
                enabled=device.type == "cuda" and args.mixed_precision != "fp32",
            ):
                logits = wrapped_model(move_batch(batch, device))
                loss = masked_bce_loss(logits, labels, sample_valid)
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            epoch_loss += float(loss.item())
            batch_count += 1
            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.eval_every_steps == 0:
                    small_metrics = evaluate(
                        wrapped_model,
                        valid_loader,
                        f"valid_small_step{global_step}",
                        output_dir,
                        device,
                        rank,
                        world_size,
                        precision=amp_dtype(args),
                    )
                    wrapped_model.train()
                    eval_count_tensor = torch.tensor(
                        local_samples_total, dtype=torch.long, device=device
                    )
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(eval_count_tensor, op=dist.ReduceOp.SUM)
                    eval_record = {
                        "kind": "step_eval",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "samples_processed_global": min(
                            int(eval_count_tensor.item()), total_samples
                        ),
                        "valid_small": small_metrics,
                    }
                    history.append(eval_record)
                    if rank == 0:
                        append_jsonl(output_dir / "eval_history.jsonl", eval_record)
                        print(json.dumps(eval_record, default=json_default), flush=True)
                        if writer:
                            for key, value in small_metrics.items():
                                if isinstance(value, (float, int)):
                                    writer.add_scalar(
                                        f"valid_small/{key}", value, global_step
                                    )
                if global_step % args.logging_steps == 0:
                    count_tensor = torch.tensor(
                        local_samples_total, dtype=torch.long, device=device
                    )
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                    global_samples_processed = min(
                        int(count_tensor.item()), total_samples
                    )
                    last_logged_samples, last_logged_time, last_throughput = (
                        update_progress(
                            output_dir,
                            writer,
                            rank,
                            epoch + 1,
                            global_step,
                            global_samples_processed,
                            total_samples,
                            start,
                            last_logged_samples,
                            last_logged_time,
                            epoch_loss / max(1, batch_count),
                            scheduler.get_last_lr()[0],
                            device,
                        )
                    )
        if batch_count % args.gradient_accumulation_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        valid_metrics = evaluate(
            wrapped_model,
            valid_full_loader,
            f"valid_epoch{epoch + 1}",
            output_dir,
            device,
            rank,
            world_size,
            precision=amp_dtype(args),
        )
        wrapped_model.train()
        record = {
            "kind": "epoch_eval",
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": epoch_loss / max(1, batch_count),
            "valid": valid_metrics,
        }
        history.append(record)
        if rank == 0:
            append_jsonl(output_dir / "eval_history.jsonl", record)
            print(json.dumps(record, default=json_default), flush=True)
            if writer:
                for key, value in valid_metrics.items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(f"valid/{key}", value, epoch + 1)
        score = float(valid_metrics.get("macro_auprc", float("nan")))
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = save_trainable_state(core_model)
            if rank == 0:
                core_model.save_pretrained(str(output_dir), safe_serialization=True)

        count_tensor = torch.tensor(
            local_samples_total, dtype=torch.long, device=device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        global_samples_processed = min(int(count_tensor.item()), total_samples)
        last_logged_samples, last_logged_time, last_throughput = update_progress(
            output_dir,
            writer,
            rank,
            epoch + 1,
            global_step,
            global_samples_processed,
            total_samples,
            start,
            last_logged_samples,
            last_logged_time,
            epoch_loss / max(1, batch_count),
            scheduler.get_last_lr()[0],
            device,
        )

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")
    restore_trainable_state(core_model, best_state)
    barrier()
    test_metrics = evaluate(
        wrapped_model,
        test_loader,
        "test",
        output_dir,
        device,
        rank,
        world_size,
        precision=amp_dtype(args),
    )
    elapsed = time.time() - start
    peak_memory = int(
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    if rank == 0:
        core_model.save_pretrained(str(output_dir), safe_serialization=True)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(output_dir))
        if writer:
            writer.close()
        final = {
            "status": "completed",
            "model_name": entry.name,
            "model_path": entry.url,
            "stage": args.stage,
            "best_epoch": best_epoch,
            "best_valid_macro_auprc": best_score,
            "history": history,
            "test": test_metrics,
            "train_samples": train_samples,
            "epochs": epochs,
            "global_steps": global_step,
            "samples_processed": total_samples,
            "throughput_samples_per_sec": total_samples / max(elapsed, 1e-6),
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": peak_memory,
            "trainable": summary,
            "optimization": {
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_global_batch_size": world_size
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
            },
        }
        write_json(output_dir / "metrics.json", final)
        write_json(
            output_dir / "trainer_state.json",
            {
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_metric": best_score,
                "history": history,
            },
        )
        write_json(output_dir / "eval_history.json", history)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "output_dir": str(output_dir),
                    "test": test_metrics,
                },
                default=json_default,
            ),
            flush=True,
        )
    barrier()


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    entries = select_model_entries(
        args.config,
        model_name=args.model_name,
        all_models=args.all_models,
        model_scope=args.model_scope,
        model_index=args.model_index,
    )
    if len(entries) > 1 and world_size > 1:
        raise RuntimeError(
            "Distributed execution requires one selected model; use --all with --model_index in a Slurm array"
        )
    if len(entries) > 1 and rank == 0:
        print(
            json.dumps({"selected_models": [entry.name for entry in entries]}),
            flush=True,
        )
    for entry in entries:
        run_one(args, entry, rank, world_size, device)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
