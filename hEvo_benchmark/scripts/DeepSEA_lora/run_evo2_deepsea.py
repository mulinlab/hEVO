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
DEFAULT_OUTPUT = str(
    Path(
        os.environ.get(
            "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
        )
    )
    / "results/DeepSEA_lora"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Formal DeepSEA LoRA evaluation for Evo2"
    )
    parser.add_argument(
        "--model_name", required=True, choices=["evo2_1b_base", "evo2_7b_base"]
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT)
    parser.add_argument("--output_dir")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--train_samples", type=int, default=4400000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_every_steps", type=int, default=5000)
    parser.add_argument("--eval_subset_size", type=int, default=4096)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_only", action="store_true")
    return parser


# Help uses the same parser as execution, before optional model imports.
if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()


import json  # noqa: E402 - lightweight CLI help precedes runtime imports
import math  # noqa: E402 - lightweight CLI help precedes runtime imports
import random  # noqa: E402 - lightweight CLI help precedes runtime imports
import time  # noqa: E402 - lightweight CLI help precedes runtime imports
from datetime import datetime, timedelta  # noqa: E402 - lightweight CLI help precedes runtime imports
from tempfile import gettempdir  # noqa: E402 - lightweight CLI help precedes runtime imports
from typing import Any  # noqa: E402 - lightweight CLI help precedes runtime imports

import numpy as np  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch.distributed as dist  # noqa: E402 - lightweight CLI help precedes runtime imports
from safetensors.torch import save_file  # noqa: E402 - lightweight CLI help precedes runtime imports
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch import nn  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch.nn import functional as F  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch.utils.tensorboard import SummaryWriter  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402 - lightweight CLI help precedes runtime imports

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from DeepSEA_lora.config import load_model_entries  # type: ignore
    from DeepSEA_lora.data import DeepSEAIterableDataset, validate_dataset  # type: ignore
    from DeepSEA_lora.evo2_lora_adapter import Evo2LoRALinear  # type: ignore
else:
    from .config import load_model_entries
    from .data import DeepSEAIterableDataset, validate_dataset
    from .evo2_lora_adapter import Evo2LoRALinear


NUM_LABELS = 919


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    _local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Evo2 DeepSEA evaluation requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
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


def load_evo2(model_name: str, model_path: str):
    import evo2
    import evo2.models as evo2_models

    package_dir = Path(evo2.__file__).resolve().parent
    os.chdir(str(package_dir))
    source = package_dir / evo2_models.CONFIG_MAP[model_name]
    safe_config = Path(gettempdir()) / f"{model_name}-deepsea-no-fp8.yml"
    config_text = (
        source.read_text(encoding="utf-8")
        .replace("use_fp8_input_projections: True", "use_fp8_input_projections: False")
        .replace("inference_mode: True", "inference_mode: False")
    )
    safe_config.write_text(config_text, encoding="utf-8")
    evo2_models.CONFIG_MAP[model_name] = str(safe_config)
    loaded = evo2.Evo2(model_name=model_name, local_path=model_path)
    backbone = loaded.model

    # Evo2's checkpoint loader uses inference_mode. Clone parameters/buffers
    # into ordinary autograd tensors before attaching trainable LoRA modules.
    for module in backbone.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is not None:
                module._parameters[name] = nn.Parameter(
                    parameter.detach().clone(), requires_grad=parameter.requires_grad
                )
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.detach().clone()
    return loaded, backbone


def replace_linear(
    root: nn.Module, qualified_name: str, rank: int, alpha: int, dropout: float
) -> None:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent._modules[part]
    leaf = parts[-1]
    module = parent._modules[leaf]
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"EVO2_LORA_UNSUPPORTED_TARGET: {qualified_name}: {type(module)}"
        )
    parent._modules[leaf] = Evo2LoRALinear(
        module, rank=rank, alpha=alpha, dropout=dropout
    )


def inject_lora(backbone: nn.Module, args: argparse.Namespace) -> list[str]:
    targets = [
        name
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear) and ".inner_mha_cls." in name
    ]
    if not targets:
        raise RuntimeError("EVO2_NO_ATTENTION_PROJECTION_TARGETS")
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for target in targets:
        replace_linear(
            backbone, target, args.lora_r, args.lora_alpha, args.lora_dropout
        )
    return targets


class Evo2DeepSEAClassifier(nn.Module):
    def __init__(
        self, backbone: nn.Module, hidden_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            hidden_size, NUM_LABELS, device=device, dtype=torch.bfloat16
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        backbone = self.backbone
        x = backbone.embedding_layer(input_ids)
        x, _ = backbone.stateless_forward(x, padding_mask=None)
        x = x.to(backbone.block_idx_to_device[0])
        if backbone.norm is not None:
            x = backbone.norm(x)
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


class Evo2Collator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        token_rows = []
        for feature in features:
            sequence = str(feature["sequence"])
            token_ids = self.tokenizer.tokenize(sequence.encode("ascii"))[
                : self.max_length
            ]
            token_rows.append(token_ids)
        width = max(len(row) for row in token_rows)
        input_ids = torch.zeros((len(token_rows), width), dtype=torch.long)
        attention_mask = torch.zeros((len(token_rows), width), dtype=torch.long)
        for row_index, row in enumerate(token_rows):
            input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_index, : len(row)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.stack([feature["labels"] for feature in features]),
        }


def make_evo_loader(
    dataset, tokenizer, max_length: int, batch_size: int, num_workers: int
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Evo2Collator(tokenizer, max_length),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    aurocs, auprcs = [], []
    for column in range(labels.shape[1]):
        if np.unique(labels[:, column]).size < 2:
            continue
        aurocs.append(roc_auc_score(labels[:, column], probabilities[:, column]))
        auprcs.append(
            average_precision_score(labels[:, column], probabilities[:, column])
        )
    flat_labels = labels.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "valid_label_count": int(len(aurocs)),
        "micro_auroc": float(roc_auc_score(flat_labels, flat_probabilities)),
        "micro_auprc": float(average_precision_score(flat_labels, flat_probabilities)),
        "bce_loss": float(
            np.mean(
                np.maximum(logits, 0)
                - logits * labels
                + np.log1p(np.exp(-np.abs(logits)))
            )
        ),
    }


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device: torch.device,
    output_dir: Path,
    split: str,
    rank: int,
    world_size: int,
) -> dict[str, float | int]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels").numpy()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels)
    part_path = output_dir / f".evo_eval_{split}.rank{rank}.npz"
    np.savez(
        part_path, logits=np.concatenate(all_logits), labels=np.concatenate(all_labels)
    )
    barrier()
    result = None
    if rank == 0:
        logits = []
        labels = []
        for part_rank in range(world_size):
            part = np.load(output_dir / f".evo_eval_{split}.rank{part_rank}.npz")
            logits.append(part["logits"])
            labels.append(part["labels"])
        result = metrics(np.concatenate(labels), np.concatenate(logits))
        write_json(output_dir / f"evaluation_{split}.json", result)
        for part_rank in range(world_size):
            (output_dir / f".evo_eval_{split}.rank{part_rank}.npz").unlink(
                missing_ok=True
            )
    result = broadcast_object(result, rank)
    barrier()
    return result


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    for name, value in state.items():
        named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))


def clip_gradients(parameters: list[torch.nn.Parameter], max_norm: float) -> float:
    # Evo2 is model-parallel. Reduce norms as host scalars instead of using
    # the single-device foreach implementation in torch.nn.utils.
    squared = sum(
        parameter.grad.detach().float().pow(2).sum().item()
        for parameter in parameters
        if parameter.grad is not None
    )
    norm = squared**0.5
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return norm


def save_adapter(
    model: nn.Module, output_dir: Path, args: argparse.Namespace, targets: list[str]
) -> None:
    state = trainable_state(model)
    save_file(
        state, str(output_dir / "adapter_model.safetensors"), metadata={"format": "pt"}
    )
    write_json(
        output_dir / "adapter_config.json",
        {
            "model_type": "evo2",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "target_modules": targets,
            "modules_to_save": ["classifier"],
        },
    )


def progress(
    output_dir: Path,
    step: int,
    samples: int,
    total: int,
    start: float,
    loss: float,
    lr: float,
) -> None:
    elapsed = max(time.time() - start, 1e-6)
    speed = samples / elapsed
    remaining = max(total - samples, 0)
    eta = remaining / max(speed, 1e-6)
    fraction = min(samples / max(total, 1), 1.0)
    width = 30
    bar = (
        "=" * int(width * fraction)
        + ">"
        + "." * max(0, width - int(width * fraction) - 1)
    )
    eta_time = (datetime.now() + timedelta(seconds=eta)).strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "kind": "progress",
        "step": step,
        "samples_processed": samples,
        "samples_total": total,
        "progress_percent": fraction * 100,
        "throughput_samples_per_sec": speed,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "eta_wallclock": eta_time,
        "loss": loss,
        "learning_rate": lr,
        "peak_gpu_memory_bytes": max(
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ),
    }
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(
        f"[Progress] [{bar}] {fraction * 100:6.2f}% samples={samples:,}/{total:,} "
        f"step={step:,} throughput={speed:,.2f}/s ETA={eta / 3600:.2f}h ({eta_time})",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    set_seed(args.seed)
    entries = {entry.name: entry for entry in load_model_entries(args.config)}
    if args.model_name not in entries:
        raise KeyError(f"Model not found in config.yaml: {args.model_name}")
    entry = entries[args.model_name]
    max_length = args.max_length or entry.max_length or 1024
    dataset_info = validate_dataset(args.dataset_root)
    train_samples = (
        min(args.train_samples, 128) if args.smoke_only else args.train_samples
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.output_root)
        / (
            f"{args.model_name}__evo2-lora-r{args.lora_r}-a{args.lora_alpha}-d{args.lora_dropout:g}"
            f"__len{max_length}__seed{args.seed}__{'smoke' if args.smoke_only else 'screen'}__{datetime.now():%Y%m%d_%H%M%S}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output_dir / "tensorboard")) if rank == 0 else None
    loaded, backbone = load_evo2(args.model_name, entry.url)
    targets = inject_lora(backbone, args)
    hidden_size = int(backbone.config.hidden_size)
    model = Evo2DeepSEAClassifier(backbone, hidden_size, device)
    summary = {
        "model_name": args.model_name,
        "model_path": entry.url,
        "architecture": "StripedHyena/Evo2 model-parallel",
        "parameter_dtype": "bfloat16",
        "world_size": world_size,
        "target_modules": targets,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    summary["trainable_fraction"] = (
        summary["trainable_parameters"] / summary["total_parameters"]
    )
    if rank == 0:
        write_json(output_dir / "model_summary.json", summary)
        write_json(
            output_dir / "run_manifest.json",
            {
                "model": entry.as_dict(),
                "dataset": dataset_info,
                "max_length": max_length,
                "train_samples": train_samples,
                "epochs": args.epochs,
                "seed": args.seed,
                "precision": "bf16",
                "world_size": world_size,
                "model_parallel": False,
                "data_parallel": True,
                "gradient_checkpointing": False,
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "eval_every_steps": args.eval_every_steps,
                "eval_subset_size": args.eval_subset_size,
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
        Path(output_dir / "config_snapshot.yaml").write_text(
            Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
        )
    barrier()

    train_dataset = DeepSEAIterableDataset(
        Path(args.dataset_root) / "train_out.csv",
        Path(args.dataset_root) / "train_labels.npy",
        "train",
        max_samples=train_samples,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        shuffle_buffer_size=10_000,
        seed=args.seed,
    )
    train_loader = make_evo_loader(
        train_dataset,
        loaded.tokenizer,
        max_length,
        args.train_batch_size,
        args.num_workers,
    )
    valid_dataset = DeepSEAIterableDataset(
        Path(args.dataset_root) / "valid_out.csv",
        None,
        "valid",
        max_samples=args.eval_subset_size,
        rank=rank,
        world_size=world_size,
    )
    valid_loader = make_evo_loader(
        valid_dataset, loaded.tokenizer, max_length, args.train_batch_size, 0
    )
    test_dataset = DeepSEAIterableDataset(
        Path(args.dataset_root) / "test_out.csv",
        Path(args.dataset_root) / "test_labels.npy",
        "test",
        rank=rank,
        world_size=world_size,
    )
    test_loader = make_evo_loader(
        test_dataset, loaded.tokenizer, max_length, args.train_batch_size, 0
    )

    wrapped_model: torch.nn.Module = (
        DDP(model, device_ids=[0], find_unused_parameters=False)
        if world_size > 1
        else model
    )
    trainable = [p for p in wrapped_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_updates = (
        math.ceil(
            train_samples / (args.train_batch_size * args.gradient_accumulation_steps)
        )
        * args.epochs
    )
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min((step + 1) / warmup_updates, 1.0) if step < warmup_updates else 1.0
        ),
    )
    best_score = -float("inf")
    best_state = None
    eval_history = []
    global_step = 0
    samples_processed = 0
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = wrapped_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits.float(), labels.float()
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"EVO2_NONFINITE_DEEPSEA_LOSS step={global_step}: {loss.item()}"
                )
            (loss / args.gradient_accumulation_steps).backward()
            samples_processed += int(labels.shape[0]) * world_size
            if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                clip_gradients(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % args.logging_steps == 0 or global_step == 1:
                    if rank == 0:
                        progress(
                            output_dir,
                            global_step,
                            samples_processed,
                            train_samples * args.epochs,
                            start,
                            float(loss.item()),
                            scheduler.get_last_lr()[0],
                        )
                        writer.add_scalar("train/loss", float(loss.item()), global_step)
                if global_step % args.eval_every_steps == 0:
                    valid_metrics = evaluate(
                        wrapped_model,
                        valid_loader,
                        device,
                        output_dir,
                        "valid",
                        rank,
                        world_size,
                    )
                    record = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "samples_processed": samples_processed,
                        "valid": valid_metrics,
                    }
                    eval_history.append(record)
                    if rank == 0:
                        write_json(output_dir / "eval_history.json", eval_history)
                        write_json(output_dir / "evaluation_valid.json", valid_metrics)
                        print(
                            f"[Eval] step={global_step} valid={json.dumps(valid_metrics)}",
                            flush=True,
                        )
                        writer.add_scalar(
                            "valid/macro_auprc",
                            valid_metrics["macro_auprc"],
                            global_step,
                        )
                    if valid_metrics["macro_auprc"] > best_score:
                        best_score = float(valid_metrics["macro_auprc"])
                        best_state = trainable_state(model)
                        if rank == 0:
                            save_adapter(model, output_dir, args, targets)
                    model.train()
            if samples_processed >= train_samples * (epoch + 1):
                break

    valid_metrics = evaluate(
        wrapped_model, valid_loader, device, output_dir, "valid", rank, world_size
    )
    eval_history.append(
        {
            "epoch": args.epochs,
            "global_step": global_step,
            "samples_processed": samples_processed,
            "valid": valid_metrics,
        }
    )
    if valid_metrics["macro_auprc"] > best_score:
        best_score = float(valid_metrics["macro_auprc"])
        best_state = trainable_state(model)
        if rank == 0:
            save_adapter(model, output_dir, args, targets)
    if best_state is not None:
        restore_state(model, best_state)
    if args.smoke_only:
        test_metrics = {"status": "not_run_in_smoke"}
    else:
        test_metrics = evaluate(
            wrapped_model, test_loader, device, output_dir, "test", rank, world_size
        )
    elapsed = time.time() - start
    final = {
        **summary,
        "status": "completed",
        "stage": "smoke" if args.smoke_only else "screen",
        "best_valid_macro_auprc": best_score,
        "history": eval_history,
        "test": test_metrics,
        "train_samples": train_samples,
        "epochs": args.epochs,
        "global_steps": global_step,
        "samples_processed": samples_processed,
        "throughput_samples_per_sec": samples_processed / max(elapsed, 1e-6),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": max(
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ),
    }
    if rank == 0:
        write_json(output_dir / "eval_history.json", eval_history)
        write_json(output_dir / "evaluation_test.json", test_metrics)
        write_json(
            output_dir / "trainer_state.json",
            {
                "global_step": global_step,
                "best_valid_macro_auprc": best_score,
                "elapsed_seconds": elapsed,
            },
        )
        write_json(output_dir / "metrics.json", final)
        writer.close()
        print(
            json.dumps(
                {
                    "status": final["status"],
                    "output_dir": str(output_dir),
                    "test": test_metrics,
                },
                default=str,
            ),
            flush=True,
        )
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
