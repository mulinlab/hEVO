from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(
        description="Recompute DeepSEA category metrics from a LoRA run"
    )
    p.add_argument("--run_dir", required=True)
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs/models.yaml"),
    )
    p.add_argument(
        "--dataset_root",
        default=str(
            Path(
                os.environ.get(
                    "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
                )
            )
            / "datasets/DeepSEA"
        ),
    )
    p.add_argument(
        "--metadata",
        default=str(
            Path(
                os.environ.get(
                    "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
                )
            )
            / "datasets/DeepSEA/deepsea_metadata.tsv"
        ),
    )
    p.add_argument("--output", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Index of this independent evaluation shard",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of independent evaluation shards",
    )
    p.add_argument(
        "--expected_shards",
        type=int,
        default=None,
        help="Require this contiguous set of prediction rank files when aggregating",
    )
    p.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Aggregate existing .category_eval.rank*.npz files without model inference",
    )
    return p


# Help uses the same parser as execution, before optional model imports.
if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()


from concurrent.futures import ThreadPoolExecutor  # noqa: E402 - lightweight CLI help precedes runtime imports
import json  # noqa: E402 - lightweight CLI help precedes runtime imports
import math  # noqa: E402 - lightweight CLI help precedes runtime imports

import numpy as np  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch.distributed as dist  # noqa: E402 - lightweight CLI help precedes runtime imports
from safetensors.torch import load_file  # noqa: E402 - lightweight CLI help precedes runtime imports
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402 - lightweight CLI help precedes runtime imports

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from DeepSEA_lora.adapters import load_model_and_tokenizer  # noqa: E402
from DeepSEA_lora.config import load_model_entries  # noqa: E402
from DeepSEA_lora.data import DeepSEAIterableDataset, make_loader  # noqa: E402
from DeepSEA_lora.run_deepsea_lora import configure_peft  # noqa: E402


NUM_LABELS = 919
CATEGORY_RANGES = {"DHS": (0, 125), "TF": (125, 815), "HM": (815, 919)}


def args() -> argparse.Namespace:
    p = build_parser()
    return p.parse_args()


def setup_dist() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        torch.cuda.set_device(local_rank)
        return rank, world, torch.device("cuda", local_rank)
    return rank, world, torch.device("cpu")


def load_category_metadata(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or len(lines) - 1 != NUM_LABELS:
        raise ValueError(
            f"Expected 919 metadata rows, got {max(0, len(lines) - 1)}: {path}"
        )
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    if len(rows) != NUM_LABELS:
        raise ValueError(f"Expected 919 non-empty metadata rows, got {len(rows)}")
    # This is the published DeepSEA metadata layout. The second column is the
    # feature name, not the category; the category is defined by the row block.
    if any(row[1] != "DNase" for row in rows[:125]):
        raise ValueError(
            "Metadata first 125 rows are not all DNase; refusing guessed category boundaries"
        )
    if any(row[1].startswith("H") is False for row in rows[815:]):
        raise ValueError(
            "Metadata last 104 rows are not histone marks; refusing guessed category boundaries"
        )
    return {
        "source": str(path),
        "num_labels": NUM_LABELS,
        "categories": {
            name: {"start": start, "end": end, "count": end - start}
            for name, (start, end) in CATEGORY_RANGES.items()
        },
        "first_features": [rows[i][1] for i in range(3)],
        "boundary_features": [rows[124][1], rows[125][1], rows[814][1], rows[815][1]],
        "last_features": [rows[i][1] for i in range(916, 919)],
    }


def _column_metrics(args: tuple[np.ndarray, np.ndarray]) -> tuple[float, float] | None:
    target, column_logits = args
    if np.unique(target).size < 2:
        return None
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(column_logits.astype(np.float64), -40, 40))
    )
    return (
        float(roc_auc_score(target, probabilities)),
        float(average_precision_score(target, probabilities)),
    )


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    column_args = ((labels[:, col], logits[:, col]) for col in range(labels.shape[1]))
    workers = min(16, max(1, labels.shape[1]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        column_results = list(pool.map(_column_metrics, column_args))
    valid_results = [item for item in column_results if item is not None]
    roc_values = [item[0] for item in valid_results]
    pr_values = [item[1] for item in valid_results]
    flat_y = labels.reshape(-1)
    flat_p = probabilities.reshape(-1)
    return {
        "macro_auroc": float(np.mean(roc_values)) if roc_values else math.nan,
        "macro_auprc": float(np.mean(pr_values)) if pr_values else math.nan,
        "valid_label_count": len(roc_values),
        "micro_auroc": float(roc_auc_score(flat_y, flat_p))
        if np.unique(flat_y).size > 1
        else math.nan,
        "micro_auprc": float(average_precision_score(flat_y, flat_p)),
        "bce_loss": float(
            np.mean(
                np.maximum(logits, 0)
                - logits * labels
                + np.log1p(np.exp(-np.abs(logits)))
            )
        ),
    }


def write_category_metrics(
    run_dir: Path,
    category_info: dict[str, object],
    part_paths: list[Path],
    output: Path,
) -> dict[str, object]:
    all_labels = []
    all_logits = []
    for part_path in part_paths:
        with np.load(part_path) as item:
            all_labels.append(item["labels"])
            all_logits.append(item["logits"])
    labels = np.concatenate(all_labels, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    if (
        labels.ndim != 2
        or logits.shape != labels.shape
        or labels.shape[1] != NUM_LABELS
    ):
        raise ValueError(
            f"Invalid category prediction shape: labels={labels.shape}, logits={logits.shape}; "
            f"expected (*, {NUM_LABELS})"
        )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    result: dict[str, object] = {
        "status": "completed",
        "model_name": manifest["model"]["name"],
        "run_dir": str(run_dir),
        "test_samples": int(labels.shape[0]),
        "num_labels": int(labels.shape[1]),
        "category_definition": category_info,
        "overall": metrics(labels, logits),
        "categories": {
            name: metrics(labels[:, start:end], logits[:, start:end])
            for name, (start, end) in CATEGORY_RANGES.items()
        },
    }
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    cli = args()
    run_dir = Path(cli.run_dir).resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    model_info = manifest["model"]
    if cli.aggregate_only:
        category_info = load_category_metadata(Path(cli.metadata))
        part_paths = sorted(run_dir.glob(".category_eval.rank*.npz"))
        if not part_paths:
            raise FileNotFoundError(
                f"No existing category prediction files found in {run_dir}"
            )
        if cli.expected_shards is not None:
            expected = {
                f".category_eval.rank{rank}.npz" for rank in range(cli.expected_shards)
            }
            actual = {path.name for path in part_paths}
            if actual != expected:
                missing, unexpected = (
                    sorted(expected - actual),
                    sorted(actual - expected),
                )
                raise ValueError(
                    f"Prediction shard mismatch; missing={missing}, unexpected={unexpected}"
                )
        result = write_category_metrics(
            run_dir,
            category_info,
            part_paths,
            Path(cli.output) if cli.output else run_dir / "category_metrics.json",
        )
        for part_path in part_paths:
            part_path.unlink(missing_ok=True)
        print(json.dumps(result, indent=2), flush=True)
        return
    if cli.num_shards < 1 or not 0 <= cli.shard_index < cli.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    model_name = str(model_info["name"])
    entry = next(x for x in load_model_entries(cli.config) if x.name == model_name)
    rank, world, device = setup_dist()
    global_rank = cli.shard_index * world + rank
    global_world = cli.num_shards * world
    if rank == 0:
        category_info = load_category_metadata(Path(cli.metadata))
    else:
        category_info = None
    if world > 1:
        payload = [category_info]
        dist.broadcast_object_list(payload, src=0)
        category_info = payload[0]

    backend = manifest.get("attention_backend", "eager")
    if backend == "not_applicable":
        backend = "eager"
    model, tokenizer, _config, _family, targets = load_model_and_tokenizer(
        model_name, entry.url, attention_backend=backend
    )
    adapter_config = json.loads(
        (run_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    peft_args = type(
        "PeftArgs",
        (),
        {
            "lora_r": int(adapter_config.get("r", 8)),
            "lora_alpha": int(
                adapter_config.get("lora_alpha", adapter_config.get("alpha", 16))
            ),
            "lora_dropout": float(
                adapter_config.get("lora_dropout", adapter_config.get("dropout", 0.05))
            ),
        },
    )()
    # Reuse the training suite's bnb-disabling PEFT setup, then restore the
    # exact saved trainable state without importing PeftModel.from_pretrained.
    model = configure_peft(model, targets, peft_args)
    raw_state = load_file(str(run_dir / "adapter_model.safetensors"), device="cpu")
    state = {}
    for key, value in raw_state.items():
        # PEFT versions used by the training runs serialized the default
        # adapter without the newer ".default" component in LoRA keys.
        key = key.replace(".lora_A.weight", ".lora_A.default.weight")
        key = key.replace(".lora_B.weight", ".lora_B.default.weight")
        if key.endswith("base_model.model.classifier.weight"):
            key = key.replace(
                "classifier.weight", "classifier.modules_to_save.default.weight"
            )
        elif key.endswith("base_model.model.classifier.bias"):
            key = key.replace(
                "classifier.bias", "classifier.modules_to_save.default.bias"
            )
        state[key] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys in {run_dir}: {unexpected[:8]}")
    if missing:
        print(f"[adapter] missing non-trainable/base keys: {missing[:8]}", flush=True)
    precision = str(manifest.get("precision", "fp16"))
    # NTv3's convolutional stem consumes a continuous FP32 representation
    # produced inside its custom remote-code forward path.  Keeping the
    # checkpoint in FP16 makes that stem receive FP32 activations with FP16
    # weights and fails in conv1d.  NTv3 is small, so use an explicit FP32
    # evaluation path without changing the saved adapter or metric protocol.
    if _family == "ntv3":
        precision = "fp32"
    if device.type == "cuda":
        if precision == "bf16":
            model = model.to(device=device, dtype=torch.bfloat16)
        elif precision == "fp32":
            model = model.to(device=device, dtype=torch.float32)
        else:
            model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    model.eval()

    eval_batch = cli.batch_size or int(manifest.get("per_device_eval_batch_size", 1))
    dataset = DeepSEAIterableDataset(
        Path(cli.dataset_root) / "test_out.csv",
        Path(cli.dataset_root) / "test_labels.npy",
        "test",
        rank=global_rank,
        world_size=global_world,
        shuffle=False,
    )
    loader = make_loader(
        dataset,
        tokenizer,
        int(manifest.get("max_length", entry.max_length)),
        eval_batch,
        False,
        cli.num_workers,
    )
    local_logits: list[np.ndarray] = []
    local_labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").numpy().astype(np.float32, copy=False)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            enabled = device.type == "cuda" and precision != "fp32"
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
                logits = model(batch).float().cpu().numpy()
            local_labels.append(labels)
            local_logits.append(logits.astype(np.float32, copy=False))

    out = Path(cli.output) if cli.output else run_dir / "category_metrics.json"
    part = run_dir / f".category_eval.rank{global_rank}.npz"
    np.savez(
        part, labels=np.concatenate(local_labels), logits=np.concatenate(local_logits)
    )
    if world > 1:
        dist.barrier()
    if rank == 0:
        if cli.num_shards == 1:
            result = write_category_metrics(
                run_dir,
                category_info,
                [
                    run_dir / f".category_eval.rank{part_rank}.npz"
                    for part_rank in range(global_world)
                ],
                out,
            )
            for part_rank in range(global_world):
                (run_dir / f".category_eval.rank{part_rank}.npz").unlink(
                    missing_ok=True
                )
            print(json.dumps(result, indent=2), flush=True)
        else:
            print(
                f"Completed category prediction shard {cli.shard_index + 1}/{cli.num_shards}",
                flush=True,
            )
    if world > 1:
        # Do not add a second NCCL barrier here. Rank 0 computes 885 per-label
        # metrics on CPU and can take longer than NCCL's watchdog timeout;
        # non-zero ranks can safely exit after the first barrier because all
        # prediction files have already been written.
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
