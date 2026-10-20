from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def build_parser():
    p = argparse.ArgumentParser(description="Recompute Evo2 DeepSEA category metrics")
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
    return p


# Help uses the same parser as execution, before optional model imports.
if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()


import json  # noqa: E402 - lightweight CLI help precedes runtime imports
import math  # noqa: E402 - lightweight CLI help precedes runtime imports

import numpy as np  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch  # noqa: E402 - lightweight CLI help precedes runtime imports
from safetensors.torch import load_file  # noqa: E402 - lightweight CLI help precedes runtime imports
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402 - lightweight CLI help precedes runtime imports

from run_evo2_deepsea import (  # noqa: E402 - lightweight CLI help precedes runtime imports
    Evo2DeepSEAClassifier,
    inject_lora,
    load_evo2,
    make_evo_loader,
)
from config import load_model_entries  # noqa: E402 - lightweight CLI help precedes runtime imports
from data import DeepSEAIterableDataset  # noqa: E402 - lightweight CLI help precedes runtime imports

NUM_LABELS = 919
CATEGORY_RANGES = {"DHS": (0, 125), "TF": (125, 815), "HM": (815, 919)}


def parse_args() -> argparse.Namespace:
    p = build_parser()
    return p.parse_args()


def category_info(path: Path) -> dict[str, object]:
    rows = [
        line.split("\t")
        for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]
        if line.strip()
    ]
    if (
        len(rows) != NUM_LABELS
        or any(row[1] != "DNase" for row in rows[:125])
        or any(not row[1].startswith("H") for row in rows[815:])
    ):
        raise ValueError(
            "DeepSEA metadata does not match the published DHS(125), TF(690), HM(104) layout"
        )
    return {
        "source": str(path),
        "num_labels": NUM_LABELS,
        "categories": {
            k: {"start": a, "end": b, "count": b - a}
            for k, (a, b) in CATEGORY_RANGES.items()
        },
        "boundary_features": [rows[124][1], rows[125][1], rows[814][1], rows[815][1]],
    }


def metric(y: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    p = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
    roc, pr = [], []
    for col in range(y.shape[1]):
        if np.unique(y[:, col]).size < 2:
            continue
        roc.append(float(roc_auc_score(y[:, col], p[:, col])))
        pr.append(float(average_precision_score(y[:, col], p[:, col])))
    flat_y, flat_p = y.reshape(-1), p.reshape(-1)
    return {
        "macro_auroc": float(np.mean(roc)) if roc else math.nan,
        "macro_auprc": float(np.mean(pr)) if pr else math.nan,
        "valid_label_count": len(roc),
        "micro_auroc": float(roc_auc_score(flat_y, flat_p)),
        "micro_auprc": float(average_precision_score(flat_y, flat_p)),
        "bce_loss": float(
            np.mean(
                np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
            )
        ),
    }


def main() -> None:
    cli = parse_args()
    run_dir = Path(cli.run_dir).resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    model_name = str(manifest["model"]["name"])
    entry = next(x for x in load_model_entries(cli.config) if x.name == model_name)
    if not torch.cuda.is_available():
        raise RuntimeError("Evo2 category evaluation requires CUDA")
    device = torch.device("cuda", 0)
    loaded, backbone = load_evo2(model_name, entry.url)
    adapter_cfg = json.loads(
        (run_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    lora_args = type(
        "Args",
        (),
        {
            "lora_r": int(adapter_cfg.get("r", 8)),
            "lora_alpha": int(
                adapter_cfg.get("alpha", adapter_cfg.get("lora_alpha", 16))
            ),
            "lora_dropout": float(
                adapter_cfg.get("dropout", adapter_cfg.get("lora_dropout", 0.05))
            ),
        },
    )()
    targets = inject_lora(backbone, lora_args)
    model = Evo2DeepSEAClassifier(backbone, int(backbone.config.hidden_size), device)
    state = load_file(str(run_dir / "adapter_model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Evo2 adapter keys: {unexpected[:8]}")
    if missing:
        print(f"[adapter] missing base/non-trainable keys: {missing[:8]}", flush=True)
    model.eval()
    dataset = DeepSEAIterableDataset(
        Path(cli.dataset_root) / "test_out.csv",
        Path(cli.dataset_root) / "test_labels.npy",
        "test",
    )
    batch_size = max(1, int(manifest.get("train_batch_size", 1)))
    loader = make_evo_loader(
        dataset,
        loaded.tokenizer,
        int(manifest.get("max_length", entry.max_length)),
        batch_size,
        0,
    )
    logits_list, labels_list = [], []
    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            labels = batch.pop("labels").numpy().astype(np.float32, copy=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            logits_list.append(pred.float().cpu().numpy())
            labels_list.append(labels)
            if index % 1000 == 0:
                print(
                    f"[Progress] test_samples={sum(x.shape[0] for x in labels_list):,}",
                    flush=True,
                )
    y, z = np.concatenate(labels_list), np.concatenate(logits_list)
    result = {
        "status": "completed",
        "model_name": model_name,
        "run_dir": str(run_dir),
        "test_samples": int(y.shape[0]),
        "num_labels": int(y.shape[1]),
        "category_definition": category_info(Path(cli.metadata)),
        "overall": metric(y, z),
        "categories": {
            k: metric(y[:, a:b], z[:, a:b]) for k, (a, b) in CATEGORY_RANGES.items()
        },
        "target_modules": targets,
    }
    (run_dir / "category_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
