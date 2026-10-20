"""Replay real caches through isolated old/new evaluator functions.

AST loading deliberately excludes optional model imports, not evaluator logic.
This checks training/metrics equivalence; it does not certify cache provenance
or fresh model inference. Inputs are read-only and outputs are kept separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import types
from pathlib import Path


def evaluator_source(root: Path, filename: str, *, required: bool = True) -> Path:
    """Resolve both historical flat snapshots and the organized evaluator."""
    for directory in (
        root,
        root / "core",
        root / "adapters",
        root / "runtime",
        root / "compat",
    ):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError(f"Evaluator source {filename} not found under {root}")
    return root / filename


def load_definitions(path, namespace, names=None):
    tree = ast.parse(Path(path).read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and (names is None or n.name in names)
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="scripts/evaluate directory"
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["probe", "variant"], required=True)
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="Fail unless metrics and every saved array match a baseline replay exactly",
    )
    args = parser.parse_args()
    import h5py
    import numpy as np
    import torch
    import tqdm
    import sklearn.metrics as metrics
    from sklearn.preprocessing import label_binarize
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    torch.set_num_threads(2)
    namespace = dict(
        np=np,
        torch=torch,
        tqdm=tqdm,
        nn=nn,
        Dataset=Dataset,
        DataLoader=DataLoader,
        label_binarize=label_binarize,
        SEED=42,
        os=os,
        json=json,
        h5py=h5py,
    )
    namespace.update(
        {
            name: getattr(metrics, name)
            for name in [
                "accuracy_score",
                "f1_score",
                "matthews_corrcoef",
                "roc_auc_score",
                "average_precision_score",
            ]
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.kind == "probe":
        common = load_definitions(
            evaluator_source(args.source, "probe_common.py"), dict(namespace)
        )
        namespace["_probe_common"] = types.SimpleNamespace(**common)
        names = {
            "EmbeddingDataset",
            "collate_emb_fn",
            "FlexibleProbe",
            "eval_probe_on_embeddings",
            "train_probe_on_embeddings",
            "bootstrap_ci",
        }
        for protocol in ["standard", "pareto"]:
            source = args.source / (
                "probe_single_seq_input.py"
                if protocol == "standard"
                else "probe_single_seq_input_Pareto.py"
            )
            if protocol == "pareto":
                try:
                    source = evaluator_source(args.source, "probe_pareto_engine.py")
                except FileNotFoundError:
                    source = evaluator_source(
                        args.source, "probe_single_seq_input_Pareto.py"
                    )
            env = dict(namespace)
            # Shared definitions may replace identical classes in the new engine.
            env.update({k: v for k, v in common.items() if k in names})
            load_definitions(source, env, names)
            for task in ["enhancers", "splice_sites_all"]:
                path = (
                    args.benchmark
                    / "datasets_emb_H5"
                    / "NT_Pareto"
                    / task
                    / f"DNABERT2_{task}.h5"
                )
                with h5py.File(path) as handle:
                    arrays = {
                        f"{split}_{name}": handle[split][name][:]
                        for split in ["train", "val"]
                        for name in ["embeddings", "labels"]
                    }
                random.seed(40)
                np.random.seed(40)
                torch.manual_seed(40)
                model = env["FlexibleProbe"](
                    arrays["train_embeddings"].shape[1],
                    len(np.unique(arrays["train_labels"])),
                )
                loaders = []
                for split in ["train", "val"]:
                    data = env["EmbeddingDataset"](
                        torch.from_numpy(arrays[f"{split}_embeddings"]),
                        torch.from_numpy(arrays[f"{split}_labels"]),
                    )
                    loaders.append(
                        DataLoader(
                            data,
                            batch_size=64,
                            shuffle=split == "train",
                            collate_fn=env["collate_emb_fn"],
                        )
                    )
                model = env["train_probe_on_embeddings"](
                    model,
                    *loaders,
                    epochs=10,
                    lr=0.001,
                    weight_decay=0.0,
                    device=torch.device("cpu"),
                )
                point, labels, predictions, probabilities = env[
                    "eval_probe_on_embeddings"
                ](model, loaders[1], device=torch.device("cpu"), return_samples=True)
                bootstrap = env["bootstrap_ci"](
                    labels,
                    predictions,
                    probabilities,
                    n_bootstrap=1000,
                    random_state=40,
                )
                key = f"{protocol}_{task}"
                np.savez(
                    args.output / f"{key}.npz",
                    labels=labels,
                    predictions=predictions,
                    probabilities=probabilities,
                    **{
                        f"weight_{k}": v.detach().numpy()
                        for k, v in model.state_dict().items()
                    },
                )
                results[key] = dict(
                    point=point,
                    bootstrap=bootstrap,
                    train_count=len(arrays["train_labels"]),
                    test_count=len(labels),
                    cache=str(path),
                    cache_array_sha256=hashlib.sha256(
                        b"".join(v.tobytes() for v in arrays.values())
                    ).hexdigest(),
                    cache_provenance="legacy cache without source metadata; numerical replay only",
                )
    else:
        names = {
            "load_llrs_from_h5",
            "_llrs_to_scores",
            "_bootstrap_ci_for_metric",
            "bootstrap_ci_for_auroc",
            "bootstrap_ci_for_auprc",
        }
        env = load_definitions(
            args.source / "zeroshot_paried_seq_input.py", namespace, names
        )
        for task, mode in [
            ("Causal_Variants", "neg_llr"),
            ("Rare_Variants", "neg_llr"),
            ("MPRA", "abs_llr"),
        ]:
            path = (
                args.benchmark
                / "datasets_H5"
                / "hEvo-large-variant-moe-20260918"
                / task
                / task
                / "hEvo-large-variant-moe-20260918.hdf5"
            )
            llrs, labels = env["load_llrs_from_h5"](str(path))
            raw, y = llrs.numpy(), labels.numpy()
            scores = env["_llrs_to_scores"](raw, mode)
            with h5py.File(path) as handle:
                metadata = {k: str(v) for k, v in handle.attrs.items()}
                details = handle["details"][:]
            np.savez(args.output / f"{task}.npz", llrs=raw, labels=y, scores=scores)
            results[task] = dict(
                count=len(y),
                auroc=float(metrics.roc_auc_score(y, scores)),
                auprc=float(metrics.average_precision_score(y, scores)),
                auroc_bootstrap=env["bootstrap_ci_for_auroc"](raw, y, score_mode=mode),
                auprc_bootstrap=env["bootstrap_ci_for_auprc"](raw, y, score_mode=mode),
                cache=str(path),
                cache_metadata=metadata,
                score_mode=mode,
                details_sha256=hashlib.sha256(
                    b"\n".join(
                        x if isinstance(x, bytes) else str(x).encode() for x in details
                    )
                ).hexdigest(),
                cache_provenance="retained historical checkpoint cache; numerical replay only",
            )
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    if args.compare_with is not None:
        baseline = json.loads((args.compare_with / "metrics.json").read_text())
        if json.loads(json.dumps(results)) != baseline:
            raise AssertionError("Replay metric records differ from baseline")
        for path in args.compare_with.glob("*.npz"):
            with np.load(path) as old, np.load(args.output / path.name) as new:
                if set(old.files) != set(new.files):
                    raise AssertionError(f"Replay array fields differ: {path.name}")
                for field in old.files:
                    np.testing.assert_array_equal(
                        old[field], new[field], err_msg=f"{path.name}/{field}"
                    )
        print("PASS: metrics and all replay arrays exactly match baseline")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
