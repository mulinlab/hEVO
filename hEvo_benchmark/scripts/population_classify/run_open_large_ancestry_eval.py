#!/usr/bin/env python3
"""Run the existing DNAModelsBenchmark ancestry metrics for hEvo-open-large."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ["DNAMODELS_BENCHMARK_ROOT"]).expanduser().resolve()
SUITE = ROOT / "scripts/populaton_classify"
INPUT_DIR = ROOT / "datasets/1kg_popu_classification/jsonl_2000v_1500s_511bp"
BASE_DIR = ROOT / "results/1kg_popu_classification/2000v_1500s_511bp"
MODEL = "hEvo-open-large"
EMB_DIR = BASE_DIR / f"{MODEL}_popu_emb"
RUN_DIR = BASE_DIR / f"{MODEL}_analysis"
GT_PATH = ROOT / "results/ancestry_analysis/1kg_final_for_admixture.5.Q.merged"


def main() -> None:
    sys.path.insert(0, str(SUITE))
    import classify
    import dim_reduction_analysis as dim_reduction

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    expected = {path.stem for path in INPUT_DIR.glob("*.jsonl")}
    observed = {path.stem for path in EMB_DIR.glob("*.npy")}
    if len(expected) != 1500 or observed != expected:
        raise RuntimeError(
            f"Incomplete embeddings: inputs={len(expected)}, outputs={len(observed)}, "
            f"missing={len(expected - observed)}, unexpected={len(observed - expected)}"
        )
    for path in sorted(EMB_DIR.glob("*.npy")):
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (3844,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid embedding: {path} shape={vector.shape}")

    # Keep the suite's model-discovery code limited to this complete model.
    isolated_base = RUN_DIR / "model_input"
    isolated_base.mkdir(exist_ok=True)
    model_link = isolated_base / f"{MODEL}_popu_emb"
    if not model_link.exists():
        model_link.symlink_to(EMB_DIR, target_is_directory=True)
    elif model_link.resolve() != EMB_DIR.resolve():
        raise RuntimeError(f"Unexpected model input link: {model_link}")

    classify.BASE_DIR = str(isolated_base)
    classify.RESULTS_DIR = str(RUN_DIR / "classification_results")
    classify.main()
    classification_path = (
        RUN_DIR / "classification_results/classification_benchmark.csv"
    )
    classification = pd.read_csv(classification_path)
    if (
        classification["model"].tolist() != [MODEL]
        or int(classification.iloc[0]["n_samples"]) != 1500
    ):
        raise RuntimeError(f"Unexpected classification results: {classification_path}")

    dim_reduction.PLOT_DIR = str(RUN_DIR / "plots")
    dim_reduction.JSON_DIR = str(RUN_DIR / "json")
    Path(dim_reduction.PLOT_DIR).mkdir(exist_ok=True)
    Path(dim_reduction.JSON_DIR).mkdir(exist_ok=True)
    dim_reduction.process_model(MODEL, str(EMB_DIR))

    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("OMP_NUM_THREADS", "4")
    subprocess.run(
        [
            sys.executable,
            str(SUITE / "Regression_with_admixture.py"),
            "--base-dir",
            str(isolated_base),
            "--gt-path",
            str(GT_PATH),
            "--report-dir",
            str(RUN_DIR / "report"),
            "--target-model",
            MODEL,
            "--num-workers",
            "1",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            str(SUITE / "benchmark_heldout_admixture_similarity.py"),
            "--base-dir",
            str(BASE_DIR),
            "--gt-path",
            str(GT_PATH),
            "--output-dir",
            str(RUN_DIR / "heldout"),
            "--models",
            MODEL,
            "hEVO-Large",
        ],
        check=True,
        env=env,
    )
    fewshot_paths = sorted((RUN_DIR / "report").glob("overall_model_summary_*.csv"))
    if not fewshot_paths:
        raise RuntimeError("Few-shot summary was not produced")
    fewshot = pd.read_csv(fewshot_paths[-1])
    fewshot_row = fewshot.loc[fewshot["Model"].eq(MODEL)]
    if len(fewshot_row) != 1:
        raise RuntimeError(f"Unexpected few-shot summary: {fewshot_paths[-1]}")
    heldout_path = RUN_DIR / "heldout/heldout_admixture_similarity_by_population.csv"
    heldout = pd.read_csv(heldout_path)
    heldout_overall = heldout.loc[heldout["SuperPopBlock"].eq("ALL")]
    if set(heldout_overall["Model"]) != {MODEL, "hEVO-Large"}:
        raise RuntimeError(f"Unexpected held-out comparison: {heldout_path}")
    unsupervised = json.loads((RUN_DIR / "json" / f"{MODEL}_metrics.json").read_text())
    summary = {
        "model": MODEL,
        "status": "complete",
        "n_samples": 1500,
        "classification": classification.iloc[0].to_dict(),
        "unsupervised": unsupervised,
        "fewshot": fewshot_row.iloc[0].to_dict(),
        "heldout_overall": heldout_overall.to_dict(orient="records"),
    }
    (RUN_DIR / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )
    manifest = {
        "model": MODEL,
        "input_dir": str(INPUT_DIR),
        "embedding_dir": str(EMB_DIR),
        "ground_truth": str(GT_PATH),
        "samples": len(observed),
        "embedding_shape": [3844],
        "extraction": {
            "batch_size": 16,
            "max_length": 512,
            "aggregation": "multi_stats",
            "topk": 500,
        },
        "classification": "5-fold stratified logistic regression, seed 42",
        "fewshot": "10-60 samples per group, 10 repeats, seeds 10-19, Ridge alpha 1",
        "heldout": "60 train and 50 heldout per group, seed 42, Ridge alpha 1",
        "comparison_model": "hEVO-Large",
        "suite": str(SUITE),
    }
    (RUN_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"All hEvo-open-large ancestry evaluations completed: {RUN_DIR}")


if __name__ == "__main__":
    main()
