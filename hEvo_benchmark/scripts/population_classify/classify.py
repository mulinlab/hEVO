#!/usr/bin/env python
"""Run cross-validated super-population classification from sample embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from embedding_loader import (
    available_model_names,
    format_fused_model_note,
    load_embedding_rows_for_model,
)


def load_model_dataset(base_dir: Path, model_name: str):
    rows = load_embedding_rows_for_model(str(base_dir), model_name)
    if not rows:
        return None, None

    print(f"  [load] {format_fused_model_note(model_name)} -> {len(rows)} samples")
    x = np.vstack([vec for _, _, vec in rows]).astype(np.float32)
    y = np.array([pop_label for _, pop_label, _ in rows])
    return x, y


def run_classification(x: np.ndarray, y: np.ndarray, model_name: str) -> dict | None:
    if len(set(y)) < 2:
        print(f"  [skip] {model_name}: not enough classes")
        return None

    y_encoded = LabelEncoder().fit_transform(y)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    # Fit scaling independently inside each training fold. Fitting it once on
    # all samples leaks held-out fold distribution information into CV.
    pipeline = make_pipeline(StandardScaler(), clf)
    scoring = {
        "accuracy": "accuracy",
        "mcc": make_scorer(matthews_corrcoef),
        "f1_macro": "f1_macro",
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, x, y_encoded, cv=skf, scoring=scoring)

    return {
        "model": model_name,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(len(set(y))),
        "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "mcc_mean": float(np.mean(cv_results["test_mcc"])),
        "mcc_std": float(np.std(cv_results["test_mcc"])),
        "f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir or args.base_dir / "classification_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_model_metrics = []

    for model_name in available_model_names(str(args.base_dir)):
        try:
            x, y = load_model_dataset(args.base_dir, model_name)
        except Exception as exc:
            print(f"  [skip] {model_name}: {exc}")
            continue
        if x is None or len(x) <= 10:
            print(f"  [skip] {model_name}: too few samples")
            continue

        print(f"  [eval] {format_fused_model_note(model_name)}")
        result = run_classification(x, y, model_name)
        if result is not None:
            all_model_metrics.append(result)
            print(
                f"  [result] Acc: {result['accuracy_mean']:.4f} | "
                f"MCC: {result['mcc_mean']:.4f}"
            )

    df_results = pd.DataFrame(all_model_metrics)
    if not df_results.empty:
        df_results = df_results.sort_values(
            ["accuracy_mean", "mcc_mean", "model"],
            ascending=[False, False, True],
        )
    output_csv = results_dir / "classification_benchmark.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"Finished. Results written to {output_csv}")


if __name__ == "__main__":
    main()
