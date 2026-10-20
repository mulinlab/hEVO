#!/usr/bin/env python3
"""Validate GPN-MSA ancestry vectors and update S33-S35 plus Fig. 4 a-c."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler


EXPECTED_SAMPLES = 1500
EXPECTED_EMBEDDING_WIDTH = 2308
EXPECTED_SHOTS = list(range(10, 61))
MODEL_NAME = "GPN-MSA"
TABLE_PREFIXES = ("S33", "S34", "S35")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_path(tables_dir: Path, prefix: str) -> Path:
    matches = list(tables_dir.glob(f"{prefix}_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {prefix}_*.csv under {tables_dir}, found {matches}"
        )
    return matches[0]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def compute_tsne_metrics(embedding_paths: list[Path]) -> tuple[float, float]:
    labels = np.asarray([path.stem.split("_", 1)[0] for path in embedding_paths])
    vectors = np.stack(
        [np.load(path, allow_pickle=False) for path in embedding_paths]
    ).astype(np.float32)
    scaled = StandardScaler().fit_transform(vectors)
    reduced = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
        metric="cosine",
    ).fit_transform(scaled)
    return float(silhouette_score(reduced, labels)), float(
        calinski_harabasz_score(reduced, labels)
    )


def validate_embeddings(run_root: Path, preflight: dict) -> list[Path]:
    embedding_dir = run_root / "results" / "GPN-MSA_popu_emb"
    manifest_dir = run_root / "results" / "manifests"
    samples = json.loads(
        (run_root / "config" / "sample_manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        row["file_name"].removesuffix(".jsonl") + ".npy": row for row in samples
    }
    paths = sorted(embedding_dir.glob("*.npy"))
    if len(paths) != EXPECTED_SAMPLES or {path.name for path in paths} != set(expected):
        raise RuntimeError(
            f"Expected exactly {EXPECTED_SAMPLES} GPN-MSA vectors; found {len(paths)} "
            f"(missing={len(set(expected) - {p.name for p in paths})}, "
            f"extra={len({p.name for p in paths} - set(expected))})"
        )

    unique_vectors: set[bytes] = set()
    for path in paths:
        row = expected[path.name]
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (EXPECTED_EMBEDDING_WIDTH,) or not np.isfinite(vector).all():
            raise RuntimeError(f"Invalid GPN-MSA vector at {path}: {vector.shape}")
        unique_vectors.add(np.asarray(vector, dtype=np.float32).tobytes())
        manifest_path = manifest_dir / f"{path.stem}.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing extraction manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_fields = {
            "status": "complete",
            "model": MODEL_NAME,
            "sample_id": row["sample_id"],
            "population": row["population"],
            "input_file": row["input_file"],
            "input_sha256": row["input_sha256"],
            "input_rows": 10000,
            "locus_roster_sha256": preflight["locus_roster_sha256"],
            "checkpoint_weights_sha256": preflight["checkpoint"]["weights_sha256"],
            "msa_cache_forward_sha256": preflight["msa_window_cache"]["forward_sha256"],
            "msa_cache_reverse_sha256": preflight["msa_window_cache"]["reverse_sha256"],
            "embedding_shape": [EXPECTED_EMBEDDING_WIDTH],
        }
        differences = [
            key for key, value in expected_fields.items() if manifest.get(key) != value
        ]
        if differences:
            raise RuntimeError(
                f"Manifest mismatch for {path.name}: {', '.join(differences)}"
            )
        if manifest.get("output_sha256") != sha256_file(path):
            raise RuntimeError(f"Output checksum mismatch for {path}")
    if len(unique_vectors) < 2:
        raise RuntimeError(
            "All sample embeddings are identical; genotype-specific signal is missing"
        )
    return paths


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def refresh_fig4(
    repo_root: Path, tables_dir: Path, python: Path, env: dict[str, str]
) -> dict[str, dict[str, str]]:
    scripts = repo_root / "figures" / "scripts" / "Fig4"
    output_dir = repo_root / "figures" / "Fig4"
    commands = [
        [
            python,
            scripts / "a_unsupervised.py",
            "--tables-dir",
            tables_dir,
            "--output-dir",
            output_dir,
        ],
        [
            python,
            scripts / "b_mcc.py",
            "--tables-dir",
            tables_dir,
            "--output-dir",
            output_dir,
        ],
        [
            python,
            scripts / "c_fewshot.py",
            "--tables-dir",
            tables_dir,
            "--output-dir",
            output_dir,
        ],
        [
            python,
            scripts / "A4_integrated.py",
            "--tables-dir",
            tables_dir,
            "--output-dir",
            output_dir,
        ],
    ]
    for command in commands:
        run_command([str(value) for value in command], env)

    result: dict[str, dict[str, str]] = {}
    for stem in ("a_unsupervised", "b_mcc", "c_fewshot", "A4_integrated"):
        files = {}
        for extension in ("pdf", "svg", "png"):
            path = output_dir / f"{stem}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Figure output missing or empty: {path}")
            files[path.name] = sha256_file(path)
        result[stem] = files
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--regression-script", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    tables_dir = args.tables_dir.expanduser().resolve()
    config_dir = run_root / "config"
    preflight = json.loads((config_dir / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "preflight_complete":
        raise RuntimeError("GPN-MSA preflight is not complete")

    embeddings = validate_embeddings(run_root, preflight)
    s33_path, s34_path, s35_path = [
        table_path(tables_dir, prefix) for prefix in TABLE_PREFIXES
    ]
    s33_old = pd.read_csv(s33_path)
    s34_old = pd.read_csv(s34_path)
    s35_old = pd.read_csv(s35_path)
    baseline_models = set(s34_old["model"].astype(str)) - {MODEL_NAME}
    if len(baseline_models) != 33:
        raise RuntimeError(
            f"Expected 33 existing models before adding GPN-MSA, found {len(baseline_models)}"
        )
    if set(s33_old["model"].astype(str)) - {MODEL_NAME} != baseline_models:
        raise RuntimeError("S33 and S34 model rosters differ")
    if set(s35_old["model"].astype(str)) - {MODEL_NAME} != baseline_models:
        raise RuntimeError("S35 and S34 model rosters differ")

    sys.path.insert(0, str(repo_root / "figures" / "scripts"))
    from common.model_names import canonical_model_name
    from Fig4._backend import BACKEND

    def fig4_frame(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["model_display"] = prepared["model"].map(canonical_model_name)
        prepared["family"] = prepared["model"].map(BACKEND.model_family)
        prepared["algorithm_identity"] = prepared["model"].map(
            BACKEND.algorithm_identity
        )
        return prepared

    silhouette, ch_index = compute_tsne_metrics(embeddings)
    s33 = s33_old[s33_old["model"].astype(str) != MODEL_NAME].copy()
    s33.loc[len(s33)] = {
        "model": MODEL_NAME,
        "model_display": canonical_model_name(MODEL_NAME),
        "tsne_silhouette": silhouette,
        "tsne_ch_index": ch_index,
        "rank_score": 0.0,
    }
    s33["model_display"] = s33["model"].map(canonical_model_name)
    s33["rank_score"] = s33["tsne_silhouette"] + 0.15 * np.log10(
        s33["tsne_ch_index"].astype(float)
    )
    s33 = s33.sort_values(["rank_score", "model"], ascending=[False, True]).reset_index(
        drop=True
    )

    stage_dir = run_root / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_link = stage_dir / "GPN-MSA_popu_emb"
    expected_target = (run_root / "results" / "GPN-MSA_popu_emb").resolve()
    if model_link.is_symlink():
        if model_link.resolve() != expected_target:
            raise RuntimeError(f"Unexpected stage symlink target: {model_link}")
    elif model_link.exists():
        raise RuntimeError(
            f"Refusing to replace non-symlink staging path: {model_link}"
        )
    else:
        model_link.symlink_to(expected_target, target_is_directory=True)

    eval_dir = run_root / "results" / "classification"
    eval_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    task_tmp = run_root / "tmp" / "evaluation"
    task_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(task_tmp)
    env["TEMP"] = str(task_tmp)
    env["TMP"] = str(task_tmp)
    env["MPLBACKEND"] = "Agg"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    classifier = repo_root / "scripts" / "population_classify" / "classify.py"
    run_command(
        [
            str(args.python),
            str(classifier),
            "--base-dir",
            str(stage_dir),
            "--results-dir",
            str(eval_dir),
        ],
        env,
    )
    classification_path = eval_dir / "classification_benchmark.csv"
    classification = pd.read_csv(classification_path)
    if set(classification["model"].astype(str)) != {MODEL_NAME}:
        raise RuntimeError("GPN-MSA classification produced an unexpected model roster")
    cls_row = classification.iloc[0]
    if int(cls_row["n_samples"]) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"GPN-MSA classification used {cls_row['n_samples']} samples"
        )
    s34 = s34_old[s34_old["model"].astype(str) != MODEL_NAME].copy()
    s34.loc[len(s34)] = {
        "model": MODEL_NAME,
        "model_display": canonical_model_name(MODEL_NAME),
        "classification_accuracy": float(cls_row["accuracy_mean"]),
        "classification_mcc": float(cls_row["mcc_mean"]),
        "classification_samples": EXPECTED_SAMPLES,
    }
    s34["model_display"] = s34["model"].map(canonical_model_name)
    s34 = s34.sort_values(
        ["classification_accuracy", "classification_mcc", "model"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    gt_path = (
        benchmark_root
        / "results"
        / "ancestry_analysis"
        / "1kg_final_for_admixture.5.Q.merged"
    )
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    regression = (
        args.regression_script
        or benchmark_root
        / "scripts"
        / "population_classify"
        / "Regression_with_admixture.py"
    )
    regression = regression.expanduser().resolve()
    if not regression.is_file():
        raise FileNotFoundError(regression)
    fewshot_dir = run_root / "results" / "fewshot"
    fewshot_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(args.python),
            str(regression),
            "--base-dir",
            str(stage_dir),
            "--gt-path",
            str(gt_path),
            "--report-dir",
            str(fewshot_dir),
            "--min-shots",
            "10",
            "--max-shots",
            "60",
            "--num-repeats",
            "10",
            "--seed-start",
            "10",
            "--alpha",
            "1",
            "--target-model",
            MODEL_NAME,
            "--num-workers",
            "1",
        ],
        env,
    )
    reports = sorted(
        fewshot_dir.glob("fewshot_all_runs_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not reports:
        raise RuntimeError("GPN-MSA few-shot report was not produced")
    all_runs = pd.read_csv(reports[-1])
    all_runs = all_runs[all_runs["Model"].astype(str) == MODEL_NAME]
    expected_runs = {(shot, seed) for shot in EXPECTED_SHOTS for seed in range(10, 20)}
    observed_runs = set(zip(all_runs["Shot"].astype(int), all_runs["Seed"].astype(int)))
    if observed_runs != expected_runs or len(all_runs) != len(expected_runs):
        raise RuntimeError(
            f"GPN-MSA few-shot run matrix incomplete: expected {len(expected_runs)}, found {len(all_runs)}"
        )
    summary = (
        all_runs.groupby("Shot")["Avg_Corr"]
        .agg(["mean", "std"])
        .reindex(EXPECTED_SHOTS)
    )
    new_s35 = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "model_display": canonical_model_name(MODEL_NAME),
                "shot": shot,
                "pearson_r_mean": float(summary.loc[shot, "mean"]),
                "pearson_r_std": float(summary.loc[shot, "std"]),
            }
            for shot in EXPECTED_SHOTS
        ]
    )
    s35 = pd.concat(
        [s35_old[s35_old["model"].astype(str) != MODEL_NAME], new_s35],
        ignore_index=True,
    )
    s35["model_display"] = s35["model"].map(canonical_model_name)
    s35 = s35.sort_values(["model", "shot"]).reset_index(drop=True)

    if len(s33) != 34 or s33["model"].nunique() != 34:
        raise RuntimeError("S33 does not contain exactly 34 unique model rows")
    if len(s34) != 34 or s34["model"].nunique() != 34:
        raise RuntimeError("S34 does not contain exactly 34 unique model rows")
    if len(s35) != 34 * len(EXPECTED_SHOTS) or s35.duplicated(["model", "shot"]).any():
        raise RuntimeError("S35 does not contain a complete 34 x 51 model/shot matrix")
    for frame, metric in ((s33, "rank_score"), (s34, "classification_mcc")):
        selected = BACKEND.representative_rows(fig4_frame(frame), metric)
        if MODEL_NAME not in set(selected["model"].astype(str)):
            raise RuntimeError(
                f"GPN-MSA would not be selected for Fig. 4 metric {metric}"
            )
    endpoint = (
        s35[s35["shot"] == 60]
        .sort_values("pearson_r_mean", ascending=False)
        .reset_index(drop=True)
    )
    if MODEL_NAME not in set(
        BACKEND.representative_rows(fig4_frame(endpoint), "pearson_r_mean")[
            "model"
        ].astype(str)
    ):
        raise RuntimeError(
            "GPN-MSA would not be selected for the Fig. 4c endpoint ranking"
        )
    for frame, column in (
        (s33, "rank_score"),
        (s34, "classification_mcc"),
        (s35, "pearson_r_mean"),
    ):
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite values in {column}")

    backup_dir = run_root / "backups" / "tables"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = {}
    for path in (s33_path, s34_path, s35_path):
        backup = backup_dir / f"{timestamp}_{path.name}"
        backup.write_bytes(path.read_bytes())
        backups[path.name] = {"path": str(backup), "sha256": sha256_file(backup)}

    atomic_csv(s33, s33_path)
    atomic_csv(s34, s34_path)
    atomic_csv(s35, s35_path)
    figures = refresh_fig4(repo_root, tables_dir, args.python, env)

    report = {
        "status": "complete",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "samples": EXPECTED_SAMPLES,
        "embedding_width": EXPECTED_EMBEDDING_WIDTH,
        "metrics": {
            "tsne_silhouette": silhouette,
            "tsne_ch_index": ch_index,
            "classification_accuracy": float(cls_row["accuracy_mean"]),
            "classification_mcc": float(cls_row["mcc_mean"]),
            "fewshot_pearson_r_at_60": float(summary.loc[60, "mean"]),
        },
        "tables": {
            path.name: {"rows": len(frame), "sha256": sha256_file(path)}
            for path, frame in ((s33_path, s33), (s34_path, s34), (s35_path, s35))
        },
        "backups": backups,
        "figures": figures,
        "fewshot_report": str(reports[-1]),
        "preflight": str(config_dir / "preflight.json"),
    }
    report_path = run_root / "config" / "finalization_report.json"
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
