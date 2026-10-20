#!/usr/bin/env python3
"""Validate GPN-Star ancestry embeddings and update S33-S35 reproducibly."""

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

FIGURE_SCRIPTS = Path(__file__).resolve().parents[2] / "figures" / "scripts"
if str(FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SCRIPTS))
from common.model_names import canonical_model_name  # noqa: E402 - local path/runtime configuration precedes this import


MODELS = ("GPN-Star-M", "GPN-Star-P", "GPN-Star-V")
EXPECTED_EMBEDDING_WIDTH = 3076
EXPECTED_ROWS_PER_SAMPLE = 10000
EXPECTED_BASELINE_MODELS = 30
EXPECTED_SAMPLES = 1500
EXPECTED_SHOTS = list(range(10, 61))
WINDOW_LENGTH = 511
CENTER_OFFSET = 255


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_path(tables: Path, prefix: str) -> Path:
    matches = list(tables.glob(f"{prefix}_*.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {prefix}_*.csv under {tables}; found {matches}"
        )
    return matches[0]


def validate_shared_locus_roster(
    run_root: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Prove that all samples use the same 10,000 coordinate/ref rows.

    The MSA preflight queried every locus in the reference sample. The
    per-sample extractors also check each non-reference allele against the
    coordinate MSA. This roster check extends the preflight to homozygous-ref
    rows, which are intentionally represented as exact zero deltas without a
    model forward pass.
    """
    input_paths = [
        Path(line)
        for line in (run_root / "config/input_files.txt").read_text().splitlines()
        if line
    ]
    if len(input_paths) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Expected 1500 inputs, found {len(input_paths)}")

    reference_digest = None
    reference_loci: set[tuple[str, int]] = set()
    expected_identity = None
    expected_inputs: dict[str, dict[str, object]] = {}
    for file_index, path in enumerate(input_paths):
        digest = hashlib.sha256()
        row_count = 0
        sample_id = population = None
        local_loci: set[tuple[str, int]] = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                current_identity = (str(row["sample"]), str(row["sample_popu"]))
                if sample_id is None:
                    sample_id, population = current_identity
                    if (
                        expected_identity is not None
                        and current_identity in expected_identity
                    ):
                        raise RuntimeError(
                            f"Duplicate input identity {current_identity} in {path}"
                        )
                elif current_identity != (sample_id, population):
                    raise RuntimeError(f"Mixed sample identity in {path}:{line_number}")

                sequence = str(row["seq"]).upper()
                reference = str(row["ref_base"]).upper()
                alternate = str(row["alt_base"]).upper()
                if (
                    len(sequence) != WINDOW_LENGTH
                    or sequence[CENTER_OFFSET] != alternate
                ):
                    raise RuntimeError(
                        f"Invalid sequence/alternate at {path}:{line_number}"
                    )
                if reference not in "ACGT" or alternate not in "ACGT":
                    raise RuntimeError(f"Invalid alleles at {path}:{line_number}")
                chrom = str(row["chr"])
                position = int(row["center_pos"])
                if position <= CENTER_OFFSET:
                    raise RuntimeError(
                        f"Invalid position at {path}:{line_number}: {chrom}:{position}"
                    )
                local_loci.add((chrom, position))
                digest.update(f"{chrom}\t{position}\t{reference}\n".encode("ascii"))
                row_count += 1

        if row_count != EXPECTED_ROWS_PER_SAMPLE:
            raise RuntimeError(f"{path}: expected 10000 rows, found {row_count}")
        if file_index == 0:
            reference_digest = digest.hexdigest()
            reference_loci = local_loci
            expected_identity = set()
        elif digest.hexdigest() != reference_digest:
            raise RuntimeError(
                f"Coordinate/reference roster differs from the first sample: {path}"
            )
        if (sample_id, population) in expected_identity:
            raise RuntimeError(f"Duplicate sample identity {(sample_id, population)}")
        expected_identity.add((sample_id, population))
        artifact_name = f"{population}_{sample_id}.npy"
        if artifact_name in expected_inputs:
            raise RuntimeError(f"Duplicate artifact identity {artifact_name}")
        expected_inputs[artifact_name] = {
            "input_file": str(path.resolve()),
            "sample_id": sample_id,
            "population": population,
            "input_rows": row_count,
            "input_sha256": sha256(path),
        }

    if len(reference_loci) != EXPECTED_ROWS_PER_SAMPLE:
        raise RuntimeError(
            f"Expected 10000 unique loci in the reference sample, found {len(reference_loci)}"
        )
    return (
        {
            "files": len(input_paths),
            "rows_per_file": EXPECTED_ROWS_PER_SAMPLE,
            "unique_loci_per_file": len(reference_loci),
            "coordinate_reference_sha256": reference_digest,
            "all_samples_share_coordinate_reference_roster": True,
        },
        expected_inputs,
    )


def backfill_output_hashes(
    run_root: Path, expected_inputs: dict[str, dict[str, object]]
) -> int:
    """Add hashes to valid legacy manifests written before output hashing existed."""
    updated = 0
    for model in MODELS:
        embeddings = run_root / "results" / model / "embeddings"
        manifests = run_root / "results" / model / "manifests"
        for embedding in sorted(embeddings.glob("*.npy")):
            expected = expected_inputs.get(embedding.name)
            manifest_path = manifests / f"{embedding.stem}.json"
            if expected is None or not manifest_path.is_file():
                raise RuntimeError(
                    f"Cannot hash unrecognized or unmanifested output: {embedding}"
                )
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            vector = np.load(embedding, allow_pickle=False)
            if (
                vector.shape != (EXPECTED_EMBEDDING_WIDTH,)
                or not np.isfinite(vector).all()
            ):
                raise RuntimeError(f"Cannot hash invalid embedding vector: {embedding}")
            identity_matches = (
                record.get("status") == "complete"
                and record.get("model") == model
                and record.get("input_file") == expected["input_file"]
                and record.get("input_sha256") == expected["input_sha256"]
                and record.get("sample_id") == expected["sample_id"]
                and record.get("population") == expected["population"]
                and record.get("input_rows") == EXPECTED_ROWS_PER_SAMPLE
                and record.get("embedding_shape") == [EXPECTED_EMBEDDING_WIDTH]
            )
            if not identity_matches:
                raise RuntimeError(
                    f"Refusing to hash output with mismatched manifest: {manifest_path}"
                )
            actual_hash = sha256(embedding)
            recorded_hash = record.get("output_sha256")
            if recorded_hash is not None:
                if recorded_hash != actual_hash:
                    raise RuntimeError(f"Output hash mismatch for {embedding}")
                continue
            record["output_sha256"] = actual_hash
            temp = manifest_path.with_suffix(manifest_path.suffix + ".partial")
            temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, manifest_path)
            updated += 1
    return updated


def validate_embeddings(
    run_root: Path, expected_inputs: dict[str, dict[str, object]]
) -> dict[str, list[Path]]:
    input_paths = [
        Path(line)
        for line in (run_root / "config/input_files.txt").read_text().splitlines()
        if line
    ]
    if len(input_paths) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} inputs, found {len(input_paths)}"
        )
    expected_names = set(expected_inputs)
    if len(expected_names) != EXPECTED_SAMPLES:
        raise RuntimeError("Input manifest has duplicate sample/population identities")

    validated: dict[str, list[Path]] = {}
    for model in MODELS:
        out_dir = run_root / "results" / model / "embeddings"
        manifest_dir = run_root / "results" / model / "manifests"
        files = sorted(out_dir.glob("*.npy"))
        names = {path.name for path in files}
        if names != expected_names:
            raise RuntimeError(
                f"{model}: embedding roster mismatch; missing={len(expected_names - names)}, "
                f"extra={len(names - expected_names)}"
            )
        if len(files) != EXPECTED_SAMPLES:
            raise RuntimeError(f"{model}: expected 1500 embeddings, found {len(files)}")
        model_paths: set[str] = set()
        msa_store_rosters: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
        for embedding in files:
            vector = np.load(embedding, allow_pickle=False)
            if (
                vector.shape != (EXPECTED_EMBEDDING_WIDTH,)
                or not np.isfinite(vector).all()
            ):
                raise RuntimeError(
                    f"{model}: invalid vector in {embedding}: {vector.shape}"
                )
            manifest = manifest_dir / f"{embedding.stem}.json"
            if not manifest.is_file():
                raise RuntimeError(f"{model}: missing manifest {manifest}")
            record = json.loads(manifest.read_text(encoding="utf-8"))
            expected = expected_inputs[embedding.name]
            model_path = record.get("model_path")
            msa_stores = record.get("msa_stores")
            msa_species = record.get("msa_species")
            if (
                record.get("status") != "complete"
                or record.get("model") != model
                or not isinstance(model_path, str)
                or not Path(model_path).is_absolute()
                or record.get("input_sha256") != expected["input_sha256"]
                or record.get("input_file") != expected["input_file"]
                or record.get("sample_id") != expected["sample_id"]
                or record.get("population") != expected["population"]
                or record.get("input_rows") != expected["input_rows"]
                or record.get("input_rows") != EXPECTED_ROWS_PER_SAMPLE
                or record.get("embedding_shape") != [EXPECTED_EMBEDDING_WIDTH]
                or not isinstance(msa_stores, list)
                or not msa_stores
                or not all(
                    isinstance(path, str) and Path(path).is_absolute()
                    for path in msa_stores
                )
                or not isinstance(msa_species, list)
                or len(msa_species) != len(msa_stores)
                or not all(
                    isinstance(count, int) and count > 0 for count in msa_species
                )
            ):
                raise RuntimeError(
                    f"{model} {embedding.name}: extraction manifest identity/integrity "
                    f"mismatch {manifest}"
                )
            output_sha = record.get("output_sha256")
            if not isinstance(output_sha, str) or output_sha != sha256(embedding):
                raise RuntimeError(
                    f"{model} {embedding.name}: missing or mismatched output_sha256 "
                    f"in {manifest}"
                )
            model_paths.add(model_path)
            msa_store_rosters.add((tuple(msa_stores), tuple(msa_species)))
        if len(model_paths) != 1:
            raise RuntimeError(
                f"{model}: inconsistent model paths across extraction manifests"
            )
        if len(msa_store_rosters) != 1:
            raise RuntimeError(
                f"{model}: inconsistent MSA configuration across manifests"
            )
        validated[model] = files
    return validated


def compute_tsne_metrics(files: list[Path]) -> tuple[float, float]:
    labels = np.asarray([path.name.split("_", 1)[0] for path in files])
    vectors = np.stack([np.load(path, allow_pickle=False) for path in files]).astype(
        np.float32
    )
    scaled = StandardScaler().fit_transform(vectors)
    reduced = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
        metric="cosine",
    ).fit_transform(scaled)
    return (
        float(silhouette_score(reduced, labels)),
        float(calinski_harabasz_score(reduced, labels)),
    )


def run_command(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def refresh_figures(
    repo_root: Path, tables: Path, python: Path
) -> dict[str, dict[str, str]]:
    """Redraw every requested integrated figure and ancestry-linked Fig. 4 panel."""
    scripts = repo_root / "figures/scripts"
    figures = repo_root / "figures"
    commands = [
        [
            python,
            scripts / "Fig1/A4_integrated.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figures / "Fig1",
        ],
        [
            python,
            scripts / "Fig4/A4_integrated.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figures / "Fig4",
        ],
        [
            python,
            scripts / "Fig4/a_unsupervised.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figures / "Fig4",
        ],
        [
            python,
            scripts / "Fig4/b_mcc.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figures / "Fig4",
        ],
        [
            python,
            scripts / "Fig4/c_fewshot.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figures / "Fig4",
        ],
        [
            python,
            scripts / "Fig5/A4_integrated.py",
            "--tables-dir",
            tables,
            "--output-dir",
            figures / "Fig5",
            "--export-panels",
        ],
        [python, scripts / "Fig5/validate_fig5_release.py"],
    ]
    for command in commands:
        run_command([str(value) for value in command])

    figure_report: dict[str, dict[str, str]] = {}
    artifacts = {
        "Fig1": ["A4_integrated"],
        "Fig4": ["A4_integrated", "a_unsupervised", "b_mcc", "c_fewshot"],
        "Fig5": ["A4_integrated"],
    }
    for figure, stems in artifacts.items():
        figure_dir = figures / figure
        for stem in stems:
            outputs = {}
            for extension in ("pdf", "svg", "png"):
                output_path = figure_dir / f"{stem}.{extension}"
                if not output_path.is_file():
                    raise RuntimeError(
                        f"Missing refreshed figure output: {output_path}"
                    )
                actual_hash = sha256(output_path)
                outputs[output_path.name] = actual_hash
            figure_report[f"{figure}/{stem}"] = outputs
    return figure_report


def atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument(
        "--regression-script",
        type=Path,
        default=Path(os.environ["HEVO_REGRESSION_SCRIPT"]).expanduser()
        if os.environ.get("HEVO_REGRESSION_SCRIPT")
        else None,
        help="Regression_with_admixture.py; defaults to the benchmark root's scripts/population_classify",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    repo_root = args.repo_root.resolve()
    tables = args.tables_dir.resolve()
    s33_path = table_path(tables, "S33")
    s34_path = table_path(tables, "S34")
    s35_path = table_path(tables, "S35")

    locus_roster_report, expected_inputs = validate_shared_locus_roster(run_root)
    hash_backfills = backfill_output_hashes(run_root, expected_inputs)
    print(
        f"[validation] added output hashes to {hash_backfills} legacy manifests",
        flush=True,
    )
    embeddings = validate_embeddings(run_root, expected_inputs)
    s33_old = pd.read_csv(s33_path)
    s34_old = pd.read_csv(s34_path)
    s35_old = pd.read_csv(s35_path)
    baseline_models = set(s34_old["model"].astype(str))
    if len(baseline_models) != EXPECTED_BASELINE_MODELS:
        raise RuntimeError(
            f"S34 baseline roster has {len(baseline_models)} models, expected 30"
        )
    if set(s33_old["model"].astype(str)) != baseline_models:
        raise RuntimeError("S33 and S34 baseline model rosters differ")
    if set(s35_old["model"].astype(str)) != baseline_models:
        raise RuntimeError("S35 and S34 baseline model rosters differ")

    stage = run_root / "stage/embeddings"
    for model in MODELS:
        link = stage / f"{model}_popu_emb"
        expected_target = (run_root / "results" / model / "embeddings").resolve()
        if not link.is_symlink() or link.resolve() != expected_target:
            raise RuntimeError(f"Missing or invalid staged embedding link: {link}")
    expected_roster = baseline_models | set(MODELS)
    actual_roster = {
        path.name.removesuffix("_popu_emb")
        for path in stage.iterdir()
        if path.name.endswith("_popu_emb")
    }
    if actual_roster != expected_roster:
        raise RuntimeError(
            f"Staged model roster mismatch; missing={expected_roster - actual_roster}, "
            f"extra={actual_roster - expected_roster}"
        )

    print("[S33] compute t-SNE metrics for the three GPN-Star models", flush=True)
    s33_new = s33_old[~s33_old["model"].isin(MODELS)].copy().reset_index(drop=True)
    for model in MODELS:
        silhouette, ch_index = compute_tsne_metrics(embeddings[model])
        s33_new.loc[len(s33_new)] = {
            "model": model,
            "model_display": model,
            "tsne_silhouette": silhouette,
            "tsne_ch_index": ch_index,
            "rank_score": 0.0,
        }
    s33_new["model_display"] = s33_new["model"].map(canonical_model_name)
    s33_new["rank_score"] = s33_new["tsne_silhouette"] + 0.15 * np.log10(
        s33_new["tsne_ch_index"].astype(float)
    )
    s33_new = s33_new.sort_values(
        ["rank_score", "model"], ascending=[False, True]
    ).reset_index(drop=True)
    if len(s33_new) != 33 or s33_new["model"].nunique() != 33:
        raise RuntimeError("S33 did not produce one row for each of 33 models")

    classifier = repo_root / "scripts/population_classify/classify.py"
    classification_dir = run_root / "results/classification"
    run_command(
        [
            sys.executable,
            str(classifier),
            "--base-dir",
            str(stage),
            "--results-dir",
            str(classification_dir),
        ]
    )
    classification = pd.read_csv(classification_dir / "classification_benchmark.csv")
    if set(classification["model"].astype(str)) != expected_roster:
        raise RuntimeError("S34 classification output has the wrong 33-model roster")
    if set(classification["n_samples"].astype(int)) != {EXPECTED_SAMPLES}:
        raise RuntimeError(
            "S34 classification did not use 1500 samples for every model"
        )
    s34_new = (
        pd.DataFrame(
            {
                "model": classification["model"],
                "model_display": classification["model"].map(canonical_model_name),
                "classification_accuracy": classification["accuracy_mean"],
                "classification_mcc": classification["mcc_mean"],
                "classification_samples": classification["n_samples"].astype(int),
            }
        )
        .sort_values(
            ["classification_accuracy", "classification_mcc", "model"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    if len(s34_new) != 33 or s34_new["model"].nunique() != 33:
        raise RuntimeError("S34 did not produce one row for each of 33 models")

    gt_path = (
        benchmark_root / "results/ancestry_analysis/1kg_final_for_admixture.5.Q.merged"
    )
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    fewshot_dir = run_root / "results/fewshot"
    regression = args.regression_script or (
        benchmark_root / "scripts/population_classify/Regression_with_admixture.py"
    )
    regression = regression.expanduser().resolve()
    if not regression.is_file():
        raise FileNotFoundError(
            f"Regression script not found: {regression}; pass --regression-script "
            "or set HEVO_REGRESSION_SCRIPT"
        )
    run_command(
        [
            sys.executable,
            str(regression),
            "--base-dir",
            str(stage),
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
            "--num-workers",
            str(min(8, os.cpu_count() or 1)),
        ]
    )
    reports = sorted(
        fewshot_dir.glob("fewshot_all_runs_*.csv"), key=lambda p: p.stat().st_mtime
    )
    if not reports:
        raise RuntimeError("Few-shot regression report was not generated")
    all_runs = pd.read_csv(reports[-1])
    all_runs = all_runs[all_runs["Model"].isin(MODELS)]
    expected_runs = {
        (model, shot, seed)
        for model in MODELS
        for shot in EXPECTED_SHOTS
        for seed in range(10, 20)
    }
    observed_runs = set(
        zip(
            all_runs["Model"],
            all_runs["Shot"].astype(int),
            all_runs["Seed"].astype(int),
        )
    )
    if observed_runs != expected_runs:
        raise RuntimeError(
            f"Few-shot GPN-Star runs incomplete; missing={len(expected_runs - observed_runs)}, "
            f"extra={len(observed_runs - expected_runs)}"
        )
    s35_new_rows = []
    for model in MODELS:
        subset = all_runs[all_runs["Model"] == model]
        summary = (
            subset.groupby("Shot")["Avg_Corr"]
            .agg(["mean", "std"])
            .reindex(EXPECTED_SHOTS)
        )
        for shot in EXPECTED_SHOTS:
            s35_new_rows.append(
                {
                    "model": model,
                    "model_display": model,
                    "shot": shot,
                    "pearson_r_mean": float(summary.loc[shot, "mean"]),
                    "pearson_r_std": float(summary.loc[shot, "std"]),
                }
            )
    s35_new = pd.concat([s35_old, pd.DataFrame(s35_new_rows)], ignore_index=True)
    s35_new["model_display"] = s35_new["model"].map(canonical_model_name)
    if (
        len(s35_new) != 33 * len(EXPECTED_SHOTS)
        or s35_new.duplicated(["model", "shot"]).any()
    ):
        raise RuntimeError("S35 is not a complete unique 33-model x 51-shot matrix")
    numeric = s35_new[["pearson_r_mean", "pearson_r_std"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 1] < 0).any():
        raise RuntimeError("S35 contains invalid correlation summaries")

    backup_dir = run_root / "backups/tables"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = {}
    for path in (s33_path, s34_path, s35_path):
        backup = backup_dir / f"{timestamp}_{path.name}"
        backup.write_bytes(path.read_bytes())
        backups[path.name] = {"path": str(backup), "sha256": sha256(backup)}

    atomic_csv(s33_new, s33_path)
    atomic_csv(s34_new, s34_path)
    atomic_csv(s35_new, s35_path)
    figures = refresh_figures(repo_root, tables, Path(sys.executable))
    report = {
        "status": "complete",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "models": list(MODELS),
        "embedding_count_per_model": EXPECTED_SAMPLES,
        "embedding_width": EXPECTED_EMBEDDING_WIDTH,
        "input_locus_roster": locus_roster_report,
        "tables": {
            s33_path.name: {"rows": len(s33_new), "sha256": sha256(s33_path)},
            s34_path.name: {"rows": len(s34_new), "sha256": sha256(s34_path)},
            s35_path.name: {"rows": len(s35_new), "sha256": sha256(s35_path)},
        },
        "backups": backups,
        "fewshot_run_source": str(reports[-1]),
        "figures": figures,
    }
    report_path = run_root / "results/table_update_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
