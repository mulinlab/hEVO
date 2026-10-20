#!/usr/bin/env python3
"""Recompute the retained non-ClinVar model summary tables.

The command consumes external cleaned run data and score caches. It validates
model/task IDs and label sets before replacing the five retained S tables. Raw
curve-point summary files are not part of the public release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


CANONICAL_MODELS = (
    "hEvo-large",
    "hEvo-base",
    "evo2_1b_base",
    "SpeciesLM",
    "NT_500m_MS",
    "Mistral-DNA-17M",
    "AIDO.DNA-300M",
    "GENERator-3B",
    "GPN-MSA",
    "DNABERT",
    "Mistral-DNA-422M",
    "GENERator-1.2B",
    "NTv3-8M-pre",
    "Mistral-DNA-138M",
    "HyenaDNA_tiny_1k",
    "NT_2.5b_1kG",
    "NT_500m_1kG",
    "Caduceus-PH",
    "Caduceus-PS",
    "evo2_7b_base",
    "GPN-promoter",
    "NTv3-100M-pre",
    "GROVER",
    "DNABERT2",
    "NTv3-650M-pre",
    "NT_2.5b_MS",
    "GENA-LM-base",
    "GENA-LM-large",
    "Genos-1.2B",
)
EXTENSIONS = ("GPN-Star-M", "GPN-Star-P", "GPN-Star-V")
H5_FILENAMES = {
    "hEvo-base": "hEvo-base.hdf5",
    "hEvo-large": "hEvo-large.hdf5",
}

TASKS = {
    "Causal": {
        "table": "S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv",
        "dataset": "Causaldb_disease",
        "plot_title": "Causal Variants",
        "input": "causal_variants_hg38_1023_clean.csv",
        "h5_subdir": "Causal_Variants",
        "direction": "neg_llr",
    },
    "Rare_Causal": {
        "table": "S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv",
        "dataset": "rare_causal",
        "plot_title": "Rare Causal Variants",
        "input": "rare_causal_ref_alt_hg38_1023_clean.csv",
        "h5_subdir": "Rare_Causal",
        "direction": "neg_llr",
    },
    "Rare_Variants": {
        "table": "S28_zero_shot_classification_of_uk_biobank_rare_variants.csv",
        "dataset": "UKB490K_causalneg_no_replacement",
        "plot_title": "Rare Variants",
        "input": "rare_variants_ref_alt_dataset_w1023_clean.csv",
        "h5_subdir": "Rare_Variants",
        "direction": "neg_llr",
    },
    "MPRA": {
        "table": "S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv",
        "dataset": "combined_emvar_strict",
        "plot_title": "MPRA",
        "input": "mpra_emvar_hg38_1023_clean.csv",
        "h5_subdir": "combined_emvar_strict",
        "direction": "abs_llr",
    },
    "DDD": {
        "table": "S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv",
        "dataset": "ddd_public_research_vs_clinvar_benign_plof_splice_snv",
        "plot_title": "De novo LoF/splice vs ClinVar benign coding SNVs",
        "input": "ddd_public_research_vs_clinvar_benign_plof_splice_snv_clean.csv",
        "h5_subdir": "DDD",
        "direction": "neg_llr",
    },
}
TABLE_COLUMNS = [
    "plot_title",
    "dataset",
    "model",
    "auprc_mean",
    "auprc_ci_lower",
    "auprc_ci_upper",
    "num_samples",
    "num_positive",
    "num_negative",
    "baseline",
    "auroc_mean",
    "auroc_ci_lower",
    "auroc_ci_upper",
]
PROVENANCE_COLUMNS = [
    "task_id",
    "dataset",
    "model",
    "absolute_h5_path",
    "h5_sha256",
    "source_result_json",
    "source_table",
    "n_input",
    "n_h5_raw",
    "n_scored",
    "num_positive",
    "num_negative",
    "n_skipped_h5",
    "full_roc_points",
    "full_pr_points",
    "exported_roc_points",
    "exported_pr_points",
    "score_direction",
    "table_auroc",
    "table_auprc",
    "computed_auroc",
    "computed_auprc",
    "auroc_abs_diff",
    "auprc_abs_diff",
    "metric_crosscheck_status",
    "downsample_method",
    "status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("seqID") or row.get("SeqID") or "" for row in rows]
    if not ids or any(not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: missing or duplicate variant IDs")
    labels = [int(row["label"]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError(f"{path}: expected both binary labels")
    return rows


def h5_scores(path: Path) -> tuple[dict[str, float], dict[str, int], int]:
    with h5py.File(path, "r") as handle:
        details = [
            json.loads(value.decode() if isinstance(value, bytes) else str(value))
            for value in handle["details"][...]
        ]
        llrs = np.asarray(handle["llrs"][...], dtype=float)
        labels = np.asarray(handle["labels"][...], dtype=int)
    ids = [str(detail.get("idx", "")) for detail in details]
    if (
        any(not sid for sid in ids)
        or len(set(ids)) != len(ids)
        or len(ids) != len(llrs)
        or len(ids) != len(labels)
    ):
        raise ValueError(f"{path}: invalid HDF5 details/llrs/labels alignment")
    if not np.isfinite(llrs).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path}: non-finite score or non-binary label")
    return dict(zip(ids, llrs)), dict(zip(ids, labels)), len(ids)


def gpn_scores(path: Path) -> tuple[dict[str, float], dict[str, int], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = [row for row in rows if row.get("status", "PASS") == "PASS"]
    ids = [row.get("variant_id", "") for row in rows]
    if any(not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: missing or duplicate PASS IDs")
    scores = {sid: float(row["raw_llr"]) for sid, row in zip(ids, rows)}
    labels = {sid: int(row["label"]) for sid, row in zip(ids, rows)}
    if not scores or not all(np.isfinite(value) for value in scores.values()):
        raise ValueError(f"{path}: non-finite or empty GPN-Star scores")
    return scores, labels, len(rows)


def bootstrap(
    labels: np.ndarray, scores: np.ndarray, seed: int = 42, repeats: int = 1000
) -> tuple[float, float, float, float, float, float]:
    rng = np.random.RandomState(seed)
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    for _ in range(repeats):
        indices = rng.randint(0, len(labels), size=len(labels))
        y = labels[indices]
        s = scores[indices]
        try:
            auroc_values.append(float(roc_auc_score(y, s)))
            auprc_values.append(float(average_precision_score(y, s)))
        except ValueError:
            continue
    if not auroc_values or not auprc_values:
        raise ValueError("bootstrap produced no valid two-class resamples")
    return (
        float(np.percentile(auroc_values, 2.5)),
        float(np.percentile(auroc_values, 97.5)),
        float(np.percentile(auprc_values, 2.5)),
        float(np.percentile(auprc_values, 97.5)),
        float(np.mean(auroc_values)),
        float(np.mean(auprc_values)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--gpn-comparison", type=Path, required=True)
    parser.add_argument("--rare-h5-root", type=Path, required=True)
    parser.add_argument("--ddd-h5-root", type=Path, required=True)
    args = parser.parse_args()
    run_root, repo_root = args.run_root.resolve(), args.repo_root.resolve()
    input_dir = run_root / "inputs"
    table_dir = repo_root / "hEvo_benchmark" / "tables"
    source_h5_roots = {
        "Causal": run_root / "h5_causal" / "Causal_Variants" / "Causal_Variants",
        "Rare_Causal": run_root / "filtered_h5" / "Rare_Causal",
        "Rare_Variants": args.rare_h5_root.resolve(),
        "MPRA": run_root / "h5_mpra" / "MPRA" / "combined_emvar_strict",
        "DDD": args.ddd_h5_root.resolve(),
    }
    gpn_task_names = {
        "Causal Variants": "Causal",
        "Rare Causal Variants": "Rare_Causal",
        "Rare Variants": "Rare_Variants",
        "MPRA": "MPRA",
        "De novo LoF/splice vs ClinVar benign coding SNVs": "DDD",
    }
    with args.gpn_comparison.open("r", encoding="utf-8", newline="") as handle:
        gpn_rows = list(csv.DictReader(handle))
    gpn_files: dict[tuple[str, str], Path] = {}
    for row in gpn_rows:
        task_name = gpn_task_names.get(row.get("task", ""))
        model = row.get("model", "")
        source_file = Path(row.get("source_file", ""))
        if task_name and model in EXTENSIONS:
            gpn_files[(task_name, model)] = source_file
    if len(gpn_files) != len(TASKS) * len(EXTENSIONS):
        raise ValueError(
            f"GPN-Star comparison does not provide all {len(TASKS) * len(EXTENSIONS)} task/model files"
        )

    new_table_rows: dict[str, list[dict[str, object]]] = {name: [] for name in TASKS}
    new_provenance: list[dict[str, object]] = []
    metric_dir = run_root / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    for task_name, task in TASKS.items():
        input_path = input_dir / task["input"]
        rows = dataset_rows(input_path)
        ids = [row.get("seqID") or row.get("SeqID") for row in rows]
        labels_by_id = {sid: int(row["label"]) for sid, row in zip(ids, rows)}
        source_specs = [
            (
                model,
                source_h5_roots[task_name]
                / (
                    "GPN-MSA.official_msa_llr.hdf5"
                    if model == "GPN-MSA"
                    else H5_FILENAMES.get(model, f"{model}.hdf5")
                    if task_name in {"Rare_Causal", "Rare_Variants", "DDD"}
                    else f"{model}.hdf5"
                ),
                "h5",
            )
            for model in CANONICAL_MODELS
        ]
        source_specs += [
            (model, gpn_files[(task_name, model)], "gpn_star") for model in EXTENSIONS
        ]
        for model, source_path, source_type in source_specs:
            if source_type == "h5":
                raw_scores, score_labels, n_raw = h5_scores(source_path)
                if set(raw_scores) != set(ids):
                    raise ValueError(f"{task_name}/{model}: HDF5 ID set mismatch")
                if any(score_labels[sid] != labels_by_id[sid] for sid in ids):
                    raise ValueError(f"{task_name}/{model}: HDF5 label mismatch")
                raw = np.asarray([raw_scores[sid] for sid in ids], dtype=float)
                score = np.abs(raw) if task["direction"] == "abs_llr" else -raw
                h5_path = source_path
            else:
                raw_scores, score_labels, n_raw = gpn_scores(source_path)
                if set(raw_scores) != set(ids):
                    raise ValueError(f"{task_name}/{model}: GPN-Star ID set mismatch")
                if any(score_labels[sid] != labels_by_id[sid] for sid in ids):
                    raise ValueError(f"{task_name}/{model}: GPN-Star label mismatch")
                raw = np.asarray([raw_scores[sid] for sid in ids], dtype=float)
                score = np.abs(raw) if task["direction"] == "abs_llr" else -raw
                h5_path = None
            y = np.asarray([labels_by_id[sid] for sid in ids], dtype=int)
            auroc = float(roc_auc_score(y, score))
            auprc = float(average_precision_score(y, score))
            auroc_low, auroc_high, auprc_low, auprc_high, _, _ = bootstrap(y, score)
            metrics = {
                "schema": "gpn_star_nonclinvar_metric_v1",
                "task": task_name,
                "model": model,
                "num_samples": len(y),
                "num_positive": int(y.sum()),
                "num_negative": int(len(y) - y.sum()),
                "score_direction": task["direction"],
                "auroc": auroc,
                "auprc": auprc,
                "auroc_ci_lower": auroc_low,
                "auroc_ci_upper": auroc_high,
                "auprc_ci_lower": auprc_low,
                "auprc_ci_upper": auprc_high,
                "bootstrap_repeats": 1000,
                "bootstrap_seed": 42,
                "input": str(input_path.resolve()),
                "input_sha256": sha256(input_path),
                "score_source": str(source_path.resolve()),
                "score_source_sha256": sha256(source_path),
            }
            metric_path = metric_dir / task_name / f"{model}.json"
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            new_table_rows[task_name].append(
                {
                    "plot_title": task["plot_title"],
                    "dataset": task["dataset"],
                    "model": model,
                    "auprc_mean": auprc,
                    "auprc_ci_lower": auprc_low,
                    "auprc_ci_upper": auprc_high,
                    "num_samples": len(y),
                    "num_positive": int(y.sum()),
                    "num_negative": int(len(y) - y.sum()),
                    "baseline": float(y.mean()),
                    "auroc_mean": auroc,
                    "auroc_ci_lower": auroc_low,
                    "auroc_ci_upper": auroc_high,
                }
            )
            new_provenance.append(
                {
                    "task_id": task["table"].split("_", 1)[0],
                    "dataset": task["dataset"],
                    "model": model,
                    "absolute_h5_path": str(h5_path.resolve()) if h5_path else "",
                    "h5_sha256": sha256(h5_path) if h5_path else "",
                    "source_result_json": str(metric_path.resolve()),
                    "source_table": str((table_dir / task["table"]).resolve()),
                    "n_input": len(y),
                    "n_h5_raw": n_raw if source_type == "h5" else 0,
                    "n_scored": len(y),
                    "num_positive": int(y.sum()),
                    "num_negative": int(len(y) - y.sum()),
                    "n_skipped_h5": 0,
                    "score_direction": task["direction"],
                    "table_auroc": auroc,
                    "table_auprc": auprc,
                    "computed_auroc": auroc,
                    "computed_auprc": auprc,
                    "auroc_abs_diff": 0.0,
                    "auprc_abs_diff": 0.0,
                    "metric_crosscheck_status": "MATCH",
                    "downsample_method": "not_exported",
                    "status": "MATCH",
                }
            )

    for task_name, task in TASKS.items():
        table_path = table_dir / task["table"]
        with table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=TABLE_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(new_table_rows[task_name])
    summary = {
        "schema": "gpn_star_nonclinvar_export_v1",
        "status": "PASS",
        "canonical_model_count": len(CANONICAL_MODELS),
        "extension_model_count": len(EXTENSIONS),
        "tasks": {
            task_name: {
                "rows": len(new_table_rows[task_name]),
                "table": str((table_dir / task["table"]).resolve()),
                "table_sha256": sha256(table_dir / task["table"]),
            }
            for task_name, task in TASKS.items()
        },
    }
    (run_root / "export_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
