#!/usr/bin/env python3
"""Reuse retained raw predictions for a separate, descriptive cLLR audit.

Primary tables and raw source files are read-only. Calibration artifacts are
downloaded from immutable checkpoint revisions, hashed and compared to local
assets. Missing sources become explicit coverage records.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))
from calibrate_gpn_star_scores import Reference, calibrate, evaluate, sha256
from adapters.gpn_star_input import normalize


def publication_summary(records, source_roots=()):
    """Keep scalar scientific results and export portable provenance labels."""
    frame = pd.DataFrame(
        [
            {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
            for r in records
        ]
    )
    roots = sorted(
        (
            (str(Path(path).resolve()).rstrip("/") + "/", label + "://")
            for label, path in source_roots
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def portable(value):
        if not isinstance(value, str):
            return value
        for source, label in roots:
            value = value.replace(source, label)
        value = re.sub(r"/(?:mnt/[^/]+/)?home/[^/\s]+/", "source://", value)
        if value.startswith("/"):
            return "external://" + Path(value).name
        return value

    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(portable)
    return frame


def keys(frame):
    chrom = frame.chrom.astype(str).str.replace(r"^chr", "", regex=True)
    return (
        chrom
        + ":"
        + frame.pos.astype(int).astype(str)
        + ":"
        + frame.ref
        + ">"
        + frame.alt
    )


def select_target(raw, input_path, task, mode):
    # Reuse the authoritative evaluator's cleaned input and labels, rather
    # than a pilot roster or historical labels from an earlier score file.
    from argparse import Namespace

    target, _ = normalize(
        Namespace(input=input_path, max_records=0, task_name=task, score_mode=mode)
    )
    target = target[target.input_status.eq("PASS")].copy()
    raw = raw.copy()
    raw["key"] = keys(raw)
    target["key"] = keys(target)
    duplicates = raw[raw.key.duplicated(keep=False)]
    if (
        not duplicates.empty
        and duplicates.groupby("key").raw_llr.nunique(dropna=False).gt(1).any()
    ):
        raise ValueError("conflicting predictions for a duplicate variant key")
    raw = raw.drop_duplicates("key")
    merged = target[["key", "variant_id", "label"]].merge(
        raw.drop(columns=["variant_id", "label"]),
        on="key",
        how="left",
        validate="many_to_one",
    )
    if merged.model_id.isna().any():
        raise ValueError(
            "authoritative input has variants missing from retained raw predictions"
        )
    merged["score_mode"] = mode
    return merged.drop(columns="key")


def paired_top1(frame, column):
    points = []
    # Each prespecified positive has its own trio. Distinct DDD positives
    # can share a genomic position; grouping only by coordinates conflates
    # those candidate sets and would incorrectly drop valid comparisons.
    grouped = frame.copy()
    grouped["candidate_group"] = (
        grouped.variant_id.astype(str).str.split("__samepos_other_", n=1).str[0]
    )
    for _, group in grouped.groupby("candidate_group", sort=True):
        if len(group) != 3 or group.label.astype(int).sum() != 1:
            raise ValueError(
                "DDD group must have one positive and two other ALT candidates"
            )
        positive = group.loc[group.label.astype(int).eq(1), column].iloc[0]
        other = group.loc[group.label.astype(int).eq(0), column].max()
        points.append(float(positive > other))
    return float(np.mean(points)), len(points)


def binary_metrics(frame, mode):
    raw = pd.to_numeric(frame.raw_llr, errors="coerce")
    use = np.isfinite(raw) & frame.status.eq("PASS")
    y = frame.loc[use, "label"].astype(int)
    values = raw.loc[use]
    if mode == "neg_llr":
        values = -values
    elif mode == "abs_llr":
        values = values.abs()
    return {
        "raw_source_n": len(y),
        "raw_source_auprc": float(average_precision_score(y, values)),
        "raw_source_auroc": float(roc_auc_score(y, values)),
    }


def write_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def run(args):
    repo = Path(__file__).resolve().parents[2]
    configs = yaml.safe_load((repo / "configs/gpn_star.yaml").read_text())["models"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    calibrations = {}
    calibration_records = []
    for model, cfg in configs.items():
        url = f"https://huggingface.co/{cfg['hf_repo']}/resolve/{cfg['revision']}/calibration_table/llr.parquet"
        p = output / "calibration" / model / "llr.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": model, "revision": cfg["revision"], "url": url}
        try:
            receipt = p.with_suffix(".provenance.json")
            if not p.exists() or not receipt.exists():
                with urlopen(url, timeout=30) as response:
                    data = response.read()
                p.write_bytes(data)
                record["sha256"] = sha256(p)
                receipt.write_text(json.dumps(record, indent=2))
            else:
                previous = json.loads(receipt.read_text())
                if previous["url"] != url or previous["sha256"] != sha256(p):
                    raise ValueError("cached calibration receipt mismatch")
                record.update(previous)
            local = (
                args.assets_root / "models" / model / "calibration_table/llr.parquet"
            )
            record["local_matches_pinned_official"] = local.exists() and sha256(
                local
            ) == sha256(p)
            record["status"] = "PINNED_OFFICIAL_ARTIFACT_VERIFIED"
            calibrations[model] = pd.read_parquet(p)
        except Exception as exc:
            record.update(status="CALIBRATION_UNAVAILABLE", reason=str(exc))
        calibration_records.append(record)
    reference = Reference(args.reference_fasta)
    summary = []
    refresh = args.runs_root / "20260916_nonclinvar_refresh"
    definitions = [
        ("S26", "Causal", "neg_llr"),
        ("S27", "Rare_Causal", "neg_llr"),
        ("S28", "Rare_Variants", "neg_llr"),
        ("S29", "MPRA", "abs_llr"),
        ("S30", "DDD", "neg_llr"),
    ]
    try:
        for model, cfg in configs.items():
            if model not in calibrations:
                continue
            suffix = model[-1]
            clinvar = (
                args.runs_root
                / "20260912/gpnm_corrected_affected/scores"
                / model
                / "ClinVar260615_all.tsv"
                if suffix == "M"
                else args.runs_root
                / "20260912/gpnstar_pv_current"
                / suffix
                / "scores"
                / model
                / "ClinVar260615_all.tsv"
            )
            specs = [("S23", clinvar, None, "neg_llr", None)]
            for tid, task, mode in definitions:
                manifest = refresh / "metrics" / task / (model + ".json")
                if manifest.exists():
                    info = json.loads(manifest.read_text())
                    specs.append(
                        (
                            tid,
                            Path(info["score_source"]),
                            Path(info["input"]),
                            mode,
                            info,
                        )
                    )
                else:
                    summary.append(
                        {
                            "model": model,
                            "table_id": tid,
                            "status": "SOURCE_METRIC_MANIFEST_UNAVAILABLE",
                        }
                    )
            samepos = (
                clinvar.parent / "DDD_LoF_splice_vs_same_position_other_alleles.tsv"
            )
            if suffix != "M":
                alternatives = sorted(
                    (args.runs_root / "20260911").glob(
                        f"**/scores/{model}/DDD_LoF_splice_vs_same_position_other_alleles.tsv"
                    )
                )
                samepos = alternatives[0] if alternatives else samepos
            specs.append(("S31", samepos, None, "neg_llr", None))
            for tid, source, target, mode, info in specs:
                record = {
                    "model": model,
                    "table_id": tid,
                    "source_path": str(source),
                    "revision": cfg["revision"],
                }
                try:
                    if not source.exists():
                        raise FileNotFoundError(source)
                    if info and (
                        sha256(source) != info["score_source_sha256"]
                        or sha256(target) != info["input_sha256"]
                    ):
                        raise ValueError(
                            "primary metric manifest input/source hash mismatch"
                        )
                    raw = pd.read_csv(source, sep="\t", dtype={"chrom": str})
                    if target is not None:
                        raw = select_target(raw, target, tid, mode)
                    out = calibrate(
                        raw, calibrations[model], reference, model, cfg["revision"]
                    )
                    point = evaluate(out)
                    current = pd.read_csv(next((repo / "tables").glob(tid + "_*.csv")))
                    main = current[current.model.eq(model)].iloc[0]
                    if tid == "S31":
                        complete = out[out.calibration_status.eq("PASS")].copy()
                        candidate_ids = (
                            complete.variant_id.astype(str)
                            .str.split("__samepos_other_", n=1)
                            .str[0]
                        )
                        sizes = complete.groupby(candidate_ids).variant_id.transform(
                            "size"
                        )
                        complete = complete[sizes.eq(3)]
                        r, n = paired_top1(complete, "raw_benchmark_score")
                        c, _ = paired_top1(complete, "calibrated_benchmark_score")
                        point.update(
                            raw_paired_top1=r,
                            cllr_paired_top1=c,
                            paired_groups=n,
                            primary_paired_top1=float(main.paired_top1),
                            raw_reproduces_primary=bool(
                                np.isclose(r, main.paired_top1, atol=2e-6, rtol=0)
                            ),
                        )
                    else:
                        point.update(binary_metrics(raw, mode))
                        point.update(
                            primary_auprc=float(main.auprc_mean),
                            primary_n=int(main.num_samples),
                        )
                        point["raw_reproduces_primary"] = bool(
                            point["raw_source_n"] == int(main.num_samples)
                            and np.isclose(
                                point["raw_source_auprc"],
                                main.auprc_mean,
                                atol=2e-6,
                                rtol=0,
                            )
                        )
                    score_path = output / "scores" / model / (tid + ".cllr.tsv.gz")
                    write_frame(out, score_path)
                    record.update(
                        point,
                        source_sha256=sha256(source),
                        input_path=str(target) if target else "native retained input",
                        input_sha256=sha256(target) if target else None,
                        calibrated_scores=str(score_path),
                        calibrated_scores_sha256=sha256(score_path),
                        status="COMPLETE"
                        if point["raw_reproduces_primary"]
                        else "RAW_PRIMARY_RECONCILIATION_FAILED",
                    )
                    score_path.with_suffix(".manifest.json").write_text(
                        json.dumps(record, indent=2)
                    )
                except Exception as exc:
                    record.update(status="UNAVAILABLE", reason=str(exc))
                summary.append(record)
                print(model, tid, record["status"], flush=True)
        # Retrospective tasks retain per-event directions and the five seed
        # rosters. No new winner rule or confidence interval is invented.
        event = args.source_project / "runs/fig3_ijk_eventcentric_20260918"
        event_tasks = (
            args.source_project
            / "datasets/clinvar260707/dynamic_outputs/legacyratio_event_ds1p0x_noMT_5seeds"
        )
        events = []
        for model, cfg in configs.items():
            if model not in calibrations:
                continue
            for seed in range(1, 6):
                source_parts = sorted(
                    (event / "scores" / model / f"seed{seed:02d}").glob("part*.tsv")
                )
                task_paths = sorted(
                    (event_tasks / f"seed{seed:02d}/tasks").glob("*.tsv")
                )
                if not source_parts or not task_paths:
                    summary.append(
                        {
                            "model": model,
                            "table_id": "S32",
                            "seed": seed,
                            "status": "EVENT_INPUTS_UNAVAILABLE",
                        }
                    )
                    continue
                try:
                    raw = pd.concat(
                        [
                            pd.read_csv(p, sep="\t", dtype={"chrom": str})
                            for p in source_parts
                        ],
                        ignore_index=True,
                    )
                    calibrated = calibrate(
                        raw, calibrations[model], reference, model, cfg["revision"]
                    )
                    path = output / "scores" / model / f"S32.seed{seed:02d}.cllr.tsv.gz"
                    write_frame(calibrated, path)
                    calibrated = calibrated.set_index("variant_id")
                    for task_path in task_paths:
                        task = next(
                            (
                                t
                                for t in [
                                    "blb_stable_vs_upgraded",
                                    "plp_stable_vs_downgraded",
                                    "vus_resolved_path_vs_benign",
                                ]
                                if task_path.stem.endswith(t)
                            ),
                            None,
                        )
                        if task is None:
                            continue
                        td = pd.read_csv(task_path, sep="\t")
                        full = td.merge(
                            calibrated.reset_index()[
                                [
                                    "variant_id",
                                    "calibration_status",
                                    "raw_llr",
                                    "calibrated_llr",
                                ]
                            ],
                            on="variant_id",
                            how="left",
                            validate="many_to_one",
                        )
                        subset = full[full.calibration_status.eq("PASS")]
                        y = subset.label.astype(int).to_numpy()
                        downgrade = task == "plp_stable_vs_downgraded"
                        if downgrade:
                            y = 1 - y
                        row = {
                            "model": model,
                            "seed": seed,
                            "task_name": task,
                            "cutoff_year": int(task_path.stem[:4]),
                            "input_n": len(full),
                            "matched_n": len(subset),
                            "coverage": len(subset) / len(full),
                        }
                        for kind, column in [
                            ("raw", "raw_llr"),
                            ("cllr", "calibrated_llr"),
                        ]:
                            scores = subset[column].to_numpy(float) * (
                                1 if downgrade else -1
                            )
                            k = max(1, int(np.ceil(len(y) * 0.01)))
                            precision = float(
                                y[np.argsort(-scores, kind="stable")[:k]].mean()
                            )
                            row[kind + "_precision_at_k"] = precision
                            row[kind + "_enrichment"] = precision / float(y.mean())
                        events.append(row)
                    summary.append(
                        {
                            "model": model,
                            "table_id": "S32",
                            "seed": seed,
                            "status": "COMPLETE_DESCRIPTIVE",
                            "scores_sha256": sha256(path),
                            "source_shards": [
                                {"path": str(p), "sha256": sha256(p)}
                                for p in source_parts
                            ],
                        }
                    )
                except Exception as exc:
                    summary.append(
                        {
                            "model": model,
                            "table_id": "S32",
                            "seed": seed,
                            "status": "UNAVAILABLE",
                            "reason": str(exc),
                        }
                    )
                print(model, "S32", seed, summary[-1]["status"], flush=True)
        if events:
            ed = pd.DataFrame(events)
            ed.to_csv(output / "retrospective_cllr_by_seed.csv", index=False)
            agg = (
                ed.groupby(["model", "task_name", "cutoff_year"])
                .agg(
                    raw_precision_at_k=("raw_precision_at_k", "mean"),
                    cllr_precision_at_k=("cllr_precision_at_k", "mean"),
                    raw_enrichment=("raw_enrichment", "mean"),
                    cllr_enrichment=("cllr_enrichment", "mean"),
                    seeds=("seed", "nunique"),
                    coverage=("coverage", "mean"),
                )
                .reset_index()
            )
            baseline = pd.read_csv(next((repo / "tables").glob("S32_*.csv")))
            baseline = baseline.rename(columns={"model_label": "model"})
            agg = agg.merge(
                baseline[
                    [
                        "model",
                        "task_name",
                        "cutoff_year",
                        "precision_at_k_mean",
                        "enrichment_vs_baseline_mean",
                    ]
                ],
                on=["model", "task_name", "cutoff_year"],
                how="left",
                validate="one_to_one",
            )
            agg["raw_primary_delta"] = (
                agg.raw_enrichment - agg.enrichment_vs_baseline_mean
            )
            agg.to_csv(output / "retrospective_cllr_summary.csv", index=False)
    finally:
        reference.close()
    (output / "sensitivity_manifest.json").write_text(json.dumps(summary, indent=2))
    publication_summary(
        summary,
        [
            ("runs", args.runs_root),
            ("assets", args.assets_root),
            ("source", args.source_project),
            ("output", output),
            ("repo", repo),
        ],
    ).to_csv(output / "cllr_sensitivity_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tables/revision_consistency",
    )
    run(parser.parse_args())
