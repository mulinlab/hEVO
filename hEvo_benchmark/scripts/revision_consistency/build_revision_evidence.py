#!/usr/bin/env python3
"""Reconcile manuscript evidence; never infer missing run provenance.

Run after rebuilding S49/S43. External source roots are read-only. Published
table schemas stay stable; provenance is stored in companion files.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(root, tid):
    paths = list(root.glob(tid + "_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"ambiguous table {tid}: {paths}")
    return pd.read_csv(paths[0]), paths[0]


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def protocol_table(root):
    frame, path = table(root, "S48")
    frame = frame.set_index("configuration_item")
    summary = {
        "input_unit": "30 prespecified biallelic SNVs with literature-specified selected and other alleles",
        "input_files_and_columns": "paired selected/other sequences and locus metadata; S39 contains the locus scores",
        "train_valid_test_split": "no learned classifier or train/test split; prespecified locus roster",
        "model_representation": "masked conditional nucleotide log probabilities or causal sequence log likelihoods; native GPN-MSA allele scores",
        "pooling_or_score": "selected-allele score minus other-allele score; positive values indicate directional concordance",
        "sequence_length": "model-specific paired context recorded by window_bp, center_index and tokenized_length in S39",
        "training_or_cv": "no classifier training and no cross-validation; direct paired allele scoring",
        "uncertainty_or_seed": "95% Clopper-Pearson interval for positive loci / evaluated loci; not training-seed uncertainty",
        "metrics": "positive_loci, n_loci, directional preference proportion and exact binomial CI; descriptive score summaries",
        "configuration_note": "S38 summarizes the prespecified 30-locus allele-preference analysis; directional agreement does not establish fitness or selection coefficients",
    }
    for key, value in summary.items():
        frame.loc[key, "Adaptive allele-preference evaluation"] = value
    locus = dict(summary)
    locus.update(
        input_files_and_columns="S39: variant_id, chrom, pos, selected_allele, other_allele, selected_logprob, other_logprob, locus_score, directional_correct, score_definition, benchmark_status",
        metrics="per-locus selected/other scores, their difference, directional_correct, QA and benchmark status; supplementary CH uses the 30 selected and 30 other raw score values",
        uncertainty_or_seed="no locus-level sampling interval; aggregate exact binomial intervals are in S38",
        configuration_note="S39 is direct paired scoring; no population-centroid distance or adaptive-population ranking is computed",
    )
    for key, value in locus.items():
        frame.loc[key, "Adaptive allele-preference locus-score evaluation"] = value
    column = "Zero-shot paired-sequence variant evaluation"
    frame.loc["pooling_or_score", column] = str(
        frame.loc["pooling_or_score", column]
    ).replace(
        "GPN-Star: native-MSA VEP score",
        "GPN-Star: raw, uncalibrated native-MSA VEP LLR with prespecified task direction; cLLR is a separate sensitivity analysis",
    )
    frame.reset_index().to_csv(path, index=False)


def endpoints(root, output):
    rows, score_rows = [], []

    def add(
        tid,
        group,
        metric,
        value,
        unit=None,
        aggregation="arithmetic mean of source rows",
        selected=None,
    ):
        data, path = table(root, tid)
        if selected is not None:
            data = selected
        data = data.copy()
        data["unit"] = data[unit].astype(str) if unit else group
        data[value] = pd.to_numeric(data[value], errors="coerce")
        data = data[np.isfinite(data[value])]
        for name, part in data.groupby("unit", sort=True):
            scores = part.groupby("model")[value].mean().sort_values(ascending=False)
            best = float(scores.iloc[0])
            winners = scores.index[
                np.isclose(scores, best, atol=1e-12, rtol=0)
            ].tolist()
            primary = scores[scores.index.isin(["hEvo-base", "hEvo-large"])]
            external = scores[~scores.index.str.startswith("hEvo")]
            record = {
                "endpoint_id": tid + ":" + name,
                "group": group,
                "source_table": path.name,
                "source_sha256": digest(path),
                "score_metric": metric,
                "aggregation": aggregation,
                "n_models": len(scores),
                "model_roster": ";".join(sorted(scores.index)),
                "winners": ";".join(winners),
                "winning_score": best,
                "primary_hevo_best": primary.index[0] if len(primary) else "",
                "primary_hevo_score": float(primary.iloc[0]) if len(primary) else None,
                "external_best": external.index[0] if len(external) else "",
                "external_best_score": float(external.iloc[0])
                if len(external)
                else None,
                "selected_runs": ";".join(sorted(set(part.run_id.astype(str))))
                if "run_id" in part
                else "",
                "status": "TABLE_BACKED_DESCRIPTIVE",
                "comparison_caveat": "each model uses its reported coverage; not necessarily an identical sample intersection"
                if tid
                in {"S21", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30"}
                else "",
            }
            rows.append(record)
            for model, score in scores.items():
                score_rows.append(
                    {
                        "endpoint_id": record["endpoint_id"],
                        "model": model,
                        "score": score,
                        "rank": float(
                            scores.rank(ascending=False, method="min").loc[model]
                        ),
                    }
                )

    add("S17", "NT", "MCC", "mcc_mean", "dataset", "one row per task/model")
    add(
        "S18",
        "Genomic Benchmark",
        "MCC",
        "mcc_mean",
        "dataset",
        "one row per task/model",
    )
    add("S19", "URE binary", "MCC", "mcc_mean", "benchmark_raw")
    add(
        "S20",
        "URE multiclass",
        "macro class F1",
        "per_class_f1",
        "dataset",
        "equal-weight mean across five classes",
    )
    source = root.parent / "figures/FigS6/source_data/S6_DeepSEA_source_rows.csv"
    deepsea = pd.read_csv(source)
    pairs = deepsea[["model", "run_id"]].drop_duplicates()
    if pairs.model.duplicated().any():
        raise ValueError("FigS6 has more than one selected run per model")
    add(
        "S21",
        "DeepSEA",
        "category-macro AUROC",
        "auroc",
        selected=deepsea,
        aggregation="equal-weight DHS/HM/TF mean using FigS6's existing run selection",
    )
    for tid, group in [
        ("S23", "ClinVar"),
        ("S24", "ClinVar"),
        ("S25", "ClinVar"),
        ("S26", "CAUSALdb"),
        ("S27", "Rare causal"),
        ("S28", "UKB rare variants"),
        ("S29", "MPRA"),
        ("S30", "DDD external controls"),
    ]:
        add(tid, group, "AUPRC", "auprc_mean", "dataset")
    add("S31", "DDD within site", "paired top-1 accuracy", "paired_top1", "dataset")
    data, path = table(root, "S32")
    rows.append(
        {
            "endpoint_id": "S32:retrospective ClinVar",
            "group": "Retrospective ClinVar",
            "source_table": path.name,
            "source_sha256": digest(path),
            "n_models": data.iloc[:, 0].nunique(),
            "model_roster": ";".join(sorted(data.iloc[:, 0].unique())),
            "status": "NO_PRESPECIFIED_SINGLE_WINNER_RULE",
            "comparison_caveat": "three event types across annual cutoffs; no defensible single aggregate winner inferred",
        }
    )
    add(
        "S36",
        "HAQER",
        "MCC",
        "mcc_mean",
        aggregation="mean across five negative-control replicates",
    )
    add(
        "S37",
        "HAR",
        "MCC",
        "mcc_mean",
        aggregation="mean across five negative-control replicates",
    )
    if len(rows) != 48 or len(set(r["group"] for r in rows)) != 15:
        raise ValueError("endpoint/group inventory does not reconcile")
    pd.DataFrame(rows).to_csv(output / "endpoint_ledger.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "endpoint_model_scores.csv", index=False)


def ranks(root, output):
    data, path = table(root, "S49")
    for source, part in data.groupby("source_table"):
        if set(part.source_sha256) != {digest(root / source)}:
            raise ValueError("stale S49 source: " + source)
    nmodels = data.model.nunique()
    common = data.groupby("task_unit").model.nunique()
    common = common[common.eq(nmodels)].index
    shared = data[data.task_unit.isin(common)]
    result = data.groupby("model").task_rank.agg(mean_rank="mean", n_tasks="count")
    result = result.join(
        shared.groupby("model").task_rank.agg(
            common_task_mean_rank="mean", common_task_count="count"
        )
    )
    result = result.sort_values("common_task_mean_rank")
    result.reset_index().to_csv(
        output / "common_task_rank_sensitivity.csv", index=False
    )
    save_json(
        output / "rank_inventory.json",
        {
            "models": nmodels,
            "task_units": data.task_unit.nunique(),
            "common_task_units": len(common),
            "common_task_ids": sorted(common),
            "nt_task_fraction": 18 / 45,
            "s49_sha256": digest(path),
            "description": "descriptive rank sensitivity; no statistical significance inferred",
        },
    )


def ablations(root, source_root, output, apply):
    records = []
    aliases = {
        "A01": "hEvo-fullUKB-86M",
        "A03": "hEvo-fullUKB-sliding128-86M",
        "A05": "hEvo-60k-86M",
        "A07": "hEvo-60k-sliding128-86M",
        "A09": "hEvo-1kG-86M",
        "A11": "hEvo-1kG-sliding128-86M",
    }

    def find_mcc(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("mcc"), list):
                yield obj["mcc"]
            for v in obj.values():
                yield from find_mcc(v)

    s41, p41 = table(root, "S41")
    snapshots = output / "ablation_original"
    snapshots.mkdir(exist_ok=True)
    if not (snapshots / p41.name).exists():
        (snapshots / p41.name).write_bytes(p41.read_bytes())
    s41 = pd.read_csv(snapshots / p41.name)
    for i, row in s41.iterrows():
        matched = None
        candidates = []
        for folder in ["Probe_NT", "Probe_NT_Pareto"]:
            folder = (
                source_root
                / "results"
                / folder
                / row.dataset
                / aliases[row.condition_id]
            )
            for p in folder.glob("*.json"):
                candidates.append(str(p))
                for values in find_mcc(json.loads(p.read_text())):
                    if len(values) == 3 and abs(values[0] - row.metric_mean) <= 1e-6:
                        matched = (p, values)
                        break
                if matched:
                    break
            if matched:
                break
        record = {
            "table_id": "S41",
            "condition_id": row.condition_id,
            "model_id": row.model_id,
            "dataset": row.dataset,
            "metric": row.metric,
            "published_seed": int(row.seed),
            "point_status": "SOURCE_RESULT_MATCH" if matched else "NOT_VERIFIABLE",
            "candidate_sources": candidates,
            "checkpoint_lineage_status": "NOT_VERIFIABLE",
            "training_repeats": "unknown",
            "training_budget": "unknown",
        }
        if matched:
            p, values = matched
            record.update(
                source_path=str(p),
                source_sha256=digest(p),
                source_values=values,
                ci_status="SOURCE_RESULT_MATCH"
                if np.allclose(
                    values[1:], [row.ci_lower, row.ci_upper], atol=1e-6, rtol=0
                )
                else "NOT_VERIFIABLE",
                ci_method="historical source-result bounds; generating run/bootstrap configuration not recovered",
            )
            # Bounds match a retained summary, but no generating run record:
            # preserve the estimate while withholding its uncertified interval.
            record["ci_display_status"] = "WITHHELD_NO_RUN_PROVENANCE"
            if apply:
                s41.loc[i, ["ci_lower", "ci_upper"]] = np.nan
        else:
            record.update(ci_status="NOT_VERIFIABLE", ci_display_status="WITHHELD")
            if apply:
                s41.loc[i, ["metric_mean", "ci_lower", "ci_upper"]] = np.nan
        records.append(record)
    s42, p42 = table(root, "S42")
    if not (snapshots / p42.name).exists():
        (snapshots / p42.name).write_bytes(p42.read_bytes())
    s42 = pd.read_csv(snapshots / p42.name)
    for i, row in s42.iterrows():
        p = (
            source_root
            / "results/zeroshot_Clinvar260615"
            / row.dataset
            / "hEvo-60k-sliding128-86M"
            / f"results_hEvo-60k-sliding128-86M_{row.dataset}.json"
        )
        obj = json.loads(p.read_text()) if p.is_file() else {}
        metric = row.metric.lower()
        matched = (
            row.condition_id == "A08"
            and metric in obj
            and abs(obj[metric] - row.metric_mean) <= 2e-6
        )
        record = {
            "table_id": "S42",
            "condition_id": row.condition_id,
            "model_id": row.model_id,
            "dataset": row.dataset,
            "metric": row.metric,
            "published_seed": int(row.seed),
            "point_status": "SOURCE_RESULT_MATCH" if matched else "NOT_VERIFIABLE",
            "checkpoint_lineage_status": "NOT_VERIFIABLE",
            "training_budget": "unknown",
            "ci_status": "NOT_VERIFIABLE",
            "ci_display_status": "WITHHELD_NO_RUN_PROVENANCE",
        }
        if matched:
            bounds = [
                obj.get(metric + "_bootstrap_lower"),
                obj.get(metric + "_bootstrap_upper"),
            ]
            record.update(
                source_path=str(p),
                source_sha256=digest(p),
                source_values=[obj[metric], *bounds],
                ci_status="SOURCE_RESULT_MATCH"
                if None not in bounds
                and np.allclose(bounds, [row.ci_lower, row.ci_upper], atol=2e-6, rtol=0)
                else "NOT_VERIFIABLE",
            )
            if apply:
                s42.loc[i, ["ci_lower", "ci_upper"]] = np.nan
        else:
            if apply:
                s42.loc[i, "evidence_status"] = "NOT_VERIFIABLE"
                s42.loc[
                    i,
                    [
                        "metric_mean",
                        "ci_lower",
                        "ci_upper",
                        "num_samples",
                        "num_positive",
                        "num_negative",
                        "baseline",
                    ],
                ] = np.nan
        records.append(record)
    save_json(
        output / "uncertainty_provenance.json",
        {
            "default_new_bootstrap": {
                "seed": 42,
                "repeats": 1000,
                "percentiles": [2.5, 97.5],
                "unit": "test sequence or unique variant; paired stage differences share resamples",
            },
            "note": "source-summary concordance verifies retained numbers, not a matched pretraining experiment; prediction/run lineage is still required",
            "rows": records,
        },
    )
    if apply:
        s41.to_csv(p41, index=False, na_rep="NA")
        s42.to_csv(p42, index=False, na_rep="NA")


def resources(root, source_root, output):
    detail, _ = table(root, "S44")
    records = []
    profile_records = {}
    for model in ["GPN-Star-M", "GPN-Star-P", "GPN-Star-V"]:
        p = (
            source_root
            / "runs/fig5_gpn_star_nt_resource_profiles_20261003/manifests"
            / (model + ".json")
        )
        if p.is_file():
            profile_records[model] = (p, json.loads(p.read_text()))
    for row in detail.itertuples():
        candidates = [
            root.parent / "results/Probe_NT_Pareto" / row.source_file,
            source_root / "results/Probe_NT_Pareto" / row.source_file,
        ]
        actual = next(
            (p for p in candidates if p.is_file() and digest(p) == row.source_sha256),
            None,
        )
        obj = json.loads(actual.read_text()) if actual else {}
        meta = obj.get("_run_metadata", {})
        protocol = meta.get("protocol", {})
        record = {
            "model": row.model,
            "task": row.task,
            "source_sha256": row.source_sha256,
            "source_path": str(actual) if actual else row.source_file,
            "source_status": "HASH_VERIFIED" if actual else "PUBLISHED_HASH_ONLY",
            "gpu_model": meta.get("gpu_model", "unknown"),
            "software_versions": json.dumps(meta["software_versions"])
            if "software_versions" in meta
            else "unknown",
            "dtype": protocol.get("dtype", meta.get("dtype", "unknown")),
            "actual_batch_size": protocol.get("batch_size", "unknown"),
            "input_length": row.sequence_length,
            "sample_count": "unknown",
            "warmup": "no separate warm-up in documented tracker",
            "timer_scope": protocol.get(
                "measurement",
                "documented tracker: batched representation extraction, transfers and CPU output; historical run configuration incomplete",
            ),
            "memory_scope": "incremental peak allocated CUDA memory above loaded-model baseline (MiB)",
            "comparison_status": "CONFIGURATION_SPECIFIC_NOT_CERTIFIED_MATCHED_HARDWARE",
        }
        if row.model in profile_records:
            p, profile = profile_records[row.model]
            measured = profile.get("task_measurements", {}).get(row.task, {})
            source_values = [
                measured.get(k, np.nan)
                for k in ["latency_ms", "throughput_bps", "peak_vram_mb"]
            ]
            if np.allclose(
                source_values,
                [row.latency_ms, row.throughput_bps, row.vram_mb],
                atol=1e-9,
                rtol=1e-10,
            ):
                record.update(
                    sample_count=measured.get("n_samples", "unknown"),
                    profile_manifest_path=str(p),
                    profile_manifest_sha256=digest(p),
                    sample_count_status="RECORDED_PROFILE_MATCHES_RESOURCE_VALUES",
                )
        records.append(record)
    pd.DataFrame(records).to_csv(output / "resource_run_manifest.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tables",
    )
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tables/revision_consistency",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--audit-ablations",
        action="store_true",
        help="Opt-in only: ablation work was deferred by the user on 2026-10-10",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.apply:
        protocol_table(args.tables_dir)
    endpoints(args.tables_dir, args.output_dir)
    ranks(args.tables_dir, args.output_dir)
    if args.audit_ablations:
        ablations(args.tables_dir, args.source_project, args.output_dir, args.apply)
    resources(args.tables_dir, args.source_project, args.output_dir)
    print("Evidence written to", args.output_dir)


if __name__ == "__main__":
    main()
