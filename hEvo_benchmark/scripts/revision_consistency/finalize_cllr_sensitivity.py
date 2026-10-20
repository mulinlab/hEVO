#!/usr/bin/env python3
"""Reconcile candidate groups, add allele-preference sensitivity and plot it."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_cllr_sensitivity import paired_top1, write_frame, publication_summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))
from calibrate_gpn_star_scores import Reference, calibrate, sha256

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "adaptive_allele_preference_benchmark")
)
from allele_direction import orient_alleles


def run(args):
    repo = Path(__file__).resolve().parents[2]
    output = args.output_dir
    manifest_path = output / "sensitivity_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    primary = pd.read_csv(next((repo / "tables").glob("S31_*.csv"))).set_index("model")
    for record in manifest:
        if record.get("table_id") != "S31" or "calibrated_scores" not in record:
            continue
        frame = pd.read_csv(record["calibrated_scores"], sep="\t")
        valid = frame[frame.calibration_status.eq("PASS")].copy()
        ids = valid.variant_id.str.split("__samepos_other_", n=1).str[0]
        valid = valid[valid.groupby(ids).variant_id.transform("size").eq(3)]
        raw, n = paired_top1(valid, "raw_benchmark_score")
        cllr, _ = paired_top1(valid, "calibrated_benchmark_score")
        row = primary.loc[record["model"]]
        match = bool(
            n == row.groups_valid
            and np.isclose(raw, row.paired_top1, atol=2e-6, rtol=0)
        )
        record.update(
            raw_paired_top1=raw,
            cllr_paired_top1=cllr,
            paired_groups=n,
            raw_reproduces_primary=match,
            primary_metric="prespecified positive-variant candidate-trio top-1 accuracy",
            candidate_group_rule="variant_id before __samepos_other_; distinct positives can share coordinates",
            status="COMPLETE" if match else "RAW_PRIMARY_RECONCILIATION_FAILED",
        )
        Path(record["calibrated_scores"]).with_suffix(".manifest.json").write_text(
            json.dumps(record, indent=2)
        )
    cfg = yaml.safe_load((repo / "configs/gpn_star.yaml").read_text())["models"]
    s38 = pd.read_csv(next((repo / "tables").glob("S38_*.csv"))).set_index("model_id")
    s39 = pd.read_csv(next((repo / "tables").glob("S39_*.csv")))
    ref = Reference(args.reference_fasta)
    try:
        for model, settings in cfg.items():
            record = {"model": model, "table_id": "S38/S39"}
            try:
                source = (
                    args.source_project
                    / "runs/fig4_gpn_raw_scores_20260919/scores"
                    / (model + ".tsv")
                )
                frame = orient_alleles(pd.read_csv(source, sep="\t"))
                comparison = frame.merge(
                    s39[s39.model_id.eq(model)][["variant_id", "locus_score"]],
                    on="variant_id",
                    suffixes=("", "_table"),
                    validate="one_to_one",
                )
                if len(comparison) != 30 or not np.allclose(
                    comparison.locus_score,
                    comparison.locus_score_table,
                    atol=1e-6,
                    rtol=0,
                ):
                    raise ValueError(
                        "retained native scores do not reproduce the S39 locus roster"
                    )
                frame["raw_llr"] = frame.locus_score * np.where(
                    frame.selected_allele.eq(frame.alt), 1, -1
                )
                frame["score_mode"] = "selected_other"
                frame["status"] = np.where(
                    frame.benchmark_status.eq("PASS"), "PASS", "UNAVAILABLE"
                )
                calibration_path = output / "calibration" / model / "llr.parquet"
                receipt = json.loads(
                    calibration_path.with_suffix(".provenance.json").read_text()
                )
                if receipt["revision"] != settings["revision"] or receipt[
                    "sha256"
                ] != sha256(calibration_path):
                    raise ValueError("calibration receipt mismatch")
                scores = calibrate(
                    frame,
                    pd.read_parquet(calibration_path),
                    ref,
                    model,
                    settings["revision"],
                )
                eligible = scores.calibration_status.eq("PASS")
                n = int(eligible.sum())
                raw_k = int(scores.loc[eligible, "raw_benchmark_score"].gt(0).sum())
                cllr_k = int(
                    scores.loc[eligible, "calibrated_benchmark_score"].gt(0).sum()
                )
                match = bool(
                    n == s38.loc[model, "n_loci"]
                    and raw_k == s38.loc[model, "positive_loci"]
                )
                path = output / "scores" / model / "S38.cllr.tsv.gz"
                write_frame(scores, path)
                record.update(
                    status="COMPLETE" if match else "RAW_PRIMARY_RECONCILIATION_FAILED",
                    raw_reproduces_primary=match,
                    calibrated_rows=n,
                    raw_positive_loci=raw_k,
                    cllr_positive_loci=cllr_k,
                    raw_preference_fraction=raw_k / n,
                    cllr_preference_fraction=cllr_k / n,
                    cllr_preference_ci_low=float(
                        beta.ppf(0.025, cllr_k, n - cllr_k + 1)
                    )
                    if cllr_k
                    else 0.0,
                    cllr_preference_ci_high=float(
                        beta.ppf(0.975, cllr_k + 1, n - cllr_k)
                    )
                    if cllr_k < n
                    else 1.0,
                    source_path=str(source),
                    source_sha256=sha256(source),
                    scores_sha256=sha256(path),
                    derivation="raw ALT-minus-REF LLR recovered from native selected-minus-other log-probability difference and prespecified allele orientation",
                )
            except Exception as exc:
                record.update(status="UNAVAILABLE", reason=str(exc))
            manifest = [
                r
                for r in manifest
                if not (r.get("model") == model and r.get("table_id") == "S38/S39")
            ]
            manifest.append(record)
    finally:
        ref.close()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    summary = publication_summary(
        manifest, [("source", args.source_project), ("output", output), ("repo", repo)]
    )
    summary.to_csv(output / "cllr_sensitivity_summary.csv", index=False)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    titles = {
        "S23": "ClinVar",
        "S26": "CAUSALdb",
        "S27": "Rare causal",
        "S28": "UKB rare variants",
        "S29": "MPRA",
        "S30": "DDD external controls",
        "S31": "DDD candidate-trio top-1",
        "S38/S39": "Allele preference (30 loci)",
    }
    colors = {"GPN-Star-M": "#347A44", "GPN-Star-P": "#305CB4", "GPN-Star-V": "#9A499A"}
    markers = {"GPN-Star-M": "s", "GPN-Star-P": "^", "GPN-Star-V": "o"}
    line_styles = {"GPN-Star-M": "-", "GPN-Star-P": "--", "GPN-Star-V": ":"}
    for ax, (tid, title) in zip(axes.flat, titles.items()):
        part = summary[(summary.table_id == tid) & summary.status.eq("COMPLETE")]
        raw_column, cal_column = (
            ("raw_paired_top1", "cllr_paired_top1")
            if tid == "S31"
            else (
                ("raw_preference_fraction", "cllr_preference_fraction")
                if tid == "S38/S39"
                else ("raw_auprc", "cllr_auprc")
            )
        )
        for _, row in part.iterrows():
            ax.plot(
                [0, 1],
                [row[raw_column], row[cal_column]],
                marker=markers[row.model],
                linestyle=line_styles[row.model],
                color=colors[row.model],
                linewidth=1.1,
                label=row.model,
            )
        ax.set_title(title)
        ax.set_xticks([0, 1], ["raw", "cLLR"])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("AUPRC" if tid not in {"S31", "S38/S39"} else "Fraction")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
    )
    fig.suptitle(
        "GPN-Star calibration sensitivity: identical eligible variants within each comparison",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "Descriptive point comparisons; main raw tables unchanged. Dataset/head protocols and model coverage differ between tasks.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output / "cllr_sensitivity.pdf")
    fig.savefig(output / "cllr_sensitivity.png", dpi=180)
    plt.close(fig)
    print(summary.groupby(["table_id", "status"]).size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tables/revision_consistency",
    )
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    run(parser.parse_args())
