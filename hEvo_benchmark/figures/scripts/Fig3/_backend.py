"""Fig.3 重构版共享数据适配层。

这里仅把五类静态 source-data 合并成当前 v2 绘图函数需要的内存表，
并补上原 read_sources() 计算的 event_prevalence；所有数值都来自 GitLab tables
的独立副本，不在此重新计算模型指标。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.legacy_loader import load_module
from common.model_names import canonical_model_name
from common.panel_runtime import panel_tables_dir


BACKEND = load_module("Fig3/plot_figure3_abcdefghijk_a4_integrated_v2.py")


def _family(model: str) -> str:
    name = str(model)
    if name.startswith("hEvo-"): return "hEVO"
    if name.startswith(("NT_", "NTv")): return "NT"
    if name.startswith("evo2_"): return "Evo 2"
    if name.startswith("GPN-Star-"): return "GPN-Star"
    if name.startswith("GPN"): return "GPN"
    return "Other"

def _display(model: str) -> str:
    return canonical_model_name(model)

def _normalise_benchmark(path: Path, task: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model", "auprc_mean", "auprc_ci_lower", "auprc_ci_upper",
                "num_samples", "num_positive", "num_negative", "baseline"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["task"] = task
    frame["score"] = pd.to_numeric(frame["auprc_mean"], errors="raise")
    frame["ci_lower"] = pd.to_numeric(frame["auprc_ci_lower"], errors="raise")
    frame["ci_upper"] = pd.to_numeric(frame["auprc_ci_upper"], errors="raise")
    frame["baseline_value"] = pd.to_numeric(frame["baseline"], errors="raise")
    for column in ("num_samples", "num_positive", "num_negative"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["num_samples"] <= 0).any():
        raise ValueError(f"{path} has a nonpositive sample count")
    if not (frame["num_positive"] + frame["num_negative"]).eq(frame["num_samples"]).all():
        raise ValueError(f"{path} has inconsistent positive/negative/sample counts")
    if not (frame["baseline_value"] - frame["num_positive"] / frame["num_samples"]).abs().le(1e-12).all():
        raise ValueError(f"{path} baseline does not equal each model's positive-class fraction")
    frame["model_display"] = frame["model"].map(_display)
    frame["family"] = frame["model"].map(_family)
    frame["task_rank"] = frame.groupby("task")["score"].rank(method="min", ascending=False).astype(int)
    frame["within_view_rank"] = frame["task_rank"]
    frame["mean_auprc"] = frame["score"]
    return frame

STAR_MODELS = frozenset({"GPN-Star-M", "GPN-Star-P", "GPN-Star-V"})
LEGACY_REPRESENTATIVES = frozenset({
    "hEvo-base", "hEvo-large", "GPN-MSA", "GPN-promoter", "evo2_7b_base",
    "NTv3-650M-pre", "AIDO.DNA-300M", "NT_2.5b_MS", "SpeciesLM",
    "Genos-1.2B", "GENERator-3B", "GENA-LM-large", "DNABERT2",
    "Mistral-DNA-422M", "Caduceus-PH", "HyenaDNA_tiny_1k", "GROVER",
})
PLOT_REPRESENTATIVES = LEGACY_REPRESENTATIVES - {"GPN-promoter"}


def _best_star_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one deterministic, highest-AUPRC GPN-Star row per task."""

    stars = frame.loc[frame["model"].isin(STAR_MODELS)].copy()
    if stars.empty:
        return stars
    return (
        stars.sort_values(["task", "score", "model"], ascending=[True, False, True], kind="stable")
        .groupby("task", sort=False, as_index=False, group_keys=False)
        .head(1)
    )


def load_sources(data_dir: Path | None = None, *, static_name: str = "S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv",
                 temporal_name: str = "S32_retrospective_enrichment_of_clinvar_reclassification_events_by_cutoff_year.csv"):
    """Read the canonical S26-S36 tables and derive only display aliases in memory."""
    root = panel_tables_dir("Fig3", data_dir)
    clinvar_files = [
        ("all", root / "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv"),
        ("coding", root / "S24_zero_shot_classification_of_coding_clinvar_variants.csv"),
        ("noncoding", root / "S25_zero_shot_classification_of_noncoding_clinvar_variants.csv"),
    ]
    static_files = [
        ("causal", root / "S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv"),
        ("rare_causal", root / "S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv"),
        ("rare_ukb", root / "S28_zero_shot_classification_of_uk_biobank_rare_variants.csv"),
        ("mpra", root / "S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv"),
        ("ddd_clinvar", root / "S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv"),
    ]
    clinvar = pd.concat([_normalise_benchmark(path, task) for task, path in clinvar_files], ignore_index=True)
    # The consolidated S23 is the single source for all ClinVar models.
    star_clinvar = clinvar.loc[
        clinvar["task"].eq("all") & clinvar["model"].isin(STAR_MODELS)
    ].copy()
    clinvar = pd.concat([
        clinvar.loc[clinvar["task"].ne("all") & clinvar["model"].isin(LEGACY_REPRESENTATIVES)],
        clinvar.loc[clinvar["task"].eq("all") & clinvar["model"].isin(PLOT_REPRESENTATIVES)],
        _best_star_rows(star_clinvar),
    ], ignore_index=True)
    static = pd.concat([_normalise_benchmark(path, task) for task, path in static_files], ignore_index=True)
    static = pd.concat([
        static.loc[static["model"].isin(PLOT_REPRESENTATIVES)],
        _best_star_rows(static),
    ], ignore_index=True)
    for task in BACKEND.core.CLINVAR_META:
        if len(clinvar.loc[clinvar["task"].eq(task)]) != 17:
            raise ValueError(f"Fig3 ClinVar representative contract failed for {task}")
    for task in BACKEND.core.STATIC_META:
        if len(static.loc[static["task"].eq(task)]) != 17:
            raise ValueError(f"Fig3 static representative contract failed for {task}")
    temporal_path = root / temporal_name
    temporal = pd.read_csv(temporal_path)
    required = {"model_label", "task_name", "cutoff_year", "precision_at_k_mean", "enrichment_vs_baseline_mean"}
    if not required.issubset(temporal.columns):
        raise ValueError(f"{temporal_path} is missing columns: {sorted(required - set(temporal.columns))}")
    temporal = temporal.copy()
    temporal["event_prevalence"] = temporal["precision_at_k_mean"] / temporal["enrichment_vs_baseline_mean"]
    if set(clinvar["task"]) != set(BACKEND.core.CLINVAR_META):
        raise ValueError("Fig3 ClinVar task set mismatch")
    if set(static["task"]) != set(BACKEND.core.STATIC_META):
        raise ValueError("Fig3 static task set mismatch")
    return clinvar, static, temporal


def static_checks(static: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """构造 v2 绘图所需的样本量/基线检查字典。"""

    checks: dict[str, dict[str, float | int]] = {}
    for task in BACKEND.core.STATIC_META:
        first = static.loc[static["task"].eq(task)].iloc[0]
        checks[task] = {
            "num_samples": int(first.get("num_samples", 0)),
            "num_positive": int(first.get("num_positive", 0)),
            "num_negative": int(first.get("num_negative", 0)),
            "baseline": float(first["baseline_value"]),
        }
    return checks


def placement_table(data_dir: Path | None = None, *, name: str = "S22_schematic_coordinates_for_variant_effect_and_allele_frequency_benchmarks.csv") -> pd.DataFrame:
    """读取 architecture placement 表；a/b 使用同一真实 placement 快照。"""

    root = panel_tables_dir("Fig3", data_dir)
    table = pd.read_csv(root / name)
    required = {"dataset", "panel", "class", "allele_frequency", "vertical_coordinate"}
    if not required.issubset(table.columns):
        raise ValueError("Fig3 architecture placement fields are incomplete")
    # Keep the observed frequencies, intervals, and original coordinates from S22 as source data. Plot positions follow the reference PDF
    # category axis for illustration and must not be interpreted as observed allele frequencies.
    table = table.copy()
    table["source_panel"] = table["panel"]
    table["display_panel_letter"] = table["panel"].map({"a": "b", "b": "a"})
    table["source_allele_frequency"] = pd.to_numeric(table["allele_frequency"], errors="raise")
    schematic_coordinates = [
        BACKEND.REFERENCE_ARCH_COORDS[str(row["panel"])][str(row["dataset"])][str(row["class"])]
        for _, row in table.iterrows()
    ]
    table["display_x_fraction"] = [coordinate[0] for coordinate in schematic_coordinates]
    table["display_y_fraction"] = [coordinate[1] for coordinate in schematic_coordinates]
    return table
