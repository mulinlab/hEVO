#!/usr/bin/env python3
"""Read-only reader for the Probe_NT_Pareto JSON result tree."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = PROJECT_ROOT / "figures" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from common.model_names import canonical_model_name

DEFAULT_PARETO_DIR = Path(
    os.environ.get("HEVO_PARETO_DIR", str(PROJECT_ROOT / "results" / "Probe_NT_Pareto"))
)
HEVO_VARIANTS = {
    "hEvo-1kG-86M",
    "hEvo-1kG-sliding128-86M",
    "hEvo-60k-86M",
    "hEvo-base",
    "hEvo-large",
    "hEvo-fullUKB-86M",
}
PLOT_HEVO = {"hEvo-base", "hEvo-large"}
FAMILY_MAP = {
    "hEvo": sorted(HEVO_VARIANTS),
    "NT v2": ["NT_2.5b_1kG", "NT_2.5b_MS", "NT_500m_1kG", "NT_500m_MS"],
    "NT v3": ["NTv3-650M-pre", "NTv3-100M-pre", "NTv3-8M-pre"],
    "DNABert": ["DNABERT", "DNABERT2"],
    "GENA-LM": ["GENA-LM-base", "GENA-LM-large"],
    "Mistral-DNA": ["Mistral-DNA-17M", "Mistral-DNA-138M", "Mistral-DNA-422M"],
    "Caduceus": ["Caduceus-PH", "Caduceus-PS"],
    "GPN": ["GPN-promoter", "GPN-MSA", "GPN-Star-M", "GPN-Star-P", "GPN-Star-V"],
    "GENERator": ["GENERator-1.2B", "GENERator-3B"],
    "Evo 2": ["evo2_1b_base", "evo2_7b_base"],
    "SpeciesLM": ["SpeciesLM"],
    "HyenaDNA": ["HyenaDNA_tiny_1k"],
    "AIDO.DNA": ["AIDO.DNA-300M"],
    "Genos": ["Genos-1.2B"],
    "GROVER": ["GROVER"],
}
MODEL_TO_FAMILY = {model: family for family, models in FAMILY_MAP.items() for model in models}
FAMILY_COLORS = {
    "hEvo": "#B35C1E",
    "NT v2": "#65717F",
    "NT v3": "#7D8793",
    "DNABert": "#9AA4AF",
    "GENA-LM": "#8792A0",
    "Mistral-DNA": "#A7AFB8",
    "Caduceus": "#9AA6B5",
    "GPN": "#0B2E83",
    "GENERator": "#7F8B99",
    "Evo 2": "#0E7C86",
    "SpeciesLM": "#8994A0",
    "HyenaDNA": "#A1AAB5",
    "AIDO.DNA": "#8C98A5",
    "Genos": "#7F8B99",
    "GROVER": "#A0A9B3",
    "Other": "#8B96A5",
}


def display_name(model: str) -> str:
    return canonical_model_name(model)


def family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, FAMILY_COLORS["Other"])


def _read_rows(root: Path) -> tuple[list[dict], set[str], set[str]]:
    rows: list[dict] = []
    tasks: set[str] = set()
    models: set[str] = set()
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task = task_dir.name
        tasks.add(task)
        for model_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            json_path = model_dir / f"results_{model_dir.name}_{task}.json"
            if not json_path.is_file():
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            for model_key, content in payload.items():
                task_record = content.get(task, {})
                result = task_record.get("no_projection", {})
                if not result:
                    continue
                perf = result.get("performance", {})
                eff = result.get("efficiency", {})
                mcc = (perf.get("mcc") or [np.nan])[0]
                model = model_dir.name
                models.add(model)
                rows.append({
                    "model": model,
                    "task": task,
                    "mcc": pd.to_numeric(mcc, errors="coerce"),
                    "params_m": pd.to_numeric(eff.get("total_params_M", np.nan), errors="coerce"),
                    "throughput_bps": pd.to_numeric(eff.get("backbone_throughput_bps", np.nan), errors="coerce"),
                    "latency_ms": pd.to_numeric(eff.get("backbone_latency_ms", np.nan), errors="coerce"),
                    "vram_mb": pd.to_numeric(eff.get("backbone_peak_vram_mb", np.nan), errors="coerce"),
                })
    return rows, tasks, models


def collect_pareto_summary(pareto_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(pareto_dir).resolve() if pareto_dir is not None else DEFAULT_PARETO_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Probe_NT_Pareto directory not found: {root}")
    rows, tasks, raw_models = _read_rows(root)
    frame = pd.DataFrame(rows)
    expected_tasks = sorted(tasks)
    if len(expected_tasks) != 18:
        raise ValueError(f"Probe_NT_Pareto expected 18 tasks; got {len(expected_tasks)}")
    frame = frame[~frame["model"].isin(HEVO_VARIANTS - PLOT_HEVO)].copy()
    kept = sorted(frame["model"].unique())
    unregistered = sorted(set(kept) - set(MODEL_TO_FAMILY))
    if unregistered:
        raise ValueError(f"Pareto source contains unregistered models: {unregistered}")
    numeric = ["mcc", "params_m", "throughput_bps", "latency_ms", "vram_mb"]
    summaries = []
    for model, group in frame.groupby("model", sort=False):
        valid_mcc = group["mcc"].notna()
        record = {"model": model, "family": MODEL_TO_FAMILY.get(model, "Other"), "mean_mcc": group.loc[valid_mcc, "mcc"].mean(), "n_perf": int(valid_mcc.sum()), "n_tasks": len(expected_tasks)}
        record["missing_tasks"] = ";".join(sorted(set(expected_tasks) - set(group.loc[valid_mcc, "task"]))) or "none"
        for column in numeric[1:]:
            vals = group[column].dropna()
            record[column] = float(vals.median()) if len(vals) else np.nan
            record[f"n_{column}"] = int(len(vals))
        record["n_eff"] = int(min(record[f"n_{column}"] for column in numeric[1:]))
        summaries.append(record)
    summary = pd.DataFrame(summaries).sort_values(["mean_mcc", "model"], ascending=[False, True], kind="stable").reset_index(drop=True)
    details = {
        "raw_files": len(list(root.glob("*/*/*.json"))),
        "raw_models": len(raw_models),
        "raw_tasks": len(expected_tasks),
        "display_models": len(summary),
        "excluded_hEvo_variants": sorted(HEVO_VARIANTS - PLOT_HEVO),
        "display_hEvo_ids": sorted(PLOT_HEVO),
        "display_names": {model: display_name(model) for model in sorted(PLOT_HEVO)},
        "incomplete_models": summary.loc[summary["n_perf"] < 18, ["model", "n_perf", "missing_tasks"]].to_dict("records"),
        "efficiency_incomplete_models": summary.loc[summary["n_eff"] < 18, ["model", "n_eff"]].to_dict("records"),
        "source_dir": str(root),
    }
    return summary, details


def pareto_frontier(x: np.ndarray, y: np.ndarray, x_higher_is_better: bool) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)[::-1] if not x_higher_is_better else np.argsort(x)
    current = -np.inf
    points = []
    for index in order:
        if y[index] > current:
            points.append((x[index], y[index]))
            current = y[index]
    points.sort(key=lambda pair: pair[0])
    return np.array([pair[0] for pair in points]), np.array([pair[1] for pair in points])
