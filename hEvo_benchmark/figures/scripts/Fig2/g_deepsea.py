"""Fig.2f: compact DeepSEA comparison with a dedicated GPN-Star-M row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.legacy_loader import load_module  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, register_arial_aliases  # noqa: E402
from common.public_paths import publicize_paths  # noqa: E402
from common.model_names import canonical_model_name  # noqa: E402

BACKEND = load_module("Fig2/plot_figure2_abcdef_a4_integrated_v1.py")
WIDTH_MM, HEIGHT_MM = BACKEND.G_W_MM, BACKEND.G_H_MM
SOURCE_NAME = "S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv"


def model_family(model: str) -> str:
    value = str(model).lower()
    if value.startswith("hevo"):
        return "hEVO"
    if value.startswith("evo2") or value.startswith("evo 2"):
        return "Evo 2"
    if value.startswith("nt"):
        return "NT"
    if value.startswith("gpn"):
        return "GPN"
    return "Other"


FAMILY_QUOTAS = {"hEVO": 2, "NT": 3, "Evo 2": 1, "GPN": 2, "Other": 4}
FORCED_MODELS = {"HyenaDNA_tiny_1k": "author-requested inclusion in the Fig.2g main plot"}


def select_family_representatives(plot):
    """Keep the family roster and explicitly include author-requested models."""

    frame = plot.copy()
    frame["model_family"] = frame["model"].map(model_family)
    selected_parts = []
    for family in ("hEVO", "NT", "Evo 2", "GPN", "Other"):
        family_frame = frame.loc[frame["model_family"].eq(family)].sort_values(
            ["recomputed_mean", "display_model"], ascending=[False, True], kind="stable"
        )
        quota = FAMILY_QUOTAS[family]
        if len(family_frame) < quota:
            raise ValueError(f"S21 family {family} has {len(family_frame)} models; need {quota}")
        if family == "hEVO":
            # Preserve the main-panel primary base/large roster. Public-data
            # variants are reported in FigS6, so selecting run24 must not
            # silently replace the primary base model by its test ranking.
            primary = family_frame.loc[family_frame.model.isin({"hEvo-base", "hEvo-large"})]
            if set(primary.model) != {"hEvo-base", "hEvo-large"}:
                raise ValueError("Fig2f requires both primary hEvo scales")
            selected_parts.append(primary)
        else:
            selected_parts.append(family_frame.head(quota))
    selected = pd.concat(selected_parts, ignore_index=True)
    forced = frame.loc[frame["model"].isin(FORCED_MODELS)].copy()
    if not forced.empty:
        forced = forced.loc[~forced["model"].isin(set(selected["model"]))]
        selected = pd.concat([selected, forced], ignore_index=True)
    selected = selected.sort_values(
        ["recomputed_mean", "display_model"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    selected["overall_rank"] = range(1, len(selected) + 1)
    summary = {
        "selection_rule": "fixed family quotas; retain the primary hEvo-base/hEvo-large main-panel roster, select top mean-AUROC models within other families, and append explicitly requested models; public-data variants are reported in FigS6",
        "family_quotas": FAMILY_QUOTAS,
        "forced_models": FORCED_MODELS,
        "source_models": int(frame["model"].nunique()),
        "plotted_models": int(selected["model"].nunique()),
        "families": {
            family: {
                "source_models": int(group["model"].nunique()),
                "selected_models": [str(x) for x in selected.loc[selected["model_family"].eq(family), "display_model"]],
            }
            for family, group in frame.groupby("model_family", sort=False)
        },
    }
    return selected, summary


def load_all_models(tables_dir: Path | None = None):
    """Load and aggregate all S21 model rows for reuse by supplementary panels."""

    root = panel_tables_dir("Fig2", tables_dir)
    BACKEND.deepsea.SOURCE_TABLE = root / SOURCE_NAME
    _, plot, source_summary = BACKEND.deepsea.prepare_data()
    plot = plot.copy()
    plot["display_model"] = plot["model"].map(canonical_model_name)
    return root, plot, source_summary


def _load_selected(tables_dir_arg: Path | None):
    root, plot, source_summary = load_all_models(tables_dir_arg)
    selected, selection_summary = select_family_representatives(plot)
    return root, selected, source_summary, selection_summary


def build_panel(tables_dir: Path | None = None, *, panel_letter: str = "f"):
    """Build the compact panel from current S21 with family-level selection."""

    _, selected, _, _ = _load_selected(tables_dir)
    register_arial_aliases()
    return BACKEND.draw_compact_g(selected, panel_letter=panel_letter)


def _export(output_dir: Path, tables_dir_arg: Path | None):
    root, selected, source_summary, selection_summary = _load_selected(tables_dir_arg)
    register_arial_aliases()
    result = export_drawing(
        BACKEND.draw_compact_g(selected, panel_letter="f"),
        panel_output_dir("Fig2", output_dir),
        "g_deepsea",
        input_files(root / SOURCE_NAME),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
    )
    payload = result["qa"]
    payload["source_summary"] = source_summary
    payload["family_selection"] = selection_summary
    payload["display_metric"] = "mean AUROC"
    payload["task_markers"] = "omitted in Fig.2g; retained in FigS4-c"
    return result


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    return _export(output_dir or panel_output_dir("Fig2"), tables_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig2"))
    args = parser.parse_args()
    _export(args.output_dir, args.tables_dir)


if __name__ == "__main__":
    main()
