"""Fig.2a：7 个 Genomic Benchmark 任务的模型级 MCC matrix。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.legacy_loader import load_module  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.public_paths import publicize_paths  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import crop_drawing_clipped, export_drawing, register_arial_aliases  # noqa: E402

BACKEND = load_module("Fig2/plot_figure2_abcd_a4_integrated_v1.py")
WIDTH_MM, HEIGHT_MM = 190.0, 69.0
FIG2A_REFERENCE_MODELS = (
    "hEvo-large", "hEvo-base", "GENA-LM-large", "NT_2.5b_MS", "SpeciesLM",
    "evo2_7b_base", "AIDO.DNA-300M", "DNABERT2", "GROVER", "DNABERT",
    "Mistral-DNA-422M", "NTv3-100M-pre", "Genos-1.2B", "GPN-promoter",
    "GENERator-3B", "Caduceus-PS", "HyenaDNA_tiny_1k",
)
GPN_STAR_MODELS = {"GPN-Star-M", "GPN-Star-P", "GPN-Star-V"}


def _apply_reference_display_scope(gb_cells, gb_rows, gb_data_summary):
    """Drop GPN-Star rows and any incomplete model row from Fig.2a."""

    incomplete = set(gb_data_summary.get("incomplete_models", {}))
    selected = set(gb_rows["model"].astype(str))
    source_models = set(gb_cells["model"].astype(str))
    excluded_star = {model for model in selected if model.startswith("GPN-Star-")}
    excluded_star_series = {model for model in source_models if model.startswith("GPN-Star-")}
    excluded_incomplete = (selected & incomplete) - excluded_star
    excluded = excluded_star | excluded_incomplete
    keep = selected - excluded
    if not keep:
        raise ValueError("Fig.2a display scope removed every genomic-benchmark model")
    rows = gb_rows.loc[gb_rows["model"].astype(str).isin(keep)].copy()
    cells = gb_cells.loc[gb_cells["model"].astype(str).isin(keep)].copy()
    expected_tasks = len(BACKEND.cd.CONFIGS["gb"].tasks)
    task_counts = cells.groupby("model")["dataset"].nunique()
    if not task_counts.eq(expected_tasks).all() or cells["mcc_mean"].isna().any():
        raise ValueError("Fig.2a display scope still contains an incomplete model row")
    if rows["model"].astype(str).str.startswith("GPN-Star-").any():
        raise ValueError("Fig.2a must not display a GPN-Star model")
    report = {
        "selected_model_count": len(selected),
        "excluded_gpn_star_models": sorted(excluded_star),
        "excluded_gpn_star_series": sorted(excluded_star_series),
        "excluded_incomplete_models": sorted(excluded_incomplete),
        "displayed_model_count": int(rows["model"].nunique()),
        "displayed_rows_have_all_tasks": True,
        "displayed_gpn_star_models": [],
    }
    gb_data_summary["fig2a_display_scope"] = report
    return cells, rows, report


def build_panel(tables_dir: Path | None = None, *, panel_letter: str = "a"):
    """读取当前 Genomic/NT 表，并截取 Genomic matrix panel。"""

    root = panel_tables_dir("Fig2", tables_dir)
    BACKEND.cd.TABLE_DIR = root
    BACKEND.cd.CONFIGS["gb"] = replace(BACKEND.cd.CONFIGS["gb"], table_file="S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv")
    BACKEND.cd.CONFIGS["nt"] = replace(BACKEND.cd.CONFIGS["nt"], table_file="S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv")
    gb_cells, gb_summary, gb_data_summary = BACKEND.cd.read_and_validate(
        BACKEND.cd.CONFIGS["gb"],
        allow_incomplete=True,
    )
    nt_cells, nt_summary, nt_data_summary = BACKEND.cd.read_and_validate(BACKEND.cd.CONFIGS["nt"])
    gb_rows, _ = BACKEND.selected_with_identity(gb_summary)
    # The supplied reference uses NTv2-2.5B-MS in the genomic panel and
    # NTv2-2.5B-1kG in the companion NT panel.
    nt_identity = "NTv1-MS"
    swap_mask = gb_rows["algorithm_identity"].astype(str).eq(nt_identity)
    replacement = gb_summary.loc[gb_summary["model"].astype(str).eq("NT_2.5b_MS")].copy()
    if int(swap_mask.sum()) != 1 or len(replacement) != 1:
        raise ValueError("Fig.2a reference requires one NTv2-2.5B-MS row")
    replacement["algorithm_identity"] = nt_identity
    replacement["row_key"] = nt_identity
    gb_rows = pd.concat([gb_rows.loc[~swap_mask], replacement], ignore_index=True)
    missing_reference_models = set(FIG2A_REFERENCE_MODELS) - set(gb_rows["model"].astype(str))
    if missing_reference_models:
        raise ValueError(
            f"Fig.2a reference roster is missing selected models: {sorted(missing_reference_models)}"
        )
    gb_rows = gb_rows.loc[gb_rows["model"].astype(str).isin(FIG2A_REFERENCE_MODELS)].copy()
    gb_rows = gb_rows.sort_values(
        ["mean_mcc", "model_display"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    gb_rows["plot_rank"] = range(1, len(gb_rows) + 1)
    gb_cells, gb_rows, display_scope = _apply_reference_display_scope(
        gb_cells, gb_rows, gb_data_summary
    )
    nt_rows, _ = BACKEND.selected_with_identity(nt_summary)
    register_arial_aliases()
    panel = crop_drawing_clipped(
        BACKEND.draw_compact_cd(
            gb_cells,
            nt_cells,
            gb_rows,
            nt_rows,
            gb_letter=panel_letter,
        ),
        0.0,
        0.0,
        WIDTH_MM,
        HEIGHT_MM,
        source_clip_height_mm=68.5,
    )
    if tuple(gb_rows["model"].astype(str)) != FIG2A_REFERENCE_MODELS:
        raise ValueError("Fig.2a model order differs from the supplied PDF reference")
    panel._fig2a_display_models = FIG2A_REFERENCE_MODELS  # type: ignore[attr-defined]
    gb_data_summary["figure_scope"] = {
        "panel": "Fig.2a genomic benchmark",
        "excluded_gpn_star_series": display_scope["excluded_gpn_star_series"],
        "excluded_incomplete_model_rows": display_scope["excluded_incomplete_models"],
        "displayed_rows_have_all_seven_tasks": display_scope["displayed_rows_have_all_tasks"],
    }
    panel._benchmark_source_summary = {"genomic_benchmark": gb_data_summary, "nt_benchmark": nt_data_summary}  # type: ignore[attr-defined]
    panel._fig2a_display_scope = display_scope  # type: ignore[attr-defined]
    return panel


def _finalize_qa(result, panel) -> None:
    import json

    summary = panel._benchmark_source_summary  # type: ignore[attr-defined]
    gb, nt = summary["genomic_benchmark"], summary["nt_benchmark"]
    scope_ok = (
        gb.get("source_rows") == 232
        and gb.get("rows") == 232
        and gb.get("source_missing_cells") == 6
        and gb.get("missing_cells") == 6
        and gb.get("source_rank_mismatch_count") == 0
        and gb.get("display_excluded_cells") == []
    )
    display_scope = getattr(panel, "_fig2a_display_scope", {})
    display_ok = (
        display_scope.get("displayed_rows_have_all_tasks") is True
        and display_scope.get("displayed_gpn_star_models") == []
        and set(display_scope.get("excluded_gpn_star_series", [])) == GPN_STAR_MODELS
        and display_scope.get("displayed_model_count") == display_scope.get("selected_model_count")
        - len(display_scope.get("excluded_gpn_star_models", []))
        - len(display_scope.get("excluded_incomplete_models", []))
    )
    rank_ok = nt.get("rank_mismatch_count") == 0
    result["qa"]["benchmark_source_summary"] = summary
    result["qa"]["status"] = "PASS" if scope_ok and rank_ok and display_ok else "FAIL"
    result["qa"]["fig2a_display_scope"] = display_scope
    result["qa"]["status_note"] = (
        "Fig.2a excludes GPN-Star rows and incomplete selected model rows, so no NA cells are shown. "
        "Other benchmark panels retain their own source-supported model scope."
        if result["qa"]["status"] == "PASS"
        else "Genomic Benchmark scope or NT rank validation failed."
    )


def export_panel(output_dir: Path | None = None):
    root = panel_tables_dir()
    sources = [root / name for name in ("S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv", "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv")]
    panel = build_panel()
    result = export_drawing(panel, panel_output_dir("Fig2", output_dir), "d_nt_benchmark", input_files(*sources), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
    _finalize_qa(result, panel)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig2"))
    args = parser.parse_args()
    drawing = build_panel(args.tables_dir)
    sources = [args.tables_dir / name for name in ("S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv", "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv")]
    result = export_drawing(drawing, args.output_dir, "d_nt_benchmark", input_files(*sources), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
    _finalize_qa(result, drawing)


if __name__ == "__main__":
    main()
