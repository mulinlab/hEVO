"""Fig.2b：18 个 Nucleotide Transformer 任务的模型级 MCC matrix。"""

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
from common.reportlab_common import crop_drawing_clipped, export_drawing, mm_drawing, pad_drawing, register_arial_aliases  # noqa: E402

BACKEND = load_module("Fig2/plot_figure2_abcd_a4_integrated_v1.py")
WIDTH_MM, HEIGHT_MM = 190.0, 73.0
STANDALONE_VERTICAL_PAD_MM = (4.0, 5.0)  # bottom, top
HUMAN_TASKS = (
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
)
DEMO_TASKS = ("demo_coding_vs_intergenomic_seqs", "demo_human_or_worm")
GPN_STAR_MODELS = ("GPN-Star-M", "GPN-Star-P", "GPN-Star-V")
FIG2B_DISPLAY_MODELS = (
    "hEvo-base", "evo2_7b_base", "hEvo-large", "GPN-Star-M", "GENA-LM-large",
    "NT_2.5b_1kG", "SpeciesLM", "AIDO.DNA-300M", "DNABERT2", "GPN-MSA",
    "DNABERT", "Genos-1.2B", "Mistral-DNA-422M", "GROVER", "NTv3-100M-pre",
    "Caduceus-PS", "GPN-promoter", "GENERator-1.2B", "HyenaDNA_tiny_1k",
)


def _annotate_source_summary(summary: dict) -> tuple[bool, bool]:
    gb_summary = summary["genomic_benchmark"]
    expected_incomplete = {
        **{model: sorted(DEMO_TASKS) for model in GPN_STAR_MODELS},
    }
    actual_incomplete = {
        model: sorted(tasks)
        for model, tasks in gb_summary.get("incomplete_models", {}).items()
    }
    human_counts = gb_summary.get("task_model_counts", {})
    source_scope_ok = (
        gb_summary.get("source_rows") == 232
        and gb_summary.get("rows") == 232
        and gb_summary.get("models") == 34
        and gb_summary.get("tasks") == 7
        and gb_summary.get("duplicate_cells") == 0
        and gb_summary.get("source_missing_cells") == len(GPN_STAR_MODELS) * len(DEMO_TASKS)
        and gb_summary.get("missing_cells") == len(GPN_STAR_MODELS) * len(DEMO_TASKS)
        and gb_summary.get("source_rank_mismatch_count") == 0
        and actual_incomplete == expected_incomplete
        and all(human_counts.get(task) == 34 for task in HUMAN_TASKS)
        and all(human_counts.get(task) == 31 for task in DEMO_TASKS)
        and gb_summary.get("display_excluded_cells") == []
    )
    nt_rank_ok = summary["nt_benchmark"].get("rank_mismatch_count") == 0
    gb_summary["scope_summary"] = {
        "coordinate_supported_human_tasks": list(HUMAN_TASKS),
        "gpn_star_missing_demo_tasks": list(DEMO_TASKS),
        "gpn_star_models_with_human_results": list(GPN_STAR_MODELS),
        "gpn_msa_demo_task_values_present": True,
        "source_rows": gb_summary.get("source_rows"),
        "source_missing_cells": gb_summary.get("source_missing_cells"),
        "expected_missing_cells": len(GPN_STAR_MODELS) * len(DEMO_TASKS),
        "expected_missing_by_model": expected_incomplete,
        "display_excluded_cells": gb_summary.get("display_excluded_cells", []),
        "scope_valid": source_scope_ok,
    }
    summary["nt_benchmark"]["rank_consistency_valid"] = nt_rank_ok
    return source_scope_ok, nt_rank_ok


def _drawing(root: Path, panel_letter: str):
    BACKEND.cd.TABLE_DIR = root
    BACKEND.cd.CONFIGS["gb"] = replace(BACKEND.cd.CONFIGS["gb"], table_file="S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv")
    BACKEND.cd.CONFIGS["nt"] = replace(BACKEND.cd.CONFIGS["nt"], table_file="S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv")
    gb_cells, gb_summary, gb_data_summary = BACKEND.cd.read_and_validate(
        BACKEND.cd.CONFIGS["gb"],
        allow_incomplete=True,
    )
    nt_cells, nt_summary, nt_data_summary = BACKEND.cd.read_and_validate(BACKEND.cd.CONFIGS["nt"])
    gb_rows, _ = BACKEND.selected_with_identity(gb_summary)
    nt_rows, _ = BACKEND.selected_with_identity(nt_summary)
    nt_rows = nt_rows.sort_values(
        ["mean_mcc", "model_display"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    nt_rows["plot_rank"] = range(1, len(nt_rows) + 1)
    full = BACKEND.draw_compact_cd(
        gb_cells, nt_cells, gb_rows, nt_rows, nt_letter=panel_letter
    )
    panel = crop_drawing_clipped(full, 0.0, 69.0, WIDTH_MM, HEIGHT_MM)
    if tuple(nt_rows["model"].astype(str)) != FIG2B_DISPLAY_MODELS:
        raise ValueError("Fig.2b model order differs from the current S17 data")
    panel._fig2b_display_models = FIG2B_DISPLAY_MODELS  # type: ignore[attr-defined]
    source_summary = {"genomic_benchmark": gb_data_summary, "nt_benchmark": nt_data_summary}
    _annotate_source_summary(source_summary)
    panel._benchmark_source_summary = source_summary  # type: ignore[attr-defined]
    return panel


def build_panel(tables_dir: Path | None = None, *, panel_letter: str = "b"):
    """读取当前 Genomic/NT 表，并截取 NT matrix panel。"""

    root = panel_tables_dir("Fig2", tables_dir)
    register_arial_aliases()
    return _drawing(root, panel_letter)


def export_panel(output_dir: Path | None = None):
    root = panel_tables_dir()
    sources = [root / name for name in ("S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv", "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv")]
    panel = build_panel()
    bottom_pad, top_pad = STANDALONE_VERTICAL_PAD_MM
    source_panel = panel
    panel = pad_drawing(source_panel, bottom_mm=bottom_pad, top_mm=top_pad)
    panel._benchmark_source_summary = source_panel._benchmark_source_summary  # type: ignore[attr-defined]
    result = export_drawing(panel, panel_output_dir("Fig2", output_dir), "c_genomic_benchmark", input_files(*sources), width_mm=WIDTH_MM, height_mm=HEIGHT_MM + bottom_pad + top_pad)
    _finalize_qa(result, panel)
    return result


def _finalize_qa(result, panel) -> None:
    """Record the expected GPN-Star demo omissions and gate current source data."""

    import json

    summary = panel._benchmark_source_summary  # type: ignore[attr-defined]
    source_scope_ok, nt_rank_ok = _annotate_source_summary(summary)
    result["qa"]["benchmark_source_summary"] = summary
    result["qa"]["status"] = "PASS" if source_scope_ok and nt_rank_ok else "FAIL"
    result["qa"]["status_note"] = (
        "S18 source coverage and S17 reported task ranks pass validation. The companion Fig.2a panel "
        "omits GPN-Star and incomplete model rows, leaving complete seven-task rows only."
        if result["qa"]["status"] == "PASS"
        else "Source scope or rank validation failed; inspect benchmark_source_summary."
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig2"))
    args = parser.parse_args()
    source_panel = build_panel(args.tables_dir)
    bottom_pad, top_pad = STANDALONE_VERTICAL_PAD_MM
    drawing = pad_drawing(source_panel, bottom_mm=bottom_pad, top_mm=top_pad)
    drawing._benchmark_source_summary = source_panel._benchmark_source_summary  # type: ignore[attr-defined]
    sources = [args.tables_dir / name for name in ("S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv", "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv")]
    result = export_drawing(drawing, args.output_dir, "c_genomic_benchmark", input_files(*sources), width_mm=WIDTH_MM, height_mm=HEIGHT_MM + bottom_pad + top_pad)
    _finalize_qa(result, drawing)


if __name__ == "__main__":
    main()
