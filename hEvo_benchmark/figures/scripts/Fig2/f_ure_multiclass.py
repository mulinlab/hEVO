"""Fig.2d：URE 五分类 F1 compact spectrum。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.legacy_loader import load_module  # noqa: E402
from common.model_names import canonical_model_name  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, pad_drawing, register_arial_aliases  # noqa: E402
from common.public_paths import publicize_paths  # noqa: E402

BACKEND = load_module("Fig2/plot_figure2_abcdef_a4_integrated_v1.py")
WIDTH_MM, HEIGHT_MM = BACKEND.F_W_MM, BACKEND.F_H_MM
STANDALONE_BOTTOM_PAD_MM = 16.0
REFERENCE_MODEL_ORDER = (
    "hEvo-large",
    "GPN-Star-P",
    "hEvo-base",
    "Genos-1.2B",
    "SpeciesLM",
    "evo2_7b_base",
    "NT_2.5b_1kG",
    "GENA-LM-large",
    "AIDO.DNA-300M",
    "DNABERT2",
    "NTv3-100M-pre",
    "DNABERT",
    "GROVER",
    "Mistral-DNA-422M",
    "GPN-promoter",
    "GENERator-1.2B",
    "GPN-MSA",
    "HyenaDNA_tiny_1k",
    "Caduceus-PS",
)
REFERENCE_CLASS_ORDER = ("ELS", "PLS", "Insulator", "Silencer", "URE")
REFERENCE_CLASS_LABELS = {
    "ELS": "Enhancer",
    "PLS": "Promoter",
    "Insulator": "Insulator",
    "Silencer": "Silencer",
    "URE": "URE",
}
FAMILY_COLORS = {
    "hEVO": "#B35C1E",
    "Evo 2": "#0E7C86",
    "NT": "#65717F",
    "GPN": "#0B2E83",
    "Other": "#9AA3AF",
}
DISPLAY_LABEL_OVERRIDES = {
    "NT_2.5b_1kG": "NTv2-2.5B-1kG",
    "NTv3-100M-pre": "NTv3-100M",
    "GPN-promoter": "GPN-Promoter",
    "HyenaDNA_tiny_1k": "HyenaDNA-1K",
}


def _family(model: str) -> str:
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


def _draw_reference_panel(summary, *, panel_letter: str = "d"):
    """Draw the complete 19-model F1 rank view from the supplied Fig. 2 PDF."""

    models = set(summary["model"].astype(str))
    missing = set(REFERENCE_MODEL_ORDER) - models
    if missing:
        raise ValueError(f"Fig.2d source roster is missing models: {sorted(missing)}")
    focused = summary.loc[summary["model"].astype(str).isin(REFERENCE_MODEL_ORDER)].copy()
    macro = focused.loc[focused["metric"].eq("Macro-F1")].copy()
    if len(macro) != len(REFERENCE_MODEL_ORDER):
        raise ValueError("Fig.2d needs one Macro-F1 row for each of its 19 models")
    macro = macro.sort_values(["mean_f1", "model_display"], ascending=[False, True], kind="stable")
    ordered_models = macro["model"].astype(str).tolist()
    # Preserve the reference roster; order rows by the current source scores.
    model_labels = {
        str(model): DISPLAY_LABEL_OVERRIDES.get(str(model), canonical_model_name(str(model)))
        for model in macro["model"].astype(str)
    }
    class_rows = focused.loc[focused["metric"].isin(REFERENCE_CLASS_ORDER)].copy()
    values = class_rows.pivot(index="model", columns="metric", values="mean_f1")
    ranks = class_rows.pivot(index="model", columns="metric", values="rank")
    if values.reindex(REFERENCE_MODEL_ORDER)[list(REFERENCE_CLASS_ORDER)].isna().any().any():
        raise ValueError("Fig.2d class-specific F1 values must be complete for all reference models")

    width, height = WIDTH_MM, HEIGHT_MM
    drawing = Drawing(width * mm, height * mm)
    drawing.add(Rect(0, 0, width * mm, height * mm, fillColor=HexColor("#FFFFFF"), strokeColor=None))
    style = BACKEND.prev.base.export_base
    add_marker = BACKEND.prev.base.cd.add_marker
    style.add_text(drawing, 0.2, 58.4, panel_letter, 14.0, bold=True, color="#000000")
    style.add_text(drawing, 1.0, 53.6, "F1 rank", 5.4, bold=True, color="#000000")

    plot_left, plot_right = 28.2, 51.9
    plot_bottom, plot_top = 8.8, 50.2
    axis_x = 52.5
    x_positions = np.linspace(plot_left + 0.7, plot_right - 0.7, len(REFERENCE_CLASS_ORDER))
    y_min, y_max = 0.30, 0.95

    def y_f1(value: float) -> float:
        return plot_bottom + (float(value) - y_min) / (y_max - y_min) * (plot_top - plot_bottom)

    style.add_text(
        drawing,
        float(np.mean(x_positions)),
        53.6,
        "hEVO-large rank",
        5.4,
        bold=True,
        anchor="middle",
        color="#000000",
    )
    for tick in (0.30, 0.50, 0.70, 0.95):
        y = y_f1(tick)
        style.add_line(drawing, plot_left, y, axis_x, y, color="#E6E9ED", width=0.42)
        style.add_line(drawing, axis_x, y, axis_x + 0.55, y, color="#000000", width=0.45)
        style.add_text(drawing, axis_x + 1.0, y - 0.55, f"{tick:g}", 6.0, color="#000000")
    style.add_line(drawing, axis_x, plot_bottom, axis_x, plot_top, color="#000000", width=0.5)
    style.add_line(drawing, plot_left, plot_bottom, axis_x, plot_bottom, color="#000000", width=0.5)
    for x, metric in zip(x_positions, REFERENCE_CLASS_ORDER):
        style.add_line(drawing, float(x), plot_bottom, float(x), plot_bottom - 0.55, color="#000000", width=0.42)
        label = String(0, 0, REFERENCE_CLASS_LABELS[metric], fontName=style.FONT, fontSize=6.0,
                       fillColor=HexColor("#000000"), textAnchor="middle")
        group = Group(label)
        group.transform = (
            float(np.cos(np.radians(72.0))),
            float(np.sin(np.radians(72.0))),
            -float(np.sin(np.radians(72.0))),
            float(np.cos(np.radians(72.0))),
            (float(x) + (0.2 if metric == "URE" else 0.0)) * mm,
            (3.93 if metric == "URE" else 3.9) * mm,
        )
        drawing.add(group)

    # Keep model labels at the same order as the plotted Macro-F1 ranks.
    label_top = 48.1
    label_step = 2.01
    for rank_index, model in enumerate(ordered_models, start=1):
        y = label_top - (rank_index - 1) * label_step
        focus = model in {"hEvo-large", "hEvo-base"}
        style.add_text(drawing, 2.7, y - 0.5, str(rank_index), 4.5, bold=focus, anchor="end", color="#000000")
        style.add_fitted_text(
            drawing, 3.6, y - 0.5, str(model_labels[model]), 4.5, 23.8,
            bold=focus, color="#000000", min_size=4.2,
        )

    draw_order = sorted(ordered_models, key=lambda model: model in {"hEvo-large", "hEvo-base"})
    marker_for_family = {"hEVO": "circle", "Evo 2": "triangle", "NT": "square", "GPN": "diamond", "Other": "circle"}
    for model in draw_order:
        family = _family(model)
        color = FAMILY_COLORS[family]
        points = [
            (float(x), y_f1(float(values.loc[model, metric])))
            for x, metric in zip(x_positions, REFERENCE_CLASS_ORDER)
        ]
        focus_large = model == "hEvo-large"
        focus_base = model == "hEvo-base"
        line_width = 1.45 if focus_base else 1.05 if focus_large else 0.52
        for index, ((x1, y1), (x2, y2)) in enumerate(zip(points[:-1], points[1:])):
            line = Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm,
                        strokeColor=HexColor(color), strokeWidth=line_width)
            if focus_large:
                line.strokeDashArray = [2.0, 1.4]
            drawing.add(line)
        for x, y in points:
            marker = "diamond" if focus_base else "triangle" if focus_large else marker_for_family[family]
            fill = color if focus_base else "#FFFFFF"
            add_marker(drawing, x, y, marker, fill=fill, stroke=color)
        if focus_large:
            for x, metric in zip(x_positions, REFERENCE_CLASS_ORDER):
                rank = int(ranks.loc[model, metric])
                score_y = y_f1(float(values.loc[model, metric]))
                offset = 1.45 if score_y < plot_top - 2.0 else -1.8
                style.add_text(drawing, float(x), score_y + offset, f"#{rank}", 4.7,
                               bold=True, anchor="middle", color="#000000")

    _rank_order = tuple(ordered_models)
    drawing._fig2d_display_models = _rank_order  # type: ignore[attr-defined]
    drawing._gpn_star_ure_selection = {  # type: ignore[attr-defined]
        "candidates": ["GPN-Star-M", "GPN-Star-P", "GPN-Star-V"],
        "selected": ["GPN-Star-P"],
        "selection_rule": "the supplied Figure 2 reference roster",
    }
    return drawing


def build_panel(tables_dir: Path | None = None, *, panel_letter: str = "d"):
    """读取 multiclass source-data，按五类 Macro-F1 选择代表模型并绘图。"""

    root = panel_tables_dir("Fig2", tables_dir)
    BACKEND.multi.SOURCE_TABLE = root / "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv"
    _, plot_long, summary, _, _ = BACKEND.multi.prepare_data()
    # Keep source validation in ``prepare_data`` but hide open-series models
    # from the main-figure display layer.
    open_mask = plot_long["model"].astype(str).str.lower().str.startswith("hevo-open-")
    plot_long = plot_long.loc[~open_mask].copy()
    summary_open_mask = summary["model"].astype(str).str.lower().str.startswith("hevo-open-")
    summary = summary.loc[~summary_open_mask].copy()
    register_arial_aliases()
    return _draw_reference_panel(summary, panel_letter=panel_letter)


def export_panel(output_dir: Path | None = None):
    source = panel_tables_dir() / "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv"
    source_drawing = build_panel()
    drawing = pad_drawing(source_drawing, bottom_mm=STANDALONE_BOTTOM_PAD_MM)
    result = export_drawing(drawing, panel_output_dir("Fig2", output_dir), "f_ure_multiclass", input_files(source), width_mm=WIDTH_MM, height_mm=HEIGHT_MM + STANDALONE_BOTTOM_PAD_MM)
    result["qa"]["gpn_star_selection"] = source_drawing._gpn_star_ure_selection  # type: ignore[attr-defined]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig2"))
    args = parser.parse_args()
    source_drawing = build_panel(args.tables_dir)
    drawing = pad_drawing(source_drawing, bottom_mm=STANDALONE_BOTTOM_PAD_MM)
    result = export_drawing(drawing, args.output_dir, "f_ure_multiclass", input_files(args.tables_dir / "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv"), width_mm=WIDTH_MM, height_mm=HEIGHT_MM + STANDALONE_BOTTOM_PAD_MM)
    result["qa"]["gpn_star_selection"] = source_drawing._gpn_star_ure_selection  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
