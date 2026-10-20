"""Fig.2e：URE binary MCC 分布/模型轨迹 panel。

旧整合器内部将该对象命名为 ``e_drawing``，其历史视觉字母为 ``f``；这里
保留原低层绘图内容和位置语义，文件名按当前数据语义明确写成 e_ure_binary。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reportlab.graphics.shapes import Rect
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.legacy_loader import load_module  # noqa: E402
from common.model_names import canonical_model_name  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, mm_drawing, pad_drawing, register_arial_aliases  # noqa: E402

BACKEND = load_module("Fig2/plot_figure2_abcdef_a4_integrated_v1.py")
WIDTH_MM, HEIGHT_MM = 61.0, 63.0
STANDALONE_LEFT_PAD_MM = 8.0
REFERENCE_MODELS = (
    "hEvo-large",
    "GPN-Star-P",
    "Genos-1.2B",
    "evo2_7b_base",
    "NT_2.5b_MS",
    "GENA-LM-large",
    "hEvo-base",
    "SpeciesLM",
    "NTv3-650M-pre",
    "GENERator-3B",
    "AIDO.DNA-300M",
    "DNABERT",
    "DNABERT2",
    "GPN-promoter",
    "GROVER",
    "Mistral-DNA-422M",
    "GPN-MSA",
    "HyenaDNA_tiny_1k",
    "Caduceus-PS",
)

FAMILY_COLORS = {
    "hEVO": "#B35C1E",
    "Evo 2": "#0E7C86",
    "NT": "#65717F",
    "GPN": "#0B2E83",
    "Other": "#9AA3AF",
}
DISPLAY_LABEL_OVERRIDES = {
    "NT_2.5b_MS": "NTv2-2.5B-MS",
    "NTv3-650M-pre": "NTv3-650M",
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


def _draw_reference_panel(raw, summary, *, panel_letter: str = "e"):
    """Draw all 19 source models with five set means and a mean-MCC marker."""

    models = set(summary["model"].astype(str))
    missing = set(REFERENCE_MODELS) - models
    if missing:
        raise ValueError(f"Fig.2e source roster is missing models: {sorted(missing)}")
    summary = summary.loc[summary["model"].astype(str).isin(REFERENCE_MODELS)].copy()
    raw = raw.loc[raw["model"].astype(str).isin(REFERENCE_MODELS)].copy()
    summary = summary.sort_values(["mean_mcc", "model_display"], ascending=[False, True], kind="stable")
    # Preserve the reference roster; order rows by the current source scores.

    drawing = mm_drawing(WIDTH_MM, HEIGHT_MM)
    drawing.add(Rect(0, 0, drawing.width, drawing.height, fillColor=HexColor("#FFFFFF"), strokeColor=None))
    style = BACKEND.prev.base.export_base
    add_marker = BACKEND.prev.base.cd.add_marker
    style.add_text(drawing, -0.8, 56.8, panel_letter, 14.0, bold=True, color="#000000")
    style.add_text(drawing, -0.5, 52.83, "URE versus enhancer", 6.0, bold=True, color="#000000")

    model_x = 1.5
    plot_x0, plot_x1 = 25.5, 50.3
    mean_x = 55.7
    style.add_text(drawing, (plot_x0 + plot_x1) / 2 + 0.1, 52.83, "Five sets", 6.0, bold=True, anchor="middle", color="#000000")
    style.add_text(drawing, mean_x, 52.83, "Mean MCC", 6.0, bold=True, anchor="middle", color="#000000")

    # Compact key for the five source values, their range, and the overall mean.
    legend_y = 57.08
    add_marker(drawing, 14.9, legend_y + 0.4, "circle", fill="#FFFFFF", stroke="#65717F")
    style.add_text(drawing, 16.5, legend_y, "Set", 5.4, color="#000000")
    style.add_line(drawing, 26.4, legend_y + 0.4, 29.8, legend_y + 0.4, color="#9AA3AF", width=0.75)
    style.add_text(drawing, 30.0, legend_y, "Range", 5.4, color="#000000")
    add_marker(drawing, 44.6, legend_y + 0.4, "circle", fill="#65717F", stroke="#65717F")
    style.add_text(drawing, 46.0, legend_y, "Mean", 5.4, color="#000000")

    top_y, row_step = 49.0, 2.15
    set_offsets = (-0.42, -0.21, 0.0, 0.21, 0.42)
    for index, row in enumerate(summary.itertuples(index=False)):
        y = top_y - index * row_step
        model = str(row.model)
        family = _family(model)
        color = FAMILY_COLORS[family]
        focus = model in {"hEvo-large", "hEvo-base"}
        if focus:
            drawing.add(Rect(1.2 * mm, (y - 0.80) * mm, (WIDTH_MM - 2.3) * mm, 1.60 * mm,
                             fillColor=HexColor("#FCF2E7"), strokeColor=None))
        elif index % 2:
            drawing.add(Rect(1.2 * mm, (y - 0.80) * mm, (WIDTH_MM - 2.3) * mm, 1.60 * mm,
                             fillColor=HexColor("#FAFBFC"), strokeColor=None))
        display_model = DISPLAY_LABEL_OVERRIDES.get(model, canonical_model_name(model))
        style.add_text(drawing, model_x, y - 0.50, display_model, 6.0,
                       bold=focus, color="#000000")

        values = raw.loc[raw["model"].astype(str).eq(model)].sort_values("dataset_order")["mcc_mean"].astype(float).to_numpy()
        if len(values) != 5:
            raise ValueError(f"Fig.2e needs five URE control-set values for {model}")
        map_x = lambda value: plot_x0 + (float(value) - 0.25) / 0.75 * (plot_x1 - plot_x0)
        low_x, high_x = map_x(float(values.min())), map_x(float(values.max()))
        style.add_line(drawing, low_x, y, high_x, y, color="#AEB6BF", width=0.62)
        for value, offset in zip(values, set_offsets):
            add_marker(drawing, map_x(float(value)), y + offset, "circle",
                       fill="#FFFFFF", stroke=color)
        mean_value = float(row.mean_mcc)
        add_marker(drawing, map_x(mean_value), y, "circle", fill=color, stroke=color)
        style.add_text(drawing, 60.8, y - 0.50, f"{mean_value:.3f}", 5.7,
                       bold=focus, anchor="end", color="#000000")

    axis_y = 8.8
    style.add_line(drawing, plot_x0, axis_y, plot_x1, axis_y, color="#000000", width=0.5)
    for tick in (0.25, 0.50, 0.75, 1.00):
        x = plot_x0 + (tick - 0.25) / 0.75 * (plot_x1 - plot_x0)
        style.add_line(drawing, x, axis_y, x, axis_y - 0.65, color="#000000", width=0.4)
        label = "1" if tick == 1.0 else f"{tick:g}"
        style.add_text(drawing, x, 5.63, label, 5.5, anchor="middle", color="#000000")
    style.add_text(drawing, (plot_x0 + plot_x1) / 2, 1.88, "MCC", 6.2, anchor="middle", color="#000000")
    drawing._fig2e_display_models = tuple(summary["model"].astype(str))  # type: ignore[attr-defined]
    return drawing


def build_panel(tables_dir: Path | None = None, *, panel_letter: str = "e"):
    """将 URE binary source table 注入兼容链并返回 compact drawing。"""

    root = panel_tables_dir("Fig2", tables_dir)
    BACKEND.prev.ure.SOURCE_TABLE = root / "S19_binary_classification_across_five_ure_enhancer_control_sets.csv"
    raw, summary, _ = BACKEND.prev.ure.prepare_data()
    # Open-series variants are absent from the supplied reference roster.
    raw = raw.loc[
        ~raw["model"].astype(str).str.lower().str.startswith("hevo-open-")
    ].copy()
    summary = summary.loc[
        ~summary["model"].astype(str).str.lower().str.startswith("hevo-open-")
    ].copy()
    missing = set(REFERENCE_MODELS) - set(summary["model"].astype(str))
    if missing:
        raise ValueError(f"Fig.2e reference models missing from S19: {sorted(missing)}")
    register_arial_aliases()
    return _draw_reference_panel(raw, summary, panel_letter=panel_letter)


def export_panel(output_dir: Path | None = None):
    source = panel_tables_dir() / "S19_binary_classification_across_five_ure_enhancer_control_sets.csv"
    drawing = pad_drawing(build_panel(), left_mm=STANDALONE_LEFT_PAD_MM)
    return export_drawing(drawing, panel_output_dir("Fig2", output_dir), "e_ure_binary", input_files(source), width_mm=WIDTH_MM + STANDALONE_LEFT_PAD_MM, height_mm=HEIGHT_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig2"))
    args = parser.parse_args()
    drawing = pad_drawing(build_panel(args.tables_dir), left_mm=STANDALONE_LEFT_PAD_MM)
    export_drawing(drawing, args.output_dir, "e_ure_binary", input_files(args.tables_dir / "S19_binary_classification_across_five_ure_enhancer_control_sets.csv"), width_mm=WIDTH_MM + STANDALONE_LEFT_PAD_MM, height_mm=HEIGHT_MM)


if __name__ == "__main__":
    main()
