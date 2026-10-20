"""Fig.1a：正式 cohort 在全量 S1 PCA 坐标系中的二维 PC1–PC2 panel。

本文件把原 Fig.1 A4 入口中 `draw_panel_a()` 的调用独立出来。数据仍是
正式 63,809 人的 IID 名单；PC 坐标从全量 S1 表按 IID 提取，坐标轴范围沿用
550,237 人全集的范围。`build_panel()` 返回相同的 ReportLab Drawing，A4 入口直接复用它。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
SCRIPTS_ROOT = HERE.parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.data_io import require_columns  # noqa: E402
from common.legacy_loader import load_module  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import cache_dir, tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, mm_drawing, register_arial_aliases  # noqa: E402


BACKEND = load_module("Fig1/plot_figure1_a4_integrated.py")
WIDTH_MM, HEIGHT_MM = 84.0, 92.0
S1_SOURCE_NAME = "S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv"
S2_SOURCE_NAME = "S2_principal_component_coordinates_of_the_final_63_809_participant_cohort.csv"
SOURCE_GROUPS = set(BACKEND.COHORT_ORDER)


def _reference_axis_fonts(drawing):
    """Match the reference's regular-weight PC axis labels."""
    def visit(item):
        if getattr(item, "text", None) in ("PC1", "PC2"):
            item.fontName = "Arial"
        for child in getattr(item, "contents", []):
            visit(child)
    visit(drawing)


def _load_final_cohort_on_s1_axes(tables_dir: Path | None = None):
    root = panel_tables_dir("Fig1", tables_dir)
    s1 = pd.read_csv(root / S1_SOURCE_NAME, dtype={"IID": str}, low_memory=False)
    s2 = pd.read_csv(root / S2_SOURCE_NAME, dtype={"IID": str}, low_memory=False)
    require_columns(s1, ["IID", "Group", "PC1", "PC2"], "Fig1a S1")
    require_columns(s2, ["IID", "Group"], "Fig1a S2")
    if len(s1) != 550237 or s1["IID"].nunique() != len(s1):
        raise ValueError("Fig1a: expected 550,237 unique prescreening-cohort S1 IIDs")
    if len(s2) != 63809 or s2["IID"].nunique() != len(s2):
        raise ValueError("Fig1a: expected 63,809 unique final-cohort IIDs")
    if set(s2["Group"]) != SOURCE_GROUPS:
        raise ValueError("Fig1a: unexpected source-cohort label set")

    s1_pcs = s1[["PC1", "PC2"]].apply(pd.to_numeric, errors="coerce")
    if s1_pcs.isna().any().any():
        raise ValueError("Fig1a: S1 PC1/PC2 coordinates contain missing or non-numeric values")
    s1["PC1"] = s1_pcs["PC1"]
    s1["PC2"] = s1_pcs["PC2"]

    matched = s2[["IID", "Group"]].merge(
        s1[["IID", "Group", "PC1", "PC2"]],
        on="IID",
        how="left",
        suffixes=("_final", "_prescreening"),
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if (matched["_merge"] != "both").any():
        missing = int((matched["_merge"] != "both").sum())
        raise ValueError(f"Fig1a: {missing} final-cohort IIDs are absent from S1")
    if (matched["Group_final"] != matched["Group_prescreening"]).any():
        raise ValueError("Fig1a: S1 and S2 cohort labels differ for matched IIDs")

    table = matched[["IID", "Group_final", "PC1", "PC2"]].rename(
        columns={"Group_final": "Group"}
    )
    domains = (
        BACKEND.nice_domain(s1["PC1"].to_numpy(float), pad=0.03),
        BACKEND.nice_domain(s1["PC2"].to_numpy(float), pad=0.03),
    )
    return table, domains, root


def build_panel(tables_dir: Path | None = None):
    """按 S2 IID 取 S1 PC 坐标，并沿用全量 S1 的绘图范围。"""

    table, domains, _ = _load_final_cohort_on_s1_axes(tables_dir)
    register_arial_aliases()
    BACKEND.DENSE_DIR = cache_dir("pca")
    drawing = mm_drawing(WIDTH_MM, HEIGHT_MM)
    BACKEND.draw_panel_a_2d(
        drawing,
        table,
        0.0,
        0.0,
        WIDTH_MM,
        HEIGHT_MM,
        domains=domains,
    )
    _reference_axis_fonts(drawing)
    return drawing


def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    """独立输出 Fig.1a 的 PNG/PDF/SVG 和 QA JSON。"""

    _, _, data_root = _load_final_cohort_on_s1_axes(tables_dir)
    out = panel_output_dir("Fig1", output_dir)
    sources = (data_root / S1_SOURCE_NAME, data_root / S2_SOURCE_NAME)
    return export_drawing(
        build_panel(data_root),
        out,
        "a_pca_2d",
        input_files(*sources),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=panel_tables_dir("Fig1"))
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig1"))
    args = parser.parse_args()
    export_panel(args.output_dir, args.tables_dir)


if __name__ == "__main__":
    main()
