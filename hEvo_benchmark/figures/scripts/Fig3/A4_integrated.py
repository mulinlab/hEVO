"""Fig.3 a–k 的 A4 矢量整合入口。

panel 脚本负责数据读取和局部 Drawing；本文件只按 v2 的绝对毫米坐标
组合它们，并补回整页共用的 benchmark/model/temporal legend。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reportlab.graphics.shapes import Group, Rect
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.panel_runtime import input_files, panel_tables_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, mm_drawing, register_arial_aliases  # noqa: E402
from Fig3 import a_phenotypic_af, b_fitness_daf, c_clinvar, d_causal_all, e_causal_rare, f_ukb_rare, g_ddd, h_mpra, i_clinvar_upgrade, j_clinvar_downgrade, k_vus_resolution  # noqa: E402
from Fig3._backend import BACKEND  # noqa: E402

PAGE_W_MM, PAGE_H_MM = 210.0, 297.0
PANEL_W_MM, BENCH_H_MM, TEMP_H_MM = 64.6666666667, 55.5, 55.0
XS = [5.0, 72.6666666667, 140.3333333334]
PNG_DPI = 300
PNG_TARGET_PIXELS = (2480, 3508)
FIG3_SOURCE_NAMES = (
    "S22_schematic_coordinates_for_variant_effect_and_allele_frequency_benchmarks.csv",
    "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv",
    "S24_zero_shot_classification_of_coding_clinvar_variants.csv",
    "S25_zero_shot_classification_of_noncoding_clinvar_variants.csv",
    "S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv",
    "S27_zero_shot_classification_of_rare_candidate_causal_gwas_variants.csv",
    "S28_zero_shot_classification_of_uk_biobank_rare_variants.csv",
    "S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv",
    "S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv",
    "S32_retrospective_enrichment_of_clinvar_reclassification_events_by_cutoff_year.csv",
)


def translated(panel, x: float, y: float) -> Group:
    """把已裁剪的 panel Drawing 平移到 A4 左下坐标。"""

    group = Group(*panel.contents)
    group.translate(x * mm, y * mm)
    return group


def build_panel(tables_dir: Path | None = None):
    """调用 a–k 的 build_panel()，保持 v2 主图的三行布局。"""

    root = panel_tables_dir("Fig3", tables_dir)
    register_arial_aliases()
    page = mm_drawing(PAGE_W_MM, PAGE_H_MM)
    page.add(Rect(0, 0, page.width, page.height, fillColor=HexColor("#FFFFFF"), strokeColor=None))
    page.add(translated(a_phenotypic_af.build_panel(root), 5.0, 225.5))
    page.add(translated(b_fitness_daf.build_panel(root), 107.0, 225.5))
    BACKEND.core.add_text(
        page,
        PAGE_W_MM / 2,
        216.0,
        "Frequency categories are schematic and do not imply matched controls.",
        5.5,
        anchor="middle",
    )
    BACKEND.draw_dataset_legend(page)
    BACKEND.draw_model_legend(page, 194.2, PAGE_W_MM, include_note=False)
    # Move DDD from g to d, then shift the former d–f panels into e–g.
    row_middle = (
        c_clinvar.build_panel(root),
        g_ddd.build_panel(root, panel_letter="d"),
        d_causal_all.build_panel(root, panel_letter="e"),
    )
    for panel, x in zip(row_middle, XS):
        page.add(translated(panel, x, 130.0))
    row_lower = (
        e_causal_rare.build_panel(root, panel_letter="f"),
        f_ukb_rare.build_panel(root, panel_letter="g"),
        h_mpra.build_panel(root),
    )
    for panel, x in zip(row_lower, XS):
        page.add(translated(panel, x, 66.5))
    temporal_panels = (
        j_clinvar_downgrade.build_panel(root),
        i_clinvar_upgrade.build_panel(root),
        k_vus_resolution.build_panel(root),
    )
    for panel, x in zip(temporal_panels, XS):
        page.add(translated(panel, x, 4.5))
    # The temporal legend is shared across panels i–k in the current v2 layout.
    BACKEND.core.draw_temporal_legend(page, 60.2)
    return page


def export_panel(output_dir: Path | None = None):
    root = panel_tables_dir("Fig3")
    out = output_dir or figure_dir("Fig3")
    return export_drawing(
        build_panel(root),
        out,
        "A4_integrated",
        input_files(*(root / name for name in FIG3_SOURCE_NAMES)),
        width_mm=PAGE_W_MM,
        height_mm=PAGE_H_MM,
        png_dpi=PNG_DPI,
        png_target_pixels=PNG_TARGET_PIXELS,
        normalize_blank_right_edge=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig3"))
    args = parser.parse_args()
    root = args.tables_dir if args.tables_dir is not None else tables_dir()
    export_drawing(
        build_panel(root),
        args.output_dir,
        "A4_integrated",
        input_files(*(root / name for name in FIG3_SOURCE_NAMES)),
        width_mm=PAGE_W_MM,
        height_mm=PAGE_H_MM,
        png_dpi=PNG_DPI,
        png_target_pixels=PNG_TARGET_PIXELS,
        normalize_blank_right_edge=True,
    )


if __name__ == "__main__":
    main()
