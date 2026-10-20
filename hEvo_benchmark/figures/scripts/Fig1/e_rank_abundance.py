"""Fig.1e：单倍型 rank-abundance 曲线。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.data_io import require_columns  # noqa: E402
from common.legacy_loader import load_module  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing, mm_drawing, pad_drawing, register_arial_aliases  # noqa: E402

BACKEND = load_module("Fig1/plot_figure1_a4_integrated.py")
WIDTH_MM, HEIGHT_MM = 46.0, 52.0
STANDALONE_RIGHT_PAD_MM = 4.0


def build_panel(tables_dir: Path | None = None):
    """读取已解析的 rank/count 表，不在绘图层重新估计 rank。"""

    root = panel_tables_dir("Fig1", tables_dir)
    source = root / "S8_rank_abundance_profile_of_distinct_haplotype_sequences.csv"
    table = pd.read_csv(source)
    require_columns(table, ["rank", "count"], "Fig1e")
    if not table["rank"].is_monotonic_increasing or not table["count"].is_monotonic_decreasing:
        raise ValueError("Fig1e: rank-abundance ordering is invalid")
    dedup = pd.read_csv(root / "S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv")
    require_columns(
        dedup,
        ["chrom", "n_free_windows", "unique_windows", "sample_count_in_vcf"],
        "Fig1e dedup contract",
    )
    if set(dedup["sample_count_in_vcf"].astype(int)) != {63809}:
        raise ValueError("Fig1e: S7 must be the 63,809-sample final-corpus chromosome run")
    if int(table["rank"].max()) > int(dedup["unique_windows"].sum()):
        raise ValueError("Fig1e: global distinct count exceeds per-chromosome distinct sum")
    register_arial_aliases()
    drawing = mm_drawing(WIDTH_MM, HEIGHT_MM)
    BACKEND.draw_panel_e(drawing, table, 0.0, 0.0, WIDTH_MM, HEIGHT_MM)
    return drawing


def export_panel(output_dir: Path | None = None):
    root = panel_tables_dir("Fig1")
    source = root / "S8_rank_abundance_profile_of_distinct_haplotype_sequences.csv"
    dedup_source = root / "S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv"
    drawing = pad_drawing(build_panel(), right_mm=STANDALONE_RIGHT_PAD_MM)
    return export_drawing(drawing, panel_output_dir("Fig1", output_dir), "e_rank_abundance", input_files(source, dedup_source), width_mm=WIDTH_MM + STANDALONE_RIGHT_PAD_MM, height_mm=HEIGHT_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig1"))
    args = parser.parse_args()
    drawing = pad_drawing(build_panel(args.tables_dir), right_mm=STANDALONE_RIGHT_PAD_MM)
    export_drawing(drawing, args.output_dir, "e_rank_abundance", input_files(args.tables_dir / "S8_rank_abundance_profile_of_distinct_haplotype_sequences.csv", args.tables_dir / "S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv"), width_mm=WIDTH_MM + STANDALONE_RIGHT_PAD_MM, height_mm=HEIGHT_MM)


if __name__ == "__main__":
    main()
