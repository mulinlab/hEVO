"""Fig.2c：三种基因组元件任务中原代表模型的 CH 对照。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.legacy_loader import load_module  # noqa: E402
from common.panel_runtime import input_files, panel_tables_dir, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing  # noqa: E402

BACKEND = load_module("Fig2/plot_fig2ab_s11_s16.py")
WIDTH_MM, HEIGHT_MM = 190.0, 53.0
FIG2C_REFERENCE_MODEL_ORDER = (
    "hEvo-large",
    "hEvo-base",
    "GPN-Star-V",
    "GROVER",
    "GENA-LM-large",
    "NT_2.5b_1kG",
    "GPN-MSA",
    "Mistral-DNA-422M",
    "Caduceus-PS",
    "DNABERT2",
    "HyenaDNA_tiny_1k",
    "GPN-promoter",
    "DNABERT",
    "SpeciesLM",
    "evo2_7b_base",
    "NTv3-650M-pre",
    "AIDO.DNA-300M",
    "Genos-1.2B",
    "GENERator-1.2B",
)
TABLE_FILES = (
    "S11_seed_level_representation_separation_for_encode_candidate_cis_regulatory_elements.csv",
    "S12_summary_of_representation_separation_for_encode_candidate_cis_regulatory_elements.csv",
    "S13_seed_level_representation_separation_for_gencode_gene_structure_classes.csv",
    "S14_summary_of_representation_separation_for_gencode_gene_structure_classes.csv",
    "S15_seed_level_representation_separation_for_curated_refseq_gene_structure_classes.csv",
    "S16_summary_of_representation_separation_for_curated_refseq_gene_structure_classes.csv",
)


def _configure(root: Path) -> None:
    """Point the table reader at the active benchmark tables directory."""
    BACKEND.TABLE_DIR = root


def build_panel(
    tables_dir: Path | None = None,
    *,
    panel_letter: str = "c",
    standalone_legend: bool = True,
    height_mm: float = HEIGHT_MM,
):
    """Read S11–S16 and draw the 19-model roster in the supplied Fig. 2 PDF."""

    root = panel_tables_dir("Fig2", tables_dir)
    _configure(root)
    combined, _, _ = BACKEND.load_data()
    expected_models = set(FIG2C_REFERENCE_MODEL_ORDER)
    available_models = set(combined["model_name"].astype(str))
    if not expected_models.issubset(available_models):
        raise ValueError(
            "Fig.2c reference roster is missing source models: "
            f"{sorted(expected_models - available_models)}"
        )
    selected = combined.loc[combined["model_name"].astype(str).isin(expected_models)].copy()
    selected_models = list(FIG2C_REFERENCE_MODEL_ORDER)
    hevo_models = [name for name in selected_models if name.startswith("hEvo-")]
    if set(hevo_models) != {"hEvo-large", "hEvo-base"}:
        raise ValueError(f"Fig.2c hEvo representatives must be large/base only, got {hevo_models}")
    from common.reportlab_common import mm_drawing, register_arial_aliases

    register_arial_aliases()
    drawing = mm_drawing(WIDTH_MM, height_mm)
    BACKEND.draw_panel_a(
        drawing,
        selected,
        0.0,
        0.0,
        WIDTH_MM,
        height_mm,
        panel_letter=panel_letter,
        standalone_legend=standalone_legend,
        model_label_size=6.0,
        model_order=FIG2C_REFERENCE_MODEL_ORDER,
    )
    drawing._refactor_combined = combined  # type: ignore[attr-defined]
    drawing._fig2c_display_models = selected_models  # type: ignore[attr-defined]
    drawing._fig2c_hevo_models = hevo_models  # type: ignore[attr-defined]
    drawing._fig2c_selection_source = "supplied Figure 2 reference PDF"  # type: ignore[attr-defined]
    return drawing


def export_panel(output_dir: Path | None = None):
    root = panel_tables_dir()
    files = [root / name for name in TABLE_FILES]
    drawing = build_panel()
    return export_drawing(drawing, panel_output_dir("Fig2", output_dir), "a_model_ch", input_files(*files), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir())
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig2"))
    args = parser.parse_args()
    root = panel_tables_dir("Fig2", args.tables_dir)
    _configure(root)
    drawing = build_panel(root)
    files = [root / name for name in TABLE_FILES]
    export_drawing(drawing, args.output_dir, "a_model_ch", input_files(*files), width_mm=WIDTH_MM, height_mm=HEIGHT_MM)


if __name__ == "__main__":
    main()
