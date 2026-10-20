"""Assemble Fig.1 on A4 with a vector PDF architecture panel and vertical rank radars."""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pypdf import PdfReader, PdfWriter, Transformation
from pathlib import Path
from reportlab.graphics.shapes import Group, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from common.data_io import sha256
from common.panel_runtime import input_files, panel_tables_dir
from common.paths import figure_dir
from common.public_paths import publicize_paths
from common.reportlab_common import export_drawing, mm_drawing, register_arial_aliases
from Fig1 import (a_pca_3d, b_haplotype_coverage, c_haplotype_diversity,
                  d_ancestry_composition, e_rank_abundance, f_ancestry_sharing,
                  h_family_radar)

PAGE_W_MM, PAGE_H_MM = 210.0, 297.0
# Lower-left coordinates. The top 8 mm of panel g hold its label.
G_X_MM, G_Y_MM, G_W_MM, G_H_MM = 8.0, 8.0, 118.0, 128.0
G_ARTWORK_W_MM, G_ARTWORK_H_MM = 118.0, 120.0
H_X_MM, H_Y_MM = 130.0, 8.0
FIG1_SOURCE_NAMES = (
    "S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv",
    "S2_principal_component_coordinates_of_the_final_63_809_participant_cohort.csv",
    "S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv",
    "S5_coverage_of_uk_biobank_reference_haplotypes_by_the_selected_subset.csv",
    "S6_local_haplotype_diversity_after_sequential_cohort_integration.csv",
    "S7_chromosome_level_haplotype_window_counts_and_sequence_deduplication.csv",
    "S8_rank_abundance_profile_of_distinct_haplotype_sequences.csv",
    "S9_haplotype_sequence_sharing_across_five_ancestry_groups.csv",
    "S10_best_model_family_ranks_across_functional_variant_and_evolutionary_tasks.csv",
    "S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv",
    "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv",
    "S19_binary_classification_across_five_ure_enhancer_control_sets.csv",
    "S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv",
    "S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv",
    "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv",
    "S24_zero_shot_classification_of_coding_clinvar_variants.csv",
    "S25_zero_shot_classification_of_noncoding_clinvar_variants.csv",
    "S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv",
    "S28_zero_shot_classification_of_uk_biobank_rare_variants.csv",
    "S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv",
    "S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv",
    "S31_within_site_top_rank_discrimination_of_ddd_de_novo_alleles.csv",
    "S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv",
    "S34_supervised_classification_of_1000_genomes_ancestry_groups.csv",
    "S35_few_shot_regression_of_ancestry_proportions.csv",
    "S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv",
    "S37_classification_of_human_accelerated_regions_across_negative_controls.csv",
    "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv",
)


def translated(panel, x, y):
    group = Group(*panel.contents)
    group.translate(x * mm, y * mm)
    return group


def relabel_panel(panel, old, new):
    """Change the embedded panel letter without altering panel artwork."""
    def visit(item):
        if isinstance(item, String) and item.text == old:
            item.text = new
            return True
        for child in getattr(item, "contents", []):
            if visit(child):
                return True
        return False

    if not visit(panel):
        raise ValueError(f"Panel letter {old!r} was not found")
    return panel


def revise_panel_text(panel, replacements):
    """Apply the reader-facing labels from the supplied Fig. 1 reference."""
    found = set()

    def visit(item):
        if isinstance(item, String) and item.text in replacements:
            found.add(item.text)
            item.text = replacements[item.text]
        for child in getattr(item, "contents", []):
            visit(child)

    visit(panel)
    missing = set(replacements) - found
    if missing:
        raise ValueError(f"Fig.1 reference labels were not found: {sorted(missing)}")
    return panel


def build_panel(tables_dir=None):
    root = panel_tables_dir("Fig1", tables_dir)
    register_arial_aliases()
    page = mm_drawing(PAGE_W_MM, PAGE_H_MM)
    page.add(Rect(0, 0, page.width, page.height,
                  fillColor=HexColor("#FFFFFF"), strokeColor=None))
    # Match the reader-facing order in the supplied reference: coverage,
    # PCA, ancestry composition, haplotype diversity, rank abundance, sharing.
    # The two columns share outer edges and the three horizontal bands meet
    # on common baselines so the reordered panels remain aligned on A4.
    coverage = b_haplotype_coverage.build_panel(root)
    pca = revise_panel_text(
        a_pca_3d.build_panel(root), {"1000 Genomes (3,202)": "1000G (3,202)"}
    )
    ancestry = revise_panel_text(
        d_ancestry_composition.build_panel(root),
        {"EAS2K": "Internal Chinese", "AFR1K": "CAAPA"},
    )
    diversity = revise_panel_text(
        c_haplotype_diversity.build_panel(root),
        {
            "+ 1000 Genomes": "+ 1000G",
            "+ East Asian cohort": "+ Internal Chinese",
            "+ Singapore 10K": "+ SG10K",
            "+ African cohort": "+ CAAPA",
            "Unique haplotypes per 100 kb": "Cohort-balanced haplotypes per 100 kb",
        },
    )
    sharing = revise_panel_text(
        f_ancestry_sharing.build_panel(root),
        {"Top 12 = 89.7% of reported total": "Top 12 intersections (89.7%)"},
    )
    radar = revise_panel_text(
        h_family_radar.build_panel(root),
        {
            "ClinVar (3)": "ClinVar (1)",
            "De novo (2)": "De novo (1)",
            "GWAS causal (1)": "GWAS causal (2)",
        },
    )
    placements = (
        (relabel_panel(coverage, "b", "a"), 8.0, 247.0),
        (relabel_panel(pca, "a", "b"), 110.0, 197.0),
        (relabel_panel(ancestry, "d", "c"), 8.0, 197.0),
        (relabel_panel(diversity, "c", "d"), 8.0, 144.0),
        (e_rank_abundance.build_panel(root), 104.0, 144.0),
        (sharing, 156.0, 144.0),
        (radar, H_X_MM, H_Y_MM),
    )
    for panel, x, y in placements:
        page.add(translated(panel, x, y))
    page.add(String((G_X_MM + 0.3) * mm, (G_Y_MM + 123.0) * mm,
                    "g", fontName="Arial-Bold", fontSize=14,
                    fillColor=HexColor("#000000")))
    return page


def export_panel(output_dir=None, tables_dir=None, architecture_pdf=None):
    root = panel_tables_dir("Fig1", tables_dir)
    out = Path(output_dir) if output_dir is not None else figure_dir("Fig1")
    architecture = Path(architecture_pdf) if architecture_pdf else figure_dir("Fig1") / "Fig1g_120mm.pdf"
    source = PdfReader(str(architecture))
    if len(source.pages) != 1:
        raise ValueError("Architecture PDF must contain exactly one page")
    result = export_drawing(build_panel(root), out, "A4_integrated",
                            input_files(*(root / name for name in FIG1_SOURCE_NAMES), architecture),
                            width_mm=PAGE_W_MM, height_mm=PAGE_H_MM)
    # Merge the supplied architecture as vector PDF content, preserving its aspect ratio.
    artwork = source.pages[0]
    box = artwork.cropbox
    scale = min(G_ARTWORK_W_MM * mm / float(box.width),
                G_ARTWORK_H_MM * mm / float(box.height))
    placed_w, placed_h = float(box.width) * scale, float(box.height) * scale
    placed_x = G_X_MM * mm + (G_ARTWORK_W_MM * mm - placed_w) / 2
    placed_y = G_Y_MM * mm + (G_ARTWORK_H_MM * mm - placed_h) / 2
    pdf = Path(result["pdf"])
    page = PdfReader(str(pdf)).pages[0]
    transform = (Transformation().translate(-float(box.left), -float(box.bottom))
                 .scale(scale).translate(placed_x, placed_y))
    page.merge_transformed_page(artwork, transform, over=True)
    writer = PdfWriter()
    writer.add_page(page)
    with pdf.open("wb") as handle:
        writer.write(handle)
    subprocess.run(["pdftoppm", "-singlefile", "-png", "-r", "600",
                    str(pdf), str(Path(result["png"]).with_suffix(""))], check=True)
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(result["svg"])], check=True)
    qa = result["qa"]
    for kind in ("pdf", "png", "svg"):
        qa["outputs"][kind]["sha256"] = sha256(Path(result[kind]))
    qa["outputs"]["svg"]["bytes"] = Path(result["svg"]).stat().st_size
    qa["backend"] = "ReportLab + pypdf vector PDF composition"
    qa["fig1g"] = {
        "status": "EMBEDDED_VECTOR_PDF",
        "source_pdf": architecture.name,
        "source_sha256": sha256(architecture),
        "placed_artwork_xywh_mm": [placed_x / mm, placed_y / mm, placed_w / mm, placed_h / mm],
        "aspect_ratio_preserved": True,
        "panel_slot_xywh_mm": [G_X_MM, G_Y_MM, G_W_MM, G_H_MM],
        "artwork_slot_xywh_mm": [G_X_MM, G_Y_MM, G_ARTWORK_W_MM, G_ARTWORK_H_MM],
        "coordinate_origin": "page bottom left",
        "panel_label_preserved": "g",
    }
    qa["fig1h"] = {
        "panel_slot_xywh_mm": [H_X_MM, H_Y_MM, 64.0, 128.0],
        "radial_scale": [0.0, 1.0],
        "value_column": "display_radius",
        "missing_values": "open edges; no imputation; note shown only when unavailable rows exist",
        "reference_style": {"font": "Arial", "label_pt": 6.0,
                            "title_pt": 6.6, "panel_letter_pt": 14.0,
                            "family_line_pt": [1.1, 0.85], "spoke_pt": 0.3,
                            "outer_polygon_pt": 0.4},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=figure_dir("Fig1"))
    parser.add_argument("--architecture-pdf", type=Path, help="Single-page architecture PDF for panel g")
    args = parser.parse_args()
    export_panel(args.output_dir, args.tables_dir, args.architecture_pdf)


if __name__ == "__main__":
    main()
