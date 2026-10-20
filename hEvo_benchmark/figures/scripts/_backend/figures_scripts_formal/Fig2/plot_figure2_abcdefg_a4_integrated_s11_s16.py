#!/usr/bin/env python3
"""Rebuild the integrated Fig. 2 using the released S15--S20 data for a/b.

Panels c--g are delegated to the previously validated renderers.  The legacy
Fig. 2a/b vector-recovery input is never read by this entrypoint.
"""

from __future__ import annotations

# === Comment index ===
# [Role] Current Fig. 2 integration entry point: a/b use S11–S16, while c–g reuse the validated legacy plotting chain.
# [Calls] updated_ab creates the new a/b panels; legacy supplies data processing and panel Drawings for c–g without reading legacy_vector_recovered.
# [Layout] Place panels on the A4 page using CONTENT_X, AB_Y, CD_Y, and F/E/G coordinates; shift AB_Y_MM up by 4 mm to avoid overlap between a/b and c.
# [Export] export_main writes the main PDF/SVG and uses Poppler to create a 600 dpi PNG; main() also writes QA and a/b source data.
# [Status] Outputs remain DATA_QC_DRAFT; comments must not change the analysis or source data for c–g.

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import plot_fig2ab_s11_s16 as updated_ab  # noqa: E402
import plot_figure2_abcdef_a4_integrated_v1 as legacy  # noqa: E402


OUT_DIR = PROJECT_ROOT / "figures" / "formal" / "Fig2" / "figure2_abcdefg_s11_s16"
STEM = "Figure2_abcdefg_A4_integrated_S11_S16_DATA_QC_DRAFT"

# The 76-mm a/b drawing and the 142-mm c/d drawing otherwise overlap by 4 mm
# at the A4 assembly boundary.  Move only the new a/b block upward into the
# available top margin; the c-g drawing logic and coordinates remain unchanged.
AB_Y_MM = legacy.prev.base.AB_Y_MM + 4.0


def register_hpc_fonts() -> None:
    """Register the user-supplied Arial faces used by legacy renderers."""

    formal_root = HERE.parents[2]
    if str(formal_root) not in sys.path:
        sys.path.insert(0, str(formal_root))
    from figures.scripts.common.font_config import register_reportlab_arial

    register_reportlab_arial()


def render_png(pdf: Path, png: Path) -> None:
    subprocess.run(
        ["pdftoppm", "-f", "1", "-l", "1", "-singlefile", "-png", "-r", "600", str(pdf), str(png.with_suffix(""))],
        check=True,
    )


def export_main(page: Drawing, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{STEM}.pdf"
    svg = output_dir / f"{STEM}.svg"
    png = output_dir / f"{STEM}.png"
    renderPDF.drawToFile(page, str(pdf))
    renderSVG.drawToFile(page, str(svg))
    render_png(pdf, png)
    return {"pdf": pdf, "svg": svg, "png": png}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    register_hpc_fonts()
    combined, selected, metadata = updated_ab.load_data()
    ab_drawing = updated_ab.build_drawing(combined, selected)

    gb_cells, gb_summary, _ = legacy.prev.base.cd.read_and_validate(legacy.prev.base.cd.CONFIGS["gb"])
    nt_cells, nt_summary, _ = legacy.prev.base.cd.read_and_validate(legacy.prev.base.cd.CONFIGS["nt"])
    gb_rows, _ = legacy.prev.base.selected_with_identity(gb_summary)
    nt_rows, _ = legacy.prev.base.selected_with_identity(nt_summary)
    cd_drawing = legacy.prev.base.draw_compact_cd(gb_cells, nt_cells, gb_rows, nt_rows)

    ure_raw, ure_summary, _ = legacy.prev.ure.prepare_data()
    ure_selected_raw, ure_selected, _ = legacy.prev.select_ure_representatives(ure_raw, ure_summary)
    e_drawing = legacy.draw_distribution_e(ure_selected_raw, ure_selected)

    _, plot_long, multi_summary, _, source_summary = legacy.multi.prepare_data()
    selected_f, variants, classes, selection_summary = legacy.select_multiclass_representatives(plot_long, multi_summary)
    f_drawing = legacy.draw_compact_f(selected_f, variants, classes)
    deepsea_raw, deepsea_plot, deepsea_source_summary = legacy.deepsea.prepare_data()
    g_drawing = legacy.draw_compact_g(deepsea_plot)

    page = Drawing(legacy.PAGE_W_MM * mm, legacy.PAGE_H_MM * mm)
    page.add(Rect(0, 0, page.width, page.height, fillColor=HexColor(legacy.WHITE), strokeColor=None))
    page.add(legacy.prev.base.translated_group(ab_drawing, legacy.prev.base.CONTENT_X_MM, AB_Y_MM))
    page.add(legacy.prev.base.translated_group(cd_drawing, legacy.prev.base.CONTENT_X_MM, legacy.prev.base.CD_Y_MM))
    page.add(legacy.prev.base.translated_group(f_drawing, legacy.F_X_MM, legacy.F_Y_MM))
    page.add(legacy.prev.base.translated_group(e_drawing, legacy.E_X_MM, legacy.E_Y_MM))
    page.add(legacy.prev.base.translated_group(g_drawing, legacy.G_X_MM, legacy.G_Y_MM))

    outputs = export_main(page, args.output_dir)
    source_info = updated_ab.write_sources(args.output_dir, combined, selected, metadata)
    qa = {
        "status": "DATA_QC_DRAFT",
        "legacy_input_used": False,
        "updated_panels": ["a", "b"],
        "unchanged_panels": ["c", "d", "e", "f", "g"],
        "datasets": metadata["datasets"],
        "models_in_source": 33,
        "representative_models": 18,
        "selection_rule": "one representative per legacy algorithm group; highest mean within-dataset t-SNE rank percentile, then mean raw t-SNE CH, then raw model ID",
        "raw_ch_axes": "independent per dataset; no cross-dataset raw CH averaging",
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
        "fig2ab_source_data": source_info,
        "lower_panel_source_summary": source_summary,
        "lower_panel_deepsea_source_summary": deepsea_source_summary,
        "lower_panel_selection_summary_rows": len(selection_summary),
        "notes": [
            "S11-S16 source and summary tables were validated before rendering.",
            "Silhouette was not plotted because the released S11-S16 workflow used --skip-silhouette.",
            "Existing lower-panel renderers were reused without changing their analysis logic.",
        ],
    }


if __name__ == "__main__":
    main()
