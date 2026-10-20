"""FigS1a: updated 550,237-row S1 source-cohort PCA overview."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

HERE = Path(__file__).resolve()
SCRIPTS_ROOT = HERE.parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.data_io import sha256  # noqa: E402
from common.panel_runtime import input_files, panel_output_dir  # noqa: E402
from common.paths import tables_dir, figure_dir  # noqa: E402
from common.reportlab_common import export_drawing  # noqa: E402
from FigS1 import _pca  # noqa: E402


WIDTH_MM, HEIGHT_MM = 84.0, 96.0
STEM = "a_pca"


def build_panel(tables_dir: Path | None = None):
    frame, _, _ = _pca.load_source(tables_dir)
    return _pca.crop_module(frame, "a")


def export_panel(tables_dir: Path | None = None, output_dir: Path | None = None):
    frame, source, validation = _pca.load_source(tables_dir)
    out = panel_output_dir("FigS1", output_dir)
    result = export_drawing(
        _pca.crop_module(frame, "a"),
        out,
        STEM,
        input_files(source),
        width_mm=WIDTH_MM,
        height_mm=HEIGHT_MM,
    )
    _pca.embed_svg_images(result["svg"])
    qa = result["qa"]
    qa.update(
        {
            "panel": "a",
            "data_status": "DATA_QC_DRAFT",
            "data_validation": validation,
            "plotting_contract": {
                "all_observed_rows": True,
                "downsampling": False,
                "quantile_clipping": False,
                "colors_encode": "source cohort, not genetic ancestry",
            },
        }
    )
    reader = PdfReader(str(result["pdf"]))
    page = reader.pages[0]
    resources = page.get("/Resources", {})
    fonts = resources.get("/Font", {}) if resources else {}
    font_names = [str(ref.get_object().get("/BaseFont", "")) for ref in fonts.values()]
    text = page.extract_text() or ""
    with Image.open(result["png"]) as image:
        pixels = list(image.size)
    expected_pixels = [round(WIDTH_MM / 25.4 * 600), round(HEIGHT_MM / 25.4 * 600)]
    checks = {
        "single_page": len(reader.pages) == 1,
        "expected_page_size": abs(float(page.mediabox.width) * 25.4 / 72 - WIDTH_MM) < 0.05
        and abs(float(page.mediabox.height) * 25.4 / 72 - HEIGHT_MM) < 0.05,
        "arial_compatible_embedded": any("Arial" in name or "LiberationSans" in name for name in font_names),
        "png_600_dpi_dimensions": all(abs(actual - expected) <= 1 for actual, expected in zip(pixels, expected_pixels)),
        "required_labels_present": all(label in text for label in ("PC1", "PC2")),
        "svg_has_editable_text": "<text" in result["svg"].read_text(encoding="utf-8"),
        "nonempty_artifacts": all(path.stat().st_size > 1000 for path in (result["pdf"], result["svg"], result["png"])),
    }
    qa["checks"] = checks
    qa["all_artifact_checks_pass"] = all(checks.values())
    qa["submission_ready"] = False
    qa["font_names"] = font_names
    qa["png_pixels"] = pixels
    qa["raster_contract"] = {
        "allowed": True,
        "purpose": "dense PCA point-density layer for 550,237 observations",
        "not_stitched_composite": True,
    }
    qa["outputs"]["svg"]["sha256"] = sha256(result["svg"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir("FigS1"))
    parser.add_argument("--output-dir", type=Path, default=figure_dir("FigS1"))
    args = parser.parse_args()
    export_panel(args.tables_dir, args.output_dir)


if __name__ == "__main__":
    main()
