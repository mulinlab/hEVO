#!/usr/bin/env python3
"""Rebuild S1a-e from local tables and integrate the five reference panels."""
from __future__ import annotations

import hashlib
import json
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.units import mm

BENCHMARK = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = BENCHMARK / "figures" / "scripts"
OUT = BENCHMARK / "figures" / "FigS1"
TABLES = BENCHMARK / "tables"
BASE_PDF = OUT / "FigureS1_abcd_A4_compact_v2.pdf"
E_PDF = OUT / "e_cohort_haplotype_workflow.pdf"
PDF = OUT / "A4_integrated.pdf"
PNG = OUT / "A4_integrated.png"
SVG = OUT / "A4_integrated.svg"

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render_png(pdf_path: Path, output_path: Path, dpi: int) -> None:
    subprocess.run(
        ["pdftoppm", "-singlefile", "-png", "-r", str(dpi), str(pdf_path), str(output_path.with_suffix(""))],
        check=True,
        capture_output=True,
        text=True,
    )


def build() -> dict[str, object]:
    from FigS1 import _build_abcd_v8 as base

    base.main()

    from FigS1 import _layout_abcd_v2 as layout

    layout.main()

    from FigS1 import _e_workflow_open_v3 as panel_e

    panel_e.export_panel(tables_dir=TABLES, output_dir=OUT)

    base_reader = PdfReader(BASE_PDF)
    e_reader = PdfReader(E_PDF)
    if len(base_reader.pages) != 1 or len(e_reader.pages) != 1:
        raise ValueError("S1 base and panel e must each be one page")
    e_page = e_reader.pages[0]
    e_width_mm = float(e_page.mediabox.width) / mm
    e_height_mm = float(e_page.mediabox.height) / mm
    if abs(e_width_mm - 188.0) > 0.01 or abs(e_height_mm - 48.0) > 0.01:
        raise ValueError(f"S1e source size changed: {e_width_mm:.3f} x {e_height_mm:.3f} mm")

    tx_mm, ty_mm = 8.1, 66.7
    writer = PdfWriter()
    writer.add_page(base_reader.pages[0])
    writer.pages[0].merge_transformed_page(
        e_page, Transformation().translate(tx=tx_mm * mm, ty=ty_mm * mm)
    )
    writer.add_metadata({
        "/Title": "Figure S1 | Cohort structure, phasing quality and haplotype corpus construction",
        "/Author": "hEVO",
        "/Subject": "A4 integration of the five reference panels, rebuilt from local tables",
    })
    with PDF.open("wb") as handle:
        writer.write(handle)

    render_png(PDF, PNG, 600)
    pdftocairo = shutil.which("pdftocairo")
    if pdftocairo is None:
        raise FileNotFoundError("pdftocairo is required to export the integrated SVG")
    subprocess.run(
        [pdftocairo, "-svg", str(PDF), str(SVG)],
        check=True,
        capture_output=True,
        text=True,
    )

    reader = PdfReader(PDF)
    page = reader.pages[0]
    text = page.extract_text() or ""
    labels = {label: text.split().count(label) for label in "abcde"}
    expected_px = [round(210 / 25.4 * 600), round(297 / 25.4 * 600)]
    with Image.open(PNG) as image:
        png_size = list(image.size)
    if len(reader.pages) != 1:
        raise ValueError("Integrated S1 must be a single A4 page")
    if abs(float(page.mediabox.width) / mm - 210) > 0.01 or abs(float(page.mediabox.height) / mm - 297) > 0.01:
        raise ValueError("Integrated S1 page is not A4 portrait")
    if labels != {label: 1 for label in "abcde"}:
        raise ValueError(f"Expected one instance of each panel label a-e, got {labels}")
    if not all(value in text for value in ("550,237", "63,809", "4.784B", "Equal-weight sampling")):
        raise ValueError("Integrated S1 is missing a checked reference label or value")
    if any(a != b for a, b in zip(png_size, expected_px)):
        raise ValueError(f"Unexpected 600-dpi PNG dimensions: {png_size}; expected {expected_px}")

    # Ensure appending e leaves the compact a-d panels pixel-identical.
    qa_dir = Path(tempfile.mkdtemp(prefix="hevo_figs1_qa_"))
    qa_base = qa_dir / "_s1_abcd_180dpi_qa"
    qa_full = qa_dir / "_s1_full_180dpi_qa"
    render_png(BASE_PDF, qa_base, 180)
    render_png(PDF, qa_full, 180)
    with Image.open(qa_base.with_suffix(".png")) as base_img, Image.open(qa_full.with_suffix(".png")) as full_img:
        base_rgb, full_rgb = base_img.convert("RGB"), full_img.convert("RGB")
        cut = round((297 - 116) * 180 / 25.4)
        unchanged = ImageChops.difference(base_rgb.crop((0, 0, base_rgb.width, cut)), full_rgb.crop((0, 0, full_rgb.width, cut))).getbbox() is None
    reference_path = Path("/tmp/supp_ref/S1.png")
    pixel_comparison = None
    if reference_path.exists():
        compare_base = qa_dir / "_s1_reference_size_qa"
        with Image.open(reference_path) as ref_img:
            ref_rgb = ref_img.convert("RGB")
            ref_width, ref_height = ref_rgb.size
        subprocess.run(
            ["pdftoppm", "-singlefile", "-png", "-scale-to", str(ref_height), str(PDF), str(compare_base)],
            check=True,
            capture_output=True,
            text=True,
        )
        with Image.open(reference_path) as ref_img, Image.open(compare_base.with_suffix(".png")) as out_img:
            ref_array = np.asarray(ref_img.convert("RGB"), dtype=np.int16)
            out_array = np.asarray(out_img.convert("RGB"), dtype=np.int16)
            if out_array.shape != ref_array.shape:
                raise ValueError(f"S1 reference raster size differs: {out_array.shape} vs {ref_array.shape}")
            delta = np.abs(out_array - ref_array)
            pixel_comparison = {
                "reference_raster": str(reference_path),
                "size_px": [ref_width, ref_height],
                "mean_absolute_channel_error": round(float(delta.mean()), 6),
                "pixels_with_any_channel_error_over_20": int((delta.max(axis=2) > 20).sum()),
                "exact_pixel_match": bool(np.array_equal(out_array, ref_array)),
            }
        compare_base.with_suffix(".png").unlink(missing_ok=True)
    qa_base.with_suffix(".png").unlink(missing_ok=True)
    qa_full.with_suffix(".png").unlink(missing_ok=True)
    if not unchanged:
        raise ValueError("Adding S1e changed the rendered S1a-d region")

    report = {
        "page_mm": [210, 297],
        "page_count": 1,
        "panel_labels": labels,
        "expected_panels": ["a", "b", "c", "d", "e"],
        "panel_e_native_size_mm": [188, 48],
        "panel_e_translation_mm": [tx_mm, ty_mm],
        "panel_e_scale": 1,
        "a_to_d_pixels_unchanged_after_appending_e": unchanged,
        "png_size_px": png_size,
        "expected_png_size_px_at_600dpi": expected_px,
        "reference_pdf_pixel_comparison": pixel_comparison,
        "source_files": {
            "a_d_base_pdf_sha256": sha256(BASE_PDF),
            "e_panel_pdf_sha256": sha256(E_PDF),
            "integrated_pdf_sha256": sha256(PDF),
            "integrated_png_sha256": sha256(PNG),
            "integrated_svg_sha256": sha256(SVG),
        },
    }
    shutil.rmtree(qa_dir)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    build()
