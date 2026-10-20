"""ReportLab panel 导出和 HPC 字体兼容层。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Group, Rect
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm

from .data_io import sha256
from .font_config import font_metadata, register_reportlab_arial
from .public_paths import publicize_paths, recorded_path


_REGISTERED = False


def register_arial_aliases() -> None:
    """Register the user-supplied Arial faces used by all ReportLab figures."""

    global _REGISTERED
    if _REGISTERED:
        return
    register_reportlab_arial()
    _REGISTERED = True


def export_drawing(
    drawing: Drawing,
    output_dir: Path,
    stem: str,
    input_files: list[Path],
    width_mm: float | None = None,
    height_mm: float | None = None,
    png_dpi: int = 600,
    png_target_pixels: tuple[int, int] | None = None,
    normalize_blank_right_edge: bool = False,
) -> dict[str, object]:
    """导出单 panel 的 PDF/SVG/PNG，并写入可复核 QA JSON。"""

    register_arial_aliases()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    renderPDF.drawToFile(drawing, str(pdf))
    renderSVG.drawToFile(drawing, str(svg))
    executable = shutil.which("pdftoppm") or "pdftoppm"
    subprocess.run(
        [executable, "-f", "1", "-l", "1", "-singlefile", "-png", "-r", str(png_dpi), str(pdf), str(png.with_suffix(""))],
        check=True,
    )
    normalized_blank_right_edge = False
    if png_target_pixels is not None:
        with Image.open(png) as image:
            image.load()
            if image.size != tuple(png_target_pixels):
                expected_width, expected_height = png_target_pixels
                right_edge_is_blank = (
                    image.size == (expected_width + 1, expected_height)
                    and normalize_blank_right_edge
                    and all(
                        image.getpixel((image.width - 1, y))[:3] == (255, 255, 255)
                        for y in range(image.height)
                    )
                )
                if not right_edge_is_blank:
                    raise ValueError(
                        f"{stem}: rendered PNG size {image.size} does not match "
                        f"target {png_target_pixels} without a non-content boundary normalization"
                    )
                image.crop((0, 0, expected_width, expected_height)).save(
                    png,
                    dpi=(png_dpi, png_dpi),
                )
                normalized_blank_right_edge = True
    with Image.open(png) as image:
        pixels = list(image.size)
        png_mode = image.mode
        png_dpi_info = image.info.get("dpi")
        image.verify()
    version_result = subprocess.run([executable, "-v"], capture_output=True, text=True, check=False)
    version_text = (version_result.stdout + version_result.stderr).splitlines()
    qa = {
        "status": "PASS",
        "status_scope": "render_and_artifact_integrity_only",
        "backend": "ReportLab",
        "inputs": [{"path": recorded_path(path), "sha256": sha256(path)} for path in input_files],
        "outputs": {
            "pdf": {"path": recorded_path(pdf), "sha256": sha256(pdf), "pages": len(PdfReader(str(pdf)).pages)},
            "svg": {"path": recorded_path(svg), "sha256": sha256(svg), "bytes": svg.stat().st_size},
            "png": {"path": recorded_path(png), "sha256": sha256(png), "pixels": pixels},
        },
        "expected_size_mm": [width_mm, height_mm],
        "rendering": {
            "renderer": Path(executable).name,
            "renderer_version": version_text[0] if version_text else "unknown",
            "dpi_requested": png_dpi,
            "dpi_metadata": list(png_dpi_info) if png_dpi_info else None,
            "font": font_metadata(),
            "png_mode": png_mode,
            "target_pixels": list(png_target_pixels) if png_target_pixels else None,
            "normalized_blank_right_edge": normalized_blank_right_edge,
        },
    }
    qa = publicize_paths(qa)
    return {"drawing": drawing, "pdf": pdf, "svg": svg, "png": png, "qa": qa}


def mm_drawing(width_mm: float, height_mm: float) -> Drawing:
    """创建毫米坐标的空白 Drawing。"""

    register_arial_aliases()
    return Drawing(width_mm * mm, height_mm * mm)


def pad_drawing(
    drawing: Drawing,
    *,
    left_mm: float = 0.0,
    bottom_mm: float = 0.0,
    right_mm: float = 0.0,
    top_mm: float = 0.0,
) -> Drawing:
    """Add white canvas margins around a standalone panel without scaling it."""

    if min(left_mm, bottom_mm, right_mm, top_mm) < 0:
        raise ValueError("drawing padding must be nonnegative")
    target = mm_drawing(
        drawing.width / mm + left_mm + right_mm,
        drawing.height / mm + bottom_mm + top_mm,
    )
    target.add(
        Rect(
            0,
            0,
            target.width,
            target.height,
            fillColor=HexColor("#FFFFFF"),
            strokeColor=None,
        )
    )
    contents = Group(*drawing.contents)
    contents.translate(left_mm * mm, bottom_mm * mm)
    target.add(contents)
    return target


def crop_drawing(source: Drawing, x_mm: float, y_mm: float, width_mm: float, height_mm: float) -> Drawing:
    """截取原 A4 坐标中的 panel 区域，并平移到独立画布原点。"""

    target = mm_drawing(width_mm, height_mm)
    group = Group()
    for item in source.contents:
        group.add(item)
    group.translate(-x_mm * mm, -y_mm * mm)
    target.add(group)
    return target


def crop_drawing_clipped(
    source: Drawing,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
    *,
    source_clip_height_mm: float | None = None,
) -> Drawing:
    """Copy only source shapes intersecting a crop rectangle, then translate them."""

    target = mm_drawing(width_mm, height_mm)
    group = Group()
    x0, y0 = x_mm * mm, y_mm * mm
    clip_height = height_mm if source_clip_height_mm is None else source_clip_height_mm
    x1, y1 = (x_mm + width_mm) * mm, (y_mm + clip_height) * mm
    for item in source.contents:
        bounds = item.getBounds()
        if bounds is None:
            continue
        bx0, by0, bx1, by1 = bounds
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        if isinstance(item, Rect) and bx0 <= 0 and by0 <= 0 and bx1 >= source.width and by1 >= source.height:
            continue
        group.add(item)
    group.translate(-x0, -y0)
    target.add(Rect(0, 0, target.width, target.height, fillColor=HexColor("#FFFFFF"), strokeColor=None))
    target.add(group)
    return target
