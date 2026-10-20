"""Matplotlib panel 的统一字体、导出和 QA 配置。"""

from __future__ import annotations

import json
import matplotlib as mpl
from matplotlib.figure import Figure
from PIL import Image
from pypdf import PdfReader

from .data_io import sha256
from .font_config import font_metadata, register_matplotlib_arial
from .public_paths import publicize_paths, recorded_path


def configure_matplotlib() -> None:
    """Register the real Arial faces and retain the existing export contract."""

    register_matplotlib_arial()

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def export_figure(
    fig: Figure,
    output_dir: Path,
    stem: str,
    input_files: list[Path],
    width_mm: float,
    height_mm: float,
) -> dict[str, object]:
    """导出 Matplotlib panel 并写入 QA JSON。"""

    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    with Image.open(png) as image:
        pixels = list(image.size)
        image.verify()
    qa = {
        "status": "PASS",
        "status_scope": "render_and_artifact_integrity_only",
        "backend": "Matplotlib",
        "inputs": [{"path": recorded_path(path), "sha256": sha256(path)} for path in input_files],
        "outputs": {
            "pdf": {"path": recorded_path(pdf), "sha256": sha256(pdf), "pages": len(PdfReader(str(pdf)).pages)},
            "svg": {"path": recorded_path(svg), "sha256": sha256(svg), "bytes": svg.stat().st_size},
            "png": {"path": recorded_path(png), "sha256": sha256(png), "pixels": pixels},
        },
        "expected_size_mm": [width_mm, height_mm],
        "font": font_metadata(),
    }
    qa = publicize_paths(qa)
    return {"figure": fig, "pdf": pdf, "svg": svg, "png": png, "qa": qa}
