"""Arial resolver shared by the legacy/formal ReportLab backends."""

from __future__ import annotations

import os
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


_FONT_FILES = {
    "Arial": "arial.ttf",
    "Arial-Bold": "arialbd.ttf",
    "Arial-Italic": "ariali.ttf",
    "Arial-BoldItalic": "arialbi.ttf",
}
_REGISTERED = False


def _directories() -> list[Path]:
    values: list[Path] = []
    configured = os.environ.get("HEVO_ARIAL_DIR")
    if configured:
        values.append(Path(configured).expanduser())
    values.extend(
        [
            Path.home() / ".local" / "share" / "fonts" / "Arial",
            Path.home() / ".fonts" / "Arial",
            Path("/usr/local/share/fonts/Arial"),
            Path("/usr/share/fonts/truetype/msttcorefonts"),
        ]
    )
    return list(dict.fromkeys(values))


def resolve_arial_paths() -> dict[str, Path]:
    for directory in _directories():
        paths = {name: directory / filename for name, filename in _FONT_FILES.items()}
        if all(path.is_file() for path in paths.values()):
            return paths
    searched = ", ".join(str(path) for path in _directories())
    raise FileNotFoundError(
        "Arial TTF files are required (arial.ttf, arialbd.ttf, ariali.ttf, "
        f"arialbi.ttf). Searched: {searched}; set HEVO_ARIAL_DIR if needed."
    )


def register_reportlab_arial() -> dict[str, str]:
    global _REGISTERED
    paths = resolve_arial_paths()
    if not _REGISTERED:
        for name, path in paths.items():
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily(
            "Arial",
            normal="Arial",
            bold="Arial-Bold",
            italic="Arial-Italic",
            boldItalic="Arial-BoldItalic",
        )
        _REGISTERED = True
    return {name: str(path) for name, path in paths.items()}
