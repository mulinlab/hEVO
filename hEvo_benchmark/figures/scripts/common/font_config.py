"""Resolve and register the user-supplied Arial fonts for all figure backends.

The plotting code uses the logical ReportLab/Matplotlib family name ``Arial``.
This module makes that name resolve to the real Arial TTF files installed in a
user font directory, without embedding a machine-specific absolute path in the
repository.  ``HEVO_ARIAL_DIR`` may be set for a project-specific font bundle;
the normal user font directory is the default.
"""

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
_REPORTLAB_REGISTERED = False
_MPL_REGISTERED = False


def _font_directories() -> list[Path]:
    directories: list[Path] = []
    configured = os.environ.get("HEVO_ARIAL_DIR")
    if configured:
        directories.append(Path(configured).expanduser())
    directories.extend(
        [
            Path.home() / ".local" / "share" / "fonts" / "Arial",
            Path.home() / ".fonts" / "Arial",
            Path("/usr/local/share/fonts/Arial"),
            Path("/usr/share/fonts/truetype/msttcorefonts"),
            Path("/usr/share/fonts/truetype/msttcorefonts"),
        ]
    )
    unique: list[Path] = []
    for directory in directories:
        if directory not in unique:
            unique.append(directory)
    return unique


def resolve_arial_paths() -> dict[str, Path]:
    """Return the four required Arial faces, failing closed if absent."""

    for directory in _font_directories():
        paths = {name: directory / filename for name, filename in _FONT_FILES.items()}
        if all(path.is_file() for path in paths.values()):
            return paths
    searched = ", ".join(str(path) for path in _font_directories())
    raise FileNotFoundError(
        "Arial regular/bold/italic/bold-italic TTF files are required. "
        f"Searched: {searched}. Set HEVO_ARIAL_DIR to a directory containing "
        "arial.ttf, arialbd.ttf, ariali.ttf, and arialbi.ttf."
    )


def register_reportlab_arial() -> dict[str, str]:
    """Register real Arial faces under stable names used by ReportLab."""

    global _REPORTLAB_REGISTERED
    paths = resolve_arial_paths()
    if not _REPORTLAB_REGISTERED:
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
        _REPORTLAB_REGISTERED = True
    return {name: str(path) for name, path in paths.items()}


def register_matplotlib_arial() -> dict[str, str]:
    """Add real Arial faces to Matplotlib and return the resolved paths."""

    from matplotlib import font_manager

    global _MPL_REGISTERED
    paths = resolve_arial_paths()
    if not _MPL_REGISTERED:
        for path in paths.values():
            font_manager.fontManager.addfont(str(path))
        _MPL_REGISTERED = True
    return {name: str(path) for name, path in paths.items()}


def font_metadata() -> dict[str, object]:
    """Return traceable font identity information for QA manifests."""

    paths = resolve_arial_paths()
    return {
        "family": "Arial",
        "source": "installed TrueType font files; installation path omitted",
        "filenames": {name: path.name for name, path in paths.items()},
    }
