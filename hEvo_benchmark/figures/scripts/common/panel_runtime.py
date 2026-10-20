"""Shared runtime helpers for table-backed figure panels."""
from __future__ import annotations
from pathlib import Path
from .paths import tables_dir, figure_dir

def panel_tables_dir(figure_or_override=None, override: Path | None = None) -> Path:
    candidate = override if override is not None else (figure_or_override if isinstance(figure_or_override, Path) else None)
    return tables_dir(candidate)

def panel_data_dir(figure_or_override=None, override: Path | None = None) -> Path:
    return panel_tables_dir(figure_or_override, override)

def panel_output_dir(figure: str, override: Path | None = None) -> Path:
    return (Path(override).expanduser().resolve() if override is not None else figure_dir(figure))

def input_files(*paths: Path) -> list[Path]:
    checked = []
    for path in paths:
        path = Path(path).resolve()
        if path.is_dir():
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        checked.append(path)
    return checked
