"""Repository-local paths for the hEvo publication figures."""
from __future__ import annotations
from pathlib import Path
import os
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT
TABLES_ROOT = REPO_ROOT / "tables"
FIGURE_ROOT = REPO_ROOT / "figures"

def tables_dir(override: Path | None = None) -> Path:
    path = Path(override).expanduser().resolve() if override is not None else Path(os.environ.get('HEVO_TABLES_DIR', TABLES_ROOT)).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"tables directory is missing: {path}")
    return path

def table_path(filename: str, override: Path | None = None) -> Path:
    path = tables_dir(override) / filename
    if not path.is_file():
        raise FileNotFoundError(f"required table is missing: {path}")
    return path

def figure_dir(figure: str) -> Path:
    path = FIGURE_ROOT / figure
    path.mkdir(parents=True, exist_ok=True)
    return path

def cache_dir(name: str) -> Path:
    """Return a per-user temporary cache, overridable for cluster deployments."""

    configured = os.environ.get("HEVO_FIGURE_CACHE_DIR")
    root = Path(configured).expanduser() if configured else Path(tempfile.gettempdir()) / "hevo_figures_runtime"
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def data_dir(figure: str | None = None) -> Path:
    return tables_dir()
