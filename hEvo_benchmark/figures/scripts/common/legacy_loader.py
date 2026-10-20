"""Load frozen low-level geometry backends bundled with this repository."""
from __future__ import annotations
import importlib
import sys
from pathlib import Path
from .paths import REPO_ROOT

UPSTREAM_ROOT = REPO_ROOT / "figures" / "scripts" / "_backend"
FORMAL_ROOT = UPSTREAM_ROOT / "figures_scripts_formal"
COMMON_ROOT = UPSTREAM_ROOT / "figures" / "scripts" / "common"
for candidate in (UPSTREAM_ROOT, COMMON_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

def load_module(relative_path: str):
    path = FORMAL_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"bundled implementation is missing: {path}")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    return importlib.import_module(path.stem)
