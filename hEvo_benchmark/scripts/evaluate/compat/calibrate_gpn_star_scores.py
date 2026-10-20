#!/usr/bin/env python3
"""Compatibility entry point for :mod:`revision_consistency.calibrate_gpn_star_scores`."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

EVALUATE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = EVALUATE_DIR.parent
for _path in (SCRIPTS_DIR, EVALUATE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_implementation = importlib.import_module(
    "revision_consistency.calibrate_gpn_star_scores"
)
for _name, _value in vars(_implementation).items():
    if _name not in {"__name__", "__loader__", "__package__", "__spec__"}:
        globals()[_name] = _value

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
