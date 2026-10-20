#!/usr/bin/env python3
"""Compatibility entry point for merging GPN-Star score shards.

The original node4 launcher used this historical path, while the maintained
implementation lives under ``scripts/evaluate``.  Keep both entry points
working so that completed shards can be merged without rerunning inference.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    implementation = (
        Path(__file__).resolve().parents[1]
        / "runtime"
        / "gpn_star_merge.py"
    )
    if not implementation.is_file():
        raise FileNotFoundError(
            f"maintained merge implementation is missing: {implementation}"
        )
    runpy.run_path(str(implementation), run_name="__main__")
