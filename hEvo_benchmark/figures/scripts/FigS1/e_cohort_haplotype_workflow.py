#!/usr/bin/env python3
"""Export FigS1e, the open five-step haplotype corpus workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS_ROOT = HERE.parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.paths import tables_dir, figure_dir  # noqa: E402
from FigS1 import _e_workflow_open_v3 as backend  # noqa: E402


WIDTH_MM, HEIGHT_MM = 188.0, 48.0
PNG_DPI = 600
STEM = "e_cohort_haplotype_workflow"


def build_panel(tables_dir: Path | None = None):
    """Build the reviewed open workflow after checking its displayed counts."""
    return backend.build_panel(tables_dir)


def export_panel(tables_dir: Path | None = None, output_dir: Path | None = None):
    """Write standalone PDF/SVG/PNG and the panel QA JSON."""
    out = output_dir or figure_dir("FigS1")
    return backend.export_panel(tables_dir=tables_dir, output_dir=out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", "--data-dir", dest="tables_dir", type=Path, default=tables_dir("FigS1"))
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=figure_dir("FigS1"))
    args = parser.parse_args()
    print(export_panel(args.tables_dir, args.output_dir))


if __name__ == "__main__":
    main()
