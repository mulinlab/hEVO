#!/usr/bin/env python3
"""Reproduce the supplied seven-panel Fig5 from local S41/S42/S43/S49 tables."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
SCRIPT_ROOT = HERE.parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from Fig5._pdf_reference import build_reference, export


def build_panel(tables_dir=None, **kwargs):
    return build_reference(Path(tables_dir or HERE.parents[3] / "tables"))


def export_panel(output_dir=None, tables_dir=None, export_panels=False):
    benchmark = HERE.parents[3]
    return export(
        Path(tables_dir or benchmark / "tables"),
        Path(output_dir or benchmark / "figures" / "Fig5"),
        export_panels=export_panels,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export-panels", action="store_true",
                        help="Refresh standalone panels from the same generated PDF")
    args = parser.parse_args()
    print(export_panel(args.output_dir, args.tables_dir, args.export_panels))


if __name__ == "__main__":
    main()
