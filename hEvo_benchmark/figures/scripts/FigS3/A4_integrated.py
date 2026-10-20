#!/usr/bin/env python3
"""Render FigS3's Genomic Benchmark and NT Benchmark matrices."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from common.supplementary_v8_figures import export_figure

def build_panel(tables_dir: Path | None = None, **_: object):
    return export_figure("FigS3", tables_dir=tables_dir)

def export_panel(output_dir: Path | None = None, tables_dir: Path | None = None):
    return export_figure("FigS3", tables_dir=tables_dir, output_dir=output_dir)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(export_panel(args.output_dir, args.tables_dir))

if __name__ == "__main__":
    main()
