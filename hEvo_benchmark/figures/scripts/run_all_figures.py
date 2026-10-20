#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve()
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
JOBS = [
    ("Fig1", "A4_integrated"), ("Fig2", "A4_integrated"),
    ("Fig3", "A4_integrated"), ("Fig4", "A4_integrated"),
    ("Fig5", "A4_integrated"),
    ("FigS1", "A4_integrated"), ("FigS2", "A4_integrated"),
    ("FigS3", "A4_integrated"), ("FigS4", "A4_integrated"),
    ("FigS5", "A4_integrated"), ("FigS6", "A4_integrated"),
    ("FigS7", "A4_integrated"),
    ("FigS8", "A4_integrated"), ("FigS9", "A4_integrated"),
    ("FigS10", "A4_integrated"), ("FigS11", "A4_integrated"),
    ("FigS12", "A4_integrated"), ("FigS13", "A4_integrated"),
]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", nargs="+", choices=[name for name, _ in JOBS],
                        help="Render only the selected figures, in publication order.")
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    canonical_out = HERE.parent.parent.resolve()
    # The legacy S1/S2/S7 builders write to their registered figure folders.
    # Keep the batch output coherent by requiring the canonical destination.
    if out != canonical_out:
        raise ValueError(f"--output-dir must be the canonical figure directory: {canonical_out}")
    if args.tables_dir is not None:
        import os
        os.environ["HEVO_TABLES_DIR"] = str(args.tables_dir.resolve())
    for figure, module_name in JOBS:
        if args.figures is not None and figure not in args.figures:
            continue
        if figure in {"FigS2", "FigS8", "FigS9"}:
            script = HERE.parent / figure / f"{module_name}.py"
            subprocess.run([sys.executable, str(script)], check=True, cwd=HERE.parents[2])
            print(f"{figure}: {out/figure/'A4_integrated.pdf'}")
            continue
        module = importlib.import_module(f"{figure}.{module_name}")
        if figure == "FigS1":
            result = module.build()
        elif figure == "FigS7":
            result = module.export()
        elif figure in {"Fig5", "FigS3", "FigS10", "FigS11", "FigS12", "FigS13"}:
            result = module.export_panel(output_dir=out/figure, tables_dir=args.tables_dir)
        else:
            result = module.export_panel(output_dir=out/figure)
        print(f"{figure}: {out / figure / 'A4_integrated.pdf'}", flush=True)
if __name__ == "__main__":
    main()
