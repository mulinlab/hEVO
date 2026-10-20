#!/usr/bin/env python3
"""Compatibility entry point retained for older figure build commands.

Figure builds no longer emit generated source manifests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve()
BENCHMARK = HERE.parents[2]


def build(figure: str, root: Path) -> None:
    """Preserve the old callable API without writing a manifest file."""
    del root
    print(f"{figure}: generated source manifests are disabled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=[f"FigS{number}" for number in range(1, 8)],
        help="Retained for compatibility; defaults to FigS1 through FigS7.",
    )
    args = parser.parse_args()
    figures = args.figures or [f"FigS{number}" for number in range(1, 8)]
    for figure in figures:
        build(figure, args.benchmark_root.resolve())


if __name__ == "__main__":
    main()
