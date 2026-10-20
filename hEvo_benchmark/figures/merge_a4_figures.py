#!/usr/bin/env python3
"""Merge all main and supplementary A4 figures into one bookmarked PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


MAIN_FIGURES = tuple(f"Fig{index}" for index in range(1, 6))
SUPPLEMENTARY_FIGURES = tuple(f"FigS{index}" for index in range(1, 14))
DEFAULT_FIGURES = MAIN_FIGURES + SUPPLEMENTARY_FIGURES
A4_POINTS = (595.2756, 841.8898)
A4_TOLERANCE_POINTS = 2.0


def page_size(page) -> tuple[float, float]:
    return float(page.mediabox.width), float(page.mediabox.height)


def is_a4(width: float, height: float) -> bool:
    portrait = A4_POINTS
    landscape = A4_POINTS[::-1]
    return any(
        abs(width - expected_width) <= A4_TOLERANCE_POINTS
        and abs(height - expected_height) <= A4_TOLERANCE_POINTS
        for expected_width, expected_height in (portrait, landscape)
    )


def parse_args() -> argparse.Namespace:
    figures_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=figures_dir,
        help="Directory containing Fig*/A4_integrated.pdf (default: script directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=figures_dir / "hEVO_all_figures_A4.pdf",
        help="Merged PDF path.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=list(DEFAULT_FIGURES),
        help="Ordered figure directory names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.figures_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if len(args.figures) != len(set(args.figures)):
        raise ValueError("--figures contains duplicate names")

    writer = PdfWriter()
    for page_index, figure_name in enumerate(args.figures):
        source = figures_dir / figure_name / "A4_integrated.pdf"
        if not source.is_file():
            raise FileNotFoundError(f"missing A4 figure: {source}")
        reader = PdfReader(source)
        if len(reader.pages) != 1:
            raise ValueError(f"{source} must contain exactly one page; found {len(reader.pages)}")
        width, height = page_size(reader.pages[0])
        if not is_a4(width, height):
            raise ValueError(
                f"{source} is not A4: {width:.3f} x {height:.3f} pt"
            )
        writer.append(source, import_outline=False)
        writer.add_outline_item(figure_name, page_index)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)

    merged = PdfReader(output)
    if len(merged.pages) != len(args.figures):
        raise RuntimeError(
            f"merged page count mismatch: expected {len(args.figures)}, found {len(merged.pages)}"
        )
    print(f"wrote {output} ({len(args.figures)} pages)")


if __name__ == "__main__":
    main()
