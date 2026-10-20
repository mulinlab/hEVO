"""Focused reference-style panel for the Fig. 4e HAQER comparison."""
from pathlib import Path

from Fig4.e_haqer_gpn_comparison import (
    REFGENE,
    build_panel as _draw_panel,
    source_files,
)

PAGE_W, PAGE_H = 595.2755905512, 841.8897637795


def build_panel(root, *, figure):
    rect = [104.88 / PAGE_W, (PAGE_H - 601.8) / PAGE_H,
            467.72 / PAGE_W, (601.8 - 415.28) / PAGE_H]
    return _draw_panel(root, figure=figure, rect=rect)
