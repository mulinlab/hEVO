from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

PALETTE = {
    "hevo_blue": "#0B5FA5",
    "hevo_blue_light": "#DCEAF7",
    "evo_green": "#3C8D5A",
    "evo_green_light": "#DCEFE4",
    "ancestry_purple": "#8B79B9",
    "ancestry_purple_light": "#ECE7F6",
    "neutral_navy": "#24435B",
    "soft_charcoal": "#334155",
    "text_muted": "#64748B",
    "grid_grey": "#D8E1EA",
    "panel_edge": "#C9D5E2",
    "paper_bg": "#FFFFFF",
    "panel_bg": "#FAFCFE",
}

PAPER_BACKGROUND = PALETTE["paper_bg"]
PANEL_BACKGROUND = PALETTE["panel_bg"]
PANEL_BORDER = "#000000"
TEXT_DARK = "#000000"
TEXT_MUTED = "#000000"
GRID_GREY = PALETTE["grid_grey"]
PURE_BLACK = "#000000"

POPULATION_COLORS = {
    "EUR": "#E9F3FF",
    "AFR": "#B7D8F7",
    "EAS": "#619BEA",
    "SAS": "#FFC7DF",
    "Others": "#FF79A7",
}

MODEL_FAMILY_COLORS = {
    "hEvo family": "#FF79A7",
    "NT family": "#619BEA",
    "Evo2 family": "#59B0B9",
}

BENCHMARK_DOMAIN_COLORS = {
    "Genomic element classification": PALETTE["hevo_blue"],
    "Variant scoring": PALETTE["evo_green"],
    "Evolutionary and ancestry analysis": PALETTE["ancestry_purple"],
}

METHOD_STAGE_COLORS = {
    "Corpus building": PALETTE["hevo_blue_light"],
    "Stage I pre-training": PALETTE["hevo_blue"],
    "Stage II refinement": PALETTE["neutral_navy"],
    "Benchmark evaluation": PALETTE["evo_green_light"],
}

DIVERSITY_COLORS = {
    "Shannon": "#70A6D4",
    "Simpson": "#FF7BA7",
}

RADAR_RING_ODD = "#F8FAFC"
RADAR_RING_EVEN = "#F1F5F9"
RADAR_RING_EDGE = GRID_GREY
RADAR_SPOKE = GRID_GREY

# Backward-compatible aliases for existing panel scripts.
FAMILY_COLORS = MODEL_FAMILY_COLORS
CATEGORY_ACCENTS = BENCHMARK_DOMAIN_COLORS


def apply_pub_rcparams(base_font_size: float = 10) -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["font.size"] = base_font_size
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.facecolor"] = PAPER_BACKGROUND
    plt.rcParams["axes.facecolor"] = PAPER_BACKGROUND
    plt.rcParams["savefig.facecolor"] = PAPER_BACKGROUND
    plt.rcParams["text.color"] = PURE_BLACK
    plt.rcParams["axes.edgecolor"] = PURE_BLACK
    plt.rcParams["axes.labelcolor"] = PURE_BLACK
    plt.rcParams["axes.titlecolor"] = PURE_BLACK
    plt.rcParams["xtick.color"] = PURE_BLACK
    plt.rcParams["ytick.color"] = PURE_BLACK
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.minor.width"] = 0.8
    plt.rcParams["ytick.minor.width"] = 0.8


def style_black_axes(
    ax: plt.Axes,
    *,
    labelsize: float = 13,
    tick_labelsize: float = 11,
    spine_width: float = 1.0,
    tick_width: float = 1.0,
    tick_length: float = 4.0,
    hide_top_right: bool = True,
    grid: bool = False,
) -> None:
    ax.grid(grid)
    ax.xaxis.label.set_color(PURE_BLACK)
    ax.yaxis.label.set_color(PURE_BLACK)
    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)
    ax.tick_params(
        axis="both",
        which="major",
        colors=PURE_BLACK,
        labelcolor=PURE_BLACK,
        labelsize=tick_labelsize,
        width=tick_width,
        length=tick_length,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        colors=PURE_BLACK,
        width=max(0.8, tick_width * 0.8),
        length=max(2.0, tick_length * 0.6),
    )
    for spine_name, spine in ax.spines.items():
        if hide_top_right and spine_name in {"top", "right"}:
            spine.set_visible(False)
            continue
        spine.set_visible(True)
        spine.set_color(PURE_BLACK)
        spine.set_linewidth(spine_width)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_color(PURE_BLACK)
        tick_label.set_fontweight("normal")


def text_color_for_fill(hex_color: str) -> str:
    r, g, b = to_rgb(hex_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.52 else TEXT_DARK
