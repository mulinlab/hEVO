"""Render the formal PDF-style HAR/HAQER genome-browser figure.

The renderer is deliberately separate from the legacy plots and the earlier
mockup.  It uses one compact coordinate canvas per locus, a fixed physical
figure size, a shared track order, direct labels, and a strict source manifest.
It never invents values for missing tracks.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from prepare_formal_tracks import PANEL_META, TRACK_SPECS


DEFAULT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "results/final_har_haqer_panel/formal_main_figure"
MANIFEST_NAME = "Figure4g_h_HAR_HAQER_tracks_manifest.json"
FORMAL_BASENAME = "Figure4g_h_HAR_HAQER_formal"
PANEL_BASENAMES = {
    "g": "Figure4g_HAR_NPAS3_formal",
    "h": "Figure4h_HAQER_chr19p13.3_formal",
}

HAR_SCRIPT = Path("results/literature_example_tracks_beautified/beautify_har_npas3_optimized.py")
HAQER_SCRIPT = Path("results/haqer_meaningful_region_chr19/plot_haqer_chr19_candidate.py")
MOCKUP_SCRIPT = Path("results/final_har_haqer_panel/plot_har_haqer_mockup.py")


# Colors are centralized so the whole figure keeps the low-saturation tutorial
# vocabulary rather than inheriting colors from the legacy plotting scripts.
PALETTE = {
    "paper": "#FFFFFF",
    "navy": "#173F63",
    "text": "#171717",
    "orange": "#C86A24",
    "orange_soft": "#E7B08D",
    "teal": "#178C96",
    "teal_soft": "#8BC8CC",
    "gold": "#B4873F",
    "charcoal": "#485058",
    "terracotta": "#B86B3E",
    "green": "#5F8E6B",
    "blue": "#4775A4",
    "purple": "#7A6D9D",
    "gerp": "#778A81",
    "pink_hit": "#D58A7C",
    "purple_near": "#8672A8",
    "blue_other": "#A5C4D8",
    "spotlight": "#F2DCCB",
    "separator": "#8E9391",
    "baseline": "#7F8585",
    "muted": "#7B858A",
    "light": "#D9DEDC",
}

TRACK_COLORS = {
    "hevo": PALETTE["orange"],
    "evo2": PALETTE["teal"],
    "peaks": PALETTE["orange"],
    "known": PALETTE["navy"],
    "h3k27ac": PALETTE["gold"],
    "atac": PALETTE["charcoal"],
    "human_ape_delta": PALETTE["terracotta"],
    "ccre_cage": PALETTE["gold"],
    "phastcons": PALETTE["green"],
    "human_branch_phylop": PALETTE["blue"],
    "ape_divergence": PALETTE["purple"],
    "gnocchi": PALETTE["green"],
    "cadd": PALETTE["gold"],
    "gerp": PALETTE["gerp"],
    "linsight": PALETTE["blue"],
    "genes": PALETTE["navy"],
}

FIXED_SCALES: dict[str, tuple[float, float]] = {
    "hevo": (0.0, 1.0),
    "phastcons": (0.0, 1.0),
    "human_branch_phylop": (-5.0, 6.0),
    "gerp": (-5.0, 6.0),
    "linsight": (0.0, 1.0),
}

ROW_HEIGHTS = {
    "score": 3.1,
    "peaks": 1.85,
    "intervals": 1.90,
    "signal": 1.75,
    "genes": 3.0,
}
ROW_GAP = 0.22
GROUP_GAP = 0.68
TITLE_SPACE = 2.8
X_AXIS_SPACE = 2.8

# The full NPAS3 transcript spans most of the HAR panel.  For the compact
# tutorial-style genome row, show only the segment surrounding the two
# project-highlighted NPAS3 HARs while retaining the full locus x-axis.
GENE_FOCUS_WINDOWS: dict[str, dict[str, tuple[int, int]]] = {
    "g": {"NPAS3": (33_560_000, 33_720_000)},
}


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        style="teacher",
        root=root,
        output_dir=root / "results/final_har_haqer_panel/formal_main_figure",
        track_dir=None,
        input_npz=None,
        peaks_tsv=None,
        overlap_json=None,
        gold_csv=None,
        raw_har_csv=None,
        haqer_bed=None,
        refgene_gz=None,
        h3k27ac_bw=None,
        phastcons_bw=None,
        ccre_bb=None,
        cadd_position_tsv=None,
        phylop_position_tsv=None,
        cadd_track_tsv=None,
        phylop_json=None,
        evo2_track_npz=None,
        evo2_track_field="evo2_1b_raw_likelihood",
    )


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}; run prepare_formal_tracks.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "formal_har_haqer_tracks_v1":
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema_version')}")
    return payload


def entries_by_key(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(entry["panel"]), str(entry["key"])): entry for entry in manifest["tracks"]}


def strict_failures(manifest: dict[str, Any]) -> list[str]:
    return [
        f"{entry['panel']}:{entry['key']}={entry['status']}"
        for entry in manifest["tracks"]
        if entry.get("required_for_formal", True)
        and (entry.get("status") != "verified_real" or entry.get("formal_eligible") is False)
    ]


def failures_for_panels(manifest: dict[str, Any], panels: list[dict[str, Any]]) -> list[str]:
    panel_ids = {str(panel["panel"]) for panel in panels}
    return [
        f"{entry['panel']}:{entry['key']}={entry['status']}"
        for entry in manifest["tracks"]
        if str(entry["panel"]) in panel_ids
        and entry.get("required_for_formal", True)
        and (entry.get("status") != "verified_real" or entry.get("formal_eligible") is False)
    ]


def configure_source_modules(root: Path) -> tuple[ModuleType, ModuleType]:
    har = load_module(root / HAR_SCRIPT, "formal_har_source")
    haqer = load_module(root / HAQER_SCRIPT, "formal_haqer_source")
    args = source_args(root)
    har.configure_paths(args)
    haqer.configure_paths(args)
    return har, haqer


def collect_panels(root: Path) -> list[dict[str, Any]]:
    helper = load_module(root / MOCKUP_SCRIPT, "formal_existing_data_helpers")
    har, haqer = configure_source_modules(root)
    return [helper.collect_har(har), helper.collect_haqer(haqer)]


def finite_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return
    start = int(indices[0])
    previous = start
    for index in indices[1:]:
        index = int(index)
        if index != previous + 1:
            yield start, previous + 1
            start = index
        previous = index
    yield start, previous + 1


def read_text_rows(path: Path) -> list[dict[str, str]]:
    opener = path.open
    if path.name.endswith(".gz"):
        import gzip

        opener = gzip.open  # type: ignore[assignment]
    with opener("rt", encoding="utf-8", errors="replace", newline="") as handle:  # type: ignore[arg-type]
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def find_field(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def read_signal_file(path: Path, chrom: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npz"):
        payload = np.load(path, allow_pickle=False)
        x = None
        y = None
        for name in ("x", "center", "centers", "position", "positions"):
            if name in payload:
                x = np.asarray(payload[name], dtype=float)
                break
        for name in ("evo2_1b_raw_likelihood", "values", "value", "score", "scores", "likelihood", "signal", "track"):
            if name in payload:
                y = np.asarray(payload[name], dtype=float)
                break
        if x is None or y is None:
            raise ValueError(f"Cannot infer x/value arrays from {path}")
        return x.reshape(-1), y.reshape(-1)
    if suffixes.endswith(".json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for x_name in ("x", "center", "position", "positions"):
                for y_name in ("values", "value", "score", "scores", "likelihood", "signal"):
                    if x_name in payload and y_name in payload:
                        return np.asarray(payload[x_name], dtype=float), np.asarray(payload[y_name], dtype=float)
            rows = payload.get("track") or payload.get("data") or payload.get("values")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                return rows_to_signal(rows, chrom, start, end)
        raise ValueError(f"Cannot infer x/value arrays from {path}")
    if suffixes.endswith(".bw") or suffixes.endswith(".bigwig") or suffixes.endswith(".bigwig"):
        try:
            import pyBigWig
        except ImportError as exc:
            raise RuntimeError("pyBigWig is required for BigWig formal tracks") from exc
        bins = 1200
        with pyBigWig.open(str(path)) as handle:
            values = handle.stats(chrom, start, end, nBins=bins, exact=True)
        x = np.linspace(start, end, bins, endpoint=False, dtype=float) + (end - start) / bins / 2
        return x, np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)
    rows = read_text_rows(path)
    return rows_to_signal(rows, chrom, start, end)


def rows_to_signal(rows: list[dict[str, Any]], chrom: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        row_chrom = find_field(row, ("chrom", "chromosome", "chr"))
        if row_chrom not in (None, "", chrom, chrom.removeprefix("chr"), "chr" + chrom.removeprefix("chr")):
            continue
        left = as_float(find_field(row, ("start", "pos", "position", "center", "x")))
        right = as_float(find_field(row, ("end",)))
        if not np.isfinite(left):
            continue
        if np.isfinite(right):
            x = (left + right) / 2.0
            if right <= start or left >= end:
                continue
        else:
            x = left
        value = as_float(find_field(row, ("value", "score", "scores", "likelihood", "signal", "phylop", "gerp", "linsight")))
        if np.isfinite(x) and start <= x <= end:
            xs.append(float(x))
            ys.append(float(value))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def read_binned_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_text_rows(path)
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        left = as_float(find_field(row, ("start",)))
        right = as_float(find_field(row, ("end",)))
        value = as_float(find_field(row, ("value", "score")))
        if np.isfinite(left) and np.isfinite(right):
            xs.append((left + right) / 2.0)
            ys.append(value)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def load_entry_signal(entry: dict[str, Any], panel: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    path_value = entry.get("path")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or path.stat().st_size == 0:
        return None
    if entry["key"] in {"gerp", "linsight"}:
        return read_binned_track(path)
    return read_signal_file(path, str(panel["chrom"]), int(panel["start"]), int(panel["end"]))


def panel_model_signal(panel: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(panel["x"], dtype=float),
        np.asarray(panel["raw"], dtype=float),
        np.asarray(panel["smooth"], dtype=float),
    )


def categories_for_panel(panel: dict[str, Any]) -> list[str]:
    if "categories" in panel:
        return [str(value) for value in panel["categories"]]
    return [str(row.get("category", "other_non_gold")) for row in panel["peaks"]]


def category_color(category: str) -> str:
    if category in {"overlaps_gold", "direct", "direct_hit"}:
        return PALETTE["pink_hit"]
    if category in {"near_known_har", "near_known", "near"}:
        return PALETTE["purple_near"]
    return PALETTE["blue_other"]


def interval_span(start: int, end: int, region_start: int, region_end: int, minimum: int) -> tuple[int, int]:
    start = max(region_start, int(start))
    end = min(region_end, int(end))
    if end <= start:
        return start, end
    width = end - start
    if width >= minimum:
        return start, end
    center = (start + end) // 2
    left = max(region_start, center - minimum // 2)
    right = min(region_end, left + minimum)
    left = max(region_start, right - minimum)
    return left, right


def spotlight_spans(panel: dict[str, Any]) -> list[tuple[int, int]]:
    start = int(panel["start"])
    end = int(panel["end"])
    minimum = 3_000 if end - start >= 500_000 else 900
    spans: list[tuple[int, int]] = []
    for peak, category in zip(panel["peaks"], categories_for_panel(panel)):
        if category not in {"overlaps_gold", "direct", "direct_hit"}:
            continue
        left = int(peak.get("peak_start", peak.get("start", start)))
        right = int(peak.get("peak_end", peak.get("end", end)))
        spans.append(interval_span(left, right, start, end, minimum))
    merged: list[list[int]] = []
    for left, right in sorted(spans):
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    # Every exact peak/known-HAR overlap is highlighted.  The previous four-
    # span cap hid valid right-side overlaps from the reader.
    return [(left, right) for left, right in merged]


def row_layout(visible_keys: set[str] | None = None) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    specs = [spec for spec in TRACK_SPECS if visible_keys is None or spec["key"] in visible_keys]
    current = TITLE_SPACE
    for index, spec in enumerate(specs):
        height = ROW_HEIGHTS[spec["kind"]]
        rows.append({**spec, "height": height})
        current += height
        current += ROW_GAP
        next_group = specs[index + 1]["group"] if index + 1 < len(specs) else None
        if next_group is not None and next_group != spec["group"]:
            current += GROUP_GAP
    total = current + X_AXIS_SPACE
    cursor = total - TITLE_SPACE
    for index, row in enumerate(rows):
        top = cursor
        bottom = top - row["height"]
        row["top"] = top
        row["bottom"] = bottom
        row["center"] = (top + bottom) / 2
        row["baseline"] = bottom + row["height"] * 0.22
        cursor = bottom - ROW_GAP
        next_group = rows[index + 1]["group"] if index + 1 < len(rows) else None
        if next_group is not None and next_group != row["group"]:
            cursor -= GROUP_GAP
    return rows, total


def fixed_or_dynamic_scale(key: str, values_by_panel: dict[str, np.ndarray]) -> tuple[float, float]:
    if key in FIXED_SCALES:
        return FIXED_SCALES[key]
    values = np.concatenate([value[np.isfinite(value)] for value in values_by_panel.values() if np.any(np.isfinite(value))]) if any(np.any(np.isfinite(value)) for value in values_by_panel.values()) else np.asarray([])
    if values.size == 0:
        return 0.0, 1.0
    low = min(0.0, float(np.nanmin(values)))
    high = float(np.nanmax(values))
    pad = max((high - low) * 0.08, 0.05)
    return low - pad, high + pad


def normalise(values: np.ndarray, scale: tuple[float, float], row: dict[str, Any]) -> np.ndarray:
    low, high = scale
    inner_low = row["bottom"] + row["height"] * 0.18
    inner_high = row["bottom"] + row["height"] * 0.88
    clipped = np.asarray(values, dtype=float)
    finite = np.isfinite(clipped)
    if np.any((clipped[finite] < low) | (clipped[finite] > high)):
        observed_low = float(np.nanmin(clipped[finite]))
        observed_high = float(np.nanmax(clipped[finite]))
        raise ValueError(
            f"Track values [{observed_low}, {observed_high}] exceed plotting scale [{low}, {high}]"
        )
    return inner_low + (clipped - low) / (high - low) * (inner_high - inner_low)


def draw_finite_line(ax, x: np.ndarray, y: np.ndarray, color: str, lw: float, alpha: float, fill_to: float | None = None) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    for left, right in finite_runs(mask):
        if right - left < 1:
            continue
        ax.plot(x[left:right], y[left:right], color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=5)
        if fill_to is not None:
            ax.fill_between(x[left:right], y[left:right], fill_to, color=color, alpha=0.18, linewidth=0, zorder=3)


def draw_signal_row(ax, row: dict[str, Any], x: np.ndarray, values: np.ndarray, scale: tuple[float, float], color: str, fill: bool = True, lw: float = 0.70) -> None:
    if x.size == 0 or values.size == 0:
        return
    y = normalise(values, scale, row)
    draw_finite_line(ax, x, y, color, lw, 0.96, row["baseline"] if fill else None)


def ccre_color(label: str) -> str:
    lower = label.lower()
    if "ctcf" in lower:
        return PALETTE["teal"]
    if "pls" in lower or "promoter" in lower:
        return PALETTE["orange"]
    if "els" in lower or "enhancer" in lower:
        return PALETTE["gold"]
    return PALETTE["blue"]


def draw_interval_row(ax, row: dict[str, Any], intervals: list[dict[str, Any]], region_start: int, region_end: int, color: str | None = None) -> None:
    for index, item in enumerate(intervals):
        left = max(region_start, int(item.get("start", region_start)))
        right = min(region_end, int(item.get("end", region_end)))
        if right <= left:
            continue
        face = color or str(item.get("color") or ccre_color(str(item.get("class", ""))))
        y = row["bottom"] + row["height"] * (0.20 + 0.55 * (index % 2))
        height = row["height"] * 0.26
        ax.add_patch(patches.Rectangle((left, y), right - left, height, facecolor=face, edgecolor="none", alpha=0.90, zorder=6))


def draw_peak_row(ax, row: dict[str, Any], panel: dict[str, Any]) -> None:
    start = int(panel["start"])
    end = int(panel["end"])
    height = row["height"]
    # Keep the two callers in one compact track, but on separate lanes.  The
    # model colours are deliberately the same orange/teal used by the score
    # tracks, so a reader can compare score and calls without a new palette.
    lanes = (
        ("hEVO", panel.get("peaks", []), PALETTE["orange"], PALETTE["orange_soft"], 0.61),
        ("Evo 2", panel.get("evo2_peaks", []), PALETTE["teal"], PALETTE["teal_soft"], 0.17),
    )
    for label, peaks, hit_color, other_color, y_fraction in lanes:
        y = row["bottom"] + height * y_fraction
        bar_height = height * 0.20
        for peak in peaks:
            left = int(peak.get("peak_start", peak.get("start", start)))
            right = int(peak.get("peak_end", peak.get("end", end)))
            summit = int(peak.get("summit_center", peak.get("summit", (left + right) // 2)))
            is_hit = bool(peak.get("overlaps_known", False))
            color = hit_color if is_hit else other_color
            ax.add_patch(
                patches.Rectangle(
                    (left, y),
                    max(1, right - left),
                    bar_height,
                    facecolor=color,
                    edgecolor=hit_color,
                    linewidth=0.18,
                    alpha=0.98,
                    zorder=8,
                )
            )
            ax.vlines(
                summit,
                y - height * 0.06,
                y + bar_height + height * 0.06,
                color=hit_color,
                lw=0.36,
                alpha=0.9,
                zorder=9,
            )
        # A white-backed lane label makes the two colours self-explanatory in
        # SVG/PDF and remains legible when the first peak is near the left edge.
        ax.text(
            start + (end - start) * 0.004,
            y + bar_height * 0.5,
            label,
            ha="left",
            va="center",
            fontsize=4.2,
            color=hit_color,
            bbox={"facecolor": PALETTE["paper"], "edgecolor": "none", "pad": 0.35},
            clip_on=True,
            zorder=11,
        )


def draw_known_row(ax, row: dict[str, Any], panel: dict[str, Any]) -> None:
    minimum = 2_400 if int(panel["end"]) - int(panel["start"]) >= 500_000 else 700
    for index, item in enumerate(panel["known"]):
        left, right = interval_span(int(item["start"]), int(item["end"]), int(panel["start"]), int(panel["end"]), minimum)
        if right <= left:
            continue
        y = row["bottom"] + row["height"] * (0.21 + 0.42 * (index % 2))
        ax.add_patch(patches.Rectangle((left, y), right - left, row["height"] * 0.30, facecolor=PALETTE["navy"], edgecolor="none", alpha=0.96, zorder=8))
        label = str(item.get("label", item.get("name", "")))
        if panel["panel"] == "h" and label and label not in {"HAR", "HAQER"}:
            ax.text((left + right) / 2, row["top"] - 0.05, label, ha="center", va="top", fontsize=5.2, color=PALETTE["navy"], clip_on=True, zorder=9)


def assign_gene_levels(genes: list[dict[str, Any]], max_levels: int = 2) -> list[int]:
    ordered = sorted(enumerate(genes), key=lambda item: (int(item[1]["start"]), int(item[1]["end"])))
    occupied: list[list[tuple[int, int]]] = [[] for _ in range(max_levels)]
    result = [0] * len(genes)
    for original_index, gene in ordered:
        start = int(gene["start"])
        end = int(gene["end"])
        level = 0
        while level < max_levels and any(start < right and left < end for left, right in occupied[level]):
            level += 1
        level = min(level, max_levels - 1)
        occupied[level].append((start, end))
        result[original_index] = level
    return result


def draw_genes(ax, row: dict[str, Any], panel: dict[str, Any]) -> None:
    genes = list(panel["genes"])
    levels = assign_gene_levels(genes)
    for gene, level in zip(genes, levels):
        left = max(int(panel["start"]), int(gene["start"]))
        right = min(int(panel["end"]), int(gene["end"]))
        if right <= left:
            continue
        name = str(gene.get("name", gene.get("symbol", "gene")))
        focus = GENE_FOCUS_WINDOWS.get(str(panel["panel"]), {}).get(name)
        if focus is not None:
            left = max(left, int(focus[0]))
            right = min(right, int(focus[1]))
            if right <= left:
                continue
        highlight = name in {"NPAS3", "SHC2", "ODF3L2"}
        color = PALETTE["navy"] if highlight else PALETTE["muted"]
        y = row["bottom"] + row["height"] * (0.30 + 0.30 * (1 - level))
        ax.plot([left, right], [y, y], color=color, lw=1.10 if highlight else 0.55, solid_capstyle="round", zorder=7)
        strand = str(gene.get("strand", "+"))
        arrow_x = right if strand != "-" else left
        ax.plot([arrow_x], [y], marker=">" if strand != "-" else "<", markersize=3.0 if highlight else 2.0, color=color, zorder=8)
        if highlight:
            ax.text(left, y + row["height"] * 0.18, name, ha="left", va="bottom", fontsize=6.0, fontweight="bold", style="italic", color=color, clip_on=True, zorder=9)


def draw_cadd(ax, row: dict[str, Any], panel: dict[str, Any], scale: tuple[float, float]) -> None:
    records = list(panel["cadd"])
    if not records:
        return
    x = np.asarray([float(item["position"]) for item in records], dtype=float)
    values = np.asarray([float(item["max_phred"]) for item in records], dtype=float)
    draw_signal_row(ax, row, x, values, scale, TRACK_COLORS["cadd"], fill=True, lw=0.58)
    y20 = normalise(np.asarray([20.0]), scale, row)[0]
    ax.plot([int(panel["start"]), int(panel["end"])], [y20, y20], color=PALETTE["muted"], lw=0.38, ls=(0, (2, 2)), zorder=4)
    ax.text(int(panel["end"]), y20 + 0.08, "20", ha="right", va="bottom", fontsize=5.2, color=PALETTE["text"], clip_on=True)


def draw_row_baseline(ax, row: dict[str, Any], start: int, end: int) -> None:
    ax.plot([start, end], [row["baseline"], row["baseline"]], color=PALETTE["baseline"], lw=0.34, alpha=0.92, zorder=2)


def format_position(divisor: float) -> FuncFormatter:
    return FuncFormatter(lambda value, _position: f"{value / divisor:.1f}" if divisor == 1_000_000 else f"{value / divisor:.0f}")


def manual_axis_ticks(ax: plt.Axes, start: int, end: int, x_label: str) -> None:
    """Draw ticks inside the reserved x-axis row so g labels cannot collide with h."""
    step = 100_000 if end - start >= 500_000 else 20_000
    first = int(math.ceil(start / step) * step)
    ticks = np.arange(first, end + 1, step, dtype=float)
    divisor = 1_000_000.0 if end >= 1_000_000 else 1_000.0
    for value in ticks:
        ax.plot([value, value], [0.72, 0.98], color=PALETTE["text"], lw=0.42, zorder=10)
        label = f"{value / divisor:.1f}" if divisor == 1_000_000.0 else f"{value / divisor:.0f}"
        ax.text(value, 0.58, label, ha="center", va="top", fontsize=5.6, color=PALETTE["text"], clip_on=False, zorder=10)
    ax.text((start + end) / 2, 0.02, x_label, ha="center", va="bottom", fontsize=6.1, color=PALETTE["text"], clip_on=False, zorder=10)


def dynamic_values_for_key(key: str, panels: list[dict[str, Any]], entries: dict[tuple[str, str], dict[str, Any]]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for panel in panels:
        panel_id = str(panel["panel"])
        if key == "h3k27ac":
            values = np.asarray(panel["h3k27ac"], dtype=float)
        elif key == "phastcons":
            values = np.asarray(panel["phastcons"], dtype=float)
        elif key == "cadd":
            values = np.asarray([float(item["max_phred"]) for item in panel["cadd"]], dtype=float)
        elif key in {"peaks", "known", "ccre_cage", "genes"}:
            values = np.asarray([], dtype=float)
        else:
            loaded = load_entry_signal(entries.get((panel_id, key), {}), panel)
            values = loaded[1] if loaded is not None else np.asarray([], dtype=float)
        output[panel_id] = values
    return output


def make_extra_intervals(entry: dict[str, Any], panel: dict[str, Any]) -> list[dict[str, Any]]:
    path_value = entry.get("path")
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists() or path.stat().st_size == 0:
        return []
    if path.suffix.lower() in {".bb", ".bigbed"}:
        try:
            import pyBigWig
        except ImportError as exc:
            raise RuntimeError("pyBigWig is required for BigBed interval tracks") from exc
        rows: list[dict[str, Any]] = []
        with pyBigWig.open(str(path)) as handle:
            for left, right, rest in handle.entries(str(panel["chrom"]), int(panel["start"]), int(panel["end"])) or []:
                fields = str(rest).split("\t")
                rows.append({"start": int(left), "end": int(right), "class": fields[0] if fields else "cCRE"})
        return rows
    rows = read_text_rows(path)
    output: list[dict[str, Any]] = []
    for item in rows:
        chrom = find_field(item, ("chrom", "chromosome", "chr"))
        if chrom not in (None, "", panel["chrom"], str(panel["chrom"]).removeprefix("chr"), "chr" + str(panel["chrom"]).removeprefix("chr")):
            continue
        left = as_float(find_field(item, ("start", "pos", "position")))
        right = as_float(find_field(item, ("end",)))
        if not np.isfinite(left) or not np.isfinite(right) or right <= int(panel["start"]) or left >= int(panel["end"]):
            continue
        output.append({"start": int(left), "end": int(right), "class": str(find_field(item, ("class", "name", "label")) or "CAGE")})
    return output


def draw_panel(fig: plt.Figure, ax: plt.Axes, panel: dict[str, Any], entries: dict[tuple[str, str], dict[str, Any]], scales: dict[str, tuple[float, float]], letter: str, rows: list[dict[str, Any]], total: float, label_x: float, group_x: float, letter_x: float = 0.008) -> None:
    panel_id = str(panel["panel"])
    start = int(panel["start"])
    end = int(panel["end"])
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xlim(start, end)
    ax.set_ylim(0, total)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["text"])
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(axis="x", colors=PALETTE["text"], labelbottom=False, width=0.42, length=0, pad=0)
    ax.set_xticks([])
    manual_axis_ticks(ax, start, end, str(PANEL_META[panel_id]["x_label"]))

    for left, right in spotlight_spans(panel):
        ax.axvspan(left, right, color=PALETTE["spotlight"], alpha=0.60, zorder=0)

    for row in rows:
        draw_row_baseline(ax, row, start, end)
        key = row["key"]
        if key == "hevo":
            x, raw, smooth = panel_model_signal(panel)
            draw_signal_row(ax, row, x, raw, scales["hevo"], PALETTE["orange_soft"], fill=False, lw=0.35)
            draw_signal_row(ax, row, x, smooth, scales["hevo"], TRACK_COLORS["hevo"], fill=True, lw=0.78)
        elif key == "evo2":
            loaded = load_entry_signal(entries.get((panel_id, key), {}), panel)
            if loaded is not None:
                draw_signal_row(ax, row, loaded[0], loaded[1], scales["evo2"], TRACK_COLORS["evo2"], fill=True, lw=0.72)
        elif key == "peaks":
            draw_peak_row(ax, row, panel)
        elif key == "known":
            draw_known_row(ax, row, panel)
        elif key == "h3k27ac":
            draw_signal_row(ax, row, np.asarray(panel["x"], dtype=float), np.asarray(panel["h3k27ac"], dtype=float), scales["h3k27ac"], TRACK_COLORS[key], fill=True, lw=0.58)
        elif key == "ccre_cage":
            draw_interval_row(ax, row, list(panel["ccres"]), start, end)
            extra = make_extra_intervals(entries.get((panel_id, key), {}), panel)
            if extra:
                draw_interval_row(ax, row, extra, start, end, color=PALETTE["terracotta"])
        elif key == "phastcons":
            draw_signal_row(ax, row, np.asarray(panel["x"], dtype=float), np.asarray(panel["phastcons"], dtype=float), scales["phastcons"], TRACK_COLORS[key], fill=True, lw=0.58)
        elif key in {"atac", "human_ape_delta", "human_branch_phylop", "ape_divergence", "gnocchi"}:
            loaded = load_entry_signal(entries.get((panel_id, key), {}), panel)
            if loaded is not None:
                draw_signal_row(ax, row, loaded[0], loaded[1], scales[key], TRACK_COLORS[key], fill=True, lw=0.58)
        elif key == "cadd":
            draw_cadd(ax, row, panel, scales["cadd"])
        elif key in {"gerp", "linsight"}:
            loaded = load_entry_signal(entries.get((panel_id, key), {}), panel)
            if loaded is not None:
                draw_signal_row(ax, row, loaded[0], loaded[1], scales[key], TRACK_COLORS[key], fill=True, lw=0.56)
            if key == "gerp":
                zero = normalise(np.asarray([0.0]), scales[key], row)[0]
                ax.plot([start, end], [zero, zero], color=PALETTE["separator"], lw=0.35, zorder=4)
        elif key == "genes":
            draw_genes(ax, row, panel)

    group_ranges: dict[str, list[float]] = {}
    for row in rows:
        group_ranges.setdefault(row["group"], []).append(float(row["center"]))
        y_fig = ax.get_position().y0 + ax.get_position().height * row["center"] / total
        label = row["label"]
        if row["key"] == "known":
            label = "Known HARs" if panel_id == "g" else "Known HAQERs"
        fig.text(label_x, y_fig, label, ha="right", va="center", fontsize=5.35, color=PALETTE["text"])
    for group, centers in group_ranges.items():
        y_fig = ax.get_position().y0 + ax.get_position().height * float(np.mean(centers)) / total
        fig.text(group_x, y_fig, group, ha="center", va="center", rotation=90, fontsize=4.85, fontweight="bold", color=PALETTE["navy"])

    for index, row in enumerate(rows):
        if index == 0 or row["group"] == rows[index - 1]["group"]:
            continue
        y = ax.get_position().y0 + ax.get_position().height * (row["top"] + GROUP_GAP * 0.42) / total
        fig.add_artist(Line2D([0.045, ax.get_position().x1], [y, y], transform=fig.transFigure, color=PALETTE["separator"], lw=0.42, zorder=12))

    ax.text((start + end) / 2, total - 0.82, PANEL_META[panel_id]["title"], ha="center", va="center", fontsize=7.6, fontweight="bold", color=PALETTE["navy"])
    if panel_id == "g":
        meta = f"known recovered: hEVO {panel.get('gold_recovery', 'NA')}"
    else:
        meta = f"known recovered: HAQER {panel.get('direct_recovery', 'NA')}"
    hevo_stats = panel.get("hevo_peak_stats", {})
    evo2_stats = panel.get("evo2_peak_stats", {})
    def peak_rate_text(name: str, stats: dict[str, Any]) -> str:
        count = int(stats.get("peak_count", 0))
        hits = int(stats.get("peak_hit_count", 0))
        rate = stats.get("peak_hit_rate")
        return f"{name} {hits}/{count} ({100.0 * float(rate):.1f}%)" if rate is not None else f"{name} NA"

    def peak_rate_value(stats: dict[str, Any]) -> str:
        count = int(stats.get("peak_count", 0))
        hits = int(stats.get("peak_hit_count", 0))
        rate = stats.get("peak_hit_rate")
        return f"{hits}/{count} ({100.0 * float(rate):.1f}%)" if rate is not None else "NA"

    if panel.get("comparison_mode"):
        call_meta = f"hEVO peak hit: {peak_rate_value(hevo_stats)} | Evo 2 peak hit: {peak_rate_value(evo2_stats)}"
    else:
        call_meta = f"{meta} | peak hit rate: {peak_rate_text('hEVO', hevo_stats)} | {peak_rate_text('Evo 2', evo2_stats)}"
    # Keep the longer two-model statistic line below the centered title in
    # the reserved title band; otherwise it collides with the title at 180 mm.
    ax.text(end, total - 1.62, call_meta, ha="right", va="center", fontsize=4.7, color=PALETTE["text"])
    letter_y = ax.get_position().y1 - 0.003
    fig.text(letter_x, letter_y, letter, ha="left", va="top", fontsize=12.5, fontweight="bold", color=PALETTE["text"])


def compute_scales(panels: list[dict[str, Any]], entries: dict[tuple[str, str], dict[str, Any]]) -> dict[str, tuple[float, float]]:
    scales: dict[str, tuple[float, float]] = {}
    keys = [spec["key"] for spec in TRACK_SPECS]
    for key in keys:
        if key in FIXED_SCALES:
            scales[key] = FIXED_SCALES[key]
            continue
        values = dynamic_values_for_key(key, panels, entries)
        scales[key] = fixed_or_dynamic_scale(key, values)
    return scales


def mpl_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.55,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "savefig.facecolor": PALETTE["paper"],
    })


SMOOTHING_CONTRACT = {
    "kernel": "gaussian",
    "sigma_bp": 512,
    "truncate": 8,
}

PEAK_CALL_CONTRACT = {
    "score_threshold": 0.50,
    "prominence": 0.03,
    "distance_bp": 768,
    "width_bp": 128,
    "merge_gap_bp": 1024,
}


def figure_summary(
    panels: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: Path,
    base: Path,
    incomplete: bool,
    width_mm: float,
    height_mm: float,
    gap_mm: float,
    scales: dict[str, tuple[float, float]],
    panel_mode: str,
) -> dict[str, Any]:
    entries = entries_by_key(manifest)
    track_inputs = {}
    for panel in panels:
        panel_id = str(panel["panel"])
        for spec in TRACK_SPECS:
            entry = entries.get((panel_id, spec["key"]), {})
            track_inputs[f"{panel_id}:{spec['key']}"] = {
                "status": entry.get("status"),
                "formal_eligible": entry.get("formal_eligible"),
                "path": entry.get("path"),
                "sha256": entry.get("sha256"),
                "source_sha256": entry.get("source_sha256"),
                "source_url": entry.get("source_url"),
                "assembly": entry.get("assembly"),
            }
    suffix = base.name.removeprefix(FORMAL_BASENAME)
    summary = {
        "figure": base.name,
        "style": "teacher_pdf_genome_browser_formal",
        "backend": "python-matplotlib",
        "panel_mode": panel_mode,
        "incomplete_layout_preview": incomplete,
        "figure_size_mm": [width_mm, height_mm],
        "inter_panel_gap_mm": gap_mm,
        "panel_order": [str(panel["panel"]) for panel in panels],
        "track_order": [spec["key"] for spec in TRACK_SPECS],
        "track_labels": {spec["key"]: spec["label"] for spec in TRACK_SPECS},
        "scales": {key: [float(value[0]), float(value[1])] for key, value in scales.items()},
        "manifest": str(output_dir / MANIFEST_NAME),
        "outputs": {
            "png": str(base.parent / f"{base.name}.png"),
            "pdf": str(base.parent / f"{base.name}.pdf"),
            "svg": str(base.parent / f"{base.name}.svg"),
        },
        "missing_or_unverified_tracks": failures_for_panels(manifest, panels),
        "track_inputs": track_inputs,
        "smoothing": SMOOTHING_CONTRACT,
        "peak_call": PEAK_CALL_CONTRACT,
        "panels": {
            str(panel["panel"]): {
                "region": f"{panel['chrom']}:{panel['start']}-{panel['end']}",
                "called_peak_count": int(panel["peak_count"]),
                "gold_recovery": panel.get("gold_recovery"),
                "direct_recovery": panel.get("direct_recovery"),
                "spotlight_count": len(spotlight_spans(panel)),
                "hevo_peak_stats": panel.get("hevo_peak_stats"),
                "evo2_peak_stats": panel.get("evo2_peak_stats"),
            }
            for panel in panels
        },
        "source_policy": "all formal tracks must be verified_real; no synthetic values or missing-to-zero conversion",
        "gene_display_focus_windows": {
            panel: {name: [int(start), int(end)] for name, (start, end) in windows.items()}
            for panel, windows in GENE_FOCUS_WINDOWS.items()
        },
    }
    summary_path = output_dir / f"{base.name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def save_figure(fig: plt.Figure, base: Path) -> None:
    # Deliberately omit bbox_inches='tight': fixed physical dimensions are part
    # of the figure contract, including the combined-panel gap.
    fig.savefig(base.parent / f"{base.name}.png", dpi=600)
    fig.savefig(base.parent / f"{base.name}.pdf")
    fig.savefig(base.parent / f"{base.name}.svg")


def render_single(
    panel: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    incomplete: bool,
    scales: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    mpl_style()
    rows, total = row_layout()
    width_mm = float(manifest["style_contract"].get("figure_width_mm", 180.0))
    height_mm = 72.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor=PALETTE["paper"])
    left, right, bottom, top = 0.215, 0.988, 0.075, 0.965
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    draw_panel(
        fig,
        ax,
        panel,
        entries_by_key(manifest),
        scales,
        str(panel["panel"]),
        rows,
        total,
        label_x=0.205,
        group_x=0.046,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_incomplete_layout_preview" if incomplete else ""
    base = output_dir / f"{PANEL_BASENAMES[str(panel['panel'])]}{suffix}"
    save_figure(fig, base)
    plt.close(fig)
    return figure_summary(
        [panel], manifest, output_dir, base, incomplete,
        width_mm, height_mm, 0.0, scales, str(panel["panel"]),
    )


def render(panels: list[dict[str, Any]], manifest: dict[str, Any], output_dir: Path, incomplete: bool) -> dict[str, Any]:
    mpl_style()
    rows, total = row_layout()
    width_mm = float(manifest["style_contract"]["figure_width_mm"])
    height_mm = float(manifest["style_contract"]["figure_height_mm"])
    gap_mm = float(manifest["style_contract"]["panel_gap_mm"])
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor=PALETTE["paper"])
    left = 0.215
    right = 0.988
    bottom = 0.055
    top = 0.965
    gap = gap_mm / height_mm
    panel_height = (top - bottom - gap) / 2.0
    axes_by_panel: dict[str, plt.Axes] = {}
    for panel, y in zip(panels, (bottom + panel_height + gap, bottom)):
        axes_by_panel[str(panel["panel"])] = fig.add_axes([left, y, right - left, panel_height])
    entries = entries_by_key(manifest)
    scales = compute_scales(panels, entries)
    for panel in panels:
        draw_panel(fig, axes_by_panel[str(panel["panel"])], panel, entries, scales, str(panel["panel"]), rows, total, label_x=0.205, group_x=0.046)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_incomplete_layout_preview" if incomplete else ""
    base = output_dir / f"{FORMAL_BASENAME}{suffix}"
    save_figure(fig, base)
    plt.close(fig)
    return figure_summary(
        panels, manifest, output_dir, base, incomplete,
        width_mm, height_mm, gap_mm, scales, "both",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the formal PDF-style HAR/HAQER genome-browser figure.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--panel", choices=("both", "g", "h"), default="both", help="Render both panels or one independent locus panel")
    parser.add_argument("--allow-incomplete", action="store_true", help="Render only as a non-publication layout preview when inputs are missing")
    args = parser.parse_args()
    root = args.root
    output_dir = args.output_dir or root / "results/final_har_haqer_panel/formal_main_figure"
    manifest_path = args.manifest or output_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    failures = strict_failures(manifest)
    if failures and not args.allow_incomplete:
        raise SystemExit("Formal renderer blocked by strict track manifest: " + ", ".join(failures))
    panels = collect_panels(root)
    incomplete = bool(failures)
    entries = entries_by_key(manifest)
    scales = compute_scales(panels, entries)
    if args.panel == "both":
        summary = render(panels, manifest, output_dir, incomplete=incomplete)
    else:
        selected = next(panel for panel in panels if str(panel["panel"]) == args.panel)
        selected_failures = failures_for_panels(manifest, [selected])
        summary = render_single(selected, manifest, output_dir, bool(selected_failures), scales)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
