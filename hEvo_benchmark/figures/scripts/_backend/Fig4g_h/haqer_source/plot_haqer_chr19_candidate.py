from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np

try:
    import pyBigWig
except ImportError:  # pragma: no cover
    pyBigWig = None


DEFAULT_ROOT = Path(__file__).resolve().parents[5]
ROOT = DEFAULT_ROOT
DEFAULT_OUTDIR = ROOT / "results/haqer_meaningful_region_chr19"
DEFAULT_TEACHER_OUTDIR = ROOT / "results/haqer_meaningful_region_chr19_teacher_style"
OUTDIR = DEFAULT_OUTDIR
TRACK_DIR = DEFAULT_OUTDIR
STYLE_MODE = "legacy"

SCAN_DIR = ROOT / "results/haqer_test_region_search_hardneg_86m/target050/val_region_29"
SCORE_NPZ = SCAN_DIR / "val_region_29_normal.smoothed_scores.npz"
PEAKS_TSV = SCAN_DIR / "val_region_29_normal.peaks.tsv"
OVERLAP_JSON = ROOT / "results/final_hardneg_top1_86m/HAQER_top1_chr19_0_1000000_normal_with_ucsc.overlap_summary.json"
HAQER_BED = ROOT / "results/literature_example_tracks_beautified/haqer.hg38.bed"
REFGENE_GZ = ROOT / "reference/hg38_refGene.txt.gz"
H3K27AC_BW = ROOT / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig"
PHASTCONS_BW = ROOT / "external_tracks/ucsc/hg38.phastCons100way.bw"
CCRE_BB = ROOT / "external_tracks/ucsc/encodeCcreCombined.hg38.bb"
CADD_POSITION_TSV = DEFAULT_OUTDIR / "HAQER_chr19_CADD_GRCh38_v1.7_peak_position_scores.tsv"
PHYLOP_POSITION_TSV = DEFAULT_OUTDIR / "HAQER_chr19_phyloP100way_peak_position_scores.tsv"
CADD_TRACK_TSV = DEFAULT_OUTDIR / "HAQER_chr19_CADD_GRCh38_v1.7_binned_track.tsv"
PHYLOP_JSON = DEFAULT_OUTDIR / "tmp_phyloP_chr19_345000_485000.json"
EVO2_TRACK_NPZ = ROOT / "results/final_har_haqer_panel/evo2_1b_hevo_protocol/scores/haqer/h_blocks_12_mlp_l3_tracks.npz"
EVO2_TRACK_FIELD = "evo2_1b_raw_likelihood"

CHROM = "chr19"
REGION_START = 345_000
REGION_END = 485_000

PINK = "#F06292"
PINK_DARK = "#C2185B"
BLUE = "#2F6FAB"
GREEN = "#2F7D4F"
GREY = "#D9E2EC"
TEXT = "#111111"
MUTED = "#56616F"
PURPLE = "#8B79B9"
ORANGE = "#E69F00"
PDF_NAVY = "#173F63"
PDF_ORANGE = "#C85A0A"
PDF_TEAL = "#008A93"
PDF_GOLD = "#B07C32"
PDF_GREEN = "#5A9367"
PDF_BLUE = "#4C79A8"
PDF_PURPLE = "#7667A8"
PDF_GREY = "#B8B4AE"
PDF_LIGHT_GREY = "#E7E4DE"
PDF_GRID = "#E3E4E1"
PDF_EDGE = "#AEB7BB"
CCRE_COLORS = {
    "PLS": PDF_ORANGE,
    "pELS": PDF_GOLD,
    "dELS": PDF_GOLD,
    "CTCF-only": PDF_TEAL,
    "DNase-H3K4me3": PDF_BLUE,
    "cCRE": PDF_GREY,
}
FONT_SCALE = 1.35
MIN_INTERVAL_DISPLAY_BP = 1_800
MIN_CCRE_DISPLAY_BP = 1_400
NEAR_HAQER_DISTANCE_BP = 10_000
HIT_SPOTLIGHT_PADDING_BP = 8_000

STYLE = {
    "raw": PDF_GREY,
    "score": PDF_ORANGE,
    "known": PDF_NAVY,
    "peak_hit": PDF_ORANGE,
    "peak_near": PDF_BLUE,
    "peak_other": PDF_LIGHT_GREY,
    "gene": "#68737A",
    "gene_hi": PDF_NAVY,
    "h3k27ac": PDF_GOLD,
    "phastcons": PDF_GREEN,
    "phylop": PDF_PURPLE,
    "cadd": PDF_GOLD,
    "text": PDF_NAVY,
    "muted": "#68737A",
    "grid": PDF_GRID,
    "panel_bg": "#FFFFFF",
    "panel_edge": PDF_EDGE,
    "paper_bg": "#FFFFFF",
    "ccre_enh": PDF_GOLD,
    "ccre_pls": PDF_ORANGE,
    "ccre_ctcf": PDF_TEAL,
    "ccre_dnase": PDF_BLUE,
    "evo2": PDF_TEAL,
}


def fs(size: float) -> float:
    return size * FONT_SCALE


def genomic_tick_formatter(region_end: int) -> FuncFormatter:
    if region_end >= 1_000_000:
        return FuncFormatter(lambda value, _pos: f"{value / 1_000_000:.1f} Mb")
    return FuncFormatter(lambda value, _pos: f"{value / 1_000:.0f} kb")


def apply_genomic_xaxis(fig, region_end: int) -> None:
    formatter = genomic_tick_formatter(region_end)
    for axis in fig.axes:
        axis.xaxis.set_major_formatter(formatter)
        axis.xaxis.get_offset_text().set_visible(False)


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def distance(a0: int, a1: int, b0: int, b1: int) -> int:
    if overlaps(a0, a1, b0, b1):
        return 0
    if a1 <= b0:
        return b0 - a1
    return a0 - b1


def display_span(start: int, end: int, min_width_bp: int, region_start: int, region_end: int) -> tuple[int, int]:
    width = end - start
    if width >= min_width_bp:
        return start, end
    center = (start + end) // 2
    half = min_width_bp // 2
    display_start = max(region_start, center - half)
    display_end = min(region_end, display_start + min_width_bp)
    display_start = max(region_start, display_end - min_width_bp)
    return display_start, display_end


def merge_spans(spans: list[tuple[int, int]], region_start: int, region_end: int) -> list[tuple[int, int]]:
    clipped = [
        (max(region_start, start), min(region_end, end))
        for start, end in spans
        if start < region_end and end > region_start
    ]
    clipped = [(start, end) for start, end in clipped if start < end]
    if not clipped:
        return []
    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def inverse_spans(spans: list[tuple[int, int]], region_start: int, region_end: int) -> list[tuple[int, int]]:
    inverse: list[tuple[int, int]] = []
    cursor = region_start
    for start, end in spans:
        if cursor < start:
            inverse.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < region_end:
        inverse.append((cursor, region_end))
    return inverse


def load_scores() -> dict[str, np.ndarray]:
    data = np.load(SCORE_NPZ)
    mask = (
        (data["chromosome"] == CHROM) & data["valid_mask"] & (data["start"] < REGION_END) & (data["end"] > REGION_START)
    )
    return {
        "x": data["start"][mask] + (data["end"][mask] - data["start"][mask]) / 2,
        "raw": data["raw_scores"][mask],
        "smooth": data["smoothed_scores"][mask],
    }


def load_peaks() -> list[dict[str, int | float | str]]:
    peaks: list[dict[str, int | float | str]] = []
    with PEAKS_TSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            start = int(row["peak_start"])
            end = int(row["peak_end"])
            if row["chromosome"] == CHROM and overlaps(start, end, REGION_START, REGION_END):
                peaks.append(
                    {
                        "start": start,
                        "end": end,
                        "summit": int(row["summit_center"]),
                        "score": float(row["smoothed_score"]),
                        "prominence": float(row["prominence"]),
                    }
                )
    return peaks


def load_haqers() -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with HAQER_BED.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, name, score, *_ = line.rstrip().split("\t")
            start_i = int(start)
            end_i = int(end)
            if chrom == CHROM and overlaps(start_i, end_i, REGION_START, REGION_END):
                rows.append({"start": start_i, "end": end_i, "name": name, "score": score})
    rows.sort(key=lambda item: int(item["start"]))
    return rows


def load_genes() -> list[dict[str, int | str]]:
    by_name: dict[str, dict[str, int | str]] = {}
    with gzip.open(REFGENE_GZ, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip().split("\t")
            if len(parts) < 13 or parts[2] != CHROM:
                continue
            tx_start = int(parts[4])
            tx_end = int(parts[5])
            if not overlaps(tx_start, tx_end, REGION_START, REGION_END):
                continue
            name = parts[12] or parts[1]
            current = by_name.get(name)
            if current is None:
                by_name[name] = {"name": name, "start": tx_start, "end": tx_end, "strand": parts[3]}
            else:
                current["start"] = min(int(current["start"]), tx_start)
                current["end"] = max(int(current["end"]), tx_end)
    genes = list(by_name.values())
    genes.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return genes


def load_bigwig(path: Path, bins: int) -> np.ndarray:
    if pyBigWig is None:
        raise RuntimeError("pyBigWig is required for bigWig tracks.")
    with pyBigWig.open(str(path)) as handle:
        values = handle.stats(CHROM, REGION_START, REGION_END, nBins=bins, exact=True)
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float32)


def infer_ccre_class(fields: list[str]) -> str:
    joined = " ".join(fields)
    for label in ["CTCF-only", "DNase-H3K4me3", "pELS", "dELS", "PLS"]:
        if label in joined:
            return label
    return "cCRE"


def load_ccres() -> list[dict[str, int | str]]:
    if pyBigWig is None:
        raise RuntimeError("pyBigWig is required for bigBed tracks.")
    rows: list[dict[str, int | str]] = []
    with pyBigWig.open(str(CCRE_BB)) as handle:
        for start, end, rest in handle.entries(CHROM, REGION_START, REGION_END) or []:
            fields = rest.split("\t") if rest else []
            ccre_class = infer_ccre_class(fields)
            rows.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "class": ccre_class,
                    "color": CCRE_COLORS.get(ccre_class, CCRE_COLORS["cCRE"]),
                }
            )
    rows.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return rows


def load_cadd_position_scores() -> list[dict[str, float | int | str]]:
    if not CADD_POSITION_TSV.exists():
        return []
    rows: list[dict[str, float | int | str]] = []
    with CADD_POSITION_TSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("max_phred"):
                rows.append(
                    {
                        "source": row["source"],
                        "index": int(row["index"]),
                        "position": int(row["position"]),
                        "max_phred": float(row["max_phred"]),
                    }
                )
    return rows


def load_phylop_position_scores() -> list[dict[str, float | int | str]]:
    if not PHYLOP_POSITION_TSV.exists():
        return []
    rows: list[dict[str, float | int | str]] = []
    with PHYLOP_POSITION_TSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("phyloP"):
                rows.append(
                    {
                        "source": row["source"],
                        "index": int(row["index"]),
                        "position": int(row["position"]),
                        "phyloP": float(row["phyloP"]),
                    }
                )
    return rows


def load_cadd_binned_track() -> list[dict[str, float | int | str]]:
    if not CADD_TRACK_TSV.exists():
        return []
    rows: list[dict[str, float | int | str]] = []
    with CADD_TRACK_TSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("max_phred"):
                rows.append(
                    {
                        "position": int(row["position"]),
                        "bin_start": int(row["bin_start"]),
                        "bin_end": int(row["bin_end"]),
                        "max_phred": float(row["max_phred"]),
                    }
                )
    return rows


def load_phylop_json_track(bins: int) -> dict[str, np.ndarray]:
    if not PHYLOP_JSON.exists():
        return {"x": np.asarray([], dtype=np.float64), "values": np.asarray([], dtype=np.float32)}
    payload = json.loads(PHYLOP_JSON.read_text(encoding="utf-8"))
    sums = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    span = REGION_END - REGION_START
    for item in payload.get("phyloP100way", []):
        start = int(item["start"])
        if start < REGION_START or start >= REGION_END:
            continue
        idx = min(bins - 1, max(0, int((start - REGION_START) * bins / span)))
        sums[idx] += float(item["value"])
        counts[idx] += 1
    values = np.full(bins, np.nan, dtype=np.float32)
    mask = counts > 0
    values[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    x = np.linspace(REGION_START, REGION_END, bins, endpoint=False) + span / bins / 2
    return {"x": x, "values": values}


def load_evo2_track(path: Path, field: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Evo2 track is required for teacher-style rendering: {path}")
    payload = np.load(path, allow_pickle=False)
    x = np.asarray(payload["centers"], dtype=np.int64)
    values = np.asarray(payload[field], dtype=np.float32)
    if x.shape != values.shape:
        raise ValueError(f"Evo2 track coordinate/value shape mismatch: {x.shape} vs {values.shape}")
    mask = (x >= REGION_START) & (x <= REGION_END) & np.isfinite(values)
    if not np.any(mask):
        raise ValueError(f"Evo2 track has no finite values in {CHROM}:{REGION_START}-{REGION_END}")
    return {"x": x[mask], "values": values[mask], "path": str(path), "field": field}


def summarize(
    peaks: list[dict[str, int | float | str]], haqers: list[dict[str, int | str]]
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for peak in peaks:
        best = None
        for haqer in haqers:
            ov = max(0, min(int(peak["end"]), int(haqer["end"])) - max(int(peak["start"]), int(haqer["start"])))
            dist = distance(int(peak["start"]), int(peak["end"]), int(haqer["start"]), int(haqer["end"]))
            candidate = (dist, -ov, haqer)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            continue
        rows.append(
            {
                "chrom": CHROM,
                "peak_start": int(peak["start"]),
                "peak_end": int(peak["end"]),
                "summit": int(peak["summit"]),
                "smoothed_score": float(peak["score"]),
                "nearest_haqer": str(best[2]["name"]),
                "distance_bp": int(best[0]),
                "overlap_bp": int(-best[1]),
            }
        )
    return rows


def assign_levels(items: list[dict[str, int | str]], max_levels: int = 4) -> list[int]:
    level_ends = [-(10**12)] * max_levels
    levels: list[int] = []
    for item in items:
        start = int(item["start"])
        level = 0
        for idx, current_end in enumerate(level_ends):
            if start > current_end + 1500:
                level = idx
                break
        level_ends[level] = int(item["end"])
        levels.append(level)
    return levels


def setup_track_axis(ax, label: str, show_x: bool = False) -> None:
    ax.set_facecolor(STYLE["panel_bg"])
    ax.set_xlim(REGION_START, REGION_END)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(STYLE["panel_edge"])
    ax.spines["bottom"].set_color(STYLE["panel_edge"])
    ax.grid(axis="y", color=STYLE["grid"], linewidth=0.5, alpha=0.35)
    ax.tick_params(axis="both", labelsize=fs(8), colors=STYLE["text"])
    ax.set_ylabel(label, fontsize=fs(9), color=STYLE["text"])
    if not show_x:
        ax.tick_params(axis="x", labelbottom=False)


def draw_interval_track(ax, rows: list[dict[str, int | str]], label: str, color: str, height: float = 0.36) -> None:
    for row_index, row in enumerate(rows):
        x0 = int(row["start"]) / 1000
        width = (int(row["end"]) - int(row["start"])) / 1000
        ax.add_patch(patches.Rectangle((x0, 0.5 - height / 2), width, height, facecolor=color, edgecolor=color, lw=0.8))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    setup_track_axis(ax, label)


def draw_ccre_group(ax, ccres: list[dict[str, int | str]], classes: set[str], label: str, color: str) -> None:
    selected = [item for item in ccres if str(item["class"]) in classes]
    for item in selected:
        start = int(item["start"])
        end = int(item["end"])
        display_start, display_end = display_span(start, end, MIN_CCRE_DISPLAY_BP, REGION_START, REGION_END)
        ax.add_patch(
            patches.Rectangle(
                (display_start, 0.20),
                display_end - display_start,
                0.60,
                facecolor=color,
                edgecolor="none",
                alpha=0.84,
            )
        )
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    setup_track_axis(ax, "")
    ax.text(
        -0.012,
        0.5,
        label,
        transform=ax.get_yaxis_transform(),
        color="black",
        fontsize=fs(8),
        ha="right",
        va="center",
        clip_on=False,
    )


def draw_ccres(ax, ccres: list[dict[str, int | str]]) -> None:
    placed: list[dict[str, int]] = []
    stacked: list[dict[str, int | str]] = []
    for ccre in ccres:
        level = 0
        while any(
            item["level"] == level
            and overlaps(int(ccre["start"]), int(ccre["end"]), item["start"], item["end"])
            for item in placed
        ):
            level += 1
        placed.append({"start": int(ccre["start"]), "end": int(ccre["end"]), "level": level})
        stacked.append({**ccre, "level": level})
    max_level = max([int(item["level"]) for item in stacked], default=0)
    for ccre in stacked:
        y = max_level - int(ccre["level"])
        start = int(ccre["start"])
        end = int(ccre["end"])
        display_start, display_end = display_span(start, end, MIN_CCRE_DISPLAY_BP, REGION_START, REGION_END)
        ax.add_patch(
            patches.Rectangle(
                (display_start, y - 0.15),
                display_end - display_start,
                0.30,
                facecolor=str(ccre["color"]),
                edgecolor="none",
                alpha=1.00,
            )
        )
    ax.set_ylim(-0.5, max_level + 0.5)
    ax.set_yticks([])
    setup_track_axis(ax, "cCRE")


def draw_cadd_track(ax, rows: list[dict[str, float | int | str]]) -> None:
    if rows:
        x = np.asarray([int(r["position"]) for r in rows], dtype=np.float64)
        y = np.asarray([float(r["max_phred"]) for r in rows], dtype=np.float32)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        ax.plot(x, y, color=STYLE["cadd"], lw=0.78, alpha=0.96)
        ax.fill_between(x, y, 0, color=STYLE["cadd"], alpha=0.24, linewidth=0)
    ax.axhline(10, color=STYLE["grid"], lw=0.65, ls="--", alpha=0.75)
    ax.axhline(20, color=STYLE["muted"], lw=0.65, ls="--", alpha=0.65)
    max_score = max([float(r["max_phred"]) for r in rows], default=25.0)
    ax.set_ylim(0, max(25, max_score + 2))
    setup_track_axis(ax, "CADD")


def draw_phylop_track(ax, track: dict[str, np.ndarray]) -> None:
    x = np.asarray(track.get("x", []), dtype=np.float64)
    y = np.asarray(track.get("values", []), dtype=np.float32)
    valid = np.isfinite(y)
    if np.any(valid):
        ax.plot(x[valid], y[valid], color=STYLE["phylop"], lw=0.68, alpha=0.90)
        ax.fill_between(x[valid], y[valid], 0, color=STYLE["phylop"], alpha=0.20, linewidth=0)
    values = y[valid].tolist()
    ax.axhline(0, color="#BDBDBD", lw=0.65, alpha=0.8)
    ax.set_ylim(min(-2, min(values, default=-2) - 0.5), max(6, max(values, default=6) + 0.5))
    setup_track_axis(ax, "PhyloP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the HAQER chr19 candidate locus tracks.")
    parser.add_argument("--style", choices=["legacy", "teacher"], default="legacy")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--track-dir", type=Path, default=None)
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--peaks-tsv", type=Path, default=None)
    parser.add_argument("--overlap-json", type=Path, default=None)
    parser.add_argument("--haqer-bed", type=Path, default=None)
    parser.add_argument("--refgene-gz", type=Path, default=None)
    parser.add_argument("--h3k27ac-bw", type=Path, default=None)
    parser.add_argument("--phastcons-bw", type=Path, default=None)
    parser.add_argument("--ccre-bb", type=Path, default=None)
    parser.add_argument("--cadd-position-tsv", type=Path, default=None)
    parser.add_argument("--phylop-position-tsv", type=Path, default=None)
    parser.add_argument("--cadd-track-tsv", type=Path, default=None)
    parser.add_argument("--phylop-json", type=Path, default=None)
    parser.add_argument("--evo2-track-npz", type=Path, default=None)
    parser.add_argument("--evo2-track-field", default=EVO2_TRACK_FIELD)
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global ROOT, OUTDIR, TRACK_DIR, STYLE_MODE
    global SCORE_NPZ, PEAKS_TSV, OVERLAP_JSON, HAQER_BED, REFGENE_GZ
    global H3K27AC_BW, PHASTCONS_BW, CCRE_BB
    global CADD_POSITION_TSV, PHYLOP_POSITION_TSV, CADD_TRACK_TSV, PHYLOP_JSON
    global EVO2_TRACK_NPZ, EVO2_TRACK_FIELD

    ROOT = args.root
    OUTDIR = args.output_dir or (
        DEFAULT_OUTDIR if args.style == "legacy" else ROOT / "results/haqer_meaningful_region_chr19_teacher_style"
    )
    TRACK_DIR = args.track_dir or ROOT / "results/haqer_meaningful_region_chr19"
    STYLE_MODE = args.style
    scan_dir = ROOT / "results/haqer_test_region_search_hardneg_86m/target050/val_region_29"
    SCORE_NPZ = args.input_npz or scan_dir / "val_region_29_normal.smoothed_scores.npz"
    PEAKS_TSV = args.peaks_tsv or scan_dir / "val_region_29_normal.peaks.tsv"
    OVERLAP_JSON = args.overlap_json or ROOT / "results/final_hardneg_top1_86m/HAQER_top1_chr19_0_1000000_normal_with_ucsc.overlap_summary.json"
    HAQER_BED = args.haqer_bed or ROOT / "results/literature_example_tracks_beautified/haqer.hg38.bed"
    REFGENE_GZ = args.refgene_gz or ROOT / "reference/hg38_refGene.txt.gz"
    H3K27AC_BW = args.h3k27ac_bw or ROOT / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig"
    PHASTCONS_BW = args.phastcons_bw or ROOT / "external_tracks/ucsc/hg38.phastCons100way.bw"
    CCRE_BB = args.ccre_bb or ROOT / "external_tracks/ucsc/encodeCcreCombined.hg38.bb"
    CADD_POSITION_TSV = args.cadd_position_tsv or TRACK_DIR / "HAQER_chr19_CADD_GRCh38_v1.7_peak_position_scores.tsv"
    PHYLOP_POSITION_TSV = args.phylop_position_tsv or TRACK_DIR / "HAQER_chr19_phyloP100way_peak_position_scores.tsv"
    CADD_TRACK_TSV = args.cadd_track_tsv or TRACK_DIR / "HAQER_chr19_CADD_GRCh38_v1.7_binned_track.tsv"
    PHYLOP_JSON = args.phylop_json or TRACK_DIR / "tmp_phyloP_chr19_345000_485000.json"
    EVO2_TRACK_NPZ = args.evo2_track_npz or ROOT / "results/final_har_haqer_panel/evo2_1b_hevo_protocol/scores/haqer/h_blocks_12_mlp_l3_tracks.npz"
    EVO2_TRACK_FIELD = args.evo2_track_field


def teacher_axis(ax, label: str, show_x: bool = False) -> None:
    setup_track_axis(ax, label, show_x=show_x)
    ax.set_ylabel(
        label,
        fontsize=fs(8.5),
        color=STYLE["text"],
        labelpad=34,
        rotation=0,
        ha="right",
        va="center",
    )
    if not show_x:
        ax.tick_params(axis="x", labelbottom=False)


def teacher_group_label(ax, label: str) -> None:
    ax.text(
        -0.19,
        0.5,
        label,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=fs(8.5),
        fontweight="bold",
        color=STYLE["text"],
    )


def draw_evo2_track(ax, track: dict[str, object]) -> None:
    x = np.asarray(track["x"], dtype=np.int64)
    y = np.asarray(track["values"], dtype=np.float32)
    finite = np.isfinite(y)
    if not np.any(finite):
        raise ValueError("Evo2 track contains no finite values")
    x = x[finite]
    y = y[finite]
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    span = max(ymax - ymin, 1e-6)
    ax.fill_between(x, ymin, y, color=STYLE["evo2"], alpha=0.14, linewidth=0)
    ax.plot(x, y, color=STYLE["evo2"], lw=0.78, alpha=0.92)
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.16 * span)
    ax.text(
        0.995, 0.82, "experimental non-FP8",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=fs(6.5), color=STYLE["muted"],
    )
    teacher_axis(ax, "EVO 2 likelihood score")


def draw_teacher_intervals(ax, rows, start_key: str, end_key: str, label: str, color_fn, text_fn=None) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for row_index, row in enumerate(rows):
        start = int(row[start_key])
        end = int(row[end_key])
        display_start, display_end = display_span(start, end, MIN_INTERVAL_DISPLAY_BP, REGION_START, REGION_END)
        ax.add_patch(
            patches.Rectangle(
                (display_start, 0.24),
                display_end - display_start,
                0.52,
                facecolor=color_fn(row),
                edgecolor="none",
                alpha=0.86,
            )
        )
        if text_fn is not None:
            text = text_fn(row)
            if text:
                ax.text(
                    display_start,
                    0.79 + 0.12 * (row_index % 2),
                    text,
                    fontsize=fs(6.6),
                    color=STYLE["text"],
                    ha="center",
                    va="bottom",
                    rotation=0,
                    clip_on=False,
                )
    teacher_axis(ax, label)


def categorize_peaks(peaks: list[dict[str, int | float | str]], haqers: list[dict[str, int | str]]) -> list[str]:
    categories: list[str] = []
    for peak in peaks:
        nearest = min(
            distance(int(peak["start"]), int(peak["end"]), int(haqer["start"]), int(haqer["end"]))
            for haqer in haqers
        )
        if nearest == 0:
            categories.append("overlaps_gold")
        elif nearest <= NEAR_HAQER_DISTANCE_BP:
            categories.append("near_known_har")
        else:
            categories.append("other_non_gold")
    return categories


def render_teacher_figure(
    scores: dict[str, np.ndarray],
    peaks: list[dict[str, int | float | str]],
    haqers: list[dict[str, int | str]],
    genes: list[dict[str, int | str]],
    ccres: list[dict[str, int | str]],
    h3k27ac: np.ndarray,
    phastcons: np.ndarray,
    cadd_track: list[dict[str, float | int | str]],
    phylop_track: dict[str, np.ndarray],
    evo2_track: dict[str, object],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    categories = categorize_peaks(peaks, haqers)
    peak_rows = [{**peak, "category": category} for peak, category in zip(peaks, categories)]
    fig, axes = plt.subplots(
        10,
        1,
        figsize=(12.5, 10.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 2.2, 0.85, 0.85, 1.0, 0.85, 1.0, 1.0, 1.1, 1.65]},
    )
    score_ax, evo2_ax, peak_ax, known_ax, h3_ax, ccre_ax, cons_ax, phylop_ax, cadd_ax, gene_ax = axes
    fig.patch.set_facecolor(STYLE["paper_bg"])
    score_ax.text(
        -0.16,
        1.08,
        "h",
        transform=score_ax.transAxes,
        fontsize=fs(15),
        fontweight="bold",
        color=STYLE["text"],
        ha="left",
        va="bottom",
        clip_on=False,
    )

    hit_spans: list[tuple[int, int]] = []
    score_ax.plot(scores["x"], scores["raw"], color=STYLE["raw"], lw=0.35, alpha=0.24, label="raw score")
    score_ax.plot(
        scores["x"],
        scores["smooth"],
        color=STYLE["score"],
        lw=1.25,
        alpha=0.92,
        label="smoothed HAQER-head score",
    )
    for peak, category in zip(peaks, categories):
        color = {
            "overlaps_gold": STYLE["peak_hit"],
            "near_known_har": STYLE["peak_near"],
            "other_non_gold": STYLE["peak_other"],
        }[category]
        if category == "overlaps_gold":
            hit_spans.append((int(peak["start"]), int(peak["end"])))
        score_ax.axvspan(
            int(peak["start"]),
            int(peak["end"]),
            color=color,
            alpha=0.15 if category != "other_non_gold" else 0.10,
            linewidth=0,
        )
        score_ax.axvline(int(peak["summit"]), color=color, lw=0.65, alpha=0.78)
        score_ax.scatter(
            [int(peak["summit"])],
            [float(peak["score"])],
            s=13 if category != "other_non_gold" else 7,
            color=color,
            zorder=5,
        )
    spotlight_spans = merge_spans(
        [(start - HIT_SPOTLIGHT_PADDING_BP, end + HIT_SPOTLIGHT_PADDING_BP) for start, end in hit_spans],
        REGION_START,
        REGION_END,
    )
    for start, end in spotlight_spans:
        mask = (scores["x"] >= start) & (scores["x"] <= end)
        score_ax.plot(scores["x"][mask], scores["smooth"][mask], color=STYLE["score"], lw=1.95, alpha=1.0)
    score_ax.set_title("HAQER | chr19p13.3 | hg38", fontsize=fs(14), fontweight="bold", color=STYLE["text"], pad=10)
    direct_hits = sum(category == "overlaps_gold" for category in categories)
    score_ax.text(
        0.995,
        0.93,
        f"direct HAQER recovery: {direct_hits}/{len(haqers)}   |   called peaks: {len(peaks)}",
        transform=score_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fs(7.2),
        color=STYLE["text"],
    )
    score_ax.legend(loc="upper left", frameon=False, fontsize=fs(7.5), ncol=2)
    teacher_axis(score_ax, "hEvo-base score")
    teacher_group_label(score_ax, "MODEL")
    draw_evo2_track(evo2_ax, evo2_track)

    draw_teacher_intervals(
        peak_ax,
        peak_rows,
        "start",
        "end",
        "Peak calls",
        lambda row: {
            "overlaps_gold": STYLE["peak_hit"],
            "near_known_har": STYLE["peak_near"],
            "other_non_gold": STYLE["peak_other"],
        }[str(row["category"])],
        lambda row: "",
    )
    draw_teacher_intervals(
        known_ax,
        haqers,
        "start",
        "end",
        "Known HAQERs",
        lambda row: STYLE["known"],
        lambda row: str(row.get("name", "")),
    )

    h3_ax.plot(scores["x"], h3k27ac, color=STYLE["h3k27ac"], lw=0.68, alpha=0.82)
    teacher_axis(h3_ax, "H3K27ac")
    teacher_group_label(h3_ax, "REG.")
    draw_ccres(ccre_ax, ccres)
    teacher_axis(ccre_ax, "cCRE")

    cons_ax.plot(scores["x"], phastcons, color=STYLE["phastcons"], lw=0.58, alpha=0.76)
    teacher_axis(cons_ax, "phastCons")
    teacher_group_label(cons_ax, "EVOL.")
    draw_phylop_track(phylop_ax, phylop_track)
    teacher_axis(phylop_ax, "phyloP")

    draw_cadd_track(cadd_ax, cadd_track)
    teacher_axis(cadd_ax, "CADD")
    teacher_group_label(cadd_ax, "REF.\nSCORES")

    gene_levels = assign_levels(genes, max_levels=4)
    max_level = max(gene_levels, default=0)
    for gene, level in zip(genes, gene_levels):
        y = max_level - level
        start = max(int(gene["start"]), REGION_START)
        end = min(int(gene["end"]), REGION_END)
        symbol = str(gene["name"])
        highlight = symbol in {"SHC2", "ODF3L2"}
        color = STYLE["gene_hi"] if highlight else STYLE["gene"]
        gene_ax.plot([start, end], [y, y], color=color, lw=1.8 if highlight else 0.9, alpha=0.95 if highlight else 0.50)
        if highlight:
            gene_ax.text(start, y + 0.22, symbol, color=color, fontsize=fs(10), fontweight="bold", va="bottom")
    gene_ax.set_ylim(-0.6, max_level + 0.8)
    gene_ax.set_yticks([])
    teacher_axis(gene_ax, "RefSeq genes", show_x=True)
    teacher_group_label(gene_ax, "GENOME")
    gene_ax.set_xlabel(f"{CHROM} position (kb, hg38)", fontsize=fs(9.5))

    apply_genomic_xaxis(fig, REGION_END)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.92, bottom=0.08, hspace=0.08)
    output_base = OUTDIR / "HAQER_chr19_345k_485k_teacher_style_with_evo2"
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot() -> None:
    if STYLE_MODE == "teacher":
        return plot_teacher()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    peaks = load_peaks()
    haqers = load_haqers()
    genes = load_genes()
    ccres = load_ccres()
    h3k27ac = load_bigwig(H3K27AC_BW, len(scores["x"]))
    phastcons = load_bigwig(PHASTCONS_BW, len(scores["x"]))
    cadd_rows = load_cadd_position_scores()
    phylop_rows = load_phylop_position_scores()
    cadd_track = load_cadd_binned_track()
    phylop_track = load_phylop_json_track(len(scores["x"]))
    summary_rows = summarize(peaks, haqers)

    with (OUTDIR / "HAQER_chr19_345k_485k_candidate_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    overlap = json.loads(OVERLAP_JSON.read_text(encoding="utf-8"))
    metadata = {
        "region": f"{CHROM}:{REGION_START}-{REGION_END}",
        "official_haqer_count": len(haqers),
        "predicted_peak_count": len(peaks),
        "predicted_peaks_directly_overlapping_haqer": sum(row["distance_bp"] == 0 for row in summary_rows),
        "predicted_peaks_within_10kb_of_haqer": sum(row["distance_bp"] <= 10_000 for row in summary_rows),
        "parent_1mb_summary": {
            "num_peaks": overlap["num_peaks"],
            "num_gold_intervals": overlap["num_gold_intervals"],
            "num_peaks_overlapping_gold": overlap["num_peaks_overlapping_gold"],
            "num_gold_intervals_hit": overlap["num_gold_intervals_hit"],
        },
        "official_haqers": haqers,
        "tracks": {
            "H3K27ac": str(H3K27AC_BW),
            "phastCons100way": str(PHASTCONS_BW),
            "CADD_position_scores": str(CADD_POSITION_TSV),
            "CADD_binned_track": str(CADD_TRACK_TSV),
            "phyloP100way_position_scores": str(PHYLOP_POSITION_TSV),
            "phyloP100way_json": str(PHYLOP_JSON),
            "cCRE": str(CCRE_BB),
            "cCRE_count": len(ccres),
            "CADD_point_count": len(cadd_rows),
            "phyloP_point_count": len(phylop_rows),
            "CADD_track_bin_count": len(cadd_track),
            "phyloP_track_bin_count": int(np.isfinite(phylop_track["values"]).sum()) if len(phylop_track["values"]) else 0,
        },
    }
    (OUTDIR / "HAQER_chr19_345k_485k_candidate_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(
        8,
        1,
        figsize=(10, 12.9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 0.9, 0.9, 0.85, 0.85, 1.1, 1.0, 1.75]},
    )
    (
        ax_score,
        ax_h3,
        ax_cons,
        ax_cadd,
        ax_phylop,
        ax_ccre,
        ax_gene,
        ax_interval,
    ) = axes
    fig.patch.set_facecolor(STYLE["paper_bg"])

    peak_categories = []
    hit_spans: list[tuple[int, int]] = []
    for peak in peaks:
        nearest = min(
            distance(int(peak["start"]), int(peak["end"]), int(haqer["start"]), int(haqer["end"])) for haqer in haqers
        )
        if nearest == 0:
            category = "overlaps_gold"
        elif nearest <= NEAR_HAQER_DISTANCE_BP:
            category = "near_known_har"
        else:
            category = "other_non_gold"
        peak_categories.append(category)
        if category == "overlaps_gold":
            hit_spans.append((int(peak["start"]), int(peak["end"])))

    ax_score.plot(scores["x"], scores["raw"], color=STYLE["raw"], lw=0.35, alpha=0.20, label="raw score", zorder=1)
    ax_score.plot(
        scores["x"], scores["smooth"], color=STYLE["score"], lw=1.08, alpha=0.66, label="smoothed score", zorder=2
    )
    for peak, category in zip(peaks, peak_categories):
        if category == "overlaps_gold":
            color = STYLE["peak_hit"]
            alpha = 0.18
            summit_alpha = 0.78
        elif category == "near_known_har":
            color = STYLE["peak_near"]
            alpha = 0.20
            summit_alpha = 0.85
        else:
            color = STYLE["peak_other"]
            alpha = 0.16
            summit_alpha = 0.40
        ax_score.axvspan(int(peak["start"]), int(peak["end"]), color=color, alpha=alpha, linewidth=0, zorder=2.2)
        ax_score.axvline(int(peak["summit"]), color=color, lw=0.75, alpha=summit_alpha, zorder=4.5)
        ax_score.scatter(
            [int(peak["summit"])],
            [float(peak["score"])],
            s=14 if category != "other_non_gold" else 8,
            color=color,
            zorder=5,
        )
    spotlight_spans = merge_spans(
        [(start - HIT_SPOTLIGHT_PADDING_BP, end + HIT_SPOTLIGHT_PADDING_BP) for start, end in hit_spans],
        REGION_START,
        REGION_END,
    )
    for start, end in spotlight_spans:
        mask = (scores["x"] >= start) & (scores["x"] <= end)
        ax_score.plot(scores["x"][mask], scores["smooth"][mask], color=STYLE["score"], lw=1.95, alpha=1.0, zorder=4)
    ax_score.set_ylabel("Score", fontsize=fs(10), color=STYLE["text"])
    ax_score.set_title(
        "Predicted HAQER-associated regulatory peaks on chr19p13.3",
        fontsize=fs(14),
        fontweight="bold",
        color=STYLE["text"],
        pad=10,
    )
    ax_score.legend(loc="upper right", frameon=False, fontsize=fs(8))
    ax_score.text(
        0.012,
        0.92,
        "Gold hit: 3/3\nCalled peaks: 13\nNear-HAQER non-gold peaks: 3",
        transform=ax_score.transAxes,
        fontsize=fs(9),
        va="top",
        ha="left",
        color=STYLE["text"],
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": STYLE["paper_bg"],
            "edgecolor": STYLE["panel_edge"],
            "alpha": 0.94,
        },
    )
    setup_track_axis(ax_score, "Score")

    ax_h3.plot(scores["x"], h3k27ac, color=STYLE["h3k27ac"], lw=0.62, alpha=0.78)
    setup_track_axis(ax_h3, "H3K27ac")
    ax_cons.plot(scores["x"], phastcons, color=STYLE["phastcons"], lw=0.55, alpha=0.72)
    setup_track_axis(ax_cons, "phastCons")
    draw_cadd_track(ax_cadd, cadd_track)
    draw_phylop_track(ax_phylop, phylop_track)
    draw_ccres(ax_ccre, ccres)

    gene_levels = assign_levels(genes, max_levels=4)
    max_level = max(gene_levels, default=0)
    for gene, level in zip(genes, gene_levels):
        y = max_level - level
        start = max(int(gene["start"]), REGION_START)
        end = min(int(gene["end"]), REGION_END)
        symbol = str(gene["name"])
        highlight = symbol in {"SHC2", "ODF3L2"}
        color = STYLE["gene_hi"] if highlight else STYLE["gene"]
        ax_gene.plot([start, end], [y, y], color=color, lw=1.8 if highlight else 0.9, alpha=0.95 if highlight else 0.50)
        if highlight:
            ax_gene.text(start, y + 0.22, symbol, color=color, fontsize=fs(10), fontweight="bold", va="bottom")
    ax_gene.set_ylim(-0.6, max_level + 0.8)
    ax_gene.set_yticks([])
    setup_track_axis(ax_gene, "")

    ax_interval.set_ylim(0, 1.05)
    ax_interval.set_yticks([])
    ax_interval.set_xlabel(f"{CHROM} position", fontsize=fs(10))
    ax_interval.set_ylabel("")
    ax_interval.set_xlim(REGION_START, REGION_END)
    for haqer in haqers:
        display_start, display_end = display_span(
            int(haqer["start"]), int(haqer["end"]), MIN_INTERVAL_DISPLAY_BP, REGION_START, REGION_END
        )
        ax_interval.add_patch(
            patches.Rectangle(
                (display_start, 0.60),
                display_end - display_start,
                0.30,
                facecolor=STYLE["known"],
                edgecolor="none",
                alpha=0.86,
            )
        )
    for peak, category in zip(peaks, peak_categories):
        color = {
            "overlaps_gold": STYLE["peak_hit"],
            "near_known_har": STYLE["peak_near"],
            "other_non_gold": STYLE["peak_other"],
        }[category]
        display_start, display_end = display_span(
            int(peak["start"]), int(peak["end"]), MIN_INTERVAL_DISPLAY_BP, REGION_START, REGION_END
        )
        ax_interval.add_patch(
            patches.Rectangle(
                (display_start, 0.14),
                display_end - display_start,
                0.32,
                facecolor=color,
                edgecolor="none",
                alpha=0.82 if category != "other_non_gold" else 0.74,
            )
        )
    ax_interval.text(
        -0.012,
        0.74,
        "Known peaks",
        transform=ax_interval.get_yaxis_transform(),
        color="black",
        fontsize=fs(9),
        ha="right",
        va="center",
        clip_on=False,
    )
    ax_interval.text(
        -0.012,
        0.29,
        "Predicted peaks",
        transform=ax_interval.get_yaxis_transform(),
        color="black",
        fontsize=fs(9),
        ha="right",
        va="center",
        clip_on=False,
    )
    setup_track_axis(ax_interval, "", show_x=True)

    apply_genomic_xaxis(fig, REGION_END)
    fig.tight_layout()
    fig.savefig(OUTDIR / "HAQER_chr19_345k_485k_candidate_full_tracks.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "HAQER_chr19_345k_485k_candidate_full_tracks.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_teacher() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    peaks = load_peaks()
    haqers = load_haqers()
    genes = load_genes()
    ccres = load_ccres()
    h3k27ac = load_bigwig(H3K27AC_BW, len(scores["x"]))
    phastcons = load_bigwig(PHASTCONS_BW, len(scores["x"]))
    cadd_rows = load_cadd_position_scores()
    phylop_rows = load_phylop_position_scores()
    cadd_track = load_cadd_binned_track()
    phylop_track = load_phylop_json_track(len(scores["x"]))
    evo2_track = load_evo2_track(EVO2_TRACK_NPZ, EVO2_TRACK_FIELD)
    summary_rows = summarize(peaks, haqers)

    if summary_rows:
        with (OUTDIR / "HAQER_chr19_345k_485k_teacher_style_with_evo2_summary.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(summary_rows)

    overlap = json.loads(OVERLAP_JSON.read_text(encoding="utf-8"))
    categories = categorize_peaks(peaks, haqers)
    metadata = {
        "style": "teacher",
        "region": f"{CHROM}:{REGION_START}-{REGION_END}",
        "score_npz": str(SCORE_NPZ),
        "peaks_tsv": str(PEAKS_TSV),
        "overlap_json": str(OVERLAP_JSON),
        "official_haqer_count": len(haqers),
        "predicted_peak_count": len(peaks),
        "predicted_peaks_directly_overlapping_haqer": sum(category == "overlaps_gold" for category in categories),
        "predicted_peaks_within_10kb_of_haqer": sum(
            row["distance_bp"] <= 10_000 for row in summary_rows
        ),
        "parent_1mb_summary": {
            "num_peaks": overlap["num_peaks"],
            "num_gold_intervals": overlap["num_gold_intervals"],
            "num_peaks_overlapping_gold": overlap["num_peaks_overlapping_gold"],
            "num_gold_intervals_hit": overlap["num_gold_intervals_hit"],
        },
        "official_haqers": haqers,
        "tracks": {
            "H3K27ac": str(H3K27AC_BW),
            "phastCons100way": str(PHASTCONS_BW),
            "CADD_position_scores": str(CADD_POSITION_TSV),
            "CADD_binned_track": str(CADD_TRACK_TSV),
            "phyloP100way_position_scores": str(PHYLOP_POSITION_TSV),
            "phyloP100way_json": str(PHYLOP_JSON),
            "cCRE": str(CCRE_BB),
            "cCRE_count": len(ccres),
            "CADD_point_count": len(cadd_rows),
            "phyloP_point_count": len(phylop_rows),
            "CADD_track_bin_count": len(cadd_track),
            "phyloP_track_bin_count": int(np.isfinite(phylop_track["values"]).sum())
            if len(phylop_track["values"])
            else 0,
            "EVO 2 likelihood score (experimental non-FP8)": {
                "path": str(EVO2_TRACK_NPZ),
                "field": EVO2_TRACK_FIELD,
            },
        },
        "layout": ["MODEL", "REG.", "EVOL.", "REF. SCORES", "GENOME"],
        "output": str(OUTDIR / "HAQER_chr19_345k_485k_teacher_style_with_evo2"),
        "evo2_track_npz": str(EVO2_TRACK_NPZ),
        "evo2_track_field": EVO2_TRACK_FIELD,
    }
    (OUTDIR / "HAQER_chr19_345k_485k_teacher_style_with_evo2_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_teacher_figure(scores, peaks, haqers, genes, ccres, h3k27ac, phastcons, cadd_track, phylop_track, evo2_track)


if __name__ == "__main__":
    args = parse_args()
    configure_paths(args)
    plot()
