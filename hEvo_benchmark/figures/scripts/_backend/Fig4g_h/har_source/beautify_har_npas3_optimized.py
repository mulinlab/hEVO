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
SOURCE_DIR = ROOT / "results/literature_example_tracks"
DEFAULT_OUTDIR = ROOT / "results/literature_example_tracks_beautified"
DEFAULT_TEACHER_OUTDIR = ROOT / "results/literature_example_tracks_beautified_teacher_style"
OUTDIR = DEFAULT_OUTDIR
TRACK_DIR = DEFAULT_OUTDIR
STYLE_MODE = "legacy"

INPUT_NPZ = SOURCE_DIR / "HAR_NPAS3_optimized_with_tracks.smoothed_scores.npz"
PEAKS_TSV = SOURCE_DIR / "HAR_NPAS3_optimized_with_tracks.peaks.tsv"
OVERLAP_JSON = SOURCE_DIR / "HAR_NPAS3_optimized_with_tracks.overlap_summary.json"
GOLD_CSV = ROOT / "data/processed/01_HAR_positive_windows_512_full.csv"
RAW_HAR_CSV = ROOT / "data/raw/HARs.CSV"
REFGENE_GZ = ROOT / "reference/hg38_refGene.txt.gz"
H3K27AC_BW = ROOT / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig"
PHASTCONS_BW = ROOT / "external_tracks/ucsc/hg38.phastCons100way.bw"
CCRE_BB = ROOT / "external_tracks/ucsc/encodeCcreCombined.hg38.bb"
CADD_POSITION_TSV = TRACK_DIR / "HAR_NPAS3_CADD_GRCh38_v1.7_peak_position_scores.tsv"
PHYLOP_POSITION_TSV = TRACK_DIR / "HAR_NPAS3_phyloP100way_peak_position_scores.tsv"
CADD_TRACK_TSV = TRACK_DIR / "HAR_NPAS3_CADD_GRCh38_v1.7_binned_track.tsv"
PHYLOP_JSON = TRACK_DIR / "tmp_phyloP_chr14_33000000_34000000.json"
EVO2_TRACK_NPZ = ROOT / "results/final_har_haqer_panel/evo2_1b_hevo_protocol/scores/har/g_blocks_6_mlp_l3_tracks.npz"
EVO2_TRACK_FIELD = "evo2_1b_raw_likelihood"

NEAR_HAR_DISTANCE_BP = 6000
HIT_SPOTLIGHT_PADDING_BP = 8000
FONT_SCALE = 1.35
MIN_INTERVAL_DISPLAY_BP = 1800
MIN_CCRE_DISPLAY_BP = 1400
HIGHLIGHT_HARS = [
    {
        "start": 33611611,
        "end": 33612150,
        "label": "NPAS3-HAR",
        "note": "local HAR with dELS/CTCF cCRE annotation",
    },
    {
        "start": 33661114,
        "end": 33661912,
        "label": "HARsv2_0745",
        "note": "Neuron HARDatabase NPAS3 HAR",
    },
]

PALETTE = {
    "pdf_navy": "#173F63",
    "pdf_orange": "#C85A0A",
    "pdf_teal": "#008A93",
    "pdf_gold": "#B07C32",
    "pdf_green": "#5A9367",
    "pdf_blue": "#4C79A8",
    "pdf_purple": "#7667A8",
    "pdf_grey": "#B8B4AE",
    "pdf_light_grey": "#E7E4DE",
    "soft_charcoal": "#243746",
    "text_muted": "#68737A",
    "grid_grey": "#E3E4E1",
    "panel_edge": "#AEB7BB",
    "paper_bg": "#FFFFFF",
    "panel_bg": "#FFFFFF",
}

STYLE = {
    "raw": PALETTE["pdf_grey"],
    "score": PALETTE["pdf_orange"],
    "evo2": PALETTE["pdf_teal"],
    "gold": PALETTE["pdf_green"],
    "known_peak": PALETTE["pdf_navy"],
    "known_har": PALETTE["pdf_navy"],
    "known_har_hi": PALETTE["pdf_teal"],
    "peak_gold": PALETTE["pdf_orange"],
    "peak_near_har": PALETTE["pdf_blue"],
    "peak_other": PALETTE["pdf_light_grey"],
    "gene": PALETTE["text_muted"],
    "gene_hi": PALETTE["pdf_navy"],
    "h3k27ac": PALETTE["pdf_gold"],
    "phastcons": PALETTE["pdf_green"],
    "phylop": PALETTE["pdf_purple"],
    "cadd": PALETTE["pdf_gold"],
    "text": PALETTE["pdf_navy"],
    "muted": PALETTE["text_muted"],
    "grid": PALETTE["grid_grey"],
    "panel_bg": PALETTE["panel_bg"],
    "panel_edge": PALETTE["panel_edge"],
    "paper_bg": PALETTE["paper_bg"],
    "ccre_enh": PALETTE["pdf_gold"],
    "ccre_pls": PALETTE["pdf_orange"],
    "ccre_ctcf": PALETTE["pdf_teal"],
    "ccre_dnase": PALETTE["pdf_blue"],
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


def overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def distance_to_interval(start: int, end: int, target_start: int, target_end: int) -> int:
    if overlaps(start, end, target_start, target_end):
        return 0
    if end <= target_start:
        return target_start - end
    return start - target_end


def load_gold(path: Path, chrom: str, region_start: int, region_end: int) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["chromosome"] != chrom:
                continue
            start = int(row["start"])
            end = int(row["end"])
            if overlaps(start, end, region_start, region_end):
                rows.append({"start": start, "end": end, "label": row.get("region_type", "gold")})
    rows.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return rows


def load_peaks(path: Path) -> list[dict[str, int | float | str]]:
    peaks: list[dict[str, int | float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            peaks.append(
                {
                    "chromosome": row["chromosome"],
                    "peak_start": int(row["peak_start"]),
                    "peak_end": int(row["peak_end"]),
                    "summit_center": int(row["summit_center"]),
                    "raw_score": float(row["raw_score"]),
                    "smoothed_score": float(row["smoothed_score"]),
                    "prominence": float(row["prominence"]),
                    "width_bp": float(row["width_bp"]),
                }
            )
    return peaks


def load_known_hars(path: Path, chrom: str, region_start: int, region_end: int) -> list[dict[str, int | str]]:
    hars: list[dict[str, int | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["染色体"] != chrom:
                continue
            start = int(row["起始位置"])
            end = int(row["终止位置"])
            if overlaps(start, end, region_start, region_end):
                label = "HAR"
                for item in HIGHLIGHT_HARS:
                    if start == item["start"] and end == item["end"]:
                        label = str(item["label"])
                hars.append(
                    {
                        "start": start,
                        "end": end,
                        "label": label,
                        "ccre": row.get("cCRE类型", "."),
                        "gene": row.get("GENCODE最近基因", "NPAS3"),
                    }
                )
    hars.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return hars


def load_bigwig(path: Path, chrom: str, region_start: int, region_end: int, bins: int) -> np.ndarray:
    if pyBigWig is None:
        raise RuntimeError("pyBigWig is required for bigWig tracks.")
    with pyBigWig.open(str(path)) as handle:
        values = handle.stats(chrom, region_start, region_end, nBins=bins, exact=True)
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float32)


def infer_ccre_class(fields: list[str]) -> str:
    joined = " ".join(fields)
    for label in ["CTCF-only", "DNase-H3K4me3", "pELS", "dELS", "PLS"]:
        if label in joined:
            return label
    return "cCRE"


def load_ccres(path: Path, chrom: str, region_start: int, region_end: int) -> list[dict[str, int | str]]:
    if pyBigWig is None:
        raise RuntimeError("pyBigWig is required for bigBed tracks.")
    colors = {
        "PLS": PALETTE["pdf_orange"],
        "pELS": PALETTE["pdf_gold"],
        "dELS": PALETTE["pdf_gold"],
        "CTCF-only": PALETTE["pdf_teal"],
        "DNase-H3K4me3": PALETTE["pdf_blue"],
        "cCRE": PALETTE["pdf_grey"],
    }
    rows: list[dict[str, int | str]] = []
    with pyBigWig.open(str(path)) as handle:
        for start, end, rest in handle.entries(chrom, region_start, region_end) or []:
            fields = rest.split("\t") if rest else []
            ccre_class = infer_ccre_class(fields)
            rows.append({"start": int(start), "end": int(end), "class": ccre_class, "color": colors[ccre_class]})
    rows.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return rows


def load_cadd_position_scores(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row.get("max_phred"):
                continue
            rows.append(
                {
                    "source": row["source"],
                    "index": int(row["index"]),
                    "chrom": row["chrom"],
                    "position": int(row["position"]),
                    "interval_start": int(row["interval_start"]),
                    "interval_end": int(row["interval_end"]),
                    "max_phred": float(row["max_phred"]),
                    "best_ref": row.get("best_ref", ""),
                    "best_alt": row.get("best_alt", ""),
                }
            )
    return rows


def load_phylop_position_scores(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row.get("phyloP"):
                continue
            rows.append(
                {
                    "source": row["source"],
                    "index": int(row["index"]),
                    "chrom": row["chrom"],
                    "position": int(row["position"]),
                    "interval_start": int(row["interval_start"]),
                    "interval_end": int(row["interval_end"]),
                    "phyloP": float(row["phyloP"]),
                }
            )
    return rows


def load_cadd_binned_track(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        return []
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row.get("max_phred"):
                continue
            rows.append(
                {
                    "position": int(row["position"]),
                    "bin_start": int(row["bin_start"]),
                    "bin_end": int(row["bin_end"]),
                    "max_phred": float(row["max_phred"]),
                }
            )
    return rows


def load_phylop_json_track(path: Path, region_start: int, region_end: int, bins: int) -> dict[str, np.ndarray]:
    if not path.exists():
        return {"x": np.asarray([], dtype=np.float64), "values": np.asarray([], dtype=np.float32)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    sums = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    span = region_end - region_start
    for item in payload.get("phyloP100way", []):
        start = int(item["start"])
        if start < region_start or start >= region_end:
            continue
        idx = min(bins - 1, max(0, int((start - region_start) * bins / span)))
        sums[idx] += float(item["value"])
        counts[idx] += 1
    values = np.full(bins, np.nan, dtype=np.float32)
    mask = counts > 0
    values[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    x = np.linspace(region_start, region_end, bins, endpoint=False) + span / bins / 2
    return {"x": x, "values": values}


def load_evo2_track(path: Path, field: str, chrom: str, region_start: int, region_end: int) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Evo2 track is required for teacher-style rendering: {path}")
    payload = np.load(path, allow_pickle=False)
    x = np.asarray(payload["centers"], dtype=np.int64)
    values = np.asarray(payload[field], dtype=np.float32)
    if x.shape != values.shape:
        raise ValueError(f"Evo2 track coordinate/value shape mismatch: {x.shape} vs {values.shape}")
    mask = (x >= region_start) & (x <= region_end) & np.isfinite(values)
    if not np.any(mask):
        raise ValueError(f"Evo2 track has no finite values in {chrom}:{region_start}-{region_end}")
    return {"x": x[mask], "values": values[mask], "path": str(path), "field": field}


def load_genes(path: Path, chrom: str, region_start: int, region_end: int) -> list[dict[str, int | str]]:
    by_symbol: dict[tuple[str, str], dict[str, int | str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 13 or row[2] != chrom:
                continue
            start = int(row[4])
            end = int(row[5])
            if not overlaps(start, end, region_start, region_end):
                continue
            key = (row[12], row[3])
            if key not in by_symbol:
                by_symbol[key] = {"start": start, "end": end, "strand": row[3], "symbol": row[12]}
            else:
                by_symbol[key]["start"] = min(int(by_symbol[key]["start"]), start)
                by_symbol[key]["end"] = max(int(by_symbol[key]["end"]), end)
    genes = list(by_symbol.values())
    genes.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return genes


def classify_peaks(
    peaks: list[dict[str, int | float | str]],
    overlap_rows: list[dict[str, object]],
) -> list[dict[str, int | float | str]]:
    annotated: list[dict[str, int | float | str]] = []
    for index, peak in enumerate(peaks):
        row = overlap_rows[index]
        peak_start = int(peak["peak_start"])
        peak_end = int(peak["peak_end"])
        nearest_har = min(
            distance_to_interval(peak_start, peak_end, int(item["start"]), int(item["end"])) for item in HIGHLIGHT_HARS
        )
        if bool(row["overlaps_gold"]):
            category = "overlaps_gold"
        elif nearest_har <= NEAR_HAR_DISTANCE_BP:
            category = "near_known_har"
        else:
            category = "other_non_gold"
        annotated.append({**peak, "peak_index": index, "category": category, "nearest_highlight_har_bp": nearest_har})
    return annotated


def draw_interval_row(
    axis,
    intervals: list[dict[str, int | str]],
    y: float,
    height: float,
    color: str,
    alpha: float,
    label: str,
) -> None:
    for interval in intervals:
        axis.add_patch(
            patches.Rectangle(
                (int(interval["start"]), y),
                int(interval["end"]) - int(interval["start"]),
                height,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
            )
        )
    if intervals:
        axis.text(axis.get_xlim()[0], y + height + 0.06, label, color=color, fontsize=fs(9), va="bottom")


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


def draw_score_spotlight(
    axis,
    centers: np.ndarray,
    smoothed_scores: np.ndarray,
    hit_spans: list[tuple[int, int]],
    region_start: int,
    region_end: int,
) -> None:
    spotlight_spans = merge_spans(
        [(start - HIT_SPOTLIGHT_PADDING_BP, end + HIT_SPOTLIGHT_PADDING_BP) for start, end in hit_spans],
        region_start=region_start,
        region_end=region_end,
    )
    for start, end in spotlight_spans:
        mask = (centers >= start) & (centers <= end)
        axis.plot(centers[mask], smoothed_scores[mask], color=STYLE["score"], lw=1.95, alpha=1.0, zorder=4)


def draw_known_hars(axis, known_hars: list[dict[str, int | str]], region_start: int, region_end: int) -> None:
    for har in known_hars:
        start = int(har["start"])
        end = int(har["end"])
        display_start, display_end = display_span(start, end, MIN_INTERVAL_DISPLAY_BP, region_start, region_end)
        is_highlight = str(har["label"]) != "HAR"
        color = STYLE["known_har_hi"] if is_highlight else STYLE["known_har"]
        axis.add_patch(
            patches.Rectangle(
                (display_start, 0.22),
                display_end - display_start,
                0.34,
                facecolor=color,
                edgecolor="none",
                alpha=0.78,
            )
        )
        if is_highlight:
            axis.text(start, 0.55, str(har["label"]), color=color, fontsize=fs(8), va="bottom", rotation=30)
    axis.text(region_start, 0.56, "Known HARs", color=STYLE["known_har_hi"], fontsize=fs(9), va="bottom")


def draw_gene_track(axis, genes: list[dict[str, int | str]], region_start: int, region_end: int) -> None:
    placed: list[dict[str, int | str]] = []
    for gene in genes:
        level = 0
        while any(
            item["level"] == level
            and overlaps(int(gene["start"]), int(gene["end"]), int(item["start"]), int(item["end"]))
            for item in placed
        ):
            level += 1
        placed.append({**gene, "level": level})
    max_level = max([int(item["level"]) for item in placed], default=0)
    for gene in placed:
        y = max_level - int(gene["level"])
        start = int(gene["start"])
        end = int(gene["end"])
        symbol = str(gene["symbol"])
        is_npas3 = symbol == "NPAS3"
        color = STYLE["gene_hi"] if is_npas3 else STYLE["gene"]
        axis.plot([start, end], [y, y], color=color, lw=1.8 if is_npas3 else 0.9, alpha=0.95 if is_npas3 else 0.50)
        if is_npas3:
            axis.text(start, y + 0.22, symbol, color=color, fontsize=fs(10), fontweight="bold", va="bottom")
    axis.set_ylim(-0.6, max_level + 0.8)
    axis.set_yticks([])
    axis.set_ylabel("")
    axis.set_xlim(region_start, region_end)


def draw_ccres(axis, ccres: list[dict[str, int | str]], region_start: int, region_end: int) -> None:
    placed: list[dict[str, int]] = []
    stacked: list[dict[str, int | str]] = []
    for ccre in ccres:
        level = 0
        while any(
            item["level"] == level and overlaps(int(ccre["start"]), int(ccre["end"]), item["start"], item["end"])
            for item in placed
        ):
            level += 1
        placed.append({"start": int(ccre["start"]), "end": int(ccre["end"]), "level": level})
        stacked.append({**ccre, "level": level})
    max_level = max([int(item["level"]) for item in stacked], default=0)
    for ccre in stacked:
        y = max_level - int(ccre["level"])
        axis.add_patch(
            patches.Rectangle(
                (int(ccre["start"]), y - 0.15),
                int(ccre["end"]) - int(ccre["start"]),
                0.30,
                facecolor=str(ccre["color"]),
                edgecolor="none",
                alpha=1.00,
            )
        )
    axis.set_xlim(region_start, region_end)
    axis.set_ylim(-0.5, max_level + 0.5)
    axis.set_yticks([])
    axis.set_ylabel("cCRE", fontsize=fs(9))


def draw_ccre_group_track(
    axis,
    ccres: list[dict[str, int | str]],
    classes: set[str],
    label: str,
    color: str,
    region_start: int,
    region_end: int,
) -> None:
    selected = [ccre for ccre in ccres if str(ccre["class"]) in classes]
    for ccre in selected:
        start = int(ccre["start"])
        end = int(ccre["end"])
        display_start, display_end = display_span(start, end, MIN_CCRE_DISPLAY_BP, region_start, region_end)
        axis.add_patch(
            patches.Rectangle(
                (display_start, 0.20),
                display_end - display_start,
                0.60,
                facecolor=color,
                edgecolor="none",
                alpha=0.84,
            )
        )
    axis.set_xlim(region_start, region_end)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_ylabel("")
    axis.text(
        -0.012,
        0.5,
        label,
        transform=axis.get_yaxis_transform(),
        color="black",
        fontsize=fs(8),
        ha="right",
        va="center",
        clip_on=False,
    )


def draw_cadd_track(
    axis,
    cadd_rows: list[dict[str, float | int | str]],
    region_start: int,
    region_end: int,
) -> None:
    if cadd_rows:
        x = np.asarray([int(row["position"]) for row in cadd_rows], dtype=np.float64)
        y = np.asarray([float(row["max_phred"]) for row in cadd_rows], dtype=np.float32)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        axis.plot(x, y, color=STYLE["cadd"], lw=0.78, alpha=0.96)
        axis.fill_between(x, y, 0, color=STYLE["cadd"], alpha=0.24, linewidth=0)
    axis.axhline(10, color=STYLE["grid"], lw=0.65, ls="--", alpha=0.75)
    axis.axhline(20, color=STYLE["muted"], lw=0.65, ls="--", alpha=0.65)
    max_score = max([float(row["max_phred"]) for row in cadd_rows], default=25.0)
    axis.set_ylim(0, max(25, max_score + 2))
    axis.set_xlim(region_start, region_end)
    axis.set_ylabel("CADD", fontsize=fs(9))


def draw_phylop_track(
    axis,
    phylop_track: dict[str, np.ndarray],
    region_start: int,
    region_end: int,
) -> None:
    x = np.asarray(phylop_track.get("x", []), dtype=np.float64)
    y = np.asarray(phylop_track.get("values", []), dtype=np.float32)
    valid = np.isfinite(y)
    if np.any(valid):
        axis.plot(x[valid], y[valid], color=STYLE["phylop"], lw=0.68, alpha=0.90)
        axis.fill_between(x[valid], y[valid], 0, color=STYLE["phylop"], alpha=0.20, linewidth=0)
    values = y[valid].tolist()
    min_value = min(values, default=-2.0)
    max_value = max(values, default=6.0)
    axis.axhline(0, color="#BDBDBD", lw=0.65, ls="-", alpha=0.8)
    axis.set_ylim(min(-2, min_value - 0.5), max(6, max_value + 0.5))
    axis.set_xlim(region_start, region_end)
    axis.set_ylabel("PhyloP", fontsize=fs(9), color=STYLE["text"])


def setup_axis(axis) -> None:
    axis.set_facecolor(STYLE["panel_bg"])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(STYLE["panel_edge"])
    axis.spines["bottom"].set_color(STYLE["panel_edge"])
    axis.grid(axis="y", color=STYLE["grid"], linewidth=0.5, alpha=0.35)
    axis.tick_params(axis="both", labelsize=fs(8), colors=STYLE["text"])


def render_figure(
    name: str,
    tracks: dict[str, object],
    full: bool,
) -> None:
    chrom = str(tracks["chrom"])
    centers = np.asarray(tracks["centers"])
    raw_scores = np.asarray(tracks["raw_scores"])
    smoothed_scores = np.asarray(tracks["smoothed_scores"])
    region_start = int(tracks["region_start"])
    region_end = int(tracks["region_end"])
    peaks = list(tracks["peaks"])
    gold = list(tracks["gold"])
    known_hars = list(tracks["known_hars"])
    genes = list(tracks["genes"])

    if full:
        fig, axes = plt.subplots(
            8,
            1,
            figsize=(15, 12.9),
            sharex=True,
            gridspec_kw={"height_ratios": [3.5, 0.9, 0.9, 0.85, 0.85, 1.1, 1.0, 1.75]},
        )
        (
            score_ax,
            h3_ax,
            cons_ax,
            cadd_ax,
            phylop_ax,
            ccre_ax,
            gene_ax,
            interval_ax,
        ) = axes
    else:
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(18, 7.4),
            sharex=True,
            gridspec_kw={"height_ratios": [3.6, 0.85, 1.0, 1.75]},
        )
        score_ax, har_ax, gene_ax, interval_ax = axes

    fig.patch.set_facecolor(STYLE["paper_bg"])
    score_ax.plot(centers, raw_scores, color=STYLE["raw"], lw=0.35, alpha=0.20, label="raw score", zorder=1)
    score_ax.plot(centers, smoothed_scores, color=STYLE["score"], lw=1.08, alpha=0.66, label="smoothed score", zorder=2)
    hit_spans: list[tuple[int, int]] = []
    for peak in peaks:
        category = str(peak["category"])
        if category == "overlaps_gold":
            color = STYLE["peak_gold"]
            alpha = 0.18
            summit_alpha = 0.78
            hit_spans.append((int(peak["peak_start"]), int(peak["peak_end"])))
        elif category == "near_known_har":
            color = STYLE["peak_near_har"]
            alpha = 0.20
            summit_alpha = 0.85
        else:
            color = STYLE["peak_other"]
            alpha = 0.16
            summit_alpha = 0.40
        score_ax.axvspan(
            int(peak["peak_start"]), int(peak["peak_end"]), color=color, alpha=alpha, linewidth=0, zorder=2.2
        )
        score_ax.axvline(int(peak["summit_center"]), color=color, lw=0.75, alpha=summit_alpha, zorder=4.5)
        score_ax.scatter(
            [int(peak["summit_center"])],
            [float(peak["smoothed_score"])],
            s=14 if category != "other_non_gold" else 8,
            color=color,
            zorder=5,
        )
        if category == "near_known_har":
            score_ax.text(
                int(peak["summit_center"]),
                float(peak["smoothed_score"]) + 0.035,
                f"Peak {peak['peak_index']}",
                color=STYLE["peak_near_har"],
                fontsize=fs(8),
                ha="center",
                va="bottom",
                zorder=6,
            )
    draw_score_spotlight(score_ax, centers, smoothed_scores, hit_spans, region_start, region_end)
    score_ax.set_ylabel("Score", fontsize=fs(10), color=STYLE["text"])
    score_ax.set_title(
        "Predicted HAR-associated regulatory peaks near NPAS3",
        fontsize=fs(14),
        fontweight="bold",
        color=STYLE["text"],
        pad=10,
    )
    score_ax.legend(loc="upper right", frameon=False, fontsize=fs(8))
    score_ax.text(
        0.012,
        0.92,
        "Gold hit: 17/32\nCalled peaks: 24\nNear-HAR non-gold peaks: 2",
        transform=score_ax.transAxes,
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
    setup_axis(score_ax)

    if full:
        h3_ax.plot(centers, np.asarray(tracks["h3k27ac"]), color=STYLE["h3k27ac"], lw=0.62, alpha=0.78)
        h3_ax.set_ylabel("H3K27ac", fontsize=fs(9), color=STYLE["text"])
        setup_axis(h3_ax)
        cons_ax.plot(centers, np.asarray(tracks["phastcons"]), color=STYLE["phastcons"], lw=0.55, alpha=0.72)
        cons_ax.set_ylabel("phastCons", fontsize=fs(9), color=STYLE["text"])
        setup_axis(cons_ax)
        draw_cadd_track(cadd_ax, list(tracks["cadd_track"]), region_start, region_end)
        setup_axis(cadd_ax)
        draw_phylop_track(phylop_ax, dict(tracks["phylop_track"]), region_start, region_end)
        setup_axis(phylop_ax)
        ccres = list(tracks["ccres"])
        draw_ccres(ccre_ax, ccres, region_start, region_end)
        setup_axis(ccre_ax)
        draw_gene_track(gene_ax, genes, region_start, region_end)
        setup_axis(gene_ax)
    else:
        draw_known_hars(har_ax, known_hars, region_start, region_end)
        har_ax.set_ylim(0, 1.0)
        har_ax.set_yticks([])
        har_ax.set_ylabel("HAR", fontsize=fs(9), color=STYLE["text"])
        setup_axis(har_ax)
        draw_gene_track(gene_ax, genes, region_start, region_end)
        setup_axis(gene_ax)

    interval_ax.set_ylim(0, 1.05)
    interval_ax.set_yticks([])
    interval_ax.set_xlabel(f"{chrom} position", fontsize=fs(10))
    interval_ax.set_ylabel("")
    interval_ax.set_xlim(region_start, region_end)
    for interval in gold:
        start = int(interval["start"])
        end = int(interval["end"])
        display_start, display_end = display_span(start, end, MIN_INTERVAL_DISPLAY_BP, region_start, region_end)
        interval_ax.add_patch(
            patches.Rectangle(
                (display_start, 0.60),
                display_end - display_start,
                0.30,
                facecolor=STYLE["gold"],
                edgecolor="none",
                alpha=0.86,
            )
        )
    for peak in peaks:
        category = str(peak["category"])
        color = {
            "overlaps_gold": STYLE["peak_gold"],
            "near_known_har": STYLE["peak_near_har"],
            "other_non_gold": STYLE["peak_other"],
        }[category]
        start = int(peak["peak_start"])
        end = int(peak["peak_end"])
        display_start, display_end = display_span(start, end, MIN_INTERVAL_DISPLAY_BP, region_start, region_end)
        interval_ax.add_patch(
            patches.Rectangle(
                (display_start, 0.14),
                display_end - display_start,
                0.32,
                facecolor=color,
                edgecolor="none",
                alpha=0.82 if category != "other_non_gold" else 0.74,
            )
        )
    interval_ax.text(
        -0.012,
        0.74,
        "Known peaks",
        transform=interval_ax.get_yaxis_transform(),
        color="black",
        fontsize=fs(9),
        ha="right",
        va="center",
        clip_on=False,
    )
    interval_ax.text(
        -0.012,
        0.29,
        "Predicted peaks",
        transform=interval_ax.get_yaxis_transform(),
        color="black",
        fontsize=fs(9),
        ha="right",
        va="center",
        clip_on=False,
    )
    setup_axis(interval_ax)

    apply_genomic_xaxis(fig, region_end)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the HAR NPAS3 literature example tracks.")
    parser.add_argument("--style", choices=["legacy", "teacher"], default="legacy")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--track-dir", type=Path, default=None)
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--peaks-tsv", type=Path, default=None)
    parser.add_argument("--overlap-json", type=Path, default=None)
    parser.add_argument("--gold-csv", type=Path, default=None)
    parser.add_argument("--raw-har-csv", type=Path, default=None)
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
    global ROOT, SOURCE_DIR, OUTDIR, TRACK_DIR, STYLE_MODE
    global INPUT_NPZ, PEAKS_TSV, OVERLAP_JSON, GOLD_CSV, RAW_HAR_CSV, REFGENE_GZ
    global H3K27AC_BW, PHASTCONS_BW, CCRE_BB
    global CADD_POSITION_TSV, PHYLOP_POSITION_TSV, CADD_TRACK_TSV, PHYLOP_JSON
    global EVO2_TRACK_NPZ, EVO2_TRACK_FIELD

    ROOT = args.root
    SOURCE_DIR = ROOT / "results/literature_example_tracks"
    TRACK_DIR = args.track_dir or ROOT / "results/literature_example_tracks_beautified"
    STYLE_MODE = args.style
    OUTDIR = args.output_dir or (
        ROOT / "results/literature_example_tracks_beautified"
        if STYLE_MODE == "legacy"
        else ROOT / "results/literature_example_tracks_beautified_teacher_style"
    )

    INPUT_NPZ = args.input_npz or SOURCE_DIR / "HAR_NPAS3_optimized_with_tracks.smoothed_scores.npz"
    PEAKS_TSV = args.peaks_tsv or SOURCE_DIR / "HAR_NPAS3_optimized_with_tracks.peaks.tsv"
    OVERLAP_JSON = args.overlap_json or SOURCE_DIR / "HAR_NPAS3_optimized_with_tracks.overlap_summary.json"
    GOLD_CSV = args.gold_csv or ROOT / "data/processed/01_HAR_positive_windows_512_full.csv"
    RAW_HAR_CSV = args.raw_har_csv or ROOT / "data/raw/HARs.CSV"
    REFGENE_GZ = args.refgene_gz or ROOT / "reference/hg38_refGene.txt.gz"
    H3K27AC_BW = args.h3k27ac_bw or ROOT / "external_tracks/ucsc/wgEncodeBroadHistoneH1hescH3k27acStdSig.bigWig"
    PHASTCONS_BW = args.phastcons_bw or ROOT / "external_tracks/ucsc/hg38.phastCons100way.bw"
    CCRE_BB = args.ccre_bb or ROOT / "external_tracks/ucsc/encodeCcreCombined.hg38.bb"
    CADD_POSITION_TSV = args.cadd_position_tsv or TRACK_DIR / "HAR_NPAS3_CADD_GRCh38_v1.7_peak_position_scores.tsv"
    PHYLOP_POSITION_TSV = args.phylop_position_tsv or TRACK_DIR / "HAR_NPAS3_phyloP100way_peak_position_scores.tsv"
    CADD_TRACK_TSV = args.cadd_track_tsv or TRACK_DIR / "HAR_NPAS3_CADD_GRCh38_v1.7_binned_track.tsv"
    PHYLOP_JSON = args.phylop_json or TRACK_DIR / "tmp_phyloP_chr14_33000000_34000000.json"
    EVO2_TRACK_NPZ = args.evo2_track_npz or ROOT / "results/final_har_haqer_panel/evo2_1b_hevo_protocol/scores/har/g_blocks_6_mlp_l3_tracks.npz"
    EVO2_TRACK_FIELD = args.evo2_track_field


def teacher_axis(axis, label: str, region_start: int, region_end: int, show_x: bool = False) -> None:
    setup_axis(axis)
    axis.set_xlim(region_start, region_end)
    axis.set_ylabel(
        label,
        fontsize=fs(8.5),
        color=STYLE["text"],
        labelpad=34,
        rotation=0,
        ha="right",
        va="center",
    )
    if not show_x:
        axis.tick_params(axis="x", labelbottom=False)


def teacher_group_label(axis, label: str) -> None:
    axis.text(
        -0.19,
        0.5,
        label,
        transform=axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=fs(8.5),
        fontweight="bold",
        color=STYLE["text"],
    )


def draw_evo2_track(axis, track: dict[str, object], region_start: int, region_end: int) -> None:
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
    axis.fill_between(x, ymin, y, color=STYLE["evo2"], alpha=0.14, linewidth=0)
    axis.plot(x, y, color=STYLE["evo2"], lw=0.78, alpha=0.92)
    axis.set_ylim(ymin - 0.08 * span, ymax + 0.16 * span)
    axis.text(
        0.995,
        0.82,
        "experimental non-FP8",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=fs(6.5),
        color=STYLE["muted"],
    )
    teacher_axis(axis, "EVO 2 likelihood score", region_start, region_end)


def draw_teacher_intervals(
    axis,
    rows: list[dict[str, int | float | str]],
    start_key: str,
    end_key: str,
    region_start: int,
    region_end: int,
    label: str,
    color_fn,
    text_fn=None,
) -> None:
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    for row_index, row in enumerate(rows):
        start = int(row[start_key])
        end = int(row[end_key])
        display_start, display_end = display_span(start, end, MIN_INTERVAL_DISPLAY_BP, region_start, region_end)
        color = color_fn(row)
        axis.add_patch(
            patches.Rectangle(
                (display_start, 0.24),
                display_end - display_start,
                0.52,
                facecolor=color,
                edgecolor="none",
                alpha=0.86,
            )
        )
        if text_fn is not None:
            text = text_fn(row)
            if text:
                axis.text(
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
    teacher_axis(axis, label, region_start, region_end)


def render_teacher_figure(tracks: dict[str, object], evo2_track: dict[str, object]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    chrom = str(tracks["chrom"])
    centers = np.asarray(tracks["centers"])
    raw_scores = np.asarray(tracks["raw_scores"])
    smoothed_scores = np.asarray(tracks["smoothed_scores"])
    region_start = int(tracks["region_start"])
    region_end = int(tracks["region_end"])
    peaks = list(tracks["peaks"])
    gold = list(tracks["gold"])
    known_hars = list(tracks["known_hars"])
    genes = list(tracks["genes"])

    fig, axes = plt.subplots(
        10,
        1,
        figsize=(15, 10.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 2.2, 0.85, 0.85, 1.0, 0.85, 1.0, 1.0, 1.1, 1.65]},
    )
    score_ax, evo2_ax, peak_ax, known_ax, h3_ax, ccre_ax, cons_ax, phylop_ax, cadd_ax, gene_ax = axes
    fig.patch.set_facecolor(STYLE["paper_bg"])
    score_ax.text(
        -0.16,
        1.08,
        "g",
        transform=score_ax.transAxes,
        fontsize=fs(15),
        fontweight="bold",
        color=STYLE["text"],
        ha="left",
        va="bottom",
        clip_on=False,
    )

    hit_spans: list[tuple[int, int]] = []
    score_ax.plot(centers, raw_scores, color=STYLE["raw"], lw=0.35, alpha=0.24, label="raw score", zorder=1)
    score_ax.plot(
        centers,
        smoothed_scores,
        color=STYLE["score"],
        lw=1.25,
        alpha=0.92,
        label="smoothed hEvo head score",
        zorder=2,
    )
    for peak in peaks:
        category = str(peak["category"])
        color = {
            "overlaps_gold": STYLE["peak_gold"],
            "near_known_har": STYLE["peak_near_har"],
            "other_non_gold": STYLE["peak_other"],
        }[category]
        if category == "overlaps_gold":
            hit_spans.append((int(peak["peak_start"]), int(peak["peak_end"])))
        score_ax.axvspan(
            int(peak["peak_start"]),
            int(peak["peak_end"]),
            color=color,
            alpha=0.15 if category != "other_non_gold" else 0.10,
            linewidth=0,
            zorder=2.2,
        )
        score_ax.axvline(int(peak["summit_center"]), color=color, lw=0.65, alpha=0.78, zorder=4)
        score_ax.scatter(
            [int(peak["summit_center"])],
            [float(peak["smoothed_score"])],
            s=13 if category != "other_non_gold" else 7,
            color=color,
            zorder=5,
        )
        if category == "near_known_har":
            score_ax.text(
                int(peak["summit_center"]),
                float(peak["smoothed_score"]) + 0.035,
                f"Peak {peak['peak_index']}",
                color=STYLE["peak_near_har"],
                fontsize=fs(7.2),
                ha="center",
                va="bottom",
            )
    draw_score_spotlight(score_ax, centers, smoothed_scores, hit_spans, region_start, region_end)
    score_ax.set_title("HAR | NPAS3 | hg38", fontsize=fs(14), fontweight="bold", color=STYLE["text"], pad=10)
    score_ax.text(
        0.995,
        0.93,
        f"gold interval recovery: {tracks['gold_recovery']}   |   called peaks: {len(peaks)}",
        transform=score_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fs(7.2),
        color=STYLE["text"],
    )
    score_ax.legend(loc="upper left", frameon=False, fontsize=fs(7.5), ncol=2)
    teacher_axis(score_ax, "hEvo-base score", region_start, region_end)
    teacher_group_label(score_ax, "MODEL")

    draw_evo2_track(evo2_ax, evo2_track, region_start, region_end)

    draw_teacher_intervals(
        peak_ax,
        peaks,
        "peak_start",
        "peak_end",
        region_start,
        region_end,
        "Peak calls",
        lambda row: {
            "overlaps_gold": STYLE["peak_gold"],
            "near_known_har": STYLE["peak_near_har"],
            "other_non_gold": STYLE["peak_other"],
        }[str(row["category"])],
        lambda row: f"P{row['peak_index']}" if str(row["category"]) == "near_known_har" else "",
    )
    draw_teacher_intervals(
        known_ax,
        known_hars,
        "start",
        "end",
        region_start,
        region_end,
        "Known HARs",
        lambda row: STYLE["known_har_hi"] if str(row.get("label", "HAR")) != "HAR" else STYLE["known_har"],
        lambda row: str(row.get("label", "")) if str(row.get("label", "HAR")) != "HAR" else "",
    )

    h3_ax.plot(centers, np.asarray(tracks["h3k27ac"]), color=STYLE["h3k27ac"], lw=0.68, alpha=0.82)
    teacher_axis(h3_ax, "H3K27ac", region_start, region_end)
    teacher_group_label(h3_ax, "REG.")
    draw_ccres(ccre_ax, list(tracks["ccres"]), region_start, region_end)
    teacher_axis(ccre_ax, "cCRE", region_start, region_end)

    cons_ax.plot(centers, np.asarray(tracks["phastcons"]), color=STYLE["phastcons"], lw=0.58, alpha=0.76)
    teacher_axis(cons_ax, "phastCons", region_start, region_end)
    teacher_group_label(cons_ax, "EVOL.")
    draw_phylop_track(phylop_ax, dict(tracks["phylop_track"]), region_start, region_end)
    teacher_axis(phylop_ax, "phyloP", region_start, region_end)

    draw_cadd_track(cadd_ax, list(tracks["cadd_track"]), region_start, region_end)
    teacher_axis(cadd_ax, "CADD", region_start, region_end)
    teacher_group_label(cadd_ax, "REF.\nSCORES")

    draw_gene_track(gene_ax, genes, region_start, region_end)
    teacher_axis(gene_ax, "RefSeq genes", region_start, region_end, show_x=True)
    teacher_group_label(gene_ax, "GENOME")
    gene_ax.set_xlabel(f"{chrom} position", fontsize=fs(9.5))

    apply_genomic_xaxis(fig, region_end)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.92, bottom=0.08, hspace=0.08)
    output_base = OUTDIR / "HAR_NPAS3_teacher_style_with_evo2"
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_paths(args)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = np.load(INPUT_NPZ, allow_pickle=False)
    chromosomes = payload["chromosome"]
    chrom = str(chromosomes[0])
    centers = payload["center"].astype(np.int64)
    raw_scores = payload["raw_scores"].astype(np.float32)
    smoothed_scores = payload["smoothed_scores"].astype(np.float32)
    region_start = int(payload["start"].min())
    region_end = int(payload["end"].max())
    valid_mask = centers >= region_start

    overlap_summary = json.loads(OVERLAP_JSON.read_text(encoding="utf-8"))
    peaks = classify_peaks(load_peaks(PEAKS_TSV), overlap_summary["peak_rows"])
    gold = load_gold(GOLD_CSV, chrom, region_start, region_end)
    known_hars = load_known_hars(RAW_HAR_CSV, chrom, region_start, region_end)
    genes = load_genes(REFGENE_GZ, chrom, region_start, region_end)

    tracks: dict[str, object] = {
        "chrom": chrom,
        "centers": centers[valid_mask],
        "raw_scores": raw_scores[valid_mask],
        "smoothed_scores": smoothed_scores[valid_mask],
        "region_start": region_start,
        "region_end": region_end,
        "peaks": peaks,
        "gold": gold,
        "gold_recovery": f"{overlap_summary['num_gold_intervals_hit']}/{overlap_summary['num_gold_intervals']}",
        "known_hars": known_hars,
        "genes": genes,
    }
    tracks["h3k27ac"] = load_bigwig(H3K27AC_BW, chrom, region_start, region_end, len(centers))[valid_mask]
    tracks["phastcons"] = load_bigwig(PHASTCONS_BW, chrom, region_start, region_end, len(centers))[valid_mask]
    tracks["cadd_position_scores"] = load_cadd_position_scores(CADD_POSITION_TSV)
    tracks["phylop_position_scores"] = load_phylop_position_scores(PHYLOP_POSITION_TSV)
    tracks["cadd_track"] = load_cadd_binned_track(CADD_TRACK_TSV)
    tracks["phylop_track"] = load_phylop_json_track(PHYLOP_JSON, region_start, region_end, len(centers))
    tracks["ccres"] = load_ccres(CCRE_BB, chrom, region_start, region_end)

    if STYLE_MODE == "teacher":
        evo2_track = load_evo2_track(EVO2_TRACK_NPZ, EVO2_TRACK_FIELD, chrom, region_start, region_end)
        tracks["evo2_track"] = evo2_track
        render_teacher_figure(tracks, evo2_track)
    else:
        render_figure("HAR_NPAS3_optimized_clean_annotated", tracks, full=False)
        render_figure("HAR_NPAS3_optimized_full_tracks_annotated", tracks, full=True)

    category_counts: dict[str, int] = {}
    ccre_class_counts: dict[str, int] = {}
    for ccre in list(tracks["ccres"]):
        ccre_class = str(ccre["class"])
        ccre_class_counts[ccre_class] = ccre_class_counts.get(ccre_class, 0) + 1
    cadd_rows = list(tracks["cadd_position_scores"])
    cadd_source_counts: dict[str, int] = {}
    for row in cadd_rows:
        source = str(row["source"])
        cadd_source_counts[source] = cadd_source_counts.get(source, 0) + 1
    phylop_rows = list(tracks["phylop_position_scores"])
    phylop_source_counts: dict[str, int] = {}
    for row in phylop_rows:
        source = str(row["source"])
        phylop_source_counts[source] = phylop_source_counts.get(source, 0) + 1
    near_har_peaks = []
    for peak in peaks:
        category = str(peak["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
        if category == "near_known_har":
            near_har_peaks.append(
                {
                    "peak_index": peak["peak_index"],
                    "peak_start": peak["peak_start"],
                    "peak_end": peak["peak_end"],
                    "summit_center": peak["summit_center"],
                    "nearest_highlight_har_bp": peak["nearest_highlight_har_bp"],
                }
            )
    summary_name = (
        "HAR_NPAS3_teacher_style_with_evo2_summary.json"
        if STYLE_MODE == "teacher"
        else "HAR_NPAS3_optimized_beautified_summary.json"
    )
    (OUTDIR / summary_name).write_text(
        json.dumps(
            {
                "style": STYLE_MODE,
                "layout": ["MODEL", "REG.", "EVOL.", "REF. SCORES", "GENOME"],
                "output": str(OUTDIR / "HAR_NPAS3_teacher_style_with_evo2") if STYLE_MODE == "teacher" else None,
                "evo2_track_npz": str(EVO2_TRACK_NPZ) if STYLE_MODE == "teacher" else None,
                "evo2_track_field": EVO2_TRACK_FIELD if STYLE_MODE == "teacher" else None,
                "source_peaks_tsv": str(PEAKS_TSV),
                "source_overlap_summary": str(OVERLAP_JSON),
                "num_peaks": len(peaks),
                "num_gold_intervals": overlap_summary["num_gold_intervals"],
                "num_gold_intervals_hit": overlap_summary["num_gold_intervals_hit"],
                "category_counts": category_counts,
                "ccre_class_counts": ccre_class_counts,
                "cadd_position_scores_tsv": str(CADD_POSITION_TSV),
                "cadd_binned_track_tsv": str(CADD_TRACK_TSV),
                "cadd_source": "CADD GRCh38-v1.7 API, binned maximum PHRED over sampled positions across the displayed locus",
                "cadd_source_counts": cadd_source_counts,
                "phylop_position_scores_tsv": str(PHYLOP_POSITION_TSV),
                "phylop_json": str(PHYLOP_JSON),
                "phylop_source": "UCSC phyloP100way region track, binned mean values across the displayed locus",
                "phylop_source_counts": phylop_source_counts,
                "full_tracks": [
                    "EVO 2 likelihood score (experimental non-FP8)",
                    "H3K27ac",
                    "phastCons",
                    "CADD GRCh38-v1.7 binned max PHRED continuous track",
                    "phyloP100way binned mean continuous track",
                    "cCRE combined classes",
                    "NPAS3 gene",
                    "Known peaks",
                    "Predicted peaks",
                ],
                "near_har_distance_bp": NEAR_HAR_DISTANCE_BP,
                "near_har_peaks": near_har_peaks,
                "highlight_hars": HIGHLIGHT_HARS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
