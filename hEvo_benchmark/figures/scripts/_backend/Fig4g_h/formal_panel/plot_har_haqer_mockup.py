from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "results/final_har_haqer_panel"
HAR_SCRIPT = Path("results/literature_example_tracks_beautified/beautify_har_npas3_optimized.py")
HAQER_SCRIPT = Path("results/haqer_meaningful_region_chr19/plot_haqer_chr19_candidate.py")

GERP_URL = (
    "https://personal.broadinstitute.org/konradk/loftee_data/GRCh38/"
    "gerp_conservation_scores.homo_sapiens.GRCh38.bw"
)
GERP_CHROM_MODE = "no_chr_prefix"
LINSIGHT_URL = "https://zenodo.org/records/3955933/files/GRCh38_LinSight.bed.gz?download=1"
LINSIGHT_INDEX_URL = "https://zenodo.org/records/3955933/files/GRCh38_LinSight.bed.gz.csi?download=1"
BIGWIG_SUMMARY_URL = "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64.v369/bigWigSummary"

PALETTE = {
    "navy": "#173F63",
    "orange": "#C85A0A",
    "teal": "#008A93",
    "gold": "#B07C32",
    "green": "#5A9367",
    "blue": "#4C79A8",
    "purple": "#7667A8",
    "grey": "#B8B4AE",
    "light_grey": "#E7E4DE",
    "text": "#243746",
    "muted": "#68737A",
    "grid": "#E3E4E1",
    "edge": "#AEB7BB",
    "paper": "#FFFFFF",
}

TRACK_COLORS = {
    "score": PALETTE["orange"],
    "raw": PALETTE["grey"],
    "peak_hit": PALETTE["orange"],
    "peak_near": PALETTE["blue"],
    "peak_other": PALETTE["light_grey"],
    "known": PALETTE["navy"],
    "known_highlight": PALETTE["teal"],
    "h3k27ac": PALETTE["gold"],
    "ccre_enhancer": PALETTE["gold"],
    "ccre_promoter": PALETTE["orange"],
    "ccre_ctcf": PALETTE["teal"],
    "ccre_other": PALETTE["blue"],
    "phastcons": PALETTE["green"],
    "phylop": PALETTE["blue"],
    "cadd": PALETTE["gold"],
    "gerp": "#6A7D72",
    "linsight": PALETTE["purple"],
    "gene": PALETTE["navy"],
    "gene_other": "#8A949A",
}


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_args(root: Path, style: str = "teacher") -> SimpleNamespace:
    return SimpleNamespace(
        style=style,
        root=root,
        output_dir=root / "results/final_har_haqer_panel",
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
    )


def configure_sources(root: Path) -> tuple[ModuleType, ModuleType]:
    har = load_module(root / HAR_SCRIPT, "har_plot_source")
    haqer = load_module(root / HAQER_SCRIPT, "haqer_plot_source")
    har.configure_paths(source_args(root))
    haqer.configure_paths(source_args(root))
    return har, haqer


def write_track_tsv(
    path: Path,
    chrom: str,
    start: int,
    end: int,
    values: np.ndarray,
    source: str,
    aggregation: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["chrom", "start", "end", "value", "source", "aggregation"])
        for index, value in enumerate(values):
            left = int(round(start + (end - start) * index / len(values)))
            right = int(round(start + (end - start) * (index + 1) / len(values)))
            writer.writerow([chrom, left, right, "nan" if not np.isfinite(value) else f"{value:.8g}", source, aggregation])


def ensure_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "HAR-HAQER-figure/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def ensure_bigwig_summary() -> Path:
    cache_root = Path(
        os.environ.get("HEVO_FIGURE_CACHE_DIR", Path(tempfile.gettempdir()) / "hevo_figures_runtime")
    ).expanduser()
    path = cache_root / "tools/bigWigSummary_linux_x86_64_v369"
    if not path.exists() or path.stat().st_size < 100_000:
        ensure_file(BIGWIG_SUMMARY_URL, path)
        path.chmod(0o755)
    return path


def query_gerp(url: str, chrom: str, start: int, end: int, bins: int) -> np.ndarray:
    tool = ensure_bigwig_summary()
    query_chrom = chrom.removeprefix("chr") if GERP_CHROM_MODE == "no_chr_prefix" else chrom
    result = subprocess.run(
        [str(tool), "-type=mean", url, query_chrom, str(start), str(end), str(bins)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    tokens = result.stdout.strip().split()
    if len(tokens) != bins:
        raise RuntimeError(f"GERP++ returned {len(tokens)} bins, expected {bins}: {result.stderr.strip()}")
    return np.asarray([np.nan if token in {"n/a", "nan", "NaN"} else float(token) for token in tokens], dtype=np.float64)


def query_linsight(url: str, index_path: Path, chrom: str, start: int, end: int, bins: int) -> np.ndarray:
    try:
        import pysam
    except ImportError as exc:  # pragma: no cover - dependency is checked in the documented environment
        raise RuntimeError("pysam is required to query the GRCh38 LINSIGHT resource") from exc

    sums = np.zeros(bins, dtype=np.float64)
    weights = np.zeros(bins, dtype=np.float64)
    bin_width = (end - start) / bins
    with pysam.TabixFile(url, index=str(index_path)) as tabix:
        for raw in tabix.fetch(chrom, start, end):
            fields = raw.split("\t")
            if len(fields) < 4:
                continue
            row_start = max(start, int(fields[1]))
            row_end = min(end, int(fields[2]))
            if row_start >= row_end:
                continue
            try:
                value = float(fields[3])
            except ValueError:
                continue
            first = max(0, int((row_start - start) / bin_width))
            last = min(bins - 1, int(np.ceil((row_end - start) / bin_width) - 1))
            for index in range(first, last + 1):
                bin_start = max(row_start, int(round(start + index * bin_width)))
                bin_end = min(row_end, int(round(start + (index + 1) * bin_width)))
                overlap = max(0, bin_end - bin_start)
                if overlap:
                    sums[index] += value * overlap
                    weights[index] += overlap
    values = np.full(bins, np.nan, dtype=np.float64)
    np.divide(sums, weights, out=values, where=weights > 0)
    return values


def prepare_reference_score_caches(root: Path, cache_dir: Path, refresh: bool = False) -> dict[str, dict[str, Path | str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "GRCh38_LinSight.bed.gz.csi"
    ensure_file(LINSIGHT_INDEX_URL, index_path)
    specs = {
        "g": {"chrom": "chr14", "start": 33_000_000, "end": 34_000_000, "bins": 800},
        "h": {"chrom": "chr19", "start": 345_000, "end": 485_000, "bins": 700},
    }
    outputs: dict[str, dict[str, Path | str]] = {}
    for panel, spec in specs.items():
        gerp_path = cache_dir / f"{panel}_GERP_GRCh38_binned_track.tsv"
        linsight_path = cache_dir / f"{panel}_LINSIGHT_GRCh38_binned_track.tsv"
        if refresh or not gerp_path.exists():
            values = query_gerp(GERP_URL, spec["chrom"], spec["start"], spec["end"], spec["bins"])
            write_track_tsv(gerp_path, spec["chrom"], spec["start"], spec["end"], values, GERP_URL, "GERP++ mean per equal-width bin")
        if refresh or not linsight_path.exists():
            values = query_linsight(LINSIGHT_URL, index_path, spec["chrom"], spec["start"], spec["end"], spec["bins"])
            write_track_tsv(linsight_path, spec["chrom"], spec["start"], spec["end"], values, LINSIGHT_URL, "LINSIGHT length-weighted mean per equal-width bin")
        outputs[panel] = {
            "gerp": gerp_path,
            "linsight": linsight_path,
            "linsight_index": index_path,
        }
    return outputs


def read_track_tsv(path: Path | None) -> dict[str, np.ndarray | str] | None:
    if path is None or not path.exists():
        return None
    x: list[float] = []
    values: list[float] = []
    source = ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                start = float(row["start"])
                end = float(row["end"])
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            x.append((start + end) / 2)
            values.append(value)
            source = row.get("source", source)
    if not x:
        return None
    return {"x": np.asarray(x), "values": np.asarray(values), "source": source, "path": str(path)}


def collect_har(har: ModuleType) -> dict[str, object]:
    payload = np.load(har.INPUT_NPZ, allow_pickle=False)
    chrom = str(payload["chromosome"][0])
    centers = payload["center"].astype(np.int64)
    raw = payload["raw_scores"].astype(np.float32)
    smooth = payload["smoothed_scores"].astype(np.float32)
    region_start = int(payload["start"].min())
    region_end = int(payload["end"].max())
    valid = centers >= region_start
    overlap = json.loads(har.OVERLAP_JSON.read_text(encoding="utf-8"))
    peaks = har.classify_peaks(har.load_peaks(har.PEAKS_TSV), overlap["peak_rows"])
    return {
        "panel": "g",
        "chrom": chrom,
        "start": region_start,
        "end": region_end,
        "x": centers[valid],
        "raw": raw[valid],
        "smooth": smooth[valid],
        "peaks": peaks,
        "known": har.load_known_hars(har.RAW_HAR_CSV, chrom, region_start, region_end),
        "genes": har.load_genes(har.REFGENE_GZ, chrom, region_start, region_end),
        "ccres": har.load_ccres(har.CCRE_BB, chrom, region_start, region_end),
        "h3k27ac": har.load_bigwig(har.H3K27AC_BW, chrom, region_start, region_end, len(centers))[valid],
        "phastcons": har.load_bigwig(har.PHASTCONS_BW, chrom, region_start, region_end, len(centers))[valid],
        "cadd": har.load_cadd_binned_track(har.CADD_TRACK_TSV),
        "phylop": har.load_phylop_json_track(har.PHYLOP_JSON, region_start, region_end, len(centers)),
        "gold_recovery": f"{overlap['num_gold_intervals_hit']}/{overlap['num_gold_intervals']}",
        "peak_count": len(peaks),
    }


def collect_haqer(haqer: ModuleType) -> dict[str, object]:
    scores = haqer.load_scores()
    peaks = haqer.load_peaks()
    known = haqer.load_haqers()
    categories = haqer.categorize_peaks(peaks, known)
    return {
        "panel": "h",
        "chrom": haqer.CHROM,
        "start": haqer.REGION_START,
        "end": haqer.REGION_END,
        "x": np.asarray(scores["x"], dtype=np.float64),
        "raw": np.asarray(scores["raw"], dtype=np.float64),
        "smooth": np.asarray(scores["smooth"], dtype=np.float64),
        "peaks": peaks,
        "categories": categories,
        "known": known,
        "genes": haqer.load_genes(),
        "ccres": haqer.load_ccres(),
        "h3k27ac": haqer.load_bigwig(haqer.H3K27AC_BW, len(scores["x"])),
        "phastcons": haqer.load_bigwig(haqer.PHASTCONS_BW, len(scores["x"])),
        "cadd": haqer.load_cadd_binned_track(),
        "phylop": haqer.load_phylop_json_track(len(scores["x"])),
        "direct_recovery": f"{sum(category == 'overlaps_gold' for category in categories)}/{len(known)}",
        "peak_count": len(peaks),
    }


def array_track(x: np.ndarray, values: np.ndarray, color: str, label: str, axis, baseline: float = 0.0, fill_alpha: float = 0.20) -> None:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return
    y = values[finite]
    xx = np.asarray(x, dtype=np.float64)[finite]
    axis.plot(xx, y, color=color, lw=0.65, alpha=0.95)
    axis.fill_between(xx, y, baseline, color=color, alpha=fill_alpha, linewidth=0)
    axis.set_ylabel(label)
    lower = min(float(np.nanmin(y)), baseline)
    upper = max(float(np.nanmax(y)), baseline)
    pad = max((upper - lower) * 0.12, 0.05)
    axis.set_ylim(lower - pad, upper + pad)


def binned_track(axis, track: dict[str, np.ndarray | str] | None, color: str, label: str, baseline: float = 0.0) -> None:
    if track is None:
        axis.set_ylabel(label)
        axis.text(0.5, 0.5, "track not supplied", transform=axis.transAxes, ha="center", va="center", color=PALETTE["muted"], fontsize=6.5)
        axis.set_ylim(0, 1)
        return
    array_track(np.asarray(track["x"]), np.asarray(track["values"]), color, label, axis, baseline=baseline, fill_alpha=0.22)


def ccre_color(ccre_class: str) -> str:
    label = ccre_class.lower()
    if "ctcf" in label:
        return TRACK_COLORS["ccre_ctcf"]
    if "pls" in label or "promoter" in label:
        return TRACK_COLORS["ccre_promoter"]
    if "els" in label or "enhancer" in label:
        return TRACK_COLORS["ccre_enhancer"]
    return TRACK_COLORS["ccre_other"]


def draw_ccres(axis, rows: list[dict[str, object]], region_start: int, region_end: int) -> None:
    axis.set_xlim(region_start, region_end)
    for row in rows:
        start = max(region_start, int(row["start"]))
        end = min(region_end, int(row["end"]))
        if start >= end:
            continue
        axis.add_patch(
            patches.Rectangle(
                (start, 0.25),
                end - start,
                0.5,
                facecolor=ccre_color(str(row.get("class", "cCRE"))),
                edgecolor="none",
                alpha=0.92,
            )
        )
    axis.set_ylim(0, 1)
    axis.set_ylabel("cCRE")


def category_color(category: str) -> str:
    return {
        "overlaps_gold": TRACK_COLORS["peak_hit"],
        "near_known_har": TRACK_COLORS["peak_near"],
        "other_non_gold": TRACK_COLORS["peak_other"],
    }.get(category, TRACK_COLORS["peak_other"])


def draw_peak_calls(axis, panel: dict[str, object]) -> None:
    start = int(panel["start"])
    end = int(panel["end"])
    peaks = list(panel["peaks"])
    categories = list(panel.get("categories", [str(row.get("category", "other_non_gold")) for row in peaks]))
    axis.set_xlim(start, end)
    for peak, category in zip(peaks, categories):
        peak_start = int(peak.get("peak_start", peak.get("start", start)))
        peak_end = int(peak.get("peak_end", peak.get("end", end)))
        summit = int(peak.get("summit_center", peak.get("summit", (peak_start + peak_end) // 2)))
        color = category_color(str(category))
        axis.vlines(summit, 0.05, 0.92, color=color, lw=0.55, linestyles=(0, (4, 4)), alpha=0.9)
        axis.add_patch(patches.Rectangle((peak_start, 0.06), peak_end - peak_start, 0.19, facecolor=color, edgecolor="none", alpha=0.96))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Peak calls")


def draw_known(axis, panel: dict[str, object]) -> None:
    start = int(panel["start"])
    end = int(panel["end"])
    axis.set_xlim(start, end)
    for row in list(panel["known"]):
        left = int(row["start"])
        right = int(row["end"])
        label = str(row.get("label", row.get("name", "")))
        color = TRACK_COLORS["known_highlight"] if label not in {"HAR", ""} else TRACK_COLORS["known"]
        axis.vlines((left + right) / 2, 0.08, 0.88, color=color, lw=0.72)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Known HARs" if panel["panel"] == "g" else "Known HAQERs")


def assign_gene_levels(genes: list[dict[str, object]], max_levels: int = 3) -> list[int]:
    levels: list[int] = []
    occupied: list[list[tuple[int, int]]] = [[] for _ in range(max_levels)]
    for gene in sorted(genes, key=lambda row: (int(row["start"]), int(row["end"]))):
        start = int(gene["start"])
        end = int(gene["end"])
        level = 0
        while level < max_levels and any(start < other_end and other_start < end for other_start, other_end in occupied[level]):
            level += 1
        level = min(level, max_levels - 1)
        occupied[level].append((start, end))
        levels.append(level)
    index = {id(gene): level for gene, level in zip(sorted(genes, key=lambda row: (int(row["start"]), int(row["end"]))), levels)}
    return [index[id(gene)] for gene in genes]


def draw_genes(axis, panel: dict[str, object]) -> None:
    start = int(panel["start"])
    end = int(panel["end"])
    genes = list(panel["genes"])
    levels = assign_gene_levels(genes)
    max_level = max(levels, default=0)
    for gene, level in zip(genes, levels):
        left = max(start, int(gene["start"]))
        right = min(end, int(gene["end"]))
        if left >= right:
            continue
        name = str(gene.get("name", gene.get("symbol", "gene")))
        highlight = name in {"NPAS3", "SHC2", "ODF3L2"}
        color = TRACK_COLORS["gene"] if highlight else TRACK_COLORS["gene_other"]
        y = max_level - level + 0.35
        axis.plot([left, right], [y, y], color=color, lw=1.4 if highlight else 0.75, alpha=0.95)
        strand = str(gene.get("strand", "+"))
        arrow_x = right if strand == "+" else left
        axis.scatter([arrow_x], [y], marker=">" if strand == "+" else "<", s=12 if highlight else 7, color=color, zorder=3)
        if highlight:
            axis.text(left, y + 0.22, name, color=color, fontsize=8.5, fontweight="bold", style="italic", va="bottom")
    axis.set_xlim(start, end)
    axis.set_ylim(0, max_level + 0.85)
    axis.set_yticks([])
    axis.set_ylabel("RefSeq genes")


def setup_axis(axis, region_start: int, region_end: int, show_x: bool = False, top_line: bool = False) -> None:
    axis.set_facecolor(PALETTE["paper"])
    axis.set_xlim(region_start, region_end)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(PALETTE["edge"])
    axis.spines["bottom"].set_color(PALETTE["edge"])
    axis.spines["top"].set_visible(top_line)
    axis.spines["top"].set_color(PALETTE["edge"])
    axis.spines["top"].set_linewidth(0.55)
    axis.spines["left"].set_linewidth(0.55)
    axis.spines["bottom"].set_linewidth(0.55)
    axis.tick_params(axis="both", colors=PALETTE["navy"], labelsize=7, width=0.45, length=2.5, pad=2)
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.35, alpha=0.6)
    if not show_x:
        axis.tick_params(axis="x", labelbottom=False)


def set_track_label(axis, label: str) -> None:
    axis.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=0, color=PALETTE["text"], fontsize=7.4)
    axis.yaxis.set_label_coords(-0.075, 0.5)


def group_label(axis, label: str) -> None:
    axis.text(
        -0.235,
        0.5,
        label,
        transform=axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["navy"],
        clip_on=False,
    )


def format_position(region_end: int) -> FuncFormatter:
    if region_end >= 1_000_000:
        return FuncFormatter(lambda value, _pos: f"{value / 1_000_000:.1f}")
    return FuncFormatter(lambda value, _pos: f"{value / 1_000:.0f}")


def draw_panel(fig, subspec, panel: dict[str, object], references: dict[str, dict[str, np.ndarray | str] | None], letter: str) -> None:
    row_specs: list[tuple[str, str, str]] = [
        ("MODEL", "score", "hEvo scoring-head score"),
        ("MODEL", "peaks", "Peak calls"),
        ("MODEL", "known", "Known HARs" if letter == "g" else "Known HAQERs"),
        ("REG.", "h3k27ac", "H3K27ac"),
        ("REG.", "ccre", "cCRE"),
        ("EVOL.", "phastcons", "phastCons100way"),
        ("EVOL.", "phylop", "phyloP100way"),
        ("REF. SCORES", "cadd", "CADD-PHRED"),
    ]
    if references.get("gerp") is not None:
        row_specs.append(("REF. SCORES", "gerp", "GERP++ RS"))
    if references.get("linsight") is not None:
        row_specs.append(("REF. SCORES", "linsight", "LINSIGHT"))
    row_specs.append(("GENOME", "genes", "RefSeq genes"))
    height_map = {"score": 2.0, "peaks": 0.40, "known": 0.40, "h3k27ac": 0.62, "ccre": 0.46, "phastcons": 0.62, "phylop": 0.62, "cadd": 0.62, "gerp": 0.62, "linsight": 0.62, "genes": 0.92}
    grid = GridSpecFromSubplotSpec(
        len(row_specs),
        1,
        subplot_spec=subspec,
        height_ratios=[height_map[key] for _, key, _ in row_specs],
        hspace=0.0,
    )
    axes = []
    for index in range(len(row_specs)):
        axes.append(fig.add_subplot(grid[index], sharex=axes[0] if axes else None))
    start = int(panel["start"])
    end = int(panel["end"])
    previous_group = None
    for axis, (group, key, label) in zip(axes, row_specs):
        setup_axis(axis, start, end, show_x=key == "genes", top_line=previous_group is not None and group != previous_group)
        if group != previous_group:
            group_label(axis, group)
            previous_group = group
        if key == "score":
            x = np.asarray(panel["x"], dtype=np.float64)
            raw = np.asarray(panel["raw"], dtype=np.float64)
            smooth = np.asarray(panel["smooth"], dtype=np.float64)
            axis.plot(x, raw, color=TRACK_COLORS["raw"], lw=0.30, alpha=0.25, label="raw score")
            axis.plot(x, smooth, color=TRACK_COLORS["score"], lw=0.95, alpha=0.98, label="hEvo score")
            categories = list(panel.get("categories", [str(row.get("category", "other_non_gold")) for row in panel["peaks"]]))
            for peak, category in zip(list(panel["peaks"]), categories):
                peak_start = int(peak.get("peak_start", peak.get("start", start)))
                peak_end = int(peak.get("peak_end", peak.get("end", end)))
                summit = int(peak.get("summit_center", peak.get("summit", (peak_start + peak_end) // 2)))
                color = category_color(str(category))
                axis.axvspan(peak_start, peak_end, color=color, alpha=0.10 if color != TRACK_COLORS["peak_other"] else 0.07, linewidth=0)
                axis.axvline(summit, color=color, lw=0.45, alpha=0.8)
                score = float(peak.get("smoothed_score", peak.get("score", 0.0)))
                axis.scatter([summit], [score], color=color, s=7, zorder=4)
            upper = max(float(np.nanmax(smooth)) * 1.16, 0.45)
            axis.set_ylim(0, upper)
            axis.set_ylabel(label)
            axis.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=6.5, ncol=2, handlelength=1.8, columnspacing=1.2)
            meta = f"gold recovered: {panel['gold_recovery']}" if letter == "g" else f"direct HAQER recovery: {panel['direct_recovery']}"
            axis.text(0.995, 1.02, f"{meta}  |  called peaks: {panel['peak_count']}", transform=axis.transAxes, ha="right", va="bottom", fontsize=6.5, color=PALETTE["navy"])
            axis.set_title("HAR | NPAS3 | hg38" if letter == "g" else "HAQER | chr19p13.3 | hg38", fontsize=11, fontweight="bold", color=PALETTE["navy"], pad=9)
            axis.text(-0.285, 1.18, letter, transform=axis.transAxes, fontsize=15, fontweight="bold", color=PALETTE["navy"], ha="left", va="bottom", clip_on=False)
        elif key == "peaks":
            draw_peak_calls(axis, panel)
        elif key == "known":
            draw_known(axis, panel)
        elif key == "h3k27ac":
            array_track(np.asarray(panel["x"]), np.asarray(panel["h3k27ac"]), TRACK_COLORS["h3k27ac"], label, axis, baseline=0.0, fill_alpha=0.20)
        elif key == "ccre":
            draw_ccres(axis, list(panel["ccres"]), start, end)
        elif key == "phastcons":
            array_track(np.asarray(panel["x"]), np.asarray(panel["phastcons"]), TRACK_COLORS["phastcons"], label, axis, baseline=0.0, fill_alpha=0.18)
        elif key == "phylop":
            track = panel["phylop"]
            array_track(np.asarray(track["x"]), np.asarray(track["values"]), TRACK_COLORS["phylop"], label, axis, baseline=0.0, fill_alpha=0.16)
            axis.axhline(0, color=PALETTE["edge"], lw=0.45)
        elif key == "cadd":
            rows = list(panel["cadd"])
            x = np.asarray([int(row["position"]) for row in rows], dtype=np.float64) if rows else np.asarray([])
            values = np.asarray([float(row["max_phred"]) for row in rows], dtype=np.float64) if rows else np.asarray([])
            array_track(x, values, TRACK_COLORS["cadd"], label, axis, baseline=0.0, fill_alpha=0.18)
            axis.axhline(20, color=PALETTE["muted"], lw=0.45, ls=(0, (2, 2)))
        elif key == "gerp":
            binned_track(axis, references["gerp"], TRACK_COLORS["gerp"], label, baseline=0.0)
            axis.axhline(0, color=PALETTE["edge"], lw=0.45)
        elif key == "linsight":
            binned_track(axis, references["linsight"], TRACK_COLORS["linsight"], label, baseline=0.0)
        elif key == "genes":
            draw_genes(axis, panel)
            axis.set_xlabel(f"{panel['chrom']} position ({'Mb' if end >= 1_000_000 else 'kb'})", fontsize=8.5, color=PALETTE["text"])
            axis.xaxis.set_major_formatter(format_position(end))
        set_track_label(axis, label)
    for axis in axes[:-1]:
        axis.tick_params(axis="x", labelbottom=False)


def render_combined(panels: list[dict[str, object]], references: dict[str, dict[str, dict[str, np.ndarray | str] | None]], output_dir: Path) -> tuple[Path, Path]:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.55,
        }
    )
    figsize = (14.5, 8.2)
    gap_in = 1.5 / 25.4
    top_margin = 0.965
    bottom_margin = 0.07
    available_height_in = figsize[1] * (top_margin - bottom_margin)
    spacer_ratio = (2 * gap_in) / (available_height_in - gap_in)
    fig = plt.figure(figsize=figsize, facecolor=PALETTE["paper"])
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, spacer_ratio, 1.0], hspace=0.0)
    draw_panel(fig, outer[0], panels[0], references["g"], "g")
    draw_panel(fig, outer[2], panels[1], references["h"], "h")
    fig.subplots_adjust(left=0.18, right=0.985, top=top_margin, bottom=bottom_margin, hspace=0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "HAR_HAQER_g_h_integrated_mockup"
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_base.with_suffix(".png"), output_base.with_suffix(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the compact g/h HAR-HAQER main-figure mockup.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--fetch-reference-scores", action="store_true")
    parser.add_argument("--refresh-reference-scores", action="store_true")
    parser.add_argument("--gerp-g-tsv", type=Path, default=None)
    parser.add_argument("--gerp-h-tsv", type=Path, default=None)
    parser.add_argument("--linsight-g-tsv", type=Path, default=None)
    parser.add_argument("--linsight-h-tsv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    output_dir = args.output_dir or root / "results/final_har_haqer_panel"
    cache_dir = args.cache_dir or output_dir / "reference_score_cache"
    har, haqer = configure_sources(root)
    if args.fetch_reference_scores or args.refresh_reference_scores:
        prepared = prepare_reference_score_caches(root, cache_dir, refresh=args.refresh_reference_scores)
        gerp_g = Path(prepared["g"]["gerp"])
        gerp_h = Path(prepared["h"]["gerp"])
        linsight_g = Path(prepared["g"]["linsight"])
        linsight_h = Path(prepared["h"]["linsight"])
    else:
        gerp_g = args.gerp_g_tsv or cache_dir / "g_GERP_GRCh38_binned_track.tsv"
        gerp_h = args.gerp_h_tsv or cache_dir / "h_GERP_GRCh38_binned_track.tsv"
        linsight_g = args.linsight_g_tsv or cache_dir / "g_LINSIGHT_GRCh38_binned_track.tsv"
        linsight_h = args.linsight_h_tsv or cache_dir / "h_LINSIGHT_GRCh38_binned_track.tsv"

    panels = [collect_har(har), collect_haqer(haqer)]
    references = {
        "g": {"gerp": read_track_tsv(gerp_g), "linsight": read_track_tsv(linsight_g)},
        "h": {"gerp": read_track_tsv(gerp_h), "linsight": read_track_tsv(linsight_h)},
    }
    png_path, pdf_path = render_combined(panels, references, output_dir)
    summary = {
        "style": "compact_g_h_mockup",
        "panel_order": ["g", "h"],
        "inter_panel_gap_mm": 1.5,
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "available_tracks": [
            "hEvo scoring-head score",
            "peak calls",
            "known HAR/HAQER intervals",
            "H3K27ac",
            "cCRE",
            "phastCons100way",
            "phyloP100way",
            "CADD-PHRED",
            "GERP++ RS",
            "LINSIGHT",
            "RefSeq genes",
        ],
        "reference_score_sources": {
            "GERP++ RS": {"source": GERP_URL, "assembly": "GRCh38", "chromosome_naming": GERP_CHROM_MODE},
            "LINSIGHT": {
                "source": LINSIGHT_URL,
                "index": LINSIGHT_INDEX_URL,
                "assembly": "GRCh38",
                "aggregation": "length-weighted mean per equal-width bin",
            },
        },
        "panels": {
            panel["panel"]: {
                "region": f"{panel['chrom']}:{panel['start']}-{panel['end']}",
                "called_peak_count": panel["peak_count"],
                "score_summary": panel.get("gold_recovery", panel.get("direct_recovery")),
                "GERP_cache": str(gerp_g if panel["panel"] == "g" else gerp_h),
                "LINSIGHT_cache": str(linsight_g if panel["panel"] == "g" else linsight_h),
            }
            for panel in panels
        },
        "missing_tracks": [
            name
            for name, tracks in (("g", references["g"]), ("h", references["h"]))
            for track_name, track in tracks.items()
            if track is None
        ],
    }
    (output_dir / "HAR_HAQER_g_h_integrated_mockup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
