"""Fig. 4e: selected held-out HAQER score and peak-call comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve()
BENCHMARK = HERE.parents[3]
TABLE_ROOT = BENCHMARK / "tables/fig4e_haqer_tracks"
SOURCE_TABLE = TABLE_ROOT / "Fig4e_track_source_data.csv"
SEGMENTS_JSON = TABLE_ROOT / "selected_segments.json"
REFGENE = TABLE_ROOT / "annotations/hg38_refGene.txt.gz"
ANNOTATION_FILES = [
    TABLE_ROOT / "annotations/signals.tsv",
    TABLE_ROOT / "annotations/ccres.json",
    TABLE_ROOT / "annotations/sources.json",
]
COLORS = {
    "hevo": "#B35C1E", "gpn": "#0E7C86", "gold": "#234F6C",
    "phastcons": "#648B6C", "genes": "#234F6C", "h3k27ac": "#9D762F",
    "ape_divergence": "#8975A9", "ccres": "#84668E",
}
LITERATURE_TAGS = {"HAQER0218": "†"}
SEGMENT_GAP_BP = 800
REFERENCE_LOCUS_ORDER = [
    "21:33.896", "20:61.218", "13:110.314", "17:56.793", "18:24.841",
    "22:49.596", "13:34.909", "22:48.044", "16:83.511", "15:75.240",
    "16:89.069", "20:23.156",
]


def _read_source():
    if not SOURCE_TABLE.is_file():
        raise FileNotFoundError(
            f"Missing Fig. 4e source data {SOURCE_TABLE}; run export_fig4e_gpn_source_data.py first"
        )
    with SOURCE_TABLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata_row = next(row for row in rows if row.get("track") == "__panel__")
    metadata = json.loads(metadata_row["payload_json"])
    keyed = {}
    for row in rows:
        keyed.setdefault(row["track"], []).append(row)
    return metadata, keyed


def source_files():
    # The checked-in track table is the renderer's complete data input. Keep
    # its selection and annotation companions in the figure provenance list;
    # external run reports are not needed to render this panel.
    return [SOURCE_TABLE, SEGMENTS_JSON, *ANNOTATION_FILES, REFGENE]


def _segment_rows(rows, index):
    return [row for row in rows if row.get("segment") not in (None, "") and int(row["segment"]) == index]


def _draw_signal(ax, segment, rows, top, bottom, color, *, baseline_ratio=0.10):
    values = [row for row in _segment_rows(rows, segment["index"]) if row.get("value") not in (None, "", "None")]
    if not values:
        return
    positions = np.asarray([float(row["position"]) for row in values])
    signal = np.asarray([float(row["value"]) for row in values])
    finite = np.isfinite(signal)
    if not finite.any():
        return
    x = segment["offset"] + positions[finite] - segment["start"]
    high = float(np.nanpercentile(signal[finite], 98))
    if high <= 0:
        high = float(np.nanmax(signal[finite]))
    if high <= 0:
        high = 1.0
    scaled = np.clip(signal[finite] / high, 0.0, 1.0)
    baseline = bottom - baseline_ratio * (bottom - top)
    display_y = baseline - scaled * (baseline - top - 0.10)
    ax.fill_between(x, baseline, display_y, color=color, alpha=0.17, lw=0)
    ax.plot(x, display_y, color=color, lw=0.46)


def build_panel(root=None, *, figure, rect=None, slot=None, standalone=False, reference_style=False):
    del root, standalone
    metadata, records = _read_source()
    segments = [dict(segment) for segment in metadata["segments"]]
    if reference_style:
        order = {locus: index for index, locus in enumerate(REFERENCE_LOCUS_ORDER)}
        segments.sort(
            key=lambda segment: order[
                f"{str(segment['chrom']).removeprefix('chr')}:"
                f"{(int(segment['gold_start']) + int(segment['gold_end'])) / 2e6:.3f}"
            ]
        )
        offset = 0
        for segment in segments:
            segment["offset"] = offset
            segment["display_start"] = offset
            segment["display_end"] = offset + int(segment["end"]) - int(segment["start"])
            offset = int(segment["display_end"]) + SEGMENT_GAP_BP
        display_end = int(segments[-1]["display_end"])
    else:
        display_end = max(segment["display_end"] for segment in segments)
    caller = metadata["caller"]
    hevo_cut = float(caller["hevo"]["score"])
    gpn_cut = float(metadata["caller_parameters"]["gpn"][2])

    if slot is not None:
        ax = figure.add_subplot(slot)
    else:
        ax = figure.add_axes(rect or [0.13, 0.15, 0.85, 0.72])

    if reference_style:
        tracks = [
            ("hevo", "hEVO-large score", 1.00), ("gpn", "GPN-Star-M score", 1.00),
            ("hevo_peaks", "hEVO calls", 0.48), ("known", "Known HAQER", 0.46),
            ("ape_divergence", "Ape divergence", 0.56),
            ("ccres", "ENCODE cCREs", 0.48), ("genes", "RefSeq genes", 0.84),
        ]
    else:
        tracks = [
            ("hevo", "hEVO-large score", 1.00), ("gpn", "GPN-Star-M score", 1.00),
            ("hevo_peaks", "hEVO calls", 0.48), ("gpn_peaks", "GPN calls", 0.48),
            ("known", "Known HAQER", 0.46), ("ccres", "ENCODE cCREs", 0.48),
            ("h3k27ac", "H3K27ac H1-hESC", 0.56),
            ("ape_divergence", "Ape divergence", 0.56),
            ("phastcons", "phastCons100way", 0.56), ("genes", "RefSeq genes", 0.84),
        ]
    total = sum(track[2] for track in tracks)
    ax.set_xlim(0, display_end)
    ax.set_ylim(total + 0.12, -0.12)
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.55)
    position = ax.get_position()
    y = 0.0
    row_bounds = {}
    label_artists = []
    for key, label, height in tracks:
        top, bottom = y, y + height
        row_bounds[key] = (top, bottom)
        y = bottom
        ax.axhline(bottom, color="#93A2AD", lw=0.34, zorder=1)
        label_artist = figure.text(
            position.x0 - 0.008,
            position.y0 + (total + 0.12 - (top + height / 2)) / (total + 0.24) * position.height,
            label, ha="right", va="center", fontsize=6,
        )
        label_artists.append(label_artist)

        if key in ("hevo", "gpn"):
            color = COLORS[key]
            baseline = bottom - 0.10
            amplitude = height - 0.20
            transform = lambda vals: baseline - np.asarray(vals) * amplitude
            cutoff = hevo_cut if key == "hevo" else gpn_cut
            if not reference_style:
                label_artist.set_text(f"{label} (cut {cutoff:.2f})")
            ax.hlines(transform(cutoff), 0, display_end, color=color, lw=0.45, linestyles=(0, (2, 2)))
            ax.text(display_end - 200, transform(cutoff) - 0.025, f"{cutoff:g}",
                    ha="right", va="bottom", fontsize=5.0, color=color)
            for segment in segments:
                series = _segment_rows(records[key], segment["index"])
                if not series:
                    continue
                x = segment["offset"] + np.asarray([int(row["position"]) for row in series]) - segment["start"]
                values = np.asarray([float(row["value"]) for row in series])
                display_y = transform(values)
                ax.fill_between(x, baseline, display_y, color=color, alpha=0.12, lw=0)
                ax.plot(x, display_y, color=color, lw=0.78, zorder=3)
            continue

        if key in ("hevo_peaks", "gpn_peaks"):
            model = "hevo" if key == "hevo_peaks" else "gpn"
            color = COLORS[model]
            for segment in segments:
                for peak_row in _segment_rows(records.get(key, []), segment["index"]):
                    payload = json.loads(peak_row["payload_json"])
                    left = segment["offset"] + max(segment["start"], int(peak_row["start"])) - segment["start"]
                    right = segment["offset"] + min(segment["end"], int(peak_row["end"])) - segment["start"]
                    center_y = (top + bottom) / 2
                    if right > left:
                        ax.add_patch(Rectangle((left, center_y - 0.06), right-left, 0.12,
                                               facecolor=color, alpha=0.22, lw=0, zorder=2))
                    summit = segment["offset"] + int(peak_row["summit"]) - segment["start"]
                    ax.scatter(summit, center_y, marker="v", s=15, color=color, zorder=4)
            continue

        if key == "known":
            for segment in segments:
                known_row = next(row for row in _segment_rows(records["known"], segment["index"]))
                left = segment["offset"] + int(known_row["start"]) - segment["start"]
                right = segment["offset"] + int(known_row["end"]) - segment["start"]
                ax.add_patch(Rectangle((left, top + 0.12), right-left, 0.26,
                                       facecolor=COLORS["gold"], alpha=0.92, lw=0, zorder=3))
                ax.vlines([left, right], top, bottom, color=COLORS["gold"], lw=0.48,
                          linestyle=(0, (1, 2)), alpha=0.8, zorder=2)
            continue

        if key in ("h3k27ac", "ape_divergence", "phastcons"):
            for segment in segments:
                _draw_signal(ax, segment, records[key], top, bottom, COLORS[key])
            continue

        if key == "ccres":
            for segment in segments:
                for row in _segment_rows(records[key], segment["index"]):
                    left = max(segment["start"], int(row["start"]))
                    right = min(segment["end"], int(row["end"]))
                    if left < right:
                        mid = (top + bottom) / 2
                        ax.add_patch(Rectangle(
                            (segment["offset"] + left - segment["start"], mid - 0.13),
                            right-left, 0.26, facecolor=COLORS[key], edgecolor=COLORS[key],
                            alpha=0.68, lw=0.12, zorder=3,
                        ))
            continue

        for segment in segments:
            for gene_index, row in enumerate(_segment_rows(records["genes"], segment["index"])):
                gene = json.loads(row["payload_json"])
                lane = gene_index % 2
                center_y = top + 0.54 + lane * 0.34
                left = segment["offset"] + max(segment["start"], int(row["start"])) - segment["start"]
                right = segment["offset"] + min(segment["end"], int(row["end"])) - segment["start"]
                ax.hlines(center_y, left, right, color=COLORS["genes"], lw=0.6)
                for exon_start, exon_end in gene["exons"]:
                    if int(exon_start) < segment["end"] and int(exon_end) > segment["start"]:
                        exon_left = segment["offset"] + max(int(exon_start), segment["start"]) - segment["start"]
                        exon_right = segment["offset"] + min(int(exon_end), segment["end"]) - segment["start"]
                        ax.add_patch(Rectangle((exon_left, center_y - 0.11), exon_right-exon_left,
                                               0.22, color=COLORS["genes"], lw=0))
                strand = " →" if gene["strand"] == "+" else " ←"
                ax.text((left + right) / 2, top + 0.12 + lane * 0.34,
                        gene["name"] + strand, ha="center", va="center", fontsize=4.6,
                        color=COLORS["genes"])

    ticks, tick_labels = [], []
    for index, segment in enumerate(segments):
        offset = segment["offset"]
        width = segment["end"] - segment["start"]
        ticks.append(offset + width / 2)
        if reference_style:
            tick_labels.append(f"{segment['chrom'][3:]}:{(segment['gold_start'] + segment['gold_end']) / 2e6:.3f}")
        else:
            cohort = "T" if segment["split"] == "test" else "V"
            marker = LITERATURE_TAGS.get(segment["id"], "")
            tick_labels.append(
                f"{segment['id'][-4:]}{marker} {cohort}\n"
                f"{segment['chrom'][3:]}:{(segment['gold_start'] + segment['gold_end']) / 2e6:.3f}"
            )
        if index < len(segments) - 1:
            center = offset + width + SEGMENT_GAP_BP / 2
            ax.axvline(center, color="#697780", lw=0.55, linestyle=(0, (2, 2)), alpha=0.8)
            ax.text(center, total + 0.03, "//", ha="center", va="center",
                    fontsize=5, color="#697780", clip_on=False)
    # Real bp distances within each interval; the inter-interval gaps are artificial.
    first = segments[0]
    bar_start = first["offset"] + 150
    ax.plot([bar_start, bar_start + 1000], [-0.20, -0.20], color="#222222",
            lw=0.8, clip_on=False)
    ax.vlines([bar_start, bar_start + 1000], -0.24, -0.16,
              color="#222222", lw=0.6, clip_on=False)
    ax.text(bar_start + 500, -0.28, "1 kb", ha="center", va="bottom",
            fontsize=4.8, clip_on=False)
    for segment in segments:
        midpoint = (segment["gold_start"] + segment["gold_end"]) // 2
        local_positions = [midpoint - 2000, midpoint, midpoint + 2000]
        for pos in local_positions:
            if segment["start"] <= pos <= segment["end"]:
                xx = segment["offset"] + pos - segment["start"]
                ax.plot([xx, xx], [total + 0.02, total + 0.10],
                        color="#222222", lw=0.45)
    ax.set_xticks(ticks)
    tick_fontsize = max(3.4, 4.4 - 0.06 * max(0, len(segments) - 7))
    ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
    ax.tick_params(axis="x", width=0.5, length=2, pad=2)
    ax.set_xlabel(
        "hg38 midpoint (Mb); ticks: -2, 0, +2 kb; // separate loci" if reference_style else
        "Compressed hg38 coordinates (Mb); // omitted; T=test, V=validation; common cutoff, separate calibration",
        fontsize=5.4 if not reference_style else 5.1, labelpad=2,
    )

    metrics = metadata["test_metrics"]
    hevo, gpn = metrics["hevo"], metrics["gpn"]
    title_text = (
        "Held-out HAQERs: hEVO summit hit; GPN no peak in displayed window; "
        "hEVO−GPN local score margin ≥0.15 | †COL4A2-near"
    )
    metric_text = (
        f"Test (3-seed mean; {metrics['n_common_haqers']} HAQERs/{metrics['n_test_windows']:,} windows): "
        f"AUPRC {hevo['window_auprc']:.3f}/{gpn['window_auprc']:.3f} · "
        f"AUROC {hevo['window_auroc']:.3f}/{gpn['window_auroc']:.3f} · "
        f"center-top1 {hevo['center_top1_rate']:.3f}/{gpn['center_top1_rate']:.3f} · "
        f"summit P/R/F1 hEVO {hevo['summit_precision']:.3f}/{hevo['summit_recall']:.3f}/{hevo['summit_f1']:.3f}; "
        f"GPN {gpn['summit_precision']:.3f}/{gpn['summit_recall']:.3f}/{gpn['summit_f1']:.3f}; "
        f"shared score cutoff {hevo_cut:.2f}"
    )
    title = None
    metric_artist = None
    if not reference_style:
        title = figure.text(position.x0, position.y1 + 0.018, title_text,
                            ha="left", va="center", fontsize=7, fontweight="bold")
        metric_artist = figure.text(position.x0, position.y1 + 0.005, metric_text,
                                    ha="left", va="center", fontsize=5)
    ax._haqer_title = title
    ax._haqer_header_artists = [title] if title is not None else []
    ax._haqer_caption = title
    ax._haqer_metric_text = (metric_artist,) if metric_artist is not None else ()
    ax._haqer_label_artists = label_artists
    return ax
