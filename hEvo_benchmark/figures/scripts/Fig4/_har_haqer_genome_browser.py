"""Local helper for the HAR/HAQER formal genome-browser renderer.

The drawing implementation is kept as a package-local helper.
This helper supplies package-local copied inputs and keeps large public tracks
as explicitly declared external dependencies.  Missing tracks are never
converted to zeros.
"""

from __future__ import annotations

import importlib.util
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np



PACKAGE_ROOT = Path(__file__).resolve().parents[3]
TABLES_ROOT = PACKAGE_ROOT / "tables"
DATA_ROOT = TABLES_ROOT
S40_PATH = TABLES_ROOT / "S40_genomic_context_tracks_for_human_accelerated_and_quickly_evolved_regions.csv"
S40_MANIFEST_PATH = TABLES_ROOT / "har_haqer_track_source_manifest.csv"
RAW_TRACK_ROOT = TABLES_ROOT / "har_haqer_tracks"
UPSTREAM_ROOT = PACKAGE_ROOT / "figures" / "scripts" / "_backend" / "Fig4g_h"
WIDTH_MM = 180.0
HEIGHT_MM = 72.0
# The reference PDF places two full-height locus panels one above the other.
# 72 mm per panel plus a 1.5 mm inter-panel gap gives this compact stacked size.
COMBINED_HEIGHT_MM = 145.5
REFERENCE_FONT_SCALE = 0.78
# Keep the baseline above the tick-label/title stack so the x-axis remains
# readable in both the standalone 72 mm panel and the stacked preview.
REFERENCE_AXIS_BASELINE = 1.45
COMPARISON_REGIONS = {
    "g": (33_500_000, 33_800_000),
    "h": (345_000, 485_000),
}
COMPARISON_TRACK_KEYS = {"hevo", "evo2", "peaks", "known", "genes"}
EXAMPLE_SELECTION_RELATIVE = Path("manifest") / "example_region_selection.json"
FALLBACK_EXAMPLE_SELECTION = {
    "panel": "g",
    "chrom": "chr14",
    "known_start": 33_576_209,
    "known_end": 33_576_720,
    "known_label": "HAR_NPAS3_fallback",
    "hevo_summit": 33_576_576,
    "evo2_summit": 33_561_664,
    "hevo_summit_hit": True,
    "evo2_summit_hit": False,
    "context_start": 33_500_000,
    "context_end": 33_800_000,
    "focus_start": 33_550_000,
    "focus_end": 33_600_000,
    "peak_hit_definition": "summit_inside_known_interval",
}


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_renderer() -> ModuleType:
    prepare = _load_module(
        UPSTREAM_ROOT / "formal_panel" / "prepare_formal_tracks.py",
        "fig4gh_prepare_formal_tracks",
    )
    # The frozen formal renderer imports this module by its historical name.
    sys.modules["prepare_formal_tracks"] = prepare
    renderer = _load_module(
        UPSTREAM_ROOT / "formal_panel" / "plot_har_haqer_formal.py",
        "fig4gh_formal_renderer",
    )
    renderer.HAR_SCRIPT = UPSTREAM_ROOT / "har_source" / "beautify_har_npas3_optimized.py"
    renderer.HAQER_SCRIPT = UPSTREAM_ROOT / "haqer_source" / "plot_haqer_chr19_candidate.py"
    renderer.MOCKUP_SCRIPT = UPSTREAM_ROOT / "formal_panel" / "plot_har_haqer_mockup.py"
    return renderer


def _external_paths() -> dict[str, Path]:
    manifest_value = os.environ.get("HEVO_FIG4GH_EXTERNAL_MANIFEST")
    if not manifest_value:
        raise FileNotFoundError(
            "Set HEVO_FIG4GH_EXTERNAL_MANIFEST to an external track-input manifest "
            "when using the legacy full-track renderer."
        )
    manifest_path = Path(manifest_value)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(item["key"]): Path(str(item["path"])) for item in payload["dependencies"]}


def _source_args(data_root: Path) -> SimpleNamespace:
    external = _external_paths()
    g = data_root / "g"
    h = data_root / "h"
    # RefSeq is small enough to package with the comparison renderer.  The
    # large public signal tracks remain external, but the compact comparison
    # mode must be runnable on HPC without requiring those hidden tracks.
    local_refgene = data_root / "reference" / "hg38_refGene.txt.gz"
    return SimpleNamespace(
        style="teacher",
        root=PACKAGE_ROOT,
        output_dir=PACKAGE_ROOT / "figures" / "Fig4" / "g_h",
        track_dir=None,
        input_npz=g / "HAR_NPAS3_optimized_with_tracks.smoothed_scores.npz",
        peaks_tsv=g / "HAR_NPAS3_optimized_with_tracks.peaks.tsv",
        overlap_json=g / "HAR_NPAS3_optimized_with_tracks.overlap_summary.json",
        gold_csv=None,
        raw_har_csv=g / "HARs.CSV",
        haqer_bed=h / "haqer.hg38.bed",
        refgene_gz=local_refgene if local_refgene.is_file() else external["genes"],
        h3k27ac_bw=external["h3k27ac"],
        phastcons_bw=external["phastcons"],
        ccre_bb=external["ccre_cage"],
        cadd_position_tsv=None,
        phylop_position_tsv=None,
        cadd_track_tsv=g / "HAR_NPAS3_CADD_GRCh38_v1.7_binned_track.tsv",
        phylop_json=g / "tmp_phyloP_chr14_33000000_34000000.json",
        evo2_track_npz=g / "g_blocks_6_mlp_l3_tracks.npz",
        evo2_track_field="evo2_1b_raw_likelihood",
        evo2_peaks_tsv=g / "evo2_1b_hevo_protocol_merged.peaks.tsv",
        input_npz_h=h / "val_region_29_normal.smoothed_scores.npz",
        peaks_tsv_h=h / "val_region_29_normal.peaks.tsv",
        overlap_json_h=h / "HAQER_top1_chr19_0_1000000_normal_with_ucsc.overlap_summary.json",
        cadd_track_tsv_h=h / "HAQER_chr19_CADD_GRCh38_v1.7_binned_track.tsv",
        phylop_json_h=h / "tmp_phyloP_chr19_345000_485000.json",
        evo2_track_npz_h=h / "h_blocks_12_mlp_l3_tracks.npz",
        evo2_peaks_tsv_h=h / "evo2_1b_hevo_protocol_merged.peaks.tsv",
    )


def _localize_manifest_paths(manifest: dict, data_root: Path) -> dict:
    """Use copied panel files when rendering from a staged or SSHFS path."""

    for entry in manifest.get("tracks", []):
        panel_id = str(entry.get("panel", ""))
        path_value = entry.get("path")
        if panel_id not in {"g", "h"} or not path_value:
            continue
        candidate = data_root / panel_id / Path(str(path_value)).name
        if candidate.is_file() and candidate.stat().st_size > 0:
            entry["path"] = str(candidate)
    return manifest


def _load_example_selection(data_root: Path, panel_id: str | None = None) -> dict:
    """Load the frozen, source-hashed focal-region selection if available."""

    path = data_root / EXAMPLE_SELECTION_RELATIVE
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = payload.get("selected", payload)
        if panel_id is not None and str(selected.get("panel", "")) != str(panel_id):
            selected = next(
                (
                    item
                    for item in payload.get("candidates", [])
                    if str(item.get("panel", "")) == str(panel_id)
                ),
                selected,
            )
        if isinstance(selected, dict):
            return {**FALLBACK_EXAMPLE_SELECTION, **selected}
    return dict(FALLBACK_EXAMPLE_SELECTION)


def _configure_renderer(data_root: Path) -> ModuleType:
    renderer = _load_renderer()
    args = _source_args(data_root)

    def source_args(_root: Path) -> SimpleNamespace:
        return args

    renderer.source_args = source_args
    # The original configure_paths reads the common fields.  Apply the HAQER
    # overrides after it has initialized its defaults.
    original_configure = renderer.configure_source_modules

    def configure_source_modules(_root: Path):
        har, haqer = original_configure(_root)
        # Optional public tracks are allowed to be absent in the repository
        # bundle; convert only missing optional tracks to labeled empty arrays.
        original_har_ccres = har.load_ccres
        original_har_bigwig = har.load_bigwig
        def safe_har_ccres(path, chrom, start, end):
            return [] if not Path(path).is_file() else original_har_ccres(path, chrom, start, end)
        def safe_har_bigwig(path, chrom, start, end, bins):
            return np.full(int(bins), np.nan) if not Path(path).is_file() else original_har_bigwig(path, chrom, start, end, bins)
        har.load_ccres = safe_har_ccres
        har.load_bigwig = safe_har_bigwig
        original_haqer_ccres = haqer.load_ccres
        original_haqer_bigwig = haqer.load_bigwig
        def safe_haqer_ccres():
            return [] if not Path(haqer.CCRE_BB).is_file() else original_haqer_ccres()
        def safe_haqer_bigwig(path, bins):
            return np.full(int(bins), np.nan) if not Path(path).is_file() else original_haqer_bigwig(path, bins)
        haqer.load_ccres = safe_haqer_ccres
        haqer.load_bigwig = safe_haqer_bigwig
        haqer.SCORE_NPZ = args.input_npz_h
        haqer.PEAKS_TSV = args.peaks_tsv_h
        haqer.OVERLAP_JSON = args.overlap_json_h
        haqer.HAQER_BED = args.haqer_bed
        haqer.CADD_TRACK_TSV = args.cadd_track_tsv_h
        haqer.PHYLOP_JSON = args.phylop_json_h
        haqer.EVO2_TRACK_NPZ = args.evo2_track_npz_h
        return har, haqer

    renderer.configure_source_modules = configure_source_modules
    _install_reference_style(renderer)
    return renderer


def _peak_rows(renderer: ModuleType, path: Path, panel: dict) -> list[dict[str, object]]:
    """Load copied model peak calls and retain only this panel's coordinates."""

    rows: list[dict[str, object]] = []
    for source in renderer.read_text_rows(path):
        chrom = str(source.get("chromosome", source.get("chrom", "")))
        if chrom and chrom not in {str(panel["chrom"]), str(panel["chrom"]).removeprefix("chr"), "chr" + str(panel["chrom"]).removeprefix("chr")}:
            continue
        try:
            left = int(float(source["peak_start"]))
            right = int(float(source["peak_end"]))
            summit = int(float(source.get("summit_center", (left + right) // 2)))
        except (KeyError, TypeError, ValueError):
            continue
        if right <= int(panel["start"]) or left >= int(panel["end"]):
            continue
        rows.append({**source, "peak_start": left, "peak_end": right, "summit_center": summit})
    return rows


def _peak_overlap_stats(peaks: list[dict[str, object]], known: list[dict[str, object]]) -> dict[str, object]:
    """Calculate summit and interval peak-overlap statistics.

    Summit overlap remains the formal focal-region selection rule.  Interval
    overlap is retained separately because the reference captions use it for
    the displayed peak-hit totals.
    """

    def summit_in(peak: dict[str, object], item: dict[str, object]) -> bool:
        summit = int(peak.get("summit_center", peak.get("summit", (int(peak["peak_start"]) + int(peak["peak_end"])) // 2)))
        return int(item["start"]) <= summit < int(item["end"])

    peak_hits = [any(summit_in(peak, item) for item in known) for peak in peaks]
    interval_hits = [
        any(
            int(peak["peak_start"]) < int(item["end"])
            and int(item["start"]) < int(peak["peak_end"])
            for item in known
        )
        for peak in peaks
    ]
    known_hits = [any(summit_in(peak, item) for peak in peaks) for item in known]
    n_peaks = len(peaks)
    n_known = len(known)
    hit_count = sum(peak_hits)
    known_hit_count = sum(known_hits)
    return {
        "definition": "summit_inside_known_interval",
        "peak_count": n_peaks,
        "peak_hit_count": hit_count,
        "peak_hit_rate": hit_count / n_peaks if n_peaks else None,
        "peak_interval_hit_count": sum(interval_hits),
        "peak_interval_hit_rate": sum(interval_hits) / n_peaks if n_peaks else None,
        "known_count": n_known,
        "known_hit_count": known_hit_count,
        "known_recall": known_hit_count / n_known if n_known else None,
    }


def _attach_model_peak_calls(renderer: ModuleType, args: SimpleNamespace, panels: list[dict]) -> None:
    """Attach source-traceable hEVO/Evo 2 calls for the two-colour peak lane."""

    paths = {"g": args.evo2_peaks_tsv, "h": args.evo2_peaks_tsv_h}
    for panel in panels:
        panel_id = str(panel["panel"])
        evo2_peaks = _peak_rows(renderer, paths[panel_id], panel)
        known = list(panel.get("known", []))
        for peak in evo2_peaks:
            peak["interval_overlaps_known"] = any(
                int(peak["peak_start"]) < int(item["end"])
                and int(item["start"]) < int(peak["peak_end"])
                for item in known
            )
            peak["summit_in_known"] = any(
                int(item["start"]) <= int(peak["summit_center"]) < int(item["end"])
                for item in known
            )
            peak["overlaps_known"] = peak["summit_in_known"]
        panel["evo2_peaks"] = evo2_peaks
        panel["evo2_peak_stats"] = _peak_overlap_stats(evo2_peaks, known)
        hevo_peaks = [
            {
                "peak_start": int(peak.get("peak_start", peak.get("start"))),
                "peak_end": int(peak.get("peak_end", peak.get("end"))),
                "summit_center": int(peak.get("summit_center", peak.get("summit", (int(peak.get("peak_start", peak.get("start"))) + int(peak.get("peak_end", peak.get("end")))) // 2))),
                "prominence": peak.get("prominence"),
            }
            for peak in panel.get("peaks", [])
        ]
        for peak in hevo_peaks:
            peak["interval_overlaps_known"] = any(
                int(peak["peak_start"]) < int(item["end"])
                and int(item["start"]) < int(peak["peak_end"])
                for item in known
            )
            peak["summit_in_known"] = any(
                int(item["start"]) <= int(peak["summit_center"]) < int(item["end"])
                for item in known
            )
            peak["overlaps_known"] = peak["summit_in_known"]
        panel["hevo_peak_stats"] = _peak_overlap_stats(hevo_peaks, known)


def _overlaps_region(item: dict, start: int, end: int) -> bool:
    try:
        return int(item["start"]) < end and start < int(item["end"])
    except (KeyError, TypeError, ValueError):
        return True


def _crop_panel(panel: dict, start: int, end: int, selection: dict | None = None) -> dict:
    """Create a display-only cropped view without changing source arrays."""

    cropped = deepcopy(panel)
    cropped["start"] = int(start)
    cropped["end"] = int(end)
    x = np.asarray(cropped.get("x", []), dtype=float)
    if x.size:
        keep = (x >= start) & (x <= end)
        cropped["x"] = x[keep]
        for key in ("raw", "smooth"):
            values = np.asarray(cropped.get(key, []), dtype=float)
            if values.size == keep.size:
                cropped[key] = values[keep]
    peaks = list(cropped.get("peaks", []))
    categories = list(cropped.get("categories", []))
    peak_pairs = [(peak, categories[index] if index < len(categories) else None) for index, peak in enumerate(peaks)]
    peak_pairs = [
        (peak, category)
        for peak, category in peak_pairs
        if int(peak.get("peak_start", peak.get("start", start))) < end
        and start < int(peak.get("peak_end", peak.get("end", end)))
    ]
    cropped["peaks"] = [peak for peak, _category in peak_pairs]
    if categories:
        cropped["categories"] = [category for _peak, category in peak_pairs]
    cropped["peak_count"] = len(cropped["peaks"])
    for key in ("ccres", "genes"):
        if key in cropped:
            cropped[key] = [item for item in cropped[key] if _overlaps_region(item, start, end)]
    cropped["known"] = [
        item for item in cropped.get("known", []) if _overlaps_region(item, start, end)
    ]
    if selection is not None:
        known_start = int(selection["known_start"])
        known_end = int(selection["known_end"])
        cropped["example_known"] = [
            item for item in cropped.get("known", [])
            if int(item.get("start", -1)) == known_start and int(item.get("end", -1)) == known_end
        ]
        cropped["example_selection"] = selection
    cropped["evo2_peaks"] = [
        peak
        for peak in cropped.get("evo2_peaks", [])
        if int(peak["peak_start"]) < end and start < int(peak["peak_end"])
    ]
    known = list(cropped.get("known", []))
    for peak in cropped["evo2_peaks"]:
        peak["interval_overlaps_known"] = any(
            int(peak["peak_start"]) < int(item["end"])
            and int(item["start"]) < int(peak["peak_end"])
            for item in known
        )
        peak["summit_in_known"] = any(
            int(item["start"]) <= int(peak["summit_center"]) < int(item["end"])
            for item in known
        )
        peak["overlaps_known"] = peak["summit_in_known"]
    cropped["evo2_peak_stats"] = _peak_overlap_stats(cropped["evo2_peaks"], known)
    hevo_peaks = [
        {
            "peak_start": int(peak.get("peak_start", peak.get("start"))),
            "peak_end": int(peak.get("peak_end", peak.get("end"))),
            "summit_center": int(peak.get("summit_center", peak.get("summit", (int(peak.get("peak_start", peak.get("start"))) + int(peak.get("peak_end", peak.get("end")))) // 2))),
        }
        for peak in cropped["peaks"]
    ]
    for peak in hevo_peaks:
        peak["interval_overlaps_known"] = any(
            int(peak["peak_start"]) < int(item["end"])
            and int(item["start"]) < int(peak["peak_end"])
            for item in known
        )
        peak["summit_in_known"] = any(
            int(item["start"]) <= int(peak["summit_center"]) < int(item["end"])
            for item in known
        )
        peak["overlaps_known"] = peak["summit_in_known"]
    cropped["hevo_peak_stats"] = _peak_overlap_stats(hevo_peaks, known)
    cropped["comparison_mode"] = True
    cropped["view_span_label"] = f"{(end - start) / 1000:.0f} kb"
    return cropped


def _collect_comparison_panels(renderer: ModuleType, data_root: Path) -> list[dict]:
    """Collect only source data needed by the compact comparison mode.

    The frozen full collector also opens H3K27ac, phastCons and cCRE files.
    Those tracks are intentionally hidden in comparison mode and may be
    unavailable on HPC, so load the same local score/peak/known/gene inputs
    without touching those external resources.
    """

    har, haqer = renderer.configure_source_modules(data_root)

    payload = renderer.np.load(har.INPUT_NPZ, allow_pickle=False)
    chrom = str(payload["chromosome"][0])
    centers = payload["center"].astype(renderer.np.int64)
    raw = payload["raw_scores"].astype(renderer.np.float32)
    smooth = payload["smoothed_scores"].astype(renderer.np.float32)
    region_start = int(payload["start"].min())
    region_end = int(payload["end"].max())
    valid = (centers >= region_start) & (centers <= region_end)
    overlap = json.loads(har.OVERLAP_JSON.read_text(encoding="utf-8"))
    har_peaks = har.classify_peaks(har.load_peaks(har.PEAKS_TSV), overlap["peak_rows"])
    har_panel = {
        "panel": "g",
        "chrom": chrom,
        "start": region_start,
        "end": region_end,
        "x": centers[valid],
        "raw": raw[valid],
        "smooth": smooth[valid],
        "peaks": har_peaks,
        "known": har.load_known_hars(har.RAW_HAR_CSV, chrom, region_start, region_end),
        "genes": har.load_genes(har.REFGENE_GZ, chrom, region_start, region_end),
        "ccres": [],
        "h3k27ac": renderer.np.asarray([], dtype=renderer.np.float32),
        "phastcons": renderer.np.asarray([], dtype=renderer.np.float32),
        "cadd": [],
        "phylop": {"x": renderer.np.asarray([], dtype=float), "values": renderer.np.asarray([], dtype=float)},
        "gold_recovery": f"{overlap['num_gold_intervals_hit']}/{overlap['num_gold_intervals']}",
        "peak_count": len(har_peaks),
    }

    scores = haqer.load_scores()
    haqer_peaks = haqer.load_peaks()
    haqer_known = haqer.load_haqers()
    haqer_categories = haqer.categorize_peaks(haqer_peaks, haqer_known)
    haqer_panel = {
        "panel": "h",
        "chrom": haqer.CHROM,
        "start": haqer.REGION_START,
        "end": haqer.REGION_END,
        "x": renderer.np.asarray(scores["x"], dtype=float),
        "raw": renderer.np.asarray(scores["raw"], dtype=float),
        "smooth": renderer.np.asarray(scores["smooth"], dtype=float),
        "peaks": haqer_peaks,
        "categories": haqer_categories,
        "known": haqer_known,
        "genes": haqer.load_genes(),
        "ccres": [],
        "h3k27ac": renderer.np.asarray([], dtype=renderer.np.float32),
        "phastcons": renderer.np.asarray([], dtype=renderer.np.float32),
        "cadd": [],
        "phylop": {"x": renderer.np.asarray([], dtype=float), "values": renderer.np.asarray([], dtype=float)},
        "direct_recovery": f"{sum(category == 'overlaps_gold' for category in haqer_categories)}/{len(haqer_known)}",
        "peak_count": len(haqer_peaks),
    }
    return [har_panel, haqer_panel]


def _install_reference_style(renderer: ModuleType) -> None:
    """Apply the page-2 reference styling without changing frozen upstream code."""

    # The comparison panels are intentionally independent model score tracks.
    # The upstream renderer fixes hEvo to [0, 1], which is not appropriate when
    # the Evo2 likelihood track is shown beside it.
    renderer.FIXED_SCALES = dict(renderer.FIXED_SCALES)
    renderer.FIXED_SCALES.pop("hevo", None)

    original_dynamic_values = renderer.dynamic_values_for_key

    def dynamic_values_for_key(key, panels, entries):
        if key == "hevo":
            return {
                str(panel["panel"]): np.asarray(
                    np.concatenate(
                        [np.asarray(panel.get("raw", []), dtype=float), np.asarray(panel.get("smooth", []), dtype=float)]
                    ),
                    dtype=float,
                )
                for panel in panels
            }
        return original_dynamic_values(key, panels, entries)

    renderer.dynamic_values_for_key = dynamic_values_for_key
    original_compute_scales = renderer.compute_scales

    def compute_scales(panels, entries):
        scales = original_compute_scales(panels, entries)
        for key in ("hevo", "evo2"):
            values = renderer.dynamic_values_for_key(key, panels, entries)
            scales[key] = renderer.fixed_or_dynamic_scale(key, values)
        return scales

    renderer.compute_scales = compute_scales

    def spotlight_spans(panel):
        if panel.get("comparison_mode") and panel.get("example_known"):
            item = panel["example_known"][0]
            return [renderer.interval_span(int(item["start"]), int(item["end"]), int(panel["start"]), int(panel["end"]), 3_000 if int(panel["end"]) - int(panel["start"]) >= 500_000 else 900)]
        return renderer._hevo_original_spotlight_spans(panel)

    if not hasattr(renderer, "_hevo_original_spotlight_spans"):
        renderer._hevo_original_spotlight_spans = renderer.spotlight_spans
    renderer.spotlight_spans = spotlight_spans

    def draw_peak_row(ax, row, panel):
        start = int(panel["start"])
        end = int(panel["end"])
        height = row["height"]
        lanes = (
            ("hEVO call", panel.get("peaks", []), renderer.PALETTE["orange"], renderer.PALETTE["orange_soft"], 0.61),
            ("Evo2 call", panel.get("evo2_peaks", []), renderer.PALETTE["teal"], renderer.PALETTE["teal_soft"], 0.17),
        )
        for label, peaks, hit_color, other_color, y_fraction in lanes:
            y = row["bottom"] + height * y_fraction
            bar_height = height * 0.20
            for peak in peaks:
                left = int(peak.get("peak_start", peak.get("start", start)))
                right = int(peak.get("peak_end", peak.get("end", end)))
                summit = int(peak.get("summit_center", peak.get("summit", (left + right) // 2)))
                is_hit = bool(peak.get("summit_in_known", peak.get("overlaps_known", False)))
                color = hit_color if is_hit else other_color
                ax.add_patch(
                    renderer.patches.Rectangle(
                        (left, y), max(1, right - left), bar_height,
                        facecolor=color, edgecolor=hit_color, linewidth=0.18,
                        alpha=0.98, zorder=8,
                    )
                )
                ax.vlines(summit, y - height * 0.06, y + bar_height + height * 0.06, color=hit_color, lw=0.36, alpha=0.9, zorder=9)
            ax.text(
                start + (end - start) * 0.004,
                y + bar_height * 0.5,
                label,
                ha="left", va="center", fontsize=4.2, color=hit_color,
                bbox={"facecolor": renderer.PALETTE["paper"], "edgecolor": "none", "pad": 0.35},
                clip_on=True, zorder=11,
            )

    renderer.draw_peak_row = draw_peak_row

    def reference_axis(ax, start: int, end: int, x_label: str) -> None:
        span = end - start
        step = 100_000 if span >= 500_000 else 50_000 if span >= 250_000 else 20_000
        first = int(renderer.math.ceil(start / step) * step)
        ticks = renderer.np.arange(first, end + 1, step, dtype=float)
        divisor = 1_000_000.0 if end >= 1_000_000 else 1_000.0
        ax.plot(
            [start, end],
            [REFERENCE_AXIS_BASELINE, REFERENCE_AXIS_BASELINE],
            color=renderer.PALETTE["text"],
            lw=0.55,
            zorder=10,
        )
        for value in ticks:
            ax.plot(
                [value, value],
                [REFERENCE_AXIS_BASELINE, REFERENCE_AXIS_BASELINE + 0.28],
                color=renderer.PALETTE["text"],
                lw=0.42,
                zorder=10,
            )
            if divisor == 1_000_000.0:
                decimals = 2 if end - start < 500_000 else 1
                label = f"{value / divisor:.{decimals}f}"
            else:
                label = f"{value / divisor:.0f}"
            ax.text(
                value,
                REFERENCE_AXIS_BASELINE - 0.35,
                label,
                ha="center",
                va="top",
                fontsize=5.6 * REFERENCE_FONT_SCALE,
                color=renderer.PALETTE["text"],
                clip_on=False,
                zorder=10,
            )
        ax.text(
            (start + end) / 2,
            -0.55,
            x_label,
            ha="center",
            va="bottom",
            fontsize=6.1 * REFERENCE_FONT_SCALE,
            color=renderer.PALETTE["text"],
            clip_on=False,
            zorder=10,
        )

    renderer.manual_axis_ticks = reference_axis
    original_draw_panel = renderer.draw_panel

    def draw_panel(*args, **kwargs):
        ax = args[1] if len(args) > 1 else kwargs["ax"]
        panel = args[2] if len(args) > 2 else kwargs["panel"]
        scales = args[4] if len(args) > 4 else kwargs["scales"]
        rows = args[6] if len(args) > 6 else kwargs["rows"]
        for row in rows:
            if row["key"] in {"hevo", "evo2"}:
                low, high = scales[row["key"]]
                row["label"] = "hEVO score" if row["key"] == "hevo" else "Evo2 score"
        result = original_draw_panel(*args, **kwargs)
        # The custom axis is inside the reserved x-axis row; remove the
        # upstream bottom spine at y=0 so no second line obscures the labels.
        ax.spines["bottom"].set_visible(False)
        for text in list(ax.texts):
            if "peak hit:" in str(text.get_text()) or "known recovered:" in str(text.get_text()):
                text.remove()
        def peak_caption(stats):
            # The reference captions count peak-interval overlaps. Summit
            # hits remain in the source metadata as the formal selection rule.
            count = int(stats.get("peak_interval_hit_count", 0))
            total = int(stats.get("peak_count", 0))
            rate = float(stats.get("peak_interval_hit_rate", 0.0) or 0.0) * 100.0
            return f"{count}/{total} ({rate:.1f}%)"

        hevo_stats = panel.get("hevo_peak_stats", {})
        evo2_stats = panel.get("evo2_peak_stats", {})
        ax.text(
            int(panel["end"]),
            float(args[7] if len(args) > 7 else kwargs["total"]) - 1.62,
            f"hEVO peak hit: {peak_caption(hevo_stats)} | Evo 2 peak hit: {peak_caption(evo2_stats)}",
            ha="right", va="center", fontsize=4.7, color=renderer.PALETTE["text"],
        )
        total = float(args[7] if len(args) > 7 else kwargs["total"])
        for row in rows:
            if row["key"] not in {"hevo", "evo2"}:
                continue
            low, high = scales[row["key"]]
            y = row["bottom"] + row["height"] * 0.80
            ax.text(int(panel["start"]), y, f"{low:.2f}", ha="left", va="top", fontsize=3.8, color=renderer.PALETTE["text"], bbox={"facecolor": renderer.PALETTE["paper"], "edgecolor": "none", "pad": 0.2}, zorder=12)
            ax.text(int(panel["end"]), y, f"{high:.2f}", ha="right", va="top", fontsize=3.8, color=renderer.PALETTE["text"], bbox={"facecolor": renderer.PALETTE["paper"], "edgecolor": "none", "pad": 0.2}, zorder=12)
        return result

    renderer.draw_panel = draw_panel


def _wrap_reference_text(figure, ax, renderer: ModuleType) -> None:
    """Scale explicit upstream text calls to the compact reference hierarchy."""

    if not getattr(figure, "_hevo_fig4gh_reference_text", False):
        original_figure_text = figure.text

        def figure_text(x, y, s, *args, **kwargs):
            # RefSeq genes are the terminal GENOME track, but the reference
            # layout labels this row as part of the track stack rather than
            # adding a fifth vertical group label beside REF. SCORES.
            if str(s).strip() in {"GENES", "GENOME"}:
                return None
            if "fontsize" in kwargs:
                kwargs["fontsize"] = float(kwargs["fontsize"]) * REFERENCE_FONT_SCALE
            if "fontdict" in kwargs and kwargs["fontdict"] is not None:
                kwargs["fontdict"] = dict(kwargs["fontdict"])
                if "fontsize" in kwargs["fontdict"]:
                    kwargs["fontdict"]["fontsize"] = float(kwargs["fontdict"]["fontsize"]) * REFERENCE_FONT_SCALE
            if kwargs.get("color") == renderer.PALETTE["navy"]:
                kwargs["color"] = renderer.PALETTE["text"]
            return original_figure_text(x, y, s, *args, **kwargs)

        figure.text = figure_text
        figure._hevo_fig4gh_reference_text = True

    if not getattr(ax, "_hevo_fig4gh_reference_text", False):
        original_axis_text = ax.text

        def axis_text(x, y, s, *args, **kwargs):
            if "fontsize" in kwargs:
                kwargs["fontsize"] = float(kwargs["fontsize"]) * REFERENCE_FONT_SCALE
            if "fontdict" in kwargs and kwargs["fontdict"] is not None:
                kwargs["fontdict"] = dict(kwargs["fontdict"])
                if "fontsize" in kwargs["fontdict"]:
                    kwargs["fontdict"]["fontsize"] = float(kwargs["fontdict"]["fontsize"]) * REFERENCE_FONT_SCALE
            # Only the panel title is typography that upstream colors navy;
            # known-locus labels and highlighted gene names retain navy.
            if kwargs.get("color") == renderer.PALETTE["navy"] and " | " in str(s):
                kwargs["color"] = renderer.PALETTE["text"]
            return original_axis_text(x, y, s, *args, **kwargs)

        ax.text = axis_text
        ax._hevo_fig4gh_reference_text = True


def _load_context(
    tables_dir: Path | None = None,
    *,
    mode: str = "full",
    visible_keys: set[str] | None = None,
    view_span: str = "context",
    panel_id: str | None = None,
):
    if mode not in {"full", "comparison"}:
        raise ValueError(f"Unsupported Fig4g/h mode: {mode}")
    root = Path(tables_dir) if tables_dir is not None else DATA_ROOT
    if not (root / "S40_genomic_context_tracks_for_human_accelerated_and_quickly_evolved_regions.csv").is_file():
        raise FileNotFoundError(
            "Active Fig4g/h rendering requires formal S40_genomic_context_tracks_for_human_accelerated_and_quickly_evolved_regions.csv; "
            "the legacy track bundle and S40 source table must be restored before this renderer can run."
        )
    return _load_s47_context(
        root, mode=mode, visible_keys=visible_keys, view_span=view_span, panel_id=panel_id
    )


def _s47_json(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _s47_float(value: str) -> float:
    if value in {"", "NA", "nan", "NaN", "None"}:
        return float("nan")
    return float(value)


def _s47_entries(table_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    with (table_root / "har_haqer_track_source_manifest.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            panel = str(row["panel"])
            key = str(row["track"])
            entries[(panel, key)] = {
                **row,
                "panel": panel,
                "key": key,
                "group": row.get("group", ""),
                "label": row.get("label", key),
                "kind": row.get("track_type", "signal"),
                "required_for_formal": row.get("required_for_formal", "True").lower() == "true",
                "formal_eligible": row.get("formal_eligible", "True").lower() == "true",
                "status": row.get("status", "verified_real"),
                "path": None,
            }
    return entries


def _load_s47_context(
    table_root: Path,
    *,
    mode: str,
    visible_keys: set[str] | None,
    view_span: str,
    panel_id: str | None,
):
    """Rebuild the frozen renderer inputs from the formal S40 long table."""

    if mode not in {"full", "comparison"}:
        raise ValueError(f"Unsupported Fig4g/h mode: {mode}")
    renderer = _load_renderer()
    _install_reference_style(renderer)
    entries = _s47_entries(table_root)
    rows_by_panel: dict[str, list[dict[str, str]]] = {"g": [], "h": []}
    with (table_root / "S40_genomic_context_tracks_for_human_accelerated_and_quickly_evolved_regions.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_panel.setdefault(str(row["panel"]), []).append(row)

    panels: list[dict[str, Any]] = []
    for panel_id_value in ("g", "h"):
        source_rows = rows_by_panel.get(panel_id_value, [])
        meta_row = next(row for row in source_rows if row["track"] == "__panel__")
        panel = _s47_json(meta_row.get("payload_json", "")) or {}
        panel["panel"] = panel_id_value
        panel["chrom"] = str(meta_row["chrom"])
        panel["start"] = int(meta_row["region_start"])
        panel["end"] = int(meta_row["region_end"])
        signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for track in sorted({row["track"] for row in source_rows if row["track"] not in {"__panel__", "peaks", "evo2_peaks", "known", "genes", "ccres", "cadd", "ccre_cage_extra"}}):
            selected = [row for row in source_rows if row["track"] == track]
            x = np.asarray([_s47_float(row["position"]) for row in selected], dtype=float)
            y = np.asarray([_s47_float(row["value"]) for row in selected], dtype=float)
            signals[track] = (x, y)
        panel["x"] = signals.get("hevo_raw", (np.asarray([], dtype=float), np.asarray([], dtype=float)))[0]
        panel["raw"] = signals.get("hevo_raw", (panel["x"], np.asarray([], dtype=float)))[1]
        panel["smooth"] = signals.get("hevo_smooth", (panel["x"], np.asarray([], dtype=float)))[1]
        panel["h3k27ac"] = signals.get("h3k27ac", (panel["x"], np.full(len(panel["x"]), np.nan)))[1]
        panel["phastcons"] = signals.get("phastcons", (panel["x"], np.full(len(panel["x"]), np.nan)))[1]
        panel["_signals"] = signals
        for track in ("peaks", "evo2_peaks", "known", "genes", "ccres", "cadd"):
            values = []
            for row in source_rows:
                if row["track"] == track and row.get("payload_json"):
                    payload = _s47_json(row["payload_json"])
                    if payload is not None:
                        values.append(payload)
            panel[track] = values
        panel["_extra_intervals"] = [
            _s47_json(row["payload_json"])
            for row in source_rows
            if row["track"] == "ccre_cage_extra" and row.get("payload_json")
        ]
        phylop = signals.get("human_branch_phylop", (np.asarray([], dtype=float), np.asarray([], dtype=float)))
        panel["phylop"] = {"x": phylop[0], "values": phylop[1]}
        if panel_id_value == "g":
            panel["gold_recovery"] = panel.get("gold_recovery", "NA")
        else:
            panel["direct_recovery"] = panel.get("direct_recovery", "NA")
        panels.append(panel)

    original_load = renderer.load_entry_signal
    def load_s47_signal(entry, panel):
        key = str(entry.get("key", ""))
        loaded = panel.get("_signals", {}).get(key)
        if loaded is not None:
            return loaded
        return original_load(entry, panel)
    renderer.load_entry_signal = load_s47_signal
    original_extra = renderer.make_extra_intervals
    def load_s47_intervals(entry, panel):
        if str(entry.get("key", "")) == "ccre_cage":
            return list(panel.get("_extra_intervals", []))
        return original_extra(entry, panel)
    renderer.make_extra_intervals = load_s47_intervals

    selected_panels = panels
    if mode == "comparison":
        if view_span not in {"context", "focus"}:
            raise ValueError(f"Unsupported comparison view: {view_span}")
        source_panel = next(panel for panel in panels if str(panel["panel"]) == str(panel_id or "g"))
        start, end = COMPARISON_REGIONS[str(source_panel["panel"])]
        selected_panels = [_crop_panel(source_panel, start, end, None)]
    elif panel_id is not None:
        selected_panels = [panel for panel in panels if str(panel["panel"]) == str(panel_id)]
    scales = renderer.compute_scales(selected_panels, entries)
    layout_keys = COMPARISON_TRACK_KEYS if mode == "comparison" else visible_keys
    rows, total = renderer.row_layout(layout_keys)
    return renderer, {
        "tracks": list(entries.values()),
        "style_contract": {"panel_gap_mm": 1.5},
    }, selected_panels, entries, scales, rows, total, table_root


def build_panel(
    tables_dir: Path | None = None,
    *,
    panel_id: str,
    figure=None,
    slot=None,
    mode: str = "full",
    integrated: bool = False,
    letter: str | None = None,
    visible_keys: set[str] | None = None,
    view_span: str = "context",
):
    """Build one g or h panel, directly on an optional integrated Figure."""

    renderer, manifest, panels, entries, scales, rows, total, _root = _load_context(
        tables_dir,
        mode=mode,
        visible_keys=visible_keys,
        view_span=view_span,
        panel_id=panel_id,
    )
    panel = next(item for item in panels if str(item["panel"]) == panel_id)
    renderer.mpl_style()
    if figure is None:
        figure = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
        ax = figure.add_axes([0.215, 0.075, 0.773, 0.890])
    else:
        ax = figure.add_subplot(slot)
    _wrap_reference_text(figure, ax, renderer)
    if integrated:
        position = ax.get_position()
        label_x = max(0.005, position.x0 - 0.008)
        group_x = max(0.002, position.x0 - 0.065)
        letter_x = max(0.008, position.x0 - 0.050)
    else:
        label_x = 0.205
        group_x = 0.046
        letter_x = 0.008
    renderer.draw_panel(
        figure,
        ax,
        panel,
        entries,
        scales,
        letter if letter is not None else panel_id,
        rows,
        total,
        label_x=label_x,
        group_x=group_x,
        letter_x=letter_x,
    )
    return figure if slot is None else ax


def export_panel(
    tables_dir: Path | None,
    output_dir: Path,
    panel_id: str,
    stem: str,
    *,
    mode: str = "full",
    view_span: str = "context",
    letter: str | None = None,
):
    from common.matplotlib_common import export_figure
    from common.panel_runtime import input_files

    root = Path(tables_dir) if tables_dir is not None else DATA_ROOT
    fig = build_panel(root, panel_id=panel_id, mode=mode, view_span=view_span, letter=letter)
    copied_inputs = [
        path for path in (root / "har_haqer_tracks" / panel_id).iterdir() if path.is_file()
    ]
    selection_path = root / EXAMPLE_SELECTION_RELATIVE
    if selection_path.is_file():
        copied_inputs.append(selection_path)
    result = export_figure(fig, output_dir, stem, input_files(*copied_inputs), WIDTH_MM, HEIGHT_MM)
    plt.close(fig)
    return result


def build_combined(
    tables_dir: Path | None = None,
    *,
    mode: str = "full",
    visible_keys: set[str] | None = None,
    letter_map: dict[str, str] | None = None,
    second_panel_shift_mm: float = 0.0,
    view_span: str = "context",
):
    """Build the two-panel formal preview, optionally lowering the second panel."""

    if second_panel_shift_mm < 0:
        raise ValueError("second_panel_shift_mm must be non-negative")
    renderer, manifest, panels, entries, scales, rows, total, _root = _load_context(
        tables_dir, mode=mode, visible_keys=visible_keys, view_span=view_span
    )
    renderer.mpl_style()
    base_height_mm = COMBINED_HEIGHT_MM
    canvas_height_mm = base_height_mm + second_panel_shift_mm
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, canvas_height_mm / 25.4), facecolor="white")
    left, right, bottom_fraction, top_fraction = 0.215, 0.988, 0.055, 0.965
    gap_mm = float(manifest["style_contract"]["panel_gap_mm"])
    bottom_mm = bottom_fraction * base_height_mm
    panel_height_mm = (top_fraction * base_height_mm - bottom_mm - gap_mm) / 2.0
    panel_y_mm = (
        bottom_mm + panel_height_mm + gap_mm + second_panel_shift_mm,
        bottom_mm,
    )
    axes = {}
    for panel, y_mm in zip(panels, panel_y_mm):
        axes[str(panel["panel"])] = fig.add_axes(
            [left, y_mm / canvas_height_mm, right - left, panel_height_mm / canvas_height_mm]
        )
        _wrap_reference_text(fig, axes[str(panel["panel"])], renderer)
        display_letter = (letter_map or {}).get(str(panel["panel"]), str(panel["panel"]))
        renderer.draw_panel(
            fig,
            axes[str(panel["panel"])],
            panel,
            entries,
            scales,
            display_letter,
            rows,
            total,
            label_x=0.205,
            group_x=0.046,
        )
    return fig


def external_status() -> dict[str, object]:
    deps = _external_paths()
    return {key: {"path": str(path), "exists": path.exists()} for key, path in deps.items()}
