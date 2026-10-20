#!/usr/bin/env python3
"""Build the Fig. 4e hEvo-large/GPN-Star-M source-data table."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
BENCHMARK = HERE.parents[3]
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))
if str(BENCHMARK / "scripts/har_haqer_workflow") not in sys.path:
    sys.path.insert(0, str(BENCHMARK / "scripts/har_haqer_workflow"))

from Fig4.e_haqer_gpn_example_selection import (  # noqa: E402
    DISPLAY_GAP_BP,
    GPN_RUN,
    HEVO_RUN,
    SELECTED_IDS,
    selected_segments,
    write_manifest,
)
from plot_large_comparison import genes  # noqa: E402

import importlib.util

_caller_spec = importlib.util.spec_from_file_location(
    "fig4e_peak_caller",
    BENCHMARK / "scripts/har_haqer_workflow/core/11_call_smoothed_peaks.py",
)
_caller = importlib.util.module_from_spec(_caller_spec)
_caller_spec.loader.exec_module(_caller)

SEED = 20260515
COMMON_SCORE_CUTOFF = 0.59
THRESHOLD_GRID = [round(float(value), 2) for value in np.arange(0.30, 0.901, 0.01)]
OUTPUT_ROOT = BENCHMARK / "tables/fig4e_haqer_tracks"
ANNOTATIONS = OUTPUT_ROOT / "annotations"
SOURCE_CSV = OUTPUT_ROOT / "Fig4e_track_source_data.csv"
SEGMENTS_JSON = OUTPUT_ROOT / "selected_segments.json"
FIELDS = [
    "table_id", "panel", "segment", "assembly", "chrom", "region_start", "region_end",
    "track", "track_type", "position", "start", "end", "summit", "value", "label",
    "status", "source_file", "payload_json",
]


def _read_json(path: Path):
    return json.loads(path.read_text())


def _read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _row(segment, track, kind, *, position="", start="", end="", summit="",
         value="", label="", payload=None, source=""):
    payload_json = json.dumps(
        payload, separators=(",", ":"), allow_nan=False, default=_json_default
    ) if payload is not None else ""
    segment = segment or {}
    return {
        "table_id": "Fig4e-hEvo-GPN", "panel": "hevo_gpn_comparison",
        "segment": segment.get("index", ""), "assembly": "hg38",
        "chrom": segment.get("chrom", ""),
        "region_start": segment.get("start", ""),
        "region_end": segment.get("end", ""),
        "track": track, "track_type": kind, "position": position,
        "start": start, "end": end, "summit": summit, "value": value,
        "label": label, "status": "verified_real", "source_file": source,
        "payload_json": payload_json,
    }


def _caller_config(model, lock):
    if model == "hevo":
        return [
            float(lock["sigma"]), float(lock.get("truncate", 8)), float(lock["score"]),
            float(lock["prominence"]), int(lock["distance"]), int(lock["width"]),
        ]
    return [float(x) for x in lock]


def _call(raw, valid, centers, starts, ends, config, stride):
    sigma, truncate, min_score, min_prominence, min_distance, min_width = config
    smoothed = _caller.smooth_signal(
        raw, valid,
        _caller.gaussian_kernel(sigma / max(stride, 1), truncate),
    )
    indexes = _caller.find_local_maxima(smoothed, valid)
    indexes = indexes[np.asarray([smoothed[i] >= min_score for i in indexes], dtype=bool)]
    indexes = _caller.suppress_nearby_peaks(indexes, smoothed, centers, min_distance)
    peaks = []
    for index in indexes:
        prominence, _, _ = _caller.compute_prominence(smoothed, index)
        if prominence < min_prominence:
            continue
        width, left, right = _caller.compute_width(
            smoothed, index, prominence, stride, centers
        )
        if width < min_width:
            continue
        peaks.append({
            "peak_start": int(starts[left]), "peak_end": int(ends[right]),
            "summit_center": int(centers[index]),
            "raw_score": float(raw[index]),
            "smoothed_score": float(smoothed[index]),
            "prominence": float(prominence), "width_bp": float(width),
        })
    return smoothed, peaks


def _load_signal(model, segment, config):
    if model == "hevo":
        path = HEVO_RUN / "scans/hevo" / segment["hevo_region_id"] / f"{SEED}.raw_scores.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            centers = data["center"].astype(np.int64, copy=True)
            starts = data["start"].astype(np.int64, copy=True)
            ends = data["end"].astype(np.int64, copy=True)
            valid = data["valid_mask"].astype(bool, copy=True)
            raw = data["probabilities"].astype(np.float32, copy=True)
            stride = int(data["stride"])
    else:
        path = GPN_RUN / "scans/haqer" / segment["split"] / f"{segment['gpn_region_id']}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            centers = data["center"].astype(np.int64, copy=True)
            starts = data["start"].astype(np.int64, copy=True)
            ends = data["end"].astype(np.int64, copy=True)
            valid = data["valid_mask"].astype(bool, copy=True)
            seeds = data["seeds"].astype(np.int64, copy=True)
            seed_index = int(np.flatnonzero(seeds == SEED)[0])
            raw = data["probabilities"][seed_index].astype(np.float32, copy=True)
            stride = int(data["stride"])
    if not np.isfinite(raw[valid]).all():
        raise ValueError(f"Non-finite valid score in {model} scan: {path}")
    scores, peaks = _call(raw, valid, centers, starts, ends, config, stride)
    return path, centers, valid & np.isfinite(scores), scores, peaks


def _regions(model, split):
    if model == "hevo":
        return [
            row for row in _read_json(HEVO_RUN / "regions.json")
            if row["split"] == split
        ]
    return [
        row for row in _read_json(GPN_RUN / "regions.json")
        if row.get("task") == "haqer" and row.get("split") == split
    ]


def _scan_peaks(model, region, seed, config):
    if model == "hevo":
        path = HEVO_RUN / "scans/hevo" / region["id"] / f"{seed}.raw_scores.npz"
        core_start, core_end = int(region["start"]), int(region["end"])
        chrom = region["chrom"]
    else:
        path = GPN_RUN / "scans/haqer" / region["split"] / f"{region['id']}.npz"
        core_start, core_end = int(region["core_start"]), int(region["core_end"])
        chrom = region["chrom"]
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        centers = data["center"].astype(np.int64, copy=True)
        starts = data["start"].astype(np.int64, copy=True)
        ends = data["end"].astype(np.int64, copy=True)
        valid = data["valid_mask"].astype(bool, copy=True)
        if model == "hevo":
            raw = data["probabilities"].astype(np.float32, copy=True)
        else:
            seeds = data["seeds"].astype(np.int64, copy=False)
            seed_index = int(np.flatnonzero(seeds == seed)[0])
            raw = data["probabilities"][seed_index].astype(np.float32, copy=True)
        stride = int(data["stride"])
    if not np.isfinite(raw[valid]).all():
        raise ValueError(f"Non-finite valid score in {model} {region['id']} seed {seed}")
    _, peaks = _call(raw, valid, centers, starts, ends, config, stride)
    return [
        (chrom, int(peak["summit_center"])) for peak in peaks
        if core_start <= int(peak["summit_center"]) < core_end
    ]


def _peak_metrics(peaks, gold_by_chrom, denominator):
    true_peak_count = 0
    known_hits = set()
    for chrom, summit in peaks:
        matched = [
            ident for left, right, ident in gold_by_chrom.get(chrom, [])
            if left <= summit < right
        ]
        if matched:
            true_peak_count += 1
            known_hits.update(matched)
    precision = true_peak_count / len(peaks) if peaks else 0.0
    recall = len(known_hits) / denominator if denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "n_peaks": len(peaks), "n_true_peaks": true_peak_count,
        "n_known_hits": len(known_hits), "n_known": denominator,
    }


def _metrics_for_split(split, hevo_config, gpn_config, seeds):
    all_gold = defaultdict(list)
    split_chromosomes = (
        {f"chr{chrom}" for chrom in range(13, 18)}
        if split == "val" else {f"chr{chrom}" for chrom in range(18, 23)}
    )
    for line in (BENCHMARK / "datasets/HAQER_golden_windwos/haqer.hg38.bed").read_text().splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0] not in split_chromosomes:
            continue
        chrom, start, end, ident = fields[0], int(fields[1]), int(fields[2]), fields[3]
        all_gold[chrom].append((start, end, ident))

    common_ids = None
    excluded_by_chrom = defaultdict(list)
    if split == "test":
        hevo_hits = {row["id"]: row for row in _read_tsv(HEVO_RUN / "reports/test_known_haqer_hits.tsv")}
        gpn_hits = {row["id"]: row for row in _read_tsv(GPN_RUN / "reports/test_known_HAQER_hits.tsv")}
        common_ids = {
            ident for ident, row in hevo_hits.items()
            if row["evaluable"].lower() == "true"
            and ident in gpn_hits and gpn_hits[ident]["evaluable"].lower() == "true"
        }
        if len(common_ids) != 214:
            raise ValueError(f"Expected 214 common evaluable test HAQERs; found {len(common_ids)}")
        for chrom, intervals in all_gold.items():
            for left, right, ident in intervals:
                if ident not in common_ids and ident in hevo_hits:
                    excluded_by_chrom[chrom].append((left, right, ident))
        gold_by_chrom = {
            chrom: [(left, right, ident) for left, right, ident in intervals if ident in common_ids]
            for chrom, intervals in all_gold.items()
        }
        denominator = len(common_ids)
    else:
        gold_by_chrom = all_gold
        denominator = sum(len(intervals) for intervals in all_gold.values())

    result = {}
    for seed in seeds:
        result[seed] = {}
        for model, config in (("hevo", hevo_config), ("gpn", gpn_config)):
            calls = []
            for region in _regions(model, split):
                calls.extend(_scan_peaks(model, region, seed, config))
            if split == "test":
                calls = [
                    (chrom, summit) for chrom, summit in calls
                    if not any(left <= summit < right for left, right, _ in excluded_by_chrom.get(chrom, []))
                ]
            result[seed][model] = _peak_metrics(calls, gold_by_chrom, denominator)
    return result


def _matched_test_metrics(hevo_config, gpn_config):
    hevo_hits = {row["id"]: row for row in _read_tsv(HEVO_RUN / "reports/test_known_haqer_hits.tsv")}
    gpn_hits = {row["id"]: row for row in _read_tsv(GPN_RUN / "reports/test_known_HAQER_hits.tsv")}
    common_ids = {
        ident for ident, row in hevo_hits.items()
        if row["evaluable"].lower() == "true"
        and ident in gpn_hits and gpn_hits[ident]["evaluable"].lower() == "true"
    }
    if len(common_ids) != 214:
        raise ValueError(f"Expected 214 common evaluable test HAQERs; found {len(common_ids)}")
    hevo_by_seed = {}
    gpn_by_seed = {}
    common_validation = _metrics_for_split("val", hevo_config, gpn_config, (SEED,))[SEED]
    common_test = _metrics_for_split("test", hevo_config, gpn_config, (20260515, 20260516, 20260517))
    for seed, metrics in common_test.items():
        hevo_by_seed[seed] = metrics["hevo"]
        gpn_by_seed[seed] = metrics["gpn"]

    metric_rows = {
        "hevo": _read_tsv(HEVO_RUN / "reports/test_window_metrics.tsv"),
        "gpn": _read_tsv(GPN_RUN / "reports/window_metrics.tsv"),
    }
    result = {"n_common_haqers": len(common_ids), "n_test_windows": 1216,
              "validation": common_validation}
    for model in ("hevo", "gpn"):
        rows = metric_rows[model]
        if model == "hevo":
            rows = [row for row in rows if row["model"] == "hevo"]
        else:
            rows = [row for row in rows if row["task"] == "haqer" and row["split"] == "test"]
        by_seed = {int(row["seed"]): row for row in rows}
        if set(by_seed) != {20260515, 20260516, 20260517}:
            raise ValueError(f"Missing test window metric seeds for {model}")
        result[model] = {
            "window_auprc": float(np.mean([float(row["window_auprc"]) for row in by_seed.values()])),
            "window_auroc": float(np.mean([float(row["window_auroc"]) for row in by_seed.values()])),
            "center_top1_rate": float(np.mean([float(row["center_top1_rate"]) for row in by_seed.values()])),
            "summit_precision": float(np.mean([hevo_by_seed[s]["precision"] if model == "hevo" else gpn_by_seed[s]["precision"] for s in by_seed])),
            "summit_recall": float(np.mean([hevo_by_seed[s]["recall"] if model == "hevo" else gpn_by_seed[s]["recall"] for s in by_seed])),
            "summit_f1": float(np.mean([hevo_by_seed[s]["f1"] if model == "hevo" else gpn_by_seed[s]["f1"] for s in by_seed])),
            "per_seed_peak_metrics": hevo_by_seed if model == "hevo" else gpn_by_seed,
        }
    return result


def export() -> dict:
    manifest = write_manifest(SEGMENTS_JSON)
    segments = manifest["segments"]
    if len(segments) != 12 or {row["id"] for row in segments} != set(SELECTED_IDS):
        raise ValueError("Selected-locus manifest does not contain the fixed 12 examples")
    annotation_signals_path = ANNOTATIONS / "signals.tsv"
    annotation_ccres_path = ANNOTATIONS / "ccres.json"
    annotation_sources_path = ANNOTATIONS / "sources.json"
    for path in (annotation_signals_path, annotation_ccres_path, annotation_sources_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    hevo_lock = _read_json(HEVO_RUN / "locked_callers.json")["selected"]["hevo"]
    gpn_lock_data = _read_json(GPN_RUN / "locked_callers.json")
    gpn_selection = gpn_lock_data["selected"]["haqer"]
    gpn_caller = gpn_selection["caller"]
    gpn_config = gpn_lock_data["caller_parameters"][gpn_caller]
    configs = {"hevo": _caller_config("hevo", hevo_lock), "gpn": _caller_config("gpn", gpn_config)}
    for config in configs.values():
        config[2] = COMMON_SCORE_CUTOFF
    effective_hevo = {
        "model": "hevo",
        "caller": f"s128_t{COMMON_SCORE_CUTOFF:g}_p0.1_d1024_w64_shared",
        "sigma": configs["hevo"][0], "truncate": configs["hevo"][1],
        "score": COMMON_SCORE_CUTOFF, "prominence": configs["hevo"][3],
        "distance": configs["hevo"][4], "width": configs["hevo"][5],
        "cutoff_selection": "shared validation macro-F1",
    }
    effective_gpn = {
        "model": "gpn", "base_caller": gpn_caller,
        "caller": f"{gpn_caller}_shared_t{COMMON_SCORE_CUTOFF:g}",
        "sigma": configs["gpn"][0], "truncate": configs["gpn"][1],
        "score": COMMON_SCORE_CUTOFF, "prominence": configs["gpn"][3],
        "distance": configs["gpn"][4], "width": configs["gpn"][5],
        "cutoff_selection": "shared validation macro-F1",
    }

    signal_annotations = defaultdict(list)
    for row in _read_tsv(annotation_signals_path):
        signal_annotations[(row["track"], int(row["segment"]))].append(row)
    ccre_annotations = _read_json(annotation_ccres_path)

    output_rows = []
    segment_metadata = []
    selection_deltas = {}
    for segment in segments:
        data = {"segment": segment, "models": {}, "genes": []}
        for model in ("hevo", "gpn"):
            path, centers, valid, scores, peaks = _load_signal(model, segment, configs[model])
            mask = (centers >= segment["start"]) & (centers < segment["end"])
            mask &= valid
            if not mask.any():
                raise ValueError(f"No finite {model} scores in display context for {segment['id']}")
            values = scores[mask]
            positions = centers[mask]
            data["models"][model] = {"max": float(np.max(values)), "peaks": peaks}
            selection_deltas.setdefault(segment["id"], {})[model] = data["models"][model]["max"]
            track_name = "hevo" if model == "hevo" else "gpn"
            for position, value in zip(positions, values):
                output_rows.append(_row(
                    segment, track_name, "signal", position=int(position), value=float(value),
                    label=f"{model} calibrated HAQER-head score", source=f"{model} seed {SEED}",
                ))
            in_display = [
                peak for peak in peaks
                if segment["start"] <= peak["summit_center"] < segment["end"]
            ]
            data["models"][model]["display_peaks"] = in_display
            for peak in in_display:
                payload = dict(peak)
                payload.update(
                    model=model,
                    segment=segment["index"],
                    interval_overlaps_known=(
                        peak["peak_start"] < segment["gold_end"]
                        and peak["peak_end"] > segment["gold_start"]
                    ),
                    summit_in_known=segment["gold_start"] <= peak["summit_center"] < segment["gold_end"],
                )
                output_rows.append(_row(
                    segment, "hevo_peaks" if model == "hevo" else "gpn_peaks", "peaks",
                    start=peak["peak_start"], end=peak["peak_end"],
                    summit=peak["summit_center"], value=peak["smoothed_score"],
                    payload=payload,
                    source=f"{model} seed {SEED} shared validation-selected cutoff {COMMON_SCORE_CUTOFF:g}",
                ))

        hmax = selection_deltas[segment["id"]]["hevo"]
        gmax = selection_deltas[segment["id"]]["gpn"]
        h_peaks = data["models"]["hevo"]["display_peaks"]
        g_peaks = data["models"]["gpn"]["display_peaks"]
        h_gold = [peak for peak in h_peaks if segment["gold_start"] <= peak["summit_center"] < segment["gold_end"]]
        if not h_gold:
            raise ValueError(f"hEvo does not summit-hit selected golden interval {segment['id']}")
        if g_peaks:
            raise ValueError(f"GPN has a called peak within selected display context {segment['id']}")
        if hmax - gmax < 0.15:
            raise ValueError(f"Score margin below 0.15 for {segment['id']}: {hmax-gmax:.3f}")
        if min(peak["width_bp"] for peak in h_gold) > 1536:
            raise ValueError(f"hEvo summit peak is not sharp enough for {segment['id']}")
        segment_metadata.append({
            **segment,
            "hevo_local_max": hmax,
            "gpn_local_max": gmax,
            "score_margin": hmax - gmax,
            "hevo_summits": [p["summit_center"] for p in h_gold],
            "gpn_local_call_count": len(g_peaks),
            "hevo_peak_widths_bp": [p["width_bp"] for p in h_gold],
        })

        gold_payload = {
            "id": segment["id"], "split": segment["split"],
            "start": segment["gold_start"], "end": segment["gold_end"],
            "hevo_hit": True, "gpn_hit": False,
            "score_margin": hmax - gmax,
            "literature_context": segment.get("literature_context", ""),
        }
        output_rows.append(_row(
            segment, "known", "intervals", start=segment["gold_start"], end=segment["gold_end"],
            label=segment["id"][-4:], payload=gold_payload,
            source="datasets/HAQER_golden_windwos/haqer.hg38.bed",
        ))

        genes_in_context, _ = genes(segment["chrom"], segment["start"], segment["end"])
        for gene in genes_in_context:
            exons = sorted((int(left), int(right)) for left, right in gene["exons"])
            output_rows.append(_row(
                segment, "genes", "genes", start=int(gene["start"]), end=int(gene["end"]),
                label=gene["name"], payload={
                    "segment": segment["index"], "name": gene["name"],
                    "start": int(gene["start"]), "end": int(gene["end"]),
                    "strand": gene["strand"], "exons": exons,
                }, source="RefSeq hg38 refGene",
            ))

        for key in ("h3k27ac", "ape_divergence", "phastcons"):
            for row in signal_annotations.get((key, segment["index"]), []):
                value = row["value"]
                output_rows.append(_row(
                    segment, key, "signal", position=row["position"],
                    start=int(row["start"]), end=int(row["end"]),
                    value="" if value in (None, "None", "") else float(value),
                    label=key, source=f"annotations/signals.tsv:{key}",
                ))
        for row in ccre_annotations:
            if int(row["segment"]) != segment["index"]:
                continue
            output_rows.append(_row(
                segment, "ccres", "intervals", start=int(row["start"]), end=int(row["end"]),
                label=row.get("name", ""), payload=row,
                source="ENCODE encodeCcreCombined hg38",
            ))

    metrics = _matched_test_metrics(configs["hevo"], configs["gpn"])
    validation_metrics = metrics.pop("validation")
    for model, caller in (("hevo", effective_hevo), ("gpn", effective_gpn)):
        caller.update(validation_metrics[model])
        metrics[model].update({"caller": caller["caller"], "cutoff": COMMON_SCORE_CUTOFF})
    metadata = {
        "schema": "fig4e_hevo_gpn_tracks.v1",
        "panel": "Fig4e",
        "assembly": "hg38",
        "coordinate_system": "0-based half-open",
        "selection": (
            "held-out HAQERs with hEvo summit inside gold, no GPN called peak within the displayed +/-2 kb, "
            "hEvo-GPN local smoothed score margin >= 0.15; narrow hEvo summit peaks; COL4A2-near HAQER0218 retained"
        ),
        "display_gap_bp": DISPLAY_GAP_BP,
        "primary_seed": SEED,
        "score_note": "Both callers use the same 0.59 score cutoff. The two HAQER heads were calibrated independently, so equal scores are not guaranteed to be equally calibrated probabilities.",
        "peak_hit_definition": "called summit center is inside the original 0-based half-open HAQER interval",
        "common_cutoff_selection": {
            "score": COMMON_SCORE_CUTOFF,
            "grid": {"start": THRESHOLD_GRID[0], "end": THRESHOLD_GRID[-1], "step": 0.01},
            "criterion": "maximize the macro-average of hEvo and GPN summit-hit F1 on validation; tie-break by macro recall, then fewer combined peaks",
            "selected_validation_macro_f1": float((validation_metrics["hevo"]["f1"] + validation_metrics["gpn"]["f1"]) / 2),
            "validation_metrics": validation_metrics,
        },
        "caller": {"hevo": effective_hevo, "gpn": effective_gpn},
        "caller_parameters": {"hevo": configs["hevo"], "gpn": configs["gpn"]},
        "test_metrics": metrics,
        "segments": segment_metadata,
        "annotation_sources": _read_json(annotation_sources_path),
        "tracks": ["hevo", "gpn", "hevo_peaks", "gpn_peaks", "known", "ccres",
                   "h3k27ac", "ape_divergence", "phastcons", "genes"],
    }
    output_rows.insert(0, _row(
        {}, "__panel__", "panel", payload=metadata,
        source="e_haqer_gpn_example_selection.py",
    ))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with SOURCE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "source_data": str(SOURCE_CSV),
        "selected_ids": [segment["id"] for segment in segment_metadata],
        "score_margins": {key: round(value["hevo"] - value["gpn"], 3) for key, value in selection_deltas.items()},
        "rows": len(output_rows),
        "test_metrics": metrics,
    }


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))
