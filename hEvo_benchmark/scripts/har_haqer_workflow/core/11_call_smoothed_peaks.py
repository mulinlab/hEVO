from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smooth raw sliding-window scores and call local peaks with prominence, width, and distance filters."
        )
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        required=True,
        help="Raw score archive produced by 10_scan_genome_region.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix without extension.",
    )
    parser.add_argument(
        "--score-field",
        choices=["probabilities", "logits"],
        default="probabilities",
        help="Raw score field to smooth and call peaks on.",
    )
    parser.add_argument(
        "--sigma-bp",
        type=float,
        default=128.0,
        help="Gaussian smoothing sigma in bp.",
    )
    parser.add_argument(
        "--truncate",
        type=float,
        default=4.0,
        help="Gaussian kernel truncate radius in sigmas.",
    )
    parser.add_argument(
        "--min-peak-score",
        type=float,
        default=0.5,
        help="Minimum smoothed score to keep a peak.",
    )
    parser.add_argument(
        "--min-prominence",
        type=float,
        default=0.05,
        help="Minimum peak prominence after smoothing.",
    )
    parser.add_argument(
        "--min-distance-bp",
        type=int,
        default=512,
        help="Minimum distance between retained peak summits.",
    )
    parser.add_argument(
        "--min-width-bp",
        type=int,
        default=128,
        help="Minimum peak width at half prominence.",
    )
    return parser.parse_args()


def gaussian_kernel(sigma_windows: float, truncate: float) -> np.ndarray:
    sigma_windows = max(float(sigma_windows), 1e-6)
    radius = max(1, int(math.ceil(truncate * sigma_windows)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / sigma_windows))
    kernel /= kernel.sum()
    return kernel.astype(np.float64)


def smooth_signal(
    values: np.ndarray, valid_mask: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    raw = np.where(valid_mask, values, 0.0).astype(np.float64)
    weights = valid_mask.astype(np.float64)
    numerator = np.convolve(raw, kernel, mode="same")
    denominator = np.convolve(weights, kernel, mode="same")
    smoothed = np.divide(
        numerator,
        np.clip(denominator, a_min=1e-12, a_max=None),
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    smoothed[~valid_mask] = np.nan
    return smoothed.astype(np.float32, copy=False)


def find_local_maxima(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    peaks: list[int] = []
    for i in range(1, n - 1):
        if not valid_mask[i] or np.isnan(values[i]):
            continue
        left = values[i - 1]
        right = values[i + 1]
        if np.isnan(left) or np.isnan(right):
            continue
        if values[i] >= left and values[i] > right:
            peaks.append(i)
    if (
        n >= 1
        and valid_mask[0]
        and n > 1
        and not np.isnan(values[0])
        and not np.isnan(values[1])
        and values[0] > values[1]
    ):
        peaks.insert(0, 0)
    if (
        n >= 2
        and valid_mask[-1]
        and not np.isnan(values[-1])
        and not np.isnan(values[-2])
        and values[-1] >= values[-2]
    ):
        peaks.append(n - 1)
    return np.asarray(peaks, dtype=np.int64)


def compute_prominence(values: np.ndarray, peak_idx: int) -> tuple[float, int, int]:
    peak_value = float(values[peak_idx])

    left_min = peak_value
    left_base = peak_idx
    cursor = peak_idx
    while cursor > 0:
        cursor -= 1
        if np.isnan(values[cursor]):
            break
        left_min = min(left_min, float(values[cursor]))
        left_base = cursor
        if values[cursor] > peak_value:
            break

    right_min = peak_value
    right_base = peak_idx
    cursor = peak_idx
    while cursor < values.shape[0] - 1:
        cursor += 1
        if np.isnan(values[cursor]):
            break
        right_min = min(right_min, float(values[cursor]))
        right_base = cursor
        if values[cursor] > peak_value:
            break

    contour = max(left_min, right_min)
    prominence = peak_value - contour
    return float(prominence), int(left_base), int(right_base)


def compute_width(
    values: np.ndarray,
    peak_idx: int,
    prominence: float,
    stride: int,
    centers: np.ndarray,
) -> tuple[float, int, int]:
    peak_value = float(values[peak_idx])
    cutoff = peak_value - 0.5 * prominence

    left = peak_idx
    while (
        left > 0
        and not np.isnan(values[left - 1])
        and float(values[left - 1]) >= cutoff
    ):
        left -= 1

    right = peak_idx
    while (
        right < values.shape[0] - 1
        and not np.isnan(values[right + 1])
        and float(values[right + 1]) >= cutoff
    ):
        right += 1

    left_center = int(centers[left])
    right_center = int(centers[right])
    width_bp = max(stride, right_center - left_center + stride)
    return float(width_bp), int(left), int(right)


def suppress_nearby_peaks(
    candidate_indices: np.ndarray,
    values: np.ndarray,
    centers: np.ndarray,
    min_distance_bp: int,
) -> np.ndarray:
    order = candidate_indices[np.argsort(values[candidate_indices])[::-1]]
    kept: list[int] = []
    for idx in order.tolist():
        center = int(centers[idx])
        if all(
            abs(center - int(centers[kept_idx])) >= min_distance_bp for kept_idx in kept
        ):
            kept.append(idx)
    kept.sort()
    return np.asarray(kept, dtype=np.int64)


def write_bedgraph(
    path: Path, chrom: str, starts: np.ndarray, ends: np.ndarray, values: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for start, end, value in zip(starts.tolist(), ends.tolist(), values.tolist()):
            if np.isnan(value):
                continue
            handle.write(f"{chrom}\t{start}\t{end}\t{value:.8f}\n")


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.input_npz, allow_pickle=False)
    chromosomes = payload["chromosome"]
    chrom = str(chromosomes[0]) if chromosomes.size > 0 else "unknown"
    starts = payload["start"].astype(np.int64, copy=False)
    ends = payload["end"].astype(np.int64, copy=False)
    centers = payload["center"].astype(np.int64, copy=False)
    valid_mask = payload["valid_mask"].astype(bool, copy=False)
    stride = int(payload["stride"])
    raw_scores = payload[args.score_field].astype(np.float32, copy=False)

    sigma_windows = max(float(args.sigma_bp) / max(stride, 1), 1e-6)
    kernel = gaussian_kernel(sigma_windows=sigma_windows, truncate=args.truncate)
    smoothed_scores = smooth_signal(raw_scores, valid_mask, kernel)

    candidate_indices = find_local_maxima(smoothed_scores, valid_mask)
    candidate_indices = candidate_indices[
        np.asarray(
            [smoothed_scores[idx] >= args.min_peak_score for idx in candidate_indices],
            dtype=bool,
        )
    ]
    candidate_indices = suppress_nearby_peaks(
        candidate_indices=candidate_indices,
        values=smoothed_scores,
        centers=centers,
        min_distance_bp=args.min_distance_bp,
    )

    peaks: list[dict[str, float | int | str]] = []
    for idx in candidate_indices.tolist():
        prominence, left_base_idx, right_base_idx = compute_prominence(
            smoothed_scores, idx
        )
        if prominence < args.min_prominence:
            continue
        width_bp, left_width_idx, right_width_idx = compute_width(
            smoothed_scores,
            peak_idx=idx,
            prominence=prominence,
            stride=stride,
            centers=centers,
        )
        if width_bp < args.min_width_bp:
            continue

        peaks.append(
            {
                "chromosome": chrom,
                "peak_start": int(starts[left_width_idx]),
                "peak_end": int(ends[right_width_idx]),
                "summit_start": int(starts[idx]),
                "summit_end": int(ends[idx]),
                "summit_center": int(centers[idx]),
                "raw_score": float(raw_scores[idx]),
                "smoothed_score": float(smoothed_scores[idx]),
                "prominence": float(prominence),
                "width_bp": float(width_bp),
                "left_base_start": int(starts[left_base_idx]),
                "right_base_end": int(ends[right_base_idx]),
            }
        )

    write_bedgraph(
        args.output_prefix.with_suffix(".smoothed.bedgraph"),
        chrom,
        starts,
        ends,
        smoothed_scores,
    )

    peaks_tsv = args.output_prefix.with_suffix(".peaks.tsv")
    with peaks_tsv.open("w", encoding="utf-8") as handle:
        header = [
            "chromosome",
            "peak_start",
            "peak_end",
            "summit_start",
            "summit_end",
            "summit_center",
            "raw_score",
            "smoothed_score",
            "prominence",
            "width_bp",
            "left_base_start",
            "right_base_end",
        ]
        handle.write("\t".join(header) + "\n")
        for peak in peaks:
            handle.write("\t".join(str(peak[column]) for column in header) + "\n")

    np.savez_compressed(
        args.output_prefix.with_suffix(".smoothed_scores.npz"),
        chromosome=chromosomes,
        start=starts,
        end=ends,
        center=centers,
        valid_mask=valid_mask,
        raw_scores=raw_scores,
        smoothed_scores=smoothed_scores,
        stride=np.int64(stride),
    )

    metadata = {
        "input_npz": str(args.input_npz),
        "score_field": args.score_field,
        "sigma_bp": args.sigma_bp,
        "min_peak_score": args.min_peak_score,
        "min_prominence": args.min_prominence,
        "min_distance_bp": args.min_distance_bp,
        "min_width_bp": args.min_width_bp,
        "num_peaks": len(peaks),
    }
    args.output_prefix.with_suffix(".peak_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
