#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from workflow_config import actual_s3_result_prefix, plot_output_dir
from matplotlib.ticker import FuncFormatter

from fig1_style import PURE_BLACK, style_black_axes


RESULT_PREFIX = actual_s3_result_prefix()
OUT_DIR = plot_output_dir()
OUT_STEM = "ukb_downsampling_vcf_empirical_curve"

COLORS = {
    "Selected UKB subset": "#1768A8",
    "Random UKB downsample": "#8DBFEA",
}
LINESTYLES = {
    "Selected UKB subset": "-",
    "Random UKB downsample": "--",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot empirical VCF-based UKB downsampling coverage."
    )
    parser.add_argument("--result-prefix", type=pathlib.Path, default=RESULT_PREFIX)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--out-stem", default=OUT_STEM)
    return parser.parse_args()


def percent_tick(value: float, _pos: int | None = None) -> str:
    percent = value * 100.0
    if percent >= 1:
        return f"{percent:g}%"
    return f"{percent:.4f}".rstrip("0").rstrip(".") + "%"


def y_percent(value: float, _pos: int | None = None) -> str:
    return f"{value * 100:.0f}%"


def load_curve(result_prefix: pathlib.Path) -> pd.DataFrame:
    curve_path = result_prefix.with_suffix(".curve.tsv")
    if not curve_path.exists():
        raise FileNotFoundError(f"Missing {curve_path}")
    return pd.read_csv(curve_path, sep="\t")


def plot_curve(curve: pd.DataFrame, out_dir: pathlib.Path, out_stem: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "text.color": PURE_BLACK,
            "axes.labelcolor": PURE_BLACK,
            "axes.edgecolor": PURE_BLACK,
            "xtick.color": PURE_BLACK,
            "ytick.color": PURE_BLACK,
            "axes.linewidth": 1.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.7, 4.1), dpi=300)
    focus_max = min(0.02, float(curve["reference_haplotype_frequency_threshold"].max()))
    focus_min = float(curve["reference_haplotype_frequency_threshold"].min())
    marker_thresholds = [1e-2, 1e-3, 1e-4]

    target = "Selected UKB subset"
    endpoint_offsets = {
        "Selected UKB subset": -0.006,
    }
    for target in [target]:
        sub = curve[curve["target"] == target].sort_values(
            "reference_haplotype_frequency_threshold", ascending=False
        )
        sub = sub[
            (sub["reference_haplotype_frequency_threshold"] <= focus_max)
            & (sub["reference_haplotype_frequency_threshold"] >= focus_min)
        ]
        ax.plot(
            sub["reference_haplotype_frequency_threshold"],
            sub["threshold_weighted_coverage"],
            color=COLORS[target],
            linestyle=LINESTYLES[target],
            linewidth=2.8,
            label=target,
        )
        endpoint = sub.iloc[-1]
        if endpoint["threshold_weighted_coverage"] < 0.995:
            ax.text(
                endpoint["reference_haplotype_frequency_threshold"] * 0.92,
                endpoint["threshold_weighted_coverage"] + endpoint_offsets[target],
                f"{endpoint['threshold_weighted_coverage'] * 100:.1f}%",
                color=COLORS[target],
                ha="right",
                va="center",
                fontsize=12,
            )

    selected_line = curve[curve["target"] == "Selected UKB subset"]
    for threshold in marker_thresholds:
        if threshold < focus_min or threshold > focus_max:
            continue
        nearest = selected_line.iloc[
            (selected_line["reference_haplotype_frequency_threshold"] - threshold)
            .abs()
            .argmin()
        ]
        ax.axvline(threshold, color="#B7C7DA", linestyle=":", linewidth=1.0, zorder=0)
        ax.scatter(
            [nearest["reference_haplotype_frequency_threshold"]],
            [nearest["threshold_weighted_coverage"]],
            s=32,
            color=COLORS["Selected UKB subset"],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.text(
            threshold,
            min(0.997, nearest["threshold_weighted_coverage"] + 0.006),
            f"{nearest['threshold_weighted_coverage'] * 100:.1f}%",
            color=PURE_BLACK,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Full UKB haplotype frequency threshold (log scale)")
    ax.set_ylabel("VCF-measured weighted coverage")
    ax.xaxis.set_major_formatter(FuncFormatter(percent_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(y_percent))
    ax.set_xlim(focus_max * 1.08, focus_min * 0.86)
    y_min = max(0.0, curve["threshold_weighted_coverage"].min() - 0.03)
    ax.set_ylim(y_min, 1.01)
    style_black_axes(
        ax, labelsize=13, tick_labelsize=11, spine_width=1.0, tick_width=1.0
    )
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    fig.tight_layout()

    for ext in ["png", "pdf", "svg"]:
        out = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    curve = load_curve(args.result_prefix)
    plot_curve(curve, args.out_dir, args.out_stem)
    summary_path = args.result_prefix.with_suffix(".summary.tsv")
    if summary_path.exists():
        print(pd.read_csv(summary_path, sep="\t").to_string(index=False))


if __name__ == "__main__":
    main()
