"""Shared Fig.4 e/f renderer for compact HAR and HAQER panels."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.matplotlib_common import configure_matplotlib
from common.panel_runtime import panel_tables_dir
from Fig4._backend import BACKEND
from matplotlib.lines import Line2D

WIDTH_MM, HEIGHT_MM = 88.0, 52.0
DATA_FILES = {
    "HAR": "S37_classification_of_human_accelerated_regions_across_negative_controls.csv",
    "HAQER": "S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv",
}
NEGATIVE_SETS = {
    "HAR": [2, 5, 6, 8, 10],
    "HAQER": [2, 7, 10, 13, 19],
}
X_LIMITS = {"HAR": (0.10, 0.95), "HAQER": (0.15, 0.95)}
X_TICKS = {"HAR": [0.2, 0.4, 0.6, 0.8, 0.9], "HAQER": [0.2, 0.4, 0.6, 0.8, 0.9]}
BLACK = "#000000"
WHITE = "#FFFFFF"
NEUTRAL_LIGHT = "#D7DCE2"
HEVO_WASH = "#FBF2EA"


def _negative_set_number(value: str) -> int:
    match = re.search(r"negset(\d+)$", str(value))
    if match is None:
        raise ValueError(f"Cannot parse negative-set number from {value!r}")
    return int(match.group(1))


def load_dataset(
    tables_dir: Path | None,
    dataset: str,
    *,
    representatives_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Validate source rows, normalize replicates, and rank models from MCC."""

    if dataset not in DATA_FILES:
        raise ValueError(f"Unknown dataset {dataset!r}")
    root = panel_tables_dir("Fig4", tables_dir)
    path = root / DATA_FILES[dataset]
    df = pd.read_csv(path)
    required = {"model", "benchmark_raw", "mcc_mean"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["mcc_mean"] = pd.to_numeric(df["mcc_mean"], errors="coerce")
    mcc_values = df["mcc_mean"].to_numpy(dtype=float)
    if (
        not np.isfinite(mcc_values).all()
        or ((mcc_values < -1.0) | (mcc_values > 1.0)).any()
    ):
        raise ValueError(
            f"{path} contains missing, non-finite, or out-of-range MCC values"
        )
    benchmark_raw = df["benchmark_raw"].astype(str)
    historical_rows = benchmark_raw.str.contains(r"negset\d+$", regex=True)
    canonical_prefix = f"{dataset}_combined"
    canonical_rows = benchmark_raw.str.fullmatch(
        rf"{re.escape(canonical_prefix)}_rep[1-5]"
    )
    expected_sets = NEGATIVE_SETS[dataset]
    expected_reps = [f"rep{index}" for index in range(1, len(expected_sets) + 1)]
    set_to_rep = {
        negative_set: replicate
        for negative_set, replicate in zip(expected_sets, expected_reps)
    }
    standard = df.loc[historical_rows | canonical_rows].copy()
    historical = historical_rows.loc[standard.index]
    standard["replicate"] = pd.Series(index=standard.index, dtype="object")
    historical_set_ids = standard.loc[historical, "benchmark_raw"].map(
        _negative_set_number
    )
    unexpected_sets = sorted(set(historical_set_ids) - set(expected_sets))
    if unexpected_sets:
        raise ValueError(
            f"Unexpected {dataset} negative-set IDs in {path}: {unexpected_sets}"
        )
    standard.loc[historical, "replicate"] = historical_set_ids.map(set_to_rep)
    canonical_rep_numbers = (
        standard.loc[~historical, "benchmark_raw"]
        .astype(str)
        .str.extract(r"_rep([1-5])$")[0]
        .astype(int)
    )
    standard.loc[~historical, "replicate"] = canonical_rep_numbers.map(
        {index: f"rep{index}" for index in range(1, len(expected_sets) + 1)}
    )
    auxiliary = df.loc[~(historical_rows | canonical_rows)].copy()
    if not auxiliary.empty:
        auxiliary_ok = (
            auxiliary["model"].eq("hEvo-large")
            & auxiliary["benchmark_raw"]
            .astype(str)
            .str.fullmatch(rf"{re.escape(dataset)}_rep[1-5]")
        ) | (
            auxiliary["model"].isin({"hEvo-open-base", "hEvo-open-large"})
            & auxiliary["benchmark_raw"]
            .astype(str)
            .str.fullmatch(rf"{re.escape(dataset)}_combined_rep[1-5]")
        )
        if not bool(auxiliary_ok.all()):
            bad = (
                auxiliary.loc[~auxiliary_ok, "benchmark_raw"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(f"Unsupported auxiliary HAR benchmark rows: {bad}")
        rep_numbers = (
            auxiliary["benchmark_raw"]
            .astype(str)
            .str.extract(r"rep([1-5])$")[0]
            .astype(int)
        )
        if sorted(rep_numbers.unique().tolist()) != list(
            range(1, len(expected_sets) + 1)
        ):
            raise ValueError(
                f"Auxiliary replicate numbering is incomplete for {dataset}"
            )
        auxiliary["replicate"] = rep_numbers.map(
            {
                rep_index: f"rep{rep_index}"
                for rep_index in range(1, len(expected_sets) + 1)
            }
        )
    # Normalize both naming conventions directly to the ordered rep1-rep5 labels.
    df = pd.concat([standard, auxiliary], ignore_index=True)
    if sorted(df["replicate"].unique().tolist()) != expected_reps:
        raise ValueError(
            f"Unexpected replicate labels for {dataset}: {sorted(df['replicate'].unique())}"
        )
    model_count = int(df["model"].nunique())
    if len(df) != model_count * len(expected_reps) or df["replicate"].nunique() != len(
        expected_reps
    ):
        raise ValueError(f"Unexpected shape for {dataset}: {df.shape}")
    if df.duplicated(["model", "replicate"]).any():
        raise ValueError(f"Duplicate model/negative-set rows in {path}")
    df["benchmark_raw"] = df["replicate"]
    df = df.drop(columns=["replicate"])

    all_models = (
        df.groupby("model", as_index=False, sort=True)["mcc_mean"]
        .mean()
        .rename(columns={"mcc_mean": "mean_mcc_for_sort"})
    )

    all_models["model_display"] = all_models["model"].map(BACKEND.display)
    all_models["plot_label"] = all_models["model"].map(BACKEND.compact_display)
    all_models.loc[all_models["model"].eq("NT_2.5b_1kG"), "plot_label"] = (
        "NTv1-2.5B-1kG"
    )
    all_models["family"] = all_models["model"].map(BACKEND.model_family)
    all_models["algorithm_identity"] = all_models["model"].map(
        BACKEND.algorithm_identity
    )
    all_models = all_models.sort_values(
        ["mean_mcc_for_sort", "model_display"], ascending=[False, True]
    ).reset_index(drop=True)
    representatives = BACKEND.representative_rows(all_models, "mean_mcc_for_sort")
    if representatives.empty or representatives["model"].duplicated().any():
        raise ValueError(f"Representative model selection is invalid for {dataset}")
    if not set(representatives["model"]).issubset(set(all_models["model"])):
        raise ValueError(
            f"Representative models are not covered by source data for {dataset}"
        )
    if representatives_only:
        model_rows = representatives
    else:
        # Complete panels retain the same performance-first ordering as the
        # compact Fig. 4 representative panels.
        model_rows = all_models.sort_values(
            ["mean_mcc_for_sort", "model_display"], ascending=[False, True]
        ).reset_index(drop=True)
        model_rows["plot_rank"] = np.arange(1, len(model_rows) + 1)
    df = df.loc[df["model"].isin(model_rows["model"])].copy()
    return df, model_rows.reset_index(drop=True), path


def _family_color(family: str) -> str:
    return BACKEND.MODEL_FAMILY_STYLE[family]["color"]


def draw_dataset(
    fig: mpl.figure.Figure,
    slot,
    tables_dir: Path | None,
    dataset: str,
    *,
    letter: str,
    integrated: bool,
    model_roster: list[str] | None = None,
) -> mpl.axes.Axes:
    """Draw one compact range-and-mean panel."""

    configure_matplotlib()
    df, model_rows, _ = load_dataset(
        tables_dir, dataset, representatives_only=model_roster is None
    )
    if model_roster is not None:
        missing = sorted(set(model_roster).difference(model_rows["model"]))
        if missing:
            raise ValueError(
                f"{dataset} source table is missing reference models: {missing}"
            )
        model_rows = model_rows.loc[model_rows["model"].isin(model_roster)].copy()
        model_rows = model_rows.sort_values(
            ["mean_mcc_for_sort", "model_display"], ascending=[False, True]
        ).reset_index(drop=True)
        df = df.loc[df["model"].isin(model_roster)].copy()
    ax = fig.add_subplot(slot)
    y = np.arange(len(model_rows))
    # Both panels keep the model labels, matching the supplied reference and
    # making the two datasets independently readable.
    show_labels = True
    label_size = 6.0 if integrated else 4.8

    for row_index, row in model_rows.iterrows():
        family = row["family"]
        color = _family_color(family)
        if family == "hEVO":
            ax.axhspan(
                row_index - 0.46, row_index + 0.46, color=HEVO_WASH, lw=0, zorder=0
            )
        values = (
            df.loc[df["model"].eq(row["model"]), "mcc_mean"].astype(float).to_numpy()
        )
        lower, upper = float(values.min()), float(values.max())
        mean = float(row["mean_mcc_for_sort"])
        marker = BACKEND.MODEL_FAMILY_STYLE[family]["marker"]
        # The five negative-set values remain visible as open family-shaped
        # points; the horizontal span summarizes their observed range.
        ax.plot(
            [lower, upper],
            [row_index, row_index],
            color=color,
            alpha=0.62,
            lw=0.90,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            [lower, lower],
            [row_index - 0.18, row_index + 0.18],
            color=color,
            alpha=0.62,
            lw=0.55,
            zorder=1,
        )
        ax.plot(
            [upper, upper],
            [row_index - 0.18, row_index + 0.18],
            color=color,
            alpha=0.62,
            lw=0.55,
            zorder=1,
        )
        ax.scatter(
            values,
            np.full(values.shape, row_index),
            s=12.5,
            marker=marker,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=0.52,
            zorder=2,
        )
        ax.scatter(
            mean,
            row_index,
            s=28.0,
            marker=marker,
            facecolor=color,
            edgecolor=BLACK,
            linewidth=0.52,
            zorder=3,
        )
        if family == "hEVO":
            label_x = max(upper + 0.015, mean + 0.020)
            ax.text(
                label_x,
                row_index,
                f"{mean:.3f}",
                fontsize=4.5,
                color=BLACK,
                ha="left",
                va="center",
                zorder=4,
            )

    ax.set_ylim(len(model_rows), -0.5)
    # Match the reference PDF's dataset-specific MCC framing: the lower
    # bound leaves room for the lowest observed interval without adding a
    # misleading 0.0 tick to either panel.
    ax.set_xlim(*X_LIMITS[dataset])
    ax.set_xticks(X_TICKS[dataset])
    ax.set_xlabel("MCC", fontsize=6.2 if integrated else 5.7, labelpad=1.7)
    if show_labels:
        ax.set_yticks(y)
        ax.set_yticklabels(model_rows["plot_label"], fontsize=label_size)
        for tick, family in zip(ax.get_yticklabels(), model_rows["family"]):
            tick.set_color(BLACK)
            tick.set_fontweight("bold" if family == "hEVO" else "normal")
        # Model names are explicit on the left; reserve the margin for labels.
    else:
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.tick_params(
        axis="x",
        labelsize=5.7 if integrated else 5.2,
        length=2.0,
        width=0.5,
        pad=1.74 if integrated else 1.1,
    )
    ax.tick_params(axis="y", length=1.7, width=0.45, pad=1.27 if integrated else 1.1)
    if integrated:
        ax.xaxis.labelpad = 2.34
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(False)
    BACKEND.add_panel_letter(
        fig, ax, letter, dx=-0.065 if show_labels else -0.025, dy=0.005
    )
    return ax


def add_range_mean_key(
    fig: mpl.figure.Figure, *, y: float = 0.455, fontsize: float = 4.5
) -> None:
    """Explain the compact range-and-mean encoding on the A4 page."""

    handles = [
        Line2D(
            [0], [0], color="#69727D", lw=1.2, label="MCC range across 5 negative sets"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#69727D",
            markeredgecolor=WHITE,
            markersize=3.6,
            label="Mean MCC",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=fontsize,
        handletextpad=0.30,
        columnspacing=0.90,
        bbox_to_anchor=(0.53, y),
    )


def build_panel(
    tables_dir: Path | None = None,
    *,
    figure=None,
    slot=None,
    integrated=False,
    dataset: str,
    letter: str,
    model_roster: list[str] | None = None,
):
    root = panel_tables_dir("Fig4", tables_dir)
    if figure is not None and slot is not None:
        return draw_dataset(
            figure,
            slot,
            root,
            dataset,
            letter=letter,
            integrated=integrated,
            model_roster=model_roster,
        )
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    gs = fig.add_gridspec(1, 1, left=0.27, right=0.97, bottom=0.13, top=0.92)
    ax = draw_dataset(
        fig,
        gs[0, 0],
        root,
        dataset,
        letter=letter,
        integrated=False,
        model_roster=model_roster,
    )
    BACKEND.add_panel_header(fig, (ax,), dataset)
    return fig
