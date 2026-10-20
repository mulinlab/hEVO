#!/usr/bin/env python3
"""Shared data readers and Matplotlib renderers for hEvo Fig.5 ablation."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter
from PIL import Image
from pypdf import PdfReader

FIG5_SCRIPTS = Path(__file__).resolve().parent
SCRIPTS_ROOT = FIG5_SCRIPTS.parents[0]
REPO_ROOT = FIG5_SCRIPTS.parents[2]
if str(SCRIPTS_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS_ROOT))

from common.matplotlib_common import configure_matplotlib
from common.model_names import canonical_model_name
TABLE_SCRIPTS = REPO_ROOT / "tables" / "scripts"
if str(TABLE_SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(TABLE_SCRIPTS))
from deepsea_run_selection import active_deepsea_rows

S41_NAME = "S41_effects_of_training_corpus_and_windowing_on_18_sequence_classification_tasks.csv"
S42_NAME = "S42_effects_of_corpus_windowing_and_allelic_refinement_on_clinvar_scoring.csv"
S43_NAME = "S43_model_level_prediction_performance_and_computational_resource_use.csv"
S47_NAME = "S47_architecture_training_scope_and_provenance_of_evaluated_models.csv"
GROUP_ORDER = ("Histone Markers", "Regulatory Annotation", "Splice Site Annotation")
GENOMIC_BENCHMARK_TASKS = (
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
)
S41_CONDITION_ORDER = ("A09", "A11", "A01", "A03", "A05", "A07")
S42_CONDITION_ORDER = (
    "A09",
    "A10",
    "A11",
    "A12",
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
)
S42_STRATA = ("all", "coding", "noncoding")
COHORT_ORDER = ("1000G_3K", "UKB_500K", "PopReduce_60K")
WINDOW_ORDER = ("NoSW", "SW")


COLORS = {
    "hEvo": "#B35C1E",
    "NT": "#65717F",
    "Evo 2": "#0E7C86",
    "GPN": "#0B2E83",
    "Other": "#8B96A5",
    "positive": "#174FA6",
    "negative": "#C43B3B",
    "track": "#D8DDE3",
    "text": "#111111",
}
_PARETO_FAMILY_COLORS = {
    "hEvo": "#B35C1E",
    "NT v2": "#65717F",
    "NT v3": "#7D8793",
    "DNABert": "#9AA4AF",
    "GENA-LM": "#8792A0",
    "Mistral-DNA": "#A7AFB8",
    "Caduceus": "#9AA6B5",
    "GPN": "#0B2E83",
    "GENERator": "#7F8B99",
    "Evo 2": "#0E7C86",
    "SpeciesLM": "#8994A0",
    "HyenaDNA": "#A1AAB5",
    "AIDO.DNA": "#8C98A5",
    "Genos": "#7F8B99",
    "GROVER": "#A0A9B3",
    "Other": "#8B96A5",
}


def display_name(model: str) -> str:
    return canonical_model_name(model)


def family_color(family: str) -> str:
    return _PARETO_FAMILY_COLORS.get(family, _PARETO_FAMILY_COLORS["Other"])


def pareto_frontier(
    x: np.ndarray, y: np.ndarray, x_higher_is_better: bool
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)[::-1] if not x_higher_is_better else np.argsort(x)
    current = -np.inf
    points = []
    for index in order:
        if y[index] > current:
            points.append((x[index], y[index]))
            current = y[index]
    points.sort(key=lambda pair: pair[0])
    return np.array([pair[0] for pair in points]), np.array(
        [pair[1] for pair in points]
    )


# Use a slightly enlarged Fig.1–Fig.4-compatible typography tier for the
# dense Fig5 A4 page; values are Matplotlib points.
PANEL_LETTER_PT = 14.0
PANEL_TITLE_PT = 10.0
FACET_PT = 5.8
LABEL_PT = 5.6
FAMILY_PT = 5.6
LEGEND_PT = 5.6
AXIS_PT = 6.0
TICK_PT = 5.2
SUBTITLE_PT = 5.5
# Fig5b is an internal-ablation comparison.  Use one shared, zoomed display
# scale for every task slot so small condition differences remain visible,
# while retaining the complete source confidence intervals.
# b is an internal comparison matrix: each task column gets its own display
# interval, derived from that task's observed source CI range.
RANK_STYLES = {
    1: ("D", "#C43B3B", "rank 1"),
    2: ("s", "#2E7D55", "rank 2"),
    3: ("^", "#2864B4", "rank 3"),
}
TASK_SHORT = {
    "promoter_all": "promoter",
    "promoter_no_tata": "promoter\nno-TATA",
    "promoter_tata": "promoter\nTATA",
    "enhancers": "enhancers",
    "enhancers_types": "enhancer\ntypes",
    "splice_sites_all": "splice\nall",
    "splice_sites_acceptors": "splice\nacceptors",
    "splice_sites_donors": "splice\ndonors",
}


def tables_root(tables_dir: Path | None = None) -> Path:
    if tables_dir is not None:
        return Path(tables_dir).resolve()
    env = os.environ.get("HEVO_TABLES_DIR")
    return Path(env).resolve() if env else REPO_ROOT / "tables"


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def read_s41(tables_dir: Path | None = None) -> pd.DataFrame:
    path = tables_root(tables_dir) / S41_NAME
    frame = pd.read_csv(path)
    required = {
        "condition_id",
        "condition_label",
        "training_dataset",
        "windowing",
        "model_size",
        "task_group",
        "dataset",
        "metric",
        "metric_mean",
        "ci_lower",
        "ci_upper",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    frame = _numeric(
        frame,
        (
            "metric_mean",
            "ci_lower",
            "ci_upper",
            "num_samples",
            "num_positive",
            "num_negative",
        ),
    )
    if (
        len(frame) != 108
        or frame["condition_id"].nunique() != 6
        or frame["dataset"].nunique() != 18
    ):
        raise ValueError(
            f"{path.name}: expected 108 rows, 6 conditions and 18 tasks; got {frame.shape}"
        )
    if set(frame["condition_id"]) != set(S41_CONDITION_ORDER):
        raise ValueError(f"{path.name}: unexpected condition IDs")
    if frame["metric"].nunique() != 1 or frame["metric"].iloc[0] != "MCC":
        raise ValueError(f"{path.name}: metric must be MCC only")
    if frame.duplicated(["condition_id", "dataset"]).any():
        raise ValueError(f"{path.name}: duplicate condition-task cells")
    if frame[["metric_mean", "ci_lower", "ci_upper"]].isna().any().any():
        raise ValueError(f"{path.name}: missing MCC or interval values")
    if (frame["ci_lower"] > frame["metric_mean"]).any() or (
        frame["metric_mean"] > frame["ci_upper"]
    ).any():
        raise ValueError(f"{path.name}: invalid CI ordering")
    counts = frame.groupby("condition_id")["dataset"].nunique()
    if not (counts == 18).all():
        raise ValueError(f"{path.name}: not every condition has 18 tasks")
    group_values = set(frame["task_group"].unique())
    if group_values != set(GROUP_ORDER):
        raise ValueError(f"{path.name}: unexpected task groups {sorted(group_values)}")
    return frame


def read_s42(tables_dir: Path | None = None) -> pd.DataFrame:
    path = tables_root(tables_dir) / S42_NAME
    frame = pd.read_csv(path)
    required = {
        "condition_id",
        "condition_label",
        "stage",
        "training_dataset",
        "windowing",
        "dataset",
        "metric",
        "metric_mean",
        "ci_lower",
        "ci_upper",
        "evidence_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    frame = _numeric(
        frame,
        (
            "metric_mean",
            "ci_lower",
            "ci_upper",
            "num_samples",
            "num_positive",
            "num_negative",
            "baseline",
        ),
    )
    if (
        len(frame) != 72
        or frame["condition_id"].nunique() != 12
        or frame["dataset"].nunique() != 3
    ):
        raise ValueError(
            f"{path.name}: expected 72 rows, 12 conditions and 3 strata; got {frame.shape}"
        )
    if set(frame["metric"]) != {"AUROC", "AUPRC"}:
        raise ValueError(f"{path.name}: expected AUROC and AUPRC")
    if set(frame["dataset"]) != set(S42_STRATA):
        raise ValueError(f"{path.name}: unexpected ClinVar strata")
    if frame.duplicated(["condition_id", "dataset", "metric"]).any():
        raise ValueError(f"{path.name}: duplicate condition-stratum-metric cells")
    allowed_status = {"PASS", "NOT_VERIFIABLE"}
    if not set(frame["evidence_status"]).issubset(allowed_status):
        raise ValueError(f"{path.name}: unexpected evidence status")
    if (frame.groupby(["condition_id", "dataset"])["evidence_status"].nunique() != 1).any():
        raise ValueError(f"{path.name}: inconsistent evidence status within metric pairs")
    passed = frame["evidence_status"].eq("PASS")
    numeric_fields = [
        "metric_mean", "ci_lower", "ci_upper", "num_samples",
        "num_positive", "num_negative", "baseline",
    ]
    if frame.loc[passed, numeric_fields].isna().any().any():
        raise ValueError(f"{path.name}: PASS rows contain missing values")
    if frame.loc[~passed, numeric_fields].notna().any().any():
        raise ValueError(f"{path.name}: NOT_VERIFIABLE rows carry consumable values")
    if (frame.loc[passed, "ci_lower"] > frame.loc[passed, "metric_mean"]).any() or (
        frame.loc[passed, "metric_mean"] > frame.loc[passed, "ci_upper"]
    ).any():
        raise ValueError(f"{path.name}: invalid CI ordering")
    expected_counts = {
        "all": (264192, 84580, 179612),
        "coding": (144285, 70787, 73498),
        "noncoding": (119907, 13793, 106114),
    }
    expected_baselines = {
        "all": 0.3201459544573643,
        "coding": 0.4906053990366289,
        "noncoding": 0.1150308155487169,
    }
    for dataset, expected in expected_counts.items():
        selected = frame.loc[passed & frame["dataset"].eq(dataset)]
        if selected.empty:
            continue
        observed = set(
            selected[["num_samples", "num_positive", "num_negative"]]
            .astype(int).itertuples(index=False, name=None)
        )
        if observed != {expected}:
            raise ValueError(
                f"{path.name}: {dataset} rows do not match the ClinVar stratum counts"
            )
        if not np.isclose(selected["baseline"], expected_baselines[dataset]).all():
            raise ValueError(
                f"{path.name}: {dataset} rows do not match the ClinVar stratum baseline"
            )
    return frame


def condition_label(row: pd.Series) -> str:
    cohort = {
        "UKB_500K": "UKB 500K",
        "PopReduce_60K": "PopReduce 60K",
        "1000G_3K": "1000G 3K",
    }[str(row["training_dataset"])]
    return f"{cohort} · {row['windowing']}"


def s41_task_order(frame: pd.DataFrame) -> list[str]:
    return [
        task
        for group in GROUP_ORDER
        for task in frame.loc[
            frame["task_group"].eq(group), "dataset"
        ].drop_duplicates()
    ]


def s41_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    tasks = s41_task_order(frame)
    out = frame[frame["condition_id"].isin(S41_CONDITION_ORDER)].copy()
    out["condition_order"] = out["condition_id"].map(
        {x: i for i, x in enumerate(S41_CONDITION_ORDER)}
    )
    out["task_order"] = out["dataset"].map({x: i for i, x in enumerate(tasks)})
    out = out.sort_values(["condition_order", "task_order"], kind="stable").copy()
    out["rank"] = (
        out.groupby("dataset")["metric_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    means = (
        out.groupby("condition_id", as_index=False)["metric_mean"]
        .mean()
        .rename(columns={"metric_mean": "mean_mcc"})
    )
    out = out.merge(means, on="condition_id", how="left", validate="many_to_one")
    return out


def mcc_task_display_ranges(frame: pd.DataFrame) -> dict[str, list[float]]:
    """Return independent, CI-safe display ranges for each NT task.

    The ranges are only a visual mapping; source MCC/CI values are unchanged.
    Outward rounding to 0.01 makes the local scales stable and readable.
    """
    ranges: dict[str, list[float]] = {}
    for task, group in frame.groupby("dataset", sort=False):
        source_lo = float(group["ci_lower"].min())
        source_hi = float(group["ci_upper"].max())
        span = max(source_hi - source_lo, 0.02)
        pad = max(0.01, 0.12 * span)
        lo = max(0.0, float(np.floor((source_lo - pad) * 100.0) / 100.0))
        hi = min(1.0, float(np.ceil((source_hi + pad) * 100.0) / 100.0))
        if hi <= lo:
            hi = min(1.0, lo + 0.05)
        ranges[str(task)] = [round(lo, 2), round(hi, 2)]
    return ranges


def factor_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("SW − NoSW | 1000G 3K", "1000G_3K", "SW", "1000G_3K", "NoSW"),
        ("1000G 3K − UKB 500K | NoSW", "1000G_3K", "NoSW", "UKB_500K", "NoSW"),
        ("1000G 3K − UKB 500K | SW", "1000G_3K", "SW", "UKB_500K", "SW"),
        ("SW − NoSW | UKB 500K", "UKB_500K", "SW", "UKB_500K", "NoSW"),
        ("SW − NoSW | PopReduce 60K", "PopReduce_60K", "SW", "PopReduce_60K", "NoSW"),
        (
            "PopReduce 60K − UKB 500K | NoSW",
            "PopReduce_60K",
            "NoSW",
            "UKB_500K",
            "NoSW",
        ),
        ("PopReduce 60K − UKB 500K | SW", "PopReduce_60K", "SW", "UKB_500K", "SW"),
    ]
    pieces = []
    for label, left_cohort, left_window, right_cohort, right_window in comparisons:
        left = frame[
            (frame["training_dataset"] == left_cohort)
            & (frame["windowing"] == left_window)
        ].set_index("dataset")
        right = frame[
            (frame["training_dataset"] == right_cohort)
            & (frame["windowing"] == right_window)
        ].set_index("dataset")
        joined = left[["task_group", "metric_mean"]].join(
            right[["metric_mean"]], lsuffix="_left", rsuffix="_right", how="inner"
        )
        joined["delta"] = joined["metric_mean_left"] - joined["metric_mean_right"]
        joined["comparison"] = label
        joined["dataset"] = joined.index
        pieces.append(joined.reset_index(drop=True))
    out = pd.concat(pieces, ignore_index=True)
    out["comparison_order"] = out["comparison"].map(
        {x[0]: i for i, x in enumerate(comparisons)}
    )
    return out


def s42_auprc(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[
        frame["metric"].eq("AUPRC") & frame["evidence_status"].eq("PASS")
    ].copy()
    out["condition_order"] = out["condition_id"].map(
        {x: i for i, x in enumerate(S42_CONDITION_ORDER)}
    )
    out["stratum_order"] = out["dataset"].map({x: i for i, x in enumerate(S42_STRATA)})
    return out.sort_values(["stratum_order", "condition_order"], kind="stable")


def _public_path(path: Path) -> str:
    """Return a repository-safe logical path for figure metadata."""
    resolved = Path(path).resolve()
    try:
        return f"<benchmark-root>/{resolved.relative_to(REPO_ROOT).as_posix()}"
    except ValueError:
        marker = "/results/Probe_NT_Pareto"
        text = resolved.as_posix()
        if marker in text:
            suffix = text.split(marker, 1)[1].lstrip("/")
            root = "<source-project-root>/results/Probe_NT_Pareto"
            return f"{root}/{suffix}" if suffix else root
        return resolved.name


def _task_label(task: str) -> str:
    return TASK_SHORT.get(task, task)


def _setup_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(width=0.55, length=2.2, labelsize=TICK_PT, pad=2)


def draw_panel_heading(
    ax: plt.Axes,
    letter: str,
    title: str,
    subtitle: str | None = None,
    *,
    y: float = 1.04,
) -> None:
    """Draw a Fig.1–Fig.4-style letter/title pair in axes coordinates."""

    ax.text(
        0.0,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_PT,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.text(
        0.080,
        y + 0.002,
        title,
        transform=ax.transAxes,
        fontsize=PANEL_TITLE_PT,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    if subtitle:
        ax.text(
            0.0,
            y - 0.070,
            subtitle,
            transform=ax.transAxes,
            fontsize=SUBTITLE_PT,
            va="bottom",
            ha="left",
            clip_on=False,
        )


def draw_a4_heading(
    fig: plt.Figure,
    letter: str,
    title: str,
    y_mm: float,
    subtitle: str | None = None,
    x_mm: float = 11.0,
) -> None:
    """Draw an A4 heading with an explicit left anchor and fixed baseline."""

    x = x_mm / 210.0
    fig.text(
        x,
        y_mm / 297.0,
        letter,
        fontsize=PANEL_LETTER_PT,
        fontweight="bold",
        va="top",
        ha="left",
    )
    if title:
        fig.text(
            (x_mm + 7.0) / 210.0,
            (y_mm + 0.2) / 297.0,
            title,
            fontsize=PANEL_TITLE_PT,
            fontweight="bold",
            va="top",
            ha="left",
        )
    if subtitle:
        fig.text(
            x, (y_mm - 5.0) / 297.0, subtitle, fontsize=SUBTITLE_PT, va="top", ha="left"
        )


def draw_nt_benchmark(
    ax: plt.Axes,
    frame: pd.DataFrame,
    show_header: bool = True,
    letter_pt: float | None = None,
) -> None:
    """Draw the three-cohort Mean-MCC comparison shown in reference panel b."""

    matrix = s41_matrix(frame)
    expected = {
        ("PopReduce_60K", "NoSW"),
        ("PopReduce_60K", "SW"),
        ("1000G_3K", "NoSW"),
        ("1000G_3K", "SW"),
        ("UKB_500K", "NoSW"),
        ("UKB_500K", "SW"),
    }
    observed = set(
        matrix.drop_duplicates(["condition_id"])[
            ["training_dataset", "windowing"]
        ].itertuples(index=False, name=None)
    )
    if observed != expected:
        raise ValueError(
            f"S41 cohort/window roster does not match panel b: {sorted(observed)}"
        )

    # The reference crop is 1030 x 525 px.  Use the same square millimetre
    # coordinate scale for the standalone panel and its A4 placement.
    width_mm = 183.0
    height_mm = 93.3
    ref_px = 1030.0
    px_mm = width_mm / ref_px

    def xpx(value: float) -> float:
        return value * px_mm

    def ypx(value: float) -> float:
        return height_mm - value * px_mm

    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.axis("off")

    ink = COLORS["text"]
    orange = COLORS["hEvo"]
    gray = "#8E9CAF"
    light_gray = "#9DAABA"
    label_pt = 6.2
    highlighted_label_pt = 6.2
    title_pt = 8.0
    axis_title_pt = 7.0
    panel_letter_pt = 14.0 if letter_pt is None else letter_pt
    value_pt = 6.2

    if show_header:
        ax.text(
            xpx(14),
            ypx(12),
            "b",
            fontsize=panel_letter_pt,
            fontweight="bold",
            ha="left",
            va="top",
            color=ink,
            clip_on=False,
        )
    ax.text(
        xpx(522),
        ypx(19),
        "NT benchmark",
        fontsize=title_pt,
        fontweight="bold",
        ha="center",
        va="top",
        color=ink,
    )

    # Legend markers and labels.
    legend_y = ypx(110)
    ax.plot(
        xpx(344),
        legend_y,
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgecolor=orange,
        markeredgewidth=1.0,
        linestyle="none",
    )
    ax.text(
        xpx(372),
        legend_y,
        "No sliding",
        fontsize=label_pt,
        ha="left",
        va="center",
        color=ink,
    )
    ax.plot(
        xpx(734),
        legend_y,
        marker="o",
        markersize=4.8,
        markerfacecolor=orange,
        markeredgecolor=orange,
        markeredgewidth=0.8,
        linestyle="none",
    )
    ax.text(
        xpx(762),
        legend_y,
        "Sliding",
        fontsize=label_pt,
        ha="left",
        va="center",
        color=ink,
    )

    row_specs = [
        ("PopReduce_60K", "hEVO selected (60K)", 180, True),
        ("1000G_3K", "1000G (3K)", 247, False),
        ("UKB_500K", "UKB (500K)", 315, False),
    ]
    mean_lookup = (
        matrix.groupby(["training_dataset", "windowing"], as_index=False)["metric_mean"]
        .mean()
        .set_index(["training_dataset", "windowing"])["metric_mean"]
    )
    axis_left_px, axis_right_px = 428.0, 995.0
    axis_y = ypx(375)

    def mean_x(value: float) -> float:
        return xpx(
            axis_left_px + (float(value) - 0.52) / 0.08 * (axis_right_px - axis_left_px)
        )

    for cohort, label, row_y_px, highlighted in row_specs:
        y = ypx(row_y_px)
        ax.text(
            xpx(40),
            ypx(row_y_px + 2),
            label,
            fontsize=highlighted_label_pt if highlighted else label_pt,
            fontweight="bold" if highlighted else "normal",
            ha="left",
            va="center",
            color=orange if highlighted else ink,
        )
        no_sliding = float(mean_lookup.loc[(cohort, "NoSW")])
        sliding = float(mean_lookup.loc[(cohort, "SW")])
        x_no, x_sw = mean_x(no_sliding), mean_x(sliding)
        color = orange if highlighted else gray
        ax.plot(
            [x_no, x_sw], [y, y], color=color, linewidth=0.65, solid_capstyle="round"
        )
        ax.plot(
            x_no,
            y,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            linestyle="none",
        )
        ax.plot(
            x_sw,
            y,
            marker="o",
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            linestyle="none",
        )
        value_y = ypx(row_y_px - 18)
        ax.text(
            x_no,
            value_y,
            f"{no_sliding:.3f}",
            fontsize=value_pt,
            ha="center",
            va="bottom",
            color=ink,
        )
        ax.text(
            x_sw,
            value_y,
            f"{sliding:.3f}",
            fontsize=value_pt,
            ha="center",
            va="bottom",
            color=ink,
        )

    # Reference x axis and ticks.
    ax.plot(
        [xpx(axis_left_px), xpx(axis_right_px)],
        [axis_y, axis_y],
        color=ink,
        linewidth=0.45,
    )
    for tick in (0.52, 0.54, 0.56, 0.58, 0.60):
        x = mean_x(tick)
        ax.plot([x, x], [axis_y, axis_y - xpx(7)], color=ink, linewidth=0.45)
        ax.text(
            x,
            axis_y - xpx(19),
            f"{tick:.2f}",
            fontsize=label_pt,
            ha="center",
            va="top",
            color=ink,
        )
    ax.text(
        xpx((axis_left_px + axis_right_px) / 2),
        ypx(444),
        "Mean MCC (18 tasks)",
        fontsize=axis_title_pt,
        ha="center",
        va="top",
        color=ink,
    )


def draw_s41_matrix(
    ax: plt.Axes,
    frame: pd.DataFrame,
    show_legend: bool = True,
    show_header: bool = True,
    show_xlabel: bool = True,
) -> None:
    matrix = s41_matrix(frame)
    tasks = s41_task_order(frame)
    y_map = {
        condition: len(S41_CONDITION_ORDER) - 1 - i
        for i, condition in enumerate(S41_CONDITION_ORDER)
    }
    x_map = {task: i for i, task in enumerate(tasks)}
    ax.set_xlim(-0.2, len(tasks) + 2.8)
    ax.set_ylim(-1.25, len(S41_CONDITION_ORDER) + 1.0)
    ax.set_yticks(list(y_map.values()))
    labels = []
    meta = frame.drop_duplicates("condition_id").set_index("condition_id")
    for condition in S41_CONDITION_ORDER:
        labels.append(condition_label(meta.loc[condition]))
    ax.set_yticklabels(labels, fontsize=FACET_PT)
    ax.set_xticks(np.arange(len(tasks)) + 0.5)
    ax.set_xticklabels([""] * len(tasks))
    task_ranges = mcc_task_display_ranges(frame)
    display_label = "absolute MCC; each task has an independent display range; line = source 95% CI; marker = within-task rank"
    if show_header:
        draw_panel_heading(
            ax,
            "b",
            "S41 NT benchmark | absolute MCC + within-task rank",
            "absolute MCC; each task has an independent display range; highlighted row = PopReduce 60K · SW",
            y=1.16,
        )
    if show_xlabel:
        ax.set_xlabel(display_label, fontsize=AXIS_PT, labelpad=7)
    else:
        ax.set_xlabel("")
    _setup_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    highlight_condition = "A07"  # PopReduce 60K · SW in S41.
    highlight_y = y_map[highlight_condition]
    ax.axhspan(highlight_y - 0.38, highlight_y + 0.38, color="#FFF1DF", zorder=0)
    for tick in ax.get_yticklabels():
        if tick.get_text().startswith("PopReduce 60K") and "SW" in tick.get_text():
            tick.set_fontweight("bold")
            tick.set_color(COLORS["hEvo"])
    # Each task column has its own small axis.  The endpoint labels are the
    # actual MCC display limits for that task, not normalized 0/1 values.
    local_axis_y = -0.18
    for task in tasks:
        x0 = x_map[task]
        task_lo, task_hi = task_ranges[str(task)]
        ax.plot(
            [x0 + 0.08, x0 + 0.92],
            [local_axis_y, local_axis_y],
            color="#7E8994",
            linewidth=0.45,
            zorder=1,
        )
        ax.plot(
            [x0 + 0.08, x0 + 0.08],
            [local_axis_y - 0.06, local_axis_y + 0.06],
            color="#7E8994",
            linewidth=0.45,
            zorder=1,
        )
        ax.plot(
            [x0 + 0.92, x0 + 0.92],
            [local_axis_y - 0.06, local_axis_y + 0.06],
            color="#7E8994",
            linewidth=0.45,
            zorder=1,
        )
        ax.text(
            x0 + 0.50,
            -0.25,
            f"{task_lo:.2f}–{task_hi:.2f}",
            fontsize=3.8,
            ha="center",
            va="top",
            color="#58636E",
        )
        ax.text(
            x0 + 0.50,
            -0.52,
            _task_label(task),
            fontsize=LABEL_PT,
            rotation=42,
            ha="right",
            va="top",
            color=COLORS["text"],
        )
    for group in GROUP_ORDER:
        group_tasks = [
            t
            for t in tasks
            if frame.loc[frame["dataset"].eq(t), "task_group"].iloc[0] == group
        ]
        lo = min(x_map[t] for t in group_tasks)
        hi = max(x_map[t] for t in group_tasks) + 1
        ax.axvline(lo, color="#AEB6BF", linewidth=0.5)
        ax.text(
            (lo + hi) / 2,
            len(S41_CONDITION_ORDER) + 0.67,
            group,
            fontsize=FACET_PT,
            ha="center",
            fontweight="bold",
        )
    ax.axvline(len(tasks), color="#AEB6BF", linewidth=0.5)
    ax.text(
        len(tasks) + 1.0,
        len(S41_CONDITION_ORDER) + 0.67,
        "Mean",
        fontsize=FACET_PT,
        ha="center",
        fontweight="bold",
    )
    for _, row in matrix.iterrows():
        x0, y = x_map[row["dataset"]], y_map[row["condition_id"]]
        ax.plot(
            [x0 + 0.08, x0 + 0.92],
            [y, y],
            color=COLORS["track"],
            linewidth=0.55,
            zorder=1,
        )
        task_lo, task_hi = task_ranges[str(row["dataset"])]
        scale = task_hi - task_lo
        to_slot = lambda value: float(np.clip((float(value) - task_lo) / scale, 0, 1))
        lo = x0 + 0.08 + to_slot(row["ci_lower"]) * 0.84
        hi = x0 + 0.08 + to_slot(row["ci_upper"]) * 0.84
        mid = x0 + 0.08 + to_slot(row["metric_mean"]) * 0.84
        highlighted = row["condition_id"] == highlight_condition
        ax.plot(
            [lo, hi],
            [y, y],
            color=COLORS["hEvo"] if highlighted else "#C9A88D",
            linewidth=1.25 if highlighted else 0.65,
            zorder=2,
        )
        marker, color, _ = RANK_STYLES.get(
            int(row["rank"]), ("o", COLORS["NT"], "other")
        )
        if int(row["rank"]) > 3:
            ax.plot(
                mid,
                y,
                marker=marker,
                markersize=4.0 if highlighted else 3.1,
                markerfacecolor="white",
                markeredgecolor=COLORS["hEvo"] if highlighted else COLORS["NT"],
                markeredgewidth=0.8 if highlighted else 0.55,
                zorder=3,
            )
        else:
            ax.plot(
                mid,
                y,
                marker=marker,
                markersize=4.4 if highlighted else 3.6,
                markerfacecolor=color,
                markeredgecolor=COLORS["hEvo"] if highlighted else color,
                markeredgewidth=0.8 if highlighted else 0.45,
                zorder=3,
            )
    for i, condition in enumerate(S41_CONDITION_ORDER):
        y = len(S41_CONDITION_ORDER) - 1 - i
        mean = float(
            matrix.loc[matrix["condition_id"].eq(condition), "mean_mcc"].iloc[0]
        )
        ax.plot(
            [len(tasks) + 0.08, len(tasks) + 0.92],
            [y, y],
            color=COLORS["track"],
            linewidth=0.55,
        )
        # The mean column remains a summary, but uses its own local range so
        # it is not visually tied to any individual task scale.
        mean_lo = float(matrix["metric_mean"].min())
        mean_hi = float(matrix["metric_mean"].max())
        mean_span = max(mean_hi - mean_lo, 0.02)
        mean_pad = max(0.01, 0.12 * mean_span)
        mean_lo = max(0.0, float(np.floor((mean_lo - mean_pad) * 100.0) / 100.0))
        mean_hi = min(1.0, float(np.ceil((mean_hi + mean_pad) * 100.0) / 100.0))
        mean_slot = float(np.clip((mean - mean_lo) / (mean_hi - mean_lo), 0, 1))
        highlighted = condition == highlight_condition
        ax.plot(
            len(tasks) + 0.08 + mean_slot * 0.84,
            y,
            marker="o",
            markersize=5.0 if highlighted else 3.8,
            color=COLORS["hEvo"],
            markeredgecolor="white",
            markeredgewidth=0.45 if highlighted else 0.35,
        )
        ax.text(
            len(tasks) + 1.02,
            y - 0.12,
            f"{mean:.3f}",
            fontsize=LABEL_PT,
            va="center",
            fontweight="bold" if highlighted else "normal",
            color=COLORS["hEvo"] if highlighted else COLORS["text"],
        )
    legend = [
        Line2D(
            [0],
            [0],
            marker=RANK_STYLES[1][0],
            color="none",
            markerfacecolor=RANK_STYLES[1][1],
            markeredgecolor=RANK_STYLES[1][1],
            markersize=4,
            label="rank 1",
        ),
        Line2D(
            [0],
            [0],
            marker=RANK_STYLES[2][0],
            color="none",
            markerfacecolor=RANK_STYLES[2][1],
            markeredgecolor=RANK_STYLES[2][1],
            markersize=4,
            label="rank 2",
        ),
        Line2D(
            [0],
            [0],
            marker=RANK_STYLES[3][0],
            color="none",
            markerfacecolor=RANK_STYLES[3][1],
            markeredgecolor=RANK_STYLES[3][1],
            markersize=4,
            label="rank 3",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["NT"],
            markersize=4,
            label="other rank",
        ),
    ]
    if show_legend:
        ax.legend(
            handles=legend,
            loc="lower left",
            bbox_to_anchor=(0, 0.99),
            ncol=4,
            frameon=False,
            fontsize=LEGEND_PT,
            handletextpad=0.25,
            columnspacing=0.8,
        )


def draw_clinvar_all(
    ax: plt.Axes,
    frame: pd.DataFrame,
    show_header: bool = True,
    letter_pt: float | None = None,
) -> None:
    """Draw reference panel c: ClinVar ``all`` with AUPRC and AUROC columns."""

    data = frame[
        frame["dataset"].eq("all") & frame["evidence_status"].eq("PASS")
    ].copy()
    required = {"AUPRC", "AUROC"}
    if set(data["metric"].unique()) != required:
        raise ValueError(
        f"S42 dataset=all must contain AUPRC and AUROC, got {sorted(data['metric'].unique())}"
        )
    if data[["metric_mean", "ci_lower", "ci_upper"]].isna().any().any():
        raise ValueError("S42 dataset=all contains missing metric values")

    width_mm = 183.0
    height_mm = 77.5
    ref_px = 1275.0
    px_mm = width_mm / ref_px

    def xpx(value: float) -> float:
        return value * px_mm

    def ypx(value: float) -> float:
        return height_mm - value * px_mm

    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.axis("off")

    ink = COLORS["text"]
    orange = COLORS["hEvo"]
    gray = "#8E9CAF"
    label_pt = 6.2
    stage_pt = 6.4
    title_pt = 8.2
    panel_letter_pt = 16.5 if letter_pt is None else letter_pt

    if show_header:
        ax.text(
            xpx(45),
            ypx(30),
            "c",
            fontsize=panel_letter_pt,
            fontweight="bold",
            ha="left",
            va="top",
            color=ink,
            clip_on=False,
        )
    ax.text(
        xpx(700),
        ypx(29),
        "ClinVar all",
        fontsize=title_pt,
        fontweight="bold",
        ha="center",
        va="top",
        color=ink,
    )

    legend_y = ypx(119)
    ax.plot(
        xpx(557),
        legend_y,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor=orange,
        markeredgewidth=1.0,
        linestyle="none",
    )
    ax.text(
        xpx(583),
        legend_y,
        "No sliding",
        fontsize=label_pt,
        ha="left",
        va="center",
        color=ink,
    )
    ax.plot(
        xpx(1006),
        legend_y,
        marker="o",
        markersize=4.0,
        markerfacecolor=orange,
        markeredgecolor=orange,
        markeredgewidth=0.8,
        linestyle="none",
    )
    ax.text(
        xpx(1034),
        legend_y,
        "Sliding",
        fontsize=label_pt,
        ha="left",
        va="center",
        color=ink,
    )

    cohort_specs = [
        ("PopReduce_60K", "hEVO selected (60K)", 206),
        ("1000G_3K", "1000G (3K)", 274),
        ("UKB_500K", "UKB (500K)", 342),
    ]
    stage_offsets = {"Stage I": 33, "Stage II": 0}
    axis_specs = {
        "AUPRC": (461.0, 805.0, 0.30, 1.00),
        "AUROC": (924.0, 1265.0, 0.50, 1.00),
    }
    axis_y = 385.0

    def metric_x(metric: str, value: float) -> float:
        left, right, lo, hi = axis_specs[metric]
        return xpx(left + (float(value) - lo) / (hi - lo) * (right - left))

    for cohort, label, row_y in cohort_specs:
        ax.text(
            xpx(61),
            ypx(row_y - 16),
            label,
            fontsize=label_pt,
            fontweight="bold" if cohort == "PopReduce_60K" else "normal",
            ha="left",
            va="center",
            color=ink,
        )
        for stage, offset in stage_offsets.items():
            y = ypx(row_y - offset)
            ax.text(
                xpx(338), y, stage, fontsize=stage_pt, ha="left", va="center", color=ink
            )
            for metric in ("AUPRC", "AUROC"):
                subset = data[
                    (data["training_dataset"].eq(cohort))
                    & (data["stage"].eq(stage))
                    & (data["metric"].eq(metric))
                ]
                if len(subset) != 2 or set(subset["windowing"]) != set(WINDOW_ORDER):
                    raise ValueError(
                        f"S42 all incomplete for {cohort}/{stage}/{metric}"
                    )
                subset = subset.set_index("windowing")
                no_value = float(subset.loc["NoSW", "metric_mean"])
                sw_value = float(subset.loc["SW", "metric_mean"])
                color = orange if cohort == "PopReduce_60K" else gray
                x_no = metric_x(metric, no_value)
                x_sw = metric_x(metric, sw_value)
                ax.plot(
                    [x_no, x_sw],
                    [y, y],
                    color=color,
                    linewidth=0.85,
                    solid_capstyle="round",
                )
                ax.plot(
                    x_no,
                    y,
                    marker="o",
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    linestyle="none",
                )
                ax.plot(
                    x_sw,
                    y,
                    marker="o",
                    markersize=4.0,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.8,
                    linestyle="none",
                )

    for metric in ("AUPRC", "AUROC"):
        left, right, lo, hi = axis_specs[metric]
        y = ypx(axis_y)
        ax.plot([xpx(left), xpx(right)], [y, y], color=ink, linewidth=0.45)
        ticks = (0.30, 0.60, 1.00) if metric == "AUPRC" else (0.50, 0.75, 1.00)
        for tick in ticks:
            x = metric_x(metric, tick)
            ax.plot([x, x], [y, y - xpx(7)], color=ink, linewidth=0.45)
            ax.text(
                x,
                ypx(axis_y + 20),
                f"{tick:g}",
                fontsize=label_pt,
                ha="center",
                va="top",
                color=ink,
            )
        center = (left + right) / 2
        ax.text(
            xpx(center),
            ypx(453),
            metric,
            fontsize=title_pt - 1.0,
            ha="center",
            va="top",
            color=ink,
        )


def _d_model_key(value: object) -> str:
    """Stable model key used to align the heterogeneous source-table names."""

    text = "".join(ch for ch in str(value).lower() if ch.isalnum())
    aliases = {
        "hevobase": "hEvo-base",
        "hevolarge": "hEvo-large",
        "hevoopenbase": "hEvo-open-base",
        "hevoopenlarge": "hEvo-open-large",
    }
    return aliases.get(text, str(value))


def _d_series(
    frame: pd.DataFrame, value_column: str, *, model_column: str = "model"
) -> pd.Series:
    required = {model_column, value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Panel d source missing columns {missing}")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    out = pd.DataFrame(
        {"model": frame[model_column].map(_d_model_key), "value": values}
    )
    out = out.dropna(subset=["model", "value"])
    if out.empty:
        raise ValueError(f"Panel d source has no finite values in {value_column}")
    return out.groupby("model")["value"].mean()


def _d_rank_frame(scores: pd.Series, axis: str) -> pd.DataFrame:
    scores = pd.to_numeric(scores, errors="coerce").dropna()
    scores = scores.groupby(level=0).mean()
    if scores.empty:
        raise ValueError(f"Panel d: no finite scores for {axis}")
    ranks = scores.rank(method="min", ascending=False)
    n_models = len(scores)
    return pd.DataFrame(
        {
            "axis": axis,
            "model": scores.index.astype(str),
            "score": scores.to_numpy(float),
            "rank": ranks.to_numpy(float),
            "n_models": n_models,
            # Outer means a better rank.  This is a normalized rank, not a
            # reconstructed benchmark score.
            "radius": (n_models - ranks.to_numpy(float) + 1.0) / n_models,
        }
    )


def _d_rank_profiles(
    tables_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Build the three rank-profile groups used by panel d from live tables."""

    root = tables_root(tables_dir)

    def read(name: str) -> pd.DataFrame:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    genomic_source = read(
        "S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv"
    )
    if set(genomic_source["dataset"].astype(str)) != set(GENOMIC_BENCHMARK_TASKS):
        raise ValueError("Panel d: S18 task set does not match the seven genomic benchmark tasks")
    genomic_tasks_by_model = genomic_source.groupby("model")["dataset"].apply(
        lambda values: set(values.astype(str))
    )
    incomplete_genomic_models = {
        str(model): sorted(set(GENOMIC_BENCHMARK_TASKS) - tasks)
        for model, tasks in genomic_tasks_by_model.items()
        if tasks != set(GENOMIC_BENCHMARK_TASKS)
    }
    complete_genomic_models = sorted(
        model for model, tasks in genomic_tasks_by_model.items()
        if tasks == set(GENOMIC_BENCHMARK_TASKS)
    )
    genomic = _d_series(
        genomic_source.loc[genomic_source["model"].isin(complete_genomic_models)],
        "mcc_mean",
    )
    nt = _d_series(read("S17_frozen_representation_classification_across_18_nucleotide_transformer_tasks.csv"), "mcc_mean")
    ure_binary = _d_series(
        read("S19_binary_classification_across_five_ure_enhancer_control_sets.csv"), "mcc_mean"
    )
    ure_multi = _d_series(
        read("S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv"),
        "per_class_f1",
    )
    # The binary MCC and multiclass per-class F1 summaries are the two URE
    # benchmark components used by the current Fig.1 rank summary.
    ure = pd.concat([ure_binary, ure_multi], axis=1).mean(axis=1)
    deepsea = _d_series(active_deepsea_rows(read("S21_deepsea_multilabel_prediction_after_low_rank_adaptation.csv")), "auroc")

    clinvar = pd.concat(
        [
            _d_series(read("S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv"), "auprc_mean"),
            _d_series(read("S24_zero_shot_classification_of_coding_clinvar_variants.csv"), "auprc_mean"),
            _d_series(
                read("S25_zero_shot_classification_of_noncoding_clinvar_variants.csv"), "auprc_mean"
            ),
        ],
        axis=1,
    ).mean(axis=1)
    de_novo = pd.concat(
        [
            _d_series(
                read("S30_classification_of_ddd_de_novo_variants_against_benign_clinvar_controls.csv"), "auprc_mean"
            ),
            _d_series(
                read("S31_within_site_top_rank_discrimination_of_ddd_de_novo_alleles.csv"),
                "paired_top1_bootstrap_mean",
            ),
        ],
        axis=1,
    ).mean(axis=1, skipna=False)
    gwas = _d_series(read("S26_zero_shot_classification_of_candidate_causal_gwas_variants.csv"), "auprc_mean")
    rare = _d_series(read("S28_zero_shot_classification_of_uk_biobank_rare_variants.csv"), "auprc_mean")
    mpra = _d_series(read("S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv"), "auprc_mean")

    har = _d_series(read("S37_classification_of_human_accelerated_regions_across_negative_controls.csv"), "mean_mcc_for_sort")
    haqer = _d_series(read("S36_classification_of_human_ancestor_quickly_evolved_regions_across_negative_controls.csv"), "mean_mcc_for_sort")
    ancestry = pd.concat(
        [
            _d_series(
                read("S33_unsupervised_separation_of_1000_genomes_ancestry_groups.csv"), "rank_score"
            ),
            _d_series(
                read("S34_supervised_classification_of_1000_genomes_ancestry_groups.csv"),
                "classification_mcc",
            ),
            _d_series(
                read("S35_few_shot_regression_of_ancestry_proportions.csv"),
                "pearson_r_mean",
            ),
        ],
        axis=1,
    ).mean(axis=1)

    adaptation_table = read(
        "S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv"
    )
    adaptation_table = adaptation_table[
        adaptation_table["status"].eq("PASS")
        & pd.to_numeric(adaptation_table["accuracy"], errors="coerce").notna()
    ].copy()
    if adaptation_table.empty:
        raise ValueError("Panel d: S42 has no source-backed PASS accuracy rows")
    adaptation = _d_series(adaptation_table, "accuracy", model_column="display_model")
    loci = int(
        pd.to_numeric(adaptation_table["n_loci_expected"], errors="coerce")
        .dropna()
        .mode()
        .iloc[0]
    )

    raw = {
        "Genomic": genomic,
        "NT": nt,
        "URE": ure,
        "DeepSEA": deepsea,
        "ClinVar": clinvar,
        "De novo": de_novo,
        "GWAS causal": gwas,
        "Rare variants": rare,
        "MPRA": mpra,
        "HAR": har,
        "HAQER": haqer,
        "Ancestry": ancestry,
        "Adaptive probe": adaptation,
    }
    profiles = {axis: _d_rank_frame(scores, axis) for axis, scores in raw.items()}
    for axis, profile in profiles.items():
        if not {"hEvo-base", "hEvo-large"}.issubset(set(profile["model"])):
            raise ValueError(
                f"Panel d: {axis} does not contain both hEVO reported models"
            )
    details = {
        "source_model_counts": {
            axis: len(profile) for axis, profile in profiles.items()
        },
        "adaptation_loci_expected": loci,
        "deepsea_model_count": len(profiles["DeepSEA"]),
        "source_backed_axes": {
            model: sorted(
                axis for axis, profile in profiles.items()
                if model in set(profile["model"])
            )
            for model in ("hEvo-open-base", "hEvo-open-large", "hEvo-large")
        },
        "genomic_benchmark_coverage": {
            "source_models": int(genomic_source["model"].nunique()),
            "complete_models_used_for_ranking": complete_genomic_models,
            "incomplete_models_excluded_from_ranking": incomplete_genomic_models,
            "expected_tasks_per_model": len(GENOMIC_BENCHMARK_TASKS),
            "ranking_rule": "unweighted mean MCC across all seven tasks; models missing any task are excluded from this axis",
        },
        "open_models": "measured axes shown; unmeasured axes left disconnected",
    }
    return profiles, details


def draw_d_rank_profiles(
    ax: plt.Axes,
    tables_dir: Path | None = None,
    show_header: bool = True,
    letter_pt: float | None = None,
    radar_radius: float = 114.0,
) -> dict[str, object]:
    """Draw reference panel d: individual-model rank profiles for hEVO."""

    profiles, details = _d_rank_profiles(tables_dir)
    width_px, height_px = 2320.0, 630.0
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, height_px)
    ax.axis("off")

    ink = COLORS["text"]
    base_color = "#B35C1E"
    large_color = "#7E431F"
    grid = "#C9D0D7"
    outer = "#929DA8"

    def y(top_px: float) -> float:
        return height_px - top_px

    if show_header:
        panel_letter_pt = 20.0 if letter_pt is None else letter_pt
        ax.text(
            16,
            y(56),
            "d",
            fontsize=panel_letter_pt,
            fontweight="bold",
            ha="left",
            va="top",
            color=ink,
        )

    # The main figure reports the two canonical hEVO models. Open-series rows
    # remain available in the source profiles and details, but are not drawn in
    # the main-figure display layer.
    legend_items = [
        ("hEvo-base", 300, "o", base_color, "-", True),
        ("hEvo-large", 790, "D", large_color, "-", True),
    ]
    for label, x0, marker, color, linestyle, filled in legend_items:
        ax.plot(
            [x0 - 18, x0 + 42],
            [y(86), y(86)],
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
            solid_capstyle="butt",
        )
        ax.plot(
            x0 + 12,
            y(86),
            marker=marker,
            markersize=5.0,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=1.25,
            linestyle="none",
        )
        ax.text(x0 + 63, y(86), label, fontsize=7.0, ha="left", va="center", color=ink)
    ax.text(
        720,
        y(145),
        "Reported models",
        fontsize=7.0,
        ha="center",
        va="center",
        color=ink,
    )

    groups = [
        (
            "Genomic elements [0.25-1]",
            430.0,
            382.0,
            radar_radius,
            0.25,
            [
                ("Genomic", "Genomic (7)"),
                ("NT", "NT (18)"),
                ("URE", "URE (10)"),
                ("DeepSEA", "DeepSEA (3)"),
            ],
        ),
        (
            "Variant scoring [0.35-1]",
            1210.0,
            382.0,
            radar_radius,
            0.35,
            [
                ("ClinVar", "ClinVar (3)"),
                ("De novo", "De novo (2)"),
                ("GWAS causal", "GWAS causal (1)"),
                ("Rare variants", "Rare variants (1)"),
                ("MPRA", "MPRA (1)"),
            ],
        ),
        (
            "Evolution & ancestry [0.15-1]",
            1990.0,
            382.0,
            radar_radius,
            0.15,
            [
                ("HAR", "HAR (5)"),
                ("HAQER", "HAQER (5)"),
                ("Ancestry", "Ancestry (3)"),
                (
                    "Adaptive probe",
                    f"Adaptive probe ({details['adaptation_loci_expected']})",
                ),
            ],
        ),
    ]
    for title, cx, cy, radius, _rmin, axes in groups:
        ax.text(
            cx,
            y(192),
            title,
            fontsize=7.4,
            fontweight="bold",
            ha="center",
            va="center",
            color=ink,
        )
        n = len(axes)
        angles = np.pi / 2.0 - np.arange(n) * (2.0 * np.pi / n)

        def point(rad: float, angle: float) -> tuple[float, float]:
            return cx + radius * rad * np.cos(angle), y(
                cy - radius * rad * np.sin(angle)
            )

        grid_points = [point(1.0, angle) for angle in angles]
        closed_grid = grid_points + [grid_points[0]]
        ax.plot(
            [p[0] for p in closed_grid],
            [p[1] for p in closed_grid],
            color=outer,
            linewidth=0.45,
            zorder=1,
        )
        for angle in angles:
            x1, y1 = point(0.0, angle)
            x2, y2 = point(1.0, angle)
            ax.plot([x1, x2], [y1, y2], color=grid, linewidth=0.35, zorder=1)

        for model, color, marker in (
            ("hEvo-base", base_color, "o"),
            ("hEvo-large", large_color, "D"),
        ):
            indexed = {axis: profiles[axis].set_index("model") for axis, _ in axes}
            if any(model not in frame.index for frame in indexed.values()):
                continue
            coords = [
                point(float(indexed[axis].loc[model, "radius"]), angles[i])
                for i, (axis, _) in enumerate(axes)
            ]
            closed = coords + [coords[0]]
            ax.plot(
                [p[0] for p in closed],
                [p[1] for p in closed],
                color=color,
                linewidth=1.05,
                linestyle="-",
                zorder=3,
            )
            ax.plot(
                [p[0] for p in coords],
                [p[1] for p in coords],
                marker=marker,
                linestyle="none",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=4.8,
                zorder=4,
            )

        for i, (_, label) in enumerate(axes):
            lx, ly = point(1.0, angles[i])
            dx, dy = lx - cx, ly - y(cy)
            if abs(dx) < 2:
                ha, va, ox, oy = (
                    "center",
                    ("bottom" if dy < 0 else "top"),
                    0,
                    (-35 if dy < 0 else 35),
                )
            elif dx > 0:
                ha, va, ox, oy = "left", "center", 12, 0
            else:
                ha, va, ox, oy = "right", "center", -12, 0
            ax.text(lx + ox, ly + oy, label, fontsize=6.2, ha=ha, va=va, color=ink)

    ax.text(
        2298,
        y(610),
        "Outer = better individual-model rank.",
        fontsize=6.3,
        ha="right",
        va="center",
        color=ink,
    )
    return details


def draw_factor_delta(
    axs: list[plt.Axes], frame: pd.DataFrame, show_header: bool = True
) -> None:
    delta = factor_deltas(frame)
    comparisons = list(
        delta.sort_values("comparison_order")["comparison"].drop_duplicates()
    )
    y_map = {x: len(comparisons) - 1 - i for i, x in enumerate(comparisons)}
    max_abs = max(0.01, float(np.nanmax(np.abs(delta["delta"]))))
    lim = max_abs * 1.25
    highlight_comparisons = {
        "SW − NoSW | PopReduce 60K",
        "PopReduce 60K − UKB 500K | SW",
    }
    for ax, group in zip(axs, GROUP_ORDER):
        subset = delta[delta["task_group"].eq(group)]
        ax.axvline(0, color="#333333", linewidth=0.5)
        for comparison in highlight_comparisons:
            if comparison in y_map:
                ax.axhspan(
                    y_map[comparison] - 0.38,
                    y_map[comparison] + 0.38,
                    color="#FFF1DF",
                    zorder=0,
                )
        for _, row in subset.iterrows():
            y = y_map[row["comparison"]]
            val = float(row["delta"])
            highlighted = row["comparison"] in highlight_comparisons
            ax.plot(
                val,
                y,
                marker="o",
                markersize=3.1 if highlighted else 2.5,
                color=COLORS["hEvo"] if highlighted else "#B9C0C8",
                markeredgecolor="white",
                markeredgewidth=0.35 if highlighted else 0.25,
                zorder=2,
            )
        means = subset.groupby("comparison")["delta"].mean()
        for comparison, value in means.items():
            y = y_map[comparison]
            color = COLORS["positive"] if value >= 0 else COLORS["negative"]
            highlighted = comparison in highlight_comparisons
            ax.plot(
                value,
                y,
                marker="D",
                markersize=5.0 if highlighted else 4.0,
                color=COLORS["hEvo"] if highlighted else color,
                markeredgecolor="white",
                markeredgewidth=0.45 if highlighted else 0.35,
                zorder=3,
            )
            ax.text(
                value + (0.004 if value >= 0 else -0.004),
                y,
                f"{value:+.3f}",
                fontsize=4.6,
                va="center",
                ha="left" if value >= 0 else "right",
                fontweight="bold" if highlighted else "normal",
                color=COLORS["hEvo"] if highlighted else COLORS["text"],
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.8, len(comparisons) - 0.2)
        ax.set_yticks(range(len(comparisons)))
        # y=0 is the bottom row, while y_map assigns the first comparison to
        # the top row; reverse labels so the text follows the plotted points.
        ax.set_yticklabels(
            [x.replace(" | ", "\n") for x in reversed(comparisons)], fontsize=LABEL_PT
        )
        ax.set_title(
            group.replace(" Annotation", ""),
            fontsize=FACET_PT,
            fontweight="bold",
            pad=2,
        )
        ax.set_xlabel("ΔMCC", fontsize=AXIS_PT, labelpad=3)
        _setup_ax(ax)
        if ax is not axs[0]:
            ax.tick_params(left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
    if show_header:
        draw_panel_heading(
            axs[0],
            "c",
            "S41 factor effects",
            "paired mean-MCC difference across matched NT tasks; PopReduce 60K · SW comparisons highlighted",
            y=1.16,
        )


def draw_clinvar(
    axs: list[plt.Axes],
    frame: pd.DataFrame,
    show_header: bool = True,
    show_legend: bool = True,
) -> None:
    data = s42_auprc(frame)
    if set(data["dataset"]) != set(S42_STRATA):
        unavailable = sorted(set(S42_STRATA) - set(data["dataset"]))
        raise ValueError(
            "S42 stratified ClinVar rendering is unavailable; "
            f"no PASS evidence for {unavailable}"
        )
    cohort_colors = {
        "UKB_500K": "#B35C1E",
        "PopReduce_60K": "#174FA6",
        "1000G_3K": "#0E7C86",
    }
    combos = [(cohort, window) for cohort in COHORT_ORDER for window in WINDOW_ORDER]
    y_map = {
        f"{cohort}|{window}": len(combos) - 1 - i
        for i, (cohort, window) in enumerate(combos)
    }
    for ax, stratum in zip(axs, S42_STRATA):
        subset = data[data["dataset"].eq(stratum)]
        for cohort, window in combos:
            pair = subset[
                (subset["training_dataset"].eq(cohort))
                & (subset["windowing"].eq(window))
            ].copy()
            pair = pair.sort_values("stage")
            y = y_map[f"{cohort}|{window}"]
            if len(pair) != 2:
                raise ValueError(
                    f"S42 AUPRC pair incomplete for {cohort}/{window}/{stratum}"
                )
            xvals = [0, 1]
            vals = pair["metric_mean"].to_numpy(float)
            highlighted = cohort == "PopReduce_60K" and window == "SW"
            line_color = COLORS["hEvo"] if highlighted else "#AEB7C0"
            ax.plot(
                xvals,
                vals,
                color=line_color,
                linewidth=1.35 if highlighted else 0.65,
                alpha=0.95 if highlighted else 0.7,
                zorder=3 if highlighted else 1,
            )
            for x, (_, row) in zip(xvals, pair.iterrows()):
                point_color = line_color
                ax.errorbar(
                    x,
                    row["metric_mean"],
                    yerr=[
                        [row["metric_mean"] - row["ci_lower"]],
                        [row["ci_upper"] - row["metric_mean"]],
                    ],
                    fmt="o",
                    color=point_color,
                    markerfacecolor="white"
                    if row["stage"] == "Stage I"
                    else point_color,
                    markeredgecolor=point_color,
                    markersize=4.6 if highlighted else 3.0,
                    capsize=2.0 if highlighted else 1.3,
                    linewidth=0.75 if highlighted else 0.45,
                    zorder=4 if highlighted else 2,
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stage I", "Stage II"], fontsize=LABEL_PT)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(stratum, fontsize=FACET_PT, fontweight="bold", pad=2)
        _setup_ax(ax)
        if ax is axs[0]:
            ax.set_ylabel("AUPRC", fontsize=AXIS_PT)
        else:
            ax.tick_params(left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
    if show_header:
        draw_panel_heading(
            axs[0],
            "d",
            "ClinVar variant detection | AUPRC",
            "Stage I → Stage II; PopReduce 60K · SW highlighted; points = source mean, bars = source 95% CI",
            y=1.16,
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=(COLORS["hEvo"] if c == "PopReduce_60K" else "#AEB7C0"),
            marker="o",
            markerfacecolor="white",
            markersize=4.0 if c == "PopReduce_60K" else 3.0,
            linewidth=1.2 if c == "PopReduce_60K" else 0.65,
            label=(
                "PopReduce 60K · SW (highlight)"
                if c == "PopReduce_60K"
                else c.replace("_", " ")
            ),
        )
        for c in COHORT_ORDER
    ]
    if show_legend:
        axs[-1].legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.03),
            frameon=False,
            fontsize=LEGEND_PT,
            ncol=3,
            handletextpad=0.3,
            columnspacing=0.8,
        )


def _pareto_family_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=family_color(f),
            markeredgecolor="white",
            markersize=4,
            label="hEVO" if f == "hEvo" else f,
        )
        for f in ("hEvo", "NT", "Evo 2", "GPN", "Other")
    ]


def _pareto_family_group(family: str) -> str:
    if family == "hEvo":
        return "hEvo"
    if family in {"NT v2", "NT v3"}:
        return "NT"
    if family == "Evo 2":
        return "Evo 2"
    if family == "GPN":
        return "GPN"
    return "Other"


def _pareto_group_style(group: str) -> tuple[str, str]:
    return {
        "hEvo": ("o", COLORS["hEvo"]),
        "NT": ("s", COLORS["NT"]),
        "Evo 2": ("^", COLORS["Evo 2"]),
        "GPN": ("D", COLORS["GPN"]),
        "Other": ("o", "#B8C2CE"),
    }[group]


def _pareto_family_handles_reference() -> list[Line2D]:
    handles = []
    for group in ("hEvo", "NT", "Evo 2", "GPN", "Other"):
        marker, color = _pareto_group_style(group)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=4.4,
                label="hEVO" if group == "hEvo" else group,
            )
        )
    return handles


def _pareto_supplementary_ids(
    summary: pd.DataFrame, tables_dir: Path | None
) -> tuple[dict[str, int], Path]:
    """Map display models to the 1-based S47 supplementary-table roster."""

    path = tables_root(tables_dir) / S47_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Pareto model-ID source table not found: {path}")
    roster = pd.read_csv(path)
    if "model" not in roster.columns:
        raise ValueError(f"{path.name}: missing model column")
    canonical = [str(model) for model in roster["model"]]
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{path.name}: duplicate canonical model IDs")
    model_set = set(summary["model"].astype(str))
    canonical_set = set(canonical)
    if summary["model"].duplicated().any() or not model_set.issubset(canonical_set):
        extra = sorted(model_set - canonical_set)
        raise ValueError(
            f"{path.name}: unregistered or duplicate display models; extra={extra}"
        )
    return {model: index for index, model in enumerate(canonical, start=1)}, path


def draw_pareto_single(
    ax: plt.Axes,
    *,
    tables_dir: Path | None = None,
    metric: str = "throughput_bps",
    xlabel: str = "Backbone throughput (bp/s)",
    xscale: str = "linear",
    higher_is_better: bool = True,
    letter: str = "e",
    show_legend: bool = True,
    letter_pt: float | None = None,
) -> dict[str, object]:
    """Draw one reference-style Pareto panel with source-backed model points."""

    summary = pd.read_csv(
        tables_root(tables_dir) / S43_NAME,
        na_values=["NA", ""],
        float_precision="round_trip",
    )
    _pareto_supplementary_ids(summary, tables_dir)
    if set(summary["table_id"].astype(str)) != {"S43"}:
        raise ValueError(f"{S43_NAME}: expected table_id=S43")
    details = {"source_table": S43_NAME, "display_model_count": len(summary)}
    plot = summary[
        np.isfinite(summary[metric]) & np.isfinite(summary["mean_mcc"])
    ].copy()
    if plot.empty:
        raise ValueError(f"No valid Pareto rows for {metric}")
    plot["family_group"] = plot["family"].map(_pareto_family_group)
    model_ids, model_id_source = _pareto_supplementary_ids(summary, tables_dir)
    plot["supplementary_id"] = plot["model"].map(model_ids)

    px, py = pareto_frontier(
        plot[metric].to_numpy(float), plot["mean_mcc"].to_numpy(float), higher_is_better
    )
    ax.plot(
        px,
        py,
        color="#8B98A6",
        linewidth=0.7,
        linestyle="--",
        alpha=0.8,
        drawstyle="steps-post" if higher_is_better else "steps-pre",
        zorder=1,
    )

    for group in ("Other", "NT", "Evo 2", "GPN", "hEvo"):
        subset = plot[plot["family_group"].eq(group)]
        if subset.empty:
            continue
        marker, color = _pareto_group_style(group)
        incomplete = subset["n_" + metric].to_numpy(int) < 18
        for row, is_incomplete in zip(subset.itertuples(index=False), incomplete):
            ax.scatter(
                [getattr(row, metric)],
                [row.mean_mcc],
                s=20 if group != "hEvo" else 30,
                marker=marker,
                facecolors="none" if is_incomplete else color,
                edgecolors=color,
                linewidths=0.8 if is_incomplete else 0.35,
                zorder=4 if group == "hEvo" else 3,
            )
            if group != "hEvo":
                ax.annotate(
                    str(row.supplementary_id),
                    (getattr(row, metric), row.mean_mcc),
                    xytext=(2.0, 2.0),
                    textcoords="offset points",
                    fontsize=4.4,
                    color=COLORS["text"],
                    ha="left",
                    va="bottom",
                )

    for model, yoff in (
        ("hEvo-base", 8),
        ("hEvo-large", -10),
    ):
        row = plot[plot["model"].eq(model)]
        if row.empty:
            continue
        row = row.iloc[0]
        marker = "o" if model == "hEvo-base" else "D"
        label = model
        incomplete = int(row["n_" + metric]) < 18
        ax.annotate(
            label,
            (row[metric], row["mean_mcc"]),
            xytext=(4, yoff),
            textcoords="offset points",
            fontsize=6.1,
            color=COLORS["hEvo"],
            fontweight="bold",
            ha="left",
            va="center",
        )
        # Redraw hEVO on top with its final marker geometry and coverage state.
        ax.scatter(
            [row[metric]],
            [row["mean_mcc"]],
            s=30,
            marker=marker,
            facecolors="none" if incomplete else COLORS["hEvo"],
            edgecolors=COLORS["hEvo"],
            linewidths=1.0,
            zorder=6,
        )

    ax.set_xscale(xscale)
    ax.set_ylim(0.37, 0.62)
    ax.set_yticks([0.40, 0.45, 0.50, 0.55, 0.60])
    if xscale == "linear":
        x_max = float(plot[metric].max())
        ax.set_xlim(0.0, x_max * 1.08)
        ticks = np.linspace(0.0, x_max, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0" if t == 0 else f"{t / 1e6:.2f}M" for t in ticks])
    else:
        positive = plot[metric][plot[metric] > 0]
        ax.set_xlim(float(positive.min()) * 0.72, float(positive.max()) * 1.25)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel, fontsize=7.0, labelpad=5)
    ax.set_ylabel("Mean MCC (18 tasks)", fontsize=7.0, labelpad=5)
    _setup_ax(ax)
    ax.tick_params(labelsize=5.8, length=3.0, width=0.65)
    panel_letter_pt = PANEL_LETTER_PT if letter_pt is None else letter_pt
    ax.text(
        -0.175,
        1.16,
        letter,
        transform=ax.transAxes,
        fontsize=panel_letter_pt,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )
    if xscale == "log":
        ax.text(
            1.0,
            1.16,
            "log scale",
            transform=ax.transAxes,
            fontsize=6.3,
            ha="right",
            va="top",
            color=COLORS["text"],
            clip_on=False,
        )
    if show_legend:
        ax.figure.legend(
            handles=_pareto_family_handles_reference(),
            loc="upper center",
            bbox_to_anchor=(0.58, 0.955),
            ncol=5,
            frameon=False,
            fontsize=6.5,
            handletextpad=0.28,
            columnspacing=1.15,
        )
    details.update(
        {
            "metric": metric,
            "xscale": xscale,
            "higher_is_better": higher_is_better,
            "frontier_points": len(px),
            "frontier_x_monotonic": bool(np.all(np.diff(px) >= 0))
            if len(px) > 1
            else True,
            "frontier_y_monotonic": bool(np.all(np.diff(py) >= -1e-12))
            if higher_is_better and len(py) > 1
            else (bool(np.all(np.diff(py) <= 1e-12)) if len(py) > 1 else True),
            "display_model_count": len(summary),
            "valid_metric_model_count": len(plot),
            "supplementary_model_ids": model_ids,
            "supplementary_model_id_source": _public_path(model_id_source),
            "incomplete_metric_models": plot.loc[
                plot["n_" + metric] < 18, ["model", "n_" + metric]
            ].to_dict("records"),
        }
    )
    return details


def draw_pareto(
    axs: np.ndarray,
    show_header: bool = True,
    show_legend: bool = True,
    legend_anchor: float = 0.99,
) -> dict[str, object]:
    summary = pd.read_csv(
        tables_root() / S43_NAME, na_values=["NA", ""], float_precision="round_trip"
    )
    details = {"source_table": S43_NAME, "display_model_count": len(summary)}
    metrics = [
        ("throughput_bps", "Backbone throughput (bp/s)", True),
        ("params_m", "Total predictor parameters (M)", False),
        ("latency_ms", "Backbone latency (ms)", False),
        ("vram_mb", "Peak VRAM (MB)", False),
    ]
    for ax, (column, xlabel, higher) in zip(axs.flat, metrics):
        plot = summary[
            np.isfinite(summary[column]) & np.isfinite(summary["mean_mcc"])
        ].copy()
        if plot.empty:
            raise ValueError(f"No valid Pareto rows for {column}")
        px, py = pareto_frontier(
            plot[column].to_numpy(float), plot["mean_mcc"].to_numpy(float), higher
        )
        ax.step(
            px,
            py,
            where="post" if higher else "pre",
            color="#174A7E",
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            plot[column],
            plot["mean_mcc"],
            s=12,
            c=[family_color(x) for x in plot["family"]],
            edgecolors="white",
            linewidths=0.25,
            alpha=0.75,
            zorder=2,
        )
        for model in ("hEvo-base", "hEvo-large"):
            row = plot[plot["model"].eq(model)]
            if row.empty:
                continue
            row = row.iloc[0]
            incomplete = int(row["n_perf"]) < 18 or int(row["n_eff"]) < 18
            marker = "o" if model == "hEvo-base" else "s"
            ax.scatter(
                [row[column]],
                [row["mean_mcc"]],
                s=34,
                marker=marker,
                facecolors="none" if incomplete else COLORS["hEvo"],
                edgecolors=COLORS["hEvo"],
                linewidths=0.9,
                zorder=4,
            )
            ax.annotate(
                display_name(model),
                (row[column], row["mean_mcc"]),
                xytext=(4, 4 if model == "hEvo-base" else -10),
                textcoords="offset points",
                fontsize=5.4,
                color=COLORS["hEvo"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=COLORS["hEvo"], linewidth=0.35),
            )
        ax.set_xscale("log")
        y_min, y_max = float(plot["mean_mcc"].min()), float(plot["mean_mcc"].max())
        pad = max(0.006, (y_max - y_min) * 0.12)
        ax.set_ylim(max(0, y_min - pad), min(1, y_max + pad))
        ax.set_xlabel(xlabel, fontsize=AXIS_PT, labelpad=2)
        ax.set_ylabel(
            "Mean MCC" if ax is axs.flat[0] or ax is axs.flat[2] else "",
            fontsize=AXIS_PT,
            labelpad=2,
        )
        _setup_ax(ax)
        ax.tick_params(labelsize=TICK_PT)
    if show_legend:
        axs.flat[0].figure.legend(
            handles=_pareto_family_handles(),
            loc="center",
            bbox_to_anchor=(0.50, 0.49),
            frameon=False,
            ncol=5,
            fontsize=LEGEND_PT,
            handletextpad=0.25,
            columnspacing=0.8,
        )
    if show_header:
        draw_panel_heading(
            axs.flat[0],
            "e",
            "NT performance–efficiency Pareto views",
            "PopReduce 60K · SW = hEVO-base highlighted; hEVO-large shown for scale; open marker denotes incomplete coverage",
            y=1.05,
        )
    return details


def make_figure(kind: str, tables_dir: Path | None = None):
    configure_matplotlib()
    s41 = read_s41(tables_dir)
    s42 = read_s42(tables_dir)
    if kind == "b":
        fig, ax = plt.subplots(figsize=(87 / 25.4, 44.4 / 25.4))
        draw_nt_benchmark(ax, s41)
    elif kind == "c":
        fig, ax = plt.subplots(figsize=(108 / 25.4, 45.7 / 25.4))
        draw_clinvar_all(ax, s42)
    elif kind == "d":
        fig, ax = plt.subplots(figsize=(196.2 / 25.4, 53.4 / 25.4))
        d_details = draw_d_rank_profiles(ax, tables_dir=tables_dir)
        fig._d_details = d_details
    elif kind == "e":
        fig, ax = plt.subplots(figsize=(98.3 / 25.4, 69.5 / 25.4))
        pareto_details = draw_pareto_single(
            ax,
            tables_dir=tables_dir,
            metric="throughput_bps",
            xlabel="Backbone throughput (bp/s)",
            xscale="log",
            higher_is_better=True,
            letter="e",
            show_legend=True,
        )
        fig._pareto_details = pareto_details
    elif kind == "f":
        fig, ax = plt.subplots(figsize=(98.3 / 25.4, 69.5 / 25.4))
        pareto_details = draw_pareto_single(
            ax,
            tables_dir=tables_dir,
            metric="params_m",
            xlabel="Nominal total parameters (M)",
            xscale="log",
            higher_is_better=True,
            letter="f",
            show_legend=True,
        )
        fig._pareto_details = pareto_details
    elif kind == "g":
        fig, ax = plt.subplots(figsize=(98.3 / 25.4, 69.5 / 25.4))
        pareto_details = draw_pareto_single(
            ax,
            tables_dir=tables_dir,
            metric="latency_ms",
            xlabel="Backbone latency (ms)",
            xscale="log",
            higher_is_better=False,
            letter="g",
            show_legend=True,
        )
        fig._pareto_details = pareto_details
    elif kind == "h":
        fig, ax = plt.subplots(figsize=(98.3 / 25.4, 69.5 / 25.4))
        pareto_details = draw_pareto_single(
            ax,
            tables_dir=tables_dir,
            metric="vram_mb",
            xlabel="Peak VRAM (MB)",
            xscale="log",
            higher_is_better=False,
            letter="h",
            show_legend=True,
        )
        fig._pareto_details = pareto_details
    else:
        raise ValueError(kind)
    if kind == "c" or kind == "b" or kind == "d":
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    elif kind in {"e", "f", "g", "h"}:
        fig.subplots_adjust(left=0.155, right=0.959, bottom=0.137, top=0.785)
    else:
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.82)
    return fig


def make_a4(tables_dir: Path | None = None):
    """Use the native-vector reference layout for all integrated callers."""
    from _reference import make_a4 as render_reference
    return render_reference(tables_dir)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    width_mm: float,
    height_mm: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf, svg, png = (output_dir / f"{stem}.{ext}" for ext in ("pdf", "svg", "png"))
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    with Image.open(png) as image:
        image.verify()
    reader = PdfReader(str(pdf))
    page_count = len(reader.pages)
    svg_editable_text = "<text" in svg.read_text(encoding="utf-8")
    if page_count < 1:
        raise ValueError(f"Rendered PDF has no pages: {pdf}")
    page = reader.pages[0]
    actual_size_mm = (
        float(page.mediabox.width) * 25.4 / 72,
        float(page.mediabox.height) * 25.4 / 72,
    )
    if any(
        abs(actual - expected) > 0.05
        for actual, expected in zip(actual_size_mm, (width_mm, height_mm))
    ):
        raise ValueError(
            f"Rendered PDF size {actual_size_mm} mm does not match "
            f"{(width_mm, height_mm)} mm"
        )
    if not svg_editable_text:
        raise ValueError(f"Rendered SVG has no editable text: {svg}")
    plt.close(fig)
    return {
        "pdf": str(pdf),
        "svg": str(svg),
        "png": str(png),
    }
