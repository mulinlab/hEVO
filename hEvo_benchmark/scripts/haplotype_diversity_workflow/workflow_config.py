"""Shared path defaults for the haplotype diversity workflow.

All defaults target the established HPC layout. Every path can be overridden by
an environment variable so the workflow can be used from another checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("HAPDIV_PROJECT_ROOT", "<source-project-root>")
)
DEFAULT_HEVO_DATA_ROOT = Path(os.environ.get("HEVO_DATA_ROOT", "<hEvo-data-root>"))
DEFAULT_RESULTS_ROOT = DEFAULT_PROJECT_ROOT / "results"
DEFAULT_LOG_ROOT = DEFAULT_PROJECT_ROOT / "logs"
DEFAULT_S3_SOURCE = (
    DEFAULT_RESULTS_ROOT
    / "ukb_downsampling_vcf_empirical_fullUKBB_ref50k"
    / "ukb_downsampling_empirical.curve.tsv"
)
DEFAULT_S4_SOURCE = (
    DEFAULT_RESULTS_ROOT
    / "haplotype_diversity_teacher_order"
    / "fig1_haplotype_diversity_plot_source.tsv"
)


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def project_root() -> Path:
    return _path_env("HAPDIV_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)


def hevo_data_root() -> Path:
    return _path_env("HEVO_DATA_ROOT", DEFAULT_HEVO_DATA_ROOT)


def results_root() -> Path:
    return _path_env("HAPDIV_RESULTS_ROOT", project_root() / "results")


def log_root() -> Path:
    return _path_env("HAPDIV_LOG_ROOT", project_root() / "logs")


def actual_s3_source() -> Path:
    return _path_env(
        "HAPDIV_S3_SOURCE",
        results_root()
        / "ukb_downsampling_vcf_empirical_fullUKBB_ref50k"
        / "ukb_downsampling_empirical.curve.tsv",
    )


def actual_s4_source() -> Path:
    return _path_env(
        "HAPDIV_S4_SOURCE",
        results_root()
        / "haplotype_diversity_teacher_order"
        / "fig1_haplotype_diversity_plot_source.tsv",
    )


def actual_s3_result_prefix() -> Path:
    source = actual_s3_source()
    suffix = ".curve.tsv"
    return (
        Path(str(source)[: -len(suffix)])
        if str(source).endswith(suffix)
        else source.with_suffix("")
    )


def plot_output_dir() -> Path:
    return _path_env("HAPDIV_PLOT_OUTPUT", results_root() / "haplotype_diversity_plots")
