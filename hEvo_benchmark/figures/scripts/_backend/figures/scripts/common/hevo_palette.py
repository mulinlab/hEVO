"""Canonical color semantics shared by hEVO manuscript figures.

Ancestry colors are locked to the formal HGDP + 1000 Genomes ancestry-anchor
panel.  New figures must import this mapping instead of defining local colors.
"""

from __future__ import annotations

# [Shared styles] Centralize color aliases for model families, ancestry or population groups, and directional results.
# [Usage] Plotting scripts import this module instead of redefining colors, keeping color meanings consistent across figures.
# [Note] This file provides style constants only; it does not read tables or create figures.

from typing import Final


ANCESTRY_ORDER: Final[tuple[str, ...]] = ("EUR", "AFR", "EAS", "SAS", "Others")

ANCESTRY_COLORS: Final[dict[str, str]] = {
    "EUR": "#0B2E83",     # deep hEVO blue
    "AFR": "#D97706",     # orange
    "EAS": "#0E7C86",     # teal
    "SAS": "#7655A6",     # purple
    "Others": "#8995A5",  # neutral gray
}

# Display aliases use the identical category color.
ANCESTRY_COLOR_ALIASES: Final[dict[str, str]] = {
    **ANCESTRY_COLORS,
    "Other/admixed": ANCESTRY_COLORS["Others"],
    # S5 uses the 1000 Genomes super-population label AMR while its legacy
    # indicator column is named ``Others``.  Keep the fifth-category hue
    # stable, but never treat the labels as biologically interchangeable in
    # captions or source-data contracts.
    "AMR": ANCESTRY_COLORS["Others"],
}


# Source-cohort semantics used by Figure 1 and its supporting figures.  These
# hues encode data source/cohort, not genetic ancestry.  Keeping the mapping in
# this shared module prevents main/supplement drift.
COHORT_ORDER: Final[tuple[str, ...]] = (
    "UKB", "UK10K", "EAS", "SG10K", "AFR", "TopMed", "HGDP", "1kgenome"
)

COHORT_LABELS: Final[dict[str, str]] = {
    "UKB": "UK Biobank",
    "UK10K": "UK10K",
    "EAS": "Internal Chinese cohort",
    "SG10K": "SG10K",
    "AFR": "CAAPA",
    "TopMed": "TOPMed",
    "HGDP": "HGDP",
    "1kgenome": "1000 Genomes",
}

COHORT_SHORT_LABELS: Final[dict[str, str]] = {
    "UKB": "UKB",
    "UK10K": "UK10K",
    "EAS": "Internal Chinese",
    "SG10K": "SG10K",
    "AFR": "CAAPA",
    "TopMed": "TOPMed",
    "HGDP": "HGDP",
    "1kgenome": "1KGP",
}

COHORT_COLORS: Final[dict[str, str]] = {
    "UKB": "#0B2E83",
    "TopMed": "#2F6FD6",
    "UK10K": "#6E9FDE",
    "SG10K": "#5BC0C8",
    "1kgenome": "#0E7C86",
    "HGDP": "#25A59A",
    "EAS": "#E6AD35",
    "AFR": "#B35C1E",
}
