"""Canonical model naming and family styling for hEVO benchmark figures.

The manuscript uses colour to encode a small number of prespecified model
families.  All remaining models are deliberately neutral so that a panel with
30+ rows remains legible and does not become a categorical rainbow.
"""

from __future__ import annotations

# [Shared styles] Centralize canonical model names, display names, families, and color mappings.
# [Usage] Fig. 1–4 use this module to resolve model aliases, family groups, and legend labels; original model IDs remain in the source data.
# [Note] Changes here affect labels and colors across multiple figures, so this is a shared style dependency.

from typing import Final


BLACK: Final[str] = "#000000"
WHITE: Final[str] = "#FFFFFF"

MODEL_FAMILY_STYLE: Final[dict[str, dict[str, str]]] = {
    # Locked manuscript-wide model-family colors (2026-08-11).
    "hEVO": {"color": "#B35C1E", "marker": "o"},
    "Evo 2": {"color": "#0E7C86", "marker": "^"},
    "NT": {"color": "#65717F", "marker": "s"},
    "GPN": {"color": "#0B2E83", "marker": "D"},
    "Other": {"color": "#B8BEC8", "marker": "o"},
}


MODEL_DISPLAY_ALIASES: Final[dict[str, str]] = {
    "hEvo-open-base": "hEVO-base-o",
    "hEvo-open-large": "hEVO-large-o",
    "hEvo-1kG-s128": "hEVO-1KGP-s128-86M",
    "hEvo-1kG-sliding128-86M": "hEVO-1KGP-s128-86M",
    "hEvo-fullUKB": "hEVO-full-UKB-86M",
    "hEvo-fullUKB-86M": "hEVO-full-UKB-86M",
    "hEvo-60k": "hEVO-60K-86M",
    "hEvo-60k-86M": "hEVO-60K-86M",
    "hEvo-1kG": "hEVO-1KGP-86M",
    "hEvo-1kG-86M": "hEVO-1KGP-86M",
    "evo2_1b_base": "Evo2-1B",
    "evo2_7b_base": "Evo2-7B",
    "Evo2-1B": "Evo2-1B",
    "Evo2-7B": "Evo2-7B",
    "EVO 2-1B": "Evo2-1B",
    "EVO 2-7B": "Evo2-7B",
    "Evo 2 1B base": "Evo2-1B",
    "Evo 2 7B base": "Evo2-7B",
    "NT_500m_MS": "NTv2-500M-MS",
    "NT_500m_1kG": "NTv2-500M-1kG",
    "NT_2.5b_MS": "NTv2-2.5B-MS",
    "NT_2.5b_1kG": "NTv2-2.5B-1kG",
    "NT-500M-1kG": "NTv2-500M-1kG",
    "NT-2.5B-1kG": "NTv2-2.5B-1kG",
    "NT-500M-MS": "NTv2-500M-MS",
    "NT-2.5B-MS": "NTv2-2.5B-MS",
    "NTv1-1000G-500M": "NTv2-500M-1kG",
    "NTv1-1000G-2.5B": "NTv2-2.5B-1kG",
    "NTv1-MS-2.5B": "NTv2-2.5B-MS",
    "NTv2-MS-500M": "NTv2-500M-MS",
    "NT 500M multispecies": "NTv2-500M-MS",
    "NT-v2-MS-500M": "NTv2-500M-MS",
    "NT-MS-500M": "NTv2-500M-MS",
    "NT 2.5B multispecies": "NTv2-2.5B-MS",
    "NT-MS-2.5B": "NTv2-2.5B-MS",
    "NT 500M 1000G": "NTv2-500M-1kG",
    "NT-1000G-500M": "NTv2-500M-1kG",
    "NT 2.5B 1000G": "NTv2-2.5B-1kG",
    "NT-1000G-2.5B": "NTv2-2.5B-1kG",
    "NTv3-8M-pre": "NTv3-8M",
    "NTv3-100M-pre": "NTv3-100M",
    "NTv3-650M-pre": "NTv3-650M",
    "NTv3 8M pre-trained": "NTv3-8M",
    "NTv3 100M pre-trained": "NTv3-100M",
    "NTv3 650M pre-trained": "NTv3-650M",
    "DNABERT2": "DNABERT2",
    "DNABERT-S": "DNABERT",
    "DNABERT-2": "DNABERT2",
    "DNABERT-2-117M": "DNABERT2",
    "DNABERT2-117M": "DNABERT2",
    "AIDO.DNA-300M": "AIDO.DNA-300M",
    "GPN-promoter": "GPN-Promoter",
    "GPN-MSA": "GPN-MSA",
    "HyenaDNA_tiny_1k": "HyenaDNA-1K",
    "HyenaDNA-tiny": "HyenaDNA-1K",
    "GENA-LM-base": "GENA-LM-base",
    "GENA-LM-large": "GENA-LM-large",
    "GENA-LM-base-110M": "GENA-LM-base",
    "GENA-LM-large-336M": "GENA-LM-large",
    "Caduceus-PH": "Caduceus-PH",
    "Caduceus-PS": "Caduceus-PS",
    "Caduceus-Ph": "Caduceus-PH",
    "Caduceus-PS-131K": "Caduceus-PS",
    "DNABERT": "DNABERT",
    "SpeciesLM": "SpeciesLM",
    "SpeciesLM-Metazoa": "SpeciesLM",
}


# Central manuscript registry. ``display`` is the only label intended for
# figures; ``raw_id`` remains the immutable link to result tables and code.
# ``evidence`` separates names verified in a paper/model card from project
# release identifiers that still need a manuscript citation.
MODEL_NAME_REGISTRY: Final[dict[str, dict[str, str]]] = {
    "hEvo-base": {
        "display": "hEVO-base",
        "algorithm": "hEVO",
        "parameters": "87.095M",
        "variant": "base",
        "evidence": "project-defined",
    },
    "hEvo-large": {
        "display": "hEVO-large",
        "algorithm": "hEVO",
        "parameters": "319.462M",
        "variant": "large",
        "evidence": "project-defined",
    },
    "evo2_1b_base": {
        "display": "Evo2-1B",
        "algorithm": "Evo 2",
        "parameters": "1B",
        "variant": "base; 8K context checkpoint",
        "evidence": "official-model-card",
    },
    "evo2_7b_base": {
        "display": "Evo2-7B",
        "algorithm": "Evo 2",
        "parameters": "7B",
        "variant": "base; 8K context checkpoint",
        "evidence": "official-model-card",
    },
    "NT_500m_MS": {
        "display": "NTv2-500M-MS",
        "algorithm": "Nucleotide Transformer v2",
        "parameters": "500M",
        "variant": "v2; multispecies",
        "evidence": "official-model-card-inferred-project-mapping",
    },
    "NT_2.5b_MS": {
        "display": "NTv2-2.5B-MS",
        "algorithm": "Nucleotide Transformer v1",
        "parameters": "2.5B",
        "variant": "multispecies",
        "evidence": "paper-aligned-project-mapping",
    },
    "NT_500m_1kG": {
        "display": "NTv2-500M-1kG",
        "algorithm": "Nucleotide Transformer v1",
        "parameters": "500M",
        "variant": "1000 Genomes",
        "evidence": "paper-verified",
    },
    "NT_2.5b_1kG": {
        "display": "NTv2-2.5B-1kG",
        "algorithm": "Nucleotide Transformer v1",
        "parameters": "2.5B",
        "variant": "1000 Genomes",
        "evidence": "paper-verified",
    },
    "NTv3-8M-pre": {
        "display": "NTv3-8M",
        "algorithm": "Nucleotide Transformer v3",
        "parameters": "8M",
        "variant": "pre-trained",
        "evidence": "official-model-card",
    },
    "NTv3-100M-pre": {
        "display": "NTv3-100M",
        "algorithm": "Nucleotide Transformer v3",
        "parameters": "100M",
        "variant": "pre-trained",
        "evidence": "official-model-card",
    },
    "NTv3-650M-pre": {
        "display": "NTv3-650M",
        "algorithm": "Nucleotide Transformer v3",
        "parameters": "650M",
        "variant": "pre-trained",
        "evidence": "official-model-card",
    },
    "GPN-promoter": {
        "display": "GPN-Promoter",
        "algorithm": "GPN",
        "parameters": "",
        "variant": "promoter-trained",
        "evidence": "paper-verified",
    },
    "GPN-MSA": {
        "display": "GPN-MSA",
        "algorithm": "GPN",
        "parameters": "",
        "variant": "multiple-sequence alignment",
        "evidence": "paper-verified",
    },
    "Caduceus-PH": {
        "display": "Caduceus-PH",
        "algorithm": "Caduceus",
        "parameters": "",
        "variant": "post hoc RC conjoining; 131K context; d_model 256; 16 layers",
        "evidence": "paper-verified",
    },
    "Caduceus-PS": {
        "display": "Caduceus-PS",
        "algorithm": "Caduceus",
        "parameters": "",
        "variant": "RC parameter sharing; 131K context; d_model 256; 16 layers",
        "evidence": "paper-verified",
    },
    "DNABERT": {
        "display": "DNABERT",
        "algorithm": "DNABERT",
        "parameters": "",
        "variant": "S checkpoint",
        "evidence": "project-config-verified",
    },
    "DNABERT2": {
        "display": "DNABERT2",
        "algorithm": "DNABERT-2",
        "parameters": "117M",
        "variant": "",
        "evidence": "paper-verified",
    },
    "GENA-LM-base": {
        "display": "GENA-LM-base",
        "algorithm": "GENA-LM",
        "parameters": "110M",
        "variant": "base",
        "evidence": "paper-verified",
    },
    "GENA-LM-large": {
        "display": "GENA-LM-large",
        "algorithm": "GENA-LM",
        "parameters": "336M",
        "variant": "large",
        "evidence": "paper-verified",
    },
    "HyenaDNA_tiny_1k": {
        "display": "HyenaDNA-1K",
        "algorithm": "HyenaDNA",
        "parameters": "",
        "variant": "tiny; 1K context",
        "evidence": "paper-aligned-project-mapping",
    },
    "AIDO.DNA-300M": {
        "display": "AIDO.DNA-300M",
        "algorithm": "AIDO.DNA",
        "parameters": "300M",
        "variant": "",
        "evidence": "project-release-id",
    },
    "CrossDNA-8.1M": {
        "display": "CrossDNA-8.1M",
        "algorithm": "CrossDNA",
        "parameters": "8.1M",
        "variant": "",
        "evidence": "project-release-id",
    },
    "CrossDNA-28.6M": {
        "display": "CrossDNA-28.6M",
        "algorithm": "CrossDNA",
        "parameters": "28.6M",
        "variant": "",
        "evidence": "project-release-id",
    },
    "CrossDNA-71.6M": {
        "display": "CrossDNA-71.6M",
        "algorithm": "CrossDNA",
        "parameters": "71.6M",
        "variant": "",
        "evidence": "project-release-id",
    },
    "CrossDNA-519M": {
        "display": "CrossDNA-519M",
        "algorithm": "CrossDNA",
        "parameters": "519M",
        "variant": "",
        "evidence": "project-release-id",
    },
    "Mistral-DNA-17M": {
        "display": "Mistral-DNA-17M",
        "algorithm": "Mistral-DNA",
        "parameters": "17M",
        "variant": "v1; hg38",
        "evidence": "project-config-verified",
    },
    "Mistral-DNA-138M": {
        "display": "Mistral-DNA-138M",
        "algorithm": "Mistral-DNA",
        "parameters": "138M",
        "variant": "v1; hg38",
        "evidence": "project-config-verified",
    },
    "Mistral-DNA-422M": {
        "display": "Mistral-DNA-422M",
        "algorithm": "Mistral-DNA",
        "parameters": "422M",
        "variant": "v1; hg38",
        "evidence": "project-config-verified",
    },
    "GENERator-1.2B": {
        "display": "GENERator-1.2B",
        "algorithm": "GENERator",
        "parameters": "1.2B",
        "variant": "v2; eukaryote; base",
        "evidence": "project-config-verified",
    },
    "GENERator-3B": {
        "display": "GENERator-3B",
        "algorithm": "GENERator",
        "parameters": "3B",
        "variant": "v2; eukaryote; base",
        "evidence": "project-config-verified",
    },
    "Genos-1.2B": {
        "display": "Genos-1.2B",
        "algorithm": "Genos",
        "parameters": "1.2B",
        "variant": "",
        "evidence": "project-release-id",
    },
    "SpeciesLM": {
        "display": "SpeciesLM",
        "algorithm": "SpeciesLM",
        "parameters": "",
        "variant": "metazoa; upstream; k6",
        "evidence": "project-config-verified",
    },
    "GROVER": {
        "display": "GROVER",
        "algorithm": "GROVER",
        "parameters": "",
        "variant": "",
        "evidence": "paper-verified",
    },
}


def canonical_model_name(name: str) -> str:
    """Return the manuscript display name without altering unknown models."""

    clean = str(name).strip()
    if clean in MODEL_NAME_REGISTRY:
        return MODEL_NAME_REGISTRY[clean]["display"]
    return MODEL_DISPLAY_ALIASES.get(clean, clean)


def model_registry_record(name: str) -> dict[str, str]:
    """Return a stable naming record while preserving unregistered raw IDs."""

    clean = str(name).strip()
    if clean in MODEL_NAME_REGISTRY:
        return {"raw_id": clean, **MODEL_NAME_REGISTRY[clean]}
    return {
        "raw_id": clean,
        "display": MODEL_DISPLAY_ALIASES.get(clean, clean),
        "algorithm": clean,
        "parameters": "",
        "variant": "",
        "evidence": "unregistered",
    }


def model_family(name: str) -> str:
    """Assign a model to one of the manuscript's locked visual families."""

    display = canonical_model_name(name).lower()
    if display.startswith("hevo"):
        return "hEVO"
    if display.startswith("evo 2") or display.startswith("evo2"):
        return "Evo 2"
    if display.startswith("nt-") or display.startswith("nt ") or display.startswith(("ntv1", "ntv2", "ntv3")):
        return "NT"
    if display.startswith("gpn-"):
        return "GPN"
    return "Other"


def model_style(name: str) -> dict[str, str]:
    """Return a copy of the locked style for ``name``."""

    return dict(MODEL_FAMILY_STYLE[model_family(name)])
