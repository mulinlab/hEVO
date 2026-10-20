"""Public import shim for the manuscript-wide model display-name registry."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = _SCRIPT_ROOT / "_backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from figures.scripts.common.hevo_model_style import (  # noqa: E402
    MODEL_DISPLAY_ALIASES,
    MODEL_FAMILY_STYLE,
    MODEL_NAME_REGISTRY,
    canonical_model_name,
    model_family,
    model_registry_record,
    model_style,
)

__all__ = [
    "MODEL_DISPLAY_ALIASES",
    "MODEL_FAMILY_STYLE",
    "MODEL_NAME_REGISTRY",
    "canonical_model_name",
    "model_family",
    "model_registry_record",
    "model_style",
]
