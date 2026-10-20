from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelEntry:
    name: str
    url: str
    model_type: str | None = None
    pooling: str = "mean-pooling"
    batch_size: int | None = None
    max_length: int = 1024
    base: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model_entries(config_path: str | Path) -> list[ModelEntry]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entries: list[ModelEntry] = []
    os.environ.setdefault(
        "GPN_STAR_ASSETS_ROOT",
        str(Path(os.environ.get("HPC_HOME", Path.home())) / "gpn_star_assets"),
    )
    for raw in payload.get("Models", []) or []:
        if not isinstance(raw, dict) or "name" not in raw or "url" not in raw:
            continue
        entries.append(
            ModelEntry(
                name=str(raw["name"]),
                url=os.path.expandvars(str(raw["url"])),
                model_type=str(raw.get("type"))
                if raw.get("type") is not None
                else None,
                pooling=str(raw.get("future", "mean-pooling")),
                batch_size=int(raw["batch_size"])
                if raw.get("batch_size") is not None
                else None,
                max_length=int(raw.get("max_length", 1024)),
                base=str(raw.get("base")) if raw.get("base") is not None else None,
            )
        )
    if not entries:
        raise ValueError(f"No model entries found in {path}")
    return entries


def is_standard_entry(entry: ModelEntry) -> bool:
    """Return whether the first-generation HF/PEFT adapter can handle an entry."""
    name = entry.name.lower()
    return not entry.url.lower().endswith(".pt") and not name.startswith("gpn")


def select_model_entries(
    config_path: str | Path,
    model_name: str | None = None,
    all_models: bool = False,
    model_scope: str = "standard",
    model_index: int | None = None,
) -> list[ModelEntry]:
    entries = load_model_entries(config_path)
    if model_scope not in {"standard", "all"}:
        raise ValueError("model_scope must be 'standard' or 'all'")
    if model_scope == "standard":
        entries = [entry for entry in entries if is_standard_entry(entry)]

    if model_name:
        matches = [entry for entry in entries if entry.name == model_name]
        if not matches:
            available = ", ".join(entry.name for entry in entries)
            raise KeyError(f"Unknown model {model_name!r}. Available: {available}")
        return matches
    if not all_models:
        raise ValueError("Provide --model_name or --all")
    if model_index is not None:
        if model_index < 0 or model_index >= len(entries):
            raise IndexError(f"model_index {model_index} outside [0, {len(entries)})")
        return [entries[model_index]]
    return entries
