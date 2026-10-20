"""Serialize filesystem paths without publishing checkout-specific locations."""

from __future__ import annotations

from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(f"could not find a Git repository above {current}")


def recorded_path(path: str | Path) -> str:
    """Use stable repository placeholders and hide absolute external locations."""

    resolved = Path(path).expanduser().resolve()
    root = repository_root()
    benchmark = root / "hEvo_benchmark"
    try:
        return "<benchmark-root>/" + resolved.relative_to(benchmark).as_posix()
    except ValueError:
        pass
    try:
        return "<repo-root>/" + resolved.relative_to(root).as_posix()
    except ValueError:
        return "<external-path>/" + resolved.name


def publicize_paths(value, *, _field: str | None = None):
    """Recursively replace absolute path values in QA metadata."""

    if isinstance(value, Path):
        return recorded_path(value)
    if isinstance(value, dict):
        # Keep font identity useful while removing machine-specific install
        # locations. Filenames and family names are enough to identify faces.
        is_font_metadata = value.get("family") == "Arial" and "paths" in value
        return {
            key: publicize_paths(child, _field=key)
            for key, child in value.items()
            if not (is_font_metadata and key == "paths")
        }
    if isinstance(value, (list, tuple)):
        return [publicize_paths(child, _field=_field) for child in value]
    # PDF PostScript font names conventionally start with '/'; they are
    # identifiers, not filesystem locations.
    if _field == "font_names":
        return value
    if isinstance(value, str) and Path(value).is_absolute():
        return recorded_path(value)
    return value


def write_public_json(path: Path, payload, *, ensure_ascii: bool = False) -> None:
    """Write JSON after removing machine-specific paths from public metadata."""

    import json

    path.write_text(
        json.dumps(publicize_paths(payload), indent=2, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )
