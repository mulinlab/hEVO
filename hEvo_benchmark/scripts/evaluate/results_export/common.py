"""Small shared primitives used by result validators and table exporters."""

from __future__ import annotations

import hashlib
import csv
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(
    path: Path, *, require_header: bool = False
) -> tuple[list[str], list[dict[str, str]]]:
    """Read string-valued CSV rows with the exporters' header policies."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if require_header and reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames or []), list(reader)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    *,
    atomic: bool = False,
    atomic_style: str = "suffix",
    create_parent: bool = False,
    lineterminator: str = "\n",
    extrasaction: str = "raise",
) -> None:
    """Write CSV rows, retaining each caller's direct or atomic write policy."""
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(handle: Any) -> None:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction=extrasaction,
            lineterminator=lineterminator,
        )
        writer.writeheader()
        writer.writerows(rows)

    if not atomic:
        with path.open("w", encoding="utf-8", newline="") as handle:
            write(handle)
        return

    if atomic_style == "suffix":
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            write(handle)
        temporary.replace(path)
        return
    if atomic_style != "named":
        raise ValueError(f"unsupported atomic_style: {atomic_style!r}")

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        write(handle)
    os.replace(temporary, path)
