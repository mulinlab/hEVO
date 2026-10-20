"""Shared filesystem metadata fingerprints (not content hashes)."""

import hashlib
from pathlib import Path


def filesystem_fingerprint(value: str) -> str:
    root = Path(value)
    candidates = (
        [root]
        if root.is_file()
        else sorted(item for item in root.rglob("*") if item.is_file())
    )
    digest = hashlib.sha256()
    for item in candidates:
        stat = item.stat()
        relative = item.name if root.is_file() else str(item.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()
