"""Shared ancestry categories and table I/O."""
import csv
import hashlib
from pathlib import Path

CLASSES = ["AFR", "EUR", "EAS", "SAS", "AMR", "MID", "OCE"]
FOUR = CLASSES[:4]
FIVE = FOUR + ["Others"]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def read_tsv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def write_tsv(path, rows, fields=None):
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
