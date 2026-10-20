"""Regression cases for the adaptive-allele benchmark utilities."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_io import sha256
from tokens import changed_block


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([], [], ([], [])),
        ([1, 2], [1, 2], ([], [])),
        ([1, 2, 3], [1, 4, 3], ([1], [1])),
        ([1, 2, 3], [1, 4, 5, 3], ([1], [1, 2])),
        ([1, 3], [1, 2, 3], ([], [1])),
        ([1, 2, 3], [1, 3], ([1], [])),
        ([1, 2], [3, 4], ([0, 1], [0, 1])),
        ([], [1], ([], [0])),
    ],
)
def test_changed_token_blocks(left, right, expected):
    assert changed_block(left, right) == expected
    assert changed_block(right, left) == expected[::-1]


@pytest.mark.parametrize("size", [0, 1, (1 << 20) + 17])
def test_fingerprint_across_streaming_boundary(tmp_path, size):
    data = (b"ACGT" * ((size + 3) // 4))[:size]
    path = tmp_path / "source.bin"
    path.write_bytes(data)
    assert sha256(path) == hashlib.sha256(data).hexdigest()
