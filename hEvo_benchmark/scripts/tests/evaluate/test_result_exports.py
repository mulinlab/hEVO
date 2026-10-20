"""CSV compatibility checks for shared result-export primitives."""

import hashlib

import pytest

from results_export import common
from results_export import (
    export_hevo_large_sequence_classification_tables as large_export,
    merge_gpn_msa_gb_probe_results as gb_merge,
    merge_gpn_msa_nt_native_results as nt_merge,
    merge_gpn_msa_unsupervised_results as unsupervised_merge,
    merge_gpn_star_probe_results as star_merge,
)


READERS = (
    common.read_csv,
    gb_merge.read_csv,
    nt_merge.read_csv,
    unsupervised_merge.read_csv,
    star_merge.read_csv,
    large_export.read_csv,
)


def test_read_csv_shared_semantics_and_header_policy(tmp_path):
    fixture = tmp_path / "input.csv"
    fixture.write_bytes(b"id,value\r\n1,first\r\n2,second\r\n")
    expected = (
        ["id", "value"],
        [{"id": "1", "value": "first"}, {"id": "2", "value": "second"}],
    )
    for reader in READERS:
        assert reader(fixture) == expected

    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    for reader in READERS[:-1]:
        assert reader(empty) == ([], [])
    with pytest.raises(ValueError, match="missing CSV header"):
        large_export.read_csv(empty)


def test_shared_writers_preserve_csv_bytes_and_atomic_modes(tmp_path):
    fields = ["id", "value"]
    rows = [{"id": "1", "value": "first"}, {"id": "2", "value": "second"}]
    expected = b"id,value\n1,first\n2,second\n"

    direct = tmp_path / "direct.csv"
    star_merge.write_csv(direct, fields, rows)
    assert direct.read_bytes() == expected

    suffix_atomic = tmp_path / "suffix.csv"
    unsupervised_merge.write_csv(suffix_atomic, fields, rows)
    assert suffix_atomic.read_bytes() == expected
    assert not suffix_atomic.with_suffix(".csv.tmp").exists()

    named_atomic = tmp_path / "new-parent" / "named.csv"
    large_export.write_csv_atomic(named_atomic, fields, rows)
    assert named_atomic.read_bytes() == expected


def test_atomic_writers_preserve_previous_file_when_row_validation_fails(tmp_path):
    bad_rows = [{"id": "1", "unexpected": "extra"}]
    original_bytes = b"previous,content\n"

    suffix_atomic = tmp_path / "suffix.csv"
    suffix_atomic.write_bytes(original_bytes)
    with pytest.raises(ValueError, match="fields not in fieldnames"):
        unsupervised_merge.write_csv(suffix_atomic, ["id"], bad_rows)
    assert suffix_atomic.read_bytes() == original_bytes

    named_atomic = tmp_path / "named.csv"
    named_atomic.write_bytes(original_bytes)
    with pytest.raises(ValueError, match="fields not in fieldnames"):
        large_export.write_csv_atomic(named_atomic, ["id"], bad_rows)
    assert named_atomic.read_bytes() == original_bytes


def test_shared_sha256_uses_standard_file_digest(tmp_path):
    fixture = tmp_path / "bytes.bin"
    fixture.write_bytes(b"csv helper digest fixture\n")
    assert common.sha256(fixture) == hashlib.sha256(fixture.read_bytes()).hexdigest()


def test_task_rank_policies_keep_their_existing_tie_behavior():
    rows = [
        {
            "dataset": "task",
            "model": "B",
            "mcc_mean": "0.8",
            "task_rank": "",
            "task_count": "",
        },
        {
            "dataset": "task",
            "model": "A",
            "mcc_mean": "0.8",
            "task_rank": "",
            "task_count": "",
        },
        {
            "dataset": "task",
            "model": "C",
            "mcc_mean": "0.3",
            "task_rank": "",
            "task_count": "",
        },
    ]
    competition_rows = [dict(row) for row in rows]
    gb_merge.competition_ranks(competition_rows)
    competition_ranks = {row["model"]: row["task_rank"] for row in competition_rows}
    assert competition_ranks == {"A": "1", "B": "1", "C": "3"}

    sequential_rows = [dict(row) for row in rows]
    large_export.rank_by_task(sequential_rows)
    sequential_ranks = {row["model"]: row["task_rank"] for row in sequential_rows}
    assert sequential_ranks == {"B": "1", "A": "2", "C": "3"}
