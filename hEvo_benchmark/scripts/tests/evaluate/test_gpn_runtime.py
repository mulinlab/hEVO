"""CPU-only checks for legacy GPN shard commands and moved implementations."""

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def run(command):
    subprocess.run(command, check=True, capture_output=True, text=True)


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_legacy_split_and_merge_wrappers_round_trip_tsv(tmp_path):
    scripts = Path(__file__).resolve().parents[2]
    source = tmp_path / "normalized.tsv"
    source.write_text(
        "variant_id\tsource_row\traw_llr\tstatus\n"
        "v2\t2\t-2.0\tPASS\n"
        "v1\t1\t1.0\tPASS\n"
        "v3\t3\t0.5\tSKIP\n",
        encoding="utf-8",
    )
    shard_dir = tmp_path / "shards"
    split_manifest = tmp_path / "split.json"
    run(
        [
            sys.executable,
            str(scripts / "evaluate/compat/gpn_star_split.py"),
            "--input",
            str(source),
            "--output-dir",
            str(shard_dir),
            "--shards",
            "2",
            "--manifest",
            str(split_manifest),
        ]
    )
    shard_paths = sorted(shard_dir.glob("shard_*.tsv"))
    assert len(shard_paths) == 2
    assert [len(read_tsv(path)) for path in shard_paths] == [2, 1]
    assert json.loads(split_manifest.read_text())["rows"] == 3

    merged_path = tmp_path / "merged.tsv"
    merge_manifest = tmp_path / "merge.json"
    command = [
        sys.executable,
        str(scripts / "evaluate/gpn_star_variant/merge_score_shards.py"),
    ]
    for shard in shard_paths:
        command.extend(("--input", str(shard)))
    command.extend(
        (
            "--output",
            str(merged_path),
            "--task-name",
            "fixture",
            "--manifest",
            str(merge_manifest),
        )
    )
    run(command)
    merged = read_tsv(merged_path)
    assert [row["source_row"] for row in merged] == ["1", "2", "3"]
    assert [row["variant_id"] for row in merged] == ["v1", "v2", "v3"]
    assert {row["task_name"] for row in merged} == {"fixture"}
    assert json.loads(merge_manifest.read_text())["rows"] == 3


def test_legacy_native_clinvar_aggregator_wrapper_on_small_fixture(tmp_path):
    scripts = Path(__file__).resolve().parents[2]
    data = tmp_path / "clinvar.csv"
    data.write_text("SeqID,label\nv1,1\nv2,0\n", encoding="utf-8")
    score_paths = []
    manifest_paths = []
    scores = (("v1", 1, 0.8, 0.2), ("v2", 0, 0.2, 0.8))
    for index, (seq_id, label, p_ref, p_alt) in enumerate(scores):
        score_path = tmp_path / "shard_{}.tsv".format(index)
        score = math.log(p_ref) - math.log(p_alt + 1e-10)
        score_path.write_text(
            "SeqID\tlabel\tcontext_bp\tscore\tp_ref\tp_alt\n"
            f"{seq_id}\t{label}\t6\t{score}\t{p_ref}\t{p_alt}\n",
            encoding="utf-8",
        )
        manifest_path = tmp_path / "shard_{}.json".format(index)
        manifest_path.write_text(
            json.dumps(
                {
                    "family": "generator",
                    "context_bp": 1024,
                    "data_path": str(data),
                    "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                    "model_path": "fixture-model",
                }
            ),
            encoding="utf-8",
        )
        score_paths.append(score_path)
        manifest_paths.append(manifest_path)

    output_dir = tmp_path / "aggregate"
    command = [
        sys.executable,
        str(scripts / "evaluate/compat/aggregate_native_clinvar_shards.py"),
        "--model-name",
        "fixture",
        "--data",
        str(data),
        "--output-dir",
        str(output_dir),
        "--bootstrap",
        "0",
        "--scores",
        *(str(path) for path in score_paths),
        "--manifests",
        *(str(path) for path in manifest_paths),
    ]
    run(command)
    metrics = json.loads((output_dir / "fixture.metrics.json").read_text())
    assert metrics["num_samples"] == 2
    assert metrics["num_positive"] == metrics["num_negative"] == 1
    combined = read_tsv(output_dir / "fixture.scores.tsv")
    assert [row["SeqID"] for row in combined] == ["v1", "v2"]
    assert (
        json.loads((output_dir / "fixture.manifest.json").read_text())[
            "num_samples_scored"
        ]
        == 2
    )
