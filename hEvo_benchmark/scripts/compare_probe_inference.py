#!/usr/bin/env python3
"""Run baseline/current probe CLIs on small balanced NT task samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CURRENT_CLI = ROOT / "hEvo_benchmark/scripts/evaluate/probe_single_seq_input.py"
DEFAULT_BASELINE = Path("/tmp/hevo-eval-baseline-9405e9d")
DEFAULT_NT_DATA = Path("/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/datasets/NT_formatted")
DEFAULT_MODEL = Path("/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/models/DNABERT-2-117M")
DEFAULT_OUTPUT = Path("/tmp/hevo-eval-regression-20261017/probe-inference")
TASKS = ("enhancers", "splice_sites_all")
SPLITS = ("train", "test")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--nt-data", type=Path, default=DEFAULT_NT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default="DNABERT2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_split(
    source: Path, destination: Path, task: str, split: str, per_class: int, seed: int
) -> dict[str, Any]:
    """Sample balanced rows by the final whitespace-delimited label column."""
    groups: dict[int, list[tuple[int, str, list[str]]]] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) < 2 or not fields[0]:
                continue
            try:
                label = int(fields[-1])
            except ValueError:
                continue
            groups.setdefault(label, []).append(
                (line_number, line.rstrip("\r\n"), fields)
            )

    rng = random.Random(seed + sum(ord(char) for char in task + split))
    selected = []
    for label in sorted(groups):
        rows = groups[label]
        rng.shuffle(rows)
        if len(rows) < per_class:
            raise RuntimeError(
                f"{source} has {len(rows)} rows for label {label}; need {per_class}"
            )
        selected.extend(rows[:per_class])
    selected.sort(key=lambda row: (int(row[2][-1]), row[0]))
    field_counts = sorted({len(row[2]) for row in selected})
    if len(field_counts) != 1:
        raise RuntimeError(f"{source} has inconsistent row widths: {field_counts}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(row[1] for row in selected) + "\n", encoding="utf-8"
    )
    lengths = sorted({len(row[2][0]) for row in selected})
    return {
        "source": str(source),
        "fixture": str(destination),
        "sha256": _sha256(destination),
        "rows": len(selected),
        "class_counts": {
            str(label): sum(int(row[2][-1]) == label for row in selected)
            for label in sorted(groups)
        },
        "sequence_lengths": lengths,
        "label_column_index": field_counts[0] - 1,
        "source_line_numbers": [row[0] for row in selected],
        "sequence_sha256": [
            hashlib.sha256(row[2][0].encode()).hexdigest() for row in selected
        ],
    }


def prepare_fixtures(args) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "sampling_seed": args.seed,
        "per_class": args.per_class,
        "label_column": "last field",
        "sequence_column": 0,
        "tasks": {},
    }
    fixture_dir = args.output_dir / "fixtures"
    for task in TASKS:
        task_info = {}
        for split in SPLITS:
            source = args.nt_data / task / f"{split}.txt"
            fixture = fixture_dir / task / f"{split}.txt"
            task_info[split] = sample_split(
                source, fixture, task, split, args.per_class, args.seed
            )
        manifest["tasks"][task] = task_info
    return manifest


def run_cli(
    version: str, task: str, args, fixture_info: dict[str, Any]
) -> dict[str, Any]:
    cli = (
        args.baseline / "hEvo_benchmark/scripts/evaluate/probe_single_seq_input.py"
        if version == "old"
        else CURRENT_CLI
    )
    run_dir = args.output_dir / args.model_name / task / version
    out_dir, log_dir, hdf5_dir = (run_dir / name for name in ("out", "logs", "hdf5"))
    for directory in (out_dir, log_dir, hdf5_dir):
        directory.mkdir(parents=True, exist_ok=True)
    train_file = fixture_info["train"]["fixture"]
    test_file = fixture_info["test"]["fixture"]
    command = [
        sys.executable,
        str(cli),
        "--model_paths",
        str(args.model),
        "--model_name",
        args.model_name,
        "--dataset_files",
        f"{train_file},{test_file}",
        "--dataset_name",
        task,
        "--seq_col",
        "0",
        "--label_col",
        str(fixture_info["train"]["label_column_index"]),
        "--sep",
        "space",
        "--max_length",
        "512",
        "--epochs",
        "10",
        "--bootstrap",
        "1000",
        "--batch_size",
        "64",
        "--precompute_batch_size",
        "64",
        "--num_workers",
        "0",
        "--pooling",
        "mean",
        "--recompute",
        "--out_dir",
        str(out_dir),
        "--log_dir",
        str(log_dir),
        "--hdf5_dir",
        str(hdf5_dir),
    ]
    if version == "new":
        command.extend(("--protocol", "standard"))
    environment = os.environ.copy()
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device.startswith("cuda"):
        if "," in args.device:
            raise ValueError("Use one CUDA device per comparison process")
        visible_device = args.device.removeprefix("cuda:")
        if visible_device != "cuda":
            environment["CUDA_VISIBLE_DEVICES"] = visible_device
    else:
        environment["CUDA_VISIBLE_DEVICES"] = args.device

    log_path = run_dir / "cli.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    results_files = sorted(out_dir.glob("results_*.json"))
    hdf5_files = sorted(hdf5_dir.glob("*.h5"))
    report: dict[str, Any] = {
        "version": version,
        "task": task,
        "command": command,
        "cli_path": str(cli),
        "cli_sha256": _sha256(cli),
        "return_code": completed.returncode,
        "log": str(log_path),
        "run_dir": str(run_dir),
        "status": "ok" if completed.returncode == 0 else "error",
    }
    if completed.returncode != 0:
        report["error"] = f"CLI returned exit status {completed.returncode}"
        return report
    if len(results_files) != 1 or len(hdf5_files) != 1:
        report.update(
            {
                "status": "error",
                "error": f"Expected one results JSON and HDF5; found "
                f"{len(results_files)} JSON and {len(hdf5_files)} HDF5 files",
            }
        )
        return report

    report["results_json"] = str(results_files[0])
    report["results_sha256"] = _sha256(results_files[0])
    report["results"] = json.loads(results_files[0].read_text(encoding="utf-8"))
    report["hdf5"] = str(hdf5_files[0])
    report["hdf5_sha256"] = _sha256(hdf5_files[0])
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["run_manifest"] = {
            "entrypoint_sha256": manifest.get("code", {}).get("sha256"),
            "model_fingerprints": [
                value.get("filesystem_fingerprint")
                for value in manifest.get("model_paths", [])
            ],
            "protocol": manifest.get("protocol"),
            "run_status": manifest.get("run_status"),
        }
    return report


def compare_hdf5(old_path: Path, new_path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    comparison: dict[str, Any] = {"splits": {}, "exact": True}
    with h5py.File(old_path, "r") as old, h5py.File(new_path, "r") as new:
        if set(old.keys()) != set(new.keys()):
            return {
                "exact": False,
                "error": f"Top-level HDF5 groups differ: {list(old.keys())} vs {list(new.keys())}",
            }
        for split in sorted(old.keys()):
            comparison["splits"][split] = {}
            if set(old[split].keys()) != set(new[split].keys()):
                comparison["exact"] = False
                comparison["splits"][split]["error"] = (
                    f"Datasets differ: {list(old[split].keys())} vs {list(new[split].keys())}"
                )
                continue
            for name in sorted(old[split].keys()):
                old_values = old[split][name][...]
                new_values = new[split][name][...]
                same_shape = old_values.shape == new_values.shape
                same = same_shape and np.array_equal(
                    old_values, new_values, equal_nan=True
                )
                item = {
                    "old_shape": list(old_values.shape),
                    "new_shape": list(new_values.shape),
                    "exact": bool(same),
                }
                if same_shape and old_values.dtype.kind in "fiu" and old_values.size:
                    difference = np.asarray(new_values, dtype=np.float64) - np.asarray(
                        old_values, dtype=np.float64
                    )
                    item["max_abs_diff"] = float(np.max(np.abs(difference)))
                    item["mean_abs_diff"] = float(np.mean(np.abs(difference)))
                comparison["splits"][split][name] = item
                comparison["exact"] &= bool(same)
    return comparison


def compare_task(task: str, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    if old.get("status") != "ok" or new.get("status") != "ok":
        return {
            "status": "error",
            "old_status": old.get("status"),
            "new_status": new.get("status"),
            "old_error": old.get("error"),
            "new_error": new.get("error"),
        }
    hdf5 = compare_hdf5(Path(old["hdf5"]), Path(new["hdf5"]))
    metrics_exact = old["results"] == new["results"]
    return {
        "status": "ok" if hdf5["exact"] and metrics_exact else "mismatch",
        "hdf5": hdf5,
        "metrics_exact_after_json_parse": metrics_exact,
        "old_metrics": old["results"],
        "new_metrics": new["results"],
        "old_run_manifest": old.get("run_manifest"),
        "new_run_manifest": new.get("run_manifest"),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.per_class < 1:
        raise SystemExit("--per-class must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = prepare_fixtures(args)
    fixtures_path = args.output_dir / "fixtures.json"
    fixtures_path.write_text(
        json.dumps(fixtures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report: dict[str, Any] = {
        "model": str(args.model),
        "model_name": args.model_name,
        "fixtures": str(fixtures_path),
        "sampling": fixtures,
        "tasks": {},
    }
    mismatch = False
    for task in TASKS:
        versions = {
            version: run_cli(version, task, args, fixtures["tasks"][task])
            for version in ("old", "new")
        }
        report["tasks"][task] = {
            **versions,
            "comparison": compare_task(task, versions["old"], versions["new"]),
        }
        mismatch |= report["tasks"][task]["comparison"]["status"] != "ok"

    report["status"] = "mismatch" if mismatch else "ok"
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
