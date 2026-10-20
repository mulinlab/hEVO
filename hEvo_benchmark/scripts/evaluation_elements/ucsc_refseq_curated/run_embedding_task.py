#!/usr/bin/env python3
"""Dispatch one frozen model's full UCSC embedding extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shared-extractor-dir", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    roster = json.loads((run_root / "config" / "model_roster.json").read_text(encoding="utf-8"))
    group = os.environ.get("MODEL_MEMORY_GB", "")
    group_models = roster["gpu_memory_groups_gb"].get(group, [])
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(group_models):
        raise IndexError(f"array index {index} outside memory-{group} model group of {len(group_models)}")
    model_name = group_models[index]
    spec = roster["model_specs"][model_name]
    if int(spec["memory_gb"]) != int(group):
        raise ValueError(f"resource-group mismatch for {model_name}")

    dataset_id = roster["dataset_id"]
    output_h5 = run_root / "embeddings" / dataset_id / f"{model_name}.h5"
    script = Path(__file__).resolve().parent / "extract_embeddings_h5.py"
    command = [
        sys.executable, str(script),
        "--input-csv", str(roster["dataset_csv"]),
        "--model-path", spec["model_path"],
        "--model-name", model_name,
        "--output-h5", str(output_h5),
        "--batch-size", str(spec["batch_size"]),
        "--max-length", str(spec["max_length"]),
        "--pooling", "mean",
        "--extractor-dir", str(args.shared_extractor_dir.resolve()),
        "--dataset-manifest", str(run_root / "config" / "dataset_manifest.json"),
        "--run-id", roster["run_id"],
        "--evo2-config-dir", str(Path(__file__).resolve().parents[1] / "evaluation" / "evo2_configs"),
    ]
    print(json.dumps({"model": model_name, "memory_gb": group, "output": str(output_h5), "command": command}), flush=True)
    subprocess.run(command, check=True)
    manifest_path = output_h5.with_suffix(".manifest.json")
    output_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if output_manifest.get("model_weight_sha256") != spec["model_weight_sha256_expected"]:
        raise ValueError(f"loaded model weights differ from frozen fingerprint for {model_name}")
    if output_manifest.get("pooling") != "mean" or output_manifest.get("max_length") != spec["max_length"]:
        raise ValueError(f"embedding extraction settings mismatch for {model_name}")
    digest = hashlib.sha256()
    with output_h5.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    output_manifest["output_h5_sha256"] = digest.hexdigest()
    output_manifest["fresh_extraction"] = True
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "model": model_name, "embedding_sha256": digest.hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
