#!/usr/bin/env python3
"""Prepare or execute bounded Slurm array tasks for a frozen element run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    root = args.run_dir.resolve()
    gate = read(root / "qa/dataset_gate.json")
    independent = read(root / "qa/independent_dataset_summary.json")
    models = read(root / "qa/model_preflight.json")
    if gate["status"] != "PASS" or independent["status"] != "PASS" or models["status"] != "FILES_PRESENT":
        raise ValueError("dataset and model-input gates must pass before preparing GPU tasks")
    if gate["dataset_sha256"] != independent["dataset_sha256"]:
        raise ValueError("independent dataset hashes disagree")
    code = Path(__file__).resolve().parent
    snapshot = code / "shared"
    required_adapters = ("probe_single_seq_input.py", "extract_embeddings_classes.py")
    if any(not (snapshot / name).is_file() for name in required_adapters):
        raise FileNotFoundError(f"bundled model adapters are incomplete: {snapshot}")
    tasks = []
    for model in models["models"]:
        batch_size = model["batch_size"]
        if "2.5b" in model["model_name"] or model["model_name"] == "GENERator-3B":
            batch_size = min(batch_size, 8)
        max_length = model["max_length"]
        if model["model_name"].startswith("Caduceus-"):
            max_length = max(max_length, 514)
        for dataset, expected_hash in gate["dataset_sha256"].items():
            dataset_path = root / "datasets" / f"{dataset}.csv"
            if digest(dataset_path) != expected_hash:
                raise ValueError("dataset file changed after passing the gate")
            tasks.append({"index": len(tasks), "model_name": model["model_name"], "model_path": model["model_path"],
                          "dataset_id": dataset, "dataset_sha256": expected_hash,
                          "input_csv": str(dataset_path), "output_h5": str(root / "embeddings" / dataset / f"{model['model_name']}.h5"),
                          "batch_size": batch_size, "max_length": max_length})
    if len(tasks) != 58:
        raise ValueError("expected exactly 58 model/dataset embedding tasks")
    plan = {"run_id": gate["run_id"], "protocol_id": gate["protocol_id"], "tasks": tasks,
            "extractor_dir": str(snapshot), "dataset_manifest": str(root / "datasets/dataset_manifest.json"),
            "shared_adapter_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")}, "gpu_concurrency": 4}
    plan_path = root / "evaluation_tasks.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("existing execution plan differs; use a new run")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    (root / "logs").mkdir(exist_ok=True)
    slurm = root / "slurm"
    slurm.mkdir(exist_ok=True)
    for mode, partition, cpus, mem, extra, cap in (
        ("embedding", "gpu", 4, "80G", "#SBATCH --gres=gpu:1\n", 4),
        ("reduction", "cpuPartition", 8, "64G", "", 8),
    ):
        script = slurm / f"{mode}.sbatch"
        command = shlex.join(["micromamba", "-n", "llmbm", "run", "python", str(Path(__file__).resolve()), "worker",
                              "--run-dir", str(root), "--mode", mode])
        script.write_text(f"#!/usr/bin/env bash\n#SBATCH --job-name=elements-{mode}\n#SBATCH --partition={partition}\n"
                          f"#SBATCH --cpus-per-task={cpus}\n#SBATCH --mem={mem}\n#SBATCH --array=0-57%{cap}\n{extra}"
                          f"#SBATCH --output={root}/logs/{mode}.%A_%a.out\n#SBATCH --error={root}/logs/{mode}.%A_%a.err\n"
                          "set -euo pipefail\nexport LC_ALL=C\nexport OMP_NUM_THREADS=4\nexport OPENBLAS_NUM_THREADS=4\n"
                          "export MKL_NUM_THREADS=4\nexport PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                          f"{command} --task-index \"$SLURM_ARRAY_TASK_ID\"\n")
        subprocess.run(["bash", "-n", str(script)], check=True)
    print(json.dumps({"tasks": len(tasks), "plan": str(plan_path), "slurm_dir": str(slurm), "submitted": False}))


def worker(args):
    root = args.run_dir.resolve()
    plan = read(root / "evaluation_tasks.json")
    task = plan["tasks"][args.task_index]
    if task["index"] != args.task_index:
        raise ValueError("task index mismatch")
    if digest(task["input_csv"]) != task["dataset_sha256"]:
        raise ValueError("input dataset changed")
    for name, expected in plan["shared_adapter_sha256"].items():
        if digest(Path(plan["extractor_dir"]) / name) != expected:
            raise ValueError("frozen adapter changed")
    code = Path(__file__).resolve().parent
    if args.mode == "embedding":
        command = [sys.executable, str(code / "extract_embeddings_h5.py"), "--input-csv", task["input_csv"],
                   "--model-path", task["model_path"], "--model-name", task["model_name"], "--output-h5", task["output_h5"],
                   "--batch-size", str(task["batch_size"]), "--max-length", str(task["max_length"]),
                   "--pooling", "mean", "--extractor-dir", plan["extractor_dir"],
                   "--dataset-manifest", plan["dataset_manifest"], "--run-id", plan["run_id"]]
    else:
        command = [sys.executable, str(code / "reduce_embeddings.py"), "--input-h5", task["output_h5"],
                   "--output-dir", str(root / "reductions" / task["dataset_id"] / task["model_name"]),
                   "--seeds", "42", "43", "44", "--perplexity", "100", "--tsne-iterations", "5000"]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="write execution plans and Slurm files, without submitting")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("worker")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["embedding", "reduction"], required=True)
    p.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else worker(args)


if __name__ == "__main__":
    main()
