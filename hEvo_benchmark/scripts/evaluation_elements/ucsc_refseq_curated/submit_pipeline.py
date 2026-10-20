#!/usr/bin/env python3
"""Generate and submit the gated UCSC Slurm pipeline without waiting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def submit(script: Path, dependency: list[str] | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency:
        command += ["--dependency=afterok:" + ":".join(dependency)]
    command.append(str(script))
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.strip().split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--submit", action="store_true", help="Submit jobs; without this, only generate and validate Slurm scripts")
    parser.add_argument("--allow-resubmit", action="store_true", help="Permit submitting a second independent set of jobs")
    args = parser.parse_args()
    root, repo = args.run_root.resolve(), args.repo_root.resolve()
    roster_path = root / "config" / "model_roster.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    dataset_validation = json.loads((root / "qa" / "dataset_validation.json").read_text(encoding="utf-8"))
    builder_qa = json.loads((root / "qa" / "dataset_qa.json").read_text(encoding="utf-8"))
    if roster.get("status") != "PASS" or builder_qa.get("status") != "PASS" or dataset_validation.get("status") != "PASS":
        raise ValueError("dataset, validation, or 30-model roster gate failed")
    if len(roster.get("models", [])) != 30 or len(set(roster["models"])) != 30:
        raise ValueError("expected the frozen 30-model roster")
    record_path = root / "qa" / "submission_manifest.json"
    if record_path.exists() and not args.allow_resubmit:
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if prior.get("status") in {"SUBMITTING", "SUBMITTED"} or prior.get("jobs"):
            raise FileExistsError("a prior submission is recorded; inspect current jobs before using --allow-resubmit")

    code = Path(__file__).resolve().parent
    shared_extractor_dir = code.parent / "evaluation" / "shared"
    slurm_dir, log_dir = root / "slurm", root / "logs"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}
    gpu_groups = roster["gpu_memory_groups_gb"]
    for memory, models in gpu_groups.items():
        if not models:
            continue
        script = slurm_dir / f"embedding_mem{memory}.sbatch"
        count = len(models)
        text = f'''#!/usr/bin/env bash
#SBATCH --job-name=ucsc-embed-{memory}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem={memory}G
#SBATCH --array=0-{count - 1}%{count}
#SBATCH --output={log_dir}/embedding.%A_%a.out
#SBATCH --error={log_dir}/embedding.%A_%a.err
#SBATCH --chdir={repo}
set -euo pipefail
export LC_ALL=C
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MODEL_MEMORY_GB={memory}
micromamba -n llmbm run python {code}/run_embedding_task.py \\
  --run-root {root} \\
  --shared-extractor-dir {shared_extractor_dir}
'''
        write_executable(script, text)
        generated[f"embedding_mem{memory}"] = script

    pc_script = slurm_dir / "reduce_pcatsne.sbatch"
    write_executable(pc_script, f'''#!/usr/bin/env bash
#SBATCH --job-name=ucsc-pcatsne
#SBATCH --partition=cpuPartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-29%16
#SBATCH --output={log_dir}/pcatsne.%A_%a.out
#SBATCH --error={log_dir}/pcatsne.%A_%a.err
#SBATCH --chdir={repo}
set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
micromamba -n llmbm run python {code}/reduce_pcatsne.py --run-root {root}
''')
    generated["reduce_pcatsne"] = pc_script

    umap_script = slurm_dir / "reduce_umap.sbatch"
    write_executable(umap_script, f'''#!/usr/bin/env bash
#SBATCH --job-name=ucsc-umap
#SBATCH --partition=cpuPartition
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-29%20
#SBATCH --output={log_dir}/umap.%A_%a.out
#SBATCH --error={log_dir}/umap.%A_%a.err
#SBATCH --chdir={repo}
set -euo pipefail
export LC_ALL=C
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
micromamba -n llmbm run python {code}/reduce_umap.py --run-root {root}
''')
    generated["reduce_umap"] = umap_script

    aggregate_script = slurm_dir / "aggregate.sbatch"
    write_executable(aggregate_script, f'''#!/usr/bin/env bash
#SBATCH --job-name=ucsc-ch-summary
#SBATCH --partition=cpuPartition
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output={log_dir}/aggregate.%j.out
#SBATCH --error={log_dir}/aggregate.%j.err
#SBATCH --chdir={repo}
set -euo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
micromamba -n llmbm run python {code}/aggregate_results.py --run-root {root}
''')
    generated["aggregate"] = aggregate_script

    manifest = {
        "status": "SCRIPTS_GENERATED",
        "run_id": roster["run_id"],
        "run_root": str(root),
        "dataset_sha256": roster["dataset_sha256"],
        "model_count": len(roster["models"]),
        "embedding_max_concurrency": sum(len(group) for group in gpu_groups.values()),
        "pcatsne_array_concurrency": 16,
        "umap_array_concurrency": 20,
        "scripts": {key: str(path) for key, path in generated.items()},
        "submitted_at_utc": None,
        "jobs": {},
    }
    if not args.submit:
        record_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "SCRIPTS_GENERATED", "groups": {key: len(value) for key, value in gpu_groups.items()}, "scripts": manifest["scripts"]}, indent=2))
        return

    manifest["status"] = "SUBMITTING"
    record_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gpu_job_ids = []
    for key in generated:
        if not key.startswith("embedding_mem"):
            continue
        job_id = submit(generated[key])
        gpu_job_ids.append(job_id)
        manifest["jobs"][key] = job_id
        manifest["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pc_job = submit(pc_script, gpu_job_ids)
    manifest["jobs"]["reduce_pcatsne"] = pc_job
    record_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    umap_job = submit(umap_script, gpu_job_ids)
    manifest["jobs"]["reduce_umap"] = umap_job
    record_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate_job = submit(aggregate_script, [pc_job, umap_job])
    manifest["jobs"]["aggregate"] = aggregate_job
    manifest["status"] = "SUBMITTED"
    manifest["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    record_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
