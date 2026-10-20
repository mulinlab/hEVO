"""Dispatch one model-by-dataset taxonomy-v2 reduction task."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True,
                        help="Runtime reduction task list generated in the external run directory.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol-id", default="complementary_elements_hg38_512_core128_taxonomy_v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    tasks = json.loads(args.tasks_json.expanduser().resolve().read_text())
    if not 0 <= args.task_index < len(tasks):
        raise IndexError(f"task index {args.task_index} outside task list of {len(tasks)}")
    task = tasks[args.task_index]
    command = [
        sys.executable, str(root / "reduce_taxonomy.py"),
        "--input-h5", task["input_h5"], "--output-dir", task["output_dir"],
        "--dataset-id", task["dataset_id"], "--run-id", args.run_id,
        "--dataset-sha256", task["dataset_sha256"],
        "--protocol-id", args.protocol_id,
        "--seeds", "42", "43", "44", "--perplexity", "100", "--tsne-iterations", "5000",
    ]
    print(json.dumps({"task": task, "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
