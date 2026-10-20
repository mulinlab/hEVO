#!/usr/bin/env python3
"""Read-only syntax, coverage, documentation, Ruff and optional CLI audit.

Run with Python >=3.10. Reports go to stdout; no benchmark outputs are written.
CLI smoke checks require the workflow dependencies in --python's environment.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXCLUDED = {"archive", "vendor", "__pycache__", ".ruff_cache", ".pytest_cache"}


def maintained_files(root=ROOT):
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not EXCLUDED.intersection(p.relative_to(root).parts)
    )


def cli_files(files):
    return [
        p
        for p in files
        if p.suffix == ".py"
        and not p.name.startswith("test_")
        and p.name != "check_scripts.py"
        and "argparse" in p.read_text()
        and "__main__" in p.read_text()
    ]


def validate_coverage(rows, repo=REPO):
    errors = []
    ids = set()
    for row in rows:
        if row["benchmark"] in ids:
            errors.append(f"Duplicate benchmark: {row['benchmark']}")
        ids.add(row["benchmark"])
        for field in ("inputs", "scoring", "scheduler", "summary", "readme"):
            for value in row[field].split(";"):
                if value.startswith("hEvo_benchmark/") and not (repo / value).exists():
                    errors.append(f"{row['benchmark']}: missing {field} path {value}")
        for field in ("protocol", "outputs", "dependencies", "status"):
            if not row[field].strip():
                errors.append(f"{row['benchmark']}: empty {field}")
    tasks = set(
        re.findall(
            r"^\s*- supername:\s*(\S+)",
            (repo / "hEvo_benchmark/configs/tasks.yaml").read_text(),
            re.M,
        )
    )
    covered = {g for row in rows for g in row["task_groups"].split(";") if g}
    errors.extend(f"Unmapped task group: {g}" for g in sorted(tasks - covered))
    mapped_tables = {v for row in rows for v in row["outputs"].split(";")}
    errors.extend(
        f"Unmapped source table: S{i}"
        for i in range(1, 50)
        if f"S{i}" not in mapped_tables
    )
    return errors


def documentation_errors(files):
    errors = []
    index = (ROOT / "README.md").read_text()
    for directory in sorted(
        {
            p.relative_to(ROOT).parts[0]
            for p in files
            if len(p.relative_to(ROOT).parts) > 1
        }
    ):
        if not (ROOT / directory / "README.md").is_file():
            errors.append(f"Missing workflow README: {directory}")
        if f"{directory}/README.md" not in index:
            errors.append(f"Missing workflow index entry: {directory}")
    for p in files:
        if p.suffix != ".md":
            continue
        # Ignore code fences, remote links, anchors and explicit placeholders.
        text = re.sub(r"```.*?```", "", p.read_text(), flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = target.split("#", 1)[0].split(' "', 1)[0]
            if not target or "://" in target or "<" in target:
                continue
            if not (p.parent / target).exists():
                errors.append(f"{p.relative_to(ROOT)}: broken link {target}")
    return errors


def documented_command_errors(files):
    """Validate literal flags in runnable Python examples without executing them."""
    errors = []
    for document in files:
        if document.suffix != ".md":
            continue
        for block in re.findall(
            r"```(?:bash|sh)\n(.*?)```", document.read_text(), re.S
        ):
            for line in block.replace("\\\n", " ").splitlines():
                try:
                    tokens = shlex.split(line)
                except ValueError:
                    continue  # Explicit placeholder/shell syntax, not a literal invocation.
                for index, token in enumerate(tokens):
                    if not token.endswith(".py") or "<" in token or "$" in token:
                        continue
                    candidates = [
                        REPO / token,
                        ROOT.parent / token,
                        document.parent / token,
                    ]
                    script = next((p for p in candidates if p.is_file()), None)
                    if script is None:
                        continue  # Examples referring to external runtime tools are documented separately.
                    source = script.read_text()
                    if (
                        "--input-manifest" in tokens
                        and script.name == "export_clinvar_all_subsets.py"
                    ):
                        source += (script.parent / "rebuild_gpn_subsets.py").read_text()
                    if script.name == "merge_score_shards.py":
                        source += (
                            ROOT / "evaluate/runtime/gpn_star_merge.py"
                        ).read_text()
                    if script.name == "gpn_star_merge.py":
                        source += (
                            ROOT / "evaluate/runtime/gpn_star_merge.py"
                        ).read_text()
                    if script.name == "gpn_star_split.py":
                        source += (
                            ROOT / "evaluate/runtime/gpn_star_split.py"
                        ).read_text()
                    if script.name == "aggregate_native_clinvar_shards.py":
                        source += (
                            ROOT
                            / "evaluate/results_export/aggregate_native_clinvar_shards.py"
                        ).read_text()
                    tree = ast.parse(source)
                    options = {"--help", "-h"}
                    for node in ast.walk(tree):
                        if (
                            isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "add_argument"
                        ):
                            options.update(
                                arg.value
                                for arg in node.args
                                if isinstance(arg, ast.Constant)
                                and isinstance(arg.value, str)
                                and arg.value.startswith("-")
                            )
                    for flag in tokens[index + 1 :]:
                        flag = flag.split("=", 1)[0]
                        if flag.startswith("--") and flag not in options:
                            errors.append(
                                f"{document.relative_to(ROOT)}: {script.name} has no option {flag}"
                            )
                    break
    return errors


def smoke_help(path, python, timeout, repo=REPO):
    records = []
    with tempfile.TemporaryDirectory(prefix="hevo-cli-help-") as isolated:
        for cwd in (repo, Path(isolated)):
            try:
                result = subprocess.run(
                    [python, str(path), "--help"],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                passed = result.returncode == 0 and "usage:" in result.stdout.lower()
                records.append(
                    {
                        "cwd": "repo" if cwd == repo else "external",
                        "passed": passed,
                        "returncode": result.returncode,
                        "stderr": result.stderr[-1500:],
                    }
                )
            except subprocess.TimeoutExpired:
                records.append(
                    {
                        "cwd": "repo" if cwd == repo else "external",
                        "passed": False,
                        "returncode": 124,
                        "stderr": "timeout",
                    }
                )
        if any(Path(isolated).iterdir()):
            records.append(
                {
                    "cwd": "external",
                    "passed": False,
                    "returncode": 1,
                    "stderr": "--help created files in the working directory",
                }
            )
    return {"path": str(path.relative_to(ROOT)), "checks": records}


def run_checks(args):
    files = maintained_files()
    errors = []
    python_files = [p for p in files if p.suffix == ".py"]
    shell_files = [p for p in files if p.suffix in (".sh", ".sbatch")]
    for p in python_files:
        try:
            source = p.read_text()
            ast.parse(source, filename=str(p))
            compile(source, str(p), "exec")  # No import, execution or .pyc writes.
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Syntax: {p.relative_to(ROOT)}: {exc}")
    for p in shell_files:
        result = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        if result.returncode:
            errors.append(f"Shell: {p.relative_to(ROOT)}: {result.stderr}")
    native_checks = []
    if args.native_syntax:
        for p in files:
            command = None
            if p.suffix == ".R":
                command = [
                    "Rscript",
                    "-e",
                    "invisible(parse(file=commandArgs(TRUE)[1]))",
                    str(p),
                ]
            elif p.suffix == ".cpp":
                command = [args.cxx, "-std=c++17", "-fopenmp", "-fsyntax-only"]
                for include in args.cpp_include:
                    command += ["-I", include]
                command += [str(p)]
            if command:
                try:
                    result = subprocess.run(
                        command, capture_output=True, text=True, timeout=args.timeout
                    )
                    native_checks.append(
                        {
                            "path": str(p.relative_to(ROOT)),
                            "returncode": result.returncode,
                            "output": result.stderr[-1500:],
                        }
                    )
                    if result.returncode:
                        errors.append(f"Native syntax: {p.relative_to(ROOT)}")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(f"Native syntax: {p.relative_to(ROOT)}: {exc}")
    with (ROOT / "benchmark_coverage.tsv").open() as handle:
        coverage = list(csv.DictReader(handle, delimiter="\t"))
    errors.extend(validate_coverage(coverage))
    errors.extend(documentation_errors(files))
    errors.extend(documented_command_errors(files))
    ruff = args.ruff or shutil.which("ruff")
    ruff_checks = []
    if not ruff:
        errors.append(
            "Ruff unavailable: install the pinned tools-requirements.txt or pass --ruff"
        )
    else:
        for command in (
            [ruff, "check", str(ROOT)],
            [ruff, "format", "--check", str(ROOT)],
        ):
            result = subprocess.run(command, capture_output=True, text=True)
            ruff_checks.append(
                {
                    "command": command[1:3],
                    "returncode": result.returncode,
                    "output": result.stdout + result.stderr,
                }
            )
            if result.returncode:
                errors.append(f"Ruff failed: {' '.join(command)}")
    hashes = {}
    for p in python_files:
        hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(
            str(p.relative_to(ROOT))
        )
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    if duplicates:
        errors.append("Exact duplicate maintained Python files found")
    help_results = []
    if args.cli_help:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            help_results = list(
                pool.map(
                    lambda p: smoke_help(p, args.python, args.timeout), cli_files(files)
                )
            )
        errors.extend(
            f"CLI: {row['path']}"
            for row in help_results
            if not all(c["passed"] for c in row["checks"])
        )
    ignored_counts = {}
    for name in sorted(EXCLUDED):
        ignored_counts[name] = sum(
            1
            for p in ROOT.rglob("*")
            if p.is_file() and name in p.relative_to(ROOT).parts
        )
    return {
        "status": "PASS" if not errors else "FAIL",
        "python_files": len(python_files),
        "shell_files": len(shell_files),
        "benchmark_rows": len(coverage),
        "syntax_check": "AST + compile (no imports)",
        "errors": errors,
        "exact_duplicates": duplicates,
        "excluded_local_files": ignored_counts,
        "ruff": ruff_checks,
        "native_syntax": native_checks,
        "cli_help": help_results,
        "runtime_limit": "Help/synthetic checks do not validate full GPU experiments or external artifacts.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ruff", help="Ruff executable (tools-requirements.txt pins its version)"
    )
    parser.add_argument(
        "--cli-help",
        action="store_true",
        help="Execute formal Python CLIs with --help from two working directories",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python environment with workflow dependencies",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--native-syntax",
        action="store_true",
        help="Also parse R and check C++ syntax; requires Rscript, C++17 and development headers",
    )
    parser.add_argument(
        "--cxx", default="c++", help="C++17 compiler executable for --native-syntax"
    )
    parser.add_argument(
        "--cpp-include",
        action="append",
        default=[],
        help="Development-header directory; repeat if needed",
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    report = run_checks(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
