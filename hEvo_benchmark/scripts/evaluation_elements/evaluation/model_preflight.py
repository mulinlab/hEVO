#!/usr/bin/env python3
"""Resolve the frozen table model roster without substituting checkpoints."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from string import Template

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path-override", action="append", default=[], metavar="MODEL=PATH")
    args = parser.parse_args()
    benchmark = args.repo_root / "hEvo_benchmark"
    sources = sorted((benchmark / "tables").glob("S11_*ch_index_source.csv")) + sorted((benchmark / "tables").glob("S13_*ch_index_source.csv"))
    if len(sources) != 2:
        raise ValueError("expected exactly two current CH source tables")
    rosters = []
    for path in sources:
        with path.open() as handle:
            rosters.append({row["model_name"] for row in csv.DictReader(handle)})
    if any(roster != rosters[0] for roster in rosters) or len(rosters[0]) != 29:
        raise ValueError("current source tables do not share the same 29-model roster")
    mapping_file = benchmark / "scripts/evaluate/probe_single_seq_input.py"
    mapping = {}
    for node in ast.walk(ast.parse(mapping_file.read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "model_extractor_map" for t in node.targets):
            mapping = {ast.literal_eval(k): v.id for k, v in zip(node.value.keys, node.value.values)}
    config_path = benchmark / "configs/models.yaml"
    configured = {row["name"]: row for row in yaml.safe_load(config_path.read_text())["Models"]}
    overrides = dict(value.split("=", 1) for value in args.model_path_override)
    if not set(overrides).issubset(rosters[0]):
        raise ValueError("override names must belong to the frozen roster")
    entries, errors = [], []
    for name in sorted(rosters[0]):
        info = configured.get(name)
        if info is None:
            errors.append({"model": name, "reason": "not_configured"})
            continue
        configured_path = Template(info["url"]).substitute(MODELS_ROOT=str(args.models_root), BENCHMARK_ROOT=str(benchmark))
        path = Path(overrides.get(name, configured_path))
        files = [path] if path.is_file() else sorted({p for pattern in ("*.safetensors", "pytorch_model*.bin", "*.pt", "*.ckpt") for p in path.glob(pattern)}) if path.is_dir() else []
        issues = []
        if not path.exists():
            issues.append("configured_path_missing")
        if not files:
            issues.append("weights_missing")
        if name not in mapping:
            issues.append("extractor_missing")
        entry = {"model_name": name, "model_path": str(path), "extractor_class": mapping.get(name),
                 "configured_path": configured_path, "explicit_path_override": name in overrides,
                 "batch_size": 1 if name.startswith("evo2_") else min(int(info.get("batch_size", 32)), 32),
                 "max_length": int(info.get("max_length", 1024)),
                 "weight_files": [{"path": str(p), "bytes": p.stat().st_size} for p in files],
                 "status": "BLOCKED" if issues else "FILES_PRESENT_NOT_RUNTIME_VERIFIED", "issues": issues}
        entries.append(entry)
        errors.extend({"model": name, "reason": issue, "path": str(path)} for issue in issues)
    result = {"status": "BLOCKED_MODEL_INPUTS" if errors else "FILES_PRESENT", "models": entries, "model_count": len(entries), "errors": errors,
              "models_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "extractor_map_sha256": hashlib.sha256(mapping_file.read_bytes()).hexdigest(),
              "source_table_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "note": "No runtime or weight-hash validation is implied by file presence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "model_count": len(entries), "errors": errors}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
