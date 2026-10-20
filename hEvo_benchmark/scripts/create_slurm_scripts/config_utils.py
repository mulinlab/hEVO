from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def find_benchmark_root(start_path: str | Path | None = None) -> Path:
    start = Path(start_path or __file__).resolve()
    current = start if start.is_dir() else start.parent
    for candidate in [current, *current.parents]:
        if (candidate / "config.yaml").exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to locate benchmark root from starting path: {start}"
    )


def default_config_path(start_path: str | Path | None = None) -> Path:
    return find_benchmark_root(start_path) / "config.yaml"


def read_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).resolve()
    with yaml_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _base_variables(config_path: Path) -> dict[str, str]:
    benchmark_root = config_path.resolve().parent
    repo_root = benchmark_root.parent
    values = {
        "CONFIG_DIR": str(benchmark_root),
        "BENCHMARK_ROOT": str(benchmark_root),
        "REPO_ROOT": str(repo_root),
        "HOME": str(Path.home()),
    }
    for key, value in os.environ.items():
        values[key] = value
    return values


def _resolve_string(value: str, variables: Mapping[str, str]) -> str:
    resolved = value
    for _ in range(10):
        updated = VARIABLE_PATTERN.sub(
            lambda match: variables.get(match.group(1), match.group(0)),
            resolved,
        )
        if updated == resolved:
            return updated
        resolved = updated
    return resolved


def _resolve_object(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, variables)
    if isinstance(value, list):
        return [_resolve_object(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_object(item, variables) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_file_path(base_config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_config_path.parent / candidate).resolve()


def load_config(
    config_path: str | Path | None = None,
    extra_vars: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    path = Path(config_path or default_config_path()).resolve()
    raw_config = read_yaml(path)

    variables = _base_variables(path)
    combined_config: dict[str, Any] = {}

    if isinstance(raw_config.get("PathVars"), dict):
        path_vars = {
            str(key): _resolve_string(str(value), variables)
            for key, value in raw_config["PathVars"].items()
        }
        variables.update(path_vars)

    config_files = raw_config.get("ConfigFiles", {})
    manifest_content = {
        key: value for key, value in raw_config.items() if key not in {"ConfigFiles"}
    }
    combined_config = _deep_merge(combined_config, manifest_content)

    if isinstance(config_files, dict):
        for _, relative_path in config_files.items():
            resolved_path = _resolve_config_file_path(
                path, _resolve_string(str(relative_path), variables)
            )
            fragment = read_yaml(resolved_path)
            if isinstance(fragment.get("PathVars"), dict):
                fragment_path_vars = {
                    str(key): _resolve_string(str(value), variables)
                    for key, value in fragment["PathVars"].items()
                }
                variables.update(fragment_path_vars)
            combined_config = _deep_merge(combined_config, fragment)

    if extra_vars:
        variables.update({str(key): str(value) for key, value in extra_vars.items()})

    return _resolve_object(combined_config, variables)


def get_cli_defaults(config: Mapping[str, Any], tool_name: str) -> dict[str, Any]:
    cli_defaults = config.get("CliDefaults", {})
    if not isinstance(cli_defaults, dict):
        return {}
    tool_defaults = cli_defaults.get(tool_name, {})
    return tool_defaults if isinstance(tool_defaults, dict) else {}


def choose_path(
    cli_value: Path | None, config_value: str | None, fallback: Path
) -> Path:
    if cli_value is not None:
        return cli_value
    if config_value:
        return Path(config_value)
    return fallback


def choose_value(cli_value: str | None, config_value: str | None, fallback: str) -> str:
    return cli_value or config_value or fallback


def resolve_supername(
    requested: str | None,
    available_supernames: list[str],
) -> str:
    if requested and requested in available_supernames:
        return requested
    if requested and requested not in available_supernames:
        raise KeyError(
            f"Unknown supername: {requested}. Available: {', '.join(available_supernames)}"
        )
    if len(available_supernames) == 1:
        return available_supernames[0]
    raise ValueError(
        "No valid supername resolved. Provide --supername explicitly. "
        f"Available: {', '.join(available_supernames)}"
    )
