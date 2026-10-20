#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import default_config_path


DEFAULT_OLD_ROOT = "/path/to/old/benchmark/root"
NEW_ROOT_VAR = "${BENCHMARK_ROOT}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Programmatically replace hard-coded benchmark root paths in config.yaml "
            "with ${BENCHMARK_ROOT}."
        )
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=default_config_path(__file__),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--old-root",
        default=DEFAULT_OLD_ROOT,
        help="Hard-coded benchmark root path to replace.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write changes back to the file.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix used when writing in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config_path.resolve()
    original_text = config_path.read_text(encoding="utf-8")
    updated_text = original_text.replace(args.old_root, NEW_ROOT_VAR)
    replacement_count = original_text.count(args.old_root)

    if args.in_place and updated_text != original_text:
        backup_path = config_path.with_name(config_path.name + args.backup_suffix)
        backup_path.write_text(original_text, encoding="utf-8")
        config_path.write_text(updated_text, encoding="utf-8")

    print(f"config_path={config_path}")
    print(f"replacement_count={replacement_count}")
    print(f"changed={'yes' if updated_text != original_text else 'no'}")


if __name__ == "__main__":
    main()
