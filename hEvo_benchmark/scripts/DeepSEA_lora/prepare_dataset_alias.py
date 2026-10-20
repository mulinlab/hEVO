from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import os
from pathlib import Path

import numpy as np

from data import count_csv_rows, validate_dataset


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expose existing HPC DeepSEA data under <source-project-root>/datasets"
    )
    parser.add_argument("--source", default="<hEvo-data-root>/dataset/deepsea")
    parser.add_argument(
        "--target",
        default=str(
            Path(
                os.environ.get(
                    "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
                )
            )
            / "datasets/DeepSEA"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(
            Path(
                os.environ.get(
                    "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
                )
            )
            / "datasets/DeepSEA_manifest.json"
        ),
    )
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash all data files; this reads about 51 GB",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target)
    manifest_path = Path(args.manifest)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source:
            raise FileExistsError(
                f"Target already exists and is not the requested alias: {target}"
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)

    info = validate_dataset(target)
    files = {}
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        record = {
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        if args.sha256:
            started = time.time()
            record["sha256"] = sha256_file(path)
            record["sha256_seconds"] = round(time.time() - started, 3)
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r")
            record["shape"] = list(array.shape)
            record["dtype"] = str(array.dtype)
        elif path.name.endswith(".csv"):
            record["rows"] = count_csv_rows(path)
        files[path.name] = record
    manifest = {
        "source": str(source),
        "target": str(target),
        "target_is_symlink": target.is_symlink(),
        "platform": platform.platform(),
        "created_unix": time.time(),
        "dataset_validation": info,
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
