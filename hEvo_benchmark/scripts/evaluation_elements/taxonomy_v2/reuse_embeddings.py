"""Subset existing embeddings by seqID for taxonomy-v2 datasets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PROTOCOL = "complementary_elements_hg38_512_core128_taxonomy_v2"
TASKS = {
    "encode_ccre_v4_regulatory": (5000, 5),
    "gencode_v45_gene_structure": (3000, 3),
}
def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def row_hash(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(f"{row['seqID']}\t{row['label']}\t{row['seq']}\n".encode("utf-8"))
    return h.hexdigest()


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def source_embedding(source_dir, extension_dir, task, model):
    primary = source_dir / task / f"{model}.h5"
    if primary.is_file():
        return primary
    if extension_dir is not None:
        return extension_dir / task / f"{model}.h5"
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-dir", type=Path, required=True,
                        help="Directory with the ENCODE and GENCODE source-taxonomy CSVs.")
    parser.add_argument("--taxonomy-v2-dir", type=Path, required=True,
                        help="Directory with the two taxonomy-v2 CSVs.")
    parser.add_argument("--source-embeddings-dir", type=Path, required=True,
                        help="Embedding directory containing <dataset-id>/<model>.h5.")
    parser.add_argument("--extension-embeddings-dir", type=Path,
                        help="Optional second embedding directory for additional models.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Fresh external run directory for derived embeddings and task metadata.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", action="append", dest="models",
                        help="Limit to a named model; repeat the option to select more than one.")
    args = parser.parse_args()
    source_dataset_dir = args.source_dataset_dir.expanduser().resolve()
    taxonomy_v2_dir = args.taxonomy_v2_dir.expanduser().resolve()
    source_embeddings_dir = args.source_embeddings_dir.expanduser().resolve()
    extension_embeddings_dir = (
        args.extension_embeddings_dir.expanduser().resolve()
        if args.extension_embeddings_dir else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered = set()
    for embedding_root in (source_embeddings_dir, extension_embeddings_dir):
        if embedding_root is None:
            continue
        for task in TASKS:
            discovered.update(path.stem for path in (embedding_root / task).glob("*.h5"))
    models = sorted(set(args.models) if args.models else discovered)
    if not models:
        raise FileNotFoundError("No model H5 files found; check the supplied embedding directories.")
    task_manifest = {}
    task_rows = {}
    for task, (expected_rows, expected_classes) in TASKS.items():
        optimized_path = taxonomy_v2_dir / f"{task}.csv"
        source_path = source_dataset_dir / f"{task}.csv"
        optimized = load_rows(optimized_path)
        source = load_rows(source_path)
        source_by_id = {r["seqID"]: r for r in source}
        if len(source_by_id) != len(source):
            raise ValueError(f"Duplicate seqID in source dataset {task}")
        if len(optimized) != expected_rows or len({r["label"] for r in optimized}) != expected_classes:
            raise ValueError(f"Unexpected optimized dataset dimensions for {task}")
        for row in optimized:
            original = source_by_id.get(row["seqID"])
            if original is None or original["seq"] != row["seq"] or original["label"] != row["source_subtype"]:
                raise ValueError(f"Optimized row cannot be matched exactly to source: {task}/{row['seqID']}")
        counts = {}
        for row in optimized:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        if set(counts.values()) != {1000}:
            raise ValueError(f"Optimized classes are not balanced for {task}: {counts}")
        task_rows[task] = optimized
        task_manifest[task] = {
            "sha256": sha256(optimized_path), "rows": len(optimized),
            "class_counts": dict(sorted(counts.items())), "source_sha256": sha256(source_path),
        }

    dataset_gate = {
        "status": "PASS", "run_id": args.run_id, "protocol_id": PROTOCOL,
        "assembly": "hg38", "chromosomes": "chr1-22", "window_bp": 512,
        "core_bp": 128, "seed": 42, "samples_per_class": 1000,
        "tasks": {task: sorted(info["class_counts"]) for task, info in task_manifest.items()},
        "class_counts": {task: info["class_counts"] for task, info in task_manifest.items()},
        "dataset_sha256": {task: info["sha256"] for task, info in task_manifest.items()},
        "source_dataset_sha256": {task: info["source_sha256"] for task, info in task_manifest.items()},
    }
    (output_dir / "dataset_gate.json").write_text(json.dumps(dataset_gate, indent=2) + "\n", encoding="utf-8")
    (output_dir / "embeddings").mkdir(exist_ok=True)
    derived = []
    for task, rows in task_rows.items():
        input_hash = task_manifest[task]["sha256"]
        original_by_id = {r["seqID"]: r for r in load_rows(source_dataset_dir / f"{task}.csv")}
        for model in models:
            source_h5 = source_embedding(source_embeddings_dir, extension_embeddings_dir, task, model)
            if not source_h5.is_file():
                raise FileNotFoundError(source_h5)
            target_dir = output_dir / "embeddings" / task
            target_dir.mkdir(parents=True, exist_ok=True)
            target_h5 = target_dir / f"{model}.h5"
            expected_ids = [row["seqID"] for row in rows]
            with h5py.File(source_h5, "r") as src:
                source_ids = [text(x) for x in src["seqID"][:]]
                source_labels = [text(x) for x in src["label"][:]]
                if len(set(source_ids)) != len(source_ids):
                    raise ValueError(f"Duplicate seqID in embedding: {source_h5}")
                index = {seq_id: i for i, seq_id in enumerate(source_ids)}
                missing = [seq_id for seq_id in expected_ids if seq_id not in index]
                if missing:
                    raise ValueError(f"Embedding misses {len(missing)} selected seqIDs: {source_h5}")
                for row in rows:
                    i = index[row["seqID"]]
                    if source_labels[i] != row["source_subtype"] or original_by_id[row["seqID"]]["seq"] != row["seq"]:
                        raise ValueError(f"Embedding/source row mismatch: {task}/{model}/{row['seqID']}")
                desired_indices = np.asarray([index[x] for x in expected_ids], dtype=np.int64)
                order = np.argsort(desired_indices)
                sorted_embeddings = src["embeddings"][desired_indices[order]]
                embeddings = sorted_embeddings[np.argsort(order)]
                attrs = dict(src.attrs)
            labels = np.asarray([r["label"] for r in rows], dtype=object)
            seq_ids = np.asarray(expected_ids, dtype=object)
            expected_row_hash = row_hash(rows)
            if not np.isfinite(embeddings).all():
                raise ValueError(f"Non-finite embedding values: {source_h5}")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            with h5py.File(target_h5, "w") as dst:
                dst.create_dataset("embeddings", data=embeddings, dtype="float32")
                dst.create_dataset("label", data=labels, dtype=string_dtype)
                dst.create_dataset("seqID", data=seq_ids, dtype=string_dtype)
                for key, value in attrs.items():
                    dst.attrs[key] = value
                dst.attrs["run_id"] = args.run_id
                dst.attrs["protocol_id"] = PROTOCOL
                dst.attrs["dataset_sha256"] = input_hash
                dst.attrs["input_sha256"] = input_hash
                dst.attrs["row_order_sha256"] = expected_row_hash
                dst.attrs["n_rows"] = len(rows)
                dst.attrs["n_classes"] = len(task_manifest[task]["class_counts"])
                dst.attrs["taxonomy_parent_labels"] = True
                dst.attrs["source_embedding_h5_sha256"] = sha256(source_h5)
            manifest = {
                "model_name": model, "dataset_id": task, "run_id": args.run_id,
                "protocol_id": PROTOCOL, "source_embedding_h5": str(source_h5),
                "source_embedding_h5_sha256": sha256(source_h5),
                "derived_embedding_h5": str(target_h5), "derived_embedding_h5_sha256": sha256(target_h5),
                "source_embedding_rows": len(source_ids), "selected_rows": len(rows),
                "dataset_sha256": input_hash, "input_sha256": input_hash,
                "row_order_sha256": expected_row_hash,
                "model_weight_sha256": text(attrs["model_weight_sha256"]),
                "embedding_reused_by_seqID": True, "model_inference_rerun": False,
            }
            target_h5.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            derived.append(manifest)

    tasks = []
    for task in TASKS:
        for model in models:
            tasks.append({
                "index": len(tasks), "dataset_id": task, "model_name": model,
                "input_h5": str(output_dir / "embeddings" / task / f"{model}.h5"),
                "output_dir": str(output_dir / "reductions" / task / model),
                "dataset_sha256": task_manifest[task]["sha256"],
            })
    (output_dir / "reduction_tasks.json").write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS", "run_id": args.run_id, "protocol_id": PROTOCOL,
        "model_count": len(models), "embedding_units": len(derived),
        "dataset_manifest": str(output_dir / "dataset_gate.json"),
        "datasets": task_manifest,
        "reused_embedding_units": len(derived), "reduction_units": len(tasks),
        "source_embedding_manifests": derived,
    }
    (output_dir / "reuse_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "models": models, "embedding_units": len(derived),
                      "reduction_units": len(tasks), "dataset_rows": {k: v["rows"] for k, v in task_manifest.items()}}, indent=2))


if __name__ == "__main__":
    main()
