#!/usr/bin/env python3
"""Freeze UCSC dataset/model manifests from the passed data and UMAP gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

DATASET_ID = "ucsc_refseq_curated_gene_structure"
RUN_ID = "ucsc_refseq_curated_v1_20260924"
EXPECTED_CLASSES = ("CDS", "UTR5", "UTR3", "protein_coding_intron", "noncoding_exon")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--dataset-qa", type=Path, required=True)
    parser.add_argument("--umap-manifest", type=Path, required=True)
    parser.add_argument("--base-evaluation-tasks", type=Path, required=True)
    parser.add_argument("--open-large-source-manifest", type=Path, required=True)
    parser.add_argument("--open-large-source-h5", type=Path, required=True)
    parser.add_argument("--shared-extractor-dir", type=Path, required=True)
    parser.add_argument("--output-roster", type=Path, required=True)
    parser.add_argument("--output-dataset-manifest", type=Path, required=True)
    args = parser.parse_args()

    qa = json.loads(args.dataset_qa.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or qa.get("selected_rows") != 5000:
        raise ValueError("dataset QA must be PASS for exactly 5,000 rows")
    if qa.get("selected_class_counts") != {label: 1000 for label in EXPECTED_CLASSES}:
        raise ValueError(f"unexpected dataset class counts: {qa.get('selected_class_counts')}")
    dataset_hash = sha256_file(args.dataset_csv)
    if dataset_hash != qa.get("selected_dataset_sha256"):
        raise ValueError("dataset CSV hash does not match the builder QA report")

    umap = json.loads(args.umap_manifest.read_text(encoding="utf-8"))
    models = umap.get("models", [])
    if umap.get("status") != "PASS" or len(models) != 30 or len(set(models)) != 30:
        raise ValueError("the frozen all-30 UMAP manifest is not PASS or does not contain 30 unique models")

    task_payload = json.loads(args.base_evaluation_tasks.read_text(encoding="utf-8"))
    base_tasks = {
        task["model_name"]: task
        for task in task_payload["tasks"]
        if task.get("dataset_id") == "encode_ccre_v4_regulatory"
    }
    open_manifest = json.loads(args.open_large_source_manifest.read_text(encoding="utf-8"))
    open_large_name = "hEvo-open-large"
    if open_large_name in base_tasks or open_large_name not in models:
        raise ValueError("open-large roster placement is inconsistent")
    if set(models) != set(base_tasks) | {open_large_name}:
        raise ValueError("the 30-model roster differs from the 29-model base task manifest plus open-large")

    shared_adapter = args.shared_extractor_dir / "extract_embeddings_classes.py"
    shared_mapping = args.shared_extractor_dir / "probe_single_seq_input.py"
    if not shared_adapter.is_file() or not shared_mapping.is_file():
        raise FileNotFoundError("shared model extractor adapter/mapping is incomplete")
    adapter_hash, mapping_hash = sha256_file(shared_adapter), sha256_file(shared_mapping)

    root = args.run_root.resolve()
    source_run = Path(task_payload["tasks"][0]["input_csv"]).parents[1]
    model_specs = {}
    groups = {"48": [], "64": [], "80": []}
    for name in models:
        if name == open_large_name:
            source_manifest_path = args.open_large_source_manifest
            source_h5 = args.open_large_source_h5
            profile = {
                "model_name": name,
                "model_path": open_manifest["model_path"],
                "batch_size": 32,
                "max_length": 1024,
            }
        else:
            task = base_tasks[name]
            profile = {
                "model_name": name,
                "model_path": task["model_path"],
                "batch_size": int(task["batch_size"]),
                "max_length": int(task["max_length"]),
            }
            source_manifest_path = source_run / "embeddings" / "encode_ccre_v4_regulatory" / f"{name}.manifest.json"
            source_h5 = source_run / "embeddings" / "encode_ccre_v4_regulatory" / f"{name}.h5"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not Path(profile["model_path"]).exists():
            raise FileNotFoundError(f"model path does not exist: {profile['model_path']}")
        if not source_h5.is_file():
            raise FileNotFoundError(f"source embedding evidence is missing: {source_h5}")
        if source_manifest.get("model_name") != name:
            raise ValueError(f"source embedding manifest model mismatch: {name}")
        if int(source_manifest.get("max_length", -1)) != profile["max_length"]:
            raise ValueError(f"model max_length changed relative to frozen run: {name}")
        if source_manifest.get("pooling") != "mean" or source_manifest.get("truncation") is not False:
            raise ValueError(f"source extraction contract is not mean/untruncated for {name}")
        if source_manifest.get("shared_adapter_sha256") != adapter_hash or source_manifest.get("shared_mapping_sha256") != mapping_hash:
            raise ValueError(f"shared extractor fingerprints changed for {name}")

        memory = 80 if name in {"evo2_7b_base", "GENERator-3B"} else 64 if ("2.5b" in name or name in {"evo2_1b_base", "GENERator-1.2B"}) else 48
        spec = {
            **profile,
            "memory_gb": memory,
            "model_weight_sha256_expected": source_manifest["model_weight_sha256"],
            "source_inference_precision": source_manifest.get("inference_precision", "model_default"),
            "source_evo2_config_sha256": source_manifest.get("evo2_config_sha256", ""),
            "source_embedding_manifest": str(source_manifest_path),
            "source_embedding_manifest_sha256": sha256_file(source_manifest_path),
            "source_embedding_h5": str(source_h5),
            "source_embedding_h5_sha256": sha256_file(source_h5),
            "cache_reused": False,
            "cache_reuse_reason": "Fresh extraction for the frozen UCSC dataset; the run-specific extractor fingerprint is distinct, so no prior embedding rows are reused.",
        }
        model_specs[name] = spec
        groups[str(memory)].append(name)

    dataset_manifest = {
        "status": "PASS",
        "run_id": RUN_ID,
        "protocol_id": qa["protocol_id"],
        "dataset_id": DATASET_ID,
        "dataset_csv": str(args.dataset_csv.resolve()),
        "dataset_sha256": {DATASET_ID: dataset_hash},
        "dataset_sha256_value": dataset_hash,
        "rows": 5000,
        "class_count": 5,
        "tasks": {DATASET_ID: list(EXPECTED_CLASSES)},
        "class_counts": qa["selected_class_counts"],
        "input_annotation_sha256": qa["annotation_sha256"],
        "reference_fasta_sha256": qa["fasta_sha256"],
    }
    roster = {
        "status": "PASS",
        "run_id": RUN_ID,
        "dataset_id": DATASET_ID,
        "dataset_csv": str(args.dataset_csv.resolve()),
        "dataset_sha256": dataset_hash,
        "protocol_id": qa["protocol_id"],
        "models": models,
        "model_specs": model_specs,
        "gpu_memory_groups_gb": groups,
        "model_roster_source": str(args.umap_manifest.resolve()),
        "model_roster_source_sha256": sha256_file(args.umap_manifest),
        "shared_adapter_sha256": adapter_hash,
        "shared_mapping_sha256": mapping_hash,
        "source_model_count": len(model_specs),
        "embedding_reuse_count": 0,
        "reuse_note": "47 coordinate-identical windows exist against the prior GENCODE subset; embeddings are intentionally recomputed because the new run uses a separate extractor-file fingerprint and the reused fraction is below 1%.",
        "candidate_overlap_summary": qa.get("cross_dataset_overlaps", {}),
    }
    for output_path, payload in ((args.output_roster, roster), (args.output_dataset_manifest, dataset_manifest)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "models": len(models), "memory_groups": {k: len(v) for k, v in groups.items()}, "dataset_sha256": dataset_hash}, indent=2))


if __name__ == "__main__":
    main()
