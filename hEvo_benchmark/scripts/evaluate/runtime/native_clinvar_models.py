#!/usr/bin/env python3
"""Model-specific zero-shot ClinVar evaluation for Mistral-DNA and GENERator.

Mistral-DNA follows the author's tutorial: run AutoModelForCausalLM on each
REF/ALT sequence, max-pool its per-position vocabulary logits, and rank by the
L2 distance between the two pooled vectors. GENERator follows its released
variant-effect recipe: use the upstream reference context, predict the next
token after a 6-mer-aligned context, marginalize token probabilities by their
first DNA base, and score log(P(ref) / P(alt)).

The input CSV is the benchmark's centered paired-sequence ClinVar file. For
GENERATOR, genomic coordinates are used to recover its native upstream context
from an indexed hg38 FASTA, capped at the benchmark-wide maximum of 1,024 bp
and aligned to a 6-mer boundary. For Mistral-DNA, the supplied centered 1,023 bp
sequence is used in full (within its 512-token context limit).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Support direct execution from this relocated runtime module.

_EVALUATE_ROOT = Path(__file__).resolve().parents[1]
if str(_EVALUATE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVALUATE_ROOT))

from adapters.native_variant_scorers import (  # noqa: E402
    generator_base_token_ids as _generator_base_token_ids,
    generator_score_batch as _generator_score_batch,
    mistral_score_batch as _mistral_score_batch,
)


DEFAULT_DATA = (
    Path(__file__).resolve().parents[3]
    / "datasets/newClinvar_260615/clinvar_all_1023bp.csv"
)
DEFAULT_HG38 = Path(
    os.environ.get(
        "HG38_FASTA",
        str(
            Path(
                os.environ.get(
                    "DNAMODELS_BENCHMARK_ROOT", Path.home() / "DNAModelsBenchmark"
                )
            )
            / "datasets/hg38.fa"
        ),
    )
)
DNA_BASES = frozenset("ACGT")
CSV_COLUMNS = ["SeqID", "refSeq", "altSeq", "label", "chrom", "pos"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_chromosome(fasta, chrom: object) -> str | None:
    chrom = str(chrom)
    aliases = [f"chr{chrom}", chrom]
    if chrom.upper() in {"M", "MT", "CHRM", "CHRMT"}:
        aliases.extend(["chrM", "chrMT", "MT", "M"])
    for name in aliases:
        if name in fasta.references:
            return name
    return None


def load_model(model_path: str, device: str, dtype_name: str, family: str):
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if family == "generator":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "GENERATOR tokenizer has no pad or EOS token for batching"
                )
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def mistral_score_batch(
    tokenizer,
    model,
    ref_seqs: list[str],
    alt_seqs: list[str],
    device: str,
    max_tokens: int,
) -> np.ndarray:
    return _mistral_score_batch(
        tokenizer, model, ref_seqs, alt_seqs, device, max_tokens
    )


def generator_base_token_ids(tokenizer) -> dict[str, list[int]]:
    return _generator_base_token_ids(
        tokenizer, missing_message="Tokenizer has no vocabulary tokens starting with:"
    )


def generator_score_batch(
    tokenizer,
    model,
    contexts: list[str],
    refs: list[str],
    alts: list[str],
    token_ids_by_base: dict[str, list[int]],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _generator_score_batch(
        tokenizer, model, contexts, refs, alts, token_ids_by_base, device
    )


def centered_snv(ref_seq: str, alt_seq: str) -> tuple[int, str, str] | None:
    if len(ref_seq) != len(alt_seq) or not ref_seq:
        return None
    differences = [
        i for i, (ref, alt) in enumerate(zip(ref_seq, alt_seq)) if ref != alt
    ]
    if len(differences) != 1 or differences[0] != len(ref_seq) // 2:
        return None
    position = differences[0]
    ref, alt = ref_seq[position].upper(), alt_seq[position].upper()
    if ref not in DNA_BASES or alt not in DNA_BASES:
        return None
    return position, ref, alt


def generator_context(fasta, chromosome: str, pos_1based: int, context_bp: int) -> str:
    position0 = int(pos_1based) - 1
    if position0 < 0:
        raise ValueError(f"Invalid one-based variant position: {pos_1based}")
    start = max(0, position0 - context_bp)
    sequence = fasta.fetch(chromosome, start, position0).upper()
    sequence = sequence.lstrip("N")
    remainder = len(sequence) % 6
    if remainder:
        sequence = sequence[remainder:]
    return sequence


def bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {}
    rng = np.random.RandomState(seed)
    n = len(labels)
    ap_values: list[float] = []
    auc_values: list[float] = []
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap metrics", leave=False):
        indices = rng.randint(0, n, size=n)
        bootstrap_labels = labels[indices]
        bootstrap_scores = scores[indices]
        try:
            ap_values.append(
                float(average_precision_score(bootstrap_labels, bootstrap_scores))
            )
            auc_values.append(float(roc_auc_score(bootstrap_labels, bootstrap_scores)))
        except ValueError:
            continue
    result: dict[str, float] = {}
    for name, values in (("auprc", ap_values), ("auroc", auc_values)):
        array = np.asarray(values, dtype=float)
        result[f"{name}_bootstrap_mean"] = float(np.mean(array))
        result[f"{name}_ci_lower"] = float(np.percentile(array, 2.5))
        result[f"{name}_ci_upper"] = float(np.percentile(array, 97.5))
    return result


def evaluate(args) -> dict[str, object]:
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "scores.tsv"
    scores_path = output_dir / f"{args.model_name}.{suffix}"
    metrics_path = output_dir / f"{args.model_name}.metrics.json"
    manifest_path = output_dir / f"{args.model_name}.manifest.json"
    if scores_path.exists() or metrics_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Output already exists for {args.model_name}; choose a new --output-dir"
        )

    if args.family == "generator":
        try:
            import pysam
        except ImportError as exc:
            raise RuntimeError(
                "GENERATOR native context extraction requires pysam"
            ) from exc
        reference_path = Path(args.reference_fasta).resolve()
        if not Path(str(reference_path) + ".fai").exists():
            raise FileNotFoundError(
                f"Indexed FASTA required; missing {reference_path}.fai"
            )
        fasta = pysam.FastaFile(str(reference_path))
        tokenizer, model = load_model(
            args.model_path, args.device, args.dtype, args.family
        )
        base_token_ids = generator_base_token_ids(tokenizer)
    else:
        fasta = None
        reference_path = None
        tokenizer, model = load_model(
            args.model_path, args.device, args.dtype, args.family
        )
        base_token_ids = None

    started = time.time()
    all_scores: list[float] = []
    all_labels: list[int] = []
    skip_counts: Counter[str] = Counter()
    output_fields = [
        "SeqID",
        "chrom",
        "pos",
        "label",
        "ref",
        "alt",
        "score",
        "p_ref",
        "p_alt",
        "context_bp",
    ]
    total_processed = 0

    iterator = pd.read_csv(
        data_path,
        usecols=CSV_COLUMNS,
        dtype={
            "SeqID": "string",
            "refSeq": "string",
            "altSeq": "string",
            "chrom": "string",
        },
        chunksize=args.rows_per_chunk,
    )
    with scores_path.open("w", newline="", encoding="utf-8") as score_file:
        writer = csv.DictWriter(score_file, fieldnames=output_fields, delimiter="\t")
        writer.writeheader()

        for chunk in tqdm(iterator, desc=f"{args.model_name} ClinVar", unit="chunk"):
            records: list[dict[str, object]] = []
            for row in chunk.itertuples(index=False):
                variant = centered_snv(str(row.refSeq).upper(), str(row.altSeq).upper())
                if variant is None:
                    skip_counts["not_centered_single_snv"] += 1
                    continue
                center_index, ref, alt = variant
                label = int(row.label)
                if label not in (0, 1):
                    skip_counts["invalid_label"] += 1
                    continue

                context = ""
                if args.family == "generator":
                    chromosome = resolve_chromosome(fasta, row.chrom)
                    if chromosome is None:
                        skip_counts["reference_chromosome_missing"] += 1
                        continue
                    position0 = int(row.pos) - 1
                    reference_base = fasta.fetch(
                        chromosome, position0, position0 + 1
                    ).upper()
                    if reference_base != ref:
                        skip_counts["reference_allele_mismatch"] += 1
                        continue
                    context = generator_context(
                        fasta, chromosome, int(row.pos), args.context_bp
                    )
                    if not context:
                        skip_counts["empty_upstream_context"] += 1
                        continue
                else:
                    context = str(row.refSeq).upper()

                records.append(
                    {
                        "SeqID": str(row.SeqID),
                        "chrom": str(row.chrom),
                        "pos": int(row.pos),
                        "label": label,
                        "ref": ref,
                        "alt": alt,
                        "context": context,
                        "alt_context": str(row.altSeq).upper()
                        if args.family == "mistral"
                        else "",
                    }
                )

            if not records:
                total_processed += len(chunk)
                if args.max_rows and total_processed >= args.max_rows:
                    break
                continue

            chunk_scores: list[float] = []
            chunk_p_ref: list[float] = [math.nan] * len(records)
            chunk_p_alt: list[float] = [math.nan] * len(records)
            batch_starts = range(0, len(records), args.batch_size)
            for start in tqdm(
                batch_starts,
                desc=f"{args.model_name} batches",
                unit="batch",
                leave=False,
            ):
                batch = records[start : start + args.batch_size]
                if args.family == "mistral":
                    batch_scores = mistral_score_batch(
                        tokenizer,
                        model,
                        [str(item["context"]) for item in batch],
                        [str(item["alt_context"]) for item in batch],
                        args.device,
                        args.max_tokens,
                    )
                else:
                    contexts = [str(item["context"]) for item in batch]
                    refs = [str(item["ref"]) for item in batch]
                    alts = [str(item["alt"]) for item in batch]
                    batch_scores, batch_p_ref, batch_p_alt = generator_score_batch(
                        tokenizer,
                        model,
                        contexts,
                        refs,
                        alts,
                        base_token_ids,
                        args.device,
                    )
                    chunk_p_ref[start : start + len(batch)] = batch_p_ref.tolist()
                    chunk_p_alt[start : start + len(batch)] = batch_p_alt.tolist()
                chunk_scores.extend(np.asarray(batch_scores, dtype=float).tolist())

            for i, (record, score) in enumerate(zip(records, chunk_scores)):
                writer.writerow(
                    {
                        "SeqID": record["SeqID"],
                        "chrom": record["chrom"],
                        "pos": record["pos"],
                        "label": record["label"],
                        "ref": record["ref"],
                        "alt": record["alt"],
                        "score": score,
                        "p_ref": chunk_p_ref[i],
                        "p_alt": chunk_p_alt[i],
                        "context_bp": len(str(record["context"]))
                        if args.family == "generator"
                        else len(str(record["context"])),
                    }
                )
            score_file.flush()
            all_scores.extend(chunk_scores)
            all_labels.extend(int(item["label"]) for item in records)
            total_processed += len(chunk)
            if args.max_rows and total_processed >= args.max_rows:
                break

    if fasta is not None:
        fasta.close()
    del model
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    scores = np.asarray(all_scores, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=np.int8)
    if not len(scores):
        raise RuntimeError(f"No variants were scored; skip counts: {dict(skip_counts)}")

    precision, recall, _ = precision_recall_curve(labels, scores)
    metrics: dict[str, object] = {
        "model_name": args.model_name,
        "family": args.family,
        "num_samples": int(len(labels)),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "skip_counts": dict(skip_counts),
        "rows_read": int(total_processed),
        "score_file": str(scores_path),
    }
    metrics.update(bootstrap_metrics(labels, scores, args.bootstrap, args.seed))
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest = {
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "model_path": str(Path(args.model_path).resolve()),
        "model_name": args.model_name,
        "family": args.family,
        "dtype": args.dtype,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": args.batch_size,
        "rows_per_chunk": args.rows_per_chunk,
        "benchmark_max_input_bp": 1024,
        "context_bp": args.context_bp if args.family == "generator" else None,
        "reference_fasta": str(reference_path) if reference_path is not None else None,
        "reference_fai": str(reference_path) + ".fai"
        if reference_path is not None
        else None,
        "mistral_max_tokens": args.max_tokens if args.family == "mistral" else None,
        "scoring_protocol": (
            "max_pool_per_position_causal_lm_logits_then_ref_alt_l2"
            if args.family == "mistral"
            else "left_context_next_token_base_marginal_log_p_ref_over_p_alt"
        ),
        "num_samples_scored": int(len(labels)),
        "num_samples_positive": int(labels.sum()),
        "num_samples_negative": int(len(labels) - labels.sum()),
        "skip_counts": dict(skip_counts),
        "metrics_file": str(metrics_path),
        "scores_file": str(scores_path),
        "max_rows_requested": args.max_rows or None,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Scores: {scores_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Manifest: {manifest_path}")
    return metrics


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["mistral", "generator"])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-fasta", default=str(DEFAULT_HG38))
    parser.add_argument("--context-bp", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rows-per-chunk", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    args = parser.parse_args(argv)
    if args.dtype is None:
        args.dtype = "fp32" if args.family == "mistral" else "bf16"
    if args.family == "generator" and not 1 <= args.context_bp <= 1024:
        parser.error("--context-bp must be between 1 and the 1,024 bp benchmark limit")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
