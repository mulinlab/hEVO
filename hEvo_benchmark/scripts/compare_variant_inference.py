#!/usr/bin/env python3
"""Compare baseline and refactored variant inference on real checkpoints.

The parent process selects a deterministic balanced set of real SNVs and runs
baseline/current scorers in separate Python processes. Each process loads a
fresh model from its checkpoint, which avoids module-cache and model-state
sharing between the old and new code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from compare_evaluator_caches import evaluator_source


ROOT = Path(__file__).resolve().parents[2]
EVALUATE_DIR = Path(__file__).resolve().parent / "evaluate"
DEFAULT_BASELINE = Path("/tmp/hevo-eval-baseline-9405e9d")
DEFAULT_DATA = (
    ROOT / "hEvo_benchmark/datasets/Causal_Variants/causal_variants_hg38_1023.csv"
)
DEFAULT_MODELS = Path("/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/models")
DEFAULT_FASTA = Path("/mnt/hpc/home/xuqinyi/DNAModelsBenchmark/datasets/hg38.fa")
DEFAULT_OUTPUT = Path("/tmp/hevo-eval-regression-20261017/inference")

MODEL_CONFIG = {
    "mistral_17m": {
        "name": "Mistral-DNA-17M",
        "relative_path": "Mistral-DNA/Mistral-DNA-v1-17M-hg38",
        "kind": "mistral",
    },
    "generator_1_2b": {
        "name": "GENERator-1.2B",
        "relative_path": "GENERator-v2/GENERator-v2-eukaryote-1.2b-base",
        "kind": "generator",
    },
    "hevo_base": {
        "name": "hEvo-open-base",
        "relative_path": "hEvo_benchmark/models/hEvo-open-base",
        "kind": "hevo",
    },
    "hyena_tiny": {
        "name": "HyenaDNA_tiny_1k",
        "relative_path": "LongSafari/hyenadna-tiny-1k-seqlen-hf-causal",
        "kind": "hyena",
    },
}


def _resolve_chromosome(fasta, chromosome: str) -> str | None:
    chrom = str(chromosome)
    for candidate in (chrom, f"chr{chrom}"):
        if candidate in fasta.references:
            return candidate
    if chrom.upper() in {"M", "MT", "CHRM", "CHRMT"}:
        for candidate in ("chrM", "chrMT", "MT", "M"):
            if candidate in fasta.references:
                return candidate
    return None


def select_pairs(data_path: Path, fasta_path: Path, per_class: int, seed: int):
    """Select balanced centered SNVs whose hg38 allele and context are valid."""
    import pysam

    import random

    candidates: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    with (
        pysam.FastaFile(str(fasta_path)) as fasta,
        data_path.open(newline="", encoding="utf-8") as handle,
    ):
        for row in csv.DictReader(handle):
            try:
                label = int(row["label"])
                ref_seq = row["refseq"].upper()
                alt_seq = row["altseq"].upper()
                if (
                    label not in candidates
                    or len(ref_seq) != len(alt_seq)
                    or not ref_seq
                ):
                    continue
                diffs = [i for i, (a, b) in enumerate(zip(ref_seq, alt_seq)) if a != b]
                if len(diffs) != 1 or diffs[0] != len(ref_seq) // 2:
                    continue
                offset = diffs[0]
                ref_base, alt_base = ref_seq[offset], alt_seq[offset]
                if ref_base not in "ACGT" or alt_base not in "ACGT":
                    continue
                chrom = _resolve_chromosome(fasta, row["chrom"])
                pos = int(row["pos"])
                if chrom is None or pos < 1:
                    continue
                fasta_ref = fasta.fetch(chrom, pos - 1, pos).upper()
                if fasta_ref != ref_base:
                    continue
                start = max(0, pos - 1 - 1024)
                context = fasta.fetch(chrom, start, pos - 1).upper().lstrip("N")
                context = context[len(context) % 6 :]
                if not context:
                    continue
                candidates[label].append(
                    {
                        "id": str(row.get("seqID") or row.get("rsID") or ""),
                        "label": label,
                        "ref_seq": ref_seq,
                        "alt_seq": alt_seq,
                        "ref": ref_base,
                        "alt": alt_base,
                        "chrom": chrom,
                        "pos": pos,
                        "context": context,
                        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        rows = candidates[label]
        rng.shuffle(rows)
        if len(rows) < per_class:
            raise RuntimeError(
                f"Only {len(rows)} valid label={label} records; requested {per_class}"
            )
        selected.extend(rows[:per_class])
    return sorted(selected, key=lambda row: (row["label"], row["id"]))


def _write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_worker(args):
    """Load one code version and score all requested model branches."""
    import numpy as np
    import torch

    source_dir = Path(args.source_dir).resolve()
    sys.path.insert(0, str(EVALUATE_DIR))
    sys.path.insert(0, str(source_dir))
    if (source_dir / "adapters").is_dir():
        from adapters import extract_logits_classes as extractors
        from runtime import native_clinvar_models as native
    else:
        import extract_logits_classes as extractors
        import native_clinvar_models as native

    pair_rows = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "version": args.version,
        "source_dir": str(source_dir),
        "models": {},
    }
    device = torch.device(args.device)

    for key, config in MODEL_CONFIG.items():
        model_root = (
            Path(args.repo_root) if config["kind"] == "hevo" else Path(args.model_root)
        )
        model_path = model_root / config["relative_path"]
        model_report: dict[str, Any] = {
            "model_name": config["name"],
            "model_path": str(model_path),
            "status": "error",
            "source_sha256": {
                filename: _sha256(
                    evaluator_source(source_dir, filename, required=False)
                )
                for filename in (
                    "extract_logits_classes.py",
                    "native_clinvar_models.py",
                    "native_variant_scorers.py",
                )
            },
            "checkpoint_provenance": {
                filename: _sha256(model_path / filename)
                for filename in (
                    "config.json",
                    "provenance.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                )
                if (model_path / filename).is_file()
            },
        }
        result["models"][key] = model_report
        try:
            if config["kind"] == "mistral":
                scorer = extractors.ExtractLogitForMistralDNA(
                    str(model_path), config["name"]
                )
                scorer.device = device
                scorer.model.to(device)
                scores, native_scores = [], []
                for row in pair_rows:
                    values = scorer.score_paired_sequences(
                        [row["ref_seq"]], [row["alt_seq"]], max_tokens=512
                    )
                    scores.append(float(values[0]))
                    native_values = native.mistral_score_batch(
                        scorer.tokenizer,
                        scorer.model,
                        [row["ref_seq"]],
                        [row["alt_seq"]],
                        device,
                        512,
                    )
                    native_scores.append(float(native_values[0]))
                model_report["native_score_type"] = "mistral_native_clinvar_runner"
                model_report["score_type"] = "mistral_pooled_vocab_logits_l2_v1"
            elif config["kind"] == "generator":
                scorer = extractors.ExtractLogitForGENERator(
                    str(model_path), config["name"]
                )
                scorer.device = device
                scorer.model.to(device)
                scores, p_refs, p_alts = [], [], []
                native_scores, native_p_refs, native_p_alts = [], [], []
                base_token_ids = native.generator_base_token_ids(scorer.tokenizer)
                for row in pair_rows:
                    values = scorer.score_next_base(
                        [row["context"]], [row["ref"]], [row["alt"]]
                    )
                    scores.append(float(values[0][0]))
                    p_refs.append(float(values[1][0]))
                    p_alts.append(float(values[2][0]))
                    native_values = native.generator_score_batch(
                        scorer.tokenizer,
                        scorer.model,
                        [row["context"]],
                        [row["ref"]],
                        [row["alt"]],
                        base_token_ids,
                        device,
                    )
                    native_scores.append(float(native_values[0][0]))
                    native_p_refs.append(float(native_values[1][0]))
                    native_p_alts.append(float(native_values[2][0]))
                model_report["native_score_type"] = "generator_native_clinvar_runner"
                model_report["score_type"] = "generator_next_token_base_marginal_v1"
                model_report["p_ref"] = p_refs
                model_report["p_alt"] = p_alts
            elif config["kind"] == "hevo":
                scorer = extractors.ExtractLogitForhEvo_rare(
                    str(model_path), config["name"]
                )
                scorer.device = device
                scorer.model.to(device)
                scores = [
                    float(scorer.calc_variant_llr(row["ref_seq"], row["alt_seq"]))
                    for row in pair_rows
                ]
                model_report["score_type"] = "hevo_target_masked_alt_minus_ref"
            else:
                scorer = extractors.ExtractLogitForHyena(
                    str(model_path), config["name"]
                )
                scorer.device = device
                scorer.model.to(device)
                scores = []
                for row in pair_rows:
                    ref = scorer.tokenize([row["ref_seq"]], max_length=1024)
                    alt = scorer.tokenize([row["alt_seq"]], max_length=1024)
                    ref_ids = ref["input_ids"].to(device)
                    alt_ids = alt["input_ids"].to(device)
                    ref_logits = scorer.get_logit(
                        ref_ids, ref.get("attention_mask"), return_numpy=False
                    )[0]
                    alt_logits = scorer.get_logit(
                        alt_ids, alt.get("attention_mask"), return_numpy=False
                    )[0]
                    scores.append(
                        float(
                            scorer.calc_score(alt_logits, alt_ids[0])
                            - scorer.calc_score(ref_logits, ref_ids[0])
                        )
                    )
                model_report["score_type"] = "hyena_mean_pll_alt_minus_ref"

            scores = np.asarray(scores, dtype=np.float64)
            if scores.shape != (len(pair_rows),) or not np.isfinite(scores).all():
                raise ValueError(
                    f"Expected {len(pair_rows)} finite scores, got shape={scores.shape}"
                )
            model_report.update(
                {
                    "status": "ok",
                    "scores": scores.tolist(),
                    "count": int(len(scores)),
                    "labels": [int(row["label"]) for row in pair_rows],
                    "ids": [row["id"] for row in pair_rows],
                    "context_sha256": [row["context_sha256"] for row in pair_rows],
                    "score_mean": float(scores.mean()),
                    "score_std": float(scores.std()),
                }
            )
            if config["kind"] in ("mistral", "generator"):
                native_scores = np.asarray(native_scores, dtype=np.float64)
                model_report["native_scores"] = native_scores.tolist()
                model_report["native_max_abs_diff"] = float(
                    np.max(np.abs(native_scores - scores))
                )
                if config["kind"] == "generator":
                    model_report["native_p_ref"] = native_p_refs
                    model_report["native_p_alt"] = native_p_alts
                    model_report["native_p_ref_max_abs_diff"] = float(
                        np.max(np.abs(np.asarray(native_p_refs) - np.asarray(p_refs)))
                    )
                    model_report["native_p_alt_max_abs_diff"] = float(
                        np.max(np.abs(np.asarray(native_p_alts) - np.asarray(p_alts)))
                    )
        except Exception as error:
            model_report["error"] = f"{type(error).__name__}: {error}"
        finally:
            if "scorer" in locals():
                del scorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_json(Path(args.output), result)


def compare_reports(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    comparison = {}
    for key in MODEL_CONFIG:
        old_model = old["models"].get(key, {})
        new_model = new["models"].get(key, {})
        entry: dict[str, Any] = {
            "old_status": old_model.get("status", "missing"),
            "new_status": new_model.get("status", "missing"),
            "ids_match": old_model.get("ids") == new_model.get("ids"),
            "context_hashes_match": old_model.get("context_sha256")
            == new_model.get("context_sha256"),
        }
        if old_model.get("status") == "ok" and new_model.get("status") == "ok":
            before = np.asarray(old_model["scores"], dtype=np.float64)
            after = np.asarray(new_model["scores"], dtype=np.float64)
            diff = after - before
            entry.update(
                {
                    "count": int(len(before)),
                    "max_abs_diff": float(np.max(np.abs(diff))),
                    "mean_abs_diff": float(np.mean(np.abs(diff))),
                    "rmse": float(np.sqrt(np.mean(diff**2))),
                    "old_score_mean": float(before.mean()),
                    "new_score_mean": float(after.mean()),
                    "pearson_correlation": (
                        float(np.corrcoef(before, after)[0, 1])
                        if np.std(before) and np.std(after)
                        else None
                    ),
                    "spearman_rank_correlation": _spearman(before, after),
                    "max_rank_displacement": _max_rank_displacement(before, after),
                    "number_of_rank_changes": int(
                        np.count_nonzero(_rankdata(before) != _rankdata(after))
                    ),
                    "allclose_atol_1e-6_rtol_1e-5": bool(
                        np.allclose(before, after, atol=1e-6, rtol=1e-5)
                    ),
                }
            )
            if "p_ref" in old_model and "p_ref" in new_model:
                for probability_name in ("p_ref", "p_alt"):
                    p_old = np.asarray(old_model[probability_name], dtype=np.float64)
                    p_new = np.asarray(new_model[probability_name], dtype=np.float64)
                    entry[f"{probability_name}_max_abs_diff"] = float(
                        np.max(np.abs(p_new - p_old))
                    )
            if "native_scores" in old_model and "native_scores" in new_model:
                for version_name, model in (("old", old_model), ("new", new_model)):
                    native_scores = np.asarray(model["native_scores"], dtype=np.float64)
                    extractor_scores = np.asarray(model["scores"], dtype=np.float64)
                    entry[f"{version_name}_native_max_abs_diff"] = float(
                        np.max(np.abs(native_scores - extractor_scores))
                    )
        else:
            entry["old_error"] = old_model.get("error")
            entry["new_error"] = new_model.get("error")
        comparison[key] = entry
    return comparison


def _rankdata(values):
    from scipy.stats import rankdata

    return rankdata(values, method="average")


def _spearman(before, after):
    import numpy as np

    if np.std(before) == 0 or np.std(after) == 0:
        return None
    return float(np.corrcoef(_rankdata(before), _rankdata(after))[0, 1])


def _max_rank_displacement(before, after):
    import numpy as np

    return float(np.max(np.abs(_rankdata(before) - _rankdata(after))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", choices=["old", "new"], help=argparse.SUPPRESS)
    parser.add_argument("--source-dir", help=argparse.SUPPRESS)
    parser.add_argument("--model-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pairs", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = select_pairs(args.data, args.fasta, args.per_class, args.seed)
    pairs_path = args.output_dir / "selected_pairs.json"
    _write_json(pairs_path, pair_rows)
    version_reports = {}
    python = sys.executable
    for version, source in (
        ("old", args.baseline / "hEvo_benchmark/scripts/evaluate"),
        ("new", EVALUATE_DIR),
    ):
        output_path = args.output_dir / f"{version}_inference.json"
        command = [
            python,
            str(Path(__file__).resolve()),
            "--worker",
            "--version",
            version,
            "--source-dir",
            str(source),
            "--pairs",
            str(pairs_path),
            "--output",
            str(output_path),
            "--model-root",
            str(args.models),
            "--repo-root",
            str(ROOT),
            "--device",
            args.device,
        ]
        try:
            subprocess.run(command, check=True, env=os.environ.copy())
            version_reports[version] = json.loads(
                output_path.read_text(encoding="utf-8")
            )
        except Exception as error:
            version_reports[version] = {
                "version": version,
                "models": {},
                "error": f"{type(error).__name__}: {error}",
            }

    comparison = compare_reports(
        version_reports.get("old", {"models": {}}),
        version_reports.get("new", {"models": {}}),
    )
    _write_json(args.output_dir / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
