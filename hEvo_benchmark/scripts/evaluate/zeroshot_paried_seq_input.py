#!/usr/bin/env python3
"""
使用模型logits计算LLR (Log-Likelihood Ratio) 并评估AUROC/AUPRC

用法示例:
python zeroshot_paried_seq_input.py \
  --model_paths /path/to/model \
  --model_name DNABERT2 \
  --dataset_files /path/to/data.txt \
  --dataset_name variant_dataset \
  --h5_path ./variant_scores_h5 \
  --batch_size 64 \
  --precompute_batch_size 64 \
  --num_workers 4 \
  --ref_col 1 \
  --alt_col 2 \
  --label_col 3

数据格式要求：
每行包含: idx, ref_seq, alt_seq, label (tab或space分隔)
ref_seq和alt_seq应该仅在一个位置有差异
"""

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REFERENCE_FASTA = os.environ.get(
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
GPN_STAR_MODEL_IDS = frozenset({"GPN-Star-V", "GPN-Star-M", "GPN-Star-P"})


def build_parser():
    parser = argparse.ArgumentParser(
        description="使用预计算logits计算LLR并评估AUROC/AUPRC"
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help="模型目录路径（支持多个，逗号分隔）",
    )
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument(
        "--dataset_files",
        type=str,
        required=True,
        help="数据文件路径（支持多个，逗号分隔）",
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--h5_path", type=str, required=True, help="HDF5文件根目录")
    parser.add_argument(
        "--sep",
        type=str,
        default="tab",
        help="数据文件分隔符；默认优先 tab，并自动识别逗号分隔的 CSV",
    )
    parser.add_argument("--idx_col", type=int, default=0, help="索引列位置")
    parser.add_argument("--ref_col", type=int, default=1, help="参考序列列位置")
    parser.add_argument("--alt_col", type=int, default=2, help="变异序列列位置")
    parser.add_argument("--label_col", type=int, default=3, help="标签列位置")
    parser.add_argument(
        "--chrom_col",
        type=int,
        default=None,
        help="可选染色体列；省略时尝试从含 chrom/pos 的表头自动识别",
    )
    parser.add_argument(
        "--pos_col",
        type=int,
        default=None,
        help="可选 1-based 变异坐标列；省略时尝试从含 chrom/pos 的表头自动识别",
    )
    parser.add_argument(
        "--reference_fasta",
        type=str,
        default=DEFAULT_REFERENCE_FASTA,
        help="GENERATOR 提取上游基因组上下文所用的已索引 FASTA",
    )
    parser.add_argument(
        "--context_bp",
        type=int,
        default=1024,
        help="GENERATOR 最多使用多少 bp 上游参考上下文（上限 1,024 bp）",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument(
        "--precompute_batch_size",
        type=int,
        default=32,
        help="通用模型预计算 batch；Mistral-DNA/GENERator 使用固定的原生推理 batch",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="通用模型 tokenizer 的最大 token 数；所有输入序列另受 --max_input_bp 限制",
    )
    parser.add_argument(
        "--max_input_bp",
        type=int,
        default=1024,
        help="benchmark 输入序列最大长度(bp)，不得超过协议上限 1,024 bp",
    )
    parser.add_argument(
        "--max_model_tokens",
        type=int,
        default=512,
        help="Mistral-DNA 原生评分 tokenizer token 上限",
    )
    parser.add_argument(
        "--center_index",
        type=int,
        default=None,
        help="可选的中心位置；指定后强制要求唯一 SNV 位于该零基索引",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader的工作进程数"
    )
    parser.add_argument(
        "--recompute", action="store_true", help="强制重新计算logits（忽略缓存）"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap重采样次数（0表示不计算置信区间）",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./llr_results", help="结果输出目录"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs/llr_eval", help="TensorBoard日志目录"
    )
    parser.add_argument(
        "--del_old_runs", action="store_true", help="是否删除旧的TensorBoard日志"
    )
    parser.add_argument("--use_abs", action="store_true", help="兼容旧命令：使用 |LLR|")
    parser.add_argument(
        "--score_mode",
        choices=["neg_llr", "abs_llr", "raw_llr"],
        default=None,
        help="显式指定分数方向；默认根据 --use_abs 兼容旧行为",
    )
    parser.add_argument(
        "--gpn-star-model-id", choices=sorted(GPN_STAR_MODEL_IDS), default=None
    )
    parser.add_argument("--gpn-star-model-dir", default=None)
    parser.add_argument("--gpn-star-model-revision", default=None)
    parser.add_argument("--gpn-star-msa-store", action="append", default=[])
    parser.add_argument("--gpn-star-msa-species", nargs="+", type=int, default=[])
    parser.add_argument("--gpn-star-msa-dataset", default="")
    parser.add_argument("--gpn-star-msa-revision", default="")
    parser.add_argument(
        "--gpn-star-input",
        default=None,
        help="Normalized GPN-Star TSV; defaults to the first dataset file",
    )
    parser.add_argument(
        "--gpn-star-output", default=None, help="External native-MSA score TSV"
    )
    parser.add_argument(
        "--gpn-star-manifest", default=None, help="External native-MSA run manifest"
    )
    parser.add_argument("--gpn-star-window-bp", type=int, default=1024)
    parser.add_argument("--gpn-star-device", default="auto")
    parser.add_argument("--gpn-star-max-records", type=int, default=0)
    return parser


# Help uses the same parser as execution, before optional model imports.
if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()


import hashlib  # noqa: E402 - lightweight CLI help precedes runtime imports
import json  # noqa: E402 - lightweight CLI help precedes runtime imports
import random  # noqa: E402 - lightweight CLI help precedes runtime imports
from typing import Dict, Optional, Tuple  # noqa: E402 - lightweight CLI help precedes runtime imports

import h5py  # noqa: E402 - lightweight CLI help precedes runtime imports
import matplotlib.pyplot as plt  # noqa: E402 - lightweight CLI help precedes runtime imports
import numpy as np  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch  # noqa: E402 - lightweight CLI help precedes runtime imports
import tqdm  # noqa: E402 - lightweight CLI help precedes runtime imports
from adapters.extract_logits_classes import (  # noqa: E402 - lightweight CLI help precedes runtime imports
    ExtractLogitForAIDO,
    ExtractLogitForCaduceus,
    ExtractLogitForDNABert,
    ExtractLogitForEvo2_1B,
    ExtractLogitForEvo2_7B,
    ExtractLogitForGENALM,
    ExtractLogitForGENERator,
    ExtractLogitForGenos,
    ExtractLogitForGPN,
    ExtractLogitForGROVER,
    ExtractLogitForhEvo_full,
    ExtractLogitForhEvo_rare,
    ExtractLogitForHyena,
    ExtractLogitForMistralDNA,
    ExtractLogitForNT,
    ExtractLogitForNTv3,
    ExtractLogitForSpeciesLM,
)
from core.provenance import filesystem_fingerprint  # noqa: E402 - runtime imports follow CLI help
from adapters.masked_base_scorer import (  # noqa: E402
    BPE_MASKED_BASE_MODEL_IDS,
    SCORING_METHOD as BPE_SCORING_METHOD,
    score_masked_bases,
)
from sklearn.metrics import (  # noqa: E402 - lightweight CLI help precedes runtime imports
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Dataset  # noqa: E402 - lightweight CLI help precedes runtime imports
from torch.utils.tensorboard import SummaryWriter  # noqa: E402 - lightweight CLI help precedes runtime imports

# ========= Random seed setup ========= #
SEED = 42
DNA_BASES = frozenset("ACGT")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ================================= #


# Example usage:
# extractor = ExtractLogitForhEvoPos(model_path, "hEvo-with-position")
#
# ---------- Extractor mapping ----------
model_extractor_map = {
    "AIDO.DNA-300M": ExtractLogitForAIDO,
    "DNABERT2": ExtractLogitForDNABert,
    "DNABERT": ExtractLogitForDNABert,
    "hEvo-open-base": ExtractLogitForhEvo_rare,
    "hEvo-base": ExtractLogitForhEvo_rare,
    "hEvo-large": ExtractLogitForhEvo_full,
    "hEvo-open-large": ExtractLogitForhEvo_full,
    "NT_2.5b_1kG": ExtractLogitForNT,
    "NT_500m_1kG": ExtractLogitForNT,
    "NT_500m_MS": ExtractLogitForNT,
    "NT_2.5b_MS": ExtractLogitForNT,
    "HyenaDNA_tiny_1k": ExtractLogitForHyena,
    "GPN-promoter": ExtractLogitForGPN,
    "SpeciesLM": ExtractLogitForSpeciesLM,
    "evo2_7b_base": ExtractLogitForEvo2_7B,
    "evo2_1b_base": ExtractLogitForEvo2_1B,
    "Genos-1.2B": ExtractLogitForGenos,
    "GENA-LM-base": ExtractLogitForGENALM,
    "GENA-LM-large": ExtractLogitForGENALM,
    "GROVER": ExtractLogitForGROVER,
    "NTv3-8M-pre": ExtractLogitForNTv3,
    "NTv3-100M-pre": ExtractLogitForNTv3,
    "NTv3-650M-pre": ExtractLogitForNTv3,
    "Mistral-DNA-17M": ExtractLogitForMistralDNA,
    "Mistral-DNA-138M": ExtractLogitForMistralDNA,
    "Mistral-DNA-422M": ExtractLogitForMistralDNA,
    "GENERator-1.2B": ExtractLogitForGENERator,
    "GENERator-3B": ExtractLogitForGENERator,
    "Caduceus-PS": ExtractLogitForCaduceus,
    "Caduceus-PH": ExtractLogitForCaduceus,
}

NATIVE_CAUSAL_MODEL_IDS = frozenset(
    {
        "Mistral-DNA-17M",
        "Mistral-DNA-138M",
        "Mistral-DNA-422M",
        "GENERator-1.2B",
        "GENERator-3B",
    }
)


def native_inference_batch_size(model_name: str) -> int:
    """Keep causal-model batch composition fixed for reproducible scoring."""
    return 8 if model_name == "GENERator-3B" else 16


NT_MODEL_IDS = frozenset({"NT_2.5b_1kG", "NT_500m_1kG", "NT_500m_MS", "NT_2.5b_MS"})
HEVO_MODEL_IDS = frozenset(
    {
        "hEvo-1kG-86M",
        "hEvo-1kG-sliding128-86M",
        "hEvo-open-base",
        "hEvo-base",
        "hEvo-large",
        "hEvo-open-large",
        "hEvo-fullUKB-86M",
        "hEvo-60k-86M",
    }
)
MASKED_MARGINAL_MODEL_IDS = frozenset(
    {
        "AIDO.DNA-300M",
        "DNABERT",
        "DNABERT2",
        "GPN-promoter",
        "SpeciesLM",
        "GENA-LM-base",
        "GENA-LM-large",
        "GROVER",
        "NTv3-8M-pre",
        "NTv3-100M-pre",
        "NTv3-650M-pre",
        "Caduceus-PS",
        "Caduceus-PH",
    }
)


def uses_masked_marginal_variant_scoring(extractor) -> bool:
    """Whether a model scores the allele from a reference context with its token masked."""
    model_name = getattr(extractor, "model_name", None)
    if model_name == "GPN-MSA":
        raise ValueError("GPN-MSA requires its coordinate-aware native MSA scorer")
    if isinstance(extractor, ExtractLogitForNT) or model_name in (
        HEVO_MODEL_IDS | MASKED_MARGINAL_MODEL_IDS
    ):
        return True
    if isinstance(
        extractor,
        (
            ExtractLogitForEvo2_1B,
            ExtractLogitForEvo2_7B,
            ExtractLogitForHyena,
            ExtractLogitForGenos,
        ),
    ):
        return False
    raise ValueError(
        f"No masked or causal variant scorer registered for {model_name!r}"
    )


def calc_masked_marginal_llr(
    logits: torch.Tensor,
    ref_input_ids: torch.Tensor,
    alt_input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[float, list[int]]:
    """Sum alt-vs-ref token log-probabilities at masked variant tokens."""
    changed = ref_input_ids.ne(alt_input_ids)
    if attention_mask is not None:
        changed &= attention_mask.to(changed.device).bool()
    positions = torch.where(changed)[0]
    if positions.numel() == 0:
        raise ValueError("ref/alt encodings contain no active changed token")
    token_logits = logits[positions].float()
    log_probs = torch.log_softmax(token_logits, dim=-1)
    refs = ref_input_ids[positions].to(log_probs.device)
    alts = alt_input_ids[positions].to(log_probs.device)
    score = (
        log_probs.gather(1, alts[:, None]).sum()
        - log_probs.gather(1, refs[:, None]).sum()
    )
    return float(score.item()), [int(pos) for pos in positions.detach().cpu().tolist()]


def run_gpn_star_evaluation(
    model_path: str,
    dataset_file: str,
    dataset_name: str,
    args,
) -> Dict[str, object]:
    """Dispatch GPN-Star to the official native-MSA protocol.

    GPN-Star is intentionally isolated from ``model_extractor_map``: its VEP
    score requires the published MSA layout and cannot be reproduced by the
    single-sequence logits path. ``dataset_file`` must be a normalized TSV
    produced by ``gpn_star_input.py`` (or an equivalent file with the same
    schema).
    """
    from adapters.gpn_star_adapter import run as run_native_msa
    from adapters.gpn_star_adapter import write_manifest as write_gpn_star_manifest

    model_id = args.gpn_star_model_id or args.model_name
    model_dir = Path(args.gpn_star_model_dir or model_path)
    revision = args.gpn_star_model_revision
    if not revision:
        raise ValueError("GPN-Star requires --gpn-star-model-revision")
    # argparse returns one list per occurrence when ``--gpn-star-msa-store``
    # uses ``nargs=+``.  Flatten it so both repeated single-path options and
    # one multi-path option have the expected native-MSA shape.
    msa_store = [
        store
        for group in (args.gpn_star_msa_store or [])
        for store in (group if isinstance(group, (list, tuple)) else [group])
    ]
    msa_species = list(args.gpn_star_msa_species or [])
    if not msa_store or not msa_species:
        raise ValueError(
            "GPN-Star requires --gpn-star-msa-store and --gpn-star-msa-species"
        )

    native_args = argparse.Namespace(
        model_id=model_id,
        model_dir=model_dir,
        model_revision=revision,
        msa_store=msa_store,
        msa_species=msa_species,
        msa_dataset=args.gpn_star_msa_dataset,
        msa_revision=args.gpn_star_msa_revision,
        input=Path(args.gpn_star_input or dataset_file),
        output=Path(
            args.gpn_star_output
            or os.path.join(args.out_dir, f"scores_{model_id}_{dataset_name}.tsv")
        ),
        window_bp=args.gpn_star_window_bp,
        batch_size=args.precompute_batch_size,
        device=args.gpn_star_device,
        max_records=args.gpn_star_max_records,
        manifest=(
            Path(getattr(args, "gpn_star_manifest", ""))
            if getattr(args, "gpn_star_manifest", None)
            else None
        ),
    )
    result, diagnostics = run_native_msa(native_args)
    native_args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(native_args.output, sep="\t", index=False)
    write_gpn_star_manifest(native_args, result, diagnostics)

    passed = result[result["status"] == "PASS"]
    if passed.empty:
        raise RuntimeError("GPN-Star produced no PASS rows")
    labels = np.asarray(passed["label"].astype(int), dtype=np.int64)
    scores = np.asarray(passed["benchmark_score"].astype(float), dtype=np.float64)
    if len(np.unique(labels)) != 2 or not np.isfinite(scores).all():
        raise RuntimeError("GPN-Star PASS rows do not contain finite two-class scores")

    # Reuse the common AUROC/AUPRC output contract. The native protocol's
    # benchmark_score is already oriented so larger values mean more positive;
    # represent it as a synthetic negative LLR solely for the common renderer.
    metrics = evaluate_llr_scores(
        llrs=torch.from_numpy(-scores.astype(np.float32)),
        labels=torch.from_numpy(labels),
        output_dir=args.out_dir,
        dataset_name=dataset_name,
        model_name=model_id,
        use_abs=False,
        score_mode="neg_llr",
    )
    metrics.update(
        {
            "score_mode": "gpn_star_benchmark_score",
            "gpn_star_model_id": model_id,
            "gpn_star_model_revision": revision,
            "gpn_star_score_file": str(native_args.output),
            "gpn_star_diagnostics": diagnostics,
            "gpn_star_pass_rows": int(len(passed)),
            "gpn_star_blocked_rows": int(len(result) - len(passed)),
            "evaluator_sha256": current_evaluator_sha256(),
        }
    )
    results_file = Path(args.out_dir) / f"results_{model_id}_{dataset_name}.json"
    results_file.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def validate_variant_pair(
    ref_seq: str, alt_seq: str, center_index: int | None = None
) -> tuple[int, str, str]:
    if not ref_seq or not alt_seq:
        raise ValueError("empty_sequence")
    if len(ref_seq) != len(alt_seq):
        raise ValueError("length_mismatch")
    differences = [
        index
        for index, (ref_base, alt_base) in enumerate(zip(ref_seq, alt_seq))
        if ref_base != alt_base
    ]
    if len(differences) != 1:
        raise ValueError("not_exactly_one_snv")
    position = differences[0]
    ref_base, alt_base = ref_seq[position], alt_seq[position]
    if ref_base not in DNA_BASES or alt_base not in DNA_BASES:
        raise ValueError("invalid_snv_allele")
    if center_index is not None and position != center_index:
        raise ValueError("not_at_center_index")
    return position, ref_base, alt_base


def cache_fingerprint(
    model_path: str,
    dataset_file: str,
    args,
    use_abs: bool,
    score_mode: str | None = None,
) -> dict[str, str | int | bool | None]:
    data_stat = os.stat(dataset_file)
    payload = {
        "schema": "llr-cache-v3"
        if args.model_name in NATIVE_CAUSAL_MODEL_IDS
        else "llr-cache-v2",
        "model_path": os.path.abspath(model_path),
        "model_fingerprint": filesystem_fingerprint(model_path),
        "dataset_file": os.path.abspath(dataset_file),
        "dataset_size": int(data_stat.st_size),
        "dataset_mtime_ns": int(data_stat.st_mtime_ns),
        "model_name": args.model_name,
        "max_length": int(args.max_length),
        "max_input_bp": int(getattr(args, "max_input_bp", 1024)),
        "center_index": args.center_index,
        "score_mode": score_mode or ("abs_llr" if use_abs else "neg_llr"),
    }
    if args.model_name in NATIVE_CAUSAL_MODEL_IDS:
        if args.model_name.startswith("Mistral-DNA-"):
            payload["variant_scoring_method"] = "mistral_pooled_vocab_logits_l2_v1"
            payload["max_model_tokens"] = int(getattr(args, "max_model_tokens", 512))
        else:
            payload["variant_scoring_method"] = "generator_next_token_base_marginal_v1"
            payload["context_bp"] = int(getattr(args, "context_bp", 1024))
            payload["max_input_bp"] = int(getattr(args, "max_input_bp", 1024))
            payload["chrom_col"] = getattr(args, "chrom_col", None)
            payload["pos_col"] = getattr(args, "pos_col", None)
            reference_fasta = getattr(args, "reference_fasta", None)
            payload["reference_fasta"] = (
                os.path.abspath(reference_fasta) if reference_fasta else None
            )
            if reference_fasta and os.path.isfile(reference_fasta):
                payload["reference_fasta_fingerprint"] = filesystem_fingerprint(
                    reference_fasta
                )
                fai_path = f"{reference_fasta}.fai"
                payload["reference_fai_fingerprint"] = (
                    filesystem_fingerprint(fai_path)
                    if os.path.isfile(fai_path)
                    else None
                )
        payload["native_inference_batch_size"] = native_inference_batch_size(
            args.model_name
        )
    elif args.model_name in NT_MODEL_IDS:
        # NT scoring masks the changed 6-mer before predicting its token.
        # The scoring protocol is part of the cache identity.
        payload["variant_scoring_method"] = "nt_masked_marginal_6mer_v1"
    elif args.model_name in HEVO_MODEL_IDS:
        payload["variant_scoring_method"] = "hevo_masked_marginal_v1"
    elif args.model_name in BPE_MASKED_BASE_MODEL_IDS:
        payload["variant_scoring_method"] = BPE_SCORING_METHOD
    elif args.model_name in MASKED_MARGINAL_MODEL_IDS:
        payload["variant_scoring_method"] = "masked_marginal_changed_tokens_v1"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


# ---------- Dataset ----------
class VariantPairDataset(Dataset):
    """变异序列对数据集"""

    def __init__(
        self,
        input_file: str,
        ref_col: int = 1,
        alt_col: int = 2,
        sep: str = ",",
        idx_col: int = 0,
        label_col: int = 3,
        center_index: int | None = None,
        chrom_col: int | None = None,
        pos_col: int | None = None,
        max_input_bp: int = 1024,
    ):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"未找到文件: {input_file}")

        self.samples = []
        self.variant_metadata: list[dict[str, object]] = []
        self.rejection_counts: dict[str, int] = {}
        if (chrom_col is None) != (pos_col is None):
            raise ValueError("chrom_col and pos_col must be supplied together")
        resolved_chrom_col = chrom_col
        resolved_pos_col = pos_col
        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(sep)

                # Infer optional genomic coordinates from a conventional header,
                # and skip that header instead of counting it as a malformed row.
                normalized = [item.strip().lstrip("\ufeff").lower() for item in parts]
                if any(
                    value in {"label", "refseq", "altseq", "seqid"}
                    for value in normalized
                ):
                    if resolved_chrom_col is None:
                        chrom_names = {"chrom", "chromosome", "chr"}
                        pos_names = {"pos", "position", "start"}
                        resolved_chrom_col = next(
                            (
                                i
                                for i, value in enumerate(normalized)
                                if value in chrom_names
                            ),
                            None,
                        )
                        resolved_pos_col = next(
                            (
                                i
                                for i, value in enumerate(normalized)
                                if value in pos_names
                            ),
                            None,
                        )
                        if (resolved_chrom_col is None) != (resolved_pos_col is None):
                            resolved_chrom_col = resolved_pos_col = None
                    continue

                columns = [ref_col, alt_col, idx_col, label_col]
                if resolved_chrom_col is not None and resolved_pos_col is not None:
                    columns.extend([resolved_chrom_col, resolved_pos_col])
                check_len = max(columns)
                if len(parts) <= check_len:
                    print(f"Warning: 跳过第{line_num}行，列数不足")
                    continue

                try:
                    idx = parts[idx_col].strip()
                    ref_seq = parts[ref_col].strip().upper()
                    alt_seq = parts[alt_col].strip().upper()
                    label = int(parts[label_col].strip())

                    if len(ref_seq) > max_input_bp or len(alt_seq) > max_input_bp:
                        reason = "sequence_exceeds_max_input_bp"
                        self.rejection_counts[reason] = (
                            self.rejection_counts.get(reason, 0) + 1
                        )
                        continue

                    try:
                        validate_variant_pair(
                            ref_seq, alt_seq, center_index=center_index
                        )
                    except ValueError as error:
                        reason = str(error)
                        self.rejection_counts[reason] = (
                            self.rejection_counts.get(reason, 0) + 1
                        )
                        print(f"Warning: 跳过第{line_num}行，{reason}")
                        continue
                    metadata: dict[str, object] = {}
                    if resolved_chrom_col is not None and resolved_pos_col is not None:
                        chrom = parts[resolved_chrom_col].strip()
                        pos = int(parts[resolved_pos_col].strip())
                        if not chrom or pos < 1:
                            raise ValueError("invalid genomic coordinate")
                        metadata = {"chrom": chrom, "pos": pos}
                    self.samples.append((idx, ref_seq, alt_seq, label))
                    self.variant_metadata.append(metadata)
                except (ValueError, IndexError) as e:
                    print(f"Warning: 跳过第{line_num}行，解析错误: {e}")
                    continue

        print(f"成功加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_variant_fn(batch):
    """Collate function for variant pairs"""
    indices = [item[0] for item in batch]
    ref_seqs = [item[1] for item in batch]
    alt_seqs = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    return indices, ref_seqs, alt_seqs, labels


def resolve_fasta_chromosome(fasta, chrom: str) -> str | None:
    """Resolve common chromosome aliases against an indexed reference FASTA."""
    chrom = str(chrom)
    candidates = [f"chr{chrom}", chrom]
    if chrom.upper() in {"M", "MT", "CHRM", "CHRMT"}:
        candidates.extend(["chrM", "chrMT", "MT", "M"])
    return next(
        (candidate for candidate in candidates if candidate in fasta.references), None
    )


def generator_upstream_context(
    ref_seq: str,
    variant_offset: int,
    context_bp: int,
    fasta=None,
    chrom: str | None = None,
    pos: int | None = None,
) -> tuple[str, str]:
    """Return a capped upstream context and the source used to obtain it."""
    if fasta is not None and chrom is not None and pos is not None:
        resolved = resolve_fasta_chromosome(fasta, chrom)
        if resolved is None:
            raise ValueError("reference_chromosome_missing")
        position0 = int(pos) - 1
        if position0 < 0:
            raise ValueError("invalid_genomic_coordinate")
        reference_base = fasta.fetch(resolved, position0, position0 + 1).upper()
        if reference_base != ref_seq[variant_offset]:
            raise ValueError("reference_allele_mismatch")
        start = max(0, position0 - context_bp)
        context = fasta.fetch(resolved, start, position0).upper()
        source = "reference_fasta_upstream"
    else:
        # Without genomic coordinates, use the portion of the supplied paired
        # sequence upstream of the SNV. It obeys the same bp cap but can be
        # shorter than a full genomic upstream window.
        context = ref_seq[:variant_offset][-context_bp:].upper()
        source = "paired_sequence_upstream_fallback"

    context = context.lstrip("N")
    remainder = len(context) % 6
    if remainder:
        context = context[remainder:]
    if not context:
        raise ValueError("empty_upstream_context")
    return context, source


def precompute_and_cache_native_scores_h5(
    extractor,
    dataset: VariantPairDataset,
    h5_file_path: str,
    batch_size: int = 16,
    max_input_bp: int = 1024,
    max_model_tokens: int = 512,
    context_bp: int = 1024,
    reference_fasta: str | None = None,
    cache_metadata: Optional[dict[str, str | int | bool | None]] = None,
):
    """Run the model-specific Mistral-DNA or GENERATOR variant score."""
    if not 1 <= max_input_bp <= 1024:
        raise ValueError(
            "max_input_bp must be between 1 and the 1,024 bp benchmark limit"
        )
    if not 1 <= context_bp <= max_input_bp:
        raise ValueError("GENERATOR context_bp must be between 1 and 1,024 bp")

    is_mistral = isinstance(extractor, ExtractLogitForMistralDNA)
    use_coordinates = bool(dataset.variant_metadata) and any(
        "chrom" in item and "pos" in item for item in dataset.variant_metadata
    )
    if use_coordinates and not all(
        "chrom" in item and "pos" in item for item in dataset.variant_metadata
    ):
        raise ValueError("Genomic coordinates are missing for one or more variants")

    fasta = None
    if not is_mistral and use_coordinates:
        reference_fasta = reference_fasta or DEFAULT_REFERENCE_FASTA
        if not os.path.isfile(reference_fasta):
            raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")
        if not os.path.isfile(f"{reference_fasta}.fai"):
            raise FileNotFoundError(
                f"Indexed reference FASTA required: {reference_fasta}.fai"
            )
        try:
            import pysam
        except ImportError as error:
            raise RuntimeError(
                "GENERATOR coordinate context extraction requires pysam"
            ) from error
        fasta = pysam.FastaFile(reference_fasta)

    print(
        "开始计算模型原生变异分数并写入 HDF5："
        f"{('Mistral pooled-logit L2' if is_mistral else 'GENERATOR next-base marginal log(Pref/Palt)')}"
    )
    all_scores: list[float] = []
    all_labels: list[int] = []
    detail_list: list[str] = []

    try:
        for start in tqdm.tqdm(
            range(0, len(dataset), batch_size), desc="Native variant scores"
        ):
            batch_indices: list[str] = []
            batch_ref_seqs: list[str] = []
            batch_alt_seqs: list[str] = []
            batch_refs: list[str] = []
            batch_alts: list[str] = []
            batch_labels: list[int] = []
            batch_offsets: list[int] = []
            batch_metadata: list[dict[str, object]] = []
            batch_contexts: list[str] = []
            batch_context_sources: list[str] = []

            stop = min(start + batch_size, len(dataset))
            for row_index in range(start, stop):
                idx, ref_seq, alt_seq, label = dataset[row_index]
                metadata = dataset.variant_metadata[row_index]
                try:
                    variant_offset, ref_base, alt_base = validate_variant_pair(
                        ref_seq, alt_seq
                    )
                    context = ""
                    context_source = "paired_sequence"
                    if not is_mistral:
                        context, context_source = generator_upstream_context(
                            ref_seq=ref_seq,
                            variant_offset=variant_offset,
                            context_bp=context_bp,
                            fasta=fasta,
                            chrom=str(metadata["chrom"])
                            if "chrom" in metadata
                            else None,
                            pos=int(metadata["pos"]) if "pos" in metadata else None,
                        )
                    batch_indices.append(str(idx))
                    batch_ref_seqs.append(ref_seq)
                    batch_alt_seqs.append(alt_seq)
                    batch_refs.append(ref_base)
                    batch_alts.append(alt_base)
                    batch_labels.append(int(label))
                    batch_offsets.append(variant_offset)
                    batch_metadata.append(metadata)
                    if not is_mistral:
                        batch_contexts.append(context)
                        batch_context_sources.append(context_source)
                except ValueError as error:
                    reason = str(error)
                    dataset.rejection_counts[reason] = (
                        dataset.rejection_counts.get(reason, 0) + 1
                    )

            if not batch_indices:
                continue

            if is_mistral:
                batch_scores = extractor.score_paired_sequences(
                    batch_ref_seqs, batch_alt_seqs, max_tokens=max_model_tokens
                )
                p_refs = p_alts = [None] * len(batch_scores)
            else:
                batch_scores, p_refs, p_alts = extractor.score_next_base(
                    batch_contexts, batch_refs, batch_alts
                )

            for j, (idx, score) in enumerate(zip(batch_indices, batch_scores)):
                metadata = batch_metadata[j]
                position = (
                    int(metadata["pos"]) if "pos" in metadata else batch_offsets[j]
                )
                detail = {
                    "idx": idx,
                    "position": position,
                    "ref_char": batch_refs[j],
                    "alt_char": batch_alts[j],
                    "score": float(score),
                    "llr": float(score),
                    "label": batch_labels[j],
                    "scoring_method": (
                        "mistral_pooled_vocab_logits_l2_v1"
                        if is_mistral
                        else "generator_next_token_base_marginal_v1"
                    ),
                }
                if not is_mistral:
                    detail.update(
                        {
                            "context_bp": len(batch_contexts[j]),
                            "context_source": batch_context_sources[j],
                            "p_ref": float(p_refs[j]),
                            "p_alt": float(p_alts[j]),
                        }
                    )
                all_scores.append(float(score))
                all_labels.append(batch_labels[j])
                detail_list.append(json.dumps(detail))
    finally:
        if fasta is not None:
            fasta.close()

    if not all_scores:
        raise RuntimeError(
            f"No variants were scored; rejection counts: {dataset.rejection_counts}"
        )

    os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)
    if os.path.exists(h5_file_path):
        os.remove(h5_file_path)
    with h5py.File(h5_file_path, "w") as handle:
        handle.create_dataset("llrs", data=np.asarray(all_scores, dtype=np.float32))
        handle.create_dataset("labels", data=np.asarray(all_labels, dtype=np.int32))
        detail_dtype = h5py.special_dtype(vlen=str)
        details = handle.create_dataset(
            "details", shape=(len(detail_list),), dtype=detail_dtype
        )
        for index, detail in enumerate(detail_list):
            details[index] = detail
        handle.attrs["num_samples"] = len(all_scores)
        handle.attrs["score_semantics"] = (
            "effect_distance" if is_mistral else "log_p_ref_over_p_alt"
        )
        if cache_metadata:
            for key, value in cache_metadata.items():
                handle.attrs[key] = "" if value is None else value
    print(f"Native scoring finished: {len(all_scores)} variants -> {h5_file_path}")


def precompute_masked_base_scores_h5(
    extractor, dataset, h5_file_path, batch_size, max_length, cache_metadata,
):
    """Cache BPE masked-nucleotide scores with exact paired-sequence identity."""
    scores, labels, details = [], [], []
    for start in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Masked base scores"):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        ids, refs, alts, batch_labels = collate_variant_fn(batch)
        batch_scores, variants = score_masked_bases(extractor, refs, alts, max_length)
        for idx, ref, alt, label, llr, (pos, r, a) in zip(
            ids, refs, alts, batch_labels, batch_scores, variants
        ):
            scores.append(llr)
            labels.append(int(label))
            details.append(json.dumps({"idx": str(idx), "ref_seq": ref,
                "alt_seq": alt, "label": int(label), "position": pos,
                "ref_char": r, "alt_char": a, "llr": llr,
                "scoring_method": BPE_SCORING_METHOD}))
    os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)
    with h5py.File(h5_file_path, "w") as handle:
        handle.create_dataset("llrs", data=np.asarray(scores, dtype=np.float32))
        handle.create_dataset("labels", data=np.asarray(labels, dtype=np.int32))
        handle.create_dataset("details", data=details, dtype=h5py.string_dtype())
        handle.attrs["num_samples"] = len(scores)
        for key, value in cache_metadata.items():
            handle.attrs[key] = "" if value is None else value


def precompute_and_cache_logits_h5(
    extractor,
    dataset: VariantPairDataset,
    h5_file_path: str,
    batch_size: int = 64,
    max_length: int = 1024,
    cache_metadata: Optional[dict[str, str | int | bool | None]] = None,
):
    """
    计算 ref 序列的 logits，并提取变异信息存入 HDF5 文件。

    存储内容包括：
        llrs -> (N,) 的数组，保存 LLRS
        labels -> (N,) 的数组，保存标签
        details -> (N,) 的字符串数组，每个元素是 JSON，格式如下：
            {
                "idx": str(idx),
                "ref_seq": ref_seq,
                "alt_seq": alt_seq,
                "position": position,
                "ref_char": ref_char,
                "alt_char": alt_char,
                "llr": float(llr),
                "label": int(label),
            }
    """
    print(f"开始预计算 logits 和 LLR，并保存到 {h5_file_path} ...")

    all_llrs = []
    all_labels = []
    detail_list = []

    # Process in batches.
    for i in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Precompute Logits"):
        batch_samples = [
            dataset[j] for j in range(i, min(i + batch_size, len(dataset)))
        ]
        indices, ref_seqs, alt_seqs, labels = collate_variant_fn(batch_samples)

        # Tokenize the reference sequence.
        encoded_ref = extractor.tokenize(
            ref_seqs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded_alt = extractor.tokenize(
            alt_seqs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_ref["input_ids"].to(extractor.device)
        input_ids_alt = encoded_alt["input_ids"].to(extractor.device)
        attention_mask = encoded_ref.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(extractor.device)
        attention_mask_alt = encoded_alt.get("attention_mask")
        if attention_mask_alt is not None:
            attention_mask_alt = attention_mask_alt.to(extractor.device)

        # Get logits for the reference sequence.
        with torch.inference_mode():
            if uses_masked_marginal_variant_scoring(extractor):
                ref_logits = extractor.get_masked_variant_logits(
                    ref_input_ids=input_ids,
                    alt_input_ids=input_ids_alt,
                    attention_mask=attention_mask,
                    alt_attention_mask=attention_mask_alt,
                    return_numpy=False,
                )
                alt_logits = None
            else:
                ref_logits = extractor.get_logit(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_numpy=False,
                )

                alt_logits = extractor.get_logit(
                    input_ids=input_ids_alt,
                    attention_mask=attention_mask_alt,
                    return_numpy=False,
                )
        # Process each sample.
        for j, (idx, ref_seq, alt_seq, label) in enumerate(
            zip(indices, ref_seqs, alt_seqs, labels)
        ):
            if isinstance(
                extractor,
                (
                    ExtractLogitForEvo2_1B,
                    ExtractLogitForEvo2_7B,
                    ExtractLogitForHyena,
                    ExtractLogitForGenos,
                ),
            ):
                try:
                    # Get information about the differing tokens.
                    position, ref_token, alt_token, ref_char, alt_char = (
                        extractor.get_diff_token(ref_seq, alt_seq)
                    )
                    # Calculate the log-likelihood ratio (LLR).
                    sample_logits_ref = ref_logits[j]  # (seq_len, vocab_size)
                    sample_logits_alt = alt_logits[j]  # (seq_len, vocab_size)
                    input_id_ref = input_ids[j]
                    input_id_alt = input_ids_alt[j]
                    ref_score = extractor.calc_score(
                        sample_logits_ref,
                        input_id=input_id_ref,
                    )
                    alt_score = extractor.calc_score(
                        sample_logits_alt,
                        input_id=input_id_alt,
                    )
                    llr = alt_score - ref_score
                    all_llrs.append(llr)
                    all_labels.append(label)

                    # Save only the requested fields.
                    detail_dict = {
                        "idx": str(idx),
                        "ref_seq": ref_seq,
                        "alt_seq": alt_seq,
                        "position": position,
                        "ref_char": ref_char,
                        "alt_char": alt_char,
                        "llr": float(llr),
                        "label": int(label),
                    }
                    detail_list.append(json.dumps(detail_dict))

                except Exception as e:
                    print(f"Warning: 处理样本 {idx} 时出错: {e}")
                    continue

            else:
                try:
                    llr, position = calc_masked_marginal_llr(
                        ref_logits[j],
                        input_ids[j],
                        input_ids_alt[j],
                        attention_mask[j] if attention_mask is not None else None,
                    )
                    ref_char, alt_char = ref_seq, alt_seq
                    all_llrs.append(llr)
                    all_labels.append(label)

                    # Save only the requested fields.
                    detail_dict = {
                        "idx": str(idx),
                        "ref_seq": ref_seq,
                        "alt_seq": alt_seq,
                        "position": position,
                        "ref_char": ref_char,
                        "alt_char": alt_char,
                        "llr": float(llr),
                        "label": int(label),
                    }
                    detail_list.append(json.dumps(detail_dict))

                except Exception as e:
                    print(f"Warning: 处理样本 {idx} 时出错: {e}")
                    continue

    # Write the results to HDF5.
    os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)
    if os.path.exists(h5_file_path):
        # Delete and recreate the file if it already exists.
        os.remove(h5_file_path)

    all_llrs_arr = np.array(all_llrs, dtype=np.float32)
    all_labels_arr = np.array(all_labels, dtype=np.int32)

    with h5py.File(h5_file_path, "w") as f:
        f.create_dataset("llrs", data=all_llrs_arr)
        f.create_dataset("labels", data=all_labels_arr)

        dt_string = h5py.special_dtype(vlen=str)
        detail_dset = f.create_dataset(
            "details", shape=(len(detail_list),), dtype=dt_string
        )
        for i, detail_str in enumerate(detail_list):
            detail_dset[i] = detail_str

        f.attrs["num_samples"] = len(all_llrs_arr)
        if cache_metadata:
            for key, value in cache_metadata.items():
                f.attrs[key] = "" if value is None else value

    print(f"完成预计算，共 {len(all_llrs_arr)} 个样本。结果已写入 {h5_file_path}")


def precompute_and_cache_logits_h5_with_position(
    extractor,
    dataset: VariantPairDataset,
    h5_file_path: str,
    batch_size: int = 64,
    max_length: int = 1024,
    cache_metadata: Optional[dict[str, str | int | bool | None]] = None,
):
    """
    Compute masked MLM or causal sequence scores and cache variant details.
    """
    print(f"开始预计算 logits 和 LLR，并保存到 {h5_file_path} ...")

    all_llrs = []
    all_labels = []
    detail_list = []

    # Process in batches.
    for i in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Precompute Logits"):
        batch_samples = [
            dataset[j] for j in range(i, min(i + batch_size, len(dataset)))
        ]
        indices, ref_seqs, alt_seqs, labels = collate_variant_fn(batch_samples)

        encoded_ref = extractor.tokenize(
            ref_seqs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded_alt = extractor.tokenize(
            alt_seqs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_ref["input_ids"].to(extractor.device)
        input_ids_alt = encoded_alt["input_ids"].to(extractor.device)
        attention_mask = encoded_ref.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(extractor.device)
        attention_mask_alt = encoded_alt.get("attention_mask")
        if attention_mask_alt is not None:
            attention_mask_alt = attention_mask_alt.to(extractor.device)

        # ===== Get logits =====
        with torch.inference_mode():
            if uses_masked_marginal_variant_scoring(extractor):
                ref_logits = extractor.get_masked_variant_logits(
                    ref_input_ids=input_ids,
                    alt_input_ids=input_ids_alt,
                    attention_mask=attention_mask,
                    alt_attention_mask=attention_mask_alt,
                    return_numpy=False,
                )
                alt_logits = None
            else:
                # Standard inference.
                ref_logits = extractor.get_logit(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_numpy=False,
                )
                alt_logits = extractor.get_logit(
                    input_ids=input_ids_alt,
                    attention_mask=attention_mask,
                    return_numpy=False,
                )

        # ===== Process each sample =====
        for j, (idx, ref_seq, alt_seq, label) in enumerate(
            zip(indices, ref_seqs, alt_seqs, labels)
        ):
            # Calculate the LLR according to the model type.
            # Causal language model (use calc_score).
            if isinstance(
                extractor,
                (
                    ExtractLogitForEvo2_1B,
                    ExtractLogitForEvo2_7B,
                    ExtractLogitForHyena,
                    ExtractLogitForGenos,
                ),
            ):
                try:
                    # Get information about the differing tokens.
                    position, ref_token, alt_token, ref_char, alt_char = (
                        extractor.get_diff_token(ref_seq, alt_seq)
                    )
                    # Calculate the score.
                    sample_logits_ref = ref_logits[j]
                    sample_logits_alt = alt_logits[j]
                    input_id_ref = input_ids[j]
                    input_id_alt = input_ids_alt[j]

                    ref_score = extractor.calc_score(
                        sample_logits_ref,
                        input_id=input_id_ref,
                    )
                    alt_score = extractor.calc_score(
                        sample_logits_alt,
                        input_id=input_id_alt,
                    )
                    llr = alt_score - ref_score
                    all_llrs.append(llr)
                    all_labels.append(label)

                    detail_dict = {
                        "idx": str(idx),
                        "ref_seq": ref_seq,
                        "alt_seq": alt_seq,
                        "position": position,
                        "ref_char": ref_char,
                        "alt_char": alt_char,
                        "llr": float(llr),
                        "label": int(label),
                    }
                    detail_list.append(json.dumps(detail_dict))

                except Exception as e:
                    print(f"Warning: 处理样本 {idx} 时出错: {e}")
                    continue
            else:
                # Masked MLM scoring at all changed target tokens.
                llr, position = calc_masked_marginal_llr(
                    ref_logits[j],
                    input_ids[j],
                    input_ids_alt[j],
                    attention_mask[j] if attention_mask is not None else None,
                )
                ref_char, alt_char = ref_seq, alt_seq
                all_llrs.append(llr)
                all_labels.append(label)

                detail_dict = {
                    "idx": str(idx),
                    "ref_seq": ref_seq,
                    "alt_seq": alt_seq,
                    "position": position,
                    "ref_char": ref_char,
                    "alt_char": alt_char,
                    "llr": float(llr),
                    "label": int(label),
                }
                detail_list.append(json.dumps(detail_dict))

    # ===== Write to HDF5 =====
    os.makedirs(os.path.dirname(h5_file_path), exist_ok=True)
    if os.path.exists(h5_file_path):
        os.remove(h5_file_path)

    all_llrs_arr = np.array(all_llrs, dtype=np.float32)
    all_labels_arr = np.array(all_labels, dtype=np.int32)

    with h5py.File(h5_file_path, "w") as f:
        f.create_dataset("llrs", data=all_llrs_arr)
        f.create_dataset("labels", data=all_labels_arr)

        dt_string = h5py.special_dtype(vlen=str)
        detail_dset = f.create_dataset(
            "details", shape=(len(detail_list),), dtype=dt_string
        )
        for i, detail_str in enumerate(detail_list):
            detail_dset[i] = detail_str

        f.attrs["num_samples"] = len(all_llrs_arr)
        if cache_metadata:
            for key, value in cache_metadata.items():
                f.attrs[key] = "" if value is None else value

    print(f"完成预计算，共 {len(all_llrs_arr)} 个样本。结果已写入 {h5_file_path}")


def load_llrs_from_h5(h5_file_path: str, expected_fingerprint: str | None = None):
    """
    从指定的 HDF5 文件加载 llrs 和 labels。
    如果文件不存在或内容不完整，则返回 None。
    否则返回 (llrs: torch.Tensor, labels: torch.Tensor)。
    """
    if not os.path.exists(h5_file_path):
        return None

    with h5py.File(h5_file_path, "r") as f:
        if (
            expected_fingerprint is not None
            and f.attrs.get("fingerprint") != expected_fingerprint
        ):
            return None
        if "llrs" not in f or "labels" not in f:
            return None

        llrs_np = f["llrs"][:]
        labels_np = f["labels"][:]

    llrs = torch.from_numpy(llrs_np)
    labels = torch.from_numpy(labels_np)
    return llrs, labels


def evaluate_llr_scores(
    llrs: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    dataset_name: str,
    model_name: str,
    use_abs: bool = False,
    score_mode: str | None = None,
    score_label_text: str | None = None,
) -> Dict[str, object]:
    """
    评估LLR分数的性能。

    参数:
      llrs: LLR分数 (N,)，通常为 log(P_alt) - log(P_ref)
      labels: 标签 (N,)，1 表示致病/有功能，0 表示良性/对照
      output_dir: 输出目录
      dataset_name: 数据集名称
      model_name: 模型名称
      use_abs: 是否使用绝对值进行评估。True用于不分方向的功能性变异评估，False则使用 -llr。

    返回:
      包含指标的字典
    """
    # Convert to NumPy.
    llrs_np = llrs.detach().cpu().numpy() if torch.is_tensor(llrs) else np.array(llrs)
    labels_np = (
        labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
    )

    # ================= Score conversion =================
    # The scores must be positively associated with label=1.
    score_mode = score_mode or ("abs_llr" if use_abs else "neg_llr")
    if score_mode == "abs_llr":
        scores = np.abs(llrs_np)
        score_label_text = score_label_text or "|LLR|"
    elif score_mode == "raw_llr":
        scores = llrs_np
        score_label_text = score_label_text or "LLR"
    elif score_mode == "neg_llr":
        scores = -llrs_np
        score_label_text = score_label_text or "-LLR"
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
    # ===============================================

    # Calculate metrics.
    metrics: Dict[str, object] = {
        "num_samples": int(len(labels_np)),
        "num_positive": int(np.sum(labels_np)),
        "num_negative": int(len(labels_np) - np.sum(labels_np)),
        "score_mode": score_mode,
    }

    # Calculate AUROC.
    try:
        auroc = roc_auc_score(labels_np, scores)
    except Exception as e:
        print(f"计算AUROC失败: {e}")
        auroc = float("nan")

    # Calculate AUPRC.
    try:
        auprc = average_precision_score(labels_np, scores)
    except Exception as e:
        print(f"计算AUPRC失败: {e}")
        auprc = float("nan")

    os.makedirs(output_dir, exist_ok=True)
    suffix = (
        "abs"
        if score_mode == "abs_llr"
        else ("raw" if score_mode == "raw_llr" else "neg")
    )

    # Plot the ROC curve.
    try:
        fpr, tpr, _ = roc_curve(labels_np, scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (area = {auroc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({score_label_text})\n{model_name} on {dataset_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        roc_path = os.path.join(
            output_dir, f"roc_{model_name}_{dataset_name}_{suffix}.png"
        )
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"绘制ROC曲线失败: {e}")

    # Plot the precision-recall curve.
    try:
        precision, recall, _ = precision_recall_curve(labels_np, scores)
        plt.figure(figsize=(8, 6))
        plt.plot(
            recall,
            precision,
            color="forestgreen",
            lw=2,
            label=f"PR (area = {auprc:.4f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"Precision-Recall Curve ({score_label_text})\n{model_name} on {dataset_name}"
        )
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        pr_path = os.path.join(
            output_dir, f"pr_{model_name}_{dataset_name}_{suffix}.png"
        )
        plt.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"绘制PR曲线失败: {e}")

    # Update the metrics dictionary.
    metrics.update(
        {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "llr_mean": float(np.mean(llrs_np)),
            "llr_std": float(np.std(llrs_np)),
            "raw_score_mean": float(np.mean(llrs_np)),
            "raw_score_std": float(np.std(llrs_np)),
            "score_mean": float(np.mean(scores)),
            "score_max": float(np.max(scores)),
        }
    )

    print(f"\n--- 评估结果 [{score_label_text}] ---")
    print(f"数据集: {dataset_name} | 模型: {model_name}")
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    print(f"正/负样本数: {metrics['num_positive']}/{metrics['num_negative']}")
    print(f"结果图片已保存至: {output_dir}")

    return metrics


def _llrs_to_scores(llrs: np.ndarray, score_mode: str) -> np.ndarray:
    """Convert LLR values to scores aligned with the requested score mode."""
    if score_mode == "neg_llr":
        return -llrs
    if score_mode == "abs_llr":
        return np.abs(llrs)
    if score_mode == "raw_llr":
        return llrs
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def _bootstrap_ci_for_metric(
    llrs: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    random_state: int = SEED,
    score_mode: str = "neg_llr",
) -> Tuple[float, float, float]:
    """Calculate a row-level non-parametric bootstrap CI for one metric."""
    rng = np.random.RandomState(random_state)
    n = len(labels)
    metric_values = []

    for _ in range(n_bootstrap):
        idxs = rng.randint(0, n, size=n)
        labels_boot = labels[idxs]
        llrs_boot = llrs[idxs]

        try:
            scores_boot = _llrs_to_scores(llrs_boot, score_mode)
            metric_values.append(metric_fn(labels_boot, scores_boot))
        except Exception:
            # A resample containing one class cannot define AUROC; omit it.
            continue

    metric_values = np.asarray(metric_values, dtype=float)
    metric_values = metric_values[~np.isnan(metric_values)]

    if len(metric_values) == 0:
        return (float("nan"), float("nan"), float("nan"))

    return (
        float(np.mean(metric_values)),
        float(np.percentile(metric_values, 2.5)),
        float(np.percentile(metric_values, 97.5)),
    )


def bootstrap_ci_for_auroc(
    llrs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = SEED,
    score_mode: str = "neg_llr",
) -> Tuple[float, float, float]:
    """Calculate a 95% percentile bootstrap CI for AUROC."""
    return _bootstrap_ci_for_metric(
        llrs=llrs,
        labels=labels,
        metric_fn=roc_auc_score,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        score_mode=score_mode,
    )


def bootstrap_ci_for_auprc(
    llrs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = SEED,
    score_mode: str = "neg_llr",
) -> Tuple[float, float, float]:
    """Calculate a 95% percentile bootstrap CI for AUPRC."""
    return _bootstrap_ci_for_metric(
        llrs=llrs,
        labels=labels,
        metric_fn=average_precision_score,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        score_mode=score_mode,
    )


def current_evaluator_sha256() -> str:
    """Return the SHA256 of this evaluator source file for result provenance."""
    digest = hashlib.sha256()
    with Path(__file__).resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_llr_evaluation(
    model_path: str,
    dataset_file: str,
    dataset_name: str,
    use_abs: bool,
    args,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, object]:
    """
    运行LLR评估的主流程

    参数:
      model_path: 模型路径
      dataset_file: 数据文件路径
      dataset_name: 数据集名称
      args: 命令行参数
      writer: TensorBoard writer

    返回:
      评估指标字典
    """
    model_name = args.model_name
    if model_name not in model_extractor_map:
        raise ValueError(
            f"No shared masked or causal scorer registered for {model_name!r}"
        )
    print(f"\n{'=' * 60}")
    print(f"模型: {model_name}")
    print(f"数据集: {dataset_name}")
    print(f"数据文件: {dataset_file}")
    print(f"{'=' * 60}\n")

    # Determine the HDF5 file path.
    hdf5_dir = os.path.join(args.h5_path, dataset_name)
    hdf5_path = os.path.join(hdf5_dir, f"{model_name}.hdf5")
    is_native_causal = model_name in NATIVE_CAUSAL_MODEL_IDS
    from core.score_protocol import metric_mode_for_model

    metric_score_mode = metric_mode_for_model(model_name, args.score_mode)
    cache_meta = cache_fingerprint(
        model_path,
        dataset_file,
        args,
        use_abs,
        score_mode="raw_llr" if is_native_causal else metric_score_mode,
    )

    # Parse and validate inputs first, preserving rejection counts even on a cache hit.
    sep = "\t" if args.sep == "tab" else ","
    if args.sep == "tab":
        with open(dataset_file, "r", encoding="utf-8") as source:
            first_line = next((line for line in source if line.strip()), "")
        if "\t" not in first_line and "," in first_line:
            sep = ","
    dataset = VariantPairDataset(
        input_file=dataset_file,
        ref_col=args.ref_col,
        alt_col=args.alt_col,
        sep=sep,
        idx_col=args.idx_col,
        label_col=args.label_col,
        center_index=args.center_index,
        chrom_col=args.chrom_col,
        pos_col=args.pos_col,
        max_input_bp=args.max_input_bp,
    )

    # First, try loading from the HDF5 file.
    loaded = load_llrs_from_h5(
        hdf5_path, expected_fingerprint=cache_meta["fingerprint"]
    )

    if loaded is None or args.recompute:
        if loaded is not None and args.recompute:
            print("检测到 --recompute 参数，将重新计算logits...")
        else:
            print("HDF5文件中未找到可用结果，开始计算...")

        extractor = model_extractor_map[model_name](model_path, model_name)

        # The dataset is parsed and validated before checking the cache.
        if len(dataset) == 0:
            raise RuntimeError(f"数据集为空: {dataset_file}")

        if is_native_causal:
            safe_batch_size = native_inference_batch_size(model_name)
            print(
                f"使用模型原生评分流程；max_input_bp={args.max_input_bp}, "
                f"固定推理 batch_size={safe_batch_size}"
            )
            precompute_and_cache_native_scores_h5(
                extractor=extractor,
                dataset=dataset,
                h5_file_path=hdf5_path,
                batch_size=safe_batch_size,
                max_input_bp=args.max_input_bp,
                max_model_tokens=args.max_model_tokens,
                context_bp=args.context_bp,
                reference_fasta=args.reference_fasta,
                cache_metadata=cache_meta,
            )
        elif model_name in BPE_MASKED_BASE_MODEL_IDS:
            precompute_masked_base_scores_h5(
                extractor, dataset, hdf5_path, args.precompute_batch_size,
                args.max_length, cache_meta,
            )
        else:
            # Shared single-sequence scoring uses masked MLM or causal likelihoods.
            precompute_and_cache_logits_h5_with_position(
                extractor=extractor,
                dataset=dataset,
                h5_file_path=hdf5_path,
                batch_size=args.precompute_batch_size,
                max_length=args.max_length,
                cache_metadata=cache_meta,
            )

        # Reload the data.
        loaded = load_llrs_from_h5(
            hdf5_path, expected_fingerprint=cache_meta["fingerprint"]
        )
        if loaded is None:
            raise RuntimeError("预计算后仍无法加载数据，可能存在错误")
    else:
        print("从 HDF5 文件加载预计算结果...")

    llrs, labels = loaded
    print(f"加载了 {len(llrs)} 个样本的 LLR 分数")

    # Evaluate performance.
    metrics = evaluate_llr_scores(
        llrs=llrs,
        labels=labels,
        output_dir=args.out_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        use_abs=False if is_native_causal else use_abs,
        score_mode=metric_score_mode,
        score_label_text=(
            "pooled-logit L2 distance"
            if model_name.startswith("Mistral-DNA-")
            else (
                "|LLR|"
                if metric_score_mode == "abs_llr"
                else "-LLR"
                if metric_score_mode == "raw_llr"
                else "LLR"
            )
            if model_name.startswith("GENERator-")
            else None
        ),
    )
    metrics["rejection_counts"] = dataset.rejection_counts
    metrics["cache_fingerprint"] = cache_meta["fingerprint"]
    if model_name.startswith("Mistral-DNA-"):
        metrics["variant_scoring_method"] = "mistral_pooled_vocab_logits_l2_v1"
        metrics["score_semantics"] = (
            "pooled-logit L2 distance; larger is a stronger effect score"
        )
        metrics["benchmark_max_input_bp"] = int(args.max_input_bp)
        metrics["max_model_tokens"] = int(args.max_model_tokens)
        metrics["native_inference_batch_size"] = native_inference_batch_size(model_name)
    elif model_name.startswith("GENERator-"):
        metrics["variant_scoring_method"] = "generator_next_token_base_marginal_v1"
        metrics["score_semantics"] = "log(P_ref/P_alt)"
        metrics["benchmark_max_input_bp"] = int(args.max_input_bp)
        metrics["context_bp"] = int(args.context_bp)
        metrics["native_inference_batch_size"] = native_inference_batch_size(model_name)
        metrics["reference_fasta"] = (
            os.path.abspath(args.reference_fasta)
            if any("chrom" in item for item in dataset.variant_metadata)
            else None
        )
        metrics["context_source"] = (
            "reference_fasta_upstream"
            if any("chrom" in item for item in dataset.variant_metadata)
            else "paired_sequence_upstream_fallback"
        )
    elif args.model_name in NT_MODEL_IDS:
        metrics["variant_scoring_method"] = "nt_masked_marginal_6mer_v1"
    elif args.model_name in HEVO_MODEL_IDS:
        metrics["variant_scoring_method"] = "hevo_masked_marginal_v1"
    elif args.model_name in BPE_MASKED_BASE_MODEL_IDS:
        metrics["variant_scoring_method"] = BPE_SCORING_METHOD
    elif args.model_name in MASKED_MARGINAL_MODEL_IDS:
        metrics["variant_scoring_method"] = "masked_marginal_changed_tokens_v1"

    # Record bootstrap and evaluator provenance for new runs only; historical results are unchanged.
    bootstrap_enabled = args.bootstrap > 0
    metrics["bootstrap_n"] = int(args.bootstrap)
    metrics["bootstrap_seed"] = int(SEED) if bootstrap_enabled else None
    metrics["bootstrap_method"] = (
        "nonparametric_percentile" if bootstrap_enabled else "disabled"
    )
    metrics["bootstrap_resampling_unit"] = (
        "row_iid_with_replacement" if bootstrap_enabled else None
    )
    metrics["bootstrap_percentiles"] = [2.5, 97.5] if bootstrap_enabled else None
    metrics["evaluator_sha256"] = current_evaluator_sha256()
    metrics["score_protocol_sha256"] = hashlib.sha256(
        (Path(__file__).resolve().parent / "core" / "score_protocol.py").read_bytes()
    ).hexdigest()
    metrics["score_mode"] = metric_score_mode
    metrics["task_score_mode"] = args.score_mode

    # Calculate bootstrap confidence intervals.
    if bootstrap_enabled:
        print(f"\n计算Bootstrap置信区间 (n={args.bootstrap})...")
        llrs_np = (
            llrs.detach().cpu().numpy() if torch.is_tensor(llrs) else np.asarray(llrs)
        )
        labels_np = (
            labels.detach().cpu().numpy()
            if torch.is_tensor(labels)
            else np.asarray(labels)
        )

        auroc_mean, auroc_lower, auroc_upper = bootstrap_ci_for_auroc(
            llrs_np,
            labels_np,
            n_bootstrap=args.bootstrap,
            random_state=SEED,
            score_mode=metric_score_mode,
        )
        auprc_mean, auprc_lower, auprc_upper = bootstrap_ci_for_auprc(
            llrs_np,
            labels_np,
            n_bootstrap=args.bootstrap,
            random_state=SEED,
            score_mode=metric_score_mode,
        )
        metrics["auroc_bootstrap_mean"] = auroc_mean
        metrics["auroc_bootstrap_lower"] = auroc_lower
        metrics["auroc_bootstrap_upper"] = auroc_upper
        metrics["auprc_bootstrap_mean"] = auprc_mean
        metrics["auprc_bootstrap_lower"] = auprc_lower
        metrics["auprc_bootstrap_upper"] = auprc_upper
        print(
            f"  AUROC (Bootstrap): {auroc_mean:.4f} [{auroc_lower:.4f}, {auroc_upper:.4f}]"
        )
        print(
            f"  AUPRC (Bootstrap): {auprc_mean:.4f} [{auprc_lower:.4f}, {auprc_upper:.4f}]"
        )

    # Write to TensorBoard.
    if writer is not None:
        writer.add_scalar(f"eval/{dataset_name}/auroc", metrics["auroc"], 0)
        writer.add_scalar(f"eval/{dataset_name}/auprc", metrics["auprc"], 0)
        if "auroc_bootstrap_mean" in metrics:
            writer.add_scalar(
                f"eval/{dataset_name}/auroc_bootstrap",
                metrics["auroc_bootstrap_mean"],
                0,
            )
        if "auprc_bootstrap_mean" in metrics:
            writer.add_scalar(
                f"eval/{dataset_name}/auprc_bootstrap",
                metrics["auprc_bootstrap_mean"],
                0,
            )

    # Save the results to JSON.
    results_file = os.path.join(
        args.out_dir, f"results_{model_name}_{dataset_name}.json"
    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_file}")

    return metrics


def main(argv=None):
    parser = build_parser()

    args = parser.parse_args(argv)
    if not 1 <= args.max_input_bp <= 1024:
        parser.error("--max_input_bp must be between 1 and the 1,024 bp protocol limit")
    if not 1 <= args.context_bp <= args.max_input_bp:
        parser.error("--context_bp must be between 1 and --max_input_bp")
    if args.max_model_tokens < 1:
        parser.error("--max_model_tokens must be positive")
    if (args.chrom_col is not None) != (args.pos_col is not None):
        parser.error("--chrom_col and --pos_col must be supplied together")
    if args.model_name == "GPN-MSA":
        parser.error(
            "GPN-MSA requires the coordinate-aware native MSA VEP entry point; "
            "the generic single-sequence extractor is not a native scorer"
        )
    if args.model_name and args.model_name.startswith("Mistral-DNA-"):
        if args.score_mode not in (None, "raw_llr") or args.use_abs:
            parser.error(
                "Mistral-DNA outputs an effect distance, not an LLR; "
                "use raw mode for this scorer"
            )
        args.score_mode = "raw_llr"
    if args.score_mode is None:
        args.score_mode = "abs_llr" if args.use_abs else "neg_llr"

    # Create the output directory.
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Remove old logs.
    if args.del_old_runs and os.path.exists(args.log_dir):
        import shutil

        shutil.rmtree(args.log_dir, ignore_errors=True)
        os.makedirs(args.log_dir, exist_ok=True)

    # Parse the model path and data files.
    model_paths = [p.strip() for p in args.model_paths.split(",") if p.strip()]
    dataset_files = [p.strip() for p in args.dataset_files.split(",") if p.strip()]

    if len(model_paths) == 0:
        raise ValueError("至少需要指定一个模型路径")
    if len(dataset_files) == 0:
        raise ValueError("至少需要指定一个数据文件")

    if args.model_name in GPN_STAR_MODEL_IDS or args.gpn_star_model_id is not None:
        if len(model_paths) != 1 or len(dataset_files) != 1:
            raise ValueError(
                "GPN-Star native-MSA mode accepts exactly one model and one dataset"
            )
        model_id = args.gpn_star_model_id or args.model_name
        metrics = run_gpn_star_evaluation(
            model_path=model_paths[0],
            dataset_file=dataset_files[0],
            dataset_name=args.dataset_name,
            args=args,
        )
        summary_file = os.path.join(args.out_dir, "summary_results.json")
        with open(summary_file, "w", encoding="utf-8") as handle:
            json.dump(
                {model_id: {args.dataset_name: metrics}},
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n汇总结果已保存到: {summary_file}")
        return

    # Initialize TensorBoard.
    writer = SummaryWriter(log_dir=args.log_dir)

    # Store all results.
    all_results = {}
    failed_evaluations = []

    # Evaluate each model/data-file combination.
    for model_path in model_paths:
        this_model_name = os.path.basename(model_path.rstrip("/"))
        all_results[this_model_name] = {}

        for dataset_file in dataset_files:
            # Generate a unique dataset_name for each input when multiple data files are provided.
            if len(dataset_files) > 1:
                file_basename = os.path.splitext(os.path.basename(dataset_file))[0]
                current_dataset_name = f"{args.dataset_name}_{file_basename}"
            else:
                current_dataset_name = args.dataset_name

            try:
                metrics = run_llr_evaluation(
                    model_path=model_path,
                    dataset_file=dataset_file,
                    dataset_name=current_dataset_name,
                    use_abs=args.use_abs,
                    args=args,
                    writer=writer,
                )
                all_results[this_model_name][current_dataset_name] = metrics
            except Exception as e:
                print(
                    f"\n错误: 处理 {this_model_name} on {current_dataset_name} 时失败"
                )
                print(f"错误信息: {e}")
                import traceback

                traceback.print_exc()
                failed_evaluations.append((this_model_name, current_dataset_name))
                continue

    # Save the summary results.
    summary_file = os.path.join(args.out_dir, "summary_results.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总结果已保存到: {summary_file}")

    # Print the summary table.
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    print(f"{'模型':<30} {'数据集':<30} {'AUROC':<10} {'AUPRC':<10}")
    print("-" * 80)
    for mname, datasets in all_results.items():
        for dname, metrics in datasets.items():
            auroc = metrics.get("auroc", float("nan"))
            auprc = metrics.get("auprc", float("nan"))
            print(f"{mname:<30} {dname:<30} {auroc:<10.4f} {auprc:<10.4f}")
    print("=" * 80)

    writer.close()
    if failed_evaluations:
        raise RuntimeError(f"Evaluation failed for {failed_evaluations}; see errors above")
    print("\n所有任务完成!")
    print(f"结果保存在: {args.out_dir}")
    print(f"TensorBoard日志: {args.log_dir}")
    print(f"使用以下命令查看TensorBoard: tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()
