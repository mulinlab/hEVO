#!/usr/bin/env python3
"""Score non-hEvo roster models without legacy selection inputs.

MLM scoring masks every token whose tokenized representation changes between
the two allele sequences. This avoids assuming DNA position == token index.
Causal scoring compares full-sequence token log likelihoods.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser():
    p = argparse.ArgumentParser(
        description="Independent adaptive-allele scoring for one HF model."
    )
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--kind", choices=["masked", "causal"], required=True)
    p.add_argument("--loci", type=Path, required=True)
    p.add_argument("--reference-fasta", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window-bp", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-loci", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    return p


# Help uses the same parser as execution, before optional model imports.
if __name__ == "__main__" and any(flag in sys.argv[1:] for flag in ("-h", "--help")):
    build_parser().parse_args()

import csv  # noqa: E402 - lightweight CLI help precedes runtime imports
import json  # noqa: E402 - lightweight CLI help precedes runtime imports
import math  # noqa: E402 - lightweight CLI help precedes runtime imports
from typing import Dict, List  # noqa: E402 - lightweight CLI help precedes runtime imports
import pysam  # noqa: E402 - lightweight CLI help precedes runtime imports
import torch  # noqa: E402 - lightweight CLI help precedes runtime imports
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer  # noqa: E402 - lightweight CLI help precedes runtime imports

from model_adapters import build_adapter  # noqa: E402 - lightweight CLI help precedes runtime imports
from tokens import changed_block  # noqa: E402 - lightweight CLI help precedes runtime imports


def parse_args():
    parser = build_parser()
    return parser.parse_args()


def loci(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h, delimiter="\t"))
    if not rows:
        raise ValueError("empty loci")
    for r in rows:
        if len(r["selected_allele"]) != 1 or len(r["other_allele"]) != 1:
            raise ValueError(f"{r['variant_id']}: non-SNV")
    return rows


def encode(tok, seq, limit):
    enc = tok(
        seq,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    out = {}
    for key, value in list(enc.items()):
        if isinstance(value, list):
            if value and isinstance(value[0], int):
                value = [value]
            out[key] = torch.tensor(value, dtype=torch.long)
        else:
            out[key] = (
                value.unsqueeze(0)
                if torch.is_tensor(value) and value.ndim == 1
                else value
            )
    ids = out["input_ids"][0]
    if ids.numel() > limit:
        raise ValueError(
            f"tokenized sequence length {ids.numel()} exceeds model limit {limit}"
        )
    return out


def model_logits(model, ids, attention_mask):
    try:
        out = model(
            input_ids=ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            return_dict=True,
        )
    except TypeError as exc:
        if "attention_mask" not in str(exc):
            raise
        out = model(input_ids=ids.unsqueeze(0), return_dict=True)
    return out.logits[0]


def score_masked(model, tok, ref_seq, other_seq, limit):
    er, eo = encode(tok, ref_seq, limit), encode(tok, other_seq, limit)
    ir, io = er["input_ids"][0], eo["input_ids"][0]
    diff_r, diff_o = changed_block(ir.tolist(), io.tolist())
    special = set(getattr(tok, "all_special_ids", []) or [])
    diff_r = [i for i in diff_r if int(ir[i]) not in special]
    diff_o = [i for i in diff_o if int(io[i]) not in special]
    if not diff_r or not diff_o:
        raise ValueError("no tokenized difference detected at candidate center")
    mask_id = getattr(tok, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("tokenizer has no mask token")
    masked_r = ir.clone().to(model.device)
    masked_o = io.clone().to(model.device)
    for i in diff_r:
        masked_r[i] = int(mask_id)
    for i in diff_o:
        masked_o[i] = int(mask_id)
    logp_r = torch.log_softmax(
        model_logits(model, masked_r, torch.ones_like(masked_r)).float(), dim=-1
    )
    logp_o = torch.log_softmax(
        model_logits(model, masked_o, torch.ones_like(masked_o)).float(), dim=-1
    )
    ref_score = sum(float(logp_r[i, int(ir[i])]) for i in diff_r)
    other_score = sum(float(logp_o[i, int(io[i])]) for i in diff_o)
    return ref_score, other_score, diff_r, diff_o, int(mask_id), len(ir), len(io)


def score_causal(model, tok, ref_seq, other_seq, limit):
    er, eo = encode(tok, ref_seq, limit), encode(tok, other_seq, limit)
    ir, io = er["input_ids"][0].to(model.device), eo["input_ids"][0].to(model.device)
    ar = torch.ones_like(ir)
    ao = torch.ones_like(io)
    with torch.inference_mode():
        lr = model_logits(model, ir, ar).float()
        lo = model_logits(model, io, ao).float()

    def total(logits, ids):
        return float(
            torch.log_softmax(logits[:-1], dim=-1).gather(1, ids[1:].unsqueeze(1)).sum()
        )

    return total(lr, ir), total(lo, io), [], None, int(ir.numel())


def main():
    a = parse_args()
    rows = loci(a.loci)
    rows = rows[: a.max_loci] if a.max_loci else rows
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(a.device)
    dtype = torch.bfloat16 if a.model_id.startswith("hEvo") else torch.float32
    adapter = build_adapter(a.model_id, a.model_dir, a.kind, device)
    if adapter is not None:
        return run_adapter(a, rows, adapter)
    tok = AutoTokenizer.from_pretrained(
        str(a.model_dir), trust_remote_code=True, local_files_only=True, use_fast=True
    )
    if a.kind == "masked":
        model, loading = AutoModelForMaskedLM.from_pretrained(
            str(a.model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            output_loading_info=True,
        )
    else:
        model, loading = AutoModelForCausalLM.from_pretrained(
            str(a.model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            output_loading_info=True,
        )
    missing = loading.get("missing_keys", [])
    if missing:
        raise RuntimeError("CHECKPOINT_MISSING_KEYS:" + ",".join(missing[:12]))
    model.to(device).eval()
    cfg = getattr(model, "config", None)
    limit = int(getattr(cfg, "max_position_embeddings", 10**9) or 10**9)
    results = []
    window = a.window_bp
    center = window // 2
    with pysam.FastaFile(str(a.reference_fasta)) as fasta:
        for r in rows:
            pos = int(r["pos"])
            start = pos - 1 - center
            ref_seq = fasta.fetch(r["chrom"], start, start + window).upper()
            if len(ref_seq) != window or ref_seq[center] != r.get(
                "vcf_ref", ref_seq[center]
            ):
                raise ValueError(f"{r['variant_id']}: reference mismatch")
            other_seq = (
                ref_seq[:center] + r["other_allele"].upper() + ref_seq[center + 1 :]
            )
            selected_seq = (
                ref_seq[:center] + r["selected_allele"].upper() + ref_seq[center + 1 :]
            )
            try:
                if a.kind == "masked":
                    sl, ol, diff_r, diff_o, mask_id, tok_len_r, tok_len_o = (
                        score_masked(model, tok, selected_seq, other_seq, limit)
                    )
                    diff = diff_r
                    tok_len = tok_len_r
                    tokenization_note = f"changed_token_positions_selected={','.join(map(str, diff_r))};other={','.join(map(str, diff_o))};token_lengths={tok_len_r},{tok_len_o}"
                else:
                    sl, ol, diff, mask_id, tok_len = score_causal(
                        model, tok, selected_seq, other_seq, limit
                    )
                    tokenization_note = "full_sequence_loglikelihood"
                score = sl - ol
                if not math.isfinite(score):
                    raise ValueError("nonfinite score")
                results.append(
                    {
                        **r,
                        "model_id": a.model_id,
                        "model_type": a.kind,
                        "window_bp": str(window),
                        "center_index": str(center),
                        "tokenized_length": str(tok_len),
                        "masked_token_id": "" if mask_id is None else str(mask_id),
                        "changed_token_positions": ",".join(map(str, diff)),
                        "selected_logprob": f"{sl:.10g}",
                        "other_logprob": f"{ol:.10g}",
                        "locus_score": f"{score:.10g}",
                        "tokenization_note": tokenization_note,
                        "status": "PASS",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        **r,
                        "model_id": a.model_id,
                        "model_type": a.kind,
                        "window_bp": str(window),
                        "center_index": str(center),
                        "selected_logprob": "",
                        "other_logprob": "",
                        "locus_score": "",
                        "tokenization_note": type(exc).__name__ + ":" + str(exc),
                        "status": "FAIL",
                    }
                )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cols = list(results[0])
    with a.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(results)
    manifest = {
        "model_id": a.model_id,
        "model_dir": str(a.model_dir),
        "kind": a.kind,
        "window_bp": window,
        "center_index": center,
        "loci_count": len(results),
        "pass_count": sum(x["status"] == "PASS" for x in results),
        "tokenizer_class": tok.__class__.__name__,
        "max_position_embeddings": limit,
        "legacy_selection_inputs_used": False,
        "model_class": model.__class__.__name__,
        "checkpoint_backed_head": True,
        "missing_keys": loading.get("missing_keys", []),
        "unexpected_keys": loading.get("unexpected_keys", []),
        "score_definition": "MLM changed-token masked logprob difference"
        if a.kind == "masked"
        else "causal full-sequence loglikelihood difference",
    }
    a.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["pass_count"] == len(results) else 2


def run_adapter(a, rows, adapter):
    window = a.window_bp
    center = window // 2
    cfg = getattr(getattr(adapter, "model", None), "config", None)
    limit = int(
        getattr(cfg, "max_seq_len", None)
        or getattr(cfg, "max_position_embeddings", None)
        or 10**9
    )
    results = []
    with pysam.FastaFile(str(a.reference_fasta)) as fasta:
        for r in rows:
            pos = int(r["pos"])
            start = pos - 1 - center
            ref_seq = fasta.fetch(r["chrom"], start, start + window).upper()
            if len(ref_seq) != window or ref_seq[center] != r.get(
                "vcf_ref", ref_seq[center]
            ):
                raise ValueError(f"{r['variant_id']}: reference mismatch")
            other_seq = (
                ref_seq[:center] + r["other_allele"].upper() + ref_seq[center + 1 :]
            )
            selected_seq = (
                ref_seq[:center] + r["selected_allele"].upper() + ref_seq[center + 1 :]
            )
            try:
                special = adapter.score_pair(selected_seq, other_seq, limit)
                sl, ol = special["selected_logprob"], special["other_logprob"]
                score = sl - ol
                if not math.isfinite(score):
                    raise ValueError("nonfinite score")
                results.append(
                    {
                        **r,
                        "model_id": a.model_id,
                        "model_type": adapter.kind,
                        "window_bp": str(window),
                        "center_index": str(center),
                        "tokenized_length": str(special["tokenized_length"]),
                        "masked_token_id": ""
                        if special["masked_token_id"] is None
                        else str(special["masked_token_id"]),
                        "changed_token_positions": ",".join(
                            map(str, special["changed_token_positions"])
                        ),
                        "selected_logprob": f"{sl:.10g}",
                        "other_logprob": f"{ol:.10g}",
                        "locus_score": f"{score:.10g}",
                        "tokenization_note": special["tokenization_note"],
                        "status": "PASS",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        **r,
                        "model_id": a.model_id,
                        "model_type": adapter.kind,
                        "window_bp": str(window),
                        "center_index": str(center),
                        "selected_logprob": "",
                        "other_logprob": "",
                        "locus_score": "",
                        "tokenization_note": type(exc).__name__ + ":" + str(exc),
                        "status": "FAIL",
                    }
                )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cols = list(results[0])
    with a.output.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(results)
    manifest = {
        "model_id": a.model_id,
        "model_dir": str(a.model_dir),
        "kind": adapter.kind,
        "window_bp": window,
        "center_index": center,
        "loci_count": len(results),
        "pass_count": sum(x["status"] == "PASS" for x in results),
        "tokenizer_class": type(adapter.tokenizer).__name__
        if adapter.tokenizer is not None
        else "native",
        "max_position_embeddings": limit,
        "legacy_selection_inputs_used": False,
        "score_definition": "MLM changed-token masked logprob difference"
        if adapter.kind == "masked"
        else "causal full-sequence loglikelihood difference",
        "adapter_summary": adapter.summary,
    }
    a.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["pass_count"] == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
