#!/usr/bin/env python3
"""Model-family scoring interfaces for the adaptive-allele preference benchmark.

The generic scorer is intentionally conservative: a model is scored only when
its checkpoint-backed likelihood head (or its official native inference API)
is available. No randomly initialized output head is accepted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from tokens import changed_block


SPECIAL_MODEL_IDS = frozenset(
    {
        "HyenaDNA_tiny_1k",
        "GPN-MSA",
        "GPN-promoter",
        "SpeciesLM",
        "AIDO.DNA-300M",
        "DNABERT",
        "evo2_1b_base",
        "evo2_7b_base",
    }
)


class AdapterError(RuntimeError):
    """A model cannot be scored under the declared benchmark protocol."""


class MaskedAdapter:
    kind = "masked"

    def __init__(self, model_id: str, model_dir: Path, device: torch.device):
        self.model_id = model_id
        self.model_dir = model_dir
        self.device = device
        self.tokenizer = None
        self.model = None
        self.summary: Dict[str, Any] = {
            "adapter_id": self.__class__.__name__,
            "model_id": model_id,
            "model_dir": str(model_dir),
            "kind": "masked",
        }

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _logits(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {
            k: v.to(self.device)
            for k, v in encoded.items()
            if k in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.inference_mode():
            try:
                output = self.model(**inputs, return_dict=True)
            except TypeError:
                inputs.pop("token_type_ids", None)
                output = self.model(**inputs, return_dict=True)
        if not hasattr(output, "logits"):
            raise AdapterError(f"{self.model_id}:model_output_has_no_logits")
        return output.logits[0].float()

    def score_pair(
        self, selected_sequence: str, other_sequence: str, limit: int
    ) -> Dict[str, Any]:
        selected = self.encode(selected_sequence)
        other = self.encode(other_sequence)
        selected_ids = selected["input_ids"][0]
        other_ids = other["input_ids"][0]
        if selected_ids.numel() > limit or other_ids.numel() > limit:
            raise AdapterError(f"{self.model_id}:tokenized_sequence_exceeds_limit")
        selected_diff, other_diff = changed_block(
            selected_ids.tolist(), other_ids.tolist()
        )
        special = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        selected_diff = [
            i for i in selected_diff if int(selected_ids[i]) not in special
        ]
        other_diff = [i for i in other_diff if int(other_ids[i]) not in special]
        if not selected_diff or not other_diff:
            raise AdapterError(f"{self.model_id}:no_tokenized_difference_detected")
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None:
            raise AdapterError(f"{self.model_id}:tokenizer_has_no_mask_token")
        selected_masked = selected_ids.clone()
        other_masked = other_ids.clone()
        for index in selected_diff:
            selected_masked[index] = int(mask_id)
        for index in other_diff:
            other_masked[index] = int(mask_id)
        selected_encoded = dict(selected)
        other_encoded = dict(other)
        selected_encoded["input_ids"] = selected_masked.unsqueeze(0)
        other_encoded["input_ids"] = other_masked.unsqueeze(0)
        selected_log_probs = torch.log_softmax(self._logits(selected_encoded), dim=-1)
        other_log_probs = torch.log_softmax(self._logits(other_encoded), dim=-1)
        selected_score = sum(
            float(selected_log_probs[i, int(selected_ids[i])]) for i in selected_diff
        )
        other_score = sum(
            float(other_log_probs[i, int(other_ids[i])]) for i in other_diff
        )
        return {
            "selected_logprob": selected_score,
            "other_logprob": other_score,
            "changed_token_positions": selected_diff,
            "masked_token_id": int(mask_id),
            "tokenized_length": int(selected_ids.numel()),
            "tokenization_note": (
                f"changed_token_positions_selected={','.join(map(str, selected_diff))};"
                f"other={','.join(map(str, other_diff))};"
                f"token_lengths={selected_ids.numel()},{other_ids.numel()}"
            ),
        }


class _HFHeadsummaryMixin:
    def _validate_loading(self, info: Dict[str, Any], allowed_missing: Tuple[str, ...]):
        missing = list(info.get("missing_keys", []))
        unexpected = list(info.get("unexpected_keys", []))
        real_missing = [x for x in missing if x not in allowed_missing]
        if real_missing or unexpected:
            raise AdapterError(
                f"{self.model_id}:checkpoint_loading_mismatch:"
                f"missing={real_missing[:12]}:unexpected={unexpected[:12]}"
            )
        self.summary.update(
            {
                "model_class": type(self.model).__name__,
                "tokenizer_class": type(self.tokenizer).__name__,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "allowed_missing_keys": list(allowed_missing),
                "checkpoint_backed_head": True,
            }
        )


class SpeciesLMAdapter(MaskedAdapter, _HFHeadsummaryMixin):
    """SpeciesLM uses overlapping six-mer tokens, matching the reference evaluator."""

    def __init__(self, model_id: str, model_dir: Path, device: torch.device):
        super().__init__(model_id, model_dir, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True, use_fast=True
        )
        self.model, info = AutoModelForMaskedLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            output_loading_info=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        self._validate_loading(info, ())
        self.model.to(device).eval()

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        kmers = " ".join(sequence[i : i + 6] for i in range(max(1, len(sequence) - 5)))
        return self.tokenizer(
            kmers, add_special_tokens=True, truncation=False, return_tensors="pt"
        )


class NucleotideTransformerAdapter(MaskedAdapter, _HFHeadsummaryMixin):
    """Use the checkpoint-backed MLM head for official Nucleotide Transformers."""

    def __init__(self, model_id: str, model_dir: Path, device: torch.device):
        super().__init__(model_id, model_dir, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True, use_fast=True
        )
        self.model, info = AutoModelForMaskedLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            output_loading_info=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        self._validate_loading(info, ())
        getter = getattr(self.model, "get_output_embeddings", None)
        head = getter() if callable(getter) else None
        weight = getattr(head, "weight", None)
        if weight is None or weight.ndim != 2:
            raise AdapterError(f"{model_id}:checkpoint_has_no_usable_lm_head")
        vocab_size = int(getattr(self.model.config, "vocab_size", weight.shape[0]))
        if int(weight.shape[0]) != vocab_size:
            raise AdapterError(
                f"{model_id}:lm_head_vocab_mismatch:{int(weight.shape[0])}!={vocab_size}"
            )
        if not torch.isfinite(weight.detach()).all().item():
            raise AdapterError(f"{model_id}:lm_head_contains_nonfinite_values")
        self.summary.update(
            {"head_name": "lm_head.decoder.weight", "vocab_size": vocab_size}
        )
        self.model.to(device).eval()

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            sequence, add_special_tokens=True, truncation=False, return_tensors="pt"
        )


class AIDOAdapter(MaskedAdapter, _HFHeadsummaryMixin):
    """Load AIDO's official RNABert implementation from ModelGenerator."""

    def __init__(self, model_id: str, model_dir: Path, device: torch.device):
        super().__init__(model_id, model_dir, device)
        try:
            from modelgenerator.huggingface_models.rnabert import (
                RNABertForMaskedLM,
                RNABertTokenizer,
                RNABertConfig,
            )
        except ImportError as exc:
            raise AdapterError(
                f"{model_id}:modelgenerator_rnabert_unavailable"
            ) from exc
        self.tokenizer = RNABertTokenizer(vocab_file=str(model_dir / "vocab.txt"))
        config = RNABertConfig.from_pretrained(str(model_dir))
        self.model, info = RNABertForMaskedLM.from_pretrained(
            str(model_dir),
            config=config,
            local_files_only=True,
            output_loading_info=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        # The official snapshot ties decoder weights to input embeddings and
        # omits decoder bias; the reference implementation creates it as zero.
        self._validate_loading(info, ("cls.predictions.decoder.bias",))
        self.model.to(device).eval()
        self.summary["tokenizer_protocol"] = "RNABertTokenizer; raw DNA characters"

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            sequence, add_special_tokens=True, truncation=False, return_tensors="pt"
        )


class HyenaDNAAdapter:
    """HyenaDNA complete HF CausalLM checkpoint and tokenizer."""

    kind = "causal"

    def __init__(self, model_id: str, model_dir: Path, device: torch.device):
        self.model_id = model_id
        self.model_dir = model_dir
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True, use_fast=True
        )
        self.model, loading = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            output_loading_info=True,
            torch_dtype=torch.float32,
        )
        missing = sorted(set(loading.get("missing_keys", [])))
        unexpected = sorted(set(loading.get("unexpected_keys", [])))
        mismatched = loading.get("mismatched_keys", [])
        if missing or unexpected or mismatched:
            raise AdapterError(
                f"{model_id}:checkpoint_loading_mismatch:"
                f"missing={missing[:12]}:unexpected={unexpected[:12]}:"
                f"mismatched={mismatched[:12]}"
            )
        head = getattr(self.model, "lm_head", None)
        head_weight = getattr(head, "weight", None)
        if head_weight is None:
            raise AdapterError(f"{model_id}:checkpoint_has_no_lm_head_weight")
        if not torch.isfinite(head_weight.detach()).all().item():
            raise AdapterError(f"{model_id}:lm_head_contains_nonfinite_values")
        self.model.to(device).eval()
        config = getattr(self.model, "config", None)
        self.model_limit = int(
            getattr(config, "max_seq_len", None)
            or getattr(config, "max_position_embeddings", None)
            or 10**9
        )
        self.special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        self.summary = {
            "adapter_id": self.__class__.__name__,
            "model_id": model_id,
            "model_dir": str(model_dir),
            "kind": "causal",
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "mismatched_keys": mismatched,
            "checkpoint_backed_head": True,
            "head_name": "lm_head.weight",
            "special_tokens_reserved": self.special_tokens,
            "max_sequence_tokens": self.model_limit,
        }

    def _encode(self, sequence: str) -> torch.Tensor:
        encoded = self.tokenizer(
            sequence, add_special_tokens=True, truncation=False, return_tensors="pt"
        )
        ids = encoded["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise AdapterError(
                f"{self.model_id}:unexpected_tokenizer_shape:{tuple(ids.shape)}"
            )
        if ids.shape[1] != len(sequence) + self.special_tokens:
            raise AdapterError(
                f"{self.model_id}:unexpected_token_length:bp={len(sequence)};"
                f"tokens={ids.shape[1]};special_tokens={self.special_tokens}"
            )
        if ids.shape[1] > self.model_limit:
            raise AdapterError(
                f"{self.model_id}:tokenized_sequence_exceeds_checkpoint_limit:"
                f"tokens={ids.shape[1]};limit={self.model_limit}"
            )
        return ids[0].to(self.device)

    def _sequence_log_likelihood(self, sequence: str) -> Tuple[float, torch.Tensor]:
        ids = self._encode(sequence)
        with torch.inference_mode():
            outputs = self.model(input_ids=ids.unsqueeze(0), return_dict=True)
        logits = outputs.logits[0].float()
        if logits.shape[0] != ids.shape[0]:
            raise AdapterError(
                f"{self.model_id}:logits_length_mismatch:"
                f"logits={logits.shape[0]};tokens={ids.shape[0]}"
            )
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
        score = log_probs.gather(1, ids[1:].unsqueeze(1)).sum()
        return float(score), ids

    def score_pair(
        self, selected_sequence: str, other_sequence: str, limit: int
    ) -> Dict[str, Any]:
        selected_score, selected_ids = self._sequence_log_likelihood(selected_sequence)
        other_score, other_ids = self._sequence_log_likelihood(other_sequence)
        effective_limit = min(int(limit), self.model_limit)
        if (
            selected_ids.numel() > effective_limit
            or other_ids.numel() > effective_limit
        ):
            raise AdapterError(
                f"{self.model_id}:tokenized_sequence_exceeds_limit:limit={effective_limit}"
            )
        selected_diff, _ = changed_block(selected_ids.tolist(), other_ids.tolist())
        if not selected_diff:
            raise AdapterError(f"{self.model_id}:no_tokenized_difference_detected")
        return {
            "selected_logprob": selected_score,
            "other_logprob": other_score,
            "changed_token_positions": selected_diff,
            "masked_token_id": None,
            "tokenized_length": int(selected_ids.numel()),
            "tokenization_note": (
                "full_sequence_loglikelihood; HyenaDNA byte-level DNA tokenizer;"
                f"special_tokens={self.special_tokens}"
            ),
        }


class GPNPromoterAdapter(MaskedAdapter, _HFHeadsummaryMixin):
    """GPN promoter checkpoint using the official native GPN masked-LM head."""

    def __init__(self, model_id: str, model_dir: Path, device: torch.device):
        super().__init__(model_id, model_dir, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True, use_fast=True
        )
        self.model, info = AutoModelForMaskedLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            output_loading_info=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        self._validate_loading(info, ())
        head = getattr(getattr(self.model, "cls", None), "decoder", None)
        head_weight = getattr(head, "weight", None)
        if head_weight is None:
            raise AdapterError(f"{model_id}:checkpoint_has_no_cls_decoder_weight")
        if not torch.isfinite(head_weight.detach()).all().item():
            raise AdapterError(f"{model_id}:cls_decoder_contains_nonfinite_values")
        self.summary.update(
            {
                "head_name": "cls.decoder.weight",
                "official_model_family": "GPNForMaskedLM",
            }
        )
        self.model.to(device).eval()

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            sequence, add_special_tokens=True, truncation=False, return_tensors="pt"
        )


class GPNMSAAdapter(MaskedAdapter):
    """GPN-MSA requires the published aligned-MSA inference input."""

    def __init__(self, model_id: str, model_dir: Path, device: torch.device):
        super().__init__(model_id, model_dir, device)
        raise AdapterError(
            "GPN-MSA:msa_alignment_required:provide a compatible 100-way MSA zarr;"
            " single-sequence input is not the published GPN-MSA inference protocol"
        )


class NativeCausalAdapter:
    kind = "causal"

    def __init__(self, model_id: str, model_dir: Path, device: torch.device):
        from evo2 import Evo2
        import evo2.utils as evo2_utils

        self.model_id = model_id
        self.model_dir = model_dir
        self.device = device
        checkpoint_name = (
            "evo2_1b_base" if model_id == "evo2_1b_base" else "evo2_7b_base"
        )
        if not model_dir.is_file():
            raise AdapterError(
                f"{model_id}:native_checkpoint_path_is_not_file:{model_dir}"
            )
        config_override = os.environ.get("EVO2_CONFIG_PATH", "").strip()
        if config_override:
            config_path = Path(config_override)
            if not config_path.is_file():
                raise AdapterError(
                    f"{model_id}:evo2_config_override_missing:{config_path}"
                )
            evo2_utils.CONFIG_MAP[checkpoint_name] = str(config_path)
        self.model = Evo2(checkpoint_name, str(model_dir))
        eval_target = self.model
        while not hasattr(eval_target, "eval") and hasattr(eval_target, "model"):
            eval_target = eval_target.model
        if hasattr(eval_target, "eval"):
            eval_target.eval()
        self.tokenizer = self.model.tokenizer.tokenize_batch
        self.summary = {
            "adapter_id": self.__class__.__name__,
            "model_id": model_id,
            "model_dir": str(model_dir),
            "kind": "causal",
            "model_class": type(self.model).__name__,
            "tokenizer_protocol": "Evo2 native byte tokenizer",
            "checkpoint_backed_head": True,
            "config_path": evo2_utils.CONFIG_MAP[checkpoint_name],
            "evo2_config_override": bool(config_override),
        }

    def _encode(self, sequence: str) -> torch.Tensor:
        ids = torch.tensor(self.tokenizer([sequence.encode("ascii")]), dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return ids.to(self.device)

    def _sequence_log_likelihood(self, sequence: str) -> Tuple[float, torch.Tensor]:
        ids = self._encode(sequence)
        with torch.inference_mode():
            outputs, _ = self.model(ids)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        logits = outputs[0] if outputs.ndim == 3 else outputs
        target = ids[0]
        if logits.shape[0] != target.shape[0]:
            raise AdapterError(f"{self.model_id}:native_logits_length_mismatch")
        log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
        score = log_probs.gather(1, target[1:].unsqueeze(1)).sum()
        return float(score), target

    def score_pair(
        self, selected_sequence: str, other_sequence: str, limit: int
    ) -> Dict[str, Any]:
        selected_score, selected_ids = self._sequence_log_likelihood(selected_sequence)
        other_score, other_ids = self._sequence_log_likelihood(other_sequence)
        if selected_ids.numel() > limit or other_ids.numel() > limit:
            raise AdapterError(f"{self.model_id}:tokenized_sequence_exceeds_limit")
        selected_diff, _ = changed_block(selected_ids.tolist(), other_ids.tolist())
        if not selected_diff:
            raise AdapterError(f"{self.model_id}:no_tokenized_difference_detected")
        return {
            "selected_logprob": selected_score,
            "other_logprob": other_score,
            "changed_token_positions": selected_diff,
            "masked_token_id": None,
            "tokenized_length": int(selected_ids.numel()),
            "tokenization_note": "full_sequence_loglikelihood; native Evo2 byte tokenizer",
        }


def build_adapter(model_id: str, model_dir: Path, kind: str, device: torch.device):
    if model_id in {"NT_500m_1kG", "NT_2.5b_1kG"}:
        if kind != "masked":
            raise AdapterError(f"{model_id}:expected_masked_kind")
        return NucleotideTransformerAdapter(model_id, model_dir, device)
    if model_id == "GPN-promoter":
        return GPNPromoterAdapter(model_id, model_dir, device)
    if model_id == "HyenaDNA_tiny_1k":
        if kind != "causal":
            raise AdapterError(f"{model_id}:expected_causal_kind")
        return HyenaDNAAdapter(model_id, model_dir, device)
    if model_id == "SpeciesLM":
        return SpeciesLMAdapter(model_id, model_dir, device)
    if model_id == "AIDO.DNA-300M":
        return AIDOAdapter(model_id, model_dir, device)
    if model_id == "GPN-MSA":
        return GPNMSAAdapter(model_id, model_dir, device)
    if model_id in {"evo2_1b_base", "evo2_7b_base"}:
        return NativeCausalAdapter(model_id, model_dir, device)
    if model_id in SPECIAL_MODEL_IDS:
        raise AdapterError(
            f"{model_id}:checkpoint_has_no_supported_likelihood_head;"
            " inspect full official checkpoint before scoring"
        )
    return None
