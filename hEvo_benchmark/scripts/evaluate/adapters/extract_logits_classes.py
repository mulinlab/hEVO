from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from enformer_pytorch import Enformer, seq_indices_to_one_hot
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from adapters.native_variant_scorers import (
    generator_base_token_ids,
    generator_score_batch,
    mistral_score_batch,
)


def validate_pretrained_lm_head(model, loading_info, tokenizer, *, allow_unused_pooler=False) -> None:
    """Fail closed unless a complete, finite pretrained output head was loaded."""

    missing = sorted(set(loading_info.get("missing_keys", [])))
    unexpected = sorted(set(loading_info.get("unexpected_keys", [])))
    if allow_unused_pooler:
        unused_pooler = {"bert.pooler.dense.weight", "bert.pooler.dense.bias"}
        unexpected = [key for key in unexpected if key not in unused_pooler]
    mismatched = loading_info.get("mismatched_keys", [])
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Checkpoint is not a complete language model: "
            f"missing_keys={missing}; unexpected_keys={unexpected}; "
            f"mismatched_keys={mismatched}"
        )
    get_output = getattr(model, "get_output_embeddings", None)
    head = get_output() if callable(get_output) else None
    if head is None:
        # Some custom Transformers-compatible MLMs (including hEvo's
        # FlashBertForMaskedLM) expose their pretrained head as ``cls`` but do
        # not implement the standard accessor.
        cls_head = getattr(model, "cls", None)
        predictions = getattr(cls_head, "predictions", None)
        head = getattr(predictions, "decoder", None)
        if head is None:
            pretrain_head = getattr(model, "pretrain_head", None)
            head = getattr(pretrain_head, "mlm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None or weight.ndim != 2:
        raise RuntimeError("Checkpoint has no usable pretrained LM output head")
    if not torch.isfinite(weight.detach()).all().item():
        raise RuntimeError("LM output-head weights contain non-finite values")
    output_vocab = int(weight.shape[0])
    config_vocab = int(getattr(model.config, "vocab_size", output_vocab))
    if output_vocab != config_vocab:
        raise RuntimeError(
            "LM output-head vocabulary mismatch: "
            f"head={output_vocab}, config={config_vocab}"
        )
    # Some official tokenizers declare non-emitted BOS/EOS entries beyond the
    # model vocabulary (NT-500M-1000G has IDs 4105/4106 with a 4105-way head).
    # Validate IDs that can actually enter the benchmark instead of rejecting
    # a usable upstream checkpoint solely because len(tokenizer) is larger.
    probe = tokenizer(
        ["A", "C", "G", "T", "N", "ACGTN", "ACGTACGTACGT"],
        add_special_tokens=True,
    )
    emitted_ids = {int(token_id) for row in probe["input_ids"] for token_id in row}
    if emitted_ids and max(emitted_ids) >= output_vocab:
        raise RuntimeError(
            "Tokenizer emits IDs outside the LM output vocabulary: "
            f"max_emitted_id={max(emitted_ids)}, head={output_vocab}"
        )


# ---------- Extractor ----------
class ExtractLogitBase(object):
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model, loading_info = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True, output_loading_info=True
        )
        validate_pretrained_lm_head(
            self.model, loading_info, self.tokenizer,
            allow_unused_pooler=model_name == "GROVER",
        )
        self.model.eval()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError

    def get_diff_token(
        self, ref_seq: str, alt_seq: str
    ) -> Tuple[int, int, int, str, str] | None:
        """
        接受两个碱基序列，找出第一个不同的token位置和值
        如果序列完全相同，返回 None 而不是抛出异常。
        """
        ref_tokens = self.tokenizer.encode(ref_seq, add_special_tokens=True)
        alt_tokens = self.tokenizer.encode(alt_seq, add_special_tokens=True)

        # Find the first differing token.
        for i, (ref_token, alt_token) in enumerate(zip(ref_tokens, alt_tokens)):
            if ref_token != alt_token:
                # Decode the token to obtain the actual character.
                ref_char = self.tokenizer.decode([ref_token])
                alt_char = self.tokenizer.decode([alt_token])
                return i, ref_token, alt_token, ref_char, alt_char

        # [Change] Return None instead of raising ValueError.
        print("Warning: No difference found between ref and alt sequences. Skipping...")
        return None

    def get_masked_variant_logits(
        self,
        ref_input_ids: torch.Tensor,
        alt_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Predict the changed token from the reference context with it masked.

        This is the masked-marginal score input used for MLM variants. All
        tokens changed by a variant are masked together in the reference
        context. This supports both single-token DNA vocabularies and
        overlapping subword/k-mer tokenizers.
        """
        if ref_input_ids.ndim != 2 or alt_input_ids.ndim != 2:
            raise ValueError("Variant inputs must have shape (batch, sequence)")
        if ref_input_ids.shape != alt_input_ids.shape:
            raise ValueError(
                "Ref/alt tokenized shapes differ: "
                f"ref={tuple(ref_input_ids.shape)}, alt={tuple(alt_input_ids.shape)}"
            )

        if attention_mask is None:
            pad_token_id = self.tokenizer.pad_token_id
            attention_mask = (
                torch.ones_like(ref_input_ids)
                if pad_token_id is None
                else ref_input_ids.ne(pad_token_id).long()
            )
        if alt_attention_mask is not None and not torch.equal(
            attention_mask.to(alt_attention_mask.device), alt_attention_mask
        ):
            raise ValueError("Ref/alt attention masks differ")

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise RuntimeError(
                f"{self.model_name} tokenizer does not define a mask token"
            )

        changed_tokens = ref_input_ids.ne(alt_input_ids) & attention_mask.bool()
        changed_counts = changed_tokens.sum(dim=1)
        if not torch.all(changed_counts >= 1):
            raise ValueError(
                "Each SNV must change at least one active token; "
                f"observed per-sample token differences={changed_counts.detach().cpu().tolist()}"
            )
        if torch.any(changed_counts > 6):
            raise ValueError(
                "A single SNV changed more than six tokenizer tokens; token alignment "
                "is not reliable for masked-marginal scoring: "
                f"observed per-sample token differences={changed_counts.detach().cpu().tolist()}"
            )

        masked_ref_ids = ref_input_ids.clone()
        masked_ref_ids[changed_tokens] = int(mask_token_id)
        return self.get_logit(
            input_ids=masked_ref_ids,
            attention_mask=attention_mask,
            return_numpy=return_numpy,
        )

    def calc_llr(
        self, logits: torch.Tensor, position: int, ref_token: int, alt_token: int
    ) -> float:
        """
        计算对数似然比 (LLR)。
        公式: log_softmax(logits)[alt_token] - log_softmax(logits)[ref_token]
        """
        # for t in logits:
        #     print(t.item(), end=" ")
        # print()
        # Handle the batch dimension.
        if logits.ndim == 3:
            # Assume logits has shape (Batch, Seq, Vocab) with batch_size=1.
            # Alternatively, logits may already correspond to this sample.
            token_logits = logits[0, position, :]
        elif logits.ndim == 2:
            # (Seq, Vocab)
            token_logits = logits[position, :]
        else:
            token_logits = logits

        # Use log_softmax to compute log probabilities.
        token_logits = token_logits.to(torch.float32)
        log_probs = torch.log_softmax(token_logits, dim=-1)

        # Extract and subtract the log probabilities for ref and alt.
        llr = log_probs[alt_token] - log_probs[ref_token]

        if not self.model_name.startswith("hEvo-"):
            print(
                f"Position: {position}, Ref: {ref_token}, Alt: {alt_token}, LLR: {llr.item()}"
            )
        return llr.item()


class ExtractLogitForDNABert(ExtractLogitBase):
    """DNABERT模型的Logit提取器 (修复了MosaicBERT ALiBi维度问题)"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=512,
            padding="max_length",
            truncation=truncation,
            return_tensors=return_tensors,
            # [Change] Critical fix.
            # DNABERT-S (MosaicBERT) uses ALiBi positional encoding.
            # Sequence length must be a multiple of 8 to avoid a dimension mismatch.
            # pad_to_multiple_of=32,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits  # (batch_size, seq_length, vocab_size)

        if return_numpy:
            return logits.cpu().numpy()
        return logits


class ExtractLogitForAIDO(ExtractLogitBase):
    """AIDO.DNA character-level masked LM extractor."""

    def __init__(self, model_path: str, model_name: str):
        # AIDO.DNA is an RNABert checkpoint with SwiGLU. Transformers' generic
        # Bert loader cannot instantiate it; use GenBio's official backbone
        # implementation and request its pretrained MLM adapter explicitly.
        from modelgenerator.backbones import aido_dna_300m
        from modelgenerator.backbones.base import LegacyAdapterType

        self.model_path = model_path
        self.model_name = model_name
        local_backbone = type(
            "LocalAIDODNA300M", (aido_dna_300m,), {"model_path": model_path}
        )
        self.model = local_backbone(LegacyAdapterType.MASKED_LM, None)
        self.tokenizer = self.model.tokenizer
        mlm_head = self.model.decoder.predictions
        # RNABert stores the learned output bias as ``cls.predictions.bias``;
        # Transformers 4.57 leaves the decoder alias freshly initialized while
        # loading the parent bias. Rebind it explicitly to the checkpoint
        # parameter instead of using the uninitialized alias.
        mlm_head.decoder.bias = mlm_head.bias
        if mlm_head.decoder.bias is not mlm_head.bias:
            raise RuntimeError(
                "AIDO MLM decoder bias is not tied to the pretrained bias"
            )
        if (
            not torch.isfinite(mlm_head.decoder.weight.detach()).all().item()
            or not torch.isfinite(mlm_head.bias.detach()).all().item()
        ):
            raise RuntimeError(
                "AIDO pretrained MLM head contains non-finite parameters"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # The AIDO vocabulary contains individual nucleotide symbols; its
        # WordPiece tokenizer requires whitespace between them.
        spaced = [" ".join(text.upper()) for text in texts]
        return self.tokenizer(
            spaced,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, np.ndarray]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            outputs = self.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = self.model.decoder(outputs.last_hidden_state)
        return logits.cpu().numpy() if return_numpy else logits


class ExtractLogitForhEvo(ExtractLogitBase):
    """hEvo模型的Logit提取器"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs

        if return_numpy:
            return logits.cpu().numpy()
        return logits


class ExtractLogitForhEvo_rare(ExtractLogitBase):
    """hEvo模型的Logit提取器"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            # 1. Run the backbone with bf16 automatic mixed precision.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # This applies only inside the model; it assumes model.py does not force a dtype conversion in forward.
                outputs = self.model.bert(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )

            # 2. The classifier layer computation is promoted to fp32 because its weights are fp32.
            # Explicitly keep all computations in fp32 from this point onward.
            last_hidden_states = outputs.last_hidden_state.to(torch.float32)
            # Current hEvo checkpoints expose ``cls``; the earlier
            # multi-task MLM checkpoint exposes the equivalent head as
            # ``pretrain_head.mlm_head``.  Keep the same logits-only path for
            # both layouts so legacy self-supervised checkpoints can be
            # checked without changing score semantics.
            if hasattr(self.model, "cls"):
                logits = self.model.cls(last_hidden_states)
            elif hasattr(self.model, "pretrain_head") and hasattr(
                self.model.pretrain_head, "mlm_head"
            ):
                logits = self.model.pretrain_head.mlm_head(last_hidden_states)
            else:
                raise AttributeError(
                    "hEvo checkpoint exposes neither cls nor pretrain_head.mlm_head"
                )

        return logits if not return_numpy else logits.cpu().numpy()

    def logit_to_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """针对 BERT 模型的 Log-prob 提取，取消偏移"""
        # Compute [Seq, Vocab] values in fp32 for numerical accuracy.
        softmax_logprobs = torch.log_softmax(logits.float(), dim=-1)

        # [Core fix] BERT does not require the [:-1] and [1:] position shifts.
        # Extract log_prob at the corresponding positions directly.
        logprobs = softmax_logprobs[torch.arange(len(input_ids)), input_ids]

        return logprobs

    def calc_variant_llr(self, ref_seq: str, alt_seq: str) -> float:
        """Sum ALT-minus-REF log probabilities from target-masked context."""
        ref = self.tokenize([ref_seq], max_length=1024, return_tensors="pt")
        alt = self.tokenize([alt_seq], max_length=1024, return_tensors="pt")
        logits = self.get_masked_variant_logits(
            ref["input_ids"],
            alt["input_ids"],
            attention_mask=ref.get("attention_mask"),
            alt_attention_mask=alt.get("attention_mask"),
        )
        changed = ref["input_ids"].ne(alt["input_ids"])
        if ref.get("attention_mask") is not None:
            changed &= ref["attention_mask"].bool()
        positions = torch.where(changed[0])[0]
        log_probs = torch.log_softmax(
            logits[0, positions.to(logits.device)].float(), dim=-1
        )
        refs = ref["input_ids"][0, positions].to(log_probs.device)
        alts = alt["input_ids"][0, positions].to(log_probs.device)
        return float(
            (
                log_probs.gather(1, alts[:, None]).sum()
                - log_probs.gather(1, refs[:, None]).sum()
            ).item()
        )


class ExtractLogitForhEvo_full(ExtractLogitForhEvo_rare):
    """hEvo extractor that preserves ordinary model post-backbone modules."""

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )
        logits = outputs.logits
        return logits if not return_numpy else logits.cpu().numpy()


class ExtractLogitForhEvo_stage2(ExtractLogitBase):
    """hEvo模型的Logit提取器"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            self.model = self.model.to(torch.bfloat16)
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits

        if return_numpy:
            return logits.cpu().numpy()
        return logits


class ExtractLogitForhEvo_rand(ExtractLogitBase):
    """hEvo模型的Logit提取器"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            self.model = self.model.to(torch.bfloat16)
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def logit_to_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (length, vocab)
        input_ids: (length,)
        返回每个位置对应 token 的 log-prob，形状为 (length,)
        """
        # Apply log_softmax along the vocabulary dimension.
        softmax_logprobs = torch.log_softmax(logits, dim=-1)

        # As in the batched version, align predictions with targets by predicting the token at the next position.
        softmax_logprobs = softmax_logprobs[:-1]  # (length-1, vocab)
        input_ids = input_ids[1:]  # (length-1,)

        assert softmax_logprobs.shape[0] == input_ids.shape[0]

        # Retrieve the corresponding log probability by token ID.
        logprobs = softmax_logprobs[torch.arange(len(input_ids)), input_ids]

        return logprobs  # (length-1,)

    def calc_score(
        self,
        logits: torch.Tensor,
        input_id,
        reduce_method="mean",
    ) -> float:
        """
        计算对数似然比 (LLR)。
        """

        logprobs = self.logit_to_logprobs(logits, input_id)
        logprobs = logprobs.float().cpu().numpy()

        if reduce_method == "sum":  # PLL
            reduce_func = np.sum
        elif reduce_method == "mean":  # mean PLL
            reduce_func = np.mean
        else:
            raise ValueError(f"Invalid reduce_method {reduce_method}")
        score: float = reduce_func(logprobs[: len(input_id)])
        return score


class ExtractLogitForNT(ExtractLogitBase):
    """Nucleotide Transformer MLM extractor with masked-marginal variant logits.

    NT v1/v2 tokenizes DNA into non-overlapping 6-mers. For a single-base
    substitution, exactly one sequence token should change. Variant scoring
    must predict that token from its context, so the changed reference token
    is replaced by the tokenizer's mask token before the forward pass.
    """

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def get_masked_variant_logits(
        self,
        ref_input_ids: torch.Tensor,
        alt_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        alt_attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Return reference-context logits with each changed NT token masked.

        Inputs are paired tokenizer outputs. The check is deliberately strict:
        one SNV must map to exactly one different NT token, and the paired
        encodings must have identical shapes and attention masks. This catches
        truncation or token alignment problems instead of silently scoring the
        first difference.
        """
        if ref_input_ids.ndim != 2 or alt_input_ids.ndim != 2:
            raise ValueError("NT variant inputs must have shape (batch, sequence)")
        if ref_input_ids.shape != alt_input_ids.shape:
            raise ValueError(
                "NT ref/alt tokenized shapes differ: "
                f"ref={tuple(ref_input_ids.shape)}, alt={tuple(alt_input_ids.shape)}"
            )

        if attention_mask is None:
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                attention_mask = torch.ones_like(ref_input_ids)
            else:
                attention_mask = ref_input_ids.ne(pad_token_id).long()
        if alt_attention_mask is not None and not torch.equal(
            attention_mask.to(alt_attention_mask.device), alt_attention_mask
        ):
            raise ValueError("NT ref/alt attention masks differ")

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise RuntimeError("NT tokenizer does not define a mask token")

        active_tokens = attention_mask.bool()
        changed_tokens = ref_input_ids.ne(alt_input_ids) & active_tokens
        changed_counts = changed_tokens.sum(dim=1)
        if not torch.all(changed_counts == 1):
            counts = changed_counts.detach().cpu().tolist()
            raise ValueError(
                "Each NT SNV must change exactly one active 6-mer token; "
                f"observed per-sample token differences={counts}"
            )

        positions = changed_tokens.to(torch.int64).argmax(dim=1)
        masked_ref_ids = ref_input_ids.clone()
        rows = torch.arange(ref_input_ids.shape[0], device=ref_input_ids.device)
        masked_ref_ids[rows, positions] = int(mask_token_id)
        return self.get_logit(
            input_ids=masked_ref_ids,
            attention_mask=attention_mask,
            return_numpy=return_numpy,
        )

    def calc_llr(
        self, logits: torch.Tensor, position: int, ref_token: int, alt_token: int
    ) -> float:
        """Compute log P(alt | masked context) - log P(ref | masked context)."""
        if logits.ndim == 3:
            token_logits = logits[0, position, :]
        elif logits.ndim == 2:
            token_logits = logits[position, :]
        else:
            token_logits = logits
        if max(ref_token, alt_token) >= token_logits.shape[-1]:
            raise ValueError(
                "NT token id exceeds pretrained LM head vocabulary: "
                f"ref={ref_token}, alt={alt_token}, vocab={token_logits.shape[-1]}"
            )
        log_probs = torch.log_softmax(token_logits.float(), dim=-1)
        return float((log_probs[alt_token] - log_probs[ref_token]).item())


class ExtractLogitForHyena(ExtractLogitBase):
    """HyenaDNA模型的Logit提取器"""

    def __init__(self, model_path: str, model_name: str):
        self.model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            output_loading_info=True,
        )
        missing = sorted(set(loading_info.get("missing_keys", [])))
        unexpected = sorted(set(loading_info.get("unexpected_keys", [])))
        mismatched = loading_info.get("mismatched_keys", [])
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "HyenaDNA checkpoint is not a complete CausalLM: "
                f"missing_keys={missing}; unexpected_keys={unexpected}; "
                f"mismatched_keys={mismatched}"
            )
        head = getattr(self.model, "lm_head", None)
        head_weight = getattr(head, "weight", None)
        if head_weight is None:
            raise RuntimeError("HyenaDNA checkpoint has no lm_head.weight")
        if not torch.isfinite(head_weight.detach()).all().item():
            raise RuntimeError("HyenaDNA lm_head.weight contains non-finite values")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.model_path = model_path
        self.model_name = model_name
        self.model.eval()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # max_length is the biological sequence length in bp. HyenaDNA adds
        # one BOS token, so reserve one extra tokenizer position.
        special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        token_max_length = max_length + special_tokens
        model_limit = getattr(self.model.config, "max_seq_len", None)
        if model_limit is not None and token_max_length > int(model_limit):
            raise ValueError(
                f"HyenaDNA token length {token_max_length} exceeds "
                f"checkpoint max_seq_len={model_limit}"
            )
        encoded = self.tokenizer(
            texts,
            max_length=token_max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, return_dict=True)
            logits = outputs.logits
            print(logits.shape)

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def logit_to_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (length, vocab)
        input_ids: (length,)
        返回每个位置对应 token 的 log-prob，形状为 (length,)
        """
        # Apply log_softmax along the vocabulary dimension.
        softmax_logprobs = torch.log_softmax(logits, dim=-1)

        # As in the batched version, align predictions with targets by predicting the token at the next position.
        softmax_logprobs = softmax_logprobs[:-1]  # (length-1, vocab)
        input_ids = input_ids[1:]  # (length-1,)

        assert softmax_logprobs.shape[0] == input_ids.shape[0]

        # Retrieve the corresponding log probability by token ID.
        logprobs = softmax_logprobs[torch.arange(len(input_ids)), input_ids]

        return logprobs  # (length-1,)

    def calc_score(
        self,
        logits: torch.Tensor,
        input_id,
        reduce_method="mean",
    ) -> float:
        """
        计算对数似然比 (LLR)。
        """

        logprobs = self.logit_to_logprobs(logits, input_id)
        logprobs = logprobs.float().cpu().numpy()

        if reduce_method == "sum":  # PLL
            reduce_func = np.sum
        elif reduce_method == "mean":  # mean PLL
            reduce_func = np.mean
        else:
            raise ValueError(f"Invalid reduce_method {reduce_method}")
        score: float = reduce_func(logprobs[: len(input_id)])
        return score


class ExtractLogitForGPN(ExtractLogitBase):
    """
    使用 AutoTokenizer + AutoModelForSequenceClassification 从 HF 加载模型并提供 get_embedding。
    """

    import gpn.model  # noqa: F401
    from gpn.data import GenomeMSA, Tokenizer  # noqa: F401

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if model_name == "GPN-MSA":
            # Transformers' generic dispatcher maps this model_type to the
            # older GPN class and does not preserve the checkpoint MLM head.
            # Load the checkpoint-declared GPNRoFormer architecture directly.
            from gpn.model import GPNRoFormerConfig, GPNRoFormerForMaskedLM

            config = GPNRoFormerConfig.from_pretrained(model_path)
            self.model, loading_info = GPNRoFormerForMaskedLM.from_pretrained(
                model_path,
                config=config,
                output_loading_info=True,
                trust_remote_code=True,
            )
        else:
            self.model, loading_info = AutoModelForMaskedLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                output_loading_info=True,
            )
        validate_pretrained_lm_head(self.model, loading_info, self.tokenizer)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        # print(input_ids.shape)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
            )
            logits = outputs.logits
            # print(logits.shape)
        return logits


class ExtractLogitForSpeciesLM(ExtractLogitBase):
    """
    使用 AutoTokenizer + AutoModelForSequenceClassification 从 HF 加载模型并提供 get_embedding。
    """

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # Split each text into 6-base segments.
        encoded = self.tokenizer(
            [" ".join([text[i : i + 6] for i in range(len(text))]) for text in texts],
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        ids = encoded["input_ids"]
        # print(ids)

        # Find rows that are not all zeros.
        _nonzero_rows = ids[ids.sum(dim=1) != 0]

        # print(nonzero_rows)
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )
            logits = outputs.logits
        return logits


class ExtractLogitForEvo2_1B(object):
    """
    使用 AutoTokenizer + AutoModelForSequenceClassification 从 HF 加载模型并提供 get_embedding。
    get_embedding 使用 last_hidden_state（避免返回所有 hidden_states）。
    """

    def __init__(self, model_path, model_name, **kwargs):
        from runtime.evo2_local import load_local_evo2

        self.model_name = "evo2-1b"
        self.model = load_local_evo2(
            "evo2_1b_base",
            model_path,
        )
        self.tokenizer = self.model.tokenizer.tokenize_batch
        self.detokenizer = self.model.tokenizer.detokenize
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        texts = [text.encode("ascii") for text in texts]
        input_ids = torch.tensor(
            self.tokenizer(
                texts,
            ),
            dtype=torch.int,
        )

        encoded = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).long(),
        }
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        outputs, embeddings = self.model(
            input_ids.to(self.device),
        )
        self.input_ids = input_ids
        logits = outputs[0]
        return logits

    def get_diff_token(
        self, ref_seq: List[str], alt_seq: List[str]
    ) -> Tuple[int, int, int, str, str] | None:
        """
        接受两个碱基序列，找出第一个不同的token位置和值

        返回:
          (position, ref_token, alt_token, ref_char, alt_char)
          position: 差异token在序列中的位置（从0开始）
          ref_token/alt_token: token id
          ref_char/alt_char: 实际的碱基字符
        """
        ref_seq_bytes: List[bytes] = [ref.encode() for ref in ref_seq]
        alt_seq_bytes: List[bytes] = [alt.encode() for alt in alt_seq]

        ref_tokens = self.tokenizer(ref_seq_bytes)
        alt_tokens = self.tokenizer(alt_seq_bytes)

        # Find the first differing token.
        for i, (ref_token, alt_token) in enumerate(zip(ref_tokens, alt_tokens)):
            if ref_token != alt_token:
                # Decode the token to obtain the actual character.
                ref_char = self.detokenizer(ref_token)
                alt_char = self.detokenizer(alt_token)
                return i, ref_token, alt_token, ref_char, alt_char

        raise ValueError("两个序列完全相同，未找到差异token。")

    def logit_to_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (length, vocab)
        input_ids: (length,)
        返回每个位置对应 token 的 log-prob，形状为 (length,)
        """
        # Apply log_softmax along the vocabulary dimension.
        softmax_logprobs = torch.log_softmax(logits, dim=-1)

        # As in the batched version, align predictions with targets by predicting the token at the next position.
        softmax_logprobs = softmax_logprobs[:-1]  # (length-1, vocab)
        input_ids = input_ids[1:]  # (length-1,)

        assert softmax_logprobs.shape[0] == input_ids.shape[0]

        # Retrieve the corresponding log probability by token ID.
        logprobs = softmax_logprobs[torch.arange(len(input_ids)), input_ids]

        return logprobs  # (length-1,)

    def calc_score(
        self,
        logits: torch.Tensor,
        input_id,
        reduce_method="mean",
    ) -> float:
        """
        计算对数似然比 (LLR)。
        """

        logprobs = self.logit_to_logprobs(logits, input_id)
        logprobs = logprobs.float().cpu().numpy()

        if reduce_method == "sum":  # PLL
            reduce_func = np.sum
        elif reduce_method == "mean":  # mean PLL
            reduce_func = np.mean
        else:
            raise ValueError(f"Invalid reduce_method {reduce_method}")
        score: float = reduce_func(logprobs[: len(input_id)])
        return score


class ExtractLogitForEvo2_7B(object):
    """
    使用 AutoTokenizer + AutoModelForSequenceClassification 从 HF 加载模型并提供 get_embedding。
    get_embedding 使用 last_hidden_state（避免返回所有 hidden_states）。
    """

    def __init__(self, model_path, model_name, **kwargs):
        from runtime.evo2_local import load_local_evo2

        self.model_name = "evo2-7b"
        self.model = load_local_evo2(
            "evo2_7b_base",
            model_path,
        )
        self.tokenizer = self.model.tokenizer.tokenize_batch
        self.detokenizer = self.model.tokenizer.detokenize
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        texts = [text.encode("ascii") for text in texts]
        input_ids = torch.tensor(
            self.tokenizer(
                texts,
            ),
            dtype=torch.int,
        )

        encoded = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).long(),
        }
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = "mean",
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        outputs, embeddings = self.model(
            input_ids.to(self.device),
        )
        self.input_ids = input_ids
        logits = outputs[0]
        return logits

    def get_diff_token(
        self, ref_seq: List[str], alt_seq: List[str]
    ) -> Tuple[int, int, int, str, str] | None:
        """
        接受两个碱基序列，找出第一个不同的token位置和值

        返回:
          (position, ref_token, alt_token, ref_char, alt_char)
          position: 差异token在序列中的位置（从0开始）
          ref_token/alt_token: token id
          ref_char/alt_char: 实际的碱基字符
        """
        ref_seq_bytes: List[bytes] = [ref.encode() for ref in ref_seq]
        alt_seq_bytes: List[bytes] = [alt.encode() for alt in alt_seq]

        ref_tokens = self.tokenizer(ref_seq_bytes)
        alt_tokens = self.tokenizer(alt_seq_bytes)

        # Find the first differing token.
        for i, (ref_token, alt_token) in enumerate(zip(ref_tokens, alt_tokens)):
            if ref_token != alt_token:
                # Decode the token to obtain the actual character.
                ref_char = self.detokenizer(ref_token)
                alt_char = self.detokenizer(alt_token)
                return i, ref_token, alt_token, ref_char, alt_char

        raise ValueError("两个序列完全相同，未找到差异token。")

    def logit_to_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (length, vocab)
        input_ids: (length,)
        返回每个位置对应 token 的 log-prob，形状为 (length,)
        """
        # Apply log_softmax along the vocabulary dimension.
        softmax_logprobs = torch.log_softmax(logits, dim=-1)

        # As in the batched version, align predictions with targets by predicting the token at the next position.
        softmax_logprobs = softmax_logprobs[:-1]  # (length-1, vocab)
        input_ids = input_ids[1:]  # (length-1,)

        assert softmax_logprobs.shape[0] == input_ids.shape[0]

        # Retrieve the corresponding log probability by token ID.
        logprobs = softmax_logprobs[torch.arange(len(input_ids)), input_ids]

        return logprobs  # (length-1,)

    def calc_score(
        self,
        logits: torch.Tensor,
        input_id,
        reduce_method="mean",
    ) -> float:
        """
        计算对数似然比 (LLR)。
        """

        logprobs = self.logit_to_logprobs(logits, input_id)
        logprobs = logprobs.float().cpu().numpy()

        if reduce_method == "sum":  # PLL
            reduce_func = np.sum
        elif reduce_method == "mean":  # mean PLL
            reduce_func = np.mean
        else:
            raise ValueError(f"Invalid reduce_method {reduce_method}")
        score: float = reduce_func(logprobs[: len(input_id)])
        return score


class ExtractLogitForGenos(ExtractLogitBase):
    """Genos模型的Logit提取器"""

    def __init__(self, model_path: str, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model_path = model_path
        self.model_name = model_name
        self.model.eval()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids=None,
        start_pos=None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, return_dict=True)
            logits = outputs.logits
            print(logits.shape)

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def logit_to_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: (length, vocab)
        input_ids: (length,)
        返回每个位置对应 token 的 log-prob，形状为 (length,)
        """
        # Apply log_softmax along the vocabulary dimension.
        softmax_logprobs = torch.log_softmax(logits, dim=-1)

        # As in the batched version, align predictions with targets by predicting the token at the next position.
        softmax_logprobs = softmax_logprobs[:-1]  # (length-1, vocab)
        input_ids = input_ids[1:]  # (length-1,)

        assert softmax_logprobs.shape[0] == input_ids.shape[0]

        # Retrieve the corresponding log probability by token ID.
        logprobs = softmax_logprobs[torch.arange(len(input_ids)), input_ids]

        return logprobs  # (length-1,)

    def calc_score(
        self,
        logits: torch.Tensor,
        input_id,
        reduce_method="mean",
    ) -> float:
        """
        计算对数似然比 (LLR)。
        """

        logprobs = self.logit_to_logprobs(logits, input_id)
        logprobs = logprobs.float().cpu().numpy()

        if reduce_method == "sum":  # PLL
            reduce_func = np.sum
        elif reduce_method == "mean":  # mean PLL
            reduce_func = np.mean
        else:
            raise ValueError(f"Invalid reduce_method {reduce_method}")
        score: float = reduce_func(logprobs[: len(input_id)])
        return score


class ExtractLogitForhEvoPos(ExtractLogitBase):
    """
    hEvo模型的Logit提取器 (支持绝对位置编码)

    该模型需要额外的位置信息:
    - chr_ids: 染色体ID
    - start_pos: 绝对起始位置
    """

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

        # Chromosome mapping; adjust as needed.
        self.chr_mapping = {
            "chr1": 0,
            "chr2": 1,
            "chr3": 2,
            "chr4": 3,
            "chr5": 4,
            "chr6": 5,
            "chr7": 6,
            "chr8": 7,
            "chr9": 8,
            "chr10": 9,
            "chr11": 10,
            "chr12": 11,
            "chr13": 12,
            "chr14": 13,
            "chr15": 14,
            "chr16": 15,
            "chr17": 16,
            "chr18": 17,
            "chr19": 18,
            "chr20": 19,
            "chr21": 20,
            "chr22": 21,
            "chrX": 22,
            "chrY": 23,
            "chrM": 24,
        }

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        chr_ids: Optional[List[int]] = None,
        start_positions: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        对序列进行tokenize，并添加位置信息

        参数:
            texts: 输入序列列表
            max_length: 最大序列长度
            padding: 是否padding
            truncation: 是否截断
            return_tensors: 返回类型
            chr_ids: 染色体ID列表 (可选)
            start_positions: 起始位置列表 (可选)
        """
        # Basic tokenization.
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

        batch_size = len(texts)

        # Add the chromosome ID when provided.
        if chr_ids is not None:
            if isinstance(chr_ids, list):
                chr_ids = torch.tensor(chr_ids, dtype=torch.long)
            encoded["chr_ids"] = chr_ids.unsqueeze(-1).expand(
                batch_size, encoded["input_ids"].shape[1]
            )
        else:
            # Default chromosome ID: 0.
            encoded["chr_ids"] = torch.zeros(
                (batch_size, encoded["input_ids"].shape[1]), dtype=torch.long
            )

        # Add the start position when provided.
        if start_positions is not None:
            if isinstance(start_positions, list):
                start_positions = torch.tensor(start_positions, dtype=torch.long)
            encoded["start_pos"] = start_positions.unsqueeze(-1)  # [Batch, 1]
        else:
            # Default start position: 0.
            encoded["start_pos"] = torch.zeros((batch_size, 1), dtype=torch.long)

        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        获取模型logits

        参数:
            input_ids: 输入token IDs [Batch, SeqLen]
            attention_mask: 注意力掩码 [Batch, SeqLen]
            chr_ids: 染色体IDs [Batch, SeqLen] 或 [Batch]
            start_pos: 起始位置 [Batch, 1] 或 [Batch]
            return_numpy: 是否返回numpy数组
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Move to the device.
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Process the chromosome ID.
        if chr_ids is None:
            # Default to 0.
            chr_ids = torch.zeros_like(input_ids)
        else:
            chr_ids = chr_ids.to(self.device)
            # Expand from 1D to 2D when needed.
            if chr_ids.dim() == 1:
                chr_ids = chr_ids.unsqueeze(-1).expand_as(input_ids)

        # Process the start position.
        if start_pos is None:
            # Default to 0.
            start_pos = torch.zeros(
                (input_ids.shape[0], 1), dtype=torch.long, device=self.device
            )
        else:
            start_pos = start_pos.to(self.device)
            # Ensure the shape is [Batch, 1].
            if start_pos.dim() == 1:
                start_pos = start_pos.unsqueeze(-1)

        # Run the forward pass.
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chr_ids=chr_ids,
                start_pos=start_pos,
            )
            # Extract logits according to the model output format.
            if isinstance(outputs, tuple):
                logits = outputs[1]  # (loss, logits, labels)
            else:
                logits = outputs

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def parse_variant_location(self, idx: str) -> Tuple[Optional[int], Optional[int]]:
        """
        从variant ID中解析染色体和位置信息

        假设ID格式类似: chr1:12345_A_T 或 chr1-12345-A-T

        参数:
            idx: variant ID字符串

        返回:
            (chr_id, position) 或 (None, None)
        """
        try:
            # Try several delimiter formats.
            for sep in [":", "-", "_"]:
                if sep in idx:
                    parts = idx.split(sep)
                    if len(parts) >= 2:
                        chr_str = parts[0].lower()
                        pos_str = parts[1]

                        # Extract the chromosome number.
                        chr_id = self.chr_mapping.get(chr_str)
                        if chr_id is None:
                            # Try to extract the number directly.
                            chr_num = "".join(filter(str.isdigit, chr_str))
                            if chr_num:
                                chr_id = int(chr_num) - 1  # Convert to zero-based indexing.

                        # Extract the position.
                        pos_num = "".join(filter(str.isdigit, pos_str))
                        if pos_num:
                            position = int(pos_num)
                            return chr_id, position

            return None, None
        except Exception as e:
            print(f"Warning: 无法解析variant位置信息 '{idx}': {e}")
            return None, None

    def extract_location_from_batch(
        self, indices: List[str]
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """
        从一批variant IDs中提取位置信息

        参数:
            indices: variant ID列表

        返回:
            (chr_ids, positions) 或 (None, None)
        """
        chr_ids = []
        positions = []

        for idx in indices:
            chr_id, position = self.parse_variant_location(idx)
            if chr_id is not None and position is not None:
                chr_ids.append(chr_id)
                positions.append(position)
            else:
                # Use the default value if parsing fails.
                chr_ids.append(0)
                positions.append(0)

        # Check whether all values are defaults.
        if all(c == 0 for c in chr_ids) and all(p == 0 for p in positions):
            return None, None

        return chr_ids, positions


class ExtractLogitForGENALM(ExtractLogitBase):
    """GENA-LM 模型的 Logit 提取器"""

    def __init__(self, model_path: str, model_name: str):
        # GENA-LM usually uses AutoModelForMaskedLM for zero-shot variant effect prediction (VEP).
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # GENA-LM follows standard tokenizer behavior.
        # Note: Some GENA-LM variants, such as BigBird, support very long sequences; adjust max_length as needed.
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=truncation,
            return_tensors=return_tensors,
            pad_to_multiple_of=8,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            # Run the forward pass.
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

            # Extract masked language model logits with shape (Batch, Seq, Vocab).
            logits = outputs.logits

        if return_numpy:
            return logits.cpu().numpy()
        return logits


class ExtractLogitForGROVER(ExtractLogitBase):
    """GROVER 模型的 Logit 提取器"""

    def __init__(self, model_path: str, model_name: str):
        # GENA-LM usually uses AutoModelForMaskedLM for zero-shot variant effect prediction (VEP).
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # GENA-LM follows standard tokenizer behavior.
        # Note: Some GENA-LM variants, such as BigBird, support very long sequences; adjust max_length as needed.
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=truncation,
            return_tensors=return_tensors,
            pad_to_multiple_of=8,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            # Run the forward pass.
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

            # Extract masked language model logits with shape (Batch, Seq, Vocab).
            logits = outputs.logits

        if return_numpy:
            return logits.cpu().numpy()
        return logits


class ExtractLogitForNTv3(ExtractLogitBase):
    """Nucleotide Transformer模型的Logit提取器"""

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=128,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits

        if return_numpy:
            return logits.cpu().numpy()
        return logits


class ExtractLogitForMistralDNA:
    """Mistral-DNA scorer using the tutorial's pooled-logit distance."""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model.eval()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.config.use_cache = False

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=128,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def score_paired_sequences(
        self,
        ref_seqs: List[str],
        alt_seqs: List[str],
        max_tokens: int = 512,
    ) -> np.ndarray:
        """Score each pair by L2 distance between max-pooled vocabulary logits.

        Padding positions are excluded before pooling. The returned distance is
        already oriented as an effect score (larger means more different), so
        callers must evaluate it as a raw score rather than negate it.
        """
        return mistral_score_batch(
            self.tokenizer, self.model, ref_seqs, alt_seqs, self.device, max_tokens
        )

    def get_diff_token(
        self, ref_seq: str, alt_seq: str
    ) -> Tuple[int, int, int, str, str] | None:
        """
        接受两个碱基序列，找出第一个不同的token位置和值
        如果序列完全相同，返回 None 而不是抛出异常。
        """
        ref_tokens = self.tokenizer.encode(ref_seq, add_special_tokens=True)
        alt_tokens = self.tokenizer.encode(alt_seq, add_special_tokens=True)

        # Find the first differing token.
        for i, (ref_token, alt_token) in enumerate(zip(ref_tokens, alt_tokens)):
            if ref_token != alt_token:
                # Decode the token to obtain the actual character.
                ref_char = self.tokenizer.decode([ref_token])
                alt_char = self.tokenizer.decode([alt_token])
                return i, ref_token, alt_token, ref_char, alt_char

        # [Change] Return None instead of raising ValueError.
        print("Warning: No difference found between ref and alt sequences. Skipping...")
        return None

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs["logits"]
            print(logits)
            print(logits.shape)

        if return_numpy:
            return logits.cpu().numpy()
        return logits


class ExtractLogitForEnformer:
    def __init__(self, model_path: str, model_name: str = "enformer"):
        self.model_path = model_path
        self.model_name = model_name
        self.model = Enformer.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize(self, texts: List[str], max_length: int = 196608, **kwargs):
        # Use the same logic as the Embedding class.
        all_ids = []
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        for text in texts:
            seq = text.upper()[:max_length]
            ids = [base_map.get(base, 4) for base in seq]
            if len(ids) < max_length:
                ids += [4] * (max_length - len(ids))
            all_ids.append(ids)
        return {"input_ids": torch.tensor(all_ids)}

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, np.ndarray]:
        input_ids = input_ids.to(self.device)

        with torch.inference_mode():
            # 1. Convert to one-hot encoding.
            one_hot = seq_indices_to_one_hot(input_ids).to(self.device)

            # 2. Run inference to obtain the output dictionary.
            output_dict = self.model(one_hot)

            # 3. Keep predictions for humans only.
            # Shape: (batch, 896, 5313).
            logits = output_dict["human"]

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def calc_vep(
        self, ref_logits: torch.Tensor, alt_logits: torch.Tensor, method: str = "diff"
    ) -> torch.Tensor:
        """
        Enformer 是回归模型，其变异效应通常用预测值的差异来衡量，而非 LLR
        """
        if method == "diff":
            return alt_logits - ref_logits
        elif method == "log_ratio":
            return torch.log(alt_logits + 1e-8) - torch.log(ref_logits + 1e-8)
        return alt_logits - ref_logits


class ExtractLogitForCaduceus(ExtractLogitBase):
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model, loading_info = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=True, output_loading_info=True
        )
        validate_pretrained_lm_head(self.model, loading_info, self.tokenizer)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize(
        self, texts: List[str], max_length: int = 512, **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def get_logit(
        self, input_ids: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        input_ids = input_ids.to(self.device)
        with torch.inference_mode():
            outputs = self.model(input_ids)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

        return logits.cpu().numpy() if kwargs.get("return_numpy") else logits

    def get_diff_token(
        self, ref_seq: str, alt_seq: str
    ) -> Tuple[int, int, int, str, str] | None:
        """
        接受两个碱基序列，找出第一个不同的token位置和值
        如果序列完全相同，返回 None 而不是抛出异常。
        """
        ref_tokens = self.tokenizer.encode(ref_seq, add_special_tokens=True)
        alt_tokens = self.tokenizer.encode(alt_seq, add_special_tokens=True)

        # Find the first differing token.
        for i, (ref_token, alt_token) in enumerate(zip(ref_tokens, alt_tokens)):
            if ref_token != alt_token:
                # Decode the token to obtain the actual character.
                ref_char = self.tokenizer.decode([ref_token])
                alt_char = self.tokenizer.decode([alt_token])
                return i, ref_token, alt_token, ref_char, alt_char

        # [Change] Return None instead of raising ValueError.
        print("Warning: No difference found between ref and alt sequences. Skipping...")
        return None

    def calc_llr(
        self, logits: torch.Tensor, position: int, ref_token: int, alt_token: int
    ) -> float:
        """
        计算对数似然比 (LLR)。
        公式: log_softmax(logits)[alt_token] - log_softmax(logits)[ref_token]
        """
        # for t in logits:
        #     print(t.item(), end=" ")
        # print()
        # Handle the batch dimension.
        if logits.ndim == 3:
            # Assume logits has shape (Batch, Seq, Vocab) with batch_size=1.
            # Alternatively, logits may already correspond to this sample.
            token_logits = logits[0, position, :]
        elif logits.ndim == 2:
            # (Seq, Vocab)
            token_logits = logits[position, :]
        else:
            token_logits = logits

        # Use log_softmax to compute log probabilities.
        token_logits = token_logits.to(torch.float32)
        log_probs = torch.log_softmax(token_logits, dim=-1)

        # Extract and subtract the log probabilities for ref and alt.
        llr = log_probs[alt_token] - log_probs[ref_token]

        print(
            f"Position: {position}, Ref: {ref_token}, Alt: {alt_token}, LLR: {llr.item()}"
        )
        return llr.item()


class ExtractLogitForGENERator:
    """GENERATOR scorer using next-token probabilities marginalized by base."""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("GENERATOR tokenizer has no pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype
        )
        self.model.eval()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.config.use_cache = False

        self.base_token_ids = generator_base_token_ids(
            self.tokenizer,
            missing_message="GENERATOR vocabulary has no tokens beginning with",
        )

    def tokenize(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=128,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return encoded

    def score_next_base(
        self, contexts: List[str], ref_bases: List[str], alt_bases: List[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return log(P(ref)/P(alt)) after next-token base marginalization."""
        return generator_score_batch(
            self.tokenizer,
            self.model,
            contexts,
            ref_bases,
            alt_bases,
            self.base_token_ids,
            self.device,
        )

    def get_logit(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_numpy: bool = False,
        chr_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, np.ndarray]:
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            logits = outputs.logits

        if return_numpy:
            return logits.cpu().numpy()
        return logits

    def get_diff_token(
        self, ref_seq: str, alt_seq: str
    ) -> Tuple[int, int, int, str, str] | None:
        """
        接受两个碱基序列，找出第一个不同的token位置和值
        如果序列完全相同，返回 None 而不是抛出异常。
        """
        ref_tokens = self.tokenizer.encode(ref_seq, add_special_tokens=True)
        alt_tokens = self.tokenizer.encode(alt_seq, add_special_tokens=True)

        # Find the first differing token.
        for i, (ref_token, alt_token) in enumerate(zip(ref_tokens, alt_tokens)):
            if ref_token != alt_token:
                # Decode the token to obtain the actual character.
                ref_char = self.tokenizer.decode([ref_token])
                alt_char = self.tokenizer.decode([alt_token])
                return i, ref_token, alt_token, ref_char, alt_char

        # [Change] Return None instead of raising ValueError.
        print("Warning: No difference found between ref and alt sequences. Skipping...")
        return None
