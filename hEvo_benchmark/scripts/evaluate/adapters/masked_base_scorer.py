"""Score SNVs in one identical nucleotide-masked context for BPE MLMs.

REF and ALT can have different BPE segmentations. Replacing the changed base
before tokenization avoids comparing shifted token positions or padding.
"""
from __future__ import annotations

import numpy as np
import torch

BPE_MASKED_BASE_MODEL_IDS = frozenset(
    {"DNABERT2", "GENA-LM-base", "GENA-LM-large", "GROVER"}
)
SCORING_METHOD = "bpe_masked_nucleotide_marginal_v1"


def score_masked_bases(extractor, refs, alts, max_length):
    if len(refs) != len(alts):
        raise ValueError("REF/ALT batch lengths differ")
    tokenizer = extractor.tokenizer
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError("A nucleotide-masked score requires a mask token")
    base_ids = {}
    for base in "ACGT":
        ids = tokenizer.encode(base, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
            raise ValueError(f"Base {base} has no unambiguous single-token target")
        base_ids[base] = int(ids[0])
    if len(set(base_ids.values())) != 4:
        raise ValueError("Distinct DNA bases must have distinct target IDs")
    contexts, variants = [], []
    for ref, alt in zip(refs, alts):
        if len(ref) != len(alt):
            raise ValueError("Only same-length SNVs are supported")
        changed = [i for i, (a, b) in enumerate(zip(ref, alt)) if a != b]
        if len(changed) != 1:
            raise ValueError("Exactly one changed nucleotide is required")
        pos = changed[0]
        if ref[pos] not in base_ids or alt[pos] not in base_ids:
            raise ValueError("Variant alleles must be A/C/G/T")
        contexts.append(ref[:pos] + tokenizer.mask_token + ref[pos + 1:])
        variants.append((pos, ref[pos], alt[pos]))
    encoded = tokenizer(contexts, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    inputs = encoded["input_ids"].to(extractor.device)
    masks = inputs.eq(tokenizer.mask_token_id)
    if not torch.all(masks.sum(dim=1).eq(1)):
        raise ValueError("Variant mask was truncated or tokenized ambiguously")
    attention = encoded.get("attention_mask")
    if attention is not None:
        attention = attention.to(extractor.device)
    with torch.inference_mode():
        logits = extractor.get_logit(inputs, attention_mask=attention,
                                     return_numpy=False)
        masked_logits = logits[masks].float()
        if max(base_ids.values()) >= masked_logits.shape[-1]:
            raise ValueError("Base target ID is outside LM output vocabulary")
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        scores = [float((log_probs[i, base_ids[a]] - log_probs[i, base_ids[r]]).item())
                  for i, (_, r, a) in enumerate(variants)]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite masked-base LLR")
    return scores, variants
