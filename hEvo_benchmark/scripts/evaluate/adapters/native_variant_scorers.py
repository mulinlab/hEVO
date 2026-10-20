"""Shared numerical adapters for native Mistral-DNA and GENERATOR scores.

Context construction, variant filtering, and output formats belong to each
caller. This module owns only token batching and the model scoring equations.
"""

from __future__ import annotations

import numpy as np
import torch


def mistral_score_batch(
    tokenizer,
    model,
    ref_seqs: list[str],
    alt_seqs: list[str],
    device,
    max_tokens: int,
) -> np.ndarray:
    """Return paired max-pooled vocabulary-logit L2 distances."""
    if len(ref_seqs) != len(alt_seqs):
        raise ValueError("REF and ALT batches must have equal length")
    paired_sequences = [
        sequence for pair in zip(ref_seqs, alt_seqs) for sequence in pair
    ]
    encoded = tokenizer(
        paired_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False, return_dict=True).logits.float()
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask.unsqueeze(-1) == 0, -torch.inf)
        pooled = logits.amax(dim=1)
        distances = torch.linalg.vector_norm(pooled[0::2] - pooled[1::2], dim=-1)
    return distances.cpu().numpy().astype(np.float64, copy=False)


def generator_base_token_ids(
    tokenizer,
    missing_message: str = "GENERATOR vocabulary has no tokens beginning with",
) -> dict[str, list[int]]:
    """Group GENERATOR vocabulary IDs by the first character of each token."""
    groups = {base: [] for base in "ACGT"}
    for token, token_id in tokenizer.get_vocab().items():
        if isinstance(token, str) and token and token[0] in groups:
            groups[token[0]].append(int(token_id))
    missing = [base for base, token_ids in groups.items() if not token_ids]
    if missing:
        raise ValueError(f"{missing_message} {missing}")
    return groups


def generator_score_batch(
    tokenizer,
    model,
    contexts: list[str],
    refs: list[str],
    alts: list[str],
    token_ids_by_base: dict[str, list[int]],
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log(P(ref)/P(alt)) and each marginal next-base probability."""
    if not (len(contexts) == len(refs) == len(alts)):
        raise ValueError("Context, REF, and ALT batches must have equal length")
    encoded = tokenizer(
        ["<s>" + context for context in contexts],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = (
            model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :].float()
        )
        probabilities = torch.softmax(logits, dim=-1)
        p_ref = torch.empty(len(refs), device=device, dtype=torch.float32)
        p_alt = torch.empty_like(p_ref)
        for base in "ACGT":
            indices = torch.tensor(
                token_ids_by_base[base], device=device, dtype=torch.long
            )
            base_probabilities = probabilities.index_select(dim=-1, index=indices).sum(
                dim=-1
            )
            ref_rows = [i for i, value in enumerate(refs) if value == base]
            alt_rows = [i for i, value in enumerate(alts) if value == base]
            if ref_rows:
                p_ref[ref_rows] = base_probabilities[ref_rows]
            if alt_rows:
                p_alt[alt_rows] = base_probabilities[alt_rows]
        scores = torch.log(p_ref) - torch.log(p_alt + 1e-10)
    return tuple(
        values.cpu().numpy().astype(np.float64, copy=False)
        for values in (scores, p_ref, p_alt)
    )
