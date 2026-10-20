"""Equivalence checks for the shared native variant scoring adapters."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapters.extract_logits_classes import (  # noqa: E402
    ExtractLogitForGENERator,
    ExtractLogitForMistralDNA,
)
from runtime.native_clinvar_models import (  # noqa: E402
    generator_base_token_ids as native_generator_base_token_ids,
    generator_score_batch as native_generator_score_batch,
    mistral_score_batch as native_mistral_score_batch,
)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        rows = [[ord(char) % 7 + 1 for char in text] for text in texts]
        width = max(map(len, rows), default=0)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def get_vocab(self):
        return {
            "A": 0,
            "AA": 1,
            "C": 2,
            "CC": 3,
            "G": 4,
            "GG": 5,
            "T": 6,
            "TT": 7,
            "<pad>": 8,
        }


class FakeModel:
    def __call__(self, input_ids, attention_mask, **kwargs):
        batch, length = input_ids.shape
        positions = torch.arange(length, dtype=torch.float32)[None, :, None]
        vocabulary = torch.arange(9, dtype=torch.float32)[None, None, :]
        token_signal = input_ids.float()[:, :, None]
        logits = token_signal * 0.25 + positions * 0.1 + vocabulary * 0.03
        logits = logits.expand(batch, length, 9).contiguous()
        return SimpleNamespace(logits=logits)


def extractor(model_class, tokenizer, model, base_token_ids=None):
    instance = object.__new__(model_class)
    instance.tokenizer = tokenizer
    instance.model = model
    instance.device = torch.device("cpu")
    if base_token_ids is not None:
        instance.base_token_ids = base_token_ids
    return instance


def test_mistral_native_and_extractor_batch_scores_match():
    tokenizer, model = FakeTokenizer(), FakeModel()
    refs, alts = ["ACGT", "AC"], ["ATGT", "AT"]
    native = native_mistral_score_batch(tokenizer, model, refs, alts, "cpu", 12)
    wrapped = extractor(ExtractLogitForMistralDNA, tokenizer, model)
    extracted = wrapped.score_paired_sequences(refs, alts, max_tokens=12)
    np.testing.assert_array_equal(native, extracted)
    assert native.shape == (2,)
    assert np.all(native >= 0)


def test_generator_native_and_extractor_scores_and_marginals_match():
    tokenizer, model = FakeTokenizer(), FakeModel()
    contexts, refs, alts = ["AC", "GTA"], ["A", "C"], ["G", "T"]
    token_ids = native_generator_base_token_ids(tokenizer)
    native = native_generator_score_batch(
        tokenizer, model, contexts, refs, alts, token_ids, "cpu"
    )
    wrapped = extractor(ExtractLogitForGENERator, tokenizer, model, token_ids)
    extracted = wrapped.score_next_base(contexts, refs, alts)
    for native_values, extracted_values in zip(native, extracted):
        np.testing.assert_array_equal(native_values, extracted_values)
    scores, p_ref, p_alt = native
    np.testing.assert_allclose(scores, np.log(p_ref) - np.log(p_alt + 1e-10))
    assert np.all((p_ref > 0) & (p_ref < 1))
    assert np.all((p_alt > 0) & (p_alt < 1))


@pytest.mark.parametrize(
    "score_call",
    [
        lambda: native_mistral_score_batch(
            FakeTokenizer(), FakeModel(), ["A"], [], "cpu", 8
        ),
        lambda: native_generator_score_batch(
            FakeTokenizer(),
            FakeModel(),
            ["A"],
            ["A"],
            [],
            native_generator_base_token_ids(FakeTokenizer()),
            "cpu",
        ),
    ],
)
def test_shared_adapters_reject_misaligned_batches(score_call):
    with pytest.raises(ValueError):
        score_call()
