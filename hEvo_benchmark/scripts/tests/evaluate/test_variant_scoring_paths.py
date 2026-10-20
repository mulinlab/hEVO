"""Exercise scoring routes on small tensors without loading checkpoints."""

from types import SimpleNamespace

import h5py
import pytest
import torch

from adapters.extract_logits_classes import (
    ExtractLogitForhEvo_rare,
    ExtractLogitForHyena,
)
from zeroshot_paried_seq_input import (
    load_llrs_from_h5,
    precompute_and_cache_logits_h5,
    precompute_and_cache_logits_h5_with_position,
    uses_masked_marginal_variant_scoring,
)


class ToyMaskedLM(ExtractLogitForhEvo_rare):
    def __init__(self):
        self.model_name = "hEvo-base"
        self.device = torch.device("cpu")
        self.tokenizer = SimpleNamespace(mask_token_id=4, pad_token_id=5)
        self.seen_inputs = []

    def tokenize(self, texts, **kwargs):
        ids = torch.tensor([["ACGT".index(base) for base in text] for text in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def get_logit(self, input_ids, **kwargs):
        self.seen_inputs.append(input_ids.clone())
        assert torch.all(input_ids.eq(4).any(dim=1)), "MLM received an unmasked input"
        return torch.arange(6, dtype=torch.float32).expand(*input_ids.shape, 6)


class ToyCausalLM(ExtractLogitForHyena):
    def __init__(self):
        self.model_name = "HyenaDNA_tiny_1k"
        self.device = torch.device("cpu")
        self.seen_inputs = []

    tokenize = ToyMaskedLM.tokenize

    def get_logit(self, input_ids, **kwargs):
        assert not input_ids.eq(4).any(), "Causal input was masked"
        self.seen_inputs.append(input_ids.clone())
        return torch.zeros(*input_ids.shape, 6)

    def get_diff_token(self, ref, alt):
        return next(
            (i, "ACGT".index(r), "ACGT".index(a), r, a)
            for i, (r, a) in enumerate(zip(ref, alt))
            if r != a
        )

    def calc_score(self, logits, input_id):
        return float(input_id.float().mean())


@pytest.mark.parametrize(
    "precompute",
    [
        precompute_and_cache_logits_h5,
        precompute_and_cache_logits_h5_with_position,
    ],
)
def test_mlm_masks_all_changed_targets_and_sums_contrasts(precompute, tmp_path):
    model = ToyMaskedLM()
    cache = tmp_path / "masked.hdf5"
    precompute(model, [("variant", "ACGT", "TGGT", 1)], str(cache), batch_size=1)
    assert model.seen_inputs[0].tolist() == [[4, 4, 2, 3]]
    with h5py.File(cache) as handle:
        assert handle["llrs"][:].tolist() == pytest.approx([4.0])
        assert handle["labels"][:].tolist() == [1]


def test_single_variant_helper_uses_the_same_masked_context():
    model = ToyMaskedLM()
    assert model.calc_variant_llr("ACGT", "TGGT") == pytest.approx(4.0)
    assert model.seen_inputs[0].tolist() == [[4, 4, 2, 3]]


@pytest.mark.parametrize(
    "precompute",
    [
        precompute_and_cache_logits_h5,
        precompute_and_cache_logits_h5_with_position,
    ],
)
def test_causal_route_compares_unmasked_ref_and_alt_sequences(precompute, tmp_path):
    model = ToyCausalLM()
    cache = tmp_path / "causal.hdf5"
    precompute(model, [("variant", "ACGT", "TGGT", 1)], str(cache), batch_size=1)
    assert len(model.seen_inputs) == 2
    with h5py.File(cache) as handle:
        assert handle["llrs"][:].tolist() == pytest.approx([1.0])


@pytest.mark.parametrize("model_name", ["unregistered-MLM", "GPN-MSA", "Enformer"])
def test_unsupported_models_cannot_fall_back_to_unmasked_mlm(model_name):
    with pytest.raises(ValueError):
        uses_masked_marginal_variant_scoring(SimpleNamespace(model_name=model_name))


def test_cache_without_current_protocol_fingerprint_is_rejected(tmp_path):
    cache = tmp_path / "old.hdf5"
    with h5py.File(cache, "w") as handle:
        handle.create_dataset("llrs", data=[1.0])
        handle.create_dataset("labels", data=[1])
    assert load_llrs_from_h5(str(cache), expected_fingerprint="masked-protocol") is None
