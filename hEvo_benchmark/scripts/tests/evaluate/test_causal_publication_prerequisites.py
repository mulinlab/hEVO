"""Regression checks for failures that previously left the causal roster incomplete."""
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from adapters.masked_base_scorer import score_masked_bases
from adapters.extract_logits_classes import validate_pretrained_lm_head
from runtime.evo2_local import load_local_evo2


class BaseTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 4
    unk_token_id = 5

    def encode(self, text, **kwargs):
        return ["ACGT".index(text)]

    def __call__(self, texts, **kwargs):
        if kwargs.get("add_special_tokens") is True:
            return {"input_ids": [[0, 1]] * len(texts)}
        self.contexts = texts
        return {"input_ids": torch.tensor([[0, 4, 1]] * len(texts)),
                "attention_mask": torch.ones(len(texts), 3, dtype=torch.long)}


class Model:
    device = torch.device("cpu")

    def __init__(self):
        self.tokenizer = BaseTokenizer()

    def get_logit(self, input_ids, **kwargs):
        assert input_ids[:, 1].eq(4).all()
        return torch.arange(6, dtype=torch.float32).expand(*input_ids.shape, 6)


def test_masked_base_context_is_identical_despite_allele_and_reverses_score():
    model = Model()
    scores, variants = score_masked_bases(model, ["AAC", "AGC"], ["AGC", "AAC"], 512)
    assert scores == pytest.approx([2., -2.])
    assert variants == [(1, "A", "G"), (1, "G", "A")]
    assert model.tokenizer.contexts == ["A[MASK]C", "A[MASK]C"]


@pytest.mark.parametrize("ref,alt", [("AAC", "GGC"), ("AAC", "AAC"), ("AAC", "AC")])
def test_masked_base_rejects_non_snv(ref, alt):
    with pytest.raises(ValueError):
        score_masked_bases(Model(), [ref], [alt], 512)


def test_evo_config_is_package_relative_and_override_restored(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/model.yml").write_text("{}")
    mapping = {"evo2_1b_base": "configs/model.yml"}
    seen = []

    def constructor(model_id, weights):
        seen.append(mapping[model_id])
        return SimpleNamespace(model=SimpleNamespace(eval=lambda: None))

    monkeypatch.setitem(sys.modules, "evo2", SimpleNamespace(
        __file__=str(tmp_path / "__init__.py"), Evo2=constructor))
    monkeypatch.setitem(sys.modules, "evo2.utils", SimpleNamespace(CONFIG_MAP=mapping))
    load_local_evo2("evo2_1b_base", tmp_path / "weights.pt")
    assert seen == [str(tmp_path / "configs/model.yml")]
    assert mapping["evo2_1b_base"] == "configs/model.yml"


def test_unused_grover_pooler_can_be_ignored_but_missing_lm_head_cannot():
    model = SimpleNamespace(get_output_embeddings=lambda: torch.nn.Linear(3, 6),
                            config=SimpleNamespace(vocab_size=6))
    info = {"unexpected_keys": ["bert.pooler.dense.weight", "bert.pooler.dense.bias"]}
    with pytest.raises(RuntimeError):
        validate_pretrained_lm_head(model, info, BaseTokenizer())
    validate_pretrained_lm_head(model, info, BaseTokenizer(), allow_unused_pooler=True)
    with pytest.raises(RuntimeError):
        validate_pretrained_lm_head(model, {**info, "missing_keys": ["lm_head.weight"]},
                                   BaseTokenizer(), allow_unused_pooler=True)


def test_evo_fp8_fallback_does_not_edit_installed_config(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    source = tmp_path / "configs/model.yml"
    source.write_text("use_fp8_input_projections: true\n")
    mapping = {"evo2_1b_base": "configs/model.yml"}

    def constructor(model_id, weights):
        from pathlib import Path
        assert yaml.safe_load(Path(mapping[model_id]).read_text())["use_fp8_input_projections"] is False
        return SimpleNamespace(model=SimpleNamespace(eval=lambda: None))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "evo2", SimpleNamespace(
        __file__=str(tmp_path / "__init__.py"), Evo2=constructor))
    monkeypatch.setitem(sys.modules, "evo2.utils", SimpleNamespace(CONFIG_MAP=mapping))
    result = load_local_evo2("evo2_1b_base", tmp_path / "weights.pt")
    assert result.effective_fp8_input_projections is False
    assert source.read_text() == "use_fp8_input_projections: true\n"
    assert mapping["evo2_1b_base"] == "configs/model.yml"


def test_evaluator_does_not_report_success_after_inference_failure(tmp_path, monkeypatch):
    import zeroshot_paried_seq_input as evaluator

    def failure(**kwargs):
        raise ValueError("inference failed before producing scores")

    monkeypatch.setattr(evaluator, "run_llr_evaluation", failure)
    monkeypatch.setattr(evaluator, "SummaryWriter", lambda **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--model_paths", str(tmp_path / "model"),
        "--model_name", "AIDO.DNA-300M", "--dataset_files", str(tmp_path / "data.csv"),
        "--dataset_name", "Causal_Variants", "--h5_path", str(tmp_path / "h5"),
        "--out_dir", str(tmp_path / "results"), "--log_dir", str(tmp_path / "logs")])
    with pytest.raises(RuntimeError, match="Evaluation failed"):
        evaluator.main()
