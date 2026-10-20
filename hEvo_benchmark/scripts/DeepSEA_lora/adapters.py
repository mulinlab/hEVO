from __future__ import annotations

import importlib.util
import sys
import types
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_hevo(model_path: Path, attention_backend: str = "eager"):
    data_module = _load_module(
        model_path / "data.py", f"deepsea_data_{model_path.name}"
    )
    model_module = _load_module(
        model_path / "model.py", f"deepsea_model_{model_path.name}"
    )
    if attention_backend not in {"eager", "flash_attention_2", "torch_flash_sdp"}:
        raise ValueError(f"Unsupported hEvo attention backend: {attention_backend}")
    use_flash_attention = attention_backend in {"flash_attention_2", "torch_flash_sdp"}
    if use_flash_attention and not getattr(model_module, "FLASH_ATTN_AVAILABLE", False):
        raise RuntimeError(
            "FLASH_ATTENTION_2_UNAVAILABLE: hEvo requested flash_attention_2, "
            "but flash_attn.flash_attn_interface.flash_attn_func is not importable"
        )
    config = model_module.CustomBertConfig.from_pretrained(str(model_path))
    # hEvo implements FlashAttention-2 inside its own FlashBertSelfAttention,
    # rather than through Transformers' generic _supports_flash_attn_2 hook.
    # Keep the Transformers dispatch selector on eager to bypass its support
    # gate; `config.use_flash_attention` below selects hEvo's actual kernel.
    transformers_attn_implementation = "eager"
    config.attn_implementation = transformers_attn_implementation
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = transformers_attn_implementation
    if hasattr(config, "_attn_implementation_internal"):
        config._attn_implementation_internal = transformers_attn_implementation
    config.use_flash_attention = use_flash_attention
    config.use_torch_flash_sdp = attention_backend == "torch_flash_sdp"
    backbone = model_module.FlashBertModel(config)
    state_path = model_path / "pytorch_model.bin"
    if state_path.exists():
        state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
    else:
        safetensors_path = model_path / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"No hEvo weights found in {model_path}: expected "
                "pytorch_model.bin or model.safetensors"
            )
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    backbone_state = {
        key.removeprefix("bert."): value
        for key, value in state_dict.items()
        if key.startswith("bert.")
    }
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected hEvo backbone keys: {unexpected[:10]}")
    if missing:
        print(f"[hEvo] missing backbone keys: {missing[:10]}", flush=True)
    return data_module.DNATokenizer(), config, backbone, "pooler"


def _load_standard(model_path: Path):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
    backbone = AutoModel.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    return tokenizer, config, backbone, "mean"


def _load_custom_esm(model_path: Path):
    """Load NT-MS checkpoints with the repository's ESM implementation.

    NT-MS stores a GLU-expanded intermediate projection (2 * intermediate_size)
    and its local ``modeling_esm.py`` matches that checkpoint.  Falling back to
    Transformers' built-in ESM class silently constructs the smaller projection
    and causes a state-dict shape mismatch.
    """
    package_name = f"deepsea_nt_esm_{model_path.name.replace('-', '_')}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_path)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    _load_module(model_path / "esm_config.py", f"{package_name}.esm_config")
    model_module = _load_module(
        model_path / "modeling_esm.py", f"{package_name}.modeling_esm"
    )
    config = model_module.EsmConfig.from_pretrained(str(model_path))
    config.output_hidden_states = True
    from modelgenerator.huggingface_models.rnabert.tokenization_rnabert import (
        RNABertTokenizer,
    )

    # AIDO's vocab is character-level, but its RNABert tokenizer expects the
    # caller to provide the correct tokenizer class.  AutoTokenizer sees the
    # inconsistent ``model_type=bert`` in the checkpoint and returns a
    # WordPiece BertTokenizer, which encodes an entire continuous DNA string
    # as [CLS] [UNK] [SEP].
    tokenizer = RNABertTokenizer(str(model_path / "vocab.txt"), version="v2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    backbone = model_module.EsmModel.from_pretrained(
        str(model_path),
        config=config,
        add_pooling_layer=False,
        torch_dtype=torch.float16,
    )
    return tokenizer, config, backbone, "mean"


def _load_ntv3(model_path: Path):
    """Load NTv3 through its custom AutoModelForMaskedLM registration."""
    from transformers import AutoModelForMaskedLM

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    backbone = AutoModelForMaskedLM.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    return tokenizer, config, backbone, "mean"


def _load_aido(model_path: Path):
    """Load AIDO.DNA through the bundled ModelGenerator RNABert implementation."""
    modelgenerator_root = Path(
        str(
            Path(
                os.environ.get(
                    "DNAMODELS_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
                )
            )
            / "ModelGenerator"
        )
    )
    if str(modelgenerator_root) not in sys.path:
        sys.path.insert(0, str(modelgenerator_root))
    from modelgenerator.huggingface_models.rnabert.configuration_rnabert import (
        RNABertConfig,
    )
    from modelgenerator.huggingface_models.rnabert.modeling_rnabert import RNABertModel

    config = RNABertConfig.from_pretrained(str(model_path))
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=False, use_fast=False
    )
    backbone = RNABertModel.from_pretrained(
        str(model_path),
        config=config,
        add_pooling_layer=False,
        torch_dtype=torch.float16,
    )
    return tokenizer, config, backbone, "mean"


def adapter_family(model_name: str, model_path: str) -> str:
    name = model_name.lower()
    path = model_path.lower()
    if name == "gpn-star-m":
        return "gpn_star"
    if path.endswith(".pt") or name.startswith("gpn"):
        return "unsupported"
    if name.startswith("aido"):
        return "rnabert"
    if name.startswith("ntv3"):
        return "ntv3"
    if "dnabert2" in name:
        return "dnabert2"
    if "hevo" in name or name.startswith("nt") or "gena" in name or "dnabert" in name:
        return "bert_like"
    if "grover" in name:
        return "bert_like"
    if "mistral" in name or "generator" in name:
        return "decoder_like"
    if "caduceus" in name:
        return "caduceus"
    return "auto"


def target_suffixes(family: str) -> list[str]:
    if family in {"bert_like", "rnabert", "gpn_star"}:
        return ["query", "key", "value"]
    if family == "dnabert2":
        return ["Wqkv"]
    if family == "ntv3":
        return ["query_head.linear", "key_head.linear", "value_head.linear"]
    if family == "decoder_like":
        return ["q_proj", "k_proj", "v_proj", "query", "key", "value"]
    if family == "caduceus":
        return [
            "in_proj",
            "out_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "query",
            "key",
            "value",
        ]
    return [
        "query",
        "key",
        "value",
        "q_proj",
        "k_proj",
        "v_proj",
        "in_proj",
        "out_proj",
    ]


def resolve_target_modules(model: nn.Module, family: str) -> list[str]:
    if family == "gpn_star":
        # GPN-Star also owns an embedding table named `value`; target only the
        # query/key/value Linear projections for the supervised adapter.
        projections = [
            name
            for name, module in model.named_modules()
            if name
            and name.rsplit(".", 1)[-1] in {"query", "key", "value"}
            and isinstance(module, nn.Linear)
        ]
        if not projections:
            raise RuntimeError(
                "No GPN-Star Linear query/key/value projections were found"
            )
        return projections
    suffixes = set(target_suffixes(family))
    matches: set[str] = set()
    supported = (nn.Linear, nn.Conv1d, nn.Embedding)
    for name, module in model.named_modules():
        if not name or not isinstance(module, supported):
            continue
        for suffix in suffixes:
            if name == suffix or name.endswith(f".{suffix}"):
                matches.add(suffix)
                break
    targets = [suffix for suffix in target_suffixes(family) if suffix in matches]
    if not targets:
        examples = [name for name, _ in list(model.named_modules())[:40]]
        raise RuntimeError(
            f"No LoRA target modules matched family={family}; "
            f"suffixes={sorted(suffixes)}; module examples={examples}"
        )
    return targets


class DeepSEAClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        config: Any,
        pooling: str = "mean",
        num_labels: int = 919,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.pooling = pooling
        hidden_size = (
            getattr(config, "hidden_size", None)
            or getattr(config, "d_model", None)
            or getattr(config, "embed_dim", None)
        )
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from model config")
        self.dropout = nn.Dropout(
            float(getattr(config, "hidden_dropout_prob", 0.1) or 0.1)
        )
        self.classifier = nn.Linear(int(hidden_size), num_labels)

    def _backbone_forward(self, batch: dict[str, torch.Tensor]):
        kwargs = {
            key: value
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        for key in ("source_ids", "target_species"):
            if key in batch:
                kwargs[key] = batch[key]
        if self.backbone.__class__.__name__.lower().startswith("caduceus"):
            # Caduceus' Mamba forward accepts input_ids/inputs_embeds but not
            # attention_mask.  Keep the mask in the wrapper for pooling, yet
            # do not pass it into the backbone.
            kwargs.pop("attention_mask", None)
            kwargs.pop("token_type_ids", None)
        try:
            return self.backbone(**kwargs, return_dict=True, output_hidden_states=True)
        except TypeError:
            kwargs.pop("token_type_ids", None)
            try:
                return self.backbone(
                    **kwargs, return_dict=True, output_hidden_states=True
                )
            except TypeError:
                return self.backbone(**kwargs)

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        target_species: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # PEFT's feature-extraction wrapper may forward the batch as named
        # transformer arguments instead of preserving our dictionary input.
        if isinstance(input_ids, dict) and batch is None:
            batch = input_ids
            input_ids = None
        batch = dict(batch or {})
        if input_ids is not None:
            batch["input_ids"] = input_ids
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask
        if token_type_ids is not None:
            batch["token_type_ids"] = token_type_ids
        if source_ids is not None:
            batch["source_ids"] = source_ids
        if target_species is not None:
            batch["target_species"] = target_species
        outputs = self._backbone_forward(batch)
        pooled = (
            getattr(outputs, "pooler_output", None)
            if self.pooling == "pooler"
            else None
        )
        if pooled is None:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states:
                    hidden = hidden_states[-1]
            if hidden is None:
                if isinstance(outputs, (tuple, list)):
                    hidden = outputs[0]
                else:
                    raise RuntimeError(
                        f"Cannot extract hidden states from {type(outputs)}"
                    )
            # GPN-Star keeps an explicit target-species axis: (B, L, T, H).
            # DeepSEA uses a single human target, so pool its T=1 axis first.
            if hidden.ndim == 4:
                hidden = hidden.mean(dim=2)
            mask = batch.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                weights = mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
                    1.0
                )
        return self.classifier(self.dropout(pooled))


def load_model_and_tokenizer(
    model_name: str, model_path: str, attention_backend: str = "eager"
):
    path = Path(model_path)
    family = adapter_family(model_name, model_path)
    if family == "unsupported":
        raise RuntimeError(f"UNSUPPORTED_ADAPTER: {model_name} ({model_path})")
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if "hevo" in model_name.lower():
        tokenizer, config, backbone, pooling = _load_hevo(
            path, attention_backend=attention_backend
        )
    elif family == "gpn_star":
        from gpn import register_auto_classes

        register_auto_classes("star")
        config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
        config.output_hidden_states = True
        backbone = AutoModel.from_pretrained(
            str(path), config=config, trust_remote_code=True, torch_dtype=torch.float32
        )
        # GPN-Star consumes native MSA tensors assembled by GPNStarMSACollator;
        # it does not tokenize a single sequence.
        tokenizer = None
        pooling = "mean"
    elif family == "rnabert":
        tokenizer, config, backbone, pooling = _load_aido(path)
    elif family == "ntv3":
        tokenizer, config, backbone, pooling = _load_ntv3(path)
    elif (path.name.endswith("-MS") or path.name.endswith("_MS")) and (
        (path / "esm_config.py").exists() and (path / "modeling_esm.py").exists()
    ):
        # Some NT-MS checkpoints were trained with the repository's custom
        # GLU-expanded ESM implementation, while others contain only the
        # standard Transformers checkpoint files.  Select the custom loader
        # only when its code is actually present; otherwise use the same
        # standard ESM path used during the corresponding training run.
        tokenizer, config, backbone, pooling = _load_custom_esm(path)
    else:
        tokenizer, config, backbone, pooling = _load_standard(path)
    model = DeepSEAClassifier(backbone, config, pooling=pooling)
    targets = resolve_target_modules(model, family)
    return model, tokenizer, config, family, targets


def trainable_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }
