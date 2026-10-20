from __future__ import annotations

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PEFT checkpoint keys (loads model; requires model dependencies)."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("model_path")
    options = parser.parse_args()
    from safetensors.torch import load_file

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from DeepSEA_lora.adapters import load_model_and_tokenizer  # noqa: E402
    from DeepSEA_lora.run_deepsea_lora import configure_peft  # noqa: E402

    run_dir = options.run_dir
    model_name = options.model_name
    model_path = options.model_path
    model, _tok, _cfg, _family, targets = load_model_and_tokenizer(
        model_name, model_path
    )
    args = type("A", (), {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05})()
    model = configure_peft(model, targets, args)
    state = load_file(str(run_dir / "adapter_model.safetensors"), device="cpu")
    expected = set(model.state_dict())
    saved = set(state)
    print("SAVED", sorted(saved)[:8])
    print("EXPECTED", sorted(expected)[:8])
    print("SAVED_ONLY", sorted(saved - expected)[:20])
    print("EXPECTED_ONLY", sorted(expected - saved)[:20])
    print("SAVED_CLASSIFIER", sorted(x for x in saved if "classifier" in x))


if __name__ == "__main__":
    main()
