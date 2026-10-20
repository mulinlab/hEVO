"""Resolve the installed Evo2 configuration when loading local weights."""
from pathlib import Path
from tempfile import TemporaryDirectory


def load_local_evo2(model_id, weight_path):
    import evo2
    import torch
    import yaml
    from evo2.utils import CONFIG_MAP

    previous = CONFIG_MAP[model_id]
    config_path = Path(previous)
    if not config_path.is_absolute():
        config_path = Path(evo2.__file__).resolve().parent / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Installed Evo2 config is missing: {config_path}")
    # The installed loader resolves local checkpoints against CONFIG_MAP. Keep
    # the override scoped to initialization and never depend on the job's cwd.
    config = yaml.safe_load(config_path.read_text())
    fp8_unavailable = (
        config.get("use_fp8_input_projections", False)
        and (not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9))
    )
    with TemporaryDirectory(prefix="evo2_local_config_") as directory:
        if fp8_unavailable:
            # This selects the checkpoint's ordinary BF16 linear projections,
            # avoiding a Transformer Engine FP8 kernel unsupported on RTX 3090.
            # No checkpoint tensor is initialized or replaced.
            config["use_fp8_input_projections"] = False
            config_path = Path(directory) / "runtime.yml"
            config_path.write_text(yaml.safe_dump(config))
            print("Evo2: FP8 unavailable; using local checkpoint with BF16 input projections", flush=True)
        CONFIG_MAP[model_id] = str(config_path)
        try:
            model = evo2.Evo2(model_id, str(Path(weight_path).resolve()))
            model.model.eval()
            model.effective_fp8_input_projections = bool(config.get("use_fp8_input_projections", False))
            return model
        finally:
            CONFIG_MAP[model_id] = previous
