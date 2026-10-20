"""Translate task LLR orientation to the quantity stored by each scorer."""


def metric_mode_for_model(model_name: str, task_mode: str) -> str:
    if task_mode not in {"neg_llr", "raw_llr", "abs_llr"}:
        raise ValueError(f"Unknown task score mode: {task_mode}")
    if model_name.startswith("Mistral-DNA-"):
        if task_mode != "raw_llr":
            raise ValueError("Mistral-DNA stores an effect distance, not an LLR")
        return "raw_llr"
    if model_name.startswith("GENERator-"):
        # Its cache stores REF-minus-ALT, the negative of canonical LLR.
        return {"neg_llr": "raw_llr", "raw_llr": "neg_llr", "abs_llr": "abs_llr"}[
            task_mode
        ]
    return task_mode
