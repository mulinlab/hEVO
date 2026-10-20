"""Existing DeepSEA run policy, shared by all derived summaries.

Run selection is explicit, never based on the test metric. Historical rows
remain in S21. The public large-model run is FP16/pooler/one epoch.
"""
DEEPSEA_RUN_OVERRIDES = {"hEvo-open-large": "S21_run_24"}


def active_deepsea_rows(data):
    data = data.copy()
    data["model"] = data.model.replace({"NT-2.5B-1kG": "NT_2.5b_1kG", "NT-2.5B-MS": "NT_2.5b_MS", "NT-500M-1kG": "NT_500m_1kG"})
    for model, run_id in DEEPSEA_RUN_OVERRIDES.items():
        selected = data.model.eq(model)
        if selected.any() and not data.loc[selected, "run_id"].eq(run_id).any():
            raise ValueError(f"S21 is missing the selected {model} run {run_id}")
        data = data.loc[~selected | data.run_id.eq(run_id)].copy()
    if data.duplicated(["model", "task"]).any():
        raise ValueError("S21 must have one explicitly selected run per model/task")
    return data
