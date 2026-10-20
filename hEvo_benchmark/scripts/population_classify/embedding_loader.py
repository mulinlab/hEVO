import glob
import os
from typing import Dict, List, Tuple

import numpy as np


FUSED_MODEL_SPECS = {}


def available_model_names(base_dir: str) -> List[str]:
    return sorted(
        d.replace("_popu_emb", "")
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.endswith("_popu_emb")
    )


def resolve_component_models(model_name: str) -> List[str]:
    if model_name in FUSED_MODEL_SPECS:
        return list(FUSED_MODEL_SPECS[model_name])
    if "+" in model_name:
        return [part.strip() for part in model_name.split("+") if part.strip()]
    return [model_name]


def _normalize_loaded_array(loaded: np.ndarray) -> np.ndarray:
    if loaded.dtype == object:
        vec = np.asarray(loaded.item()["embedding"], dtype=np.float32)
    elif loaded.ndim > 1:
        vec = np.asarray(np.mean(loaded, axis=0), dtype=np.float32)
    else:
        vec = np.asarray(loaded, dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def load_single_embedding_map(emb_dir: str) -> Dict[str, Tuple[str, np.ndarray]]:
    rows = {}
    for path in sorted(glob.glob(os.path.join(emb_dir, "*.npy"))):
        filename = os.path.basename(path)
        pop_label, sample_id_ext = filename.split("_", 1)
        sample_id = sample_id_ext.replace(".npy", "")
        loaded = np.load(path, allow_pickle=True)
        rows[sample_id] = (pop_label, _normalize_loaded_array(loaded))
    return rows


def load_embedding_map_for_model(
    base_dir: str, model_name: str
) -> Dict[str, Tuple[str, np.ndarray]]:
    component_models = resolve_component_models(model_name)
    component_maps = []
    for component in component_models:
        emb_dir = os.path.join(base_dir, f"{component}_popu_emb")
        if not os.path.isdir(emb_dir):
            raise RuntimeError(f"Missing embedding directory for model: {component}")
        component_maps.append(load_single_embedding_map(emb_dir))

    shared_ids = set(component_maps[0].keys())
    for current_map in component_maps[1:]:
        shared_ids &= set(current_map.keys())
    if not shared_ids:
        raise RuntimeError(f"No shared samples found for model: {model_name}")

    fused_rows = {}
    for sample_id in sorted(shared_ids):
        pop_label = component_maps[0][sample_id][0]
        fused_rows[sample_id] = (
            pop_label,
            np.concatenate(
                [current_map[sample_id][1] for current_map in component_maps], axis=0
            ).astype(np.float32),
        )
    return fused_rows


def load_embedding_rows_for_model(
    base_dir: str, model_name: str
) -> List[Tuple[str, str, np.ndarray]]:
    emb_map = load_embedding_map_for_model(base_dir, model_name)
    return [
        (sample_id, pop_label, vec) for sample_id, (pop_label, vec) in emb_map.items()
    ]


def format_fused_model_note(model_name: str) -> str:
    component_models = resolve_component_models(model_name)
    if len(component_models) == 1:
        return model_name
    return f"{model_name} ({' + '.join(component_models)})"
