"""Lightweight regression checks: parsers, cache metadata, callbacks and SLURM.

Run in the documented scientific environment; all outputs use temporary paths.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "evaluate"))
sys.path.insert(0, str(ROOT / "create_slurm_scripts"))
from config_utils import choose_path, resolve_supername  # noqa: E402 - local workflow modules
from core.probe_common import bootstrap_ci  # noqa: E402 - local workflow modules


def isolated_function(path, name, namespace):
    """Load a function AST without importing a model/runtime module."""
    tree = ast.parse(path.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "metadata", [None, {"fingerprint": "synthetic", "model": "test"}]
)
def test_legacy_cache_optional_metadata(tmp_path, metadata):
    namespace = {
        "np": np,
        "h5py": h5py,
        "os": os,
        "json": json,
        "tqdm": SimpleNamespace(tqdm=lambda values, **kw: values),
    }
    function = isolated_function(
        ROOT / "evaluate/zeroshot_paried_seq_input.py",
        "precompute_and_cache_logits_h5",
        namespace,
    )
    output = tmp_path / "empty.h5"
    function(None, [], str(output), cache_metadata=metadata)
    with h5py.File(output) as handle:
        assert (
            len(handle["llrs"]) == len(handle["labels"]) == len(handle["details"]) == 0
        )
        assert handle.attrs["num_samples"] == 0
        if metadata:
            assert handle.attrs["fingerprint"] == "synthetic"


def test_probe_bootstrap_seed_and_multiclass():
    labels = [0, 1, 2] * 8
    predictions = [0, 1, 1] * 8
    probabilities = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.6, 0.3]] * 8
    first = bootstrap_ci(labels, predictions, probabilities, 30, 42)
    second = bootstrap_ci(labels, predictions, probabilities, 30, 42)
    assert first == second
    assert {"accuracy", "mcc", "auc", "auc_macro", "auc_micro"} <= first.keys()
    assert first["auc"] == first["auc_macro"]


def test_precompute_uses_protocol_callbacks(tmp_path):
    import pickle

    import torch
    from core.probe_common import precompute_and_save_embeddings

    observed = {}

    def dataset_type(**options):
        observed.update(options)
        return [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor(1)}]

    def collate_batch(rows):
        observed["collated"] = True
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}

    extractor = SimpleNamespace(
        device="cpu",
        tokenize=lambda sequence: sequence,
        get_embedding=lambda **options: options["input_ids"].float(),
    )
    precompute_and_save_embeddings(
        extractor,
        ["AC"],
        [1],
        str(tmp_path),
        "test",
        num_workers=0,
        dataset_type=dataset_type,
        collate_batch=collate_batch,
    )
    assert observed["collated"] and observed["sequences"] == ["AC"]
    with (tmp_path / "test_batch_0.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    assert saved["embeddings"].tolist() == [[1.0, 2.0]]
    assert saved["labels"].tolist() == [1]


def test_ancestry_regression_probabilities():
    path = ROOT / "population_classify/Regression_with_admixture.py"
    softmax = isolated_function(path, "softmax", {"np": np})
    values = softmax(np.array([[1000.0, 1000.0, 1000.0], [1000.0, 1001.0, 999.0]]))
    np.testing.assert_allclose(values.sum(axis=1), 1.0)
    np.testing.assert_allclose(values[0], [1 / 3] * 3)
    assert values[1, 1] > values[1, 0] > values[1, 2]


def test_slurm_path_precedence_and_unknown_task(tmp_path):
    assert choose_path(tmp_path, "/other", Path("fallback")) == tmp_path
    assert choose_path(None, str(tmp_path), Path("fallback")) == tmp_path
    with pytest.raises(KeyError, match="Unknown supername"):
        resolve_supername("missing", ["task"])


@pytest.mark.parametrize(
    "script,evaluator,h5_option",
    [
        ("create_probe_single_seq.py", "probe_single_seq_input.py", "--hdf5-root"),
        ("create_zeroshot_paired_seq.py", "zeroshot_paried_seq_input.py", "--h5-root"),
    ],
)
def test_slurm_generator_without_submission(tmp_path, script, evaluator, h5_option):
    config = {
        "Models": [
            {
                "name": "hEvo-base",
                "url": str(tmp_path / "checkpoint"),
                "max_length": 1024,
            }
        ],
        "Tasks": [
            {
                "supername": "synthetic",
                "path": str(tmp_path),
                "subtasks": [
                    {
                        "name": "task",
                        "train_path": "/train.csv",
                        "test_path": "/test.csv",
                        "type": "classification",
                        "seqtype": "paired",
                        "classes": 2,
                        "sequence_col": 1,
                        "label_col": 3,
                        "idx_col": 0,
                        "ref_col": 1,
                        "alt_col": 2,
                        "sep": ",",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    evaluator_path = ROOT / "evaluate" / evaluator
    command = [
        sys.executable,
        str(ROOT / "create_slurm_scripts" / script),
        "--config-path",
        str(path),
        "--supername",
        "synthetic",
        "--python-exec",
        sys.executable,
        "--evaluate-script",
        str(evaluator_path),
        h5_option,
        str(tmp_path / "h5"),
    ]
    for option in ["--sbatch-root", "--slurm-root", "--results-root", "--log-root"]:
        command += [option, str(tmp_path / option[2:])]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    jobs = list((tmp_path / "sbatch-root").rglob("*")) + list(
        (tmp_path / "slurm-root").rglob("*")
    )
    scripts = [p for p in jobs if p.is_file()]
    assert scripts
    assert any(str(evaluator_path) in p.read_text() for p in scripts)
    for generated in scripts:
        subprocess.run(["bash", "-n", str(generated)], check=True, capture_output=True)


@pytest.mark.parametrize(
    "relative",
    [
        "evaluate/probe_single_seq_input.py",
        "evaluate/compat/probe_single_seq_input_Pareto.py",
        "evaluate/zeroshot_paried_seq_input.py",
        "evaluate/compat/probe_single_seq_sharded.py",
        "DeepSEA_lora/run_deepsea_lora.py",
        "DeepSEA_lora/run_evo2_deepsea.py",
        "DeepSEA_lora/evaluate_category_metrics.py",
        "DeepSEA_lora/evaluate_evo2_category_metrics.py",
        "adaptive_allele_preference_benchmark/05_score_generic_model.py",
    ],
)
def test_lightweight_help_without_site_packages(tmp_path, relative):
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / relative), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert not list(tmp_path.iterdir())
