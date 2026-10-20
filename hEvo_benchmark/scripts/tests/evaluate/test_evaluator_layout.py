"""Keep the evaluator directory focused on its two public commands."""

from pathlib import Path


def test_only_two_public_python_entrypoints_at_evaluate_root():
    root = Path(__file__).resolve().parents[2] / "evaluate"
    assert {path.name for path in root.glob("*.py")} == {
        "probe_single_seq_input.py",
        "zeroshot_paried_seq_input.py",
    }


def test_implementation_layers_are_importable_packages():
    root = Path(__file__).resolve().parents[2] / "evaluate"
    for name in ("core", "adapters", "runtime", "results_export"):
        assert (root / name / "__init__.py").is_file()
