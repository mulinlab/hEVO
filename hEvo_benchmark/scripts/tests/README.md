# Script tests

This directory keeps focused evaluator and scheduler-generator tests outside
production script entry-point folders. Run from the repository root in the
`llmbm` environment, which provides PyTorch, NumPy, pandas, h5py, scikit-learn,
PyYAML and pytest.

```bash
python -m pytest hEvo_benchmark/scripts/tests/evaluate \
  hEvo_benchmark/scripts/tests/create_slurm_scripts
python -m pytest hEvo_benchmark/scripts/test_workflows.py
```

The evaluator test `conftest.py` adds `scripts/evaluate/` to the import path;
the generator tests use a separate conftest for `scripts/create_slurm_scripts/`.
Test outputs use temporary directories. The workflow test includes lightweight
CLI and generated-Slurm syntax checks.
