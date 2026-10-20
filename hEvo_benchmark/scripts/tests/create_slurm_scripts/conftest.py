"""Keep Slurm generator modules importable for their focused tests."""

import sys
from pathlib import Path

GENERATOR_DIR = Path(__file__).resolve().parents[2] / "create_slurm_scripts"
if str(GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATOR_DIR))
