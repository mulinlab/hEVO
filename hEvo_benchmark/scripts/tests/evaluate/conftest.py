"""Keep evaluator implementation modules importable for consolidated tests."""

import sys
from pathlib import Path

EVALUATE_DIR = Path(__file__).resolve().parents[2] / "evaluate"
SCRIPTS_DIR = EVALUATE_DIR.parent
COMPAT_DIR = EVALUATE_DIR / "compat"
for path in (SCRIPTS_DIR, EVALUATE_DIR, COMPAT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
