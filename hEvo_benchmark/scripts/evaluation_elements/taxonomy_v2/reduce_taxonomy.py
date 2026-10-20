"""Run the frozen reducer with the taxonomy-v2 class/row contracts."""

import importlib.util
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_reducer", CODE_ROOT / "evaluation/reduce_embeddings.py")
reducer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reducer)
reducer.DATASET_ROWS = {
    "encode_ccre_v4_regulatory": (5000, 5),
    "gencode_v45_gene_structure": (3000, 3),
}
sys.argv[0] = str(CODE_ROOT / "evaluation/reduce_embeddings.py")
reducer.main()
