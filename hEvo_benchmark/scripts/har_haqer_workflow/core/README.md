# HAQER workflow components

The `core/` modules implement head fitting, feature extraction, genome-window
scanning, peak calling, and model adapters for the HAQER workflow. Entry-point
scripts in the parent directory combine these components for local runs.

- `16_train_peakshape_heads.py`: scoring-head training and model definitions.
- `46_bootstrap_peakshape_hardneg_from_embeddings.py`: hard-negative
  generation from embeddings.
- `11_call_smoothed_peaks.py`: score smoothing and peak calling.
- `10_scan_genome_region.py`: window scoring over a specified region.

Model and reference locations are configured through command-line arguments
or the `HAQER_*` environment variables described in the parent README. Model
runtimes and their dependencies are installed in the execution environment.

[Back to parent guide](../README.md)
