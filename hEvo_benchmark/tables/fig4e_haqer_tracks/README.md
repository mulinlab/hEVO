# Fig. 4e hEvo-large and GPN-Star-M tracks

This source bundle contains both hEvo-large and GPN-Star-M tracks used in the current Fig. 4e panel. The combined source table is `Fig4e_track_source_data.csv`.


All data needed to render Fig. 4e are stored in this directory and the retained
RefSeq annotation at `annotations/hg38_refGene.txt.gz`.
The source table contains the 12 selected
held-out HAQER intervals, primary-seed score tracks for hEvo-large and
GPN-Star-M, hEvo peak calls, the original HAQER intervals, cCREs, H3K27ac,
Ape divergence, phastCons, and RefSeq gene models. Coordinates use hg38,
0-based half-open intervals. `selected_segments.json` records the fixed
selection and the shared compressed-axis positions.

The loci were selected from validation and test chromosomes, never training:
hEvo's primary-seed summit falls inside the gold interval, GPN has no called
peak in the displayed 4 kb context, the displayed hEvo local maximum exceeds
the GPN local maximum by at least 0.15, and the hEvo summit peak is at most
1.5 kb wide. HAQER0218 illustrates a locus near COL4A2.
The GPN score track is shown for all 12 loci; the GPN call lane has no intervals
because the selected contexts contain no GPN summit calls by design.

Both models now use a shared score cutoff of 0.59. This cutoff was selected on
the validation split by maximizing macro-averaged summit-hit F1 across the two
models on a 0.01 grid from 0.30 to 0.90; the same value is used on test data.
Peak hit means a called
summit lies inside the original HAQER BED interval.


## Rebuild

1. Generate `selected_segments.json` with
   `figures/scripts/Fig4/e_haqer_gpn_example_selection.py`.
2. On AiStation, run `figures/scripts/Fig4/extract_fig4e_gpn_annotations.py`
   with `--project-root /baotianpeng/XQY/HAR_HAQER_Finetune_Final`, the segment
   manifest, and this directory's `annotations` folder as output.
3. Run `figures/scripts/Fig4/export_fig4e_gpn_source_data.py` from the HPC
   workspace. It validates the selected calls and writes
   `Fig4e_track_source_data.csv`.
4. Render `figures/scripts/Fig4/A4_integrated.py` to refresh the Fig4 PNG, PDF,
   and SVG.

[Back to parent guide](../README.md)
