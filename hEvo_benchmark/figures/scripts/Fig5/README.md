# Fig. 5 source tables and rendering

The integrated renderer is A4_integrated.py. Its four Pareto panels use the
45-task mean rank from Table S49 on the vertical axis and the arithmetic mean of
each resource metric across the 18 NT tasks from S43 on the horizontal axis.
S44 retains task-level NT measurements; S47 records the three GPN-Star model
and MSA revisions.

Run from hEvo_benchmark:

    micromamba -r <micromamba-root> -n llmbm run python figures/scripts/Fig5/A4_integrated.py --tables-dir tables --output-dir <output-dir>

## Panel methods

The supplied `assets/reference.pdf` provides the A4 layout, vector ablation
scheme, embedded Arial fonts, sizes, colors, marker shapes, and label positions.
`A4_integrated.py` calls `_pdf_reference.py`; the common PDF element renderer
updates data marks, numerical labels, and Pareto frontiers directly from the
local tables. Both point layers in the reference are updated.

- a: the reference's ablation design.
- b: mean MCC across all 18 S41 tasks for each corpus/window condition.
- c: S42 passing `all` ClinVar AUPRC for Stage I and Stage II, paired within
  each corpus and window scheme.
- d: predictor parameters (millions) from S43.
- e: embedding latency (milliseconds) from S43.
- f: embedding throughput from S43, converted from bp/s to kb/s.
- g: incremental peak VRAM (MiB) from S43.

The vertical axis in d-g is the equal-weight mean of available S49 task ranks.
Smaller mean ranks appear higher. The bottom model key gives the current overall
ordering by mean rank. The resource axes preserve the reference's logarithmic
ticks and limits. Dashed Pareto frontiers use models with full NT performance,
resource, and rank coverage; open points indicate incomplete coverage. Current
GPN-MSA S49 coverage is 45/45, so its point is filled and it is eligible for the
frontier.

Default outputs are `A4_integrated.pdf`, `A4_integrated.png` (600 dpi), and
`A4_integrated.svg` in `figures/Fig5`. Table-backed plotted values and frontier
members are exported to `source_data/`. Use `--tables-dir` and `--output-dir` to override the source and output
folders; `--export-panels` also refreshes individual panels and companion
panel filenames from the same generated vectors.

For hEvo, the checkpoint backbone contains 87,094,922 parameters for the base
models and 319,462,410 for the large models. S43–S44 predictor totals add the
task-specific probe to those checkpoint counts.

For GPN-Star, S43–S44 count the encoder backbone used to produce embeddings
and exclude the unused masked-LM output head; S47 reports the full
203,022,215-parameter checkpoint.

The GPN-Star-M/P/V NT performance scores are already present in S17. Their NT
efficiency profiles are measured on the NT test split with batch size 8, the
pinned model/MSA revisions, a 1,024-bp inference window, and a 1,000-bp
centered mean of forward/reverse embeddings. Peak VRAM is the increase above
the loaded-model baseline, matching the existing tracker.

URE multiclass rank scores average class-specific F1 values within each data
set; ancestry regression averages the shot sweep; and HAR/HAQER average the
five negative-control runs.

The current rendering path requires NumPy, pandas, pypdf, ReportLab, the
project's Arial fonts, and Poppler (`pdftoppm` and `pdftocairo`).

[Back to parent guide](../../README.md)
