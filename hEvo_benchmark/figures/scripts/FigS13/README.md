# FigS13

Run from the repository root:

```bash
python hEvo_benchmark/figures/scripts/FigS13/A4_integrated.py
```

Use `--tables-dir` and `--output-dir` to override the local table/output folders.
The default destination is `hEvo_benchmark/figures/FigS13` with
`A4_integrated.pdf`, `A4_integrated.png` (600 dpi), and `A4_integrated.svg`.

## Organization

- `A4_integrated.py`: command-line entry point.
- `_reference.py`: S43 and S49 loading, calculations, and data-element placement.
- `assets/reference.pdf`: supplied reference's vector layout, embedded Arial fonts,
  static labels, line/marker styles, and colors.
- `common/reference_pdf.py`: parses original PDF elements and rewrites the numerical
  labels, marks, intervals, and heatmap fills from local tables. The reference PDF
  is a styling asset; numerical plot values come from the S tables.
- `source_data/` in the output folder: table-backed plotted values.

Dependencies: NumPy, pandas,
pypdf, ReportLab, the project's Arial fonts, and Poppler (`pdftoppm`, `pdftocairo`).

## Data definitions

Resource values are measured model means from S43. Throughput is converted from bp/s to kb/s. Available ranks are arithmetic means of S49 task_rank; common ranks use the intersection of task units across all 32 displayed models (currently 35). Original task_rank values are retained without reranking the display subset.

[Back to parent guide](../../README.md)
