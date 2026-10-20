# FigS10

Run from the repository root:

```bash
python hEvo_benchmark/figures/scripts/FigS10/A4_integrated.py
```

Use `--tables-dir` and `--output-dir` to override the local table/output folders.
The default destination is `hEvo_benchmark/figures/FigS10` with
`A4_integrated.pdf`, `A4_integrated.png` (600 dpi), and `A4_integrated.svg`.

## Organization

- `A4_integrated.py`: command-line entry point.
- `_reference.py`: S38 and S39 loading, calculations, and data-element placement.
- `assets/reference.pdf`: supplied reference's vector layout, embedded Arial fonts,
  static labels, line/marker styles, and colors.
- `common/reference_pdf.py`: parses original PDF elements and rewrites the numerical
  labels, marks, intervals, and heatmap fills from local tables. The reference PDF
  is a styling asset; numerical plot values come from the S tables.
- `source_data/` in the output folder: table-backed plotted values.

Dependencies: NumPy, pandas,
pypdf, ReportLab, the project's Arial fonts, and Poppler (`pdftoppm`, `pdftocairo`).

## Data definitions

The 30 locus rows and 31 displayed models retain the reference order. Tile colors come from S39 directional_correct. Preference comes from S38 accuracy; raw-score CH is recalculated from 30 selected and 30 alternative log probabilities in S39.

[Back to parent guide](../../README.md)
