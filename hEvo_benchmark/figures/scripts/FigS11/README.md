# FigS11

Run from the repository root:

```bash
python hEvo_benchmark/figures/scripts/FigS11/A4_integrated.py
```

Use `--tables-dir` and `--output-dir` to override the local table/output folders.
The default destination is `hEvo_benchmark/figures/FigS11` with
`A4_integrated.pdf`, `A4_integrated.png` (600 dpi), and `A4_integrated.svg`.

## Organization

- `A4_integrated.py`: command-line entry point.
- `_reference.py`: S41 loading, calculations, and data-element placement.
- `assets/reference.pdf`: supplied reference's vector layout, embedded Arial fonts,
  static labels, line/marker styles, and colors.
- `common/reference_pdf.py`: parses original PDF elements and rewrites the numerical
  labels, marks, intervals, and heatmap fills from local tables. The reference PDF
  is a styling asset; numerical plot values come from the S tables.
- `source_data/` in the output folder: table-backed plotted values.

Dependencies: NumPy, pandas,
pypdf, ReportLab, the project's Arial fonts, and Poppler (`pdftoppm`, `pdftocairo`).

## Data definitions

Heatmap colors normalize MCC within each of the 18 tasks. Bottom dots show SW minus NoSW for each task. Diamonds are the arithmetic group mean across 10 histone tasks, 5 regulatory tasks, or 3 splice tasks.

[Back to parent guide](../../README.md)
