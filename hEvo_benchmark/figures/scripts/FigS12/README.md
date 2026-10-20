# FigS12

Run from the repository root:

```bash
python hEvo_benchmark/figures/scripts/FigS12/A4_integrated.py
```

Use `--tables-dir` and `--output-dir` to override the local table/output folders.
The default destination is `hEvo_benchmark/figures/FigS12` with
`A4_integrated.pdf`, `A4_integrated.png` (600 dpi), and `A4_integrated.svg`.

## Organization

- `A4_integrated.py`: command-line entry point.
- `_reference.py`: S17-S21, S23, S26-S30 and S33-S38 loading, calculations, and data-element placement.
- `assets/reference.pdf`: supplied reference's vector layout, embedded Arial fonts,
  static labels, line/marker styles, and colors.
- `common/reference_pdf.py`: parses original PDF elements and rewrites the numerical
  labels, marks, intervals, and heatmap fills from local tables. The reference PDF
  is a styling asset; numerical plot values come from the S tables.
- `source_data/` in the output folder: table-backed plotted values.

Dependencies: NumPy, pandas,
pypdf, ReportLab, the project's Arial fonts, and Poppler (`pdftoppm`, `pdftocairo`).

## Data definitions

The latest recorded S21 run is selected per model/task, including S21_run_24 for hEvo-open-large. Radar values are 1 - (rank - 1) / number of eligible models. Genomic uses complete seven-task models; NT uses mean MCC; URE averages S19 mean MCC and S20 macro F1; DeepSEA uses mean AUROC. GWAS averages S26/S27 AUPRC; rare variants uses S28. Ancestry averages S33 rank_score, S34 MCC, and the 60-shot S35 Pearson r. The local GPN-MSA ancestry results enter the current ranks. All other radar spokes use their corresponding table mean. HAR/HAQER ranges represent the min/max of five run means; preference intervals come from S38.

[Back to parent guide](../../README.md)
