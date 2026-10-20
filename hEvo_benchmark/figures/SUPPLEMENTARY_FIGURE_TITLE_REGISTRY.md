# Supplementary figure title registry

This index records the publication FigS1–FigS7 topics and panel organization. Panel
letters refer to outputs in the matching figure directory.

| Figure | Title | Orientation | Panel content |
|---|---|---|---|
| FigS1 | Cohort construction and population structure | A4 portrait | Five panels: a preselection-cohort PCA; b eight cohort-highlight views; c chromosome 22 phasing; d chromosome-level haplotype-window counts; e cohort-to-corpus workflow. |
| FigS2 | hEVO architecture | A4 portrait | One unlettered architecture schematic covering sequence tokenization, the bidirectional Transformer, pretraining objectives, and two-stage training. |
| FigS3 | Genomic Benchmark and nucleotide-transformer tasks | A4 portrait | Two panels: a Genomic Benchmark and b NT task-level MCC matrices. |
| FigS4 | Functional class separation in PCA representations | A4 portrait | Three side-by-side dataset columns: ENCODE cCRE, GENCODE, and UCSC RefSeq; three-seed Calinski–Harabasz summaries. |
| FigS5 | Regulatory classification | A4 portrait | Two panels: a five-class regulatory F1; b URE-versus-enhancer MCC across five datasets. |
| FigS6 | DeepSEA chromatin prediction | A4 portrait | One unlettered table with category-macro AUROC and macro AUPRC for S21 models. |
| FigS7 | Zero-shot variant prediction | A4 portrait | Two panels: a AUPRC and b AUROC across six variant tasks and all 33 models shared by the six source tables. |

Figure scripts are under `figures/scripts/FigS1` through `FigS7`. Source
tables and renderer inputs are listed in `tables/README.md` and each output
directory's `source_manifest.json`.

## Companion figures in the batch runner

These figures are rendered by `scripts/run_all_figures.py` and are separate
from the default publication merge.

| Figure | Topic | Figure guide |
|---|---|---|
| FigS10 | Locus-level directional allele preferences and CH separation | [Guide](scripts/FigS10/README.md) |
| FigS11 | Within-corpus sliding-window ablations across NT tasks | [Guide](scripts/FigS11/README.md) |
| FigS12 | Family-level radar comparisons | [Guide](scripts/FigS12/README.md) |
| FigS13 | Resource and task-rank comparisons | [Guide](scripts/FigS13/README.md) |
