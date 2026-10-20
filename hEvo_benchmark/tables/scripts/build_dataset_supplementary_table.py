#!/usr/bin/env python3
"""Build S45 dataset benchmark overview table from GitLab datasets and docs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "tables"
TASKS_YAML = ROOT / "configs" / "tasks.yaml"
OUTPUT = TABLES_DIR / "S45_overview_of_datasets_labels_and_benchmark_sample_sizes.csv"
WORKBOOK = TABLES_DIR / "hEvo_benchmark_dataset_supplementary_tables.xlsx"
SHEET_NAME = "S45_dataset_overview"


BIOLOGY = {
    "NT_Pareto": "regulatory element, histone mark, promoter, and splice-site recognition from single DNA sequences",
    "GenomicBenchmark": "human regulatory, promoter, enhancer, OCR, coding/intergenic, and species-related sequence classification",
    "URE_binary_classification": "unlabeled regulatory element discrimination against enhancer controls",
    "URE_multi_classification": "regulatory element subtype classification across URE, ELS, PLS, insulator, and silencer classes",
    "ClinVar260615": "ClinVar pathogenic versus benign SNV effect scoring from REF/ALT sequence likelihoods",
    "MPRA": "functional regulatory variant effect scoring using MPRA-supported fine-mapped variants",
    "Causal_Variants": "disease causal variant prioritization against MAF-matched negative variants",
    "Rare_Causal_Variants": "rare causal variant prioritization against matched rare negatives",
    "Rare_Variants": "UKB rare variant prioritization against rare-variant negatives",
    "Denovo_Variants": "public DDD de novo LoF/splice SNV control benchmarks",
    "Evolution": "rapidly evolving region recognition for HAQER and HAR sequence sets",
    "Adaptation_Variants": "verified adaptive haplotype retrieval from REF/ALT sequence pairs",
    "DeepSEA": "multi-label chromatin feature prediction benchmark grouped by DHS, HM, and TF annotations",
    "Ancestry_Classification": "1000 Genomes super-population ancestry classification and admixture regression",
}

SOURCE_SUMMARY = {
    "NT_Pareto": "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised exported by datasets/NT_benchmark_v2/prepare_nt_benchmark_v2.py",
    "GenomicBenchmark": "katielink/genomic-benchmarks seven human-related tasks exported by datasets/GenomicBenchmark/prepare_genomic_benchmark.py",
    "URE_binary_classification": "BED-like URE source files under datasets/URE_binary_classification/bed",
    "URE_multi_classification": "BED-like URE multiclass source files under datasets/URE_multi_classification/bed",
    "ClinVar260615": "ClinVar 2026-06-15 ANNOVAR Benign/Pathogenic source tables rebuilt by datasets/newClinvar_260615/build_clinvar_260615.py",
    "MPRA": "Siraj et al. Nature 2026 Supplementary Table 3 plus MPRA emVar annotations",
    "Causal_Variants": "CausalDB disease variant exports with MAF-matched negatives",
    "Rare_Causal_Variants": "rare causal and matched rare negative BED sources",
    "Rare_Variants": "UKB490k rare variant positives and rare negative BED pool",
    "Denovo_Variants": "DECIPHER public DDD Research Variants endpoint plus ClinVar benign controls or same-position other alleles",
    "Evolution": "curated HAQER/HAR positives with negatives from the dedicated Evolution negative pool",
    "Adaptation_Variants": "verified adaptive SNP table mapped to hg38 and 1000 Genomes phased haplotypes",
    "DeepSEA": "DeepSEA public train/valid/test bundle documented in datasets/DeepSEA/README.md",
    "Ancestry_Classification": "1000 Genomes high-coverage phased VCFs and top population-specific marker variants",
}


def load_tasks() -> list[dict]:
    payload = yaml.safe_load(TASKS_YAML.read_text(encoding="utf-8"))
    return payload["Tasks"]


def resolve_dataset_path(path_template: str) -> Path:
    return Path(path_template.replace("${BENCHMARK_ROOT}", str(ROOT)))


def read_csv_counts(path: Path, label_col: int | str, sep: str = ",") -> tuple[int, dict[str, int]]:
    if not path.exists():
        return 0, {}
    df = pd.read_csv(path, sep=sep)
    if isinstance(label_col, int):
        labels = df.iloc[:, label_col]
    else:
        labels = df[label_col]
    counts = labels.value_counts(dropna=False).sort_index()
    return int(len(df)), {str(k): int(v) for k, v in counts.items()}


def combine_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for key, value in b.items():
        out[key] = out.get(key, 0) + int(value)
    return out


def counts_to_json(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return json.dumps(counts, ensure_ascii=False, sort_keys=True)


def binary_fields(counts: dict[str, int]) -> tuple[str, str, str]:
    neg = counts.get("0", "")
    pos = counts.get("1", "")
    total = sum(counts.values()) if counts else 0
    baseline = ""
    if total and isinstance(pos, int):
        baseline = f"{pos / total:.6g}"
    return neg, pos, baseline


def task_eval_type(supername: str, subtask: dict) -> str:
    if subtask.get("seqtype") == "paired":
        return "variant_effect_scoring"
    if supername == "Adaptation_Variants":
        return "variant_effect_scoring"
    return "sequence_classification"


def add_task_rows() -> list[dict]:
    rows: list[dict] = []
    for task in load_tasks():
        supername = task["supername"]
        dataset_dir = resolve_dataset_path(task["path"])
        for sub in task["subtasks"]:
            train_path = dataset_dir / sub["train_path"].lstrip("/")
            test_path = dataset_dir / sub["test_path"].lstrip("/")
            label_col = int(sub.get("label_col", 2))
            sep = sub.get("sep", ",")
            train_n, train_counts = read_csv_counts(train_path, label_col, sep)
            test_n, test_counts = read_csv_counts(test_path, label_col, sep)
            same_file = train_path == test_path
            if same_file:
                total_n = train_n
                total_counts = train_counts
            else:
                total_n = train_n + test_n
                total_counts = combine_counts(train_counts, test_counts)
            neg, pos, baseline = binary_fields(total_counts) if int(sub.get("classes", 0)) == 2 else ("", "", "")
            rows.append(
                {
                    "table_id": "S45",
                    "dataset_group": supername,
                    "dataset_name": sub["name"],
                    "dataset_path": str(dataset_dir.relative_to(ROOT)),
                    "train_file": sub["train_path"],
                    "test_file": sub["test_path"],
                    "data_source": SOURCE_SUMMARY.get(supername, ""),
                    "biological_meaning": BIOLOGY.get(supername, ""),
                    "evaluation_type": task_eval_type(supername, sub),
                    "sequence_input_type": "paired_ref_alt" if sub.get("seqtype") == "paired" else "single_sequence",
                    "num_classes": int(sub.get("classes", "")),
                    "class_labels": "",
                    "total_samples": total_n if total_n else "",
                    "train_samples": train_n if train_n else "",
                    "test_samples": "" if same_file or not test_n else test_n,
                    "negative_samples": neg,
                    "positive_samples": pos,
                    "class_counts": counts_to_json(total_counts),
                    "auprc_baseline": baseline,
                    "count_source": "dataset_csv" if total_n else "not_tracked_in_gitlab",
                    "notes": "",
                }
            )
    return rows


def add_nt_summary_if_needed(rows: list[dict]) -> None:
    summary_path = ROOT / "datasets" / "NT_benchmark_v2" / "summary.tsv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, sep="\t")
    for row in rows:
        if row["dataset_group"] != "NT_Pareto" or row["count_source"] == "dataset_csv":
            continue
        sub = summary[summary["task"] == row["dataset_name"]]
        if sub.empty:
            continue
        total = int(sub["num_rows"].sum())
        neg = int(sub["num_negative"].sum()) if "num_negative" in sub else ""
        pos = int(sub["num_positive"].sum()) if "num_positive" in sub else ""
        counts: dict[str, int] = {}
        for value in sub["label_counts_json"]:
            counts = combine_counts(counts, {str(k): int(v) for k, v in json.loads(value).items()})
        row.update(
            {
                "total_samples": total,
                "train_samples": int(sub.loc[sub["split"] == "train", "num_rows"].sum()),
                "test_samples": int(sub.loc[sub["split"] == "test", "num_rows"].sum()),
                "negative_samples": neg if row["num_classes"] == 2 else "",
                "positive_samples": pos if row["num_classes"] == 2 else "",
                "class_counts": counts_to_json(counts),
                "auprc_baseline": f"{pos / total:.6g}" if row["num_classes"] == 2 and total else "",
                "count_source": "datasets/NT_benchmark_v2/summary.tsv",
            }
        )


def add_manual_rows(rows: list[dict]) -> None:
    # DeepSEA is intentionally not in configs/tasks.yaml because it uses a dedicated
    # multi-label evaluation pipeline. The full processed labels are not tracked.
    sample_train = ROOT / "datasets" / "DeepSEA" / "sample" / "train_labels.sample.npy"
    sample_note = ""
    if sample_train.exists():
        labels = np.load(sample_train)
        sample_note = f"smoke sample train_labels shape={tuple(labels.shape)}"
    rows.append(
        {
            "table_id": "S45",
            "dataset_group": "DeepSEA",
            "dataset_name": "DeepSEA_DHS_HM_TF_multilabel",
            "dataset_path": "datasets/DeepSEA",
            "train_file": "processed/train_out.csv",
            "test_file": "processed/test_out.csv",
            "data_source": SOURCE_SUMMARY["DeepSEA"],
            "biological_meaning": BIOLOGY["DeepSEA"],
            "evaluation_type": "sequence_classification",
            "sequence_input_type": "single_sequence_multilabel",
            "num_classes": "",
            "class_labels": "multi-label chromatin features; reported groups DHS|HM|TF",
            "total_samples": "",
            "train_samples": "",
            "test_samples": "",
            "negative_samples": "",
            "positive_samples": "",
            "class_counts": "",
            "auprc_baseline": "",
            "count_source": "full_processed_files_not_tracked_in_gitlab",
            "notes": sample_note,
        }
    )

    # Event-centric ClinVar reclassification inputs are supplied from the
    # external run directory. Their sample counts vary by seed, cutoff year,
    # and transition task, so the source TSVs remain the count authority.
    rows.append(
        {
            "table_id": "S45",
            "dataset_group": "ClinVar_Eventcentric_Reclassification",
            "dataset_name": "ClinVar260707_Eventcentric_5seed_noMT",
            "dataset_path": "datasets/ClinVar260707_Eventcentric_5seed_noMT",
            "train_file": "N/A (zero-shot scoring)",
            "test_file": "external source root/seed{01..05}/tasks/*_{task}.tsv",
            "data_source": "External ClinVar event-centric sequence and task tables; runtime assets are not tracked in GitLab",
            "biological_meaning": "retrospective ClinVar label-reclassification event enrichment",
            "evaluation_type": "variant_reclassification_event_enrichment",
            "sequence_input_type": "paired_ref_alt",
            "num_classes": 2,
            "class_labels": "task-specific 0=stable/no event|1=ClinVar label transition",
            "total_samples": "Variable by seed, cutoff year, and transition task",
            "train_samples": "N/A (zero-shot scoring)",
            "test_samples": "Variable; recorded in external task TSVs",
            "negative_samples": "Variable by task, cutoff year, and seed",
            "positive_samples": "Variable by task, cutoff year, and seed",
            "class_counts": "Variable by task, cutoff year, and seed",
            "auprc_baseline": "Variable by task, cutoff year, and seed",
            "count_source": "external_eventcentric_task_files_not_tracked_in_gitlab",
            "notes": "Three transitions: blb_stable_vs_upgraded, plp_stable_vs_downgraded, and vus_resolved_path_vs_benign; five deterministic seeds; cutoff years 2017-2025. The dataset name specifies nuclear variants only. The primary metric is Top-1% enrichment.",
        }
    )

    # Each released genomic-element CSV defines an independent unsupervised
    # clustering task. The full dataset is evaluated without a train/test split.
    elements_root = ROOT / "datasets" / "evaluation_elements"
    elements = [
        ("source_taxonomy_v1", "encode_ccre_v4_regulatory.csv", "ENCODE cCRE V4 regulatory elements", 8),
        ("source_taxonomy_v1", "gencode_v45_gene_structure.csv", "GENCODE v45 gene structure", 5),
        ("taxonomy_v2", "encode_ccre_v4_regulatory.csv", "ENCODE cCRE V4 regulatory elements", 5),
        ("taxonomy_v2", "gencode_v45_gene_structure.csv", "GENCODE v45 gene structure", 3),
        ("ucsc_refseq_curated", "ucsc_refseq_curated_gene_structure.csv", "UCSC RefSeq curated gene structure", 5),
    ]
    for release, filename, meaning, n_classes in elements:
        data_file = Path("data") / release / filename
        full_path = elements_root / data_file
        total_n = 0
        counts: dict[str, int] = {}
        if full_path.exists():
            frame = pd.read_csv(full_path, usecols=["label"])
            total_n = int(len(frame))
            counts = {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()}
        rows.append(
            {
                "table_id": "S45",
                "dataset_group": "Genomic_Elements_Unsupervised_Clustering",
                "dataset_name": f"{release}_{Path(filename).stem}",
                "dataset_path": f"datasets/evaluation_elements/{data_file.parent.as_posix()}",
                "train_file": "N/A (unsupervised evaluation; no training split)",
                "test_file": data_file.as_posix(),
                "data_source": f"Released dataset in datasets/evaluation_elements/{data_file.as_posix()}",
                "biological_meaning": meaning,
                "evaluation_type": "unsupervised_clustering",
                "sequence_input_type": "single_sequence",
                "num_classes": n_classes,
                "class_labels": "|".join(counts) if counts else "See dataset README",
                "total_samples": total_n if total_n else "Not available",
                "train_samples": "N/A (unsupervised evaluation)",
                "test_samples": total_n if total_n else "Not available",
                "negative_samples": "",
                "positive_samples": "",
                "class_counts": counts_to_json(counts) if counts else "See dataset README",
                "auprc_baseline": "",
                "count_source": f"datasets/evaluation_elements/{data_file.as_posix()}",
                "notes": "Single unsplit evaluation set; clustering separation is assessed against dataset-native labels. Protocol and label definitions are documented in datasets/evaluation_elements/README.md.",
            }
        )

    # Ancestry classification is built as JSONL and evaluated through embedding
    # classification/few-shot workflows, not as a configs/tasks.yaml probe task.
    counts = {"AFR": 300, "AMR": 300, "EAS": 300, "EUR": 300, "SAS": 300}
    rows.append(
        {
            "table_id": "S45",
            "dataset_group": "Ancestry_Classification",
            "dataset_name": "1KG_superpopulation_ancestry",
            "dataset_path": "datasets/Ancestry_Classification",
            "train_file": "jsonl_2000v_1500s_511bp/",
            "test_file": "jsonl_2000v_1500s_511bp/",
            "data_source": SOURCE_SUMMARY["Ancestry_Classification"],
            "biological_meaning": BIOLOGY["Ancestry_Classification"],
            "evaluation_type": "sequence_classification",
            "sequence_input_type": "single_sequence",
            "num_classes": 5,
            "class_labels": "AFR|AMR|EAS|EUR|SAS",
            "total_samples": "1500 samples; 15000000 sequence records in production target",
            "train_samples": "",
            "test_samples": "",
            "negative_samples": "",
            "positive_samples": "",
            "class_counts": counts_to_json(counts),
            "auprc_baseline": "",
            "count_source": "datasets/Ancestry_Classification/README.md",
            "notes": "Production JSONL is intentionally not tracked; README specifies 300 samples per super-population and 10000 records per sample.",
        }
    )


def fill_clinvar_from_source_tables(rows: list[dict]) -> None:
    mapping = {
        "ClinVar260615_all": "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv",
        "ClinVar260615_coding": "S24_zero_shot_classification_of_coding_clinvar_variants.csv",
        "ClinVar260615_noncoding": "S25_zero_shot_classification_of_noncoding_clinvar_variants.csv",
    }
    for row in rows:
        source = mapping.get(row["dataset_name"])
        if not source:
            continue
        path = TABLES_DIR / source
        if not path.exists():
            continue
        df = pd.read_csv(path)
        unique = df[["num_samples", "num_positive", "num_negative", "baseline"]].drop_duplicates()
        if len(unique) != 1:
            continue
        stats = unique.iloc[0]
        neg = int(stats["num_negative"])
        pos = int(stats["num_positive"])
        total = int(stats["num_samples"])
        row.update(
            {
                "total_samples": total,
                "train_samples": total,
                "test_samples": "",
                "negative_samples": neg,
                "positive_samples": pos,
                "class_counts": counts_to_json({"0": neg, "1": pos}),
                "auprc_baseline": f"{float(stats['baseline']):.6g}",
                "count_source": f"tables/{source}",
            }
        )


def enrich_from_readmes(rows: list[dict]) -> None:
    labels = {
        "URE_binary_classification": "0=Enhancer|1=URE",
        "URE_multi_classification": "0=URE|1=ELS|2=PLS|3=Insulator|4=Silencer",
        "ClinVar260615": "0=Benign|1=Pathogenic",
        "MPRA": "0=control without emVar support|1=fine-mapped UKBB/BBJ/GTEx variant with emVar support",
        "Causal_Variants": "0=MAF-matched negative|1=disease causal variant",
        "Rare_Causal_Variants": "0=matched rare negative|1=rare causal variant",
        "Rare_Variants": "0=rare causal negative|1=UKB490k rare variant positive",
        "Denovo_Variants": "0=control allele/benign comparator|1=DDD LoF/splice SNV",
        "Evolution": "0=intergenic negative|1=HAR/HAQER positive",
        "Adaptation_Variants": "0=REF/non-adaptive haplotype|1=verified adaptive ALT haplotype",
    }
    for row in rows:
        row["class_labels"] = row["class_labels"] or labels.get(row["dataset_group"], "")
        if row["dataset_group"] == "Denovo_Variants" and row["dataset_name"].endswith("same_position_other_alleles"):
            row["notes"] = "Grouped paired top-1 task: one DDD alternate allele and two same-position other alternate alleles per valid group."
        if row["dataset_group"] == "Evolution":
            row["notes"] = "Five fixed negative-sampling replicates; results are reported as mean across replicates."


def autosize_columns(worksheet, max_width: int = 80) -> None:
    for column_cells in worksheet.columns:
        letter = column_cells[0].column_letter
        width = 0
        for cell in column_cells:
            width = max(width, len(str(cell.value)) if cell.value is not None else 0)
        worksheet.column_dimensions[letter].width = min(max(width + 2, 10), max_width)


def write_workbook(df: pd.DataFrame) -> None:
    zh_index = pd.DataFrame(
        [
            {
                "附表编号": "S45",
                "工作表": SHEET_NAME,
                "文件名": OUTPUT.name,
                "内容": "数据集与评估任务总览，包括数据来源、生物学意义、任务类型、类别数、类别/正负样本计数和二分类 AUPRC 基线。",
                "行数": len(df),
                "列数": len(df.columns),
                "备注": "count_source 字段记录计数来源；not_tracked_in_gitlab 表示对应生成数据未纳入当前 GitLab 数据集目录。",
            }
        ]
    )
    en_index = pd.DataFrame(
        [
            {
                "table_id": "S45",
                "sheet": SHEET_NAME,
                "file_name": OUTPUT.name,
                "content": "Dataset and benchmark-task overview with source, biological meaning, task type, class count, class/binary counts, and binary AUPRC baseline.",
                "rows": len(df),
                "columns": len(df.columns),
                "note": "The count_source field records provenance; not_tracked_in_gitlab means the generated dataset is documented but not tracked in the GitLab dataset directory.",
            }
        ]
    )
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        zh_index.to_excel(writer, index=False, sheet_name="目录_中文")
        en_index.to_excel(writer, index=False, sheet_name="Index_EN")
        df.to_excel(writer, index=False, sheet_name=SHEET_NAME)
        for worksheet in writer.book.worksheets:
            autosize_columns(worksheet)
            worksheet.freeze_panes = "A2"


def main() -> None:
    rows = add_task_rows()
    add_nt_summary_if_needed(rows)
    add_manual_rows(rows)
    fill_clinvar_from_source_tables(rows)
    enrich_from_readmes(rows)
    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset_group", "dataset_name"]).reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    write_workbook(df)
    print(f"Saved {OUTPUT} rows={len(df)} cols={len(df.columns)}")
    print(f"Saved {WORKBOOK}")
    print(df[["dataset_group", "dataset_name", "evaluation_type", "num_classes", "total_samples", "class_counts", "auprc_baseline", "count_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
