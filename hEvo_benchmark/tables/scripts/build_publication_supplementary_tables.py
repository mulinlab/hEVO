#!/usr/bin/env python3
"""Reconcile the publication's 32-sheet workbook to the current repository tables.

This intentionally edits the publication workbook in place. The separate
build_supplementary_tables_excel.py exports the available repository S-series
tables and must not be used for the publication workbook.
"""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path
import sys

import pandas as pd
from openpyxl import load_workbook


TABLES = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = TABLES / "Supplementary Tables.xlsx"

SOURCE_TO_DISPLAY = {
    "evo2_7b_base": "Evo2-7B",
    "evo2_1b_base": "Evo2-1B",
    "GPN-promoter": "GPN-Promoter",
    "HyenaDNA_tiny_1k": "HyenaDNA-1K",
    "NT_2.5b_1kG": "NTv1-2.5B-1kG",
    "NT_2.5b_MS": "NTv1-2.5B-MS",
    "NT_500m_1kG": "NTv1-500M-1kG",
    "NT_500m_MS": "NTv2-500M-MS",
    "NTv3-100M-pre": "NTv3-100M",
    "NTv3-650M-pre": "NTv3-650M",
    "NTv3-8M-pre": "NTv3-8M",
    "hEvo-base": "hEVO-base",
    "hEvo-large": "hEVO-large",
    "hEvo-open-base": "hEVO-base-o",
    "hEvo-open-large": "hEVO-large-o",
}


def display_model(model: object) -> str:
    value = str(model)
    return SOURCE_TO_DISPLAY.get(value, value)


def read_csv(pattern: str) -> pd.DataFrame:
    matches = sorted(TABLES.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one source for {pattern!r}, found {matches}")
    return pd.read_csv(matches[0])


def set_values(ws, row: int, columns: list[int], values: list[object]) -> None:
    for column, value in zip(columns, values, strict=True):
        if pd.isna(value):
            ws.cell(row, column).value = "N/A"
        elif hasattr(value, "item"):
            ws.cell(row, column).value = value.item()
        else:
            ws.cell(row, column).value = value


def reconcile_genomic_benchmark(wb) -> int:
    ws = wb["S4"]
    source = read_csv("S18_frozen_representation_classification_across_seven_genomic_benchmark_tasks.csv")
    rows: dict[tuple[str, str], int] = {}
    for row in range(3, ws.max_row + 1):
        key = (str(ws.cell(row, 2).value), str(ws.cell(row, 3).value))
        if key in rows:
            raise ValueError(f"duplicate key in S4: {key}")
        rows[key] = row

    # The two standard demo tasks have real, source-backed GPN-MSA results.
    inserted = 0
    missing = []
    for item in source.itertuples(index=False):
        label = display_model(item.model)
        key = (str(item.dataset), label)
        if key not in rows:
            missing.append((item, key))
    for item, key in missing:
        label = display_model(item.model)
        # Put the missing comparator alongside the other GPN models in its task block.
        after = next(
            (
                row
                for row in range(3, ws.max_row + 1)
                if str(ws.cell(row, 2).value) == key[0]
                and str(ws.cell(row, 3).value) == "GPN-Promoter"
            ),
            None,
        )
        if after is None:
            raise ValueError(f"cannot place missing S4 source row {key}")
        insert_at = after + 1
        ws.insert_rows(insert_at)
        for column in range(1, ws.max_column + 1):
            source_cell = ws.cell(after, column)
            target_cell = ws.cell(insert_at, column)
            if source_cell.has_style:
                target_cell._style = copy(source_cell._style)
            if source_cell.number_format:
                target_cell.number_format = source_cell.number_format
            if source_cell.alignment:
                target_cell.alignment = copy(source_cell.alignment)
        row = insert_at
        ws.cell(row, 1).value = item.task_group
        ws.cell(row, 2).value = item.dataset
        ws.cell(row, 3).value = label
        set_values(ws, row, [4, 5, 6], [item.mcc_mean, item.mcc_lower, item.mcc_upper])
        inserted += 1

    rows = {}
    for row in range(3, ws.max_row + 1):
        key = (str(ws.cell(row, 2).value), str(ws.cell(row, 3).value))
        if key == ("None", "None"):
            continue
        if key in rows:
            raise ValueError(f"duplicate key in S4 after insertion: {key}")
        rows[key] = row
    source_keys = {(str(x.dataset), display_model(x.model)) for x in source.itertuples(index=False)}
    if set(rows) != source_keys:
        raise ValueError(f"S4 key mismatch after insertion: missing={source_keys-set(rows)}, extra={set(rows)-source_keys}")
    for item in source.itertuples(index=False):
        row = rows[(str(item.dataset), display_model(item.model))]
        ws.cell(row, 1).value = item.task_group
        set_values(ws, row, [4, 5, 6], [item.mcc_mean, item.mcc_lower, item.mcc_upper])
    return inserted


def reconcile_mpra(wb) -> int:
    ws = wb["S15"]
    source = read_csv("S29_zero_shot_classification_of_variants_with_mpra_supported_regulatory_effects.csv")
    rows = {str(ws.cell(row, 1).value).lower(): row for row in range(3, ws.max_row + 1)}
    if len(rows) != len(source):
        raise ValueError("S15 model roster and S29 source row count differ")
    for item in source.itertuples(index=False):
        row = rows.get(display_model(item.model).lower())
        if row is None:
            raise ValueError(f"S15 has no row for {item.model}")
        set_values(
            ws,
            row,
            list(range(2, 12)),
            [
                item.num_samples,
                item.num_positive,
                item.num_negative,
                item.baseline,
                item.auprc_mean,
                item.auprc_ci_lower,
                item.auprc_ci_upper,
                item.auroc_mean,
                item.auroc_ci_lower,
                item.auroc_ci_upper,
            ],
        )
    return len(source)


def reconcile_ure_macro_f1(wb) -> int:
    source = read_csv("S20_class_specific_f1_scores_for_five_ure_regulatory_element_classes.csv")
    ws = wb["S6"]
    classes = ["URE", "ELS", "PLS", "Insulator", "Silencer"]
    display_classes = ["URE", "Enhancer", "Promoter", "Insulator", "Silencer"]
    for row in range(3, ws.max_row + 1):
        dataset = f"URE_dataset_{str(ws.cell(row, 1).value).split()[-1]}"
        model = str(ws.cell(row, 2).value).lower()
        match = source[
            (source["dataset"] == dataset)
            & (source["model"].map(lambda value: display_model(value).lower()) == model)
        ]
        if len(match) != len(classes):
            raise ValueError(f"S6 class roster mismatch for {dataset}, {model}")
        values = [float(match.loc[match["class_label"] == label, "per_class_f1"].iloc[0]) for label in classes]
        for column, label in zip(range(3, 8), display_classes, strict=True):
            ws.cell(row, column).value = float(match.loc[match["class_label"] == classes[display_classes.index(label)], "per_class_f1"].iloc[0])
        # Resolve the existing formula to its source-backed numeric result.
        ws.cell(row, 8).value = sum(values) / len(values)
    return ws.max_row - 2


def reconcile_adaptive_alleles(wb) -> int:
    source = read_csv("S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv")
    ws = wb["S23"]
    rows = {str(ws.cell(row, 1).value).lower(): row for row in range(3, ws.max_row + 1)}
    if len(rows) != len(source):
        raise ValueError("S23 model roster and S38 source row count differ")
    for item in source.itertuples(index=False):
        label = display_model(item.model_id).lower()
        if label not in rows:
            raise ValueError(f"S23 has no row for {item.model_id}")
        row = rows[label]
        set_values(
            ws,
            row,
            list(range(2, 8)),
            [
                item.n_loci,
                item.positive_loci,
                item.accuracy,
                item.accuracy_ci_low,
                item.accuracy_ci_high,
                item.one_sided_exact_binom_p,
            ],
        )
    return len(source)


def reconcile_four_model_summary(wb) -> int:
    source = read_csv("S38_model_level_preferences_for_literature_supported_adaptive_alleles.csv")
    ws = wb["S26"]
    target_row = next(
        row
        for row in range(3, ws.max_row + 1)
        if ws.cell(row, 2).value == "Selected-allele preference"
        and ws.cell(row, 3).value == "Fraction"
    )
    by_model = {display_model(item.display_model): item.accuracy for item in source.itertuples(index=False)}
    labels = [str(ws.cell(2, column).value) for column in range(4, 8)]
    set_values(ws, target_row, [4, 5, 6, 7], [by_model[label] for label in labels])
    return 1


def reconcile_adaptive_locus_alleles(wb) -> int:
    source = read_csv("S39_locus_level_scores_for_literature_supported_adaptive_alleles.csv")
    ws = wb["S22"]
    source_rows = source.drop_duplicates("variant_id").set_index("variant_id")
    updates = 0
    evidence_notes = {
        "rs642742": (
            "Published reports describe the derived T allele as selection-associated; "
            "the benchmark scoring roster in S39 labels C as selected and T as the alternative. "
            "The literature annotation and scoring orientation are shown separately."
        ),
        "rs74653330": (
            "Derived coding allele reported under selection; this retained locus passed local "
            "reference, allele and biallelic-SNV quality checks in S39."
        ),
        "rs10166942": (
            "T allele frequency tracks latitude and has population-genetic selection evidence. "
            "The broader dbSNP record may be multiallelic; the benchmark uses the validated "
            "biallelic T/C record in S39."
        ),
        "rs1871534": (
            "Strong population-differentiation/selection candidate; the GRCh38 reference allele "
            "has a known assembly issue/fix patch, and the retained benchmark record passed "
            "local allele quality checks in S39."
        ),
    }
    for row in range(3, ws.max_row + 1):
        variant = str(ws.cell(row, 1).value)
        if variant not in source_rows.index:
            raise ValueError(f"S22 variant {variant} is absent from S39")
        item = source_rows.loc[variant]
        for column, value in [(5, item.selected_allele), (6, item.other_allele)]:
            if str(ws.cell(row, column).value) != str(value):
                ws.cell(row, column).value = str(value)
                updates += 1
        if variant in evidence_notes and ws.cell(row, 12).value != evidence_notes[variant]:
            ws.cell(row, 12).value = evidence_notes[variant]
            updates += 1
    return updates


def reconcile_model_summary(wb) -> int:
    source = read_csv("S43_model_level_prediction_performance_and_computational_resource_use.csv")
    ws = wb["S27"]
    rows = {str(ws.cell(row, 1).value).lower(): row for row in range(3, ws.max_row + 1)}
    if len(rows) != len(source):
        raise ValueError("S27 model roster and S43 source row count differ")
    fields = [
        "mean_mcc",
        "n_tasks",
        "params_m",
        "throughput_bps",
        "latency_ms",
        "vram_mb",
        "mean_rank",
        "n_rank_tasks",
    ]
    for item in source.itertuples(index=False):
        label = display_model(item.model).lower()
        if label not in rows:
            raise ValueError(f"S27 has no row for {item.model}")
        set_values(ws, rows[label], list(range(2, 10)), [getattr(item, field) for field in fields])
    return len(source)


def reconcile(workbook: Path) -> None:
    wb = load_workbook(workbook)
    expected_sheets = ["Index"] + [f"S{i}" for i in range(1, 33)]
    if wb.sheetnames != expected_sheets:
        raise ValueError(f"publication workbook sheet layout changed: {wb.sheetnames}")
    inserted = reconcile_genomic_benchmark(wb)
    ure_rows = reconcile_ure_macro_f1(wb)
    mpra = reconcile_mpra(wb)
    adaptive = reconcile_adaptive_alleles(wb)
    summary = reconcile_four_model_summary(wb)
    model_summary = reconcile_model_summary(wb)
    locus_alleles = reconcile_adaptive_locus_alleles(wb)
    wb.save(workbook)
    print(
        f"Updated {workbook}: S4 added {inserted} source-backed records; S6 reconciled {ure_rows} rows; "
        f"S15 reconciled {mpra} rows; S23 reconciled {adaptive} rows; "
        f"S22 reconciled {locus_alleles} allele cells; S26 reconciled {summary} aggregate row; "
        f"S27 reconciled {model_summary} rows."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    args = parser.parse_args()
    reconcile(args.workbook.resolve())
    table_scripts = TABLES / "scripts"
    if str(table_scripts) not in sys.path:
        sys.path.insert(0, str(table_scripts))
    from sync_hevo_training_tables import sync
    count = sync(args.workbook.resolve())
    print(f"S30 synchronized with S46/R19: {count} training configurations.")


if __name__ == "__main__":
    main()
