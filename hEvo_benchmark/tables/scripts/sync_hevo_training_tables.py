#!/usr/bin/env python3
"""Synchronize four-model training settings and the publication's S30 sheet.

Public-resource budgets are independent input fields. Blank budgets stay blank;
author-entered S30 values are preserved and synchronized to the S/R companions.
"""
from __future__ import annotations

import argparse
import csv
from copy import copy
from pathlib import Path
import tempfile
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from openpyxl import load_workbook

TABLES = Path(__file__).resolve().parents[1]
MODELS = ["hEvo-base", "hEvo-large", "hEvo-open-base", "hEvo-open-large"]
DISPLAY = dict(zip(MODELS, ["hEVO-base", "hEVO-large", "hEVO-base-o", "hEVO-large-o"]))
PUBLIC = MODELS[2:]
STAGE2 = [
    ("stage2_optimizer", "Stage II optimizer", "AdamW", "Optimizer"),
    ("stage2_learning_rate", "Stage II learning rate", 0.001, "Learning rate"),
    ("stage2_weight_decay", "Stage II weight decay", 0, "Coefficient"),
    ("stage2_batch_size", "Stage II batch size", 8, "Sequences/device"),
    ("stage2_eval_batch_size", "Stage II validation batch size", 8, "Sequences/device"),
    ("stage2_gradient_accumulation_steps", "Stage II gradient accumulation", 1, "Steps"),
    ("stage2_epochs", "Stage II epochs", 1, "Epochs"),
    ("stage2_seed", "Stage II training seed", 42, "Seed"),
    ("stage2_validation_seed", "Stage II validation seed", 59, "Seed"),
    ("stage2_sequence_length", "Stage II sequence length", 1023, "bp"),
    ("stage2_center_index", "Stage II masked centre", 511, "Zero-based index"),
    ("stage2_mask_token_id", "Stage II mask token ID", 4, "Token ID"),
    ("stage2_max_samples_per_stratum", "Stage II maximum per AF stratum", 100000, "Examples/epoch"),
    ("stage2_target_mode", "Stage II prediction target", "alt_only", "Target"),
    ("stage2_split_ratios", "Stage II train/validation/test split", "90:5:5", "Ratio"),
]


def read_table(path, delimiter=","):
    with path.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.startswith("## ")]
    reader = csv.DictReader(lines, delimiter=delimiter)
    return list(reader.fieldnames), list(reader)


def write_table(path, fields, rows, delimiter=",", notes=()):
    with path.open("w", encoding="utf-8", newline="") as handle:
        for note in notes:
            handle.write("## " + note + "\n")
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value):
    if value in (None, "", "NA", "N/A"):
        return None
    number = float(value)
    if number < 0 or not number.is_integer():
        raise ValueError(f"Budget must be a nonnegative integer: {value!r}")
    return int(number)


def reconcile_budget(csv_value, sheet_value):
    left, right = as_int(csv_value), as_int(sheet_value)
    if left is not None and right is not None and left != right:
        raise ValueError(f"Conflicting CSV/S30 budgets: {left} versus {right}")
    return left if left is not None else right


def sync(workbook: Path):
    fields, models = read_table(TABLES / "S46_architecture_and_hyperparameters_of_the_primary_hevo_models.csv")
    for key, _, value, _ in STAGE2:
        if key not in fields:
            fields.append(key)
        for model in models:
            if model.get(key, "") == "":
                model[key] = value
    lookup = {r["model"]: r for r in models}
    budget_fields, budgets = read_table(TABLES / "S46_training_run_budgets.csv")
    indexed = {r["run"]: r for r in budgets}
    wb = load_workbook(workbook)
    old = wb["S30"]
    entered = {}
    for row in old.iter_rows(values_only=True):
        model = next((m for m in PUBLIC if row[0] in (m, DISPLAY[m])), None)
        if model:
            entered[model] = (row[4], row[5])
    for model in PUBLIC:
        if model not in indexed:
            row = {key: "" for key in budget_fields}
            row.update(run=model, sequence_length_bp=1024,
                       effective_global_batch_size=lookup[model]["stage1_effective_global_batch_size"],
                       peak_learning_rate=lookup[model]["stage1_peak_learning_rate"],
                       warmup_steps=lookup[model]["stage1_warmup_steps"],
                       gradient_accumulation_steps=lookup[model]["stage1_gradient_accumulation_steps"],
                       precision=lookup[model]["stage1_precision"], training_seed=lookup[model]["stage1_seed"])
            budgets.append(row)
            indexed[model] = row
        row = indexed[model]
        step, seq = entered.get(model, (None, None))
        row["optimizer_step_budget"] = reconcile_budget(row["optimizer_step_budget"], step)
        row["sequence_budget"] = reconcile_budget(row["sequence_budget"], seq)
        row["nominal_nucleotide_positions"] = (
            row["sequence_budget"] * 1024 if row["sequence_budget"] is not None else None
        )

    # Rebuild S30 using its existing typography, styles and page settings.
    first = [list(r) for r in old.iter_rows(values_only=True)]
    first_styles = [[copy(c._style) for c in row] for row in old.iter_rows()]
    first_heights = {i: old.row_dimensions[i].height for i in range(1, old.max_row + 1)}
    marker = next(i for i, r in enumerate(first) if r[0] == "Stage I training configurations and sequence budgets")
    row_templates = {kind: [copy(c._style) for c in old[row]]
                     for kind, row in [("title", 1), ("header", 2), ("data", 17),
                                       ("budget", marker + 3), ("section", marker + 1)]}
    settings = {name: copy(getattr(old, name)) for name in
                ["sheet_properties", "sheet_view", "page_setup", "page_margins", "print_options"]}
    widths = {key: copy(value) for key, value in old.column_dimensions.items()}
    position = wb.sheetnames.index("S30")
    wb.remove(old)
    ws = wb.create_sheet("S30", position)
    for name, value in settings.items():
        if name != "sheet_view":
            setattr(ws, name, value)
    ws.sheet_view.showGridLines = settings["sheet_view"].showGridLines
    for key, value in widths.items():
        ws.column_dimensions[key] = value
    ws.freeze_panes = "B3"
    ws.print_title_rows = "1:2"

    def add(values, kind="data", height=30):
        ws.append(values)
        row = ws.max_row
        for col in range(1, 8):
            ws.cell(row, col)._style = copy(row_templates[kind][col - 1])
        ws.row_dimensions[row].height = height
        return row

    primary_rows = first[:marker]
    while primary_rows and all(value is None for value in primary_rows[-1]):
        primary_rows.pop()
    existing_labels = {row[0] for row in primary_rows}
    for i, row in enumerate(primary_rows):
        line = add(row, "title" if i == 0 else "header" if i == 1 else "data",
                   first_heights[i + 1] or 30)
        for col in range(1, 8):
            ws.cell(line, col)._style = copy(first_styles[i][col - 1])
    ws.merge_cells("A1:G1")
    for key, label, _, unit in STAGE2:
        if label in existing_labels:
            continue
        line = add([label, *[lookup[m][key] for m in MODELS], unit, "Stage II"])
        if key == "stage2_learning_rate":
            for col in range(2, 6):
                ws.cell(line, col)._style = copy(first_styles[17][col - 1])
    add([None] * 7)
    row = add(["Stage I training configurations and sequence budgets"], "section")
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    add(["Configuration", "Training corpus", "Window sampling", "Global batch",
         "Step budget", "Sequence budget", "Nominal nucleotide positions"], "header", 40)
    for record in budgets:
        name = record["run"]
        corpus = ("1000 Genomes Project + HGDP" if name in PUBLIC else
                  "1000 Genomes Project" if name.startswith("1000G") else
                  "UK Biobank full cohort" if name.startswith("UKB") else "Representative corpus")
        sampling = "Sliding, 128 bp stride" if "-SW-" in name or name in PUBLIC or name == "hEvo-large" else "Nonoverlapping"
        row = add([DISPLAY.get(name, name), corpus, sampling, as_int(record["effective_global_batch_size"]),
                   as_int(record["optimizer_step_budget"]), as_int(record["sequence_budget"]),
                   as_int(record["nominal_nucleotide_positions"])], "budget", height=35)
        if name in PUBLIC and not record["sequence_budget"]:
            ws.cell(row, 7, f'=IF(F{row}="","",F{row}*1024)')
    add([None] * 7)
    row = add(["Definitions"], "section")
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    add(["Item", "Definition"], "header")
    for label, definition in [
        ("Nominal nucleotide positions", "Sequence budget multiplied by 1,024."),
        ("Trainer arguments", "Optimizer/scheduler arguments are listed separately from DeepSpeed settings."),
        ("Public-resource variants", "Scale-specific hyperparameters are shared; each checkpoint has its own step and sequence budget."),
    ]:
        add([label, definition], height=50)

    # Only replace the S30 XML; all other paper sheets and shared strings remain intact.
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "workbook.xlsx"
        wb.save(temporary)
        with ZipFile(workbook) as source, ZipFile(temporary) as generated:
            sheets = ET.fromstring(source.read("xl/workbook.xml")).find("m:sheets", ns)
            rid = next(s.get("{" + rel + "}id") for s in sheets if s.get("name") == "S30")
            links = ET.fromstring(source.read("xl/_rels/workbook.xml.rels"))
            target = next(r.get("Target") for r in links if r.get("Id") == rid)
            part = target.lstrip("/") if target.startswith("/") else "xl/" + target
            data = generated.read(part)
            styles = ET.fromstring(source.read("xl/styles.xml")).find("m:cellXfs", ns)
            assert all(int(c.get("s", "0")) < len(styles) for c in ET.fromstring(data).findall(".//m:c", ns))
            output = Path(directory) / "updated.xlsx"
            with ZipFile(output, "w") as destination:
                for info in source.infolist():
                    destination.writestr(info, data if info.filename == part else source.read(info.filename))
        workbook.write_bytes(output.read_bytes())

    write_table(TABLES / "S46_architecture_and_hyperparameters_of_the_primary_hevo_models.csv", fields, models)
    write_table(TABLES / "S46_training_run_budgets.csv", budget_fields, budgets)
    reader = TABLES / "repository_tables/R19_hevo_architecture_and_training_configuration.tsv"
    reader_fields, reader_rows = read_table(reader, "\t")
    notes = [line[3:] for line in reader.read_text().splitlines() if line.startswith("## ")]
    for key, label, _, _ in STAGE2:
        if label not in reader_fields:
            reader_fields.append(label)
        for record in reader_rows:
            name = next(m for m in MODELS if record["Model"].lower() in (m.lower(), DISPLAY[m].lower()))
            record[label] = lookup[name][key]
            record["Model"] = DISPLAY[name]
    write_table(reader, reader_fields, reader_rows, "\t", notes)
    write_table(TABLES / "repository_tables/R19_training_run_budgets.tsv", budget_fields, budgets, "\t", [
        "Repository Table R19 companion. Stage I training configurations and independent sequence budgets",
        "Nominal nucleotide positions equal sequence budget times 1,024.",
        "Blank budget fields remain independent inputs for the corresponding public-resource model.",
    ])
    return len(budgets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=TABLES / "Supplementary Tables.xlsx")
    args = parser.parse_args()
    count = sync(args.workbook)
    print(f"S30 and S46/R19 synchronized: four models, {count} Stage I configurations.")


if __name__ == "__main__":
    main()
