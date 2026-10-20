#!/usr/bin/env python3
"""Build the formal hEvo supplementary-table workbook."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

S_RE = re.compile(r"^S(\d+)_.*\.csv$")


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer")


def collect(tables_dir: Path) -> list[tuple[str, Path]]:
    s = {}
    official_s1 = "S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv"
    for path in tables_dir.glob("S*.csv"):
        match = S_RE.match(path.name)
        if not match:
            continue
        number = int(match.group(1))
        # Family-specific sidecars and archived candidates are not formal tables.
        if "_historical_candidate" in path.stem:
            continue
        # S46_training_run_budgets.csv is a companion to the formal S46
        # architecture table, not a second table with the same identifier.
        if path.name == "S46_training_run_budgets.csv":
            continue
        if number == 23 and path.name != "S23_zero_shot_classification_of_pathogenic_and_benign_clinvar_variants.csv":
            continue
        if 1 <= number <= 50 and (number != 1 or path.name == official_s1):
            if number in s:
                raise ValueError(f"duplicate S{number}: {s[number]} and {path}")
            s[number] = path
    if set(s) != set(range(1, 51)):
        raise ValueError(f"formal S files must be S1-S50, found {sorted(s)}")

    return [(f"S{i}", s[i]) for i in range(1, 51)]


def sheet_name(table_id: str, path: Path, used: set[str]) -> str:
    stem = path.name[:-7] if path.name.endswith(".csv.gz") else path.stem
    suffix = stem.split("_", 1)[1] if "_" in stem else stem
    name = f"{table_id}_{suffix}"[:31]
    base = name
    counter = 2
    while name in used:
        marker = f"_{counter}"
        name = f"{base[:31-len(marker)]}{marker}"
        counter += 1
    used.add(name)
    return name


def index_rows(entries, names, language):
    rows = []
    for table_id, path in entries:
        df = load(path)
        rows.append({
            "附表编号" if language == "zh" else "Table ID": table_id,
            "文件名" if language == "zh" else "File name": path.name,
            "工作表名" if language == "zh" else "Sheet name": names[table_id],
            "表类型" if language == "zh" else "Table type": "S table",
            "对应S表" if language == "zh" else "Corresponding S table": "",
            "行数" if language == "zh" else "Rows": len(df),
            "列数" if language == "zh" else "Columns": len(df.columns),
        })
    return pd.DataFrame(rows)


def format_sheet(writer, name, df):
    ws = writer.sheets[name]
    for index, column in enumerate(df.columns, start=1):
        values = df[column].astype(str).head(200)
        width = min(max(len(str(column)), *(len(value) for value in values)) + 2, 60)
        ws.column_dimensions[ws.cell(row=1, column=index).column_letter].width = max(width, 10)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def build(tables_dir: Path, output: Path) -> None:
    entries = collect(tables_dir)
    used = {"目录_中文", "Index_EN"}
    names = {table_id: sheet_name(table_id, path, used) for table_id, path in entries}
    zh = index_rows(entries, names, "zh")
    en = index_rows(entries, names, "en")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        zh.to_excel(writer, sheet_name="目录_中文", index=False)
        format_sheet(writer, "目录_中文", zh)
        en.to_excel(writer, sheet_name="Index_EN", index=False)
        format_sheet(writer, "Index_EN", en)
        for table_id, path in entries:
            df = load(path)
            df.to_excel(writer, sheet_name=names[table_id], index=False)
            format_sheet(writer, names[table_id], df)
    print(f"Saved workbook: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    tables_dir = args.tables_dir.resolve()
    output = (args.output.resolve() if args.output else tables_dir / "hEvo_benchmark_supplementary_tables.xlsx")
    build(tables_dir, output)


if __name__ == "__main__":
    main()
