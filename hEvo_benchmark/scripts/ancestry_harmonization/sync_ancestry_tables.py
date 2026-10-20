"""Synchronize sample ancestry and cohort counts; preserve PCs and unrelated XLSX parts."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import zipfile
from collections import Counter
from xml.etree import ElementTree as ET

import openpyxl

S2 = "S2_principal_component_coordinates_of_the_final_63_809_participant_cohort.csv"
S4 = "S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv"
R2 = "repository_tables/R02_training_resource_genetic_principal_components.tsv"
R7 = "repository_tables/R07_ancestry_sharing_of_distinct_sequences.tsv"
R28 = "repository_tables/R28_cohort_ancestry_composition.tsv"
FIVE = ("EUR", "AFR", "EAS", "SAS", "Others")
ALIASES = {"1k genome": "1kgenome", "TOPMed": "TopMed"}
METHOD = "S1 PC classifier (kNN15; 6 PCs)"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, delimiter=","):
    text = Path(path).read_text()
    comments = [line for line in text.splitlines() if line.startswith("##")]
    reader = csv.DictReader([line for line in text.splitlines() if not line.startswith("##")],
                            delimiter=delimiter)
    return comments, reader.fieldnames, list(reader)


def encoded(rows, fields, delimiter=",", comments=()):
    stream = io.StringIO(newline="")
    for line in comments:
        stream.write(line + "\n")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--tables-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tables = args.tables_dir
    repository = Path(__file__).resolve().parents[3]
    if args.backup_dir.resolve().is_relative_to(repository):
        raise ValueError("Backups and check records must be written outside the repository")
    _, _, labels = read(args.labels, "\t")
    by_id = {row["sample"]: row for row in labels}
    if len(labels) != 63809 or len(by_id) != 63809:
        raise ValueError("Expected exactly 63,809 unique labels")
    if any(row["assigned_label"] not in FIVE for row in labels):
        raise ValueError("Unexpected ancestry class")
    _, fields2, sample_rows = read(tables / S2)
    if len(sample_rows) != 63809 or len({r["IID"] for r in sample_rows}) != 63809:
        raise ValueError("Duplicate/wrong S2 sample count")
    if {r["IID"] for r in sample_rows} != set(by_id):
        raise ValueError("S2 and labels have different ID sets")
    patches = {}
    ancestry_source = lambda row: "Cohort annotation" if row["cohort"] == "EAS" else METHOD
    for row in sample_rows:
        label = by_id[row["IID"]]
        if row["Group"] != label["cohort"]:
            raise ValueError("S2 cohort conflict")
        row["Ancestry"] = label["assigned_label"]
        row["Ancestry_label_source"] = ancestry_source(label)
    fields2 = [x for x in fields2 if x not in ("Ancestry", "Ancestry_label_source")]
    patches[S2] = encoded(sample_rows, fields2 + ["Ancestry", "Ancestry_label_source"])

    _, fields4, composition = read(tables / S4)
    for row in composition:
        cohort = ALIASES.get(row["Cohort"], row["Cohort"])
        members = labels if cohort == "All" else [r for r in labels if r["cohort"] == cohort]
        if not members or len(members) != int(row["Total_samples"]):
            raise ValueError("S4 cohort size conflict")
        counts = Counter(r["assigned_label"] for r in members)
        for label in FIVE:
            row[label + "_count"] = str(counts[label])
    patches[S4] = encoded(composition, fields4)

    comments, fields_r2, records = read(tables / R2, "\t")
    if len(records) != len(sample_rows):
        raise ValueError("R02 row count mismatch")
    old_source = "Source cohort" if "Source cohort" in fields_r2 else "Ancestry category"
    new_records = []
    for public, internal in zip(records, sample_rows):
        if any(not math.isclose(float(public[k]), float(internal[k]), rel_tol=1e-12, abs_tol=1e-15)
               for k in ("PC1", "PC2")):
            raise ValueError("R02/S2 row order not established by coordinates")
        if public[old_source] != internal["Group"]:
            raise ValueError("R02 source cohort conflict")
        new_records.append({"Record": public["Record"], "PC1": public["PC1"], "PC2": public["PC2"],
                            "Source cohort": internal["Group"], "Ancestry category": internal["Ancestry"],
                            "Label source": internal["Ancestry_label_source"]})
    comments = [line for line in comments if not line.startswith("## Column ")
                and not line.startswith("## Description:") and not line.startswith(("## Source:", "## Provenance:"))]
    comments.insert(2, "## Description: Final-resource coordinates, source cohort and five-category ancestry. Record numbers are file-local; coordinates retain source precision.")
    comments.insert(3, "## Source: S2 row order; S1-PC classifier labels with the internal Chinese cohort annotated as EAS.")
    definitions = [("Record", "File-local row number; not a participant identifier."),
                   ("PC1", "Released first principal component; unchanged."),
                   ("PC2", "Released second principal component; unchanged."),
                   ("Source cohort", "Source cohort membership, distinct from ancestry."),
                   ("Ancestry category", "EUR, AFR, EAS, SAS or Others."),
                   ("Label source", "Cohort annotation for the internal Chinese cohort; otherwise the S1-PC classifier.")]
    comments += [f"## Column {i} | {name} | {definition}" for i, (name, definition) in enumerate(definitions, 1)]
    patches[R2] = encoded(new_records, [x[0] for x in definitions], "\t", comments)

    rows28 = []
    for row in composition:
        cohort = ALIASES.get(row["Cohort"], row["Cohort"])
        members = labels if cohort == "All" else [r for r in labels if r["cohort"] == cohort]
        other = [r for r in members if r["assigned_label"] == "Others"]
        counts_other = Counter(r["assignment_reason"] for r in other)
        breakdown = [counts_other["reference_" + k] for k in ("AMR", "MID", "OCE")]
        rejected = sum(int(r["effective_rejected_flag"]) for r in other)
        if sum(breakdown) + rejected != len(other):
            raise ValueError("Others reasons do not reconcile")
        rows28.append({"Source cohort": row["Cohort"], "Individuals": row["Total_samples"],
                       **{k: row[k + "_count"] for k in FIVE},
                       "Others AMR": breakdown[0], "Others MID": breakdown[1],
                       "Others OCE": breakdown[2], "Others rejected": rejected})
    definitions28 = [("Source cohort", "Source cohort; All is the entire final resource."),
                     ("Individuals", "Number of unique individuals.")]
    definitions28 += [(k, "Number assigned to the " + k + " display category.") for k in FIVE]
    definitions28 += [("Others " + k, "Accepted " + k + " underlying class, displayed as Others.") for k in ("AMR", "MID", "OCE")]
    definitions28 += [("Others rejected", "Rejected by model confidence, distance or input quality criteria; displayed as Others.")]
    notes28 = ["## Repository Table R28. Cohort sources and ancestry composition",
               "## Description: Counts of unique final-resource individuals by source cohort and five-category ancestry.",
               "## Method: S1 PC classifier (calibrated kNN, 15 neighbors, first 6 PCs); all 3,188 internal Chinese cohort participants are EAS by cohort annotation.",
               "## Interpretation: Others combines accepted AMR/MID/OCE and rejected predictions. Model probabilities are not ancestry proportions.",
               "## Source: Same sample assignments as S2 and cohort totals as S4."]
    notes28 += [f"## Column {i} | {name} | {definition}" for i, (name, definition) in enumerate(definitions28, 1)]
    patches[R28] = encoded(rows28, [x[0] for x in definitions28], "\t", notes28)

    # Existing sequence sharing cannot be reaggregated from sample labels alone.
    r7 = (tables / R7).read_text()
    original_provenance = next(x for x in r7.splitlines() if x.startswith(("## Source:", "## Provenance:")))
    patches[R7] = r7.replace(original_provenance,
        "## Source: Supplementary Table S9; sequence-sharing counts use PC1/PC2-based ancestry annotations.").encode()
    patches["S9_ancestry_sharing_README.md"] = (
        "# S9 ancestry-label basis\n\n"
        "S9 contains sequence-sharing counts across ancestry categories. "
        "Sample annotations and cohort counts are in S2/S4 and R02/R28. "
        "R07 provides the corresponding annotated reader table and field definitions.\n").encode()

    readme_r = (tables / "repository_tables/README.md").read_text()
    readme_r = readme_r.replace("R01–R27", "R01–R28")
    lines = readme_r.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("| [R02]"):
            lines[i] = "| [R02](R02_training_resource_genetic_principal_components.tsv) | 63,809 | 6 | S2 | Coordinates and file-local records are provided; source cohort and five-category ancestry are separate fields. |"
        if line.startswith("| [R07]"):
            lines[i] = "| [R07](R07_ancestry_sharing_of_distinct_sequences.tsv) | 31 | 4 | S9 | Sequence-sharing counts across the ancestry categories in S9. |"
    if not any(line.startswith("| [R28]") for line in lines):
        i = next(i for i, line in enumerate(lines) if line.startswith("| [R27]"))
        lines.insert(i + 1, "| [R28](R28_cohort_ancestry_composition.tsv) | 9 | 11 | S4 | Cohort ancestry counts and Others breakdown, consistent with the current sample labels in R02. |")
    patches["repository_tables/README.md"] = ("\n".join(lines) + "\n").encode()
    readme = (tables / "README.md").read_text()
    if '## Cohort ancestry annotation' not in readme:
        readme += '\n## Cohort ancestry annotation\n\nS2 includes `Ancestry` and `Ancestry_label_source`; `Group` identifies the source cohort. S4 and repository R02/R28 summarize annotations from the S1 six-PC classifier, with all 3,188 internal Chinese cohort participants annotated as EAS.\n'
    patches["README.md"] = readme.encode()

    # This publication workbook has a separate numbering: cohort composition is S1.
    workbook = tables / "Supplementary_Tables_32_v15_updated.xlsx"
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=False)
    if wb["S1"].cell(2, 4).value != "EUR":
        raise ValueError("Unexpected workbook layout")
    display = {"UK Biobank": "UKB", "TOPMed": "TopMed", "SG10K": "SG10K", "UK10K": "UK10K",
               "1000 Genomes Project": "1kgenome", "Chinese cohort (this study)": "EAS",
               "HGDP": "HGDP", "CAAPA": "AFR"}
    updates = {}
    for values in wb["S1"].iter_rows(min_row=3, values_only=True):
        cohort = display[values[0]]
        row_number = len(updates) // 5 + 3
        members = [r for r in labels if r["cohort"] == cohort]
        if len(members) != values[2]:
            raise ValueError("Workbook cohort count mismatch")
        counts = Counter(r["assigned_label"] for r in members)
        for col, label in zip("DEFGH", FIVE):
            updates[col + str(row_number)] = counts[label]
    wb.close()
    with zipfile.ZipFile(workbook) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(x for x in book.find("m:sheets", ns) if x.attrib["name"] == "S1")
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(x.attrib["Target"] for x in rels if x.attrib["Id"] == rid)
        target = target.lstrip("/") if target.startswith("/") else "xl/" + target
        xml = archive.read(target).decode()
    for address, count in updates.items():
        pattern = rf'(<c\b[^>]*\br="{address}"[^>]*>)(.*?)(</c>)'
        def replace(match):
            body, n = re.subn(r"<v>[^<]*</v>", f"<v>{count}</v>", match.group(2))
            if n != 1:
                raise ValueError("Expected one numeric value in " + address)
            return match.group(1) + body + match.group(3)
        xml, n = re.subn(pattern, replace, xml, flags=re.S)
        if n != 1:
            raise ValueError("Missing/duplicate workbook cell " + address)
    if args.dry_run:
        print(json.dumps({"tables": list(patches), "workbook_sheet": "S1", "workbook_numeric_cells": len(updates)}))
        return
    if args.backup_dir.exists():
        raise FileExistsError("Use a new backup directory")
    args.backup_dir.mkdir(parents=True)
    old_hashes = {}
    for name in list(patches) + [workbook.name]:
        path = tables / name
        if path.exists():
            backup = args.backup_dir / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            old_hashes[name] = sha(path)
    # Fully stage files before applying them.
    staged = args.backup_dir / "updated"
    staged.mkdir()
    for name, data in patches.items():
        path = staged / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with zipfile.ZipFile(staged / workbook.name, "w") as archive:
        for info, data in entries:
            archive.writestr(info, xml.encode() if info.filename == target else data)
    with zipfile.ZipFile(staged / workbook.name) as archive:
        for info, data in entries:
            if info.filename != target and archive.read(info.filename) != data:
                raise ValueError("Unrelated workbook member changed")
    check_wb = openpyxl.load_workbook(staged / workbook.name, read_only=True)
    assert all(check_wb["S1"][a].value == v for a, v in updates.items())
    check_wb.close()
    for name in list(patches) + [workbook.name]:
        shutil.copy2(staged / name, tables / name)
    report = {"labels_sha256": sha(args.labels), "labels": str(args.labels),
              "before_sha256": old_hashes,
              "after_sha256": {name: sha(tables / name) for name in list(patches) + [workbook.name]},
              "workbook_changed_part": target, "unrelated_workbook_parts_byte_identical": True,
              "coordinates_and_cohorts_preserved": True, "samples": 63809,
              "overall": composition[0], "script_sha256": sha(Path(__file__))}
    (args.backup_dir / "update_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"backup": str(args.backup_dir), "changed_files": len(patches) + 1,
                      "workbook_cells": len(updates), "overall": composition[0]}))


if __name__ == "__main__":
    main()
