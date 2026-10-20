#!/usr/bin/env python3
"""Reconcile published cohort labels against phased source VCF sample IDs."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

COHORTS = {'UKBB': 'UKB', 'KD': 'EAS', 'SG10K': 'SG10K', 'TWINSUK741': 'UK10K',
           'ALSPAC740': 'UK10K', 'AFR': 'AFR', 'HGDP': 'HGDP', 'Topmed': 'TopMed'}
S1 = 'S1_genetic_principal_component_coordinates_of_the_prescreening_cohort.csv'
S2 = 'S2_principal_component_coordinates_of_the_final_63_809_participant_cohort.csv'
S4 = 'S4_cohort_sources_and_ancestry_composition_of_the_final_haplotype_corpus.csv'


def samples(path):
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            if line.startswith('#CHROM'):
                ids = line.rstrip().split('\t')[9:]
                if len(ids) != len(set(ids)):
                    raise ValueError(f'Duplicate IDs: {path}')
                return set(ids)
    raise ValueError(f'No VCF sample header: {path}')


def read(path, delimiter=','):
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return reader.fieldnames, list(reader)


def write(path, fields, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcf-root', type=Path, required=True)
    parser.add_argument('--ancestry-labels', type=Path, required=True)
    parser.add_argument('--tables-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.tables_dir
    membership, inventory = {}, []
    for directory, group in COHORTS.items():
        baseline = None
        for chrom in range(1, 23):
            files = list((args.vcf_root / directory).glob(f'chr{chrom}.*vcf.gz'))
            if len(files) != 1:
                raise ValueError(f'Expected one source VCF: {directory}, chr{chrom}')
            current = samples(files[0])
            if baseline is None:
                baseline = current
            elif current != baseline:
                raise ValueError(f'Sample drift: {directory}, chr{chrom}')
        for iid in baseline:
            if iid in membership:
                raise ValueError(f'Ambiguous source membership: {iid}')
            membership[iid] = group
        inventory.append({'source': directory, 'group': group, 'sample_count': len(baseline),
                          'chromosomes_checked': 22,
                          'sample_ids_sha256': hashlib.sha256('\n'.join(sorted(baseline)).encode()).hexdigest()})
    fields, final = read(root / S2)
    if len(final) != 63809 or len({r['IID'] for r in final}) != 63809:
        raise ValueError('S2 must contain 63,809 unique IDs')
    changes = []
    for row in final:
        iid = row['IID']
        group = membership.get(iid)
        if group is None:
            if row['Group'] != '1kgenome':
                raise ValueError(f'Unmapped source: {iid}')
            continue
        if row['Group'] != group:
            changes.append({'IID': iid, 'old_group': row['Group'], 'new_group': group})
        row['Group'] = group
    if set(membership) != {r['IID'] for r in final if r['Group'] != '1kgenome'}:
        raise ValueError('S2/source membership mismatch')
    full = samples(next((args.vcf_root / 'UKB_full').glob('chr22.*vcf.gz')))
    selected = {iid for iid, group in membership.items() if group == 'UKB'}
    if not selected <= full:
        raise ValueError('UKB coverage IDs missing from full VCF')
    _, ancestry_rows = read(args.ancestry_labels, '\t')
    ancestry = {r['sample']: r['assigned_label'] for r in ancestry_rows}
    if len(ancestry) != len(ancestry_rows) or set(ancestry) != {r['IID'] for r in final}:
        raise ValueError('Ancestry labels do not match S2')
    s4_fields, old_composition = read(root / S4)
    names = {'1k genome': '1kgenome', 'TOPMed': 'TopMed'}
    composition = []
    for old in old_composition:
        group = names.get(old['Cohort'], old['Cohort'])
        rows = final if group == 'All' else [r for r in final if r['Group'] == group]
        counts = Counter(ancestry[r['IID']] for r in rows)
        if set(counts) - {'EUR', 'AFR', 'EAS', 'SAS', 'Others'}:
            raise ValueError('Unexpected ancestry label')
        out = dict(old)
        out['Total_samples'] = len(rows)
        out['Cohort_percent_of_all_samples'] = f'{100 * len(rows) / len(final):.9f}'
        for label in ('EUR', 'AFR', 'EAS', 'SAS', 'Others'):
            out[f'{label}_count'] = counts[label]
        composition.append(out)
    # All ancestry assignments are preserved; only cohort attribution changes.
    if any(str(composition[0][k]) != old_composition[0][k] for k in s4_fields if k.endswith('_count')):
        raise ValueError('Ancestry source differs from published overall totals')
    s1_fields, prescreening = read(root / S1)
    for row in prescreening:
        if row['IID'] in membership:
            row['Group'] = membership[row['IID']]
    public_outputs = []
    # Preserve the public table's file-local record numbers and coordinates.
    for filename, source in [('R01_prescreening_genetic_principal_components.tsv', prescreening),
                             ('R02_training_resource_genetic_principal_components.tsv', final)]:
        path = root / 'repository_tables' / filename
        lines = path.read_text().splitlines()
        header_end = next(i for i, line in enumerate(lines) if not line.startswith('##'))
        records = list(csv.DictReader(lines[header_end:], delimiter='\t'))
        if len(records) != len(source):
            raise ValueError(f'Public table row mismatch: {path}')
        for record, row in zip(records, source):
            if any(not math.isclose(float(record[k]), float(row[k]), rel_tol=1e-12, abs_tol=1e-15) for k in ('PC1', 'PC2')):
                raise ValueError(f'Public table coordinate mismatch: {path}')
            source_column = 'Source cohort' if 'Source cohort' in record else 'Ancestry category'
            record[source_column] = row['Group']
        public_outputs.append((path, lines[:header_end], records))
    # Validate every input and public mirror before changing any table.
    write(root / S1, s1_fields, prescreening)
    write(root / S2, fields, final)
    write(root / S4, s4_fields, composition)
    for path, metadata, records in public_outputs:
        with path.open('w') as handle:
            handle.write('\n'.join(metadata) + '\n')
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader(); writer.writerows(records)
    summary = {'source_inventory': inventory, 'final_cohort_counts': dict(Counter(r['Group'] for r in final)),
               'corrected_labels': len(changes), 'coverage_samples': len(selected),
               'training_ukb_only': 0, 'coverage_only': 0,
               'ancestry_labels_source': str(args.ancestry_labels),
               'coverage_curve_recomputed': False}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
