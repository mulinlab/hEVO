"""Single entry point for fixed S1-six-PC ancestry annotation."""
import argparse
import json
from pathlib import Path
import yaml
from ancestry.final import predict, check, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('command', choices=['predict', 'check', 'train'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    c = yaml.safe_load(args.config.read_text())
    for key in ['s1', 'targets']:
        p = Path(c['inputs'][key])
        c['inputs'][key] = str(p if p.is_absolute() else args.config.resolve().parent / p)
    for key in ['output_dir', 'training_output_dir']:
        p = Path(c[key])
        c[key] = str(p if p.is_absolute() else args.config.resolve().parent / p)
    result = check(c) if args.command == 'check' else {'predict': predict, 'train': train}[args.command](c, args.dry_run)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
