"""Copy only JSON/text evidence; never task data, trained models, or credential ledgers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    names = {
        'a09_bridge_cli_01': ['protocol.json','completion.json','command.json',
            *['official_run/'+name for name in ('run_manifest.json','actions.json','history.json',
                'classifier_fits.json','confirmation.json','run_outcome.json','final_selection.json',
                'task_spec.json','primary_split.json','confirmation_1_split.json','confirmation_2_split.json',
                'state_1.json','state_10.json','events.jsonl')]],
        'a09_bridge_cli_verified': ['protocol.json','reuse.json','completion.json','results.json'],
        'a09_interface_checks': ['protocol.json','completion.json','tests.log'],
        'a09_legacy_cli_regression': ['protocol.json','completion.json','tests.log'],
    }
    inventory = []
    for run, files in names.items():
        for name in files:
            source = args.runs_root / run / name
            target = args.output / run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inventory.append(dict(file=target.relative_to(args.output).as_posix(), bytes=target.stat().st_size,
                                  sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (args.output / 'inventory.json').write_text(json.dumps(dict(files=inventory,
        note='Raw local byte hashes; Git line-ending normalization can change cloned file bytes. No models/data/SQLite.'),
        indent=2), encoding='utf-8')
    print(json.dumps(dict(files=len(inventory), bytes=sum(r['bytes'] for r in inventory))))


if __name__ == '__main__':
    main()
