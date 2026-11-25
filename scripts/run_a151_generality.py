"""Batch-4 A-line generality scenarios on anonymous packages.

Builds small synthetic packages, runs the real entry on each, and records what
the entry does. Every case is a development fixture, not a migration result.
No API call: the planner is local.
"""
import argparse
import csv
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / 'research' / 'a151_generality'
WORDS = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta')
CASES = ('macro_f1_anonymous', 'positive_yes', 'positive_no', 'conflict_positive_with_accuracy',
         'missing_train_member')


def rows_for(labels):
    return [[' '.join((WORDS[i % len(WORDS)], WORDS[(i * 3) % len(WORDS)], f'token{i % 7}')), label]
            for i, label in enumerate(labels)]


def write_package(path, labels, manifest, drop_train=False):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['text', 'label'])
    writer.writerows(rows_for(labels))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        if not drop_train:
            archive.writestr('train.csv', buffer.getvalue().encode('utf-8'))
        archive.writestr('validation.csv', buffer.getvalue().encode('utf-8'))
        archive.writestr('manifest.json', json.dumps(manifest, indent=2).encode('utf-8'))
    return path


def manifest(metric, labels, positive=None, goal_source='declared_manifest'):
    body = {'schema_version': 1, 'objective': 'classification', 'metric': metric,
            'modality': 'text', 'input_format': 'single_text_csv',
            'columns': [{'name': 'text', 'type': 'text'}],
            'target': 'label',
            'splits': {'train': {'format': 'csv', 'path': 'train.csv'},
                       'validation': {'format': 'csv', 'path': 'validation.csv'}},
            'goal_sources': {'primary_metric': goal_source},
            'source': {'dataset': 'synthetic development fixture', 'licence': 'internal'}}
    if positive is not None:
        body['positive_label'] = positive
    return body


def run_entry(package, run_dir):
    command = [sys.executable, '-m', 'superagent', 'run', str(package), '--run-dir', str(run_dir),
               '--planner', 'local', '--scheduler', 'ab-v1', '--portfolio', 'ab-v1',
               '--catalog-scope', 'baseline', '--fit-limit', '3', '--seed', '707',
               '--trial-timeout', '300', '--time-limit', '1800']
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=EVIDENCE / 'attempt_01')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and args.execute:
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    packages = out / 'packages'
    packages.mkdir(parents=True, exist_ok=True)
    binary = ['yes' if i % 2 else 'no' for i in range(300)]
    ternary = ['low', 'mid', 'high'] * 100

    plan = {
        'macro_f1_anonymous': dict(labels=binary, metric='macro_f1'),
        'positive_yes': dict(labels=binary, metric='binary_f1', positive='yes'),
        'positive_no': dict(labels=binary, metric='binary_f1', positive='no'),
        'conflict_positive_with_accuracy': dict(labels=binary, metric='accuracy', positive='yes'),
        'missing_train_member': dict(labels=binary, metric='accuracy', drop_train=True),
    }
    if not args.execute:
        print(json.dumps({'status': 'validated_not_executed', 'cases': list(plan)}))
        return 0

    results = []
    for name in CASES:
        spec = plan[name]
        package = packages / f'{name}.zip'
        write_package(package, spec['labels'], manifest(spec['metric'], spec['labels'],
                                                        spec.get('positive')),
                      drop_train=spec.get('drop_train', False))
        run_dir = out / name
        completed = run_entry(package, run_dir)
        outcome_path = run_dir / 'run_outcome.json'
        record = {'case': name, 'returncode': completed.returncode}
        if completed.returncode == 0 and outcome_path.is_file():
            outcome = json.loads(outcome_path.read_text(encoding='utf-8'))
            task_spec = json.loads((run_dir / 'task_spec.json').read_text(encoding='utf-8'))
            record.update({'status': outcome.get('status'),
                           'primary_metric': outcome.get('primary_metric'),
                           'metric_in_task_spec': task_spec.get('metric'),
                           'goal_resolution': task_spec.get('goal_resolution', {}).get('fields'),
                           'selected': outcome.get('selected_candidate'),
                           'validation': outcome.get('validation_score'),
                           'fits': outcome['classifier_fit_accounting']['attempted_classifier_fits']})
        else:
            record['stderr_tail'] = (completed.stderr or '')[-600:]
        results.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'goal_resolution'},
                         ensure_ascii=False), flush=True)
    (out / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n',
                                      encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
