"""Compare reproducibility evidence before pooling results from two machines. No model fits."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def package_index(protocol):
    entries = protocol['tasks']
    if isinstance(entries, dict):
        entries = entries.values()
    by_name = {e['name']: e for e in entries}
    if len(by_name) != len(entries):
        raise ValueError('Ambiguous task identifiers in protocol')
    return by_name


def compare(left, right):
    a, b = read(left / 'protocol.json'), read(right / 'protocol.json')
    checks = {}
    for key in ('batch', 'source_sha256', 'environment', 'wall_seconds', 'trial_seconds'):
        if key not in a or key not in b:
            raise ValueError('Required protocol field missing: ' + key)
        checks[key] = a[key] == b[key]
    for key in ('runner_sha256', 'driver_sha256', 'helper_sha256', 'entry_sha256', 'plan_sha256', 'outer_seed', 'inner_seed',
                'api_request_limit', 'api_requests_allowed', 'execution_order'):
        if key in a or key in b:
            checks[key] = key in a and key in b and a[key] == b[key]
    limit_a = a.get('fit_limit_each', a.get('fit_limit_per_path'))
    limit_b = b.get('fit_limit_each', b.get('fit_limit_per_path'))
    checks['actual_fit_cap'] = type(limit_a) is int and limit_a == limit_b
    tasks_a, tasks_b = package_index(a), package_index(b)
    checks['task_names'] = sorted(tasks_a) == sorted(tasks_b)
    for name in sorted(tasks_a.keys() & tasks_b.keys()):
        first, second = tasks_a[name], tasks_b[name]
        for key in ('files', 'source', 'catalog_sha256', 'candidate_ids', 'splits', 'row_ids_sha256', 'semantics'):
            checks[name + '.' + key] = first[key] == second[key]
        for folder, task, side in ((left, first, 'left'), (right, second, 'right')):
            row_path = folder / name / 'row_ids.json.gz'
            if not row_path.exists() and 'outer_seed' in task:
                row_path = folder / str(task['outer_seed']) / 'row_ids.json.gz'
            data = gzip.decompress(row_path.read_bytes())
            checks[name + '.' + side + '_row_file_integrity'] = hashlib.sha256(data).hexdigest() == task['row_ids_sha256']
    return dict(comparable=all(checks.values()), differences=[k for k, passed in checks.items() if not passed],
                checks=checks, fits=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('other', type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.other)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['comparable'] else 2)
