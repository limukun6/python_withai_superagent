"""Compare two independent BP-02 materializations with the registered B inputs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('primary', 'confirmation_1', 'confirmation_2')
FILES = {'train.zip': 'optimizer', 'features.zip': 'delivery', 'inner_splits.json': 'optimizer'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(left, right, check_files):
    reference = ROOT / 'research/bc1_preparation/results.json'
    registered = read(reference)['tasks']
    sources = {
        'plan_sha256': ROOT / 'docs/batches/BP_02_plan.md',
        'reference_sha256': reference,
        'driver_sha256': ROOT / 'scripts/materialize_b_c1_rows.py',
        'helper_sha256': ROOT / 'tools/ab_registered_rows.py',
        'a_freeze_implementation_sha256': ROOT / 'superagent/ab_run.py',
    }
    environments, results = [], []
    for run in (left, right):
        protocol, completion, outcome = [read(run / f'{name}.json') for name in ('protocol', 'completion', 'results')]
        require(protocol['batch'] == 'BP-02' and protocol['fits_allowed'] == protocol['model_api_allowed'] == 0, 'Budget or batch differs')
        require(all(protocol[key] == digest(file) for key, file in sources.items()), 'Source or registration changed')
        require(completion == dict(status='passed', tasks=2, actual_worker_payloads=6, fits=0, model_predictions=0,
                                   outer_model_scores=0, model_api_requests=0), 'Incomplete materialization')
        require(outcome.keys() == registered.keys(), 'Task set differs')
        environments.append(protocol['environment'])
        for name, item in outcome.items():
            require(item['rows_sha256'] == registered[name]['row_ids_sha256'], 'Different source row registration')
            require(set(item['splits']) == set(SPLITS) and item['canonical_files'].keys() == FILES.keys(), 'Incomplete input set')
            require(item['changed_row_order_rejected'] is True and item['changed_manifest_digest_rejected'] is True,
                    'Input rejection probes missing')
            if check_files:
                for filename, section in FILES.items():
                    require(digest(run / name / section / filename) == item['canonical_files'][filename], 'Canonical file digest differs')
                for split in SPLITS:
                    path = run / name / 'a_worker_inputs' / (split + '.joblib')
                    require(digest(path) == item['splits'][split]['local_worker_sha256'], 'Local worker digest differs')
        results.append(outcome)
    require(environments[0] != environments[1], 'Expected two different environments')
    tasks = {}
    for name in registered:
        a, b = (result[name] for result in results)
        for key in ('source', 'training_rows', 'held_rows', 'rows_sha256', 'training_content_sha256', 'held_content_sha256', 'canonical_files'):
            require(a[key] == b[key], f'{name} disagrees on {key}')
        worker_matches = {}
        for split in SPLITS:
            one, two = a['splits'][split], b['splits'][split]
            require(one['logical'].keys() == {'training', 'evaluation'} and one['logical'] == two['logical'],
                    f'{name}/{split} worker examples differ')
            worker_matches[split] = one['local_worker_sha256'] == two['local_worker_sha256']
        tasks[name] = dict(optimized_rows=a['training_rows'], held_rows=a['held_rows'], canonical_file_sha256=a['canonical_files'],
                           logical_worker_pairs_matched=len(worker_matches), serialized_worker_bytes_matched=worker_matches)
    return dict(status='passed', sources={key: digest(path) for key, path in sources.items()}, tasks=tasks,
                canonical_file_pairs_matched=sum(len(t['canonical_file_sha256']) for t in tasks.values()),
                logical_worker_pairs_matched=sum(t['logical_worker_pairs_matched'] for t in tasks.values()),
                serialized_worker_pairs_matched=sum(sum(t['serialized_worker_bytes_matched'].values()) for t in tasks.values()),
                raw_local_artifacts_rehashed=check_files, fits=0, model_api_requests=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--check-files', action='store_true', help='Rehash the local input archives and worker files in each full run.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    message = json.dumps(compare(args.left, args.right, args.check_files), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as output:
            output.write(message)
    else:
        print(message, end='')
