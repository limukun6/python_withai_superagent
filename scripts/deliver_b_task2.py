"""Run B's released validation delivery using the shared locked-down official entry."""
import argparse
import csv
import shutil
from pathlib import Path

from scripts import verify_integrated_defaults as official
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, archive, read, require, save, sha, title_rows


PLAN = ROOT / 'research/b_task2_handoff_1127/protocol.json'


def prepare(args):
    plan = read(PLAN)
    package = archive(args.course, 'titles')
    require(args.features is not None and sha(args.features) == plan['input']['features'],
            'Need the registered feature-only validation packet')
    require(sha(package) == plan['input']['raw'], 'Released archive differs from registration')
    folder = args.output / 'titles'
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.features, folder / 'features.zip')
    save(args.output / 'protocol.json', {
        'agent': source_identity(ROOT), 'runner': sha(Path(official.__file__)),
        'delivery_runner': sha(Path(__file__)), 'study': sha(PLAN),
        'packages': {'titles': plan['input']},
        'fits_per_case': plan['limits']['fits_including_refit'],
        'model_requests': plan['limits']['model_api_requests'],
    })


def score(args):
    folder = args.output / 'titles'
    sealed = read(folder / 'sealed.json')
    predictions = folder / 'delivery/predictions.csv'
    require(sealed['independent_reload'] and not sealed['official_test_access']
            and sha(predictions) == sealed['prediction_sha256'],
            'A sealed, independently reloaded prediction file is required before scoring')
    package = archive(args.course, 'titles')
    require(sha(package) == read(PLAN)['input']['raw'], 'Unexpected course archive')
    with predictions.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    truth = title_rows(package, 'val', labels=True)
    require(len(truth) == 940 and [r['row_id'] for r in rows] == [str(i) for i in range(len(truth))],
            'Designated validation rows differ')
    correct = sum(int(r['prediction']) == item[2] for r, item in zip(rows, truth))
    save(args.output / 'validation.json', dict(
        sealed, metric='accuracy', correct=correct, rows=len(truth), accuracy=correct / len(truth),
        runtime_sha256=source_identity(ROOT), split='designated validation'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'train', 'score'])
    for field in ('output', 'weights', 'course', 'features'):
        p.add_argument('--' + field, type=lambda s: Path(s).resolve(), required=field in ('output', 'course'))
    args = p.parse_args()
    args.case = 'titles'
    if args.action == 'prepare':
        prepare(args)
    else:
        spec = read(args.output / 'protocol.json')
        require(spec['delivery_runner'] == sha(Path(__file__)) and spec['study'] == sha(PLAN)
                and spec['agent'] == source_identity(ROOT), 'Frozen delivery protocol or runtime changed')
        if args.action == 'train':
            if args.weights is None:
                p.error('train requires --weights')
            official.run(args)
        else:
            score(args)
