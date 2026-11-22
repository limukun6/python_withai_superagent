"""Run B's non-image catalogue comparison on four real training sets with frozen row identities."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import joblib
import numpy as np

from scripts import bq_catalog_entry, run_b1_ab_acceptance as fixture, run_b1_05_pair_pilot as pilot
from scripts.build_external_tasks import write_member
from scripts.check_b_pair_protocol import compare
from scripts.run_b1_06_mrpc import paired_counts
from superagent.ab_run import sources
from superagent.ab_sentence import verified_directory
from superagent.contracts import TaskSpec
from superagent.data import TrainingTask, load_development_task, load_training_task
from superagent.metrics import MetricSpec
from superagent.splits import grouped_partition, overlap_audit, take


TASKS = ('coupon', 'headlines', 'bean', 'sms')
ARMS = ('original', 'expanded')
ORDER = [(task, arm) for i, task in enumerate(TASKS) for arm in (ARMS if i % 2 == 0 else ARMS[::-1])]
COUPON_SHA = {'C_train.npy': 'abd27e72fd8df1e5ddbe8843a7139d6d5190414acb1a861e886df91394c88ff7',
              'y_train.npy': 'd5979f4acf17fef499e0380d7bea415f254690bad469320e47b5d4edcf76c503'}


def coupon_training(file):
    arrays = {}
    with zipfile.ZipFile(file) as archive:
        for key, expected in COUPON_SHA.items():
            names = [n for n in archive.namelist() if Path(n).name == key and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError('Missing or ambiguous course training member')
            raw = archive.read(names[0])
            if fixture.sha(raw) != expected:
                raise ValueError('Coupon training member differs from the registered course file: ' + key)
            arrays[key] = np.load(io.BytesIO(raw), allow_pickle=key == 'C_train.npy')
    # The registered categorical workers consume string-valued categories.
    # Export plain unicode arrays so the training and prediction packets need no object pickle.
    return TrainingTask('categorical_tabular', arrays['C_train.npy'].astype(str), arrays['y_train.npy'],
                        {'metric': 'accuracy'}), dict(training_members=COUPON_SHA, opened=list(COUPON_SHA))


def pack(path, kind, x, y, metric, split):
    with zipfile.ZipFile(path, 'x') as archive:
        if kind == 'categorical_tabular':
            arrays = {'C_' + split + '.npy': x}
            if y is not None:
                arrays['y_' + split + '.npy'] = y
            for name, values in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, values, allow_pickle=False)
                write_member(archive, 'table/' + name, buffer.getvalue())
            return
        if kind == 'numeric_tabular':
            buffer = io.BytesIO()
            np.savez(buffer, **({'X': x, 'y': y} if y is not None else {'X': x}))
            data, fmt, modality = buffer.getvalue(), 'npz', 'tabular'
            metadata = {}
        elif kind == 'single_text':
            buffer = io.StringIO(newline='')
            writer = csv.writer(buffer, lineterminator='\n')
            writer.writerow(['message', 'label'] if y is not None else ['message'])
            for i, message in enumerate(x):
                writer.writerow([message, y[i]] if y is not None else [message])
            data, fmt, modality = buffer.getvalue().encode(), 'csv', 'text'
            metadata = dict(columns=[{'name': 'message', 'type': 'text'}], target='label')
        else:
            raise ValueError('Unregistered Q input kind')
        manifest = dict(schema_version=1, objective='classification', modality=modality, metric=metric,
                        splits={split: {'format': fmt, 'path': 'rows.' + fmt}}, **metadata)
        write_member(archive, 'manifest.json', json.dumps(manifest, sort_keys=True).encode())
        write_member(archive, 'rows.' + fmt, data)


def prepare_shared(root, name, args, *, outer_seed=pilot.OUTER_SEED):
    if name == 'headlines':
        return pilot.prepare_pairs(root, name, *pilot.pair_source(name, args.headlines), outer_seed=outer_seed)
    if name == 'coupon':
        task, source = coupon_training(args.coupon)
    else:
        filename = {'bean': 'dry_bean_numeric_tabular.zip', 'sms': 'sms_spam_single_text.zip'}[name]
        dataset = args.uci / filename
        task = load_training_task(dataset)
        with zipfile.ZipFile(dataset) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            training = manifest['splits']['train']['path']
            source = dict(package_sha256=fixture.sha(dataset.read_bytes()), training_member_sha256=fixture.sha(archive.read(training)),
                          opened=['manifest.json', training], published_source=manifest['source']['url'])
    metric = MetricSpec.for_task(task)
    train, held, resource = grouped_partition(task, outer_seed, 'shared_text')
    audit = overlap_audit(task, train, held)
    if audit['search_duplicate_rows']:
        raise ValueError('Duplicate rows straddle the outer development partition')
    directory = root / name
    directory.mkdir()
    held_x, held_y = take(task.X_train, held), task.y_train[held]
    pack(directory / 'train.zip', task.kind, take(task.X_train, train), task.y_train[train], metric.name, 'train')
    pack(directory / 'features.zip', task.kind, held_x, None, metric.name, 'test')
    joblib.dump(dict(X=held_x, y=held_y, row_ids=held.tolist()), directory / 'holdout.joblib')
    return pilot.register_prepared_task(directory, source, 'not_applicable', len(task.y_train), train, held, resource, audit)


def prepare(root, args, frozen):
    entries = {}
    common = {}
    for name in TASKS:
        common[name] = prepare_shared(root, name, args)
        folder = root / name
        task = load_development_task(folder / 'train.zip')
        spec = TaskSpec.from_task(task).record()
        menus = {}
        for arm in ARMS:
            experiment = name + '_' + arm
            cap = bq_catalog_entry.registered_catalog(spec, arm, train_rows=len(task.y_train))
            menus[arm] = cap
            shutil.copytree(folder, root / experiment)
            entries[experiment] = dict(common[name], name=experiment, task_group=name, portfolio=arm,
                                       metric=spec['metric']['name'], catalog_sha256=fixture.digest(cap), candidate_ids=list(cap))
        base, full = menus['original'], menus['expanded']
        if not base or next(iter(base)) != next(iter(full)) or list(base) != [s for s in full if s in base] or any(base[k] != full[k] for k in base):
            raise ValueError('Shared candidate order, metadata or initial anchor differ for ' + name)
    a10 = fixture.read(pilot.ROOT / 'research/a10_real_task_pilot/protocol.json')
    cross_machine = {}
    for old in a10['tasks']:
        name = old['name']
        if name in ('bean', 'sms'):
            cross_machine[name] = dict(source_package_matches=common[name]['source']['package_sha256'] == old['task']['source']['package_sha256'],
                primary_row_ids_match=common[name]['splits']['primary']['row_ids_sha256'] == old['splits']['primary']['row_ids_sha256'],
                primary_payload_matches=common[name]['splits']['primary']['evaluation_payload_sha256'] == old['splits']['primary']['evaluation_payload_sha256'])
    fixture.write(root / 'protocol.json', dict(batch='BQ-01', source_sha256=frozen,
        driver_sha256=fixture.sha(Path(__file__).read_bytes()), entry_sha256=fixture.sha(Path(bq_catalog_entry.__file__).read_bytes()),
        helper_sha256=fixture.sha(Path(pilot.__file__).read_bytes()), source_builder_sha256=fixture.sha((pilot.ROOT / 'scripts/build_external_tasks.py').read_bytes()),
        plan_sha256=fixture.sha((pilot.ROOT / 'docs/batches/BQ_01_plan.md').read_bytes()), environment=pilot.environment(),
        tasks=entries, execution_order=ORDER, outer_seed=pilot.OUTER_SEED, inner_seed=pilot.INNER_SEED,
        fit_limit_per_path=11, fit_limit_batch=88, wall_seconds=pilot.WALL, trial_seconds=pilot.TRIAL,
        api_requests_allowed=0, a10_cross_machine=cross_machine))
    return entries


def evaluation(root, name, paths):
    data = load_development_task(root / name / 'train.zip')
    metric = MetricSpec.for_task(data)
    held = joblib.load(root / name / 'holdout.joblib')
    counts = {label: int(np.count_nonzero(data.y_train == label)) for label in metric.labels}
    most = max(metric.labels, key=counts.get)
    majority = metric.evaluate(held['y'], np.full(len(held['y']), most))
    p = paired_counts(root, name, ARMS)
    a, b = (paths[name + '_' + arm]['delivery'] for arm in ARMS)
    return dict(**p, metric=metric.name, original_score=a['score'], expanded_score=b['score'],
                primary_difference_percentage_points=100 * (b['score'] - a['score']),
                training_majority_label=most, majority_score=majority['score'])


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture.offline(root)
    os.environ.update(SUPERAGENT_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()))
    began = time.monotonic()
    paths = {}
    try:
        if verified_directory() is None:
            raise ValueError('Q requires the same verified frozen encoder on all paths')
        frozen = fixture.digest(sources())
        entries = prepare(root, args, frozen)
        if args.reference:
            checks = compare(args.reference, root)
            fixture.write(root / 'preflight.json', checks)
            if not checks['comparable']:
                raise ValueError('Pre-fit reference mismatch: ' + ', '.join(checks['differences']))
        if args.prepare_only:
            fixture.write(root / 'completion.json', dict(status='prepared', fits=0, api_requests=0))
            return
        for name, arm in ORDER:
            key = name + '_' + arm
            entry = entries[key]
            before = len(fixture.trace(root))
            row = pilot.train_one(root / key, entry, frozen, launcher=('-m', 'scripts.bq_catalog_entry', arm, 'run'))
            record = fixture.read(root / key / 'run/research_catalog.json')
            if (row['fits'] != len(fixture.trace(root)) - before or record != dict(mode=arm, ids=entry['candidate_ids'], sha256=entry['catalog_sha256'])):
                raise ValueError('Unregistered portfolio or physical fit mismatch on ' + key)
            paths[key] = row
            fixture.write(root / 'results.json', dict(paths=paths))
        fixture.write(root / 'all_models_frozen.json', {key: r['model_sha256'] for key, r in paths.items()})
        for key, row in paths.items():
            row['delivery'] = pilot.score_one(root / key, entries[key])
        comparisons = {name: evaluation(root, name, paths) for name in TASKS}
        if fixture.digest(sources()) != frozen:
            raise ValueError('Production sources changed during Q')
        fixture.write(root / 'results.json', dict(paths=paths, tasks=comparisons))
        fixture.write(root / 'completion.json', dict(status='passed', paths=len(paths), fits=len(fixture.trace(root)),
                      api_requests=0, api_rmb=0, all_frozen_before_scoring=True, course_validation_and_test_opened=False,
                      h1_accessed=False, seconds=time.monotonic()-began))
    except Exception as error:
        fixture.write(root / 'results.json', dict(paths=paths))
        fixture.write(root / 'completion.json', dict(status='failed', error=str(error), fits_observed=len(fixture.trace(root)),
                      api_requests=0, automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coupon', 'headlines', 'uci', 'weights', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())
