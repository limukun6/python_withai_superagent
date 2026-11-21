"""Freeze two pair-text tasks, run the production scheduler, then score their frozen outputs."""
import argparse
from collections import Counter
import csv
import gzip
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as fixture
from scripts.build_external_tasks import write_member
from superagent.ab_catalog import capabilities_for
from superagent.ab_run import freeze_partitions, sources
from superagent.ab_sentence import FILES, REVISION, verified_directory
from superagent.contracts import TaskSpec
from superagent.data import TrainingTask, load_development_task
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model
from superagent.splits import grouped_partition, overlap_audit, take


ROOT = fixture.ROOT
TASKS = ('symmetric_fixture', 'headlines')
INNER_SEED = 707
OUTER_SEED = 45101
WALL = 1200
TRIAL = 300


def environment():
    libraries = ('numpy', 'scipy', 'scikit-learn', 'joblib', 'catboost', 'torch', 'transformers',
                 'tokenizers', 'safetensors', 'threadpoolctl')
    from importlib.util import find_spec
    from importlib.metadata import PackageNotFoundError
    versions = {}
    for name in libraries:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.system(), machine=platform.machine(),
                packages=versions, polars=version('polars') if find_spec('polars') else None,
                minilm_revision=REVISION if verified_directory() else None,
                minilm_files={name: digest for name, (_, digest) in FILES.items()} if verified_directory() else {})


def identities_file(path, rows):
    raw = json.dumps(rows, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    path.write_bytes(gzip.compress(raw, mtime=0))
    return fixture.sha(raw)


def pair_package(path, pairs, labels, semantics, split):
    manifest = dict(schema_version=1, objective='classification', metric='accuracy', modality='pair_text',
                    pair_semantics=semantics, columns=[{'name': c, 'type': 'text'} for c in ('left', 'right')],
                    target='label', splits={split: {'format': 'csv', 'path': 'pairs.csv'}})
    if semantics == 'preference':
        manifest['pair_label_one'] = 'left'
    buf = io.StringIO(newline='')
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerow(['left', 'right'] + (['label'] if labels is not None else []))
    for i, (left, right) in enumerate(pairs):
        writer.writerow([left, right] + ([labels[i]] if labels is not None else []))
    with zipfile.ZipFile(path, 'x') as archive:
        write_member(archive, 'manifest.json', json.dumps(manifest, sort_keys=True).encode())
        write_member(archive, 'pairs.csv', buf.getvalue().encode())


def pair_source(name, released):
    if name == 'headlines':
        with zipfile.ZipFile(released) as archive:
            matches = [p for p in archive.namelist() if Path(p).name == 'train.csv' and not p.startswith('__MACOSX/')]
            if len(matches) != 1:
                raise ValueError('Expected exactly one course training CSV')
            raw = archive.read(matches[0])
        rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        pairs = [(row['headline_A'], row['headline_B']) for row in rows]
        labels = np.asarray([int(row['label_pairwise']) for row in rows])
        if len(pairs) != 8301 or set(labels) != {0, 1}:
            raise ValueError('Released training input differs from the registered size or class domain')
        return pairs, labels, 'preference', dict(training_member_sha256=fixture.sha(raw), opened=['train.csv'])
    words = ('blue', 'green', 'rust', 'red', 'yellow', 'black', 'gray', 'brown')
    pairs, labels = [], []
    for i in range(240):
        a, b = words[i % len(words)], words[(i + 3) % len(words)]
        same = i % 3 != 1
        pairs.append((f'card {i} {a} triangle', f'{a if same else b} on card {i} triangle'))
        labels.append('same' if same else 'different')
    return pairs, np.asarray(labels), 'symmetric', dict(generator='anonymous color agreement', data_sha256=fixture.digest(pairs))


def prepare(root, name, released):
    return prepare_pairs(root, name, *pair_source(name, released), outer_seed=OUTER_SEED)


def prepare_pairs(root, name, x, y, semantics, source, *, outer_seed):
    path = root / name
    path.mkdir()
    original = TrainingTask('pair_text', x, y)
    train, held, resource = grouped_partition(original, outer_seed, 'shared_text')
    audit = overlap_audit(original, train, held)
    if audit['search_rows_sharing_text']:
        raise ValueError('A held-out text also occurs in training')
    test_x = take(x, held)
    pair_package(path / 'train.zip', take(x, train), y[train], semantics, 'train')
    pair_package(path / 'features.zip', test_x, None, semantics, 'test')
    if semantics == 'symmetric':
        pair_package(path / 'reversed.zip', [(b, a) for a, b in test_x], None, semantics, 'test')
    joblib.dump(dict(X=test_x, y=y[held], row_ids=held.tolist()), path / 'holdout.joblib')
    return register_prepared_task(path, source, semantics, len(y), train, held, resource, audit)


def register_prepared_task(path, source, semantics, source_rows, train, held, resource, audit):
    task = load_development_task(path / 'train.zip')
    if task.X_val is not None or task.y_val is not None:
        raise ValueError('Optimizer received an outer validation set')
    partition_dir = path / 'registered'
    partition_dir.mkdir()
    _, registered = freeze_partitions(task, partition_dir, INNER_SEED, MetricSpec.for_task(task))
    caps = capabilities_for(TaskSpec.from_task(task).record(), train_rows=len(train))
    ids = dict(outer=dict(train=train.tolist(), held=held.tolist()), inner={})
    for split in registered:
        record = fixture.read(partition_dir / (split + '_split.json'))
        if record.get('search_rows_sharing_text', record.get('search_duplicate_rows')) != 0:
            raise ValueError('A group occurs on both sides of an inner split')
        ids['inner'][split] = {key: record[key] for key in ('train_indices', 'search_indices')}
    return dict(name=path.name, semantics=semantics, source=source, source_rows=source_rows, inner_input_rows=len(train),
                outer_rows=len(held), grouping=resource, overlap=audit, splits=registered,
                row_ids_sha256=identities_file(path / 'row_ids.json.gz', ids),
                files={p.name: fixture.sha(p.read_bytes()) for p in sorted(path.glob('*.zip'))},
                holdout_sha256=fixture.sha((path / 'holdout.joblib').read_bytes()),
                catalog_sha256=fixture.digest(caps), candidate_ids=list(caps))


def child(command, log, timeout):
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{log.name} returned exit {result.returncode}')


def train_one(path, entry, frozen, *, launcher=('-m', 'superagent', 'run')):
    run = path / 'run'
    child([*launcher, path / 'train.zip', '--scheduler', 'ab-v1', '--portfolio', 'ab-v1',
           '--fit-limit', '11', '--seed', INNER_SEED, '--time-limit', WALL, '--trial-timeout', TRIAL,
           '--run-dir', run], path / 'train.log', WALL + 60)
    outcome = fixture.read(run / 'run_outcome.json')
    ledger = fixture.read(run / 'classifier_fits.json')
    manifest = fixture.read(run / 'run_manifest.json')
    if not (outcome['requested_planner_completed'] and outcome['full_refit_completed'] and outcome['new_api_requests'] == 0
            and outcome['source_unchanged'] and ledger['unknown_failed_attempts'] == 0
            and 1 < ledger['attempted_classifier_fits'] == ledger['completed_classifier_fits'] <= 11
            and fixture.digest(manifest['source_files']) == frozen
            and fixture.digest(manifest['capabilities']) == entry['catalog_sha256']
            and manifest['task_archive_sha256'] == entry['files']['train.zip']):
        raise ValueError('Run differs from the frozen sources, budget or capability registry')
    for split in entry['splits']:
        if fixture.read(run / (split + '_split.json')) != fixture.read(path / 'registered' / (split + '_split.json')):
            raise ValueError('Training used different rows or a different split payload')
    selection = fixture.read(run / 'final_selection.json')
    if not selection['refit'] or fixture.sha((run / selection['model_file']).read_bytes()) != selection['model_sha256']:
        raise ValueError('Final frozen model mismatch')
    return dict(name=entry['name'], fits=ledger['completed_classifier_fits'], workers=ledger['workers'],
                actions=fixture.read(run / 'actions.json'), action_counts=outcome['action_counts'],
                model_sha256=selection['model_sha256'], selected=selection['family'], seconds=outcome['elapsed_seconds'],
                promotions=fixture.read(run / 'confirmation.json')['b_scheduler_review']['promotions'])


def score_one(path, entry):
    run, relocated = path / 'run', path / 'relocated'
    relocated.mkdir()
    selection = fixture.read(run / 'final_selection.json')
    for filename in ('task_spec.json', 'final_selection.json', selection['model_file']):
        shutil.copy2(run / filename, relocated / filename)
    outputs = []
    for i, source in enumerate((run, relocated)):
        dest = path / f'pred_{i}' / 'rows.csv'
        child(['-m', 'superagent', 'predict', path / 'features.zip', '--run-dir', source, '--output', dest],
              path / f'predict_{i}.log', TRIAL)
        outputs.append(dest)
    if outputs[0].read_bytes() != outputs[1].read_bytes():
        raise ValueError('Relocation or fresh-process prediction changed rows')
    if fixture.sha((path / 'holdout.joblib').read_bytes()) != entry['holdout_sha256']:
        raise ValueError('Registered holdout changed')
    child([ROOT / 'scripts/a07_verify_export.py', '--source-root', ROOT, '--run-dir', relocated,
           '--holdout', path / 'holdout.joblib', '--predictions', outputs[0], '--output', path / 'verification.json'],
          path / 'verify.log', TRIAL)
    receipt = fixture.read(path / 'verification.json')
    if not (receipt['reload_identical'] and receipt['heldout_scored'] and receipt['rows'] == entry['outer_rows']):
        raise ValueError('Prediction rows, held labels or scores could not be verified')
    receipt.update(correct=round(receipt['accuracy'] * receipt['rows']), csv_sha256=fixture.sha(outputs[0].read_bytes()))
    if entry['semantics'] == 'symmetric':
        dest = path / 'pred_swap' / 'rows.csv'
        child(['-m', 'superagent', 'predict', path / 'reversed.zip', '--run-dir', relocated, '--output', dest],
              path / 'swap.log', TRIAL)
        model = joblib.load(relocated / selection['model_file'])['model']
        rows = joblib.load(path / 'holdout.joblib')['X']
        classes, forward = probability_with_model('pair_text', model, rows)
        reverse_classes, reverse = probability_with_model('pair_text', model, [(b, a) for a, b in rows])
        if (dest.read_bytes() != outputs[0].read_bytes() or not np.array_equal(classes, reverse_classes)
                or not np.isfinite(forward).all() or not np.array_equal(forward, reverse)):
            raise ValueError('Swapping input order changed a label or a class probability')
        receipt.update(swap_label_mismatches=0, swap_max_probability_difference=0.0, swap_rows=len(rows))
    return receipt


def run(root, released, weights):
    root.mkdir(parents=True, exist_ok=False)
    fixture.offline(root)
    os.environ.update(SUPERAGENT_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if weights:
        os.environ['SUPERAGENT_MINILM_WEIGHTS'] = str(weights.resolve())
        if verified_directory() is None:
            raise ValueError('Requested pinned encoder failed file or dependency verification')
    started = time.monotonic()
    results = []
    try:
        frozen = fixture.digest(sources())
        entries = [prepare(root, name, released) for name in TASKS]
        protocol = dict(batch='B1-05', base='5ed06c66d972a50413201e3a2f2d9f74f404d214',
                        source_sha256=frozen, runner_sha256=fixture.sha(Path(__file__).read_bytes()),
                        plan_sha256=fixture.sha((ROOT / 'docs/batches/B1_05_plan.md').read_bytes()),
                        environment=environment(), outer_seed=OUTER_SEED, inner_seed=INNER_SEED,
                        fit_limit_each=11, api_request_limit=0, wall_seconds=WALL, trial_seconds=TRIAL, tasks=entries)
        fixture.write(root / 'protocol.json', protocol)
        for entry in entries:
            results.append(train_one(root / entry['name'], entry, frozen))
            fixture.write(root / 'results.json', results)
        fixture.write(root / 'all_models_frozen.json', {r['name']: r['model_sha256'] for r in results})
        for row, entry in zip(results, entries, strict=True):
            row['delivery'] = score_one(root / entry['name'], entry)
        trace = fixture.trace(root)
        if sum(row['fits'] for row in results) != len(trace) or frozen != fixture.digest(sources()):
            raise ValueError('Independent count or production sources changed')
        fixture.write(root / 'results.json', results)
        fixture.write(root / 'completion.json', dict(status='passed', paths=len(results), fits=len(trace),
                      fits_by_classifier=dict(Counter(r['classifier'] for r in trace)), api_requests=0, api_rmb=0,
                      course_validation_and_test_opened=False, seconds=time.monotonic() - started))
    except Exception as error:
        fixture.write(root / 'results.json', results)
        fixture.write(root / 'completion.json', dict(status='failed', error=str(error), fits_observed=len(fixture.trace(root)),
                      api_requests=0, automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--weights', type=Path, help='Optional verified local frozen encoder')
    args = parser.parse_args()
    run(args.output.resolve(), args.released.resolve(), args.weights)
