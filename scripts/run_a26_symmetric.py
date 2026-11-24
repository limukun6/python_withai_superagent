"""Offline A26 regression or anonymous symmetric CLI acceptance; no H data or API."""
import argparse
import contextlib
import csv
from functools import wraps
import io
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import unittest
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

from scripts.a_batch_common import save
from scripts import run_b1_ab_acceptance as fixture
from scripts.run_a10_real_task_pilot import launch, read, sha
from superagent.ab_run import sources


def checks(root, include_coverage=False):
    suites = ['tests.test_ab_official_cli', 'tests.test_ab_bridge_adapter', 'tests.test_ab_scheduler',
        'tests.test_ab_comparison', 'tests.test_ab_confirmation_budget', 'tests.test_unified_cli.UnifiedCLITests',
        'tests.test_ab_fixed_schedule', 'tests.test_ab_flash', 'tests.test_symmetric_input',
        'tests.test_general_schema', 'tests.test_native_schema', 'tests.test_image_files',
        'tests.test_catalog_scope', 'tests.test_task_readme', 'tests.test_incremental_research_account',
        'tests.test_no_dispatch_review', 'tests.test_symmetric_text', 'tests.test_task_semantics']
    if include_coverage:
        suites.append('tests.test_candidate_coverage')
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    calls = []
    def observe(original):
        @wraps(original)
        def fit(model, *args, **kwargs):
            calls.append(dict(classifier=type(model).__name__, training_rows=args[0].shape[0] if args and hasattr(args[0],'shape') else len(args[0]) if args else None,
                test_call_sites=[dict(file=Path(frame.filename).name,line=frame.lineno,function=frame.name)
                    for frame in traceback.extract_stack() if Path(frame.filename).parent.name == 'tests']))
            return original(model, *args, **kwargs)
        return fit
    with (root / 'tests.log').open('w', encoding='utf-8') as log, \
         contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), \
         patch.object(LogisticRegression, 'fit', observe(LogisticRegression.fit)), \
         patch.object(DummyClassifier, 'fit', observe(DummyClassifier.fit)):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    save(root / 'fit_observations.json', calls)
    return dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), actual_classifier_fits=len(calls),
        expected_classifier_fits=7, preprocessing_fit_count='multiple; not counted as supervised fits')


def rows(index):
    first = ('red cat sleeps', 'blue dog runs', 'green bird sings', 'white fish swims')[index % 4]
    if index % 2:
        return (first + f' item{index}', f'item{index} ' + first), 1
    second = ('bright car races', 'large boat sails', 'tiny plane flies', 'fast train moves')[index % 4]
    return (first + f' item{index}', second + f' unit{index}'), 0


def package(path, records, labelled):
    manifest = dict(schema_version=1, objective='classification', modality='pair_text', metric='accuracy',
        pair_semantics='symmetric', pair_label_one='equivalent', target='label',
        columns=[dict(name='a', type='text'), dict(name='b', type='text')],
        splits={'train' if labelled else 'test': dict(format='csv', path='rows.csv')})
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    writer.writerow(['a', 'b', 'label'] if labelled else ['a', 'b'])
    writer.writerows([(*pair, label) if labelled else pair for pair, label in records])
    with zipfile.ZipFile(path, 'x') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('rows.csv', buffer.getvalue())


def cli(root):
    package(root / 'train.zip', [rows(i) for i in range(160)], True)
    queries = [rows(i) for i in range(200, 224)]
    package(root / 'features.zip', queries, False)
    package(root / 'swapped.zip', [((b, a), y) for (a, b), y in queries], False)
    directory = root / 'official_run'
    command = [sys.executable, '-m', 'superagent', 'run', str(root / 'train.zip'), '--run-dir', str(directory),
        '--scheduler', 'ab-v1', '--portfolio', 'ab-v1', '--fit-limit', '11', '--seed', '707',
        '--time-limit', '300', '--trial-timeout', '60']
    save(root / 'command.json', command)
    launch(command, root / 'train.log', 330)
    outcome = read(directory / 'run_outcome.json')
    if not outcome['requested_planner_completed']:
        raise ValueError('Anonymous symmetric CLI did not complete')
    selection = read(directory / 'final_selection.json')
    moved = root / 'relocated'
    moved.mkdir()
    for name in ('task_spec.json', 'final_selection.json', selection['model_file']):
        shutil.copy2(directory / name, moved / name)
    predictions = []
    for index, (run, features) in enumerate(((directory, 'features.zip'), (moved, 'features.zip'), (moved, 'swapped.zip'))):
        path = root / f'predictions_{index}.csv'
        launch([sys.executable, '-m', 'superagent', 'predict', str(root / features), '--run-dir', str(run),
                '--output', str(path)], root / f'predict_{index}.log', 60)
        predictions.append(path.read_bytes())
    if not predictions[0] == predictions[1] == predictions[2]:
        raise ValueError('Export changed after relocation or input swap')
    model = joblib.load(directory / selection['model_file'])['model']
    x = [pair for pair, _ in queries]
    proba = model.predict_proba(x)
    reverse = model.predict_proba([(b, a) for a, b in x])
    if not np.array_equal(proba, reverse):
        raise ValueError('Swapped probabilities differ')
    count = len(fixture.trace(root))
    if not 2 <= count == outcome['classifier_fit_accounting']['attempted_classifier_fits'] <= 11:
        raise ValueError('Physical fit trace disagrees with the execution budget')
    return dict(status='passed', actual_classifier_fits=count, selected_candidate=outcome['selected_candidate'],
        cli_seconds=outcome['elapsed_seconds'], exported_rows=len(x), relocated_predictions_identical=True,
        swapped_predictions_identical=True, maximum_swap_probability_error=float(np.max(abs(proba - reverse))),
        evaluation='Synthetic engineering acceptance only; no true-task accuracy or H evaluation',
        run_outcome=outcome, actions=read(directory / 'actions.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['checks', 'cli'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-coverage', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixture.offline(root)
    start = time.monotonic()
    frozen = sources()
    save(root / 'protocol.json', dict(batch='A26', phase=args.phase, source_files=frozen,
        plan_sha256=sha(fixture.ROOT / 'docs/batches/A26_symmetric_methods_plan.md'), runner_sha256=sha(Path(__file__)),
        classifier_fit_limit=7 if args.phase == 'checks' else 11, real_api_limit=0, H_data_access=False))
    try:
        value = checks(root, args.include_coverage) if args.phase == 'checks' else cli(root)
        save(root / 'results.json', value)
        save(root / 'completion.json', dict(status=value['status'], elapsed_seconds=time.monotonic() - start,
            actual_classifier_fits=value['actual_classifier_fits'], real_api_requests=0, source_unchanged=sources() == frozen,
            H_data_access=False))
        print(json.dumps({k: v for k, v in value.items() if k not in ('run_outcome', 'actions')}))
        if value['status'] != 'passed':
            raise SystemExit(1)
    except Exception as error:
        save(root / 'completion.json', dict(status='failed', error=str(error), elapsed_seconds=time.monotonic() - start,
            real_api_requests=0, observed_worker_fits=len(fixture.trace(root))))
        raise
