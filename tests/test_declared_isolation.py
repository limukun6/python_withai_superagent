"""A declared temporal/entity isolation protocol is refused, not silently downgraded."""
import io
import json
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from superagent.data import load_development_task
from superagent.splits import grouped_partition

REPO = Path(__file__).resolve().parents[1]


def make_package(path, protocol):
    buffer = io.BytesIO()
    rows = [[f'text {index}', 'yes' if index % 2 else 'no'] for index in range(40)]
    body = io.StringIO()
    body.write('text,label\n')
    for row in rows:
        body.write(','.join(row) + '\n')
    manifest = {'schema_version': 1, 'objective': 'classification', 'metric': 'accuracy',
                'modality': 'text', 'input_format': 'single_text_csv',
                'columns': [{'name': 'text', 'type': 'text'}], 'target': 'label',
                'splits': {'train': {'format': 'csv', 'path': 'train.csv'},
                           'validation': {'format': 'csv', 'path': 'validation.csv'}},
                'evaluation_protocol': protocol,
                'source': {'dataset': 'synthetic isolation fixture', 'licence': 'internal'}}
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('train.csv', body.getvalue().encode('utf-8'))
        archive.writestr('validation.csv', body.getvalue().encode('utf-8'))
        archive.writestr('manifest.json', json.dumps(manifest).encode('utf-8'))
    return path


class DeclaredIsolationTests(unittest.TestCase):
    def test_a_temporal_protocol_is_refused_by_the_partitioner(self):
        task = load_development_task(make_package(Path(_tempdir(self)) / 'temporal.zip',
                                                 {'kind': 'temporal', 'time_column': 'text'}))
        with self.assertRaises(ValueError) as caught:
            grouped_partition(task, 707, 'exact_rows')
        self.assertIn('temporal/entity evaluation protocol', str(caught.exception))

    def test_an_entity_protocol_is_refused_by_the_partitioner(self):
        task = load_development_task(make_package(Path(_tempdir(self)) / 'entity.zip',
                                                 {'kind': 'entity', 'entity_column': 'text'}))
        with self.assertRaises(ValueError):
            grouped_partition(task, 707, 'exact_rows')

    def test_a_non_object_protocol_is_refused(self):
        task = load_development_task(make_package(Path(_tempdir(self)) / 'bad.zip', 'temporal'))
        with self.assertRaises(ValueError):
            grouped_partition(task, 707, 'exact_rows')

    def test_the_entry_refuses_end_to_end_and_trains_nothing(self):
        root = Path(_tempdir(self))
        package = make_package(root / 'temporal.zip', {'kind': 'temporal', 'time_column': 'text'})
        run_dir = root / 'run'
        completed = subprocess.run([sys.executable, '-m', 'superagent', 'run', str(package),
                                    '--run-dir', str(run_dir), '--planner', 'local',
                                    '--scheduler', 'ab-v1', '--portfolio', 'ab-v1',
                                    '--catalog-scope', 'baseline', '--fit-limit', '3',
                                    '--seed', '707'],
                                   cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        # A runtime refusal is reported inside the outcome JSON on stdout, not on stderr.
        self.assertIn('temporal/entity evaluation protocol', completed.stdout)
        self.assertFalse((run_dir / 'trial_1.joblib').is_file())
        outcome = json.loads((run_dir / 'run_outcome.json').read_text(encoding='utf-8'))
        self.assertEqual(outcome['status'], 'failed')
        self.assertEqual(outcome['classifier_fit_accounting']['attempted_classifier_fits'], 0)

    def test_a_default_protocol_still_partitions(self):
        task = load_development_task(make_package(Path(_tempdir(self)) / 'plain.zip', {}))
        train, held, resource = grouped_partition(task, 707, 'exact_rows')
        self.assertEqual(len(train) + len(held), len(task.y_train))
        self.assertEqual(set(np.asarray(task.y_train)[train].tolist()),
                         set(np.asarray(task.y_train)[held].tolist()))


_TEMPDIRS = {}


def _tempdir(case):
    import tempfile
    key = id(case)
    if key not in _TEMPDIRS:
        holder = tempfile.TemporaryDirectory()
        _TEMPDIRS[key] = holder
        case.addCleanup(holder.cleanup)
    return _TEMPDIRS[key].name


if __name__ == '__main__':
    unittest.main()
