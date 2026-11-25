"""Declared temporal/entity isolation protocols: supported, checked, and never silent."""
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


def tabular_task(protocol):
    rows = 40
    matrix = np.column_stack([[f'd{index:03d}' for index in range(rows)],
                              [f'u{index % 5}' for index in range(rows)],
                              np.arange(rows)])
    labels = np.array(['no', 'yes'] * (rows // 2))
    spec = {'input_schema': {'columns': [{'name': 'date'}, {'name': 'user'}, {'name': 'value'}]},
            'evaluation_protocol': protocol}
    return SimpleNamespace(kind='categorical_tabular', X_train=matrix, y_train=labels, spec=spec)


def write_package(path, protocol):
    body = io.StringIO()
    body.write('date,user,value,label\n')
    for index in range(60):
        body.write(f'd{index:03d},u{index % 6},{index},{"yes" if index % 2 else "no"}\n')
    manifest = {'schema_version': 1, 'objective': 'classification', 'metric': 'accuracy',
                'modality': 'tabular', 'input_format': 'csv',
                'columns': [{'name': 'date', 'type': 'text'}, {'name': 'user', 'type': 'text'},
                            {'name': 'value', 'type': 'numeric'}],
                'target': 'label',
                'splits': {'train': {'format': 'csv', 'path': 'train.csv'},
                           'validation': {'format': 'csv', 'path': 'validation.csv'}},
                'evaluation_protocol': protocol,
                'source': {'dataset': 'synthetic isolation fixture', 'licence': 'internal'}}
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('train.csv', body.getvalue().encode('utf-8'))
        archive.writestr('validation.csv', body.getvalue().encode('utf-8'))
        archive.writestr('manifest.json', json.dumps(manifest).encode('utf-8'))
    return path


class TemporalProtocolTests(unittest.TestCase):
    def test_the_split_follows_the_declared_time_order(self):
        task = tabular_task({'kind': 'temporal', 'column': 'date', 'train_fraction': 0.75})
        train, held, resource = grouped_partition(task, 707)
        self.assertEqual(resource['protocol'], 'temporal')
        self.assertLess(max(task.X_train[train, 0]), min(task.X_train[held, 0]))
        self.assertEqual(len(train) + len(held), len(task.y_train))

    def test_a_missing_column_is_refused(self):
        task = tabular_task({'kind': 'temporal', 'column': 'missing'})
        with self.assertRaises(ValueError) as caught:
            grouped_partition(task, 707)
        self.assertIn('not declared in input columns', str(caught.exception))

    def test_an_out_of_range_fraction_is_refused(self):
        for fraction in (0.2, 1.0):
            with self.subTest(fraction=fraction):
                task = tabular_task({'kind': 'temporal', 'column': 'date', 'train_fraction': fraction})
                with self.assertRaises(ValueError):
                    grouped_partition(task, 707)


class EntityProtocolTests(unittest.TestCase):
    def test_no_entity_crosses_the_split(self):
        task = tabular_task({'kind': 'entity', 'column': 'user', 'train_fraction': 0.6})
        train, held, resource = grouped_partition(task, 707)
        self.assertEqual(resource['protocol'], 'entity')
        self.assertEqual(set(task.X_train[train, 1]) & set(task.X_train[held, 1]), set())

    def test_the_same_seed_gives_the_same_split(self):
        task = tabular_task({'kind': 'entity', 'column': 'user', 'train_fraction': 0.6})
        first = grouped_partition(task, 707)[0].tolist()
        second = grouped_partition(task, 707)[0].tolist()
        self.assertEqual(first, second)

    def test_the_entity_column_can_be_named_explicitly(self):
        task = tabular_task({'kind': 'entity', 'entity_column': 'user', 'train_fraction': 0.6})
        self.assertEqual(grouped_partition(task, 707)[2]['column'], 'user')


class UnsupportedProtocolTests(unittest.TestCase):
    def test_an_unknown_protocol_kind_is_refused(self):
        task = tabular_task({'kind': 'spatial'})
        with self.assertRaises(ValueError) as caught:
            grouped_partition(task, 707)
        self.assertIn('Unsupported declared evaluation protocol', str(caught.exception))

    def test_a_non_object_protocol_is_refused(self):
        task = tabular_task('temporal')
        with self.assertRaises(ValueError):
            grouped_partition(task, 707)

    def test_a_default_protocol_still_partitions(self):
        task = tabular_task({})
        train, held, _ = grouped_partition(task, 707, 'exact_rows')
        self.assertEqual(len(train) + len(held), len(task.y_train))


class RollingWindowTests(unittest.TestCase):
    def test_temporal_windows_never_share_evaluation_rows(self):
        from superagent.splits import temporal_windows
        task = tabular_task({'kind': 'temporal', 'column': 'date'})
        names = ('primary', 'confirmation_1', 'confirmation_2')
        windows = {name: (train, held) for name, train, held, _ in temporal_windows(task, task.spec['evaluation_protocol'], names)}
        searches = [set(windows[name][1].tolist()) for name in names]
        for left in range(len(searches)):
            for right in range(left + 1, len(searches)):
                self.assertEqual(searches[left] & searches[right], set())
        self.assertLess(max(task.X_train[windows['primary'][1], 0]),
                        min(task.X_train[windows['confirmation_1'][1], 0]))

    def test_entity_folds_never_share_entities(self):
        from superagent.splits import entity_windows
        task = tabular_task({'kind': 'entity', 'column': 'user'})
        names = ('primary', 'confirmation_1', 'confirmation_2')
        windows = {name: (train, held) for name, train, held, _ in entity_windows(task, 707, task.spec['evaluation_protocol'], names)}
        searches = [set(task.X_train[windows[name][1], 1].tolist()) for name in names]
        for left in range(len(searches)):
            for right in range(left + 1, len(searches)):
                self.assertEqual(searches[left] & searches[right], set())


class EndToEndTests(unittest.TestCase):
    def test_the_entry_runs_a_temporal_package_with_the_declared_split(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = write_package(root / 'temporal.zip',
                                    {'kind': 'temporal', 'column': 'date', 'train_fraction': 0.75})
            run_dir = root / 'run'
            completed = subprocess.run([sys.executable, '-m', 'superagent', 'run', str(package),
                                        '--run-dir', str(run_dir), '--planner', 'local',
                                        '--scheduler', 'ab-v1', '--portfolio', 'ab-v1',
                                        '--catalog-scope', 'baseline', '--fit-limit', '3',
                                        '--seed', '707'],
                                       cwd=REPO, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout[-800:] + completed.stderr[-800:])
            outcome = json.loads((run_dir / 'run_outcome.json').read_text(encoding='utf-8'))
            self.assertEqual(outcome['status'], 'completed')
            split = json.loads((run_dir / 'primary_split.json').read_text(encoding='utf-8'))
            self.assertEqual(split['split_resource']['protocol'], 'temporal')
            # Rolling origin: the primary window trains on the earlier 70% and is scored on the next 10%.
            self.assertEqual(split['split_resource']['window'], [0.0, 0.7, 0.8])
            self.assertEqual(split['train_rows'] + split['search_rows'], 48)


if __name__ == '__main__':
    unittest.main()
