"""Release checks: training read boundary, per-trial metric, temporal edge cases."""
import io
import json
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from superagent.data import load_development_task, load_training_task
from superagent.splits import temporal_partition

REPO = Path(__file__).resolve().parents[1]
COLUMNS = [{'name': 'date', 'type': 'categorical'}, {'name': 'user', 'type': 'categorical'}]


def write_package(path, *, with_test_labels=True):
    def rows(prefix, count):
        body = io.StringIO()
        body.write('date,user,label\n')
        for index in range(count):
            body.write(f'{prefix}{index:03d},u{index % 4},{"yes" if index % 2 else "no"}\n')
        return body.getvalue()
    manifest = {'schema_version': 1, 'objective': 'classification', 'metric': 'accuracy',
                'modality': 'tabular', 'input_format': 'csv', 'columns': COLUMNS, 'target': 'label',
                'splits': {'train': {'format': 'csv', 'path': 'train.csv'},
                           'validation': {'format': 'csv', 'path': 'validation.csv'}},
                'source': {'dataset': 'boundary fixture', 'licence': 'internal'}}
    if with_test_labels:
        manifest['splits']['test'] = {'format': 'csv', 'path': 'test.csv'}
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('train.csv', rows('d', 40).encode())
        archive.writestr('validation.csv', rows('v', 20).encode())
        if with_test_labels:
            archive.writestr('test.csv', rows('t', 20).encode())
        archive.writestr('manifest.json', json.dumps(manifest).encode())
    return path


class TrainingReadBoundaryTests(unittest.TestCase):
    def test_training_never_returns_test_labels(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            package = write_package(Path(tmp) / 'boundary.zip')
            task = load_training_task(package)
            self.assertIsNone(getattr(task, 'y_test', None))
            self.assertIsNone(getattr(task, 'X_test', None))
            self.assertFalse(task.spec.get('test_accessed'))

    def test_a_development_load_does_not_mark_the_test_as_read(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            package = write_package(Path(tmp) / 'boundary.zip')
            task = load_development_task(package)
            self.assertEqual(len(task.y_train), 40)
            self.assertEqual(len(task.y_val), 20)
            self.assertFalse(task.spec.get('test_accessed'))


class PerTrialMetricTests(unittest.TestCase):
    """Every recorded trial must carry the contract metric, not a default."""

    RUNS = [REPO / 'research' / 'task1_menu' / 'old_baseline',
            REPO / 'research' / 'task1_menu' / 'new_full']

    def test_every_trial_carries_the_contract_metric(self):
        checked = 0
        for run in self.RUNS:
            history_path = run / 'history.json'
            if not history_path.is_file():
                continue
            task_spec = json.loads((run / 'task_spec.json').read_text(encoding='utf-8'))
            expected = task_spec['metric_spec']
            for row in json.loads(history_path.read_text(encoding='utf-8')):
                self.assertEqual(row['metric_spec'], expected, row['candidate'])
                self.assertEqual(row['primary_metric'], expected['name'], row['candidate'])
                checked += 1
        self.assertGreater(checked, 0, 'no recorded run was available to check')


class TemporalEdgeCaseTests(unittest.TestCase):
    def task(self, values):
        matrix = np.asarray(values).reshape(-1, 1)
        labels = np.asarray(['no', 'yes'] * (len(values) // 2))
        spec = {'input_schema': {'columns': [{'name': 'when'}], 'modality': 'tabular'},
                'evaluation_protocol': {'kind': 'temporal', 'column': 'when'}}
        return SimpleNamespace(kind='categorical_tabular', X_train=matrix, y_train=labels, spec=spec)

    def test_equal_timestamps_keep_every_row_and_split_stably(self):
        values = ['d1'] * 20 + ['d2'] * 20
        train, held, _ = temporal_partition(self.task(values), {'kind': 'temporal', 'column': 'when'})
        self.assertEqual(len(train) + len(held), 40)
        self.assertEqual(set(train.tolist()) & set(held.tolist()), set())
        self.assertLessEqual(max(values[i] for i in train), min(values[i] for i in held))

    def test_empty_values_sort_before_real_dates(self):
        values = [''] * 10 + [f'd{index:02d}' for index in range(30)]
        train, held, _ = temporal_partition(self.task(values), {'kind': 'temporal', 'column': 'when'})
        blanks = [index for index, value in enumerate(values) if value == '']
        self.assertTrue(all(index in set(train.tolist()) for index in blanks))
        self.assertLessEqual(max(values[i] for i in train), min(values[i] for i in held))

    def test_the_same_values_give_the_same_split_twice(self):
        values = ['d1'] * 10 + ['d2'] * 10 + ['d3'] * 20
        first = temporal_partition(self.task(values), {'kind': 'temporal', 'column': 'when'})[0].tolist()
        second = temporal_partition(self.task(values), {'kind': 'temporal', 'column': 'when'})[0].tolist()
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
