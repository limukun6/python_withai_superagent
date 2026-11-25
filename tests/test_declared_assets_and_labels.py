"""Declared-but-unused positive labels and optional-asset corruption are reported honestly."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from superagent.contracts import TaskSpec
from tools.encoder_assets import MODELS, matches, verified


def task(metric, positive=None):
    spec = {'metric': metric}
    if positive is not None:
        spec['positive_label'] = positive
    return SimpleNamespace(kind='numeric_tabular', X_train=np.zeros((4, 1)),
                           y_train=np.array(['a', 'z', 'a', 'z']), spec=spec)


def positive_field(task_value):
    return TaskSpec.from_task(task_value).record()['goal_resolution']['fields']['positive_label']


class DeclaredPositiveLabelTests(unittest.TestCase):
    def test_a_positive_label_used_by_the_metric_is_confirmed(self):
        field = positive_field(task('binary_f1', 'a'))
        self.assertEqual(field['status'], 'confirmed')
        self.assertEqual(field['value'], 'a')

    def test_a_positive_label_ignored_by_the_metric_is_not_applicable(self):
        field = positive_field(task('accuracy', 'a'))
        self.assertEqual(field['status'], 'not_applicable')
        self.assertIn('does not score a single positive class', field['reason'])

    def test_macro_f1_also_leaves_a_declared_positive_label_unused(self):
        self.assertEqual(positive_field(task('macro_f1', 'z'))['status'], 'not_applicable')

    def test_an_undeclared_positive_label_on_accuracy_stays_not_applicable(self):
        field = positive_field(task('accuracy'))
        self.assertEqual(field['status'], 'not_applicable')
        self.assertNotIn('reason', field)


class OptionalAssetCorruptionTests(unittest.TestCase):
    name = sorted(MODELS)[0]

    def test_a_missing_asset_folder_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verified(Path(tmp), self.name)

    def test_a_present_but_corrupted_asset_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / self.name
            folder.mkdir()
            for filename in MODELS[self.name]['files']:
                (folder / filename).write_bytes(b'not the verified weights')
            self.assertFalse(matches(folder / next(iter(MODELS[self.name]['files'])),
                                     next(iter(MODELS[self.name]['files'].values()))))
            with self.assertRaises(ValueError):
                verified(Path(tmp), self.name)

    def test_the_error_names_the_offending_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / self.name
            folder.mkdir()
            first = next(iter(MODELS[self.name]['files']))
            (folder / first).write_bytes(b'corrupt')
            with self.assertRaises(ValueError) as caught:
                verified(Path(tmp), self.name)
            self.assertIn(first, str(caught.exception))


if __name__ == '__main__':
    unittest.main()
