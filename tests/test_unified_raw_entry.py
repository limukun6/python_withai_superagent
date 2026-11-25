import tempfile
import unittest
from pathlib import Path

import numpy as np

from superagent.data import TrainingTask, load_prediction_features, load_training_task
from superagent.prediction_evidence import fingerprint
from unified_agent import audit_partition, pair_packet


class UnifiedRawEntryTests(unittest.TestCase):
    def fixture(self, repeated):
        x = [(str(i // 2 if repeated else i), f"z{i}") for i in range(100)]
        return TrainingTask("pair_text", x, np.arange(100) % 2, {
            "metric": "accuracy", "input_schema": {"modality": "pair_text", "pair_semantics": "preference",
                "pair_label_one": "left", "columns": [{"name": name, "type": "text"} for name in ("before", "after")],
                "target": "decision"}})

    def test_renamed_columns_keep_disjoint_groups_and_feature_order(self):
        task = self.fixture(True)
        (fit, held, info), _ = audit_partition(task)
        self.assertEqual(info["folds"], 5)
        self.assertTrue({task.X_train[i][0] for i in fit}.isdisjoint(task.X_train[i][0] for i in held))
        self.assertEqual(fingerprint(audit_partition(task)[0][1].tolist()), fingerprint(held.tolist()))
        with tempfile.TemporaryDirectory() as temporary:
            train, audit = Path(temporary) / "p.zip", Path(temporary) / "q.zip"
            pair_packet(train, task, fit, training=True)
            pair_packet(audit, task, held, training=False)
            training = load_training_task(train)
            self.assertEqual(training.X_train, [task.X_train[i] for i in fit])
            self.assertEqual(load_prediction_features(audit, training.kind, training.spec), [task.X_train[i] for i in held])
            np.testing.assert_array_equal(training.y_train, task.y_train[fit])

    def test_reserve_requires_observed_dependencies_and_adequate_support(self):
        self.assertIsNone(audit_partition(self.fixture(False))[0])
        task = self.fixture(True)
        task.y_train[:] = 0
        task.y_train[0] = 1
        self.assertIsNone(audit_partition(task)[0])
        task = self.fixture(True)
        task.spec["input_schema"]["pair_semantics"] = "symmetric"
        self.assertIsNone(audit_partition(task)[0])
