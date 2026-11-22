import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier

from scripts import run_b1_ab_acceptance as acceptance


class FrozenDeliveryTests(unittest.TestCase):
    def test_verify_twice_in_new_workspaces_preserves_completed_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task, query = acceptance.packages(root)
            delivery = root / 'delivered'
            delivery.mkdir()
            (delivery / 'relocated').mkdir()
            old_predictions = delivery / 'predict' / 'rows.csv'
            old_predictions.parent.mkdir()
            old_predictions.write_text('earlier delivery\n', encoding='utf-8')

            model = DummyClassifier().fit(task.X_train, task.y_train)
            model_file = delivery / 'model.joblib'
            family = 'fixture_dummy'
            joblib.dump(dict(model=model, kind=task.kind, candidate=family), model_file)
            schema = delivery / 'task_spec.json'
            acceptance.write(schema, dict(task_kind='numeric_tabular_classification'))
            acceptance.write(delivery / 'final_selection.json', dict(
                selected_experiment='fixture', model_file=model_file.name, family=family,
                model_format='candidate_bundle_v1', model_sha256=acceptance.sha(model_file.read_bytes()),
                task_spec_sha256=acceptance.sha(schema.read_bytes())))

            def original_files():
                return {p.relative_to(delivery): (p.read_bytes(), p.stat().st_mtime_ns)
                        for p in delivery.rglob('*') if p.is_file()}

            before = original_files()
            for name in ('first_check', 'second_check'):
                check = root / name
                check.mkdir()
                receipt = acceptance.predict_twice(delivery, query, output_dir=check, task_package=root / 'test.zip')
                first = check / 'predict' / 'rows.csv'
                second = check / 'relocated' / 'predict' / 'rows.csv'
                self.assertEqual(receipt['rows_per_prediction'], len(query))
                self.assertEqual(first.read_bytes(), second.read_bytes())
                self.assertEqual(acceptance.sha(first.read_bytes()), receipt['prediction_csv_sha256'])
                self.assertEqual(original_files(), before)
