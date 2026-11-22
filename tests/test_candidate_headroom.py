from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from scripts.probe_task_headroom import legacy_probe, measure_worker, observed_coverage, paired, rankings
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import make_prediction_record


class CandidateHeadroomTests(unittest.TestCase):
    def setUp(self):
        self.metric = MetricSpec('macro_f1', ('ham', 'spam'))
        self.y = np.array(['ham'] * 8 + ['spam'] * 2)
        self.majority = np.array(['ham'] * 10)
        self.balanced = np.array(['ham'] * 6 + ['spam'] * 3 + ['ham'])
        self.payload = dict(kind='single_text', y_train=self.y, y_search=self.y, X_search=[str(i) for i in range(10)],
                            metric_spec=self.metric.record())

    def worker(self, name, predictions, digest):
        scores = self.metric.evaluate(self.y, predictions)
        return dict(prediction_record=make_prediction_record(predictions, self.y, digest),
                    metrics=dict(resolved_configuration=dict(id=name), primary_metric='macro_f1', metric_spec=self.metric.record(),
                                 search_accuracy=scores['accuracy'], search_score=scores['score']))

    def test_rank_by_macro_f1_and_pair_same_rows(self):
        data = {}
        for name, pred in [('majority', self.majority), ('balanced', self.balanced)]:
            data[name], _ = measure_worker(self.worker(name,pred,'1'*64), self.payload, '1'*64, name)
        result = rankings(data)
        self.assertEqual(result['search_score']['tied_best'], ['balanced'])
        self.assertEqual(result['search_accuracy']['tied_best'], ['majority'])
        change = paired(self.metric, self.y, self.majority, self.balanced, list(range(10)))
        self.assertEqual((change['newly_correct'],change['newly_wrong']), (1,2))
        self.assertGreater(change['right_minus_left_points'], 0)
        self.assertGreater(change['paired_bootstrap_sd_points'], 0)
        diagnostic = observed_coverage(self.metric, self.y, dict(a=self.majority, b=self.balanced), 'b')
        self.assertEqual((diagnostic['all_observed_wrong'], diagnostic['correctable_among_observed']), (1,2))

    def test_different_payload_and_wrong_primary_cannot_be_pooled(self):
        record = self.worker('one', self.majority,'0'*64)
        with self.assertRaisesRegex(ValueError,'training split'):
            measure_worker(record,self.payload,'1'*64,'one')
        record['metrics']['search_score'] = record['metrics']['search_accuracy']
        with self.assertRaisesRegex(ValueError,'scores disagree'):
            measure_worker(record,self.payload,'0'*64,'one')

    def test_legacy_reuse_tells_which_expanded_candidates_lack_measurements(self):
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / 'inner_data.joblib'
            joblib.dump(self.payload,file)
            folder = root / 'headroom'
            folder.mkdir()
            name = 'single_word_C0.3'
            joblib.dump(self.worker(name,self.majority,sha256(file.read_bytes()).hexdigest()),folder/(name+'.joblib'))
            with patch('superagent.benchmark.train_candidate', side_effect=AssertionError('unexpected fit')):
                record = legacy_probe(root,10,True)
            self.assertEqual((record['primary_metric'], record['evaluated'], len(record['missing'])), ('macro_f1',1,5))
            self.assertEqual(record['best_score'], self.metric.evaluate(self.y,self.majority)['score'])
