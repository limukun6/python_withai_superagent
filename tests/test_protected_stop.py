import unittest
from pathlib import Path
from unittest.mock import Mock

from superagent.protected_stop import review_stop


class ProtectedStopTests(unittest.TestCase):
    def state(self, limit, unknown=False):
        class Log:
            def __init__(self):
                self.records = {'final_selection.json': {'model_sha256':'incumbent'}}
            def write_json(self, name, value):
                self.records[name] = value.copy()
        executor = Mock()
        executor.limit = limit
        executor.caps = {'competitor': {'classifier_fits':1, 'refit_classifier_fits':1}}
        executor.accounting.return_value = {'attempted_classifier_fits': 8, 'unknown_failed_attempts': int(unknown)}
        executor.worker.side_effect = RuntimeError('injected worker failure')
        stop = {'members':['leader'], 'evidence':{'scores':{'primary':.8, 'confirmation_1':.78}}}
        menu = [{'action':'explore','members':['competitor'],'split':'primary'}]
        parent = {'model_sha256':'incumbent'}
        return executor, Log(), stop, menu, parent

    def run_case(self, state):
        executor, log, stop, menu, parent = state
        return review_stop(None, None, None, menu, stop, parent, executor, Path('.'), log, 3, {},
                           {'primary':{'path':Path('primary.joblib')}})

    def test_protects_full_refit_before_attempting_either_training_partition(self):
        state = self.state(10)
        executor, log, stop, _, parent = state
        self.assertEqual(self.run_case(state), (parent, stop))
        executor.worker.assert_not_called()
        self.assertEqual(log.records['stop_review.json']['status'], 'insufficient_room_or_evidence')
        self.assertEqual(log.records['final_selection.json'], parent)

    def test_known_worker_failure_keeps_prior_delivery(self):
        state = self.state(16)
        executor, log, stop, _, parent = state
        self.assertEqual(self.run_case(state), (parent, stop))
        executor.worker.assert_called_once()
        self.assertEqual(log.records['stop_review.json']['status'], 'incumbent_retained_after_error')
        self.assertEqual(log.records['final_selection.json'], parent)

    def test_unaccounted_failure_surfaces_with_the_intact_delivery(self):
        state = self.state(16, unknown=True)
        with self.assertRaisesRegex(RuntimeError, 'Unknown physical fit'):
            self.run_case(state)
        self.assertEqual(state[1].records['final_selection.json'], state[4])

    def test_export_includes_both_frozen_models_and_checks_the_winner(self):
        import hashlib
        import json
        import tempfile
        from scripts.package_unified_agent import model_files
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            agent = run / 'agent'
            agent.mkdir()
            for manifest,name in [('final_selection.json','challenger_selected_model.joblib'),
                                  ('incumbent_selection.json','selected_model.joblib')]:
                data = name.encode()
                (agent/name).write_bytes(data)
                (agent/manifest).write_text(json.dumps({'model_file':name, 'model_sha256':hashlib.sha256(data).hexdigest()}))
            (agent/'stop_review.json').write_text('{}')
            names = {item.name for item in model_files(run)}
            self.assertTrue({'selected_model.joblib','challenger_selected_model.joblib','stop_review.json'} <= names)
            (agent/'challenger_selected_model.joblib').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'disagree'):
                model_files(run)
