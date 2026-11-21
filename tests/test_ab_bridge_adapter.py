from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np

from superagent.ab_catalog import capabilities_for, untried_candidates
from superagent.ab_scheduler import next_action, selection_evidence
from superagent.models import search_candidates
from superagent.scheduler_bridge import validate_action
from tests.test_ab_scheduler import competing, recorded


class TestABBridgeAdapter(unittest.TestCase):
    def test_registered_capabilities_resolve_in_the_training_worker(self):
        kinds = ('numeric_tabular', 'mixed_tabular', 'text_tabular', 'categorical_tabular',
                 'single_text', 'pair_text', 'image')
        for kind in kinds:
            with self.subTest(kind=kind):
                task = recorded()['task']
                task.update(kind=kind, input_schema={'pair_semantics': 'preference'} if kind == 'pair_text' else {})
                caps = capabilities_for(task, train_rows=256)
                actual = search_candidates(kind, 'all')
                self.assertTrue(caps)
                for name, cap in caps.items():
                    candidate = next(entry for entry in actual if entry.family == name)
                    self.assertEqual(cap['params'], candidate.params)
                    self.assertTrue(callable(candidate.fit_predict))
                    self.assertEqual((cap['kind'], cap['classifier_fits'], cap['refit_classifier_fits']), (kind, 1, 1))
                    self.assertIs(type(cap['supports_probabilities']), bool)
                    self.assertIn('swap_label_rule', cap)

    def test_shared_untried_menu_is_isolated_from_executor_and_omits_failures(self):
        state = recorded()
        caps = capabilities_for(state['task'], train_rows=256)
        original = deepcopy(caps)
        local = untried_candidates(caps, state['history'])
        flash = untried_candidates(caps, state['history'])
        self.assertEqual(local, flash)
        local[0]['params']['injected'] = True
        self.assertEqual(caps, original)
        self.assertNotEqual(local, flash)
        failed = {'candidate': flash[0]['id'], 'status': 'failed'}
        after = untried_candidates(caps, [*state['history'], failed])
        self.assertEqual(after, flash[1:])

    def test_history_only_confirmation_passes_real_gate_with_either_evidence_spelling(self):
        state = competing()
        caps = capabilities_for(state['task'], train_rows=256)
        state['available_candidates'] = untried_candidates(caps, state['history'])
        reference = next_action(state, capabilities=caps)
        for item in state['prediction_evidence']:
            item['candidate'] = item.pop('candidate_id')
        self.assertEqual(next_action(state, capabilities=caps), reference)
        self.assertEqual(reference['action'], 'confirm')
        name = reference['candidate_id']
        self.assertNotIn(name, [c['id'] for c in state['available_candidates']])
        plan = validate_action(state, reference, caps, {reference['split_id']: {'id': reference['split_id']}})
        self.assertEqual(plan['candidate'], {'id': name, 'params': caps[name]['params']})
        state['prediction_evidence'][0]['candidate_id'] = 'another'
        self.assertIn('conflicting candidate', next_action(state, capabilities=caps)['reason'])

    def test_conflicting_metric_identity_or_parameters_do_not_authorize_another_fit(self):
        state = competing()
        caps = capabilities_for(state['task'], train_rows=256)
        state['history'][0]['metric_spec'] = dict(state['task']['metric'], labels=[1, 0])
        self.assertIn('metric identity', next_action(state, capabilities=caps)['reason'])
        with self.assertRaisesRegex(ValueError, 'metric identity'):
            selection_evidence(state)
        state = competing()
        state['history'][0]['params'] = {'C': 99}
        self.assertEqual(next_action(state, capabilities=caps)['action'], 'stop')
        state = recorded()
        del caps[state['available_candidates'][0]['id']]
        self.assertEqual(next_action(state, capabilities=caps)['action'], 'stop')

    def test_distinct_refit_and_trial_costs_match_executor_admission(self):
        state = recorded()
        state.update(history=[], prediction_evidence=[])
        name = state['available_candidates'][0]['id']
        state['available_candidates'] = [state['available_candidates'][0]]
        caps = capabilities_for(state['task'], train_rows=256)
        caps[name].update(classifier_fits=2, refit_classifier_fits=1)
        state['budget'].update(remaining_fit_allowance=3, attempted_classifier_fits=8, completed_classifier_fits=8)
        action = next_action(state, capabilities=caps)
        self.assertEqual((action['action'], action['estimated_classifier_fits']), ('explore', 2))
        plan = validate_action(state, action, caps, {'primary': {'id': 'primary'}})
        self.assertEqual(plan['expected_classifier_fits'], 2)
        caps[name].update(classifier_fits=1, refit_classifier_fits=2)
        self.assertEqual(next_action(state, capabilities=caps)['action'], 'stop')
        state['budget']['reserved_final_refit'] = 2
        action = next_action(state, capabilities=caps)
        self.assertEqual(action['action'], 'explore')
        self.assertEqual(validate_action(state, action, caps, {'primary': {}})['expected_classifier_fits'], 1)

    def test_local_and_optional_picker_share_eligible_ids_without_an_api(self):
        state = recorded()
        caps = capabilities_for(state['task'], train_rows=256)
        state['available_candidates'] = untried_candidates(caps, state['history'])
        local = next_action(state, capabilities=caps)
        seen = []

        def choose(ids):
            seen.extend(ids)
            return local['candidate_id']

        other = next_action(state, capabilities=caps, on_explore=choose)
        self.assertEqual(other['candidate_id'], local['candidate_id'])
        self.assertEqual(set(seen), {c['id'] for c in state['available_candidates']})

    def test_fixture_rejects_a_physical_fit_before_calling_classifier(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            payload, output, trace = (directory / name for name in ('payload.joblib', 'output.joblib', 'trace.jsonl'))
            x = np.arange(32).reshape(8, 4)
            y = np.arange(8) % 2
            joblib.dump(dict(kind='numeric_tabular', X_train=x, y_train=y, X_search=x, y_search=y), payload)
            env = dict(os.environ, AB_B1_OFFLINE='1', AB_B1_FIT_LOG=str(trace), AB_B1_PER_WORKER_FIT_LIMIT='0',
                       PYTHONPATH=os.pathsep.join([str(root / 'scripts/ab_acceptance_guard'), str(root)]))
            worker = subprocess.run([sys.executable, '-m', 'superagent.experiment_worker', str(payload),
                                     'numeric_tabular_logistic_C1.0', str(output)], cwd=root, env=env,
                                    timeout=45, capture_output=True, text=True)
            self.assertNotEqual(worker.returncode, 0)
            self.assertIn('Supervised fit exceeds this worker admission', worker.stderr)
            self.assertFalse(trace.exists())
            self.assertFalse(output.exists())
