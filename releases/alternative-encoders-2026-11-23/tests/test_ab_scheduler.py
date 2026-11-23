import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from superagent.ab_catalog import common_catalog
from superagent.ab_methods import FIELDS, IMAGE, TEXT
from superagent.ab_scheduler import next_action
from superagent.prediction_evidence import fingerprint

EXAMPLE = Path(__file__).resolve().parents[1]/'examples'/'ab_v1'
FIELDS_OUT = set(json.loads((EXAMPLE/'action_confirm.json').read_text()))


def recorded():
    return json.loads((EXAMPLE/'input.json').read_text())


def competing():
    state = recorded()
    # Controlled counterfactual: exchange opposite predictions while preserving counts.
    item = state['prediction_evidence'][0]
    pred = item['predictions']
    a,b = pred.index(0),pred.index(1)
    pred[a], pred[b] = pred[b],pred[a]
    item['prediction_sha256'] = fingerprint(pred)
    return state


class TestActionAdapter(unittest.TestCase):
    def test_published_example_and_one_changed_prediction_pattern(self):
        state = recorded()
        self.assertEqual(next_action(state)['action'],'explore')
        changed = competing()
        action = next_action(changed)
        self.assertEqual(set(action),FIELDS_OUT)
        self.assertEqual((action['action'],action['split_id'],action['estimated_classifier_fits']),
                         ('confirm','confirmation_1',1))
        self.assertEqual(set(action['evidence_trials']),{1,2})
        self.assertEqual(changed['history'],state['history'])

    def test_no_paid_callback_on_anchor_confirmation_stop_or_at_cap(self):
        selector = Mock(side_effect=AssertionError('unexpected planner request'))
        state = competing()
        next_action(state,on_explore=selector)
        state['history'],state['prediction_evidence'] = [],[]
        anchor = next_action(state,on_explore=selector)
        self.assertEqual(anchor['candidate_id'],state['available_candidates'][0]['id'])
        state = recorded()
        state['budget']['used_llm_requests'] = 3
        self.assertEqual(next_action(state,on_explore=selector)['action'],'explore')
        state['budget'].update(remaining_fit_allowance=1,attempted_classifier_fits=10)
        action = next_action(state,on_explore=selector)
        self.assertEqual((action['action'],action['estimated_classifier_fits']),('stop',0))
        selector.assert_not_called()

    def test_only_eligible_ids_to_picker_and_no_unadvertised_id(self):
        state = recorded()
        chosen = state['available_candidates'][-1]['id']
        def choose(ids):
            self.assertEqual(set(ids),{e['id'] for e in state['available_candidates']})
            self.assertTrue(all(isinstance(i,str) for i in ids))
            return chosen
        self.assertEqual(next_action(state,on_explore=choose)['candidate_id'],chosen)
        with self.assertRaises(ValueError):
            next_action(state,on_explore=lambda ids:'ineligible')

    def test_unknown_usage_broken_ledger_deadline_or_finished_refit(self):
        for budget in (dict(remaining_fit_allowance=None),dict(unknown_failed_attempts=1),
                       dict(fit_limit=12),dict(remaining_fit_allowance=7),
                       dict(reserved_final_refit=0),dict(used_llm_requests=4),
                       dict(time_limit_seconds=1)):
            with self.subTest(budget=budget):
                state = competing()
                state['budget'].update(budget)
                self.assertEqual(next_action(state)['action'],'stop')

    def test_drift_in_row_order_and_fingerprints_stops_spending(self):
        for key in ('row_ids','evaluation_payload_sha256','label_sha256','prediction_sha256','row_id_scope'):
            state = competing()
            item = state['prediction_evidence'][1]
            item[key] = list(reversed(item[key])) if key == 'row_ids' else '0'*64
            self.assertEqual(next_action(state)['action'],'stop',key)
        state = competing()
        p = state['prediction_evidence'][0]
        trusted = {'primary':dict(row_ids=p['row_ids'],labels=p['predictions'],
                                  evaluation_payload_sha256=p['evaluation_payload_sha256'])}
        self.assertIn('trusted local',next_action(state,trusted_partitions=trusted)['reason'])

    def test_both_sides_and_refit_must_fit_no_retry_on_failed_confirmation(self):
        state = competing()
        state['budget'].update(remaining_fit_allowance=2,attempted_classifier_fits=9)
        self.assertEqual(next_action(state)['action'],'stop')
        state = competing()
        action = next_action(state)
        state['history'].append(dict(trial=4,candidate=action['candidate_id'],split_id='confirmation_1',
                                     status='failed',classifier_fits=1))
        state['budget'].update(remaining_fit_allowance=7,attempted_classifier_fits=4)
        self.assertEqual(next_action(state)['action'],'explore')

    def test_failed_primary_and_expensive_unknown_actions(self):
        state = recorded()
        state['history'].append(dict(trial=4,candidate=state['available_candidates'][0]['id'],status='failed'))
        state['available_candidates'] = [state['available_candidates'][0],{'id':'unnamed','params':{}},
                                        {'id':'stack','params':{'maximum_classifier_fits_per_evaluation':5}}]
        self.assertEqual(next_action(state)['action'],'stop')
        state['available_candidates'][-1]['params']['maximum_classifier_fits_per_evaluation'] = 2
        self.assertEqual(next_action(state)['estimated_classifier_fits'],2)

    def test_primary_metric_and_auc_are_not_replaced_by_accuracy(self):
        state = competing()
        state['task']['metric']['name'] = 'macro_f1'
        self.assertEqual(next_action(state)['action'],'stop')
        for i,r in enumerate(state['history']):
            r.update(primary_metric='macro_f1', search_score=[.4,.89,.90][i])
        self.assertEqual(next_action(state)['action'],'explore')
        state = recorded()
        state['task']['metric']['name'] = 'roc_auc'
        for r in state['history']:
            r.update(primary_metric='roc_auc',search_score=.91)
        # Scalar AUC plus hard labels cannot certify a score-based comparison.
        self.assertEqual(next_action(state)['action'],'stop')

    def test_new_confirmation_cannot_reuse_primary_rows(self):
        state = competing()
        r = copy.deepcopy(state['history'][0]); r.update(trial=4,split_id='confirmation_1')
        p = copy.deepcopy(state['prediction_evidence'][0]);p.update(trial=4,split_id='confirmation_1')
        state['history'].append(r);state['prediction_evidence'].append(p)
        self.assertIn('reuses',next_action(state)['reason'])

    def test_registered_splits_and_task_identity(self):
        state = competing()
        state['registered_confirmation_splits'] = []
        self.assertEqual(next_action(state)['action'],'explore')
        answer = next_action(state)
        state['task'].update(name='team-task-2',filename='released_abc')
        self.assertEqual(next_action(state),answer)


class TestCommonMenu(unittest.TestCase):
    def test_score_capability_size_and_early_methods(self):
        task = recorded()['task']
        initial = common_catalog(task,train_rows=250)
        self.assertTrue(any(c['estimator']=='rbf_svm' for c in initial))
        self.assertTrue(all(c['expected_classifier_fits']==1 for c in initial))
        self.assertFalse(any(c['estimator']=='rbf_svm' for c in common_catalog(task,train_rows=99999)))
        task['metric']['name'] = 'roc_auc'
        self.assertTrue(all(c['supports_probabilities'] for c in common_catalog(task,train_rows=250)))
        task['kind'] = 'image'
        self.assertIn(IMAGE,{c['id'] for c in common_catalog(task,train_rows=250)})
        task['kind'] = 'categorical_tabular'
        from importlib.util import find_spec
        self.assertEqual(FIELDS in {c['id'] for c in common_catalog(task,train_rows=250)},find_spec('catboost') is not None)

    def test_pair_semantics_and_missing_encoder(self):
        task = recorded()['task']
        task['kind'] = 'pair_text'
        for pair in ('relation',None):
            task['input_schema'] = {'pair_semantics':pair}
            self.assertEqual(common_catalog(task,train_rows=50),[])
        task['input_schema'] = {'pair_semantics':'symmetric'}
        with patch.dict('os.environ',{'SUPERAGENT_MINILM_WEIGHTS':''}):
            symmetric = common_catalog(task,train_rows=50)
        self.assertEqual(len(symmetric),7)
        self.assertTrue(all(r['swap_label_rule']=='preserves' for r in symmetric))
        task['input_schema']['pair_semantics'] = 'preference'
        with patch.dict('os.environ',{'SUPERAGENT_MINILM_WEIGHTS':''}):
            menu = common_catalog(task,train_rows=50)
        self.assertTrue(menu)
        self.assertTrue(all(r['swap_label_rule']=='flips' for r in menu))
        self.assertNotIn(TEXT,{r['id'] for r in menu})
