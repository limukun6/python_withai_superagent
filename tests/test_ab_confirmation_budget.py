import copy
import unittest

from superagent.ab_scheduler import CONFIRMATIONS, ROW_SCOPE, _promotion_block, next_action
from superagent.prediction_evidence import fingerprint
from tests.test_ab_scheduler import FIELDS_OUT, recorded


# The two leading candidates disagree on which rows are correct. Each metric is
# calculated from the fixture labels so replay can also check evidence integrity.
LEADER = 'numeric_tabular_extra_trees_leaf1'
RIVAL = 'numeric_tabular_logistic_C0.1'
THIRD = 'numeric_tabular_logistic_C1.0'
PRIMARY = 'primary'


def prefix(primary_rival_errors=2, confirm_rival_errors=3, third_errors=12):
    n = 36
    labels = [i % 2 for i in range(n)]
    state = recorded()
    state['history'] = []
    state['prediction_evidence'] = []
    state['registered_confirmation_splits'] = list(CONFIRMATIONS)
    state['budget']['llm_request_limit'] = 0
    partitions = {}
    def add(name, split, errors):
        split_number = (PRIMARY,*CONFIRMATIONS).index(split)
        ids = list(range(n*split_number,n*(split_number+1)))
        payload = fingerprint(ids)
        predictions = labels.copy()
        for i in errors:
            predictions[i] = 1-labels[i]
        score = sum(a == b for a,b in zip(predictions,labels))/n
        trial = len(state['history'])+1
        state['history'].append(dict(trial=trial,candidate=name,split_id=split,primary_metric='accuracy',
                                     status='completed',search_score=score,classifier_fits=1,params={}))
        state['prediction_evidence'].append(dict(trial=trial,candidate_id=name,split_id=split,
            row_id_scope=ROW_SCOPE,row_ids=ids,predictions=predictions,evaluation_payload_sha256=payload,
            label_sha256=fingerprint(labels),prediction_sha256=fingerprint(predictions)))
        partitions[split] = dict(row_ids=ids, labels=labels, evaluation_payload_sha256=payload)
    add(LEADER,PRIMARY,[0,1])
    add(RIVAL,PRIMARY,list(range(5,5+primary_rival_errors)))
    add(THIRD,PRIMARY,list(range(12,12+third_errors)))
    add(LEADER,CONFIRMATIONS[0],[0,1])
    add(RIVAL,CONFIRMATIONS[0],list(range(6,6+confirm_rival_errors)))
    state['budget'].update(attempted_classifier_fits=5,completed_classifier_fits=5,remaining_fit_allowance=6)
    return state, partitions, add


class TestConfirmationBudget(unittest.TestCase):
    def test_tie_then_loss_frees_search_fit_and_cites_earlier_trials(self):
        state, trusted, _ = prefix()
        original = copy.deepcopy(state)
        decision = next_action(state, trusted_partitions=trusted)
        self.assertEqual(state,original)
        self.assertEqual(set(decision),FIELDS_OUT)
        self.assertEqual(decision['action'],'explore')
        self.assertIn(decision['candidate_id'],[e['id'] for e in state['available_candidates']])
        self.assertIn('At most 1',decision['reason'])
        self.assertTrue({1,2,4,5} <= set(decision['evidence_trials']))
        self.assertNotIn(6,decision['evidence_trials'])

    def test_original_loss_then_win_keeps_the_deciding_partition(self):
        state, trusted, _ = prefix(primary_rival_errors=3,confirm_rival_errors=1)
        decision = next_action(state,trusted_partitions=trusted)
        self.assertEqual((decision['action'],decision['candidate_id'],decision['split_id']),
                         ('confirm',LEADER,'confirmation_2'))

    def test_unbeatable_half_finished_pair_does_not_consume_the_last_search_fit(self):
        state, trusted, add = prefix(primary_rival_errors=3,confirm_rival_errors=1)
        add(LEADER,CONFIRMATIONS[1],[])
        state['budget'].update(attempted_classifier_fits=6,completed_classifier_fits=6,remaining_fit_allowance=5)
        state['available_candidates'] = []
        decision = next_action(state,trusted_partitions=trusted)
        self.assertEqual(decision['action'],'stop')
        self.assertIn('At most 1',decision['reason'])
        self.assertIn(6,decision['evidence_trials'])

    def test_half_finished_pair_can_still_reverse_the_lead(self):
        state, trusted, add = prefix(primary_rival_errors=3,confirm_rival_errors=1)
        add(LEADER,CONFIRMATIONS[1],[0])
        state['budget'].update(attempted_classifier_fits=6,completed_classifier_fits=6,remaining_fit_allowance=5)
        decision = next_action(state,trusted_partitions=trusted)
        self.assertEqual((decision['action'],decision['candidate_id'],decision['split_id']),
                         ('confirm',RIVAL,'confirmation_2'))

    def test_only_registered_partitions_count(self):
        state, trusted, _ = prefix()
        state['history'] = state['history'][:3]
        state['prediction_evidence'] = state['prediction_evidence'][:3]
        state['registered_confirmation_splits'] = ['confirmation_1']
        state['budget'].update(attempted_classifier_fits=3,completed_classifier_fits=3,remaining_fit_allowance=8)
        decision=next_action(state,trusted_partitions=trusted)
        self.assertEqual(decision['action'],'explore')
        self.assertIn('At most 1',decision['reason'])

    def test_another_candidate_can_use_a_confirmation_after_a_block(self):
        state, trusted, _ = prefix(third_errors=2)
        decision = next_action(state,trusted_partitions=trusted)
        self.assertEqual((decision['action'],decision['candidate_id'],decision['split_id']),
                         ('confirm',THIRD,'confirmation_1'))
        self.assertIn('At most 1',decision['reason'])

    def test_two_small_wins_do_not_rule_out_reaching_the_mean_gain_threshold(self):
        state, _, _ = prefix()
        for row,score in zip(state['history'],[.800,.801,.70,.800,.8015],strict=True):
            row.update(primary_metric='roc_auc',search_score=score)
        attempted = {(r['candidate'],r['split_id']):r for r in state['history']}
        records = {r['trial']:r['predictions'] for r in state['prediction_evidence']}
        self.assertIsNone(_promotion_block(LEADER,RIVAL,CONFIRMATIONS,attempted,records))
