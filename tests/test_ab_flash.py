"""No HTTP or training: split binding, equal contexts and historical accounting gates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from superagent.ab_catalog import capabilities_for,untried_candidates
from superagent.ab_flash import context_for,validate_proposal,preflight
from superagent.cli import build_parser
from superagent.task_budget import TaskBudgetLedger
from tests.test_ab_scheduler import recorded


def fixture_state():
    state=recorded()
    caps=capabilities_for(state['task'],train_rows=256)
    state['available_candidates']=untried_candidates(caps,state['history'])
    for row in state['history']:
        row['split_id']='primary'
        row['training_seconds']=12345.678
        row['metric_identity']={'status':'confirmed'}
    ids=tuple(c['id'] for c in state['available_candidates'])
    return state,caps,ids


def answer_for(state,caps,ids):
    return dict(candidate=ids[0],task_kind=state['task']['kind'],
        configuration_claim={'classifier':caps[ids[0]]['estimator']},
        reason='Test another registered configuration.',review='Compare primary observations.',
        hypothesis='Primary score will improve.',evidence_trials=[1],
        expected_change=dict(metric='search_score',reference_trial=1,direction='increase',min_delta=.001),
        slice_expectation=None)


class FlashBoundaryTests(unittest.TestCase):
    def test_contexts_have_identical_common_fields_and_no_vectors_or_time(self):
        state,caps,ids=fixture_state()
        raw=context_for(state,caps,ids,False)
        checked=context_for(state,caps,ids,True)
        self.assertEqual(raw,{k:v for k,v in checked.items() if k not in ('checks','memory')})
        wire=json.dumps(raw)
        for forbidden in ('training_seconds','elapsed_seconds','prediction_evidence','row_ids','labels_vector','X_train'):
            self.assertNotIn(forbidden,wire)
        self.assertTrue(all('split_id' in row for row in raw['history']))
        self.assertEqual(validate_proposal(answer_for(state,caps,ids),state,caps,ids).candidate,ids[0])

    def test_reject_other_split_and_ineligible_or_modified_schema(self):
        state,caps,ids=fixture_state()
        base=answer_for(state,caps,ids)
        mutations=[dict(candidate='fake'),dict(task_kind='image'),dict(extra='ignored'),
            dict(configuration_claim={'classifier':'imaginary'}),dict(evidence_trials=[999]),
            dict(expected_change=dict(metric='search_score',reference_trial=1,direction='decrease',min_delta=.001)),
            dict(slice_expectation={})]
        for change in mutations:
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_proposal({**base,**change},state,caps,ids)
        state['history'][0]['split_id']='confirmation_1'
        with self.assertRaisesRegex(ValueError,'different split'):
            validate_proposal(base,state,caps,ids)

    def test_new_or_empty_ledgers_refused_before_key_or_data(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}):
            path=Path(temp)/'ledger.sqlite'
            TaskBudgetLedger.initialize(path,new_project=True)
            args=build_parser().parse_args(['run','missing.zip','--scheduler','ab-v1','--portfolio','ab-v1',
                '--planner','checked-history','--api-key-file','missing-key','--budget-ledger',str(path),
                '--task-budget-id','fixture'])
            with self.assertRaisesRegex(ValueError,'nonempty imported history'):
                preflight(args)

    def test_available_subset_is_exact_and_detached(self):
        state,caps,ids=fixture_state()
        projected=context_for(state,caps,ids[:1],False)
        self.assertEqual([c['id'] for c in projected['available_candidates']],list(ids[:1]))
        projected['available_candidates'][0]['params']['modified']=True
        self.assertNotIn('modified',caps[ids[0]]['params'])
        with self.assertRaises(ValueError):context_for(state,caps,('fake',),False)

    def test_unknown_other_scope_cannot_be_bypassed_with_a_new_task_id(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}):
            root=Path(temp);audit=root/'history.json'
            audit.write_text(json.dumps(dict(calls=[dict(request_id='b'*64,scope='old-task',
                cost_status='unknown',course_cost_units=None,course_cost_rmb=None)])),encoding='utf-8')
            ledger=TaskBudgetLedger.initialize(root/'ledger.sqlite',history=audit,
                expected_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
            args=build_parser().parse_args(['run','missing.zip','--scheduler','ab-v1','--portfolio','ab-v1',
                '--planner','raw-history','--api-key-file','missing','--budget-ledger',str(ledger.path),
                '--task-budget-id','new-task'])
            with self.assertRaisesRegex(ValueError,'project usage remains unresolved'):
                preflight(args)


if __name__=='__main__':unittest.main()
