"""Counterexamples exercise selection, budget and evidence without fitting classifiers."""
from copy import deepcopy
import hashlib
import unittest

from superagent.ab_catalog import capabilities_for, untried_candidates
from superagent.ab_finalists import finalist_action, finalist_review
from superagent.ab_scheduler import PRIMARY, CONFIRMATIONS, ROW_SCOPE
from superagent.cli import build_parser
from superagent.ab_run import validate_options
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.scheduler_bridge import validate_action
from tests.test_ab_scheduler import recorded


class MatchedFinalistTests(unittest.TestCase):
    def fixture(self, count=5):
        state = recorded()
        caps = dict(list(capabilities_for(state['task'], train_rows=256).items())[:count])
        state.update(history=[], prediction_evidence=[], registered_confirmation_splits=list(CONFIRMATIONS))
        state['budget'].update(fit_limit=7, attempted_classifier_fits=0, completed_classifier_fits=0,
            remaining_fit_allowance=7, reserved_final_refit=1, unknown_failed_attempts=0,
            elapsed_seconds=0, time_limit_seconds=300, llm_request_limit=0, used_llm_requests=0)
        state['available_candidates'] = untried_candidates(caps, [])
        trusted = {split:dict(row_ids=list(range(i*20, (i+1)*20)), labels=[0,1]*10,
                   evaluation_payload_sha256=hashlib.sha256(split.encode()).hexdigest())
                   for i,split in enumerate((PRIMARY,*CONFIRMATIONS))}
        return state, caps, trusted

    def observe(self, state, caps, trusted, action, correct):
        split, name = action['split_id'], action['candidate_id']
        labels = trusted[split]['labels']
        pred = [value if i < correct else 1-value for i,value in enumerate(labels)]
        trial = len(state['history']) + 1
        state['history'].append(dict(trial=trial,candidate=name,params=caps[name]['params'],
            split_id=split,status='completed',primary_metric='accuracy',search_score=correct/20,
            search_accuracy=correct/20,classifier_fits=1,metric_spec=state['task']['metric']))
        state['prediction_evidence'].append(dict(trial=trial,candidate_id=name,split_id=split,
            row_id_scope=ROW_SCOPE,row_ids=trusted[split]['row_ids'],predictions=pred,
            prediction_sha256=fingerprint(pred),label_sha256=fingerprint(labels),
            evaluation_payload_sha256=trusted[split]['evaluation_payload_sha256']))
        state['available_candidates'] = untried_candidates(caps,state['history'])
        for key in ('attempted_classifier_fits','completed_classifier_fits'):state['budget'][key]+=1
        state['budget']['remaining_fit_allowance']-=1

    def primaries(self):
        state,caps,trusted=self.fixture()
        for correct in (18,17,16,15):
            action=finalist_action(state,capabilities=caps,trusted_partitions=trusted)
            self.assertEqual(action['action'],'explore')
            self.observe(state,caps,trusted,action,correct)
        return state,caps,trusted

    def test_shared_checks_overturn_initial_winner_and_preserve_refit(self):
        state,caps,trusted=self.primaries()
        old=state['history'][0]['candidate'];challenger=state['history'][1]['candidate']
        for name,correct in ((old,11),(challenger,20)):
            action=finalist_action(state,capabilities=caps,trusted_partitions=trusted)
            self.assertEqual((action['action'],action['candidate_id'],action['split_id']),('confirm',name,CONFIRMATIONS[0]))
            validate_action(state,action,caps,{s:dict(id=s) for s in trusted})
            self.observe(state,caps,trusted,action,correct)
        self.assertEqual(finalist_action(state,capabilities=caps,trusted_partitions=trusted)['action'],'stop')
        review=finalist_review(state,capabilities=caps,trusted_partitions=trusted)
        self.assertEqual(review['incumbent'],challenger)
        self.assertEqual(review['basis'],[PRIMARY,CONFIRMATIONS[0]])
        self.assertEqual(state['budget']['remaining_fit_allowance'],1)

    def test_half_pair_cannot_be_frozen_or_shortlist_replaced(self):
        state,caps,trusted=self.primaries()
        action=finalist_action(state,capabilities=caps,trusted_partitions=trusted)
        self.observe(state,caps,trusted,action,0)
        with self.assertRaisesRegex(ValueError,'Incomplete matched'):
            finalist_review(state,capabilities=caps,trusted_partitions=trusted)
        next_one=finalist_action(state,capabilities=caps,trusted_partitions=trusted)
        self.assertEqual(next_one['candidate_id'],state['history'][1]['candidate'])

    def test_corrupt_configuration_score_row_or_prediction_rejected(self):
        state,caps,trusted=self.primaries()
        for field in ('prediction_sha256','row_ids','label_sha256'):
            bad=deepcopy(state)
            bad['prediction_evidence'][0][field]=[9] if field=='row_ids' else '0'*64
            with self.subTest(field=field),self.assertRaises(ValueError):
                finalist_action(bad,capabilities=caps,trusted_partitions=trusted)
        for field,value in (('search_score',.123),('params',{}),('classifier_fits',2),('status','failed')):
            bad=deepcopy(state);bad['history'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                finalist_action(bad,capabilities=caps,trusted_partitions=trusted)

    def test_unknown_usage_does_not_request_more_work(self):
        state,caps,trusted=self.primaries();state['budget']['unknown_failed_attempts']=1
        self.assertEqual(finalist_action(state,capabilities=caps,trusted_partitions=trusted)['action'],'stop')

    def test_missing_prediction_record_cannot_enter_shortlist(self):
        state,caps,trusted=self.primaries()
        state['prediction_evidence'].pop()
        with self.assertRaisesRegex(ValueError,'bound prediction evidence'):
            finalist_action(state,capabilities=caps,trusted_partitions=trusted)

    def test_single_candidate_is_still_confirmed(self):
        state,caps,trusted=self.fixture(1)
        for split in (PRIMARY,CONFIRMATIONS[0]):
            action=finalist_action(state,capabilities=caps,trusted_partitions=trusted)
            self.assertEqual(action['split_id'],split);self.observe(state,caps,trusted,action,16)
        self.assertEqual(len(finalist_review(state,capabilities=caps,trusted_partitions=trusted)['finalists']),1)

    def test_registered_cost_and_cli_limit(self):
        state,caps,trusted=self.fixture();caps[next(iter(caps))]['classifier_fits']=2
        with self.assertRaisesRegex(ValueError,'single-fit'):
            finalist_action(state,capabilities=caps,trusted_partitions=trusted)
        parser=build_parser()
        args=parser.parse_args(['run','fixture.zip','--scheduler','ab-finalists','--portfolio','ab-v1','--fit-limit','7'])
        self.assertEqual(validate_options(args),7)
        args.fit_limit=4
        with self.assertRaises(ValueError):validate_options(args)


if __name__=='__main__':unittest.main()
