"""Real review-length failures and bounded nontruncating acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from superagent.contracts import DecisionState
from superagent.decision_architecture import ArchitecturePolicy
from superagent.events import RunLogger
from test_decision_architecture import ScriptClient
from test_proposal_consistency import wire

FIXTURE=Path(__file__).parent/'fixtures/v35_long_reviews.json'


class ReviewLengthTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.logger=RunLogger(Path(self.temp.name));self.logger.initialize_usage_ledger()
        self.cases=json.loads(FIXTURE.read_text())['cases']

    def run_case(self,case,answer):
        client=ScriptClient([wire(case['original_proposal']['raw_proposal']),answer])
        policy=ArchitecturePolicy(client,self.logger,case['architecture'])
        chosen=policy.propose(DecisionState(case['state']))
        self.assertEqual(len(client.calls),2)
        self.assertEqual(chosen.candidate,case['original_proposal']['candidate'])
        return policy,client

    def test_two_real_failed_responses_replay_without_truncation_or_extra_calls(self):
        self.assertEqual([len(c['raw_review']['reason']) for c in self.cases],[537,626])
        for case in self.cases:
            before=deepcopy(case);policy,client=self.run_case(case,case['raw_review'])
            self.assertTrue(policy.summary()['all_requested_decisions_valid'])
            self.assertEqual(policy.records[0]['status'],'review_keep')
            self.assertEqual(policy.records[0]['review']['reason'],case['raw_review']['reason'])
            self.assertEqual(policy.summary()['review_reason_length_advisories'],1)
            self.assertFalse(policy.records[0]['review_reason_length']['text_truncated'])
            self.assertEqual(case,before)

    def test_preferred_and_hard_length_boundaries(self):
        case=self.cases[0]
        for size in (500,501,1200,1201):
            response=deepcopy(case['raw_review']);response['reason']='x'*size
            with self.subTest(characters=size):
                policy,client=self.run_case(case,response)
                self.assertEqual(policy.summary()['all_requested_decisions_valid'],size<=1200)
                if size<=1200:
                    self.assertEqual(len(policy.records[0]['review']['reason']),size)
                    self.assertEqual(policy.summary()['review_reason_length_advisories'],int(size>500))
                else:
                    self.assertEqual(policy.summary()['review_fallback_count'],1)
                    with self.assertRaises(RuntimeError):policy.propose(DecisionState(case['state']))
                    self.assertEqual(len(client.calls),2)

    def test_empty_nontext_and_fabricated_reference_remain_failures(self):
        case=self.cases[0]
        for field,value in [('reason',''),('reason',None),('reason',[]),('fact_ids',['t999.search_score'])]:
            response=deepcopy(case['raw_review']);response[field]=value
            with self.subTest(field=field,value=value):
                policy,client=self.run_case(case,response)
                self.assertFalse(policy.summary()['all_requested_decisions_valid'])
                self.assertEqual(policy.summary()['review_fallback_count'],1)
                self.assertEqual(policy.records[0]['chosen_proposal'],policy.records[0]['original_proposal'])

if __name__=='__main__':unittest.main()
