from copy import deepcopy
import json
from pathlib import Path
import unittest

from superagent.explanation_audit import audit_explanation,comparison_report,visible_warning_context
from superagent.evidence_agent import state_for


class ExplanationAuditTests(unittest.TestCase):
    def fixture(self):return json.loads((Path(__file__).parent/'fixtures/v09_decision_prose.json').read_text())['cases']

    def test_real_prose_flags_keep_original_and_use_decision_time_evidence(self):
        cases=self.fixture();before=deepcopy(cases)
        results=[audit_explanation(c['proposal'],c['history_before_decision']) for c in cases]
        self.assertEqual([len(a['warnings']) for a in results],[0,0,2,0])
        issue=results[2]
        self.assertEqual({w['code'] for w in issue['warnings']},{'possible_exact_tie_contradiction','unsupported_significance_wording'})
        self.assertEqual(issue['executor_comparison']['latest_trial'],3)
        self.assertEqual(issue['executor_comparison']['best_prior_trial'],2)
        self.assertAlmostEqual(issue['executor_comparison']['delta_vs_prior_best'],-.004265691651432)
        self.assertFalse(issue['freeform_prose_fully_verified'])
        self.assertFalse(issue['changes_candidate']);self.assertFalse(issue['triggers_api_retry'])
        self.assertEqual(cases,before)

    def test_real_exact_tie_and_uncertainty_are_not_definite_contradictions(self):
        case=deepcopy(self.fixture()[2]);history=case['history_before_decision']
        history[-1]['search_accuracy']=history[-2]['search_accuracy']
        proposal=case['proposal'];proposal['assessment']['interpretation']='We cannot determine significance from these trials.'
        result=audit_explanation(proposal,history)
        self.assertEqual(result['executor_comparison']['reading'],'unchanged')
        self.assertFalse(result['warnings'])
        proposal['reason']='The result nearly tied the earlier score; the next experiment may improve it.'
        history[-1]['search_accuracy']-=.001
        self.assertFalse(audit_explanation(proposal,history)['warnings'])

    def test_no_result_and_timeout_are_never_reported_as_zero_accuracy(self):
        empty=comparison_report([]);self.assertIsNone(empty['latest_search_accuracy'])
        failed=comparison_report([{'trial':1,'candidate':'word_C0.05','status':'timeout'}])
        self.assertEqual(failed['reading'],'execution_failure')
        self.assertIsNone(failed['latest_search_accuracy'])
        self.assertNotIn('0.000000%',failed['text'])

    def test_unknown_fact_is_rejected_instead_of_becoming_evidence(self):
        case=deepcopy(self.fixture()[0]);case['proposal']['assessment']['fact_ids']=['t999.search_accuracy']
        with self.assertRaises(ValueError):audit_explanation(case['proposal'],case['history_before_decision'])

    def test_warning_context_has_codes_and_executor_facts_but_no_flagged_prose(self):
        case=self.fixture()[2];audit=audit_explanation(case['proposal'],case['history_before_decision'])
        context=visible_warning_context(audit)
        self.assertNotIn('only tied',json.dumps(context))
        self.assertNotIn('original_prose',context)
        self.assertEqual(context['executor_comparison']['reading'],'lower')

    def test_outcome_derived_review_is_hidden_from_blind_control(self):
        case=deepcopy(self.fixture()[2]);history=case['history_before_decision']
        history[-1]['explanation_audit']=audit_explanation(case['proposal'],history)
        available=[{'id':'word_C10.0','params':{'C':10.}}]
        blind=state_for({},history,available,5,'evidence_blind')
        visible=state_for({},history,available,5,'budget_routes')
        self.assertNotIn('previous_explanation_review',blind)
        self.assertEqual(blind['facts'],[])
        self.assertIn('previous_explanation_review',visible)


if __name__=='__main__':unittest.main()
