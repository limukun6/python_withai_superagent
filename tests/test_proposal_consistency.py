"""Check real regressions and conservative exclusions without a paid judge."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from superagent.contracts import DecisionState
from superagent.decision_architecture import ArchitecturePolicy
from superagent.explanation_audit import audit_explanation,proposed_review_context
from test_decision_architecture import ScriptClient,review
from superagent.events import RunLogger

FIXTURE=Path(__file__).parent/'fixtures/v33_review_directions.json'


def cases():return json.loads(FIXTURE.read_text())['cases']
def bad():return deepcopy(next(c for c in cases() if c['id']=='reviewer_trial3'))
def wire(proposal):
    return {k:deepcopy(proposal[k]) for k in ('candidate','reason','hypothesis','review','evidence_trials',
                                             'expected_change','assessment','route_plan')} | {
        'assessment':{k:deepcopy(proposal['assessment'][k]) for k in ('fact_ids','interpretation','uncertainty')}}
def inspect(case):return audit_explanation(case['proposal'],case['state']['history'],case['candidate'])


class DirectionDiagnosticTests(unittest.TestCase):
    def test_six_real_decisions_identify_two_unresolved_direction_rationales(self):
        data=cases();original=deepcopy(data)
        for case in data:
            result=inspect(case)
            warnings=[w for w in result['warnings'] if w['code']=='possible_regularization_direction_rationale_mismatch']
            self.assertEqual(bool(warnings),case['expected_direction_warning'],case['id'])
            self.assertFalse(result['changes_candidate']);self.assertFalse(result['triggers_api_retry'])
            self.assertFalse(result['freeform_prose_fully_verified'])
            if warnings:
                self.assertEqual(warnings[0]['expected_direction'],'stronger')
                self.assertEqual(warnings[0]['field'],'assessment.interpretation')
        self.assertEqual(data,original)

    def test_consistent_and_explicit_alternative_prose_is_not_forced_to_change(self):
        samples=[
            'Stronger regularization may help this configuration.',
            'Compare stronger regularization versus weaker regularization; neither is established.',
            'Weaker regularization is an alternative hypothesis; this trial checks stronger regularization.',
            'We are not claiming weaker regularization will help.',
            'Weaker regularization may not improve the score.',
            'The previous configuration used weaker regularization.',
        ]
        for text in samples:
            case=bad();case['proposal']['assessment']['interpretation']=text
            with self.subTest(text=text):self.assertFalse(inspect(case)['direction_consistency']['warnings'])

    def test_increasing_c_has_opposite_direction_and_spelling_variants(self):
        case=bad();case['candidate']={'id':'word_char_C1.0','params':{**case['candidate']['params'],'C':1.}}
        case['proposal'].update(candidate=case['candidate']['id'],reason='More regularisation may improve accuracy.',hypothesis='Test score.')
        case['proposal']['assessment']['interpretation']='Less regularization may help.'
        result=inspect(case)['direction_consistency']
        self.assertEqual(result['expected_regularization_direction'],'weaker')
        self.assertEqual([w['field'] for w in result['warnings']],['reason'])

    def test_cross_route_multi_parameter_and_unchanged_c_are_not_equated(self):
        params=bad()['candidate']['params']
        for candidate in (
            {'id':'char_C0.1','params':{**params,'C':.1,'representation':'char'}},
            {'id':'word_char_C0.1','params':{**params,'C':.1,'solver':'changed'}},
            {'id':'word_char_C0.3','params':{**params,'C':.3}},
        ):
            case=bad();case['candidate']=candidate;case['proposal']['candidate']=candidate['id']
            with self.subTest(candidate=candidate):
                result=inspect(case)['direction_consistency']
                self.assertFalse(result['applicable']);self.assertFalse(result['warnings'])

    def test_past_review_and_uncertainty_are_not_current_direction_assertions(self):
        case=bad();case['proposal']['assessment']['interpretation']='Stronger regularization is the proposed experiment.'
        case['proposal']['review']='Weaker regularization performed well earlier.'
        case['proposal']['assessment']['uncertainty']='Weaker regularization may also be useful.'
        self.assertFalse(inspect(case)['direction_consistency']['warnings'])

    def test_missing_reference_and_unrecognized_language_are_not_certified(self):
        case=bad();proposal=case['proposal'];proposal['expected_change']=None
        result=audit_explanation({**proposal,'assessment':{'fact_ids':[]}},[],case['candidate'])
        self.assertFalse(result['direction_consistency']['applicable'])
        self.assertFalse(result['freeform_prose_fully_verified'])
        case['proposal'].update(reason='测试参数',hypothesis='候选假设')
        case['proposal']['assessment']['interpretation']='不在英文短语规则覆盖范围内。'
        result=inspect(case)['direction_consistency']
        self.assertTrue(result['applicable']);self.assertFalse(result['warnings'])
        self.assertFalse(result['freeform_semantics_verified'])


class PreReviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.logger=RunLogger(self.root);self.logger.initialize_usage_ledger()

    def test_both_review_roles_see_same_proposed_facts_and_warn_before_second_call(self):
        case=bad();first=[];diagnostics=[]
        for mode in ('self_review','reviewer'):
            client=ScriptClient([wire(case['proposal']),{'action':'keep','reason':'Treat the opposing rationale as a possible alternative.',
                'fact_ids':['t2.search_score'],'replacement':None}])
            policy=ArchitecturePolicy(client,self.logger,mode);chosen=policy.propose(DecisionState(case['state']))
            first.append(client.calls[0]);diagnostics.append(client.calls[1][0]['proposed_diagnostics'])
            self.assertEqual(chosen.candidate,case['candidate']['id']);self.assertEqual(len(client.calls),2)
            self.assertTrue(policy.summary()['all_requested_decisions_valid'])
            self.assertEqual(policy.summary()['chosen_proposal_warning_count'],1)
            self.assertEqual(diagnostics[-1]['configuration_change']['reference_trial'],2)
            self.assertFalse(diagnostics[-1]['candidate_veto'])
            events=[json.loads(s) for s in self.logger.events_path.read_text().splitlines()]
            original=[e for e in events if e['kind']=='architecture_original_proposal'][-1]
            self.assertEqual(original['proposed_diagnostics'],diagnostics[-1])
        self.assertEqual(first[0],first[1]);self.assertEqual(diagnostics[0],diagnostics[1])

    def test_revision_can_repair_prose_without_changing_candidate_or_fit_opportunity(self):
        case=bad();replacement=wire(case['proposal'])
        replacement['assessment']['interpretation']='Stronger regularization is the direction being tested; its benefit is unknown.'
        reply={'action':'revise','reason':'Repair the opposing rationale while retaining the valid candidate.',
               'fact_ids':['t2.search_score'],'replacement':replacement}
        client=ScriptClient([wire(case['proposal']),reply]);policy=ArchitecturePolicy(client,self.logger,'reviewer')
        chosen=policy.propose(DecisionState(case['state']));entry=policy.records[0]
        self.assertFalse(entry['candidate_changed']);self.assertTrue(entry['proposal_changed'])
        self.assertEqual(entry['status'],'review_revise');self.assertEqual(len(client.calls),2)
        self.assertEqual(chosen.trial,3);self.assertEqual(chosen.params,case['candidate']['params'])
        self.assertEqual(len(entry['original_diagnostics']['warnings']),1)
        self.assertEqual(entry['chosen_diagnostics']['warnings'],[])
        self.assertIn('weaker regularization',entry['original_proposal']['raw_proposal']['assessment']['interpretation'])

    def test_single_logs_same_diagnostic_without_extra_call_or_candidate_veto(self):
        case=bad();client=ScriptClient([wire(case['proposal'])]);policy=ArchitecturePolicy(client,self.logger,'single')
        chosen=policy.propose(DecisionState(case['state']))
        self.assertEqual(len(client.calls),1);self.assertEqual(chosen.candidate,case['candidate']['id'])
        self.assertEqual(policy.summary()['original_proposal_warning_count'],1)
        self.assertTrue(policy.summary()['all_requested_decisions_valid'])


if __name__=='__main__':unittest.main()
