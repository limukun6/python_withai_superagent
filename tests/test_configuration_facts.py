from copy import deepcopy
import unittest

from superagent.configuration_facts import choose_reference,configuration_change,proposal_change,latest_configuration_change
from superagent.evidence_agent import state_for
from superagent.explanation_audit import audit_explanation


def row(trial,candidate,c,score=.6,status='completed'):
    return {'trial':trial,'candidate':candidate,'params':{'C':c},'status':status,'search_accuracy':score if status=='completed' else None}


class ConfigurationFactsTests(unittest.TestCase):
    def test_inverse_c_direction_does_not_predict_score_gain(self):
        old=row(1,'word_char_C0.3',.3)
        result=configuration_change(old,{'id':'word_char_C0.1','params':{'C':.1}},'explicit')
        change=result['changed_parameters'][0]
        self.assertEqual((change['before'],change['after']),(.3,.1))
        self.assertEqual(change['parameter_effect']['declared_effect'],'stronger_regularization')
        self.assertFalse(change['parameter_effect']['accuracy_effect_verified'])
        self.assertTrue(result['single_declared_parameter_change'])
        self.assertFalse(result['single_factor_attribution_supported'])
        reverse=configuration_change(row(1,'word_C0.1',.1),{'id':'word_C0.3','params':{'C':.3}},'explicit')
        self.assertEqual(reverse['changed_parameters'][0]['parameter_effect']['declared_effect'],'weaker_regularization')

    def test_declared_tree_constraints_keep_their_own_meaning(self):
        for field,before,after,effect in [('depth',8,4,'lower_depth_cap'),('l2_leaf_reg',3.,10.,'stronger_l2_penalty'),
            ('min_samples_leaf',1,4,'larger_minimum_leaf')]:
            with self.subTest(field=field):
                old={'trial':1,'candidate':'catboost_depth8_reg3.0','params':{field:before},'status':'completed'}
                new={'id':'catboost_depth4_reg10.0','params':{field:after}}
                result=configuration_change(old,new,'explicit')
                self.assertEqual(result['changed_parameters'][0]['parameter_effect']['declared_effect'],effect)

    def test_route_change_and_missing_fields_are_explicit(self):
        old={'trial':1,'candidate':'word_C0.05','params':{'C':.05,'representation':'word'},'status':'completed'}
        new={'id':'word_char_stats_C0.3','params':{'C':.3,'representation':'word_char_stats','extra_metadata':True}}
        result=configuration_change(old,new,'explicit')
        self.assertFalse(result['same_declared_route']);self.assertFalse(result['single_declared_parameter_change'])
        fields={r['parameter']:r for r in result['changed_parameters']}
        self.assertFalse(fields['extra_metadata']['before_present']);self.assertIsNone(fields['extra_metadata']['before'])
        self.assertIn('scaling',fields['C']['parameter_effect']['cross_route_note'])

    def test_explicit_reference_precedes_same_route_best_then_overall_best(self):
        history=[row(1,'word_C0.05',.05,.9),row(2,'word_char_C0.3',.3,.7),row(3,'word_char_C0.7',.7,.65)]
        candidate={'id':'word_char_C0.1','params':{'C':.1}}
        self.assertEqual(choose_reference(history,candidate,3)[0]['trial'],3)
        self.assertEqual(choose_reference(history,candidate)[0]['trial'],2)
        self.assertEqual(choose_reference(history,{'id':'char_C0.1','params':{'C':.1}})[0]['trial'],1)
        for invalid in (99,True,'2'):
            with self.assertRaises(ValueError):choose_reference(history,candidate,invalid)

    def test_failed_attempt_has_configuration_but_no_invented_measurement(self):
        history=[row(1,'word_C0.05',.05,status='timeout')]
        result=proposal_change({'expected_change':None},history,{'id':'word_C0.3','params':{'C':.3}})
        self.assertTrue(result['available']);self.assertEqual(result['reference_status'],'timeout')
        self.assertEqual(result['reference_rule'],'last_attempt_configuration_only')
        self.assertIsNone(result['selected_status']);self.assertNotIn('accuracy',result)

    def test_latest_configuration_uses_same_reference_as_latest_score_comparison(self):
        history=[row(1,'word_C0.3',.3,.9),row(2,'word_C0.7',.7,.7),row(3,'word_C0.1',.1,.8)]
        result=latest_configuration_change(history)
        self.assertEqual(result['reference_trial'],1)
        self.assertEqual(result['changed_parameters'][0]['before'],.3)

    def test_blind_control_gets_no_result_selected_configuration_reference(self):
        history=[row(1,'word_C0.3',.3,.7),row(2,'word_C0.1',.1,.6)]
        available=[{'id':'word_C0.7','params':{'C':.7}}]
        blind=state_for({},history,available,5,'evidence_blind')
        visible=state_for({},history,available,5,'budget_routes')
        self.assertNotIn('latest_configuration_comparison',blind)
        self.assertEqual(visible['latest_configuration_comparison']['reference_trial'],1)

    def test_original_contradictory_wording_is_retained_not_silently_rewritten(self):
        history=[row(1,'word_C0.05',.05,.62),row(2,'word_char_C0.3',.3,.65)]
        proposal={'candidate':'word_char_C0.1','reason':'Weaker regularization may help. Testing stronger regularization.',
            'review':'Further comparison is needed.','hypothesis':'The gap may shrink.',
            'expected_change':{'reference_trial':2},'assessment':{'fact_ids':[],'interpretation':'A hypothesis.','uncertainty':'Unknown.'}}
        original=deepcopy((proposal,history))
        result=audit_explanation(proposal,history,{'id':'word_char_C0.1','params':{'C':.1}})
        self.assertEqual(result['original_prose']['fields']['reason'],proposal['reason'])
        self.assertEqual(result['configuration_change']['changed_parameters'][0]['parameter_effect']['declared_effect'],'stronger_regularization')
        self.assertFalse(result['freeform_prose_fully_verified']);self.assertFalse(result['triggers_api_retry'])
        self.assertEqual([w['code'] for w in result['warnings']],['possible_regularization_direction_rationale_mismatch'])
        self.assertEqual((proposal,history),original)
        self.assertFalse(result['changes_candidate']);self.assertFalse(result['triggers_api_retry'])

    def test_missing_metadata_is_unknown_and_not_inferred_from_candidate_name(self):
        self.assertFalse(proposal_change({},[],None)['available'])
        self.assertFalse(configuration_change({'trial':1,'candidate':'word_C0.3'},
            {'id':'word_C0.1','params':{'C':.1}},'explicit')['available'])
        self.assertFalse(latest_configuration_change([])['available'])


if __name__=='__main__':unittest.main()
