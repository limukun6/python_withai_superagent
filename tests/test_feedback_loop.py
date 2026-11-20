"""Check that executor facts reach the next decision without contaminating the raw arm."""
from copy import deepcopy
import unittest
from superagent.contracts import DecisionState,TaskSpec
from superagent.metrics import MetricSpec
from superagent.checked_history_policy import decision_context
from superagent.explanation_audit import audit_explanation


class FeedbackLoopTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(trial=1,candidate='numeric_tabular_logistic_C0.1',params={'C':.1},status='completed',
            primary_metric='accuracy',train_score=.9,search_score=.6,train_accuracy=.9,search_accuracy=.6,
            estimator={'family':'logistic','class':'LogisticRegression','source':'fitted_estimator'},
            configuration_fact_check={'status':'contradicted','declared':'extra_trees','observed':'logistic'},
            hypothesis_check={'status':'not_supported'},proposal={'hypothesis':'a future idea','evidence_trials':[]},
            explanation_audit={'warnings':[{'code':'prose_claim_contradiction','field':'hypothesis'}]})
        self.available=[{'id':'numeric_tabular_logistic_C1.0','params':{'C':1.}}]
        self.task=TaskSpec('numeric_tabular',MetricSpec('accuracy',(0,1)),{})

    def state(self,rows=None,visible=True):
        return DecisionState.build({'primary_metric':'accuracy'},rows or [self.row],self.available,4,visible,self.task)

    def test_fitted_observation_survives_state_and_is_shared(self):
        state=self.state()
        for checked in (False,True):
            ctx=decision_context(state,checked)
            self.assertEqual(ctx['history'][0]['estimator']['family'],'logistic')

    def test_derived_checks_are_available_only_in_checked_arm(self):
        state=self.state()
        raw=decision_context(state,False);checked=decision_context(state,True)
        for row in (raw['history'][0],checked['history'][0]):
            for key in ('configuration_fact_check','hypothesis_check','explanation_audit'):
                self.assertNotIn(key,row)
        self.assertEqual(checked['checks'][0]['configuration_fact_check']['status'],'contradicted')
        self.assertEqual(checked['checks'][0]['explanation_check']['warnings'][0]['field'],'hypothesis')
        self.assertEqual(checked['memory']['configuration_checks'][0]['check']['status'],'contradicted')
        self.assertEqual(checked['memory']['hypotheses'][0]['status'],'measurement_not_supported')
        checked.pop('checks');checked.pop('memory')
        self.assertEqual(raw,checked)

    def test_future_confirmation_does_not_erase_prior_contradiction(self):
        second=deepcopy(self.row);second['trial']=2;second['candidate']='numeric_tabular_logistic_C1.0'
        second['configuration_fact_check']={'status':'confirmed','declared':'logistic','observed':'logistic'}
        memory=self.state([self.row,second]).record()['memory']
        self.assertEqual([r['check']['status'] for r in memory['configuration_checks']],['contradicted','confirmed'])
        self.assertTrue(all(r['causal_explanation_verified'] is False for r in memory['hypotheses']))

    def test_hidden_feedback_does_not_leak_fitted_measurements(self):
        record=self.state(visible=False).record()
        self.assertIsNone(record['memory'])
        self.assertNotIn('estimator',record['history'][0])
        self.assertNotIn('configuration_fact_check',record['history'][0])

    def test_hypothesis_configuration_fact_is_audited_without_assessment(self):
        proposal=dict(candidate='numeric_tabular_logistic_C1.0',task_kind='numeric_tabular',
            reason='test next configuration',hypothesis='The model is an SVM.',evidence_trials=[1])
        audit=audit_explanation(proposal,[self.row],self.available[0])
        self.assertTrue(any(w.get('field')=='hypothesis' and w['code']=='prose_claim_contradiction' for w in audit['warnings']))
        self.assertFalse(audit['freeform_prose_fully_verified'])

    def test_projection_does_not_mutate_recorded_history(self):
        state=self.state();before=state.record()
        ctx=decision_context(state,True);ctx['checks'][0]['configuration_fact_check']['status']='tampered'
        self.assertEqual(state.record(),before)

if __name__=='__main__':unittest.main()
