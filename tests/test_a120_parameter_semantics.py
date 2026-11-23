import unittest
from unittest.mock import patch

from superagent.cli import build_parser,_run
from superagent.ab_run import validate_options
from superagent.parameter_semantics import check_parameter_semantics
from superagent.explanation_audit import audit_explanation


class ParameterSemanticsTests(unittest.TestCase):
    def args(self,planner='checked-history',mode='diagnostic'):
        return build_parser().parse_args(['run','anonymous.zip','--planner',planner,'--scheduler','ab-fixed',
            '--portfolio','ab-v1','--parameter-semantics',mode])

    def test_diagnostics_are_explicit_checked_AB_only(self):
        validate_options(self.args())
        for planner in ('local','raw-history'):
            with self.assertRaisesRegex(ValueError,'checked-history'):validate_options(self.args(planner))
            validate_options(self.args(planner,'off'))
        args=self.args();args.scheduler='fixed'
        with self.assertRaisesRegex(ValueError,'shared A/B'):_run(args)

    def test_no_ambient_switch_or_accidental_default_on(self):
        import os
        with patch.dict(os.environ,{'SUPERAGENT_PARAMETER_SEMANTICS':'diagnostic'}):
            self.assertEqual(self.args(mode='off').parameter_semantics,'off')
            with patch('superagent.ab_run.command',side_effect=lambda args:args.parameter_semantics):
                self.assertEqual(_run(self.args(mode='off')),'off')
                self.assertEqual(_run(self.args()),'diagnostic')

    def test_report_has_provenance_and_never_changes_candidate(self):
        candidate={'id':'numeric_tabular_logistic_C1.0','params':{'C':1.}}
        history=[dict(trial=1,candidate='numeric_tabular_logistic_C0.1',params={'C':.1},status='completed',
            primary_metric='accuracy',search_score=.7,train_score=.8,
            estimator={'family':'logistic','source':'fitted_estimator'})]
        proposal=dict(candidate=candidate['id'],task_kind='numeric_tabular',
            reason='Increasing regularization strength from C=0.1 to C=1.0 may help.',review='No causal evidence.',hypothesis='Unknown.')
        base=audit_explanation(proposal,history,candidate)
        self.assertNotIn('parameter_semantics',base)
        report=audit_explanation(proposal,history,candidate,parameter_semantics=True,
            selected_facts={'classifier':'logistic','source':'shared_ab_catalogue'})
        row=report['parameter_semantics']['warnings'][0]
        self.assertEqual(row['reference_trial'],1)
        self.assertEqual(row['candidate_value_source'],'declared_candidate_configuration')
        self.assertEqual(row['reference_estimator_source'],'fitted_estimator')
        self.assertIn('excerpt_sha256',row['definition'])
        self.assertFalse(report['changes_candidate']);self.assertFalse(report['triggers_api_retry'])

    def test_unsupported_penalty_and_unbound_prose_do_not_certify(self):
        for penalty in (None,'none'):
            row=check_parameter_semantics({'reason':'Increasing C strengthens regularization.'},
                {'id':'numeric_tabular_logistic_C1.0','params':{'C':1.,'penalty':penalty}},[],
                {'classifier':'logistic','source':'shared_ab_catalogue'})
            self.assertFalse(row['warnings']);self.assertEqual(row['items'][0]['status'],'unknown')


if __name__=='__main__':unittest.main()
