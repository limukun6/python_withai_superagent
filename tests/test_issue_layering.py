from copy import deepcopy
import unittest

from scripts.prepare_a145_grounding_cases import create_case
from superagent.ab_flash import context_for
from superagent.ab_run import registered_capabilities
from superagent.cli import build_parser,_run
from superagent.fact_registry import build_registry
from superagent.grounded_proposal import assess,GroundingError
from superagent.grounding_issues import (CHANNEL,EXECUTION,KNOWLEDGE,MEASUREMENT,address_issues,
                                         claim_class,confirm_execution,issue_for)

MATCHED_ROW={'trial':3,'status':'completed','metric_identity':{'status':'confirmed'},
             'configuration_fact_check':{'status':'confirmed'}}


def fixture():
    spec=dict(contract_version=1,objective='classification',kind='numeric_tabular',input_schema={},
              metric={'name':'accuracy','labels':[0,1],'direction':'maximize'})
    caps=registered_capabilities(spec,200);case=create_case('object_identity',0,caps)
    state=case['state'];state['grounding_binding']=case['binding']
    registry=build_registry(state,caps,case['binding'])
    name='numeric_tabular_logistic_C1.0'
    answer=dict(candidate=name,task_kind='numeric_tabular',configuration_claim={'classifier':'logistic'},
        reason='An unverified explanation.',review='Use the measured primary score.',hypothesis='This might help.',
        evidence_trials=[1],expected_change=None,slice_expectation=None,purpose='improve',
        uncertainty='Quality is unknown.',
        grounding=dict(namespace=registry['namespace'],
                       evidence_refs=['c:'+name+'.classifier','t1.search.accuracy'],
                       assertions=[dict(kind='value',ref='c:'+name+'.classifier',value='logistic')],
                       addresses_issue_ids=[]))
    return state,caps,answer,registry,[c['id'] for c in state['available_candidates']]


class ClassTests(unittest.TestCase):
    def test_current_claim_forms_are_knowledge(self):
        state,caps,answer,registry,ids=fixture()
        for claim in (dict(kind='value',ref='task.kind',value='image'),
                      dict(kind='compare',left='t1.search.accuracy',op='>',right='t1.search.accuracy'),
                      dict(kind='parameter_relation',before='c:x.C',after='c:y.C',
                           direction='weaker_regularization')):
            self.assertEqual(claim_class(claim,registry),KNOWLEDGE,claim)

    def test_reserved_forms_need_a_run_or_a_measurement(self):
        state,caps,answer,registry,ids=fixture()
        self.assertEqual(claim_class(dict(kind='capability',candidate='x'),registry),EXECUTION)
        self.assertEqual(claim_class(dict(kind='uplift',candidate='x'),registry),MEASUREMENT)
        self.assertEqual(claim_class('not a claim',registry),MEASUREMENT)


class LayeringOffTests(unittest.TestCase):
    def test_unknown_facts_never_become_issues(self):
        state,caps,answer,registry,ids=fixture()
        self.assertIsNone(issue_for(dict(kind='value',ref='t9.search.accuracy',value=.5),registry,1,'by-class'))

    def test_old_behaviour_is_byte_compatible(self):
        state,caps,answer,registry,ids=fixture()
        issue=issue_for(dict(kind='value',ref='task.kind',value='image'),registry,1)
        self.assertNotIn('issue_class',issue);self.assertNotIn('resolution_channel',issue)
        self.assertEqual(issue['required_postcondition'],
                         'A confirmed assertion with the same bound proposition, then matching execution evidence')
        result,errors=address_issues([issue],[dict(kind='value',ref='task.kind',value='numeric_tabular')],
                                     [issue['issue_id']],registry)
        self.assertEqual(errors,[]);self.assertEqual(result[0]['status'],'addressed_pending_check')
        self.assertEqual(confirm_execution(result,MATCHED_ROW)[0]['status'],'resolved')


class LayeringByClassTests(unittest.TestCase):
    def test_a_knowledge_correction_closes_on_the_check_alone(self):
        state,caps,answer,registry,ids=fixture()
        issue=issue_for(dict(kind='value',ref='task.kind',value='image'),registry,1,'by-class')
        self.assertEqual(issue['issue_class'],KNOWLEDGE)
        self.assertEqual(issue['resolution_channel'],CHANNEL[KNOWLEDGE])
        self.assertFalse(issue['new_fit_required'])
        result,errors=address_issues([issue],[dict(kind='value',ref='task.kind',value='numeric_tabular')],
                                     [issue['issue_id']],registry,'by-class')
        self.assertEqual(errors,[])
        self.assertEqual(result[0]['status'],'resolved_semantic')
        self.assertEqual(result[0]['closed_by'],CHANNEL[KNOWLEDGE])
        self.assertIsNone(result[0]['execution_confirmation'])

    def test_a_knowledge_correction_is_not_reopened_by_a_failed_trial(self):
        state,caps,answer,registry,ids=fixture()
        issue=issue_for(dict(kind='value',ref='task.kind',value='image'),registry,1,'by-class')
        addressed,_=address_issues([issue],[dict(kind='value',ref='task.kind',value='numeric_tabular')],
                                   [issue['issue_id']],registry,'by-class')
        failed=confirm_execution(addressed,{'trial':3,'status':'failed'},'by-class')
        self.assertEqual(failed[0]['status'],'resolved_semantic')

    def test_an_execution_issue_still_waits_for_matching_evidence(self):
        state,caps,answer,registry,ids=fixture()
        issue=issue_for(dict(kind='value',ref='task.kind',value='image'),registry,1,'by-class')
        issue.update(issue_class=EXECUTION,resolution_channel=CHANNEL[EXECUTION],new_fit_required=True)
        addressed,_=address_issues([issue],[dict(kind='value',ref='task.kind',value='numeric_tabular')],
                                   [issue['issue_id']],registry,'by-class')
        self.assertEqual(addressed[0]['status'],'addressed_pending_check')
        self.assertEqual(confirm_execution(addressed,{'trial':3,'status':'failed'},'by-class')[0]['status'],'open')
        self.assertEqual(confirm_execution(addressed,MATCHED_ROW,'by-class')[0]['status'],'resolved')

    def test_a_measurement_issue_is_not_closed_by_an_unrelated_trial(self):
        state,caps,answer,registry,ids=fixture()
        issue=issue_for(dict(kind='value',ref='task.kind',value='image'),registry,1,'by-class')
        issue.update(issue_class=MEASUREMENT,resolution_channel=CHANNEL[MEASUREMENT],new_fit_required=True)
        addressed,_=address_issues([issue],[dict(kind='value',ref='task.kind',value='numeric_tabular')],
                                   [issue['issue_id']],registry,'by-class')
        self.assertEqual(confirm_execution(addressed,MATCHED_ROW,'by-class')[0]['status'],'addressed_pending_check')

    def test_grounded_assessment_returns_the_layered_status(self):
        state,caps,answer,registry,ids=fixture()
        wrong=dict(kind='value',ref='task.kind',value='image')
        issue=issue_for(wrong,registry,1,'by-class');state['grounding_issues']=[issue]
        with self.assertRaises(GroundingError):assess(answer,state,caps,ids,'by-class')
        answer['grounding']['addresses_issue_ids']=[issue['issue_id']]
        answer['grounding']['assertions'].append(dict(kind='value',ref='task.kind',value='numeric_tabular'))
        _,report=assess(answer,state,caps,ids,'by-class')
        self.assertEqual(report['issues'][0]['status'],'resolved_semantic')

    def test_layered_context_states_the_rule_and_off_context_does_not(self):
        state,caps,answer,registry,ids=fixture()
        base=context_for(state,caps,ids,True,'grounded-v1','full')
        layered=context_for(state,caps,ids,True,'grounded-v1','full','by-class')
        self.assertNotIn('Issue layering is on',base['instruction'])
        self.assertIn('does not need a new experiment',layered['instruction'])
        self.assertEqual(layered['issue_layering'],'by-class')

    def test_diagnose_still_needs_an_open_issue(self):
        """Layering removes knowledge-only reasons, not the diagnose action itself."""
        state,caps,answer,registry,ids=fixture()
        answer['purpose']='diagnose';answer['grounding']['addresses_issue_ids']=[]
        with self.assertRaisesRegex(GroundingError,'open issue'):assess(answer,state,caps,ids,'by-class')


class CliTests(unittest.TestCase):
    def test_default_is_off_and_layering_requires_grounded(self):
        self.assertEqual(build_parser().parse_args(['run','missing.zip']).issue_layering,'off')
        args=build_parser().parse_args(['run','missing.zip','--scheduler','ab-v1','--planner','checked-history',
                                        '--issue-layering','by-class'])
        with self.assertRaisesRegex(ValueError,'grounded-v1'):_run(args)


if __name__=='__main__':unittest.main()
