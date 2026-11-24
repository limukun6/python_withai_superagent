from copy import deepcopy
import unittest

from scripts.prepare_a145_grounding_cases import create_case,digest
from superagent.ab_run import registered_capabilities
from superagent.fact_registry import build_registry,check_claim
from superagent.grounded_proposal import assess,GroundingError
from superagent.grounding_issues import issue_for,confirm_execution
from superagent.ab_flash import FlashPicker,context_for
from superagent.cli import build_parser,_run


def fixture():
    spec=dict(contract_version=1,objective='classification',kind='numeric_tabular',input_schema={},metric={'name':'accuracy','labels':[0,1],'direction':'maximize'})
    caps=registered_capabilities(spec,200);case=create_case('object_identity',0,caps);state=case['state'];state['grounding_binding']=case['binding']
    reg=build_registry(state,caps,case['binding']);name='numeric_tabular_logistic_C1.0';cid='c:'+name+'.classifier'
    answer=dict(candidate=name,task_kind='numeric_tabular',configuration_claim={'classifier':'logistic'},
        reason='An unverified explanation.',review='Use the measured primary score.',hypothesis='This might help.',evidence_trials=[1],
        expected_change=None,slice_expectation=None,purpose='improve',uncertainty='Quality is unknown.',
        grounding=dict(namespace=reg['namespace'],evidence_refs=[cid,'t1.search.accuracy'],
                       assertions=[dict(kind='value',ref=cid,value='logistic')],addresses_issue_ids=[]))
    return state,caps,answer,reg


class GroundedTests(unittest.TestCase):
    def test_checked_context_and_executable_proposal_keep_raw_prose_unverified(self):
        state,caps,answer,reg=fixture();ids=[c['id'] for c in state['available_candidates']]
        packet=context_for(state,caps,ids,True,'grounded-v1','compact')
        self.assertEqual(packet['fact_registry']['namespace'],reg['namespace'])
        proposal,report=assess(answer,state,caps,ids)
        self.assertEqual(proposal.raw_proposal['reason'],'An unverified explanation.')
        self.assertTrue(proposal.reason.startswith('Checked improve'))
        self.assertFalse(report['freeform_prose_verified'])
        proposal.validate_available([{'id':c['id'],'params':c['params']} for c in state['available_candidates']],state['history'])

    def test_missing_scope_and_false_claims_are_never_certified(self):
        state,caps,answer,reg=fixture();ids=[c['id'] for c in state['available_candidates']]
        for change in ('namespace','false_claim','missing_evidence'):
            broken=deepcopy(answer)
            if change=='namespace':broken['grounding']['namespace']=digest('another run')
            if change=='false_claim':broken['grounding']['assertions'][0]['value']='rbf_svm'
            if change=='missing_evidence':broken['grounding']['evidence_refs']=[broken['grounding']['evidence_refs'][0]]
            with self.subTest(change=change),self.assertRaises(GroundingError):assess(broken,state,caps,ids)
        other=deepcopy(state);other['history'][0].pop('row_ids_sha256')
        self.assertEqual(check_claim({'kind':'value','ref':'t1.search.accuracy','value':.7},build_registry(other,caps,other['grounding_binding']))['status'],'unknown')

    def test_search_provenance_does_not_certify_training_or_another_run(self):
        state,caps,answer,reg=fixture()
        state['history'][0]['train_metrics']={'accuracy':.9}
        registry=build_registry(state,caps,state['grounding_binding'])
        self.assertEqual(check_claim({'kind':'value','ref':'t1.train.accuracy','value':.9},registry)['status'],'unknown')
        claim=dict(kind='value',ref='task.kind',value='numeric_tabular',namespace=digest('foreign'))
        self.assertEqual(check_claim(claim,registry)['status'],'unknown')
        cid=answer['grounding']['evidence_refs'][0]
        self.assertEqual(check_claim(dict(kind='value',ref=cid,value='logistic',expected_role='observed'),registry)['status'],'unknown')
        answer['grounding']['addresses_issue_ids']=[{}]
        with self.assertRaises(GroundingError):assess(answer,state,caps,[c['id'] for c in state['available_candidates']])

    def test_a_correction_is_pending_until_matching_execution(self):
        state,caps,answer,reg=fixture();ids=[c['id'] for c in state['available_candidates']]
        wrong=dict(kind='value',ref='task.kind',value='image');issue=issue_for(wrong,reg,1);state['grounding_issues']=[issue]
        with self.assertRaises(GroundingError):assess(answer,state,caps,ids)
        answer['grounding']['addresses_issue_ids']=[issue['issue_id']]
        answer['grounding']['assertions'].append(dict(kind='value',ref='task.kind',value='numeric_tabular'))
        _,report=assess(answer,state,caps,ids);pending=report['issues']
        self.assertEqual(pending[0]['status'],'addressed_pending_check')
        failed=confirm_execution(pending,{'trial':3,'status':'failed'})
        self.assertEqual(failed[0]['status'],'open')
        result=confirm_execution(pending,{'trial':3,'status':'completed','metric_identity':{'status':'confirmed'},'configuration_fact_check':{'status':'confirmed'}})
        self.assertEqual(result[0]['status'],'resolved')

    def test_default_contract_and_bad_modes_stop_before_data(self):
        self.assertEqual(build_parser().parse_args(['run','missing.zip']).proposal_contract,'strict-v1')
        for planner in ('local','raw-history'):
            args=build_parser().parse_args(['run','missing.zip','--scheduler','ab-v1','--planner',planner,'--proposal-contract','grounded-v1'])
            with self.assertRaisesRegex(ValueError,'checked-history'):_run(args)

    def test_semantic_repair_is_bounded_and_transport_failure_is_not_retried(self):
        state,caps,answer,reg=fixture();ids=[c['id'] for c in state['available_candidates']]
        wrong=deepcopy(answer);wrong['grounding']['assertions'][0]['value']='rbf_svm'
        issue=issue_for(wrong['grounding']['assertions'][0],reg)
        fixed=deepcopy(answer);fixed['grounding']['addresses_issue_ids']=[issue['issue_id']]
        class Logger:
            def event(self,*a,**kw):pass
        class Client:
            blocked=False;account=object();max_requests=3
            def __init__(self,items):self.items=iter(items);self.request_count=0
            def ask(self,*a,**kw):
                self.request_count+=1;v=next(self.items)
                if isinstance(v,Exception):raise v
                return v
        client=Client([wrong,fixed]);picker=FlashPicker(client,Logger(),True,'grounded-v1','compact')
        self.assertEqual(picker.choose(state,caps,ids),answer['candidate']);self.assertEqual(client.request_count,2)
        client=Client([RuntimeError('transport')]);picker=FlashPicker(client,Logger(),True,'grounded-v1')
        with self.assertRaisesRegex(RuntimeError,'transport'):picker.choose(state,caps,ids)
        self.assertEqual(client.request_count,1)
        client=Client([wrong,wrong]);picker=FlashPicker(client,Logger(),True,'grounded-v1')
        with self.assertRaises(GroundingError):picker.choose(state,caps,ids)
        self.assertEqual(client.request_count,2)
        client=Client([wrong]);client.max_requests=1;picker=FlashPicker(client,Logger(),True,'grounded-v1')
        with self.assertRaises(GroundingError):picker.choose(state,caps,ids)
        self.assertEqual(client.request_count,1)


if __name__=='__main__':unittest.main()
