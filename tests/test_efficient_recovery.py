from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch

import requests

from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM,MeteredResponseFormatError
from superagent.recovery import RecoveryController
from superagent.semantic_contract import semantic_training,UnresolvedTaskReport
from superagent.task_budget import TaskBudgetLedger,TaskBudgetAccount
from tests.test_semantic_contract import Client,fixture,proposal


def inspect(member):
    return dict(action='inspect_header',member=member,collection=None,delimiter=',',evidence_refs=['doc:01_overview.md:0'])


def report(field='target'):
    return dict(action='report_unresolved',reason='The supplied documents leave this meaning unspecified.',
                questions=[dict(field=field,question='Which definition is requested?',evidence_refs=['doc:01_overview.md:0'])])


class EfficientRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=fixture(self.root)

    def test_one_batch_two_headers_leaves_one_request_for_planning(self):
        client=Client([dict(action='inspect_batch',requests=[inspect('ledger.data'),inspect('query.data')]),proposal()])
        task=semantic_training(self.path,client,None,ValueError('native failure'))
        self.assertEqual(client.request_count,2);self.assertEqual(len(task.y_train),8)
        self.assertEqual(len(client.payloads[1]['tool_observations']),2)
        self.assertFalse(client.payloads[1]['tool_observations'][0]['row_values_disclosed'])

    def test_batch_rejects_three_tools_before_executing_any(self):
        client=Client([dict(action='inspect_batch',requests=[inspect('query.data')]*3),proposal()])
        with patch('superagent.semantic_contract._inspect_header',side_effect=AssertionError('opened tool')):
            semantic_training(self.path,client,None,ValueError('native'))
        self.assertEqual(client.request_count,2)

    def test_nested_batch_is_not_executed(self):
        nested=dict(action='inspect_batch',requests=[dict(action='inspect_batch',requests=[inspect('query.data')])])
        client=Client([nested,proposal()])
        with patch('superagent.semantic_contract._inspect_header',side_effect=AssertionError('opened tool')):
            semantic_training(self.path,client,None,ValueError('native'))

    def test_failed_batch_member_does_not_discard_other_observation(self):
        client=Client([dict(action='inspect_batch',requests=[inspect('missing.data'),inspect('query.data')]),proposal()])
        semantic_training(self.path,client,None,ValueError('native'))
        self.assertEqual(client.payloads[1]['tool_observations'][0]['member'],'query.data')
        self.assertEqual(len(client.payloads[1]['recovery_state']['failed_actions']),2)

    def test_two_physical_tools_total_across_requests(self):
        client=Client([inspect('query.data'),dict(action='inspect_batch',requests=[inspect('ledger.data'),inspect('query.data')]),proposal()])
        semantic_training(self.path,client,None,ValueError('native'))
        self.assertEqual(len(client.payloads[2]['tool_observations']),1)

    def test_same_unresolved_field_with_paraphrased_reply_is_stagnation(self):
        a=proposal();a['version']='semantic-task-v3';a['plan']['output']=None
        a['uncertainties']=[dict(field='target',question='Unknown target definition',evidence_refs=['doc:01_overview.md:0'])]
        b=deepcopy(a);b['uncertainties'][0]['question']='Cannot establish requested target meaning'
        client=Client([a,b,proposal()]);semantic_training(self.path,client,None,ValueError('native'))
        state=client.payloads[2]['recovery_state']
        self.assertTrue(state['failed_actions'][-1]['repeated_unresolved_without_new_evidence'])
        self.assertIn('Do not merely paraphrase',client.payloads[2]['diagnostic']['recovery_hint'])

    def test_duplicate_structure_is_not_new_semantic_evidence(self):
        controller=RecoveryController();observation={'member':'x','source':'training','fields':['a']}
        controller.observed('inspect_header',observation)
        controller.failure('contract_probe',{'uncertainty_fields':['target']})
        controller.observed('inspect_header',dict(observation,requested_source='training'))
        controller.failure('contract_probe',{'uncertainty_fields':['target'],'message':'Different words'})
        self.assertTrue(controller.state['failed_actions'][-1]['repeated_unresolved_without_new_evidence'])

    def test_explicit_unresolved_report_stops_without_another_call(self):
        client=Client([report(),proposal()]);controller=RecoveryController(client=client)
        with self.assertRaises(UnresolvedTaskReport):semantic_training(self.path,client,None,ValueError('native'),recovery=controller)
        self.assertEqual(client.request_count,1);self.assertEqual(controller.state['status'],'unresolved')
        self.assertFalse(controller.state['unresolved_report']['semantic_insufficiency_certified'])
        self.assertEqual(controller.state['unresolved_fields'],['target'])

    def test_fabricated_report_citation_is_a_repairable_action(self):
        bad=report();bad['questions'][0]['evidence_refs']=['fabricated']
        client=Client([bad,proposal()]);semantic_training(self.path,client,None,ValueError('native'))
        self.assertEqual(client.request_count,2)

    def test_optional_positive_label_cannot_be_used_as_required_stop(self):
        client=Client([report('positive_label'),proposal()]);semantic_training(self.path,client,None,ValueError('native'))
        self.assertEqual(client.request_count,2)


class MeteredFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.path=fixture(self.root);self.key=self.root/'key.txt';self.key.write_text('SYNTHETIC-TEST-KEY')
        self.ledger=TaskBudgetLedger.initialize(self.root/'synthetic.sqlite',new_project=True)
        self.account=TaskBudgetAccount(self.ledger,'test_scope','unit_test')
        (self.root/'log').mkdir();self.log=RunLogger(self.root/'log');self.log.initialize_usage_ledger()

    def client(self):
        return ExperimentLLM('https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions',str(self.key),
            'DeepSeek-V4.1-Flash',.30,max_requests=3,account=self.account)

    def transport(self,mode):
        replies=['invalid JSON',json.dumps(proposal())];calls=[]
        def post(*args,**kwargs):
            calls.append(kwargs)
            if mode=='timeout':raise requests.Timeout('synthetic timeout')
            response=MagicMock();response.__enter__.return_value=response;response.ok=True;response.status_code=200
            response.json.return_value=dict(model='DeepSeek-V4.1-Flash',usage=dict(prompt_tokens=100,completion_tokens=20),
                choices=[dict(message=dict(content=replies.pop(0)),finish_reason='length' if mode=='length' else 'stop')])
            return response
        session=MagicMock();session.__enter__.return_value=session;session.post.side_effect=post
        return patch('superagent.experiment_llm.requests.Session',return_value=session),calls

    def test_settled_invalid_json_uses_next_shared_request_and_both_costs(self):
        transport,calls=self.transport('json');client=self.client()
        with transport:task=semantic_training(self.path,client,self.log,ValueError('native'))
        self.assertEqual(len(task.y_train),8);self.assertEqual(client.request_count,2);self.assertEqual(len(calls),2)
        self.assertNotEqual(calls[0]['json']['messages'],calls[1]['json']['messages'])
        state=self.ledger.project_summary();self.assertEqual(state['requests'],2);self.assertEqual(state['unresolved_requests'],0)
        self.assertAlmostEqual(state['known_course_cost_rmb'],.00036)

    def test_transport_failure_is_not_a_format_repair(self):
        transport,calls=self.transport('timeout');client=self.client()
        with transport,self.assertRaises(Exception):semantic_training(self.path,client,self.log,ValueError('native'))
        self.assertEqual(len(calls),1);self.assertTrue(client.blocked)
        self.assertEqual(self.ledger.project_summary()['unresolved_requests'],1)

    def test_truncated_response_does_not_trigger_format_repair(self):
        transport,calls=self.transport('length');client=self.client()
        with transport,self.assertRaisesRegex(ValueError,'token limit'):semantic_training(self.path,client,self.log,ValueError('native'))
        self.assertEqual(len(calls),1);self.assertEqual(self.ledger.project_summary()['unresolved_requests'],0)

    def test_unsettled_format_exception_is_not_eligible_for_repair(self):
        client=Client([MeteredResponseFormatError('not settled'),proposal()])
        with self.assertRaises(MeteredResponseFormatError):semantic_training(self.path,client,None,ValueError('native'))
        self.assertEqual(client.request_count,1)


if __name__=='__main__':unittest.main()
