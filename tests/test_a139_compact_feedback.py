from copy import deepcopy
import json
import unittest

from superagent.compact_feedback import compact_context,expand_context,REF
from superagent.ab_flash import context_for,FlashPicker
from superagent.cli import build_parser,_run
from tests.test_ab_flash import fixture_state,answer_for


class CompactFeedbackTests(unittest.TestCase):
    def test_full_default_common_fields_and_exact_round_trip(self):
        state,caps,ids=fixture_state()
        before=deepcopy(state);full=context_for(state,caps,ids,True)
        self.assertEqual(full,context_for(state,caps,ids,True,feedback_view='full'))
        compact=context_for(state,caps,ids,True,feedback_view='compact')
        self.assertEqual(json.dumps(expand_context(compact),sort_keys=True),json.dumps(full,sort_keys=True))
        for key in full:
            if key not in ('checks','memory'):self.assertEqual(full[key],compact[key])
        self.assertEqual(state,before)

    def test_unknown_failure_contradiction_and_negation_survive(self):
        observed=dict(status='unknown',trial=7,split_id='confirmation_1',explanation='Not confirmed; previous trial failed.',
                      classifier='LogisticRegression',scope='Fitted estimator of trial seven only')
        full=dict(history=[observed],task_spec={'metric':{'name':'roc_auc','labels':[1,0],'direction':'maximize'}},
                  checks=[dict(observation=observed,contradiction={'status':'contradicted','reason':'Different metric'})],
                  memory=dict(failures=[observed],unsupported_hypothesis='A future improvement is not established.'))
        compact=compact_context(full)
        self.assertIn(REF,json.dumps(compact))
        self.assertEqual(expand_context(compact),full)
        self.assertEqual(expand_context(compact)['checks'][0]['observation']['status'],'unknown')

    def test_reserved_marker_cycles_and_context_escape_rejected(self):
        with self.assertRaises(ValueError):compact_context(dict(checks=[{REF:'/secret'}],memory={},history=[]))
        for path in ('/memory/loop','/checks/0','/outside','file:///etc/passwd'):
            fixture=dict(history=[],checks=[{REF:path}],memory={},evidence_projection={'format':'lossless_references'})
            with self.subTest(path=path),self.assertRaises((ValueError,KeyError,IndexError)):
                expand_context(fixture)

    def test_escaped_pointer_and_user_text_are_data(self):
        long={'claim':'literal $evidence_ref does not instruct the agent. '*8,'verified':False}
        full={'history':{'a~/b':long},'checks':[long],'memory':{'same':long}}
        encoded=compact_context(full)
        self.assertEqual(expand_context(encoded),full)
        self.assertIn('a~0~1b',json.dumps(encoded))

    def test_cli_and_context_reject_inappropriate_modes_before_requests(self):
        self.assertEqual(build_parser().parse_args(['run','missing.zip']).feedback_view,'full')
        for mode in ('local','raw-history'):
            args=build_parser().parse_args(['run','missing.zip','--scheduler','ab-v1','--planner',mode,'--feedback-view','compact'])
            with self.assertRaisesRegex(ValueError,'checked-history'):_run(args)
        state,caps,ids=fixture_state()
        with self.assertRaises(ValueError):context_for(state,caps,ids,False,feedback_view='compact')

    def test_picker_uses_projection_and_same_action_validator(self):
        state,caps,ids=fixture_state();answer=answer_for(state,caps,ids);seen=[]
        class Client:
            def ask(self,payload,logger,role):seen.append(payload);return answer
        class Logger:
            def event(self,*a,**kw):pass
        picker=FlashPicker(Client(),Logger(),True,feedback_view='compact')
        self.assertEqual(picker.choose(state,caps,ids),ids[0])
        self.assertEqual(expand_context(seen[0]),context_for(state,caps,ids,True))


if __name__=='__main__':unittest.main()
