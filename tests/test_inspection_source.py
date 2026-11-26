"""A source omission is repairable only by unique physical member identity."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.sa7_multiturn_cases import build
from superagent.format_sources import package_files
from superagent.semantic_contract import inspection_source,_inspect_header,evidence_workspace,resolve_training_task,OUTPUT_SCHEMA
from tests.test_semantic_contract import Client


class InspectionSourceTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.path=self.root/'train.zip';self.query=self.root/'query.zip'
        with zipfile.ZipFile(self.path,'w') as archive:
            archive.writestr('README.md','Training is data.csv. Query features are incoming.csv. Keep record_id in the output.')
            archive.writestr('data.csv','feature,target\n1,A\n2,B\n')
            archive.writestr('shared.csv','feature,target\n1,A\n')
        with zipfile.ZipFile(self.query,'w') as archive:
            archive.writestr('incoming.csv','record_id,observed\nsecret,9\n')
            archive.writestr('shared.csv','record_id,observed\nsecret,9\n')
            archive.writestr('y_test.csv','label\nPRIVATE\n')
        self.request=dict(action='inspect_header',member='incoming.csv',collection=None,delimiter=',',evidence_refs=['doc:README.md:0'])

    def inspect(self,request):
        with package_files(self.path) as task,package_files(self.query) as query:
            owner,normalized,reason=inspection_source(task,query,request)
            observation=_inspect_header(owner,evidence_workspace(task),normalized)
            return observation,reason

    def test_missing_role_resolves_unique_external_member(self):
        observed,reason=self.inspect(self.request)
        self.assertEqual(observed['source'],'prediction');self.assertEqual(reason,'unique_package_membership')
        self.assertEqual(observed['fields'],['record_id','observed']);self.assertFalse(observed['row_values_disclosed'])
        self.assertNotIn('secret',json.dumps(observed))

    def test_unique_internal_member_stays_training(self):
        observed,_=self.inspect(dict(self.request,member='data.csv'))
        self.assertEqual(observed['source'],'training');self.assertEqual(observed['fields'],['feature','target'])

    def test_explicit_wrong_role_never_falls_back(self):
        with self.assertRaises(ValueError):self.inspect(dict(self.request,source='training'))
        with self.assertRaises(ValueError):self.inspect(dict(self.request,member='data.csv',source='prediction'))

    def test_same_member_in_two_packages_needs_explicit_role(self):
        with self.assertRaisesRegex(ValueError,'both packages'):self.inspect(dict(self.request,member='shared.csv'))
        observed,reason=self.inspect(dict(self.request,member='shared.csv',source='prediction'))
        self.assertEqual(observed['fields'],['record_id','observed']);self.assertEqual(reason,'explicit')

    def test_unique_membership_does_not_allow_label_vectors_or_forged_citations(self):
        for change in [dict(member='y_test.csv'),dict(evidence_refs=['fake'])]:
            with self.subTest(change=change),self.assertRaises(ValueError):self.inspect({**self.request,**change})

    def test_source_resolution_reaches_followup_and_schema_lists_conditional_fields(self):
        cases=self.root/'cases';registry=build(cases);row=registry[0]
        oracle=json.loads((cases/'Q1.oracle.json').read_text(encoding='utf-8'))
        answer=oracle['answer'];member=answer['plan']['prediction']['member']
        request=dict(self.request,member=member,delimiter=';')
        client=Client([request,answer])
        task=resolve_training_task(cases/row['package'],client,mode='proactive',prediction_package=cases/row['query'])
        self.assertEqual(len(task.y_train),8);self.assertEqual(client.request_count,2)
        observation=client.payloads[1]['tool_observations'][0]
        self.assertEqual(observation['package_source'],'external_prediction')
        self.assertIsNone(observation['requested_source']);self.assertEqual(observation['source_resolution'],'unique_package_membership')
        variants=client.payloads[0]['output_contract_schema']['properties']['columns']['items']['oneOf']
        self.assertIn('field',variants[0]['required']);self.assertIn('start',variants[1]['required'])
        self.assertIn('class_label',variants[3]['required'])
        self.assertEqual(client.payloads[0]['output_contract_schema'],OUTPUT_SCHEMA)


if __name__=='__main__':unittest.main()
