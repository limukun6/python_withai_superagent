"""Post-hoc replay of actual tool failures plus container/role boundary checks."""
import json
from contextlib import closing
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest.mock import patch
import zipfile

from scripts.sa5_semantic_cases import build
from superagent.format_sources import package_files
from superagent.semantic_contract import _inspect_header,evidence_workspace,resolve_training_task,ContractProbeError
from tests.test_semantic_contract import Client


CAPTURED={
 'P1':dict(action='inspect_header',member='train.csv',collection='evidence_workspace',delimiter=None,
           evidence_refs=['doc:README.md:0']),
 'P2':dict(action='inspect_header',member='incoming.payload',collection='incoming',delimiter=None,
           evidence_refs=['doc:00_notice.md:0'],source='prediction'),
 'P4':dict(action='inspect_header',member='records.payload',collection='records',delimiter=None,
           evidence_refs=['doc:README.md:0'])}


class SemanticToolRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cases=self.root/'cases';build(self.cases)

    def test_three_captured_requests_now_return_actual_fields_without_row_values(self):
        expected={'P1':['signal','label'],'P2':['event_id','signal'],'P4':['reading','decision','phase']}
        for name,request in CAPTURED.items():
            with package_files(self.cases/(name+'.zip')) as files:
                result=_inspect_header(files,evidence_workspace(files),request)
            self.assertEqual(result['fields'],expected[name])
            self.assertIsNone(result['collection']);self.assertTrue(result['collection_normalized'])
            if name in ('P1','P2'):self.assertEqual(result['delimiter'],',')
            self.assertFalse(result['row_values_disclosed'])
            self.assertNotIn('FORBIDDEN_SA5_VALUE',json.dumps(result))

    def test_same_package_prediction_role_reaches_the_next_proposal(self):
        oracle=json.loads((self.cases/'P2.oracle.json').read_text(encoding='utf-8'))
        client=Client([CAPTURED['P2'],oracle['answer']])
        task=resolve_training_task(self.cases/'P2.zip',client,mode='proactive')
        self.assertEqual(client.request_count,2)
        observed=client.payloads[1]['tool_observations'][0]
        self.assertEqual(observed['package_source'],'task')
        self.assertEqual(observed['source'],'prediction')
        self.assertEqual(len(task.y_train),8)

    def test_external_package_missing_member_does_not_fall_back_to_task_package(self):
        external=self.root/'external.zip'
        with zipfile.ZipFile(external,'w') as archive:archive.writestr('other.csv','signal\n1\n')
        client=Client([CAPTURED['P2']],limit=1)
        with self.assertRaises(ValueError):
            resolve_training_task(self.cases/'P2.zip',client,mode='proactive',prediction_package=external)
        self.assertEqual(client.request_count,1)
        self.assertEqual(len(client.payloads[0]['tool_observations']),0)

    def test_named_json_container_does_not_ignore_an_invalid_selector(self):
        path=self.root/'named.zip'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('README.md','Use nested.json.')
            archive.writestr('nested.json',json.dumps({'train':[{'x':1,'y':0}], 'validation':[{'secret':'PRIVATE'}]}))
        request=dict(CAPTURED['P1'],member='nested.json',collection='absent')
        with package_files(path) as files:
            with self.assertRaises(ValueError): _inspect_header(files,evidence_workspace(files),request)

    def test_sqlite_invalid_selector_never_tries_a_default_table(self):
        db_path=self.root/'table.sqlite'
        with closing(sqlite3.connect(db_path)) as db,db:
            db.execute('CREATE TABLE train(x INTEGER, result TEXT)')
            db.execute("INSERT INTO train VALUES (1,'A')")
        path=self.root/'table.zip'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('README.md','Use the train table in container.bin.')
            archive.writestr('container.bin',db_path.read_bytes())
        from superagent.record_streams import decode_records
        with package_files(path) as files:
            workspace=evidence_workspace(files)
            with patch('superagent.semantic_contract.decode_records',wraps=decode_records) as decoder:
                with self.assertRaises(ValueError):
                    _inspect_header(files,workspace,dict(CAPTURED['P1'],member='container.bin',collection='absent'))
            self.assertEqual(decoder.call_count,1)

    def test_known_evaluation_alias_still_rejects_before_read(self):
        with package_files(self.cases/'P1.zip') as files:
            workspace=evidence_workspace(files)
            for alias in ['validation','answers','y']:
                request=dict(CAPTURED['P1'],collection=alias)
                with patch.object(files,'read',side_effect=AssertionError('forbidden read')) as reader:
                    with self.assertRaises(ContractProbeError):_inspect_header(files,workspace,request)
                reader.assert_not_called()

    def test_forged_citation_is_not_normalized_into_a_valid_request(self):
        with package_files(self.cases/'P1.zip') as files:
            workspace=evidence_workspace(files)
            request=dict(CAPTURED['P1'],evidence_refs=['invented'])
            with patch.object(files,'read',side_effect=AssertionError('forbidden read')):
                with self.assertRaises(ContractProbeError):_inspect_header(files,workspace,request)


if __name__=='__main__':unittest.main()
