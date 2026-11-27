"""Unowned native layouts may reach semantic intake; explicit contracts stay authoritative."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.events import RunLogger
from superagent.semantic_contract import resolve_training_task
from tests.test_native_semantics import fixture,binary_array
from tests.test_semantic_contract import Client


def npy_case(root):
    task,query,answer,expected=fixture(root)
    names={'readings.bin':'observations.npy','decisions.bin':'decisions.npy','incoming.bin':'arrivals.npy'}
    for path in [task,query]:
        with zipfile.ZipFile(path) as archive:members={n:archive.read(n) for n in archive.namelist()}
        with zipfile.ZipFile(path,'w') as archive:
            for name,body in members.items():
                if name=='README.md':
                    text=body.decode('utf-8')
                    for a,b in names.items():text=text.replace(a,b)
                    body=text.encode('utf-8')
                archive.writestr(names.get(name,name),body)
    for source in ['training','prediction']:
        for field in ['path','target_path']:
            if field in answer['plan'][source]:answer['plan'][source][field]=names.get(answer['plan'][source][field],answer['plan'][source][field])
    return task,query,answer,expected


class DefaultHandoffTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.task,self.query,self.answer,self.expected=npy_case(self.root)

    def test_real_default_native_contract_branch_reaches_one_shared_client(self):
        run=self.root/'run';run.mkdir();client=Client([self.answer])
        task=resolve_training_task(self.task,client,RunLogger(run),prediction_package=self.query)
        self.assertEqual(len(task.y_train),8);self.assertEqual(client.request_count,1)
        self.assertEqual(task.spec['native_input_recipe']['training']['path'],'observations.npy')
        events=[json.loads(s) for s in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(e['kind']=='semantic_reader_handoff' and e['explicit_manifest'] is False for e in events))

    def test_no_client_blocked_or_exhausted_does_not_spend(self):
        blocked=Client([self.answer]);blocked.blocked=True
        exhausted=Client([self.answer],limit=0)
        for client in [None,blocked,exhausted]:
            with self.subTest(client=client),self.assertRaises(ValueError):
                resolve_training_task(self.task,client,prediction_package=self.query)
            if client is not None:self.assertEqual(client.request_count,0)

    def test_broken_explicit_manifest_does_not_allow_reinterpretation(self):
        with zipfile.ZipFile(self.task,'a') as archive:
            archive.writestr('manifest.json',json.dumps(dict(schema_version=1,modality='tabular',metric='macro_f1',
                splits=dict(train=dict(format='npy',path='absent.npy',target_path='decisions.npy')))))
        client=Client([self.answer])
        with self.assertRaises(ValueError):resolve_training_task(self.task,client,prediction_package=self.query)
        self.assertEqual(client.request_count,0)

    def test_native_reader_success_remains_local(self):
        task=self.root/'standard.zip';query=self.root/'standard_query.zip'
        with zipfile.ZipFile(task,'w') as archive:
            archive.writestr('README.md','The primary metric is macro F1.')
            archive.writestr('X_train.npy',binary_array(np.arange(24).reshape(8,3)))
            archive.writestr('y_train.npy',binary_array(np.array(['a','b']*4)))
        with zipfile.ZipFile(query,'w') as archive:archive.writestr('X_test.npy',binary_array(self.expected))
        client=Client([RuntimeError('must stay local')]);loaded=resolve_training_task(task,client,prediction_package=query)
        self.assertEqual(len(loaded.y_train),8);self.assertEqual(client.request_count,0)

    def test_no_documentation_does_not_trigger_a_guessing_call(self):
        with zipfile.ZipFile(self.task) as archive:members={n:archive.read(n) for n in archive.namelist() if n!='README.md'}
        with zipfile.ZipFile(self.task,'w') as archive:
            for name,body in members.items():archive.writestr(name,body)
        client=Client([self.answer])
        with self.assertRaisesRegex(ValueError,'No documentation'):resolve_training_task(self.task,client,prediction_package=self.query)
        self.assertEqual(client.request_count,0)

    def test_transport_failure_propagates_without_repeated_call(self):
        client=Client([RuntimeError('synthetic transport failure')])
        with self.assertRaisesRegex(RuntimeError,'transport failure'):resolve_training_task(self.task,client,prediction_package=self.query)
        self.assertEqual(client.request_count,1);self.assertTrue(client.blocked)

    def test_actual_train_default_reaches_prefit_checkpoint(self):
        from superagent.autonomous_run import _train
        class Reached(RuntimeError):pass
        client=Client([self.answer])
        with patch('superagent.autonomous_run.standby',side_effect=Reached) as beforefit:
            with self.assertRaises(Reached):_train(self.task,self.root/'train',flash=client,fit_limit=2,prediction_package=self.query)
        self.assertEqual(len(beforefit.call_args.args[0].y_train),8);self.assertEqual(client.request_count,1)


if __name__=='__main__':unittest.main()
