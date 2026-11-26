"""Production artifact save/load and CLI boundaries; no fitting or real model calls."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch
import zipfile

import numpy as np

from superagent.cli import build_parser
from superagent.events import RunLogger
from superagent.semantic_artifact import digest,load_artifact
from superagent.semantic_contract import resolve_training_task
from scripts.sa7_multiturn_cases import build
from tests.test_proactive_probe import output_proposal
from tests.test_output_recipe import package
from tests.test_semantic_contract import Client


class SemanticArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=package(self.root/'task.zip')
        self.run=self.root/'original';self.run.mkdir();self.log=RunLogger(self.run)
        self.original=resolve_training_task(self.path,Client([output_proposal()]),self.log,mode='proactive')
        self.artifact=self.run/'semantic_artifact.json'

    def test_accepted_plan_is_saved_and_restored_without_asking_model(self):
        self.assertTrue(self.artifact.is_file());client=Client([RuntimeError('must not be asked')])
        task=resolve_training_task(self.path,client,artifact_from=self.artifact)
        np.testing.assert_array_equal(task.X_train,self.original.X_train)
        np.testing.assert_array_equal(task.y_train,self.original.y_train)
        self.assertEqual(task.spec['output_recipe'],self.original.spec['output_recipe'])
        self.assertEqual(client.request_count,0)

    def test_relocated_identical_package_still_matches(self):
        other=self.root/'different_name.zip';shutil.copyfile(self.path,other)
        task=load_artifact(self.artifact,other)
        self.assertEqual(len(task.y_train),8)

    def test_changed_training_bytes_are_refused(self):
        with zipfile.ZipFile(self.path,'a') as archive:archive.writestr('new_notice.md','Changed instructions')
        with self.assertRaisesRegex(ValueError,'different training'):load_artifact(self.artifact,self.path)

    def test_changed_prediction_bytes_are_refused(self):
        other=package(self.root/'changed.zip',duplicate=True)
        with self.assertRaisesRegex(ValueError,'different prediction'):
            load_artifact(self.artifact,self.path,other)

    def test_tampered_payload_checksum_is_refused(self):
        record=json.loads(self.artifact.read_text(encoding='utf-8'))
        record['payload']['proposal']['plan']['metric']='accuracy'
        self.artifact.write_text(json.dumps(record),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'content/version'):load_artifact(self.artifact,self.path)

    def test_tampered_task_digest_is_refused_even_after_checksum_recomputed(self):
        record=json.loads(self.artifact.read_text(encoding='utf-8'))
        record['payload']['task_sha256']='0'*64;record['payload_sha256']=digest(record['payload'])
        self.artifact.write_text(json.dumps(record),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'accepted digest'):load_artifact(self.artifact,self.path)

    def test_reader_drift_is_refused(self):
        with patch('superagent.semantic_artifact.reader_identity',return_value={}):
            with self.assertRaisesRegex(ValueError,'reader source changed'):load_artifact(self.artifact,self.path)

    def test_external_binding_requires_the_original_query_content(self):
        cases=self.root/'external';registry=build(cases);row=registry[0]
        oracle=json.loads((cases/'Q1.oracle.json').read_text(encoding='utf-8'))
        run=self.root/'external_run';run.mkdir()
        resolve_training_task(cases/row['package'],Client([oracle['answer']]),RunLogger(run),mode='proactive',prediction_package=cases/row['query'])
        artifact=run/'semantic_artifact.json'
        with self.assertRaisesRegex(ValueError,'bound external'):load_artifact(artifact,cases/row['package'])
        restored=load_artifact(artifact,cases/row['package'],cases/row['query'])
        self.assertEqual(restored.spec['input_recipe']['field_map'],{'level':'observed_level'})

    def test_explicit_image_overlay_artifact_recompiles_native_task(self):
        cases=self.root/'images';registry=build(cases);row=registry[-1]
        oracle=json.loads((cases/'Q4.oracle.json').read_text(encoding='utf-8'))
        run=self.root/'image_run';run.mkdir()
        resolve_training_task(cases/row['package'],Client([oracle['answer']]),RunLogger(run),mode='proactive',prediction_package=cases/row['query'])
        task=load_artifact(run/'semantic_artifact.json',cases/row['package'],cases/row['query'])
        self.assertEqual(task.kind,'image');self.assertEqual(task.spec['feature_shape'],[12,12,3])

    def test_directory_fresh_interpretation_continues_without_an_artifact(self):
        directory=self.root/'directory';directory.mkdir()
        with zipfile.ZipFile(self.path) as archive:archive.extractall(directory)
        run=self.root/'directory_run';run.mkdir()
        task=resolve_training_task(directory,Client([output_proposal()]),RunLogger(run),mode='proactive')
        self.assertEqual(len(task.y_train),8)
        self.assertFalse((run/'semantic_artifact.json').exists())
        self.assertIn('semantic_artifact_unavailable',(run/'events.jsonl').read_text(encoding='utf-8'))

    def test_cache_cli_does_not_initialize_an_implicit_billing_client(self):
        args=build_parser().parse_args(['unified',str(self.path),'--run-dir',str(self.root/'cached'),
                                      '--semantic-artifact-from',str(self.artifact)])
        with (patch('superagent.unified.configured_flash',side_effect=AssertionError('billing initialized')),
              patch('superagent.autonomous_run.train',return_value=0) as train):
            self.assertEqual(args.handler(args),0)
        self.assertIsNone(train.call_args.kwargs['flash'])
        self.assertEqual(train.call_args.kwargs['semantic_artifact_from'],str(self.artifact))

    def test_real_training_entry_loads_cache_before_the_first_fit(self):
        from superagent.autonomous_run import _train
        class AtCheckpoint(RuntimeError):pass
        with patch('superagent.autonomous_run.standby',side_effect=AtCheckpoint) as checkpoint:
            with self.assertRaises(AtCheckpoint):
                _train(self.path,self.root/'restored_run',fit_limit=2,semantic_artifact_from=self.artifact)
        self.assertEqual(len(checkpoint.call_args.args[0].y_train),8)
        receipt=json.loads((self.root/'restored_run/semantic_artifact_receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(receipt['new_interpretation_API_requests'],0)

    def test_solve_child_receives_cache_flag_and_local_planner_controls(self):
        from unified_agent import solve_training
        child=Mock();child.wait.return_value=0
        with patch('subprocess.Popen',return_value=child) as popen:
            code=solve_training(self.path,self.root/'solve',prediction_input=self.path,self_review='protected',
                feedback_reflection='opening-local',semantic_artifact_from=self.artifact)
        self.assertEqual(code,0);flags=popen.call_args.args[0]
        self.assertEqual(flags[flags.index('--semantic-artifact-from')+1],str(self.artifact.resolve()))
        self.assertIn('--protected-self-review',flags)

    def test_legacy_does_not_silently_ignore_cache_option(self):
        args=build_parser().parse_args(['unified',str(self.path),'--run-dir',str(self.root/'legacy'),
                                      '--policy','legacy','--semantic-artifact-from',str(self.artifact)])
        with self.assertRaisesRegex(ValueError,'autonomous policy'):args.handler(args)


if __name__=='__main__':unittest.main()
