from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import joblib

from superagent.autonomous_checkpoint import frozen_spec
from superagent.contracts import TaskSpec
from superagent.delivery_recovery import recover_prediction,bound_client,find_cached
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.prediction_adapter import apply_proposal,write_adapter,load_adapter,sha
from superagent.predict import create_predictions,PredictionAdaptationError
from superagent.semantic_contract import compile_plan,evidence_workspace,ContractProbeError
from superagent.format_sources import package_files
from superagent.task_budget import TaskBudgetLedger
from tests.test_semantic_contract import proposal,Client
from tests.test_output_recipe import ProbeClassifier,package,recipe


def adapter():
    return dict(version='prediction-adapter-v1',input=dict(format='records',member='arrival.csv',collection=None,delimiter=',',
        field_map={'signal':'sensor'}),id_fields={'Ticket':'parcel_id'},evidence_refs=['prediction:doc:README.md:0'])


class DeliveryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.task=package(self.root/'task.zip');self.query=self.root/'later.zip'
        with zipfile.ZipFile(self.query,'w') as z:
            z.writestr('README.md','arrival.csv contains sensor corresponding to the original signal; parcel_id is the unchanged original identifier.')
            z.writestr('arrival.csv','parcel_id,sensor\n0001,9\n0010,2\n')
        reply=proposal();reply['version']='semantic-task-v2';reply['plan']['output']=recipe();reply['evidence']['output']=['doc:02_dictionary.md:0']
        with package_files(self.task) as files:task=compile_plan(files,evidence_workspace(files),reply)
        metric=MetricSpec.for_task(task);self.spec=frozen_spec(task,TaskSpec.from_task(task).record(),metric)
        self.run=self.root/'run';self.model=self.run/'agent';self.model.mkdir(parents=True);log=RunLogger(self.model)
        log.write_json('task_spec.json',self.spec);joblib.dump(ProbeClassifier(),self.model/'model.joblib')
        self.selection=dict(model_file='model.joblib',model_format='bare_model',model_sha256=sha(self.model/'model.joblib'),
            task_spec_sha256=sha(self.model/'task_spec.json'),family='fixed_untrained_fixture',selected_experiment='fixture',refit=True,test_accessed=False)
        log.write_json('final_selection.json',self.selection);log.write_json('classifier_fits.json',{'attempted_classifier_fits':0})
        self.ledger=TaskBudgetLedger.initialize(self.root/'synthetic.sqlite',new_project=True)
        self.protocol=dict(task_package=str(self.task),task_package_sha256=sha(self.task),task_budget_id='fixture',budget_ledger=str(self.ledger.path))
        (self.run/'training_protocol.json').write_text(json.dumps(self.protocol),encoding='utf-8')
        self.env=dict(SUPERAGENT_TASK_BUDGET_ID='fixture',SUPERAGENT_PLANNER_LEDGER=str(self.ledger.path))

    def test_repair_and_cross_process_cached_prediction_without_training(self):
        original=self.root/'original.csv';create_predictions(self.task,self.model,original)
        with self.assertRaises(PredictionAdaptationError) as failure:create_predictions(self.query,self.model,self.root/'first.csv')
        request=dict(action='inspect_header',member='arrival.csv',source='prediction',collection=None,delimiter=',',evidence_refs=['prediction:doc:README.md:0'])
        client=Client([request,adapter()]);client.spent=0
        before={n:sha(self.model/n) for n in ['model.joblib','task_spec.json','final_selection.json','classifier_fits.json']}
        with patch('superagent.delivery_recovery.bound_client',return_value=(client,0)):
            path=recover_prediction(self.query,self.run,self.root/'fixed.csv',failure.exception)
        self.assertIsNotNone(path);self.assertEqual(client.request_count,2)
        create_predictions(self.query,self.model,self.root/'fixed.csv',prediction_adapter=path)
        self.assertEqual(original.read_bytes(),(self.root/'fixed.csv').read_bytes())
        self.assertEqual(before,{n:sha(self.model/n) for n in before})
        self.assertEqual(find_cached(self.query,self.run),path)
        env={k:v for k,v in os.environ.items() if not k.startswith(('SUPERAGENT_PLANNER_','SUPERAGENT_ACCEPTED_'))};env['SUPERAGENT_OFFLINE']='1'
        result=subprocess.run([sys.executable,'-m','unified_agent','predict',str(self.query),'--run-dir',str(self.run),
            '--output',str(self.root/'replayed.csv')],env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(original.read_bytes(),(self.root/'replayed.csv').read_bytes())
        second=subprocess.run([sys.executable,'-m','unified_agent','predict',str(self.query),'--run-dir',str(self.run),
            '--output',str(self.root/'replayed_again.csv')],env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=30)
        self.assertEqual(second.returncode,0,second.stderr)
        self.assertEqual(original.read_bytes(),(self.root/'replayed_again.csv').read_bytes())

    def test_unique_native_candidate_uses_shape_not_a_task_name(self):
        import io
        import numpy as np
        from superagent.delivery_recovery import unique_native_proposal
        from superagent.semantic_contract import evidence_workspace
        data=io.BytesIO();np.savez(data,opaque=np.zeros((3,1)))
        query=self.root/'native.zip'
        with zipfile.ZipFile(query,'w') as z:z.writestr('late.npz',data.getvalue());z.writestr('README.md','The sole array is the new feature input.')
        spec=dict(feature_shape=[1],native_input_recipe=dict(modality='tabular',training={'format':'npy'},prediction={'format':'npy'}))
        with package_files(query) as files:
            workspace=evidence_workspace(files)
            workspace['documents']=[dict(d,id='prediction:'+d['id']) for d in workspace['documents']]
            result=unique_native_proposal(files,workspace,spec)
            self.assertEqual(result[0]['input']['features_key'],'opaque')
            spec['feature_shape']=[2]
            self.assertIsNone(unique_native_proposal(files,workspace,spec))

    def test_multiple_arrays_are_not_guessed_by_local_fallback(self):
        import io
        import numpy as np
        from superagent.delivery_recovery import unique_native_proposal
        from superagent.semantic_contract import evidence_workspace
        data=io.BytesIO();np.savez(data,a=np.zeros((3,1)),b=np.ones((3,1)))
        query=self.root/'multiple.zip'
        with zipfile.ZipFile(query,'w') as z:z.writestr('late.npz',data.getvalue());z.writestr('README.md','Two alternatives require interpretation.')
        spec=dict(feature_shape=[1],native_input_recipe=dict(modality='tabular',training={'format':'npy'}))
        with package_files(query) as files:
            workspace=evidence_workspace(files);workspace['documents']=[dict(d,id='prediction:'+d['id']) for d in workspace['documents']]
            self.assertIsNone(unique_native_proposal(files,workspace,spec))

    def test_short_remaining_window_does_not_start_another_request(self):
        clock={'now':100.0}
        class SlowClient(Client):
            def ask(self,*args,**kwargs):
                result=super().ask(*args,**kwargs);clock['now']=211.0;return result
        client=SlowClient([{'malformed':True},adapter()]);client.spent=0
        with patch('superagent.delivery_recovery.bound_client',return_value=(client,0)),patch('superagent.delivery_recovery.time.monotonic',side_effect=lambda:clock['now']):
            result=recover_prediction(self.query,self.run,self.root/'out.csv',PredictionAdaptationError('input'))
        self.assertIsNone(result);self.assertEqual(client.request_count,1)

    def test_model_corruption_is_not_an_adaptation_error(self):
        (self.model/'model.joblib').write_bytes(b'tampered')
        with self.assertRaises(ValueError) as failure:create_predictions(self.task,self.model,self.root/'out.csv')
        self.assertNotIsInstance(failure.exception,PredictionAdaptationError)

    def test_output_filesystem_error_is_not_an_adaptation_error(self):
        with patch('superagent.output_recipe.atomic_text',side_effect=PermissionError('controlled write denial')):
            with self.assertRaises(PermissionError):create_predictions(self.task,self.model,self.root/'out.csv')

    def test_adapter_changes_only_query_mapping_and_existing_id_source(self):
        before=deepcopy(self.spec);after=apply_proposal(self.spec,adapter(),{'prediction:doc:README.md:0'})
        self.assertEqual(self.spec,before)
        for key in ['input_schema','metric','metric_spec','feature_shape']:self.assertEqual(after[key],before[key])
        original=deepcopy(before['output_recipe']);original['columns'][0]['field']='parcel_id'
        self.assertEqual(after['output_recipe'],original)

    def test_cannot_change_goal_output_meaning_or_add_a_column(self):
        for key,value in [('metric','accuracy'),('target','other'),('output',{'delimiter':','})]:
            bad=adapter();bad[key]=value
            with self.assertRaises(ContractProbeError):apply_proposal(self.spec,bad,{'prediction:doc:README.md:0'})
        bad=adapter();bad['id_fields']['NewColumn']='parcel_id'
        with self.assertRaises(ContractProbeError):apply_proposal(self.spec,bad,{'prediction:doc:README.md:0'})

    def test_fabricated_citation_and_target_as_id_are_rejected(self):
        bad=adapter();bad['evidence_refs']=['invented']
        with self.assertRaises(ContractProbeError):apply_proposal(self.spec,bad,{'prediction:doc:README.md:0'})
        bad=adapter();bad['id_fields']['Ticket']='decision_code'
        with self.assertRaises(ContractProbeError):apply_proposal(self.spec,bad,{'prediction:doc:README.md:0'})

    def test_input_field_mapping_cannot_add_or_duplicate_features(self):
        bad=adapter();bad['input']['field_map']['unknown']='sensor'
        with self.assertRaises(ContractProbeError):apply_proposal(self.spec,bad,{'prediction:doc:README.md:0'})

    def test_cache_is_bound_to_query_and_frozen_schema(self):
        path=self.root/'adapter.json';write_adapter(path,self.query,self.model,adapter(),{'prediction:doc:README.md:0'},{'rows':2})
        load_adapter(path,self.query,self.model,self.spec,self.selection)
        other=self.root/'other.zip';other.write_bytes(self.query.read_bytes()+b'changed')
        with self.assertRaisesRegex(ContractProbeError,'query package changed'):load_adapter(path,other,self.model,self.spec,self.selection)
        (self.model/'task_spec.json').write_text('{}')
        with self.assertRaisesRegex(ContractProbeError,'specification changed'):load_adapter(path,self.query,self.model,self.spec,self.selection)

    def test_wrong_budget_scope_stops_before_client_setup(self):
        with patch.dict(os.environ,{**self.env,'SUPERAGENT_TASK_BUDGET_ID':'other'},clear=True),patch('superagent.unified.configured_flash') as setup:
            with self.assertRaisesRegex(ContractProbeError,'original training budget scope'):bound_client(self.run,self.root)
        setup.assert_not_called()

    def test_explicit_local_control_does_not_start_paid_recovery(self):
        self.protocol['feedback_reflection']='control'
        (self.run/'training_protocol.json').write_text(json.dumps(self.protocol),encoding='utf-8')
        with patch.dict(os.environ,self.env,clear=True),patch('superagent.unified.configured_flash') as setup:
            with self.assertRaisesRegex(ContractProbeError,'local/control'):bound_client(self.run,self.root)
        setup.assert_not_called()

    def test_scope_request_count_survives_a_new_client(self):
        for _ in range(2):
            ticket=self.ledger.reserve('fixture',.01,'test','deepseek-flash');self.ledger.settle(ticket,'deepseek-flash',10,1)
        client=Client([]);client.account=SimpleNamespace(ledger=self.ledger,max_requests=3)
        with patch.dict(os.environ,self.env,clear=True),patch('superagent.unified.configured_flash',return_value=client):
            result,used=bound_client(self.run,self.root)
        self.assertEqual(used,2);self.assertEqual(result.max_requests,1)

    def test_exhausted_scope_does_not_allow_three_more_calls(self):
        for _ in range(3):
            ticket=self.ledger.reserve('fixture',.01,'test','deepseek-flash');self.ledger.settle(ticket,'deepseek-flash',10,1)
        client=Client([]);client.account=SimpleNamespace(ledger=self.ledger,max_requests=3)
        with patch.dict(os.environ,self.env,clear=True),patch('superagent.unified.configured_flash',return_value=client):
            with self.assertRaisesRegex(ContractProbeError,'no remaining shared'):bound_client(self.run,self.root)
        self.assertEqual(client.request_count,0)


if __name__=='__main__':unittest.main()
