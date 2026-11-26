"""SA3: simulated decisions, real package probes, no fitting or network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.format_intake import predict_records
from superagent.format_sources import UnsupportedFormat, package_files
from superagent.semantic_contract import (
    ContractProbeError, _inspect_header, compile_plan, evidence_workspace,
    probe_prediction_input, resolve_training_task)
from tests.test_semantic_contract import Client, fixture, proposal
from tests.test_output_recipe import package, recipe


def rewrite(path, changes, remove=()):
    with zipfile.ZipFile(path) as archive:
        members = {n: archive.read(n) for n in archive.namelist() if n not in remove}
    members.update(changes)
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in members.items(): archive.writestr(name, value)


def output_proposal():
    answer = proposal()
    answer['version'] = 'semantic-task-v2'
    answer['plan']['output'] = recipe()
    answer['evidence']['output'] = ['doc:02_dictionary.md:0']
    return answer


def inspection(member='ledger.data'):
    return dict(action='inspect_header', member=member, collection=None,
                delimiter=',', evidence_refs=['doc:01_overview.md:0'])


class ProactiveProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = package(self.root/'task.zip')

    def resolve(self, replies, **kwargs):
        client = Client(replies)
        task = resolve_training_task(self.path, client, mode='proactive', **kwargs)
        return task, client

    def test_proactive_does_not_skip_model_when_native_would_succeed(self):
        task = load_training_task  # Native reader is mocked to prove the control-flow distinction.
        with patch('superagent.semantic_contract.load_training_task', return_value=task) as native:
            result, client = self.resolve([output_proposal()])
        native.assert_not_called()
        self.assertEqual(client.request_count, 1)
        self.assertEqual(result.spec['output_recipe'], recipe())

    def test_real_native_default_is_replaced_by_documented_semantic_plan(self):
        path = self.root/'native.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('README.md', 'The target is label. Accuracy evaluates this classification. '
                'Use signal only. Predict test.csv. Output Decision is Y for accept and N for decline, no ID.')
            archive.writestr('train.csv', 'signal,label\n1,decline\n9,accept\n2,decline\n8,accept\n')
            archive.writestr('test.csv', 'signal\n9\n2\n')
        native = load_training_task(path, read_declarations=True)
        self.assertNotIn('output_recipe', native.spec)
        answer = output_proposal(); plan = answer['plan']
        plan.update(training=dict(member='train.csv', collection=None, delimiter=','),
                    prediction=dict(member='test.csv', collection=None, delimiter=','),
                    target='label', metric='accuracy', split=None)
        plan['output']['columns'] = [dict(name='Decision', kind='label', mapping={'accept':'Y','decline':'N'})]
        answer['evidence'] = {k:['doc:README.md:0'] for k in answer['evidence']}
        client = Client([answer])
        task = resolve_training_task(path, client, mode='proactive')
        self.assertEqual(task.spec['output_recipe']['columns'][0]['name'], 'Decision')
        self.assertEqual(task.spec['metric'], 'accuracy')
        self.assertEqual(client.request_count, 1)

    def test_field_alias_replays_without_changing_canonical_training_schema(self):
        rewrite(self.path, {'query.data':'event_id,sensor\n0001,9\n0010,2\n'})
        answer = output_proposal(); answer['plan']['prediction']['field_map'] = {'signal':'sensor'}
        task, _ = self.resolve([answer])
        self.assertEqual(task.spec['input_schema']['columns'], [dict(name='signal',type='numeric')])
        self.assertEqual(list(map(float,predict_records(self.path,task.kind,task.spec)[:,0])), [9.,2.])
        self.assertEqual(list(map(float,task.X_train[:,0])), list(map(float,range(8))))

    def test_external_prediction_package_has_its_own_member_namespace(self):
        rewrite(self.path, {}, remove=['query.data','sample.csv'])
        query = self.root/'later.zip'
        with zipfile.ZipFile(query, 'w') as archive:
            archive.writestr('later.data', 'event_id,sensor\n0001,9\n0010,2\n')
        answer = output_proposal()
        answer['plan']['prediction'].update(member='later.data',field_map={'signal':'sensor'})
        (self.root/'probe').mkdir()
        log = RunLogger(self.root/'probe')
        client = Client([answer])
        task = resolve_training_task(self.path,client,log,mode='proactive',prediction_package=query)
        self.assertEqual(list(map(float,predict_records(query,task.kind,task.spec)[:,0])), [9.,2.])
        payload = client.payloads[0]['evidence_workspace']
        self.assertEqual(payload['prediction_members'], [dict(name='later.data',bytes=30)])
        report = json.loads((self.root/'probe/prediction_preflight.json').read_text(encoding='utf-8'))
        self.assertTrue(report['output']['original_ids_checked'])

    def test_missing_feature_diagnostic_can_be_repaired_before_fitting(self):
        rewrite(self.path, {'query.data':'event_id,sensor\n0001,9\n0010,2\n'})
        good = output_proposal(); good['plan']['prediction']['field_map'] = {'signal':'sensor'}
        task, client = self.resolve([output_proposal(),good])
        self.assertEqual(client.request_count,2)
        self.assertIn('missing',client.payloads[1]['diagnostic']['message'])
        self.assertEqual(task.spec['input_recipe']['field_map'],{'signal':'sensor'})

    def test_external_prediction_header_tool_uses_explicit_source(self):
        query=self.root/'query.zip'
        with zipfile.ZipFile(query,'w') as archive:
            archive.writestr('later.data','event_id,sensor\nPRIVATE_ID,9\nOTHER_PRIVATE_ID,2\n')
        request=inspection('later.data'); request['source']='prediction'
        answer=output_proposal()
        answer['plan']['prediction'].update(member='later.data',field_map={'signal':'sensor'})
        task,client=self.resolve([request,answer],prediction_package=query)
        observed=client.payloads[1]['tool_observations'][0]
        self.assertEqual(observed['source'],'prediction')
        self.assertEqual(observed['fields'],['event_id','sensor'])
        self.assertNotIn('PRIVATE_ID',json.dumps(client.payloads))
        self.assertEqual(probe_prediction_input(query,task)['rows'],2)

    def test_missing_id_is_found_before_standby_or_executor(self):
        bad = output_proposal(); bad['plan']['output']['columns'][0]['field'] = 'absent_id'
        client = Client([bad,output_proposal()]); run = self.root/'official'
        from superagent.autonomous_run import _train
        class CheckpointReached(RuntimeError): pass
        with patch('superagent.autonomous_run.standby',side_effect=CheckpointReached) as checkpoint:
            with patch('superagent.autonomous_run.Executor.worker',side_effect=AssertionError('fit forbidden')):
                with self.assertRaises(CheckpointReached):
                    _train(self.path,run,fit_limit=2,flash=client,interpretation='proactive')
        self.assertEqual(checkpoint.call_count,1)
        self.assertEqual(client.request_count,2)
        self.assertIn('ID field missing',client.payloads[1]['diagnostic']['message'])
        report = json.loads((run/'prediction_preflight.json').read_text(encoding='utf-8'))
        self.assertTrue(report['performed_before_fitting'])
        self.assertTrue(report['output']['original_ids_checked'])

    def test_reader_error_cannot_echo_private_cell_into_repair_prompt(self):
        rewrite(self.path,{'query.data':'event_id,signal,sensor\n0001,PRIVATE_CELL,9\n0010,PRIVATE_CELL,2\n'})
        good=output_proposal(); good['plan']['prediction']['field_map']={'signal':'sensor'}
        task,client=self.resolve([output_proposal(),good])
        self.assertEqual(client.request_count,2)
        self.assertNotIn('PRIVATE_CELL',json.dumps(client.payloads))
        self.assertEqual(probe_prediction_input(self.path,task)['rows'],2)

    def test_logging_io_failure_does_not_consume_another_model_call(self):
        client=Client([output_proposal(),output_proposal()]); log=Mock()
        log.write_json.side_effect=OSError('disk full')
        with self.assertRaisesRegex(OSError,'disk full'):
            resolve_training_task(self.path,client,log,mode='proactive')
        self.assertEqual(client.request_count,1)

    def test_template_id_mismatch_is_pre_fit_failure(self):
        answer = output_proposal(); answer['plan']['output'] = recipe(template=True)
        rewrite(self.path, {'sample.csv':'Ticket;Answer\nwrong;\n0010;\n'})
        client = Client([answer,answer])
        with self.assertRaises(UnsupportedFormat):
            resolve_training_task(self.path,client,mode='proactive')
        self.assertIn('ID sets differ',client.payloads[1]['diagnostic']['message'])
        self.assertEqual(client.request_count,2)

    def test_header_tool_then_plan_shares_client_and_does_not_disclose_values(self):
        task, client = self.resolve([inspection(),output_proposal()])
        self.assertEqual(client.request_count,2)
        observed = client.payloads[1]['tool_observations']
        self.assertEqual(observed[0]['fields'],['signal','decision_code','cohort'])
        self.assertFalse(observed[0]['row_values_disclosed'])
        self.assertNotIn('0001',json.dumps(client.payloads))
        self.assertEqual(len(task.y_train),8)

    def test_tool_and_bad_plan_leave_only_one_shared_repair_request(self):
        bad = output_proposal(); bad['plan']['target'] = 'absent'
        task,client = self.resolve([inspection(),bad,output_proposal()])
        self.assertEqual(client.request_count,3)
        self.assertEqual(task.spec['input_schema']['target'],'decision_code')
        limited = Client([inspection(),bad,output_proposal()],limit=2)
        with self.assertRaises(UnsupportedFormat):
            resolve_training_task(self.path,limited,mode='proactive')
        self.assertEqual(limited.request_count,2)

    def test_two_tools_cannot_request_a_third_tool(self):
        client = Client([inspection(),inspection('query.data'),inspection()])
        with self.assertRaises(UnsupportedFormat):
            resolve_training_task(self.path,client,mode='proactive')
        self.assertEqual(client.request_count,3)

    def test_known_evaluation_and_label_vector_tools_reject_before_member_read(self):
        rewrite(self.path, {'validation/answers.csv':'secret', 'answers.csv':'secret'})
        with package_files(self.path) as files:
            workspace = evidence_workspace(files)
            for member in ['validation/answers.csv','answers.csv']:
                with patch.object(files,'read',side_effect=AssertionError('answer opened')) as read:
                    with self.assertRaises(ContractProbeError): _inspect_header(files,workspace,inspection(member))
                read.assert_not_called()

    def test_forged_tool_citation_rejects_before_member_read(self):
        with package_files(self.path) as files:
            workspace = evidence_workspace(files); request = inspection(); request['evidence_refs']=['invented']
            with patch.object(files,'read',side_effect=AssertionError('unauthorized read')):
                with self.assertRaisesRegex(ContractProbeError,'citations'): _inspect_header(files,workspace,request)

    def test_external_json_prediction_collection_can_be_inspected(self):
        rewrite(self.path,{'query.json':json.dumps({'test':[dict(sensor=9,event_id='PRIVATE_ID')]})})
        with package_files(self.path) as files:
            workspace=evidence_workspace(files); request=inspection('query.json')
            request.update(source='prediction',collection='test',delimiter=None)
            result=_inspect_header(files,workspace,request)
        self.assertEqual(result['fields'],['sensor','event_id'])
        self.assertNotIn('PRIVATE_ID',json.dumps(result))

    def test_evaluation_collection_is_refused_before_file_read(self):
        with package_files(self.path) as files:
            workspace=evidence_workspace(files)
            for collection in ['validation','answers','y']:
                request=inspection(); request['collection']=collection
                with patch.object(files,'read',side_effect=AssertionError('evaluation file opened')) as read:
                    with self.assertRaises(ContractProbeError): _inspect_header(files,workspace,request)
                read.assert_not_called()

    def test_prediction_cannot_select_known_validation_answers(self):
        rewrite(self.path, {'validation/answers.csv':'signal\n9\n'})
        answer = output_proposal(); answer['plan']['prediction']['member']='validation/answers.csv'
        with package_files(self.path) as files:
            workspace = evidence_workspace(files)
            with patch.object(files,'read',side_effect=AssertionError('answer opened')):
                with self.assertRaises(ContractProbeError): compile_plan(files,workspace,answer)

    def test_role_fields_cannot_be_mapped_into_predictors(self):
        for source in ['decision_code','cohort']:
            answer=output_proposal(); answer['plan']['prediction']['field_map']={'signal':source}
            with package_files(self.path) as files:
                with self.assertRaisesRegex(ContractProbeError,'target/split'):
                    compile_plan(files,evidence_workspace(files),answer)

    def manifest_package(self):
        path=self.root/'manifest.zip'
        manifest=dict(schema_version=1,modality='tabular',metric='accuracy',target='label',
            columns=[dict(name='signal',type='numeric')],splits=dict(
                train=dict(format='csv',path='train.csv'),test=dict(format='csv',path='test.csv')))
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('manifest.json',json.dumps(manifest))
            archive.writestr('README.md','Output Ticket from event_id, and Answer Y for accept or N for decline, semicolon BOM.')
            archive.writestr('train.csv','signal,label\n1,decline\n9,accept\n2,decline\n8,accept\n')
            archive.writestr('test.csv','signal,event_id\n9,0001\n2,0010\n')
        return path

    def test_manifest_allows_output_overlay_but_preserves_target_and_metric(self):
        path=self.manifest_package(); client=Client([dict(version='semantic-output-v1',output=recipe(),
            evidence_refs=['doc:README.md:0'],uncertainties=[])])
        task=resolve_training_task(path,client,mode='proactive',prediction_package=path)
        self.assertEqual(task.spec['input_schema']['target'],'label')
        self.assertEqual(task.spec['metric'],'accuracy')
        self.assertNotIn('input_recipe',task.spec)
        self.assertTrue(probe_prediction_input(path,task)['output']['original_ids_checked'])
        self.assertEqual(client.payloads[0]['authoritative_contract']['metric']['name'],'accuracy')

    def test_manifest_mutation_is_rejected_instead_of_silently_applied(self):
        path=self.manifest_package(); client=Client([output_proposal(),output_proposal()])
        with self.assertRaises(UnsupportedFormat): resolve_training_task(path,client,mode='proactive')
        self.assertIn('only an output overlay',client.payloads[1]['diagnostic']['message'])

    def test_declared_jsonl_contract_keeps_existing_adaptive_reader(self):
        path=self.manifest_package()
        with zipfile.ZipFile(path) as archive: manifest=json.loads(archive.read('manifest.json'))
        manifest['splits']=dict(train=dict(format='jsonl',path='train.jsonl'),
                                test=dict(format='jsonl',path='test.jsonl'))
        train='\n'.join(json.dumps(dict(signal=i,label='accept' if i%2 else 'decline')) for i in range(8))
        query='\n'.join(json.dumps(dict(signal=i,event_id=identity)) for i,identity in [(9,'0001'),(2,'0010')])
        rewrite(path,{'manifest.json':json.dumps(manifest),'train.jsonl':train,'test.jsonl':query},
                remove=['train.csv','test.csv'])
        client=Client([dict(version='semantic-output-v1',output=recipe(),
                           evidence_refs=['doc:README.md:0'],uncertainties=[])])
        task=resolve_training_task(path,client,mode='proactive',prediction_package=path)
        self.assertEqual(task.spec['input_schema']['target'],'label')
        self.assertEqual(task.spec['metric'],'accuracy')
        self.assertTrue(probe_prediction_input(path,task)['output']['original_ids_checked'])

    def test_proactive_without_client_does_not_silently_use_native(self):
        with self.assertRaisesRegex(UnsupportedFormat,'configured model'):
            resolve_training_task(self.manifest_package(),mode='proactive')

    def test_transport_failure_is_not_retried_after_tool(self):
        client=Client([inspection(),RuntimeError('stream disconnected'),output_proposal()])
        with self.assertRaisesRegex(RuntimeError,'stream disconnected'):
            resolve_training_task(self.path,client,mode='proactive')
        self.assertEqual(client.request_count,2)

    def test_fallback_still_uses_native_without_asking(self):
        client=Client([output_proposal()])
        task=resolve_training_task(self.manifest_package(),client)
        self.assertNotIn('output_recipe',task.spec)
        self.assertEqual(client.request_count,0)

    def test_solve_passes_mode_and_prediction_path_to_child_without_training(self):
        from unified_agent import solve_training
        query=self.root/'predict.zip'; run=self.root/'solve'
        child=Mock(); child.wait.return_value=0
        with patch('subprocess.Popen',return_value=child) as popen:
            result=solve_training(self.path,run,task_interpretation='proactive',prediction_input=query)
        self.assertEqual(result,0)
        flags=popen.call_args.args[0]
        self.assertEqual(flags[flags.index('--task-interpretation')+1],'proactive')
        self.assertEqual(flags[flags.index('--prediction-input')+1],str(query.resolve()))
        protocol=json.loads((run/'training_protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(protocol['task_interpretation'],'proactive')

    def test_unified_legacy_cannot_silently_ignore_proactive(self):
        from superagent.cli import build_parser
        args=build_parser().parse_args(['unified',str(self.path),'--run-dir',str(self.root/'legacy'),
                                      '--policy','legacy','--task-interpretation','proactive'])
        with self.assertRaisesRegex(ValueError,'autonomous policy'): args.handler(args)


if __name__=='__main__': unittest.main()
