import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from superagent.action_context import summarize_action,MAX_BYTES
from superagent.recovery import RecoveryController
from superagent.semantic_contract import semantic_training
from tests.test_semantic_contract import Client,fixture,proposal


class ActionContextTests(unittest.TestCase):
    def test_scalar_plan_keeps_failed_selection_and_mapping(self):
        value=proposal();value['plan']['target']='wrong_target'
        value['plan']['prediction']['field_map']={'signal':'sensor'}
        summary=summarize_action(value)
        self.assertEqual(summary['selected']['target'],'wrong_target')
        self.assertEqual(summary['selected']['prediction']['field_map'],{'signal':'sensor'})
        self.assertEqual(summary['selected']['training']['member'],'ledger.data')
        self.assertTrue(summary['not_executable'])

    def test_unknown_payload_rows_secret_and_code_are_not_copied(self):
        value=proposal();value['rows']=['PRIVATE_VALUE'];value['secret']='PRIVATE_VALUE'
        value['plan']['columns'][0]['values']=['PRIVATE_VALUE'];value['plan']['python']='PRIVATE_VALUE'
        text=json.dumps(summarize_action(value))
        self.assertNotIn('PRIVATE_VALUE',text);self.assertNotIn('python',text)

    def test_native_keys_and_preprocessing_are_kept(self):
        value=dict(version='semantic-native-v2',plan=dict(modality='tabular',metric='balanced_accuracy',
            training=dict(format='npz',path='data.npz',features_key='input',target_key='outcome'),
            prediction=dict(format='npy',path='query.npy')))
        selected=summarize_action(value)['selected']
        self.assertEqual(selected['training']['target_key'],'outcome')
        self.assertEqual(selected['prediction']['path'],'query.npy')

    def test_output_missing_field_remains_missing_in_summary(self):
        value=dict(version='semantic-output-v2',output=dict(columns=[dict(name='ID',kind='input_id')]))
        column=summarize_action(value)['selected_output']['columns'][0]
        self.assertEqual(column,{'name':'ID','kind':'input_id'})

    def test_declared_output_mapping_is_preserved_without_extra_cell_values(self):
        value=dict(version='semantic-output-v2',output=dict(columns=[dict(name='Decision',kind='label',mapping={'B':'ok','M':'alert'},rows=['PRIVATE'])]))
        text=summarize_action(value)['selected_output']['columns'][0]
        self.assertEqual(text['mapping'],{'B':'ok','M':'alert'});self.assertNotIn('PRIVATE',json.dumps(text))

    def test_row_role_rules_and_evidence_citations_are_preserved(self):
        value=proposal();summary=summarize_action(value)
        self.assertEqual(summary['selected']['split']['values']['train'],'learn')
        self.assertEqual(summary['evidence_refs']['target'],value['evidence']['target'])

    def test_capped_evidence_list_is_marked_as_truncated(self):
        value=proposal();value['evidence']['target']=[str(i) for i in range(10)]
        summary=summarize_action(value)
        self.assertTrue(summary['truncated']);self.assertEqual(len(summary['evidence_refs']['target']),4)

    def test_thirty_feature_mapping_is_not_silently_dropped(self):
        value=proposal();value['plan']['columns']=[dict(name=f'x{i}',type='numeric') for i in range(30)]
        value['plan']['prediction']['field_map']={f'x{i}':f'z{i}' for i in range(30)}
        summary=summarize_action(value)
        self.assertEqual(len(summary['selected']['prediction']['field_map']),30)
        self.assertFalse(summary['truncated'])

    def test_byte_cap_and_explicit_truncation(self):
        value=proposal();value['plan']['prediction']['field_map']={str(i)+'界'*180:'界'*180 for i in range(64)}
        summary=summarize_action(value)
        self.assertLessEqual(len(json.dumps(summary,ensure_ascii=False).encode()),MAX_BYTES)
        self.assertTrue(summary['truncated']);self.assertTrue(summary['not_executable'])

    def test_nonfinite_values_are_not_serialized_as_json_nan(self):
        value=proposal();value['plan']['positive_label']=float('nan')
        json.dumps(summarize_action(value),allow_nan=False)

    def test_malformed_envelopes_do_not_crash_recovery_journaling(self):
        for value in [[],{'action':[]},{'version':{}},{'action':'inspect_batch','requests':None},'private rows']:
            summary=summarize_action(value)
            self.assertLessEqual(len(json.dumps(summary).encode()),MAX_BYTES)
            self.assertNotIn('private rows',json.dumps(summary))

    def test_two_latest_summaries_in_prompt_with_older_hashes_preserved(self):
        controller=RecoveryController()
        for i in range(4):controller.failure('contract_probe',dict(code='bad'),action=dict(candidate=str(i)))
        records=controller.context()['failed_actions']
        self.assertTrue(all('action_summary' not in r for r in records[:2]))
        self.assertTrue(all('action_summary' in r for r in records[2:]))
        self.assertTrue(all('action_sha256' in r for r in records))
        self.assertTrue(all('action_summary' in r for r in controller.state['failed_actions']))

    def test_summary_and_diagnostic_bound_to_the_actual_proposal(self):
        value=proposal();controller=RecoveryController()
        controller.failure('contract_probe',{'code':'ContractProbeError','message':'Target field is absent'},action=value)
        record=controller.context()['failed_actions'][-1]
        self.assertEqual(record['action_sha256'],hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest())
        self.assertIn('prediction',record['expected_check'])
        value['plan']['target']='mutated later'
        self.assertEqual(record['action_summary']['selected']['target'],'decision_code')

    def test_real_semantic_loop_delivers_previous_action_to_next_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=fixture(Path(tmp));bad=proposal();bad['plan']['target']='absent'
            client=Client([bad,proposal()]);task=semantic_training(path,client,None,ValueError('native'))
            previous=client.payloads[1]['recovery_state']['failed_actions'][-1]
            self.assertEqual(previous['action_summary']['selected']['target'],'absent')
            self.assertIn('Target field is absent',previous['diagnostic']['message'])
            self.assertIn('previous unsuccessful choice',client.payloads[1]['recovery_instruction'])
            self.assertEqual(len(task.y_train),8);self.assertEqual(client.request_count,2)

    def test_method_candidate_summary_uses_only_registered_fields(self):
        summary=summarize_action({'candidate':'numeric_method','split':'primary'})
        self.assertEqual(summary['selected'],{'candidate':'numeric_method','split':'primary'})


if __name__=='__main__':unittest.main()
