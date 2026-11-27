"""Executor-owned applicability, never a text whitelist or model-declared risk level."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.events import RunLogger
from superagent.format_sources import package_files
from superagent.native_semantics import compile_native
from superagent.semantic_artifact import load_artifact,task_digest,digest
from superagent.semantic_contract import compile_plan,evidence_workspace,resolve_training_task,_diagnostic
from tests.test_auto_semantics import standard_case
from tests.test_native_semantics import fixture,binary_array
from tests.test_semantic_contract import Client


def note(field='positive_label',text='The positive class is not specified.'):
    return dict(field=field,question=text,evidence_refs=['doc:README.md:0'])


class ScopedQuestionsTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.path,self.query,self.answer,_=fixture(self.root)
        self.answer.update(version='semantic-native-v2',uncertainties=[note()])

    def native(self,answer=None):
        with package_files(self.path) as train,package_files(self.query) as query:
            return compile_native(train,evidence_workspace(train),answer or self.answer,prediction_files=query)

    def test_nonpositive_binary_metric_can_ignore_a_scoped_positive_question(self):
        task=self.native();item=task.spec['uncertainty_resolution']['items'][0]
        self.assertEqual(item['status'],'not_required_for_compiled_task')
        self.assertEqual(item['reason'],'metric_does_not_require_positive_class_orientation')
        self.assertEqual(item['facts']['class_count'],2)
        self.assertFalse(task.spec['uncertainty_resolution']['semantic_truth_certified'])

    def test_actual_multiclass_domain_makes_single_positive_irrelevant(self):
        with zipfile.ZipFile(self.path) as archive:members={n:archive.read(n) for n in archive.namelist()}
        members['decisions.bin']=binary_array(np.array(['a','b','c','a','b','c','a','b']))
        with zipfile.ZipFile(self.path,'w') as archive:
            for name,value in members.items():archive.writestr(name,value)
        task=self.native();item=task.spec['uncertainty_resolution']['items'][0]
        self.assertEqual(item['reason'],'multiclass_metric_has_no_single_positive_class')
        self.assertEqual(item['facts']['class_count'],3)

    def test_decision_does_not_depend_on_matching_the_free_text_language(self):
        for question in ['正类没有给出。','Positive class unknown.','Classe positive non précisée.']:
            answer=deepcopy(self.answer);answer['uncertainties']=[note(text=question)]
            with self.subTest(question=question):self.assertEqual(len(self.native(answer).y_train),8)

    def test_binary_roc_and_f1_positive_questions_are_blocking(self):
        for metric in ['roc_auc','binary_f1']:
            answer=deepcopy(self.answer);answer['plan']['metric']=metric
            with self.subTest(metric=metric),self.assertRaisesRegex(ValueError,'positive_label'):
                self.native(answer)

    def test_questioned_positive_label_is_not_ignored_when_plan_asserts_one(self):
        answer=deepcopy(self.answer);answer['plan']['positive_label']='a';answer['evidence']['positive_label']=['doc:README.md:0']
        with self.assertRaisesRegex(ValueError,'positive_label'):self.native(answer)

    def test_critical_field_questions_block_before_native_data_read(self):
        for field in ['training','prediction','target','metric','output','columns','split','modality']:
            answer=deepcopy(self.answer);answer.update(plan=None,uncertainties=[note(field)])
            with package_files(self.path) as train,package_files(self.query) as query:
                workspace=evidence_workspace(train)
                with self.subTest(field=field),patch.object(train,'read',side_effect=AssertionError('No row read')) as reader:
                    with self.assertRaisesRegex(ValueError,field):compile_native(train,workspace,answer,prediction_files=query)
                    reader.assert_not_called()

    def test_image_preprocessing_question_is_only_irrelevant_for_non_image_task(self):
        answer=deepcopy(self.answer);answer['uncertainties']=[note('image_preprocessing')]
        item=self.native(answer).spec['uncertainty_resolution']['items'][0]
        self.assertEqual(item['reason'],'decoded_task_is_not_an_image_task')
        image=self.root/'images';image.mkdir();path,query,response,_=fixture(image,'image_folder')
        response.update(version='semantic-native-v2',uncertainties=[note('image_preprocessing')])
        with package_files(path) as train,package_files(query) as prediction:
            with self.assertRaisesRegex(ValueError,'image_preprocessing'):
                compile_native(train,evidence_workspace(train),response,prediction_files=prediction)

    def test_model_cannot_add_a_nonblocking_boolean_or_fake_citation(self):
        for changes in [dict(blocking=False),dict(severity='low'),dict(evidence_refs=['invented'])]:
            answer=deepcopy(self.answer);answer['uncertainties'][0].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.native(answer)

    def test_duplicate_unknown_unbounded_and_empty_questions_are_rejected(self):
        variants=[[note(),note()], [note('candidate_choice')], [note(text='')], [note(text='x'*1001)], [note()]*9]
        for values in variants:
            answer=deepcopy(self.answer);answer['uncertainties']=values
            with self.subTest(values=values),self.assertRaises(ValueError):self.native(answer)

    def test_new_version_free_text_and_old_version_scoped_objects_are_not_silently_upgraded(self):
        for version,values in [('semantic-native-v2',['Positive not specified']),('semantic-native-v1',[note()]),
                               ('semantic-native-v1',['No positive class is specified'])]:
            answer=deepcopy(self.answer);answer.update(version=version,uncertainties=values)
            with self.subTest(version=version,values=values),self.assertRaises(ValueError):self.native(answer)

    def test_legacy_empty_questions_keep_the_legacy_version_identity(self):
        answer=deepcopy(self.answer);answer.update(version='semantic-native-v1',uncertainties=[])
        task=self.native(answer)
        self.assertNotIn('uncertainty_resolution',task.spec)
        self.assertEqual(task.spec['semantic_interpretation']['version'],'semantic-native-v1')

    def test_scalar_and_manifest_overlay_use_the_same_question_engine(self):
        for explicit,version in [(False,'semantic-task-v3'),(True,'semantic-output-v2')]:
            path,answer,expected=standard_case(self.root/f'{explicit}.zip',explicit)
            answer.update(version=version,uncertainties=[note()]);client=Client([answer])
            task=resolve_training_task(path,client,mode='auto',prediction_package=path)
            self.assertEqual(task.spec['output_recipe'],expected)
            self.assertEqual(task.spec['uncertainty_resolution']['items'][0]['field'],'positive_label')

    def test_scalar_critical_question_blocks_before_row_read(self):
        path,answer,_=standard_case(self.root/'scalar.zip');answer.update(version='semantic-task-v3',plan=None,uncertainties=[note('target')])
        with package_files(path) as files:
            workspace=evidence_workspace(files)
            with patch.object(files,'read',side_effect=AssertionError('No row read')) as reader:
                with self.assertRaisesRegex(ValueError,'target'):compile_plan(files,workspace,answer)
                reader.assert_not_called()

    def test_diagnostic_retains_field_and_machine_dependency_facts(self):
        answer=deepcopy(self.answer);answer['plan']['metric']='roc_auc'
        try:self.native(answer)
        except ValueError as error:
            diagnostic=_diagnostic(error,'contract_probe')
        self.assertEqual(diagnostic['uncertainty_fields'],['positive_label'])
        self.assertEqual(diagnostic['uncertainty_dependency']['metric'],'roc_auc')
        self.assertEqual(diagnostic['uncertainty_dependency']['class_count'],2)

    def test_artifact_roundtrips_all_three_new_response_versions(self):
        cases=[(self.path,self.query,self.answer)]
        for explicit,version in [(False,'semantic-task-v3'),(True,'semantic-output-v2')]:
            path,answer,_=standard_case(self.root/f'case_{version}.zip',explicit)
            answer.update(version=version,uncertainties=[note()]);cases.append((path,path,answer))
        for i,(path,query,answer) in enumerate(cases):
            run=self.root/f'run_{i}';run.mkdir()
            task=resolve_training_task(path,Client([answer]),RunLogger(run),mode='proactive',prediction_package=query)
            restored=load_artifact(run/'semantic_artifact.json',path,query)
            self.assertEqual(task_digest(task),task_digest(restored))
            self.assertEqual(restored.spec['uncertainty_resolution'],task.spec['uncertainty_resolution'])

    def test_changed_question_cannot_hide_behind_recomputed_envelope_hash(self):
        run=self.root/'cache';run.mkdir()
        resolve_training_task(self.path,Client([self.answer]),RunLogger(run),mode='proactive',prediction_package=self.query)
        path=run/'semantic_artifact.json';record=json.loads(path.read_text(encoding='utf-8'))
        record['payload']['proposal']['uncertainties'][0]['question']='A different question'
        record['payload_sha256']=digest(record['payload']);path.write_text(json.dumps(record),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'accepted digest'):load_artifact(path,self.path,self.query)

    def test_one_model_reply_is_sufficient_for_an_inapplicable_question(self):
        run=self.root/'one';run.mkdir();client=Client([self.answer],limit=1)
        task=resolve_training_task(self.path,client,RunLogger(run),mode='proactive',prediction_package=self.query)
        receipt=json.loads((run/'semantic_contract.json').read_text(encoding='utf-8'))
        self.assertEqual(client.request_count,1);self.assertEqual(receipt['attempt'],1)
        self.assertEqual(receipt['uncertainty_resolution'],task.spec['uncertainty_resolution'])


if __name__=='__main__':unittest.main()
