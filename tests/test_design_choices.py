from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from superagent.native_semantics import compile_native
from superagent.semantic_contract import evidence_workspace,resolve_training_task
from superagent.semantic_artifact import load_artifact,task_digest
from superagent.format_sources import package_files
from superagent.events import RunLogger
from tests.test_native_semantics import fixture
from tests.test_semantic_contract import Client


def upgrade(answer,constraints=None):
    result=deepcopy(answer);result['version']='semantic-native-v3'
    constraints=constraints or {};refs=['doc:README.md:0']
    result['design_decisions']=dict(image_preprocessing=dict(task_constraints=constraints,
        constraint_evidence={key:refs for key in constraints},reason='Choose a compact representation for this image task.',evidence_refs=refs)) if result['plan']['modality']=='image' else {}
    return result


class DesignChoicesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.path,self.query,self.answer,_=fixture(self.root,'image_folder')
        with zipfile.ZipFile(self.path) as z:members={n:z.read(n) for n in z.namelist()}
        members['README.md']=b'Classify gallery/ class-folder images. Predict incoming/. Macro F1 is primary. Export Row,Class in input order, Row starting at 1. Internal image representation is chosen by the agent.'
        with zipfile.ZipFile(self.path,'w') as z:
            for n,body in members.items():z.writestr(n,body)

    def compile(self,response):
        with package_files(self.path) as files,package_files(self.query) as queries:
            return compile_native(files,evidence_workspace(files),response,prediction_files=queries)

    def test_free_parameters_are_frozen_as_agent_choices(self):
        task=self.compile(upgrade(self.answer));decision=task.spec['design_decisions']['image_preprocessing']
        self.assertEqual(set(decision['parameter_origins'].values()),{'agent_design_choice'})
        self.assertEqual(decision['chosen'],self.answer['plan']['image_preprocessing'])
        self.assertFalse(decision['semantic_requirement_extraction_certified'])

    def test_partial_constraints_derive_per_parameter_origins(self):
        task=self.compile(upgrade(self.answer,{'width':9,'mode':'L'}));decision=task.spec['design_decisions']['image_preprocessing']
        self.assertEqual(decision['parameter_origins']['width'],'task_requirement')
        self.assertEqual(decision['parameter_origins']['height'],'agent_design_choice')

    def test_all_declared_parameters_remain_task_requirements(self):
        task=self.compile(upgrade(self.answer,dict(self.answer['plan']['image_preprocessing'])))
        self.assertEqual(set(task.spec['design_decisions']['image_preprocessing']['parameter_origins'].values()),{'task_requirement'})

    def test_declared_constraint_cannot_be_overridden_before_decode(self):
        answer=upgrade(self.answer,{'width':10})
        with patch('superagent.native_semantics.load_manifest_split',side_effect=AssertionError('decoded before constraint check')):
            with self.assertRaisesRegex(ValueError,'violates declared constraint: width'):self.compile(answer)

    def test_boolean_width_is_not_equal_to_integer_one(self):
        answer=upgrade(self.answer,{'width':True});answer['plan']['image_preprocessing']['width']=1
        with self.assertRaisesRegex(ValueError,'violates declared constraint'):self.compile(answer)

    def test_goal_cannot_be_reclassified_as_an_agent_design_field(self):
        answer=upgrade(self.answer);answer['design_decisions']['target']={'reason':'freely choose target'}
        with self.assertRaisesRegex(ValueError,'must record its preprocessing decision'):self.compile(answer)

    def test_unknown_constraint_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'unknown task constraint'):self.compile(upgrade(self.answer,{'target':'other'}))

    def test_fabricated_constraint_citation_is_rejected(self):
        answer=upgrade(self.answer,{'width':9});answer['design_decisions']['image_preprocessing']['constraint_evidence']['width']=['invented']
        with self.assertRaisesRegex(ValueError,'fabricated constraint citation'):self.compile(answer)

    def test_missing_task_fact_is_not_swallowed_by_a_design_decision(self):
        answer=upgrade(self.answer);answer['uncertainties']=[dict(field='target',question='Task target is unknown',evidence_refs=['doc:README.md:0'])]
        with self.assertRaisesRegex(ValueError,'semantics: target'):self.compile(answer)

    def test_unresolved_image_task_requirement_stays_unresolved(self):
        answer=upgrade(self.answer);answer['uncertainties']=[dict(field='image_preprocessing',question='Conflicting required task sizes',evidence_refs=['doc:README.md:0'])]
        with self.assertRaisesRegex(ValueError,'image_preprocessing'):self.compile(answer)

    def test_native_array_v3_has_no_image_design_fields(self):
        folder=self.root/'array';folder.mkdir();path,query,answer,_=fixture(folder)
        with package_files(path) as files,package_files(query) as queries:
            task=compile_native(files,evidence_workspace(files),upgrade(answer),prediction_files=queries)
        self.assertEqual(task.spec['design_decisions'],{})

    def test_missing_design_record_does_not_fill_python_defaults(self):
        answer=upgrade(self.answer);answer['design_decisions']={}
        with self.assertRaisesRegex(ValueError,'must record its preprocessing decision'):self.compile(answer)

    def test_design_record_is_in_semantic_artifact_and_digest(self):
        run=self.root/'run';run.mkdir();answer=upgrade(self.answer)
        task=resolve_training_task(self.path,Client([answer]),RunLogger(run),mode='proactive',prediction_package=self.query)
        restored=load_artifact(run/'semantic_artifact.json',self.path,self.query)
        self.assertEqual(task_digest(task),task_digest(restored))
        restored.spec['design_decisions']['image_preprocessing']['reason']='a changed design statement'
        self.assertNotEqual(task_digest(task),task_digest(restored))


if __name__=='__main__':unittest.main()
