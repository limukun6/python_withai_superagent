"""Check the predeclared oracle independently of LLM proposals and network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.sa5_semantic_cases import build, grade
from superagent.format_sources import package_files
from superagent.semantic_contract import compile_plan,evidence_workspace


class SemanticPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'cases'; self.registry=build(self.root)

    def task(self,name):
        oracle=json.loads((self.root/(name+'.oracle.json')).read_text(encoding='utf-8'))
        with package_files(self.root/(name+'.zip')) as files:
            task=compile_plan(files,evidence_workspace(files),oracle['answer'])
        return task,oracle

    def test_all_four_preregistered_oracles_compile_and_pass_real_input_probes(self):
        self.assertEqual([r['id'] for r in self.registry],['P1','P2','P3','P4'])
        for name in ['P1','P2','P3','P4']:
            with self.subTest(name=name):
                task,oracle=self.task(name)
                self.assertTrue(grade(task,self.root/(name+'.zip'),oracle)['complete'])

    def test_wrong_metric_is_not_accepted_even_when_input_decodes(self):
        task,oracle=self.task('P2'); task.spec['metric']='accuracy'
        result=grade(task,self.root/'P2.zip',oracle)
        self.assertFalse(result['checks']['metric']); self.assertFalse(result['complete'])

    def test_missing_output_is_wrong_despite_successful_training_and_prediction_intake(self):
        task,oracle=self.task('P1'); task.spec.pop('output_recipe')
        result=grade(task,self.root/'P1.zip',oracle)
        self.assertTrue(result['checks']['prediction_values']); self.assertFalse(result['complete'])

    def test_validation_sentinel_and_oracle_never_enter_document_workspace(self):
        task,oracle=self.task('P4')
        self.assertEqual(len(task.y_train),8)
        self.assertNotIn('FORBIDDEN_SA5_VALUE',task.y_train)
        with package_files(self.root/'P4.zip') as files: workspace=evidence_workspace(files)
        self.assertNotIn('FORBIDDEN_SA5_VALUE',json.dumps(workspace))
        self.assertNotIn('oracle',json.dumps(workspace))

    def test_alias_change_or_probability_class_change_fails_oracle(self):
        task,oracle=self.task('P3'); task.spec['input_recipe']['field_map']={}
        result=grade(task,self.root/'P3.zip',oracle)
        self.assertFalse(result['checks']['prediction_values'])
        task,oracle=self.task('P3'); task.spec['output_recipe']['columns'][1]['class_label']='hold'
        self.assertFalse(grade(task,self.root/'P3.zip',oracle)['checks']['output_recipe'])


if __name__=='__main__': unittest.main()
