"""No-model feasibility checks before the new SA7 live dialogue."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.sa7_multiturn_cases import build,grade
from superagent.semantic_contract import resolve_training_task
from tests.test_semantic_contract import Client


class MultiturnPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'cases';self.registry=build(self.root)

    def test_all_four_oracles_pass_with_one_structure_request_then_proposal(self):
        for row in self.registry:
            with self.subTest(case=row['id']):
                oracle=json.loads((self.root/(row['id']+'.oracle.json')).read_text(encoding='utf-8'))
                client=Client([oracle['tool_request'],oracle['answer']])
                task=resolve_training_task(self.root/row['package'],client,mode='proactive',
                    prediction_package=self.root/row['query'] if row['separate_query'] else None)
                result=grade(task,self.root/row['query'],oracle)
                self.assertTrue(result['complete'],repr(result))
                self.assertEqual(client.request_count,2)
                self.assertTrue(client.payloads[1]['tool_observations'])

    def test_image_overlay_does_not_replace_the_declared_modality(self):
        row=self.registry[3];oracle=json.loads((self.root/'Q4.oracle.json').read_text(encoding='utf-8'))
        task=resolve_training_task(self.root/row['package'],Client([oracle['answer']]),mode='proactive',prediction_package=self.root/row['query'])
        self.assertEqual(task.kind,'image');self.assertEqual(task.spec['feature_shape'],[12,12,3])
        self.assertNotIn('input_recipe',task.spec)

    def test_wrong_positive_probability_or_output_index_is_not_a_complete_contract(self):
        for index in [1,3]:
            row=self.registry[index];oracle=json.loads((self.root/(row['id']+'.oracle.json')).read_text(encoding='utf-8'))
            task=resolve_training_task(self.root/row['package'],Client([oracle['answer']]),mode='proactive',
                prediction_package=self.root/row['query'] if row['separate_query'] else None)
            task.spec['output_recipe']['columns'][0]['name']='WrongHeader'
            self.assertFalse(grade(task,self.root/row['query'],oracle)['complete'])


if __name__=='__main__':unittest.main()
