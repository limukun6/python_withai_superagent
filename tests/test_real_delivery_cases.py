"""SA8 data/semantic feasibility only; no classifier fitting or live model."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.sa8_real_delivery_cases import build
from scripts.sa7_multiturn_cases import grade
from superagent.semantic_contract import resolve_training_task
from tests.test_semantic_contract import Client


class RealDeliveryCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)/'cases';cls.registry=build(cls.root)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_three_real_modalities_decode_and_export_contracts_pass_without_fitting(self):
        for row in self.registry:
            with self.subTest(case=row['id']):
                oracle=json.loads((self.root/(row['id']+'.oracle.json')).read_text(encoding='utf-8'))
                task=resolve_training_task(self.root/row['package'],Client([oracle['answer']]),mode='proactive',prediction_package=self.root/row['query'])
                result=grade(task,self.root/row['query'],oracle)
                self.assertTrue(result['complete'],repr(result))

    def test_answers_and_oracle_are_absent_from_all_model_input_packages(self):
        for row in self.registry:
            for key in ['package','query']:
                with zipfile.ZipFile(self.root/row[key]) as archive:
                    self.assertFalse(any('oracle' in name or 'answer' in name for name in archive.namelist()))
                    if key=='query':
                        content=archive.read('test.csv').decode('utf-8-sig').splitlines()[0]
                        self.assertNotIn('class_code',content)


if __name__=='__main__':unittest.main()
