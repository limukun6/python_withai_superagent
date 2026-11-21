import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from superagent.data import TrainingTask
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.ab_registered_rows import content_digest, registered_partitioner, validated_ids


class RegisteredRowTests(unittest.TestCase):
    def setUp(self):
        self.task = TrainingTask('pair_text', [('a','b'),('b','c'),('u','v'),('r','s'),('f','g'),('h','i')],
                                 np.array([0,1,0,1,0,1]))
        self.groups = sample_groups(self.task,'shared_text')

    def record(self,fit,held):
        return dict(train_indices=fit,search_indices=held,row_ids_sha256=fingerprint(held),
                    label_sha256=fingerprint(self.task.y_train[held]))

    def test_rejects_shared_text_even_with_matching_row_and_label_hashes(self):
        row = self.record([0,3,4,5],[1,2])
        with self.assertRaisesRegex(ValueError,'group crosses'):
            validated_ids(row,self.task.y_train,self.groups)

    def test_rejects_shifted_domain_duplicates_and_mismatching_labels(self):
        for fit, held in [([0,1,2,3],[4]),([0,1,2,3],[4,4]),([0,1,2,3],[4,False])]:
            with self.subTest(held=held), self.assertRaises(ValueError):
                row=self.record([0,1,2,3],[4,5])
                row.update(train_indices=fit,search_indices=held)
                validated_ids(row,self.task.y_train,self.groups)
        row=self.record([0,1,2,3],[4,5])
        row['label_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'labels changed'):
            validated_ids(row,self.task.y_train,self.groups)

    def test_registered_provider_refuses_another_seed_task_or_file(self):
        row=dict(self.record([0,1,2,3],[4,5]),seed=707)
        manifest=dict(version=1,kind='pair_text',training_content_sha256=content_digest(self.task.X_train,self.task.y_train),
                      grouping='shared_text',splits=[row])
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'ids.json'
            file.write_text(json.dumps(manifest),encoding='utf-8')
            signature=hashlib.sha256(file.read_bytes()).hexdigest()
            select=registered_partitioner(self.task,file,signature)
            self.assertEqual(select(self.task,707,'shared_text')[1].tolist(),[4,5])
            for task,seed,mode in ((self.task,808,'shared_text'),(self.task,707,'exact_rows'),
                                  (TrainingTask('pair_text',self.task.X_train,self.task.y_train),707,'shared_text')):
                with self.assertRaises(ValueError):
                    select(task,seed,mode)
            file.write_text(json.dumps(manifest)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'digest'):
                registered_partitioner(self.task,file,signature)
