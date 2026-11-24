import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
import numpy as np
from superagent.contracts import TaskSpec,DecisionState
from superagent.data import load_training_task


class TaskGoalTests(unittest.TestCase):
    def test_manifest_and_inferred_default_have_different_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('explicit','omitted','native'):
                path=Path(tmp)/(mode+'.zip')
                with zipfile.ZipFile(path,'w') as z:
                    b=io.BytesIO();np.savez(b,X=np.arange(8).reshape(4,2),y=[0,1,0,1]);z.writestr('train.npz',b.getvalue())
                    if mode!='native':
                        m=dict(schema_version=1,modality='tabular',splits={'train':{'format':'npz','path':'train.npz'}})
                        if mode=='explicit':m['metric']='accuracy'
                        z.writestr('manifest.json',json.dumps(m))
                t=TaskSpec.from_task(load_training_task(path));r=t.record()['goal_resolution']['fields']['primary_metric']
                self.assertEqual(r['status'],'confirmed' if mode=='explicit' else 'unknown')
                self.assertEqual(t.metric.name,'accuracy')
                self.assertEqual(DecisionState.build({},[],[],2,task_contract=t).record()['task_spec'],t.record())

    def test_positive_label_declaration_changes_effective_class_order(self):
        task=SimpleNamespace(kind='numeric_tabular',X_train=np.zeros((4,1)),y_train=np.array(['a','z','a','z']),spec={'metric':'binary_f1'})
        default=TaskSpec.from_task(task)
        self.assertEqual(default.goal_resolution['fields']['positive_label']['status'],'unknown')
        task.spec['positive_label']='a';explicit=TaskSpec.from_task(task)
        self.assertEqual(explicit.metric.labels,('z','a'))
        self.assertEqual(explicit.goal_resolution['fields']['positive_label']['status'],'confirmed')

    def test_invalid_training_shape_is_rejected_before_execution(self):
        for kind,x,y in [('single_text',[['two','columns']]*4,[0,1,0,1]),('image',np.zeros((4,5)),[0,1,0,1]),
                         ('numeric_tabular',np.zeros((3,2)),[0,1,0,1])]:
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                TaskSpec.from_task(SimpleNamespace(kind=kind,X_train=x,y_train=np.array(y),spec={}))


if __name__=='__main__':unittest.main()
