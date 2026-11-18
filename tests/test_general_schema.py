import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import joblib
import numpy as np
from sklearn.datasets import make_classification

from superagent.data import load_training_task,load_development_task,TrainingTask
from superagent.general_models import ImageFlattener
from superagent.models import search_candidates,predict_with_model
from superagent.profile import profile
from superagent.schema import MixedRows
from superagent.splits import inner_split,take,overlap_audit,sample_groups
from superagent.evidence_agent import route_of
from superagent.hypothesis_agent import build_state


def npz(x,y):
    buffer=io.BytesIO();np.savez_compressed(buffer,X=x,y=y);return buffer.getvalue()


def csv_bytes(header,rows):
    buffer=io.StringIO();writer=csv.writer(buffer);writer.writerow(header);writer.writerows(rows)
    return buffer.getvalue().encode()


def package(root,manifest,train,validation=None):
    path=Path(root)/'package.zip'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('manifest.json',json.dumps(manifest))
        archive.writestr(manifest['splits']['train']['path'],train)
        if 'validation' in manifest['splits']:
            archive.writestr(manifest['splits']['validation']['path'],validation or b'READING VALIDATION HERE WOULD FAIL')
        archive.writestr('sealed_test.npz',b'SYNTHETIC TEST SENTINEL, NOT A RELEASED TASK')
    return path


def manifest(modality,fmt='npz',**extra):
    return {'schema_version':1,'modality':modality,'metric':'accuracy',
        'splits':{name:{'format':fmt,'path':name+'.'+fmt} for name in ('train','validation')},**extra}


class SchemaTests(unittest.TestCase):
    def test_train_only_loader_does_not_read_validation_or_test_and_ignores_task_name(self):
        x=np.arange(180).reshape(30,6);y=np.tile([10,20,30],10)
        with tempfile.TemporaryDirectory() as temp:
            spec=manifest('tabular',name='Do not use identity to choose any model',instructions='irrelevant data text')
            path=package(temp,spec,npz(x,y))
            original=zipfile.ZipFile.read;access=[]
            def checked(archive,name,*args,**kwargs):
                access.append(name)
                if name!='manifest.json' and name!='train.npz':raise AssertionError('Forbidden split')
                return original(archive,name,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'read',checked):task=load_training_task(path)
            self.assertEqual(task.kind,'numeric_tabular')
            self.assertEqual(set(task.y_train),{10,20,30})
            self.assertEqual(access,['manifest.json','train.npz'])
            self.assertNotIn('name',task.spec)

    def test_manifest_rejects_target_as_feature_and_development_pointing_to_test(self):
        with tempfile.TemporaryDirectory() as temp:
            spec=manifest('tabular','csv',columns=[{'name':'target','type':'numeric'}],target='target')
            path=package(temp,spec,csv_bytes(['target'],[[0],[1]]))
            with self.assertRaises(ValueError):load_training_task(path)
            spec=manifest('tabular');spec['splits']['train']['path']='heldout_test.npz'
            path=package(temp,spec,npz(np.ones((4,2)),np.array([0,1,0,1])))
            with self.assertRaises(ValueError):load_training_task(path)

    def test_numeric_multiclass_pipeline_preserves_labels_and_saved_predictions(self):
        x,y=make_classification(n_samples=90,n_features=6,n_informative=4,n_redundant=0,n_classes=3,random_state=4)
        y=np.asarray(['red','green','blue'])[y];x[0,0]=np.nan
        candidate=search_candidates('numeric_tabular')[0]
        model,pred=candidate.fit_predict(x[:75],y[:75],x[75:])
        stats=model.named_steps['imputer'].statistics_.copy()
        self.assertTrue(set(pred)<={'red','green','blue'})
        predict_with_model('numeric_tabular',model,x[75:]*10)
        np.testing.assert_array_equal(stats,model.named_steps['imputer'].statistics_)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'model.joblib';joblib.dump(model,p)
            np.testing.assert_array_equal(pred,predict_with_model('numeric_tabular',joblib.load(p),x[75:]))

    def test_mixed_schema_survives_slicing_missing_numeric_and_unseen_category(self):
        columns=[{'name':'age','type':'numeric'},{'name':'group','type':'categorical'}]
        rows=[[i if i%8 else '',['a','b','c'][i%3],['low','mid','high'][i%3]] for i in range(75)]
        with tempfile.TemporaryDirectory() as temp:
            spec=manifest('tabular','csv',columns=columns,target='outcome')
            path=package(temp,spec,csv_bytes(['age','group','outcome'],rows),csv_bytes(['age','group','outcome'],[[99,'unseen','high'],['','a','low']]))
            task=load_development_task(path)
            self.assertEqual(task.kind,'mixed_tabular')
            a,b=inner_split(task,13)
            self.assertIsInstance(take(task.X_train,a),MixedRows)
            model,pred=search_candidates(task.kind)[0].fit_predict(take(task.X_train,a),task.y_train[a],task.X_val)
            self.assertEqual(len(pred),2)
            pre=model.named_steps['features']
            before=pre.transformer_.named_transformers_['numeric'].named_steps['imputer'].statistics_.copy()
            predict_with_model(task.kind,model,task.X_val)
            np.testing.assert_array_equal(before,pre.transformer_.named_transformers_['numeric'].named_steps['imputer'].statistics_)
            state=build_state(profile(task),[],[],3,True)
            self.assertEqual(state['profile']['numeric_columns'],[0])
            self.assertEqual(overlap_audit(task,a,b)['search_duplicate_rows'],0)

    def test_single_text_has_its_own_multiclass_route_and_normalized_duplicate_groups(self):
        text=[f'{topic} story variant {i}' for i in range(12) for topic in ('science','sports','music')]
        labels=np.tile(['s','p','m'],12)
        with tempfile.TemporaryDirectory() as temp:
            spec=manifest('text','csv',columns=[{'name':'body','type':'text'}],target='category')
            path=package(temp,spec,csv_bytes(['body','category'],zip(text,labels)),csv_bytes(['body','category'],[['music story','m'],['science story','s']]))
            task=load_development_task(path)
            candidate=search_candidates(task.kind)[0]
            model,pred=candidate.fit_predict(task.X_train,task.y_train,task.X_val)
            self.assertTrue(set(pred)<={'s','p','m'})
            self.assertEqual(route_of(candidate.family),'single_word')
            self.assertEqual(profile(task)['document_count'],36)
        groups=sample_groups(TrainingTask('single_text',[' a  b ','A B','c'],np.array([0,0,1])))
        self.assertEqual(groups[0],groups[1]);self.assertNotEqual(groups[0],groups[2])

    def test_image_shape_and_duplicate_grouping_are_preserved(self):
        x=np.arange(60*4*4).reshape(60,4,4)%256;y=np.arange(60)%3
        x[1]=x[0]
        groups=sample_groups(TrainingTask('image',x,y))
        self.assertEqual(groups[0],groups[1])
        flatten=ImageFlattener().fit(x)
        self.assertEqual(flatten.transform(x).shape,(60,16))
        with self.assertRaises(ValueError):flatten.transform(np.zeros((3,5,5)))
        model,pred=search_candidates('image')[0].fit_predict(x[:48],y[:48],x[48:])
        self.assertEqual(pred.shape,(12,))
        self.assertEqual(model.named_steps['flatten'].feature_shape_,(4,4))

    def test_pair_adapter_refuses_ambiguous_label_direction_and_multiclass_targets(self):
        columns=[{'name':'left_text','type':'text'},{'name':'right_text','type':'text'}]
        with tempfile.TemporaryDirectory() as temp:
            spec=manifest('pair_text','csv',columns=columns,target='winner')
            path=package(temp,spec,csv_bytes(['left_text','right_text','winner'],[['a','b',0],['c','d',1]]))
            with self.assertRaises(ValueError):load_training_task(path)
            spec['pair_label_one']='left'
            path=package(temp,spec,csv_bytes(['left_text','right_text','winner'],[['a','b',0],['c','d',2]]))
            with self.assertRaises(ValueError):load_training_task(path)


if __name__=='__main__':unittest.main()
