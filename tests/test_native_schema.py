import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.data import load_training_task,load_development_task,load_prediction_features
from superagent.predict import create_predictions


def csv_data(header,rows):
    buf=io.StringIO();writer=csv.writer(buf);writer.writerow(header);writer.writerows(rows)
    return buf.getvalue().encode()


def array_data(**values):
    buf=io.BytesIO();np.savez_compressed(buf,**values);return buf.getvalue()


def npy(value):
    buf=io.BytesIO();np.save(buf,value);return buf.getvalue()


def package(root,files,name='unrelated_dataset_identity.zip'):
    path=Path(root)/name
    with zipfile.ZipFile(path,'w') as archive:
        for key,content in files.items():archive.writestr(key,content)
    return path


class NativeSchemaTests(unittest.TestCase):
    def test_train_csv_only_discovery_and_identity_invariance(self):
        train=csv_data(['measurement','group','label'],[[i,'a' if i%2 else 'b',i%3] for i in range(30)])
        with tempfile.TemporaryDirectory() as temp:
            path=package(temp,{'nested/train.csv':train,'val.csv':b'FORBIDDEN','test.csv':b'FORBIDDEN'})
            original=zipfile.ZipFile.read;access=[]
            def checked(archive,name,*args,**kwargs):
                self.assertEqual(name,'nested/train.csv');access.append(name)
                return original(archive,name,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'read',checked):task=load_training_task(path)
            other=package(temp,{'nested/train.csv':train},'entirely_different_task.zip')
            changed=load_training_task(other)
            self.assertEqual(task.kind,'mixed_tabular');self.assertEqual(task.spec,changed.spec)
            self.assertTrue(access);self.assertEqual(task.spec['schema_origin'],'native_v1')
            self.assertEqual(task.X_train.numeric_columns,(0,))

    def test_validation_cannot_reinfer_numeric_type_or_choose_columns(self):
        train=csv_data(['a','b','target'],[[i,'low' if i%2 else 'high',i%2] for i in range(20)])
        with tempfile.TemporaryDirectory() as temp:
            path=package(temp,{'train.csv':train,'val.csv':csv_data(['b','target','a'],[['new',1,99]])})
            task=load_development_task(path)
            self.assertEqual(task.X_val.values.tolist(),[[99.,'new']])
            path=package(temp,{'train.csv':train,'val.csv':csv_data(['a','b','target'],[['not numeric','new',1]])})
            with self.assertRaises(ValueError):load_development_task(path)

    def test_numeric_categorical_and_single_text_csv_routes(self):
        cases=[('numeric_tabular',['value','label'],[[i,i%2] for i in range(20)]),
               ('categorical_tabular',['group','label'],[['a' if i%2 else 'b',i%2] for i in range(20)]),
               ('single_text',['document','class'],[[f'Example document with several words number {i}',i%3] for i in range(20)])]
        with tempfile.TemporaryDirectory() as temp:
            for kind,header,rows in cases:
                with self.subTest(kind=kind):
                    self.assertEqual(load_training_task(package(temp,{'train.csv':csv_data(header,rows)})).kind,kind)

    def test_ambiguous_targets_and_text_semantics_fail_explicitly(self):
        cases=[(['x','label','target'],[[1,0,1],[2,1,0]]),
               (['x','answer'],[[1,0],[2,1]]),
               (['x','label'],[['',0],['',1]])]
        with tempfile.TemporaryDirectory() as temp:
            for header,rows in cases:
                with self.subTest(header=header),self.assertRaises(ValueError):
                    load_training_task(package(temp,{'train.csv':csv_data(header,rows)}))

    def test_explicit_manifest_resolves_target_and_type_ambiguity(self):
        spec={'schema_version':1,'modality':'tabular','metric':'accuracy','target':'answer',
              'columns':[{'name':'x','type':'categorical'}],
              'splits':{'train':{'path':'train.csv','format':'csv'}}}
        with tempfile.TemporaryDirectory() as temp:
            task=load_training_task(package(temp,{'manifest.json':json.dumps(spec),'train.csv':csv_data(['x','answer'],[[1,0],[2,1]])}))
            self.assertEqual(task.kind,'categorical_tabular');self.assertEqual(task.spec['schema_origin'],'manifest_v1')

    def test_native_arrays_train_only_and_no_object_deserialization(self):
        x=np.arange(30*4*4).reshape(30,4,4);y=np.arange(30)%3
        with tempfile.TemporaryDirectory() as temp:
            for files in ({'X_train.npy':npy(x),'y_train.npy':npy(y)}, {'train.npz':array_data(X=x,y=y)}):
                path=package(temp,{**files,'y_test.npy':b'FORBIDDEN','val.npz':b'FORBIDDEN'})
                task=load_training_task(path)
                self.assertEqual(task.kind,'image');self.assertEqual(task.spec['feature_shape'],[4,4])
            bad=package(temp,{'X_train.npy':npy(np.asarray([['a'],['b']],dtype=object)),'y_train.npy':npy([0,1])})
            with self.assertRaisesRegex(ValueError,'Object arrays'):load_training_task(bad)

    def test_duplicate_layout_or_wrong_validation_dimensions_are_rejected(self):
        x=np.ones((10,2));y=np.arange(10)%2
        cases=[{'train.npz':array_data(X=x,y=y),'other/train.npz':array_data(X=x,y=y)},
               {'train.npz':array_data(X=x,y=y),'train.csv':csv_data(['x','label'],[[1,0],[2,1]])}]
        with tempfile.TemporaryDirectory() as temp:
            for files in cases:
                with self.assertRaises(ValueError):load_training_task(package(temp,files))
            path=package(temp,{'train.npz':array_data(X=x,y=y),'val.npz':array_data(X=np.ones((2,3)),y=[0,1])})
            with self.assertRaisesRegex(ValueError,'dimensions'):load_development_task(path)

    def test_feature_only_prediction_preserves_frozen_csv_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path=package(temp,{'train.csv':csv_data(['a','b','label'],[[i,'category',i%2] for i in range(20)])})
            task=load_training_task(path)
            pred=package(temp,{'test.csv':csv_data(['b','label','a'],[['new','TARGET MUST BE IGNORED',91]])},'features.zip')
            x=load_prediction_features(pred,task.kind,task.spec)
            self.assertEqual(x.values.tolist(),[[91.,'new']]);self.assertEqual(x.columns,('a','b'))
            spec={**task.spec['input_schema'],'schema_version':1,'metric':'accuracy',
                  'splits':{'test':{'path':'test.csv','format':'csv'}}}
            spec['columns']=list(reversed(spec['columns']))
            changed=package(temp,{'manifest.json':json.dumps(spec),'test.csv':csv_data(['a','b'],[[91,'new']])},'changed.zip')
            with self.assertRaisesRegex(ValueError,'frozen'):load_prediction_features(changed,task.kind,task.spec)

    def test_feature_only_arrays_do_not_load_targets_or_training_data(self):
        with tempfile.TemporaryDirectory() as temp:
            for fmt in ('npy','npz'):
                spec={'input_schema':{'modality':'tabular'},'input_format':fmt,'feature_shape':[2]}
                files={'X_test.npy':npy(np.ones((3,2))),'y_test.npy':b'FORBIDDEN'} if fmt=='npy' else {
                    'test.npz':array_data(X=np.ones((3,2)),y=np.asarray([object()],dtype=object))}
                path=package(temp,{**files,'train.npz':b'FORBIDDEN'})
                original=zipfile.ZipFile.read
                def checked(archive,name,*args,**kwargs):
                    self.assertIn(name,('X_test.npy','test.npz'));return original(archive,name,*args,**kwargs)
                with patch.object(zipfile.ZipFile,'read',checked):x=load_prediction_features(path,'numeric_tabular',spec)
                self.assertEqual(x.shape,(3,2))

    def test_empty_label_and_test_path_cannot_be_training_data(self):
        with tempfile.TemporaryDirectory() as temp:
            for files in ({'train.csv':csv_data(['x','label'],[[1,''],[2,'a']])},
                          {'test/train.npz':array_data(X=np.ones((2,3)),y=[0,1])}):
                with self.assertRaises(ValueError):load_training_task(package(temp,files))

    def test_modified_frozen_schema_fails_before_model_or_prediction_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'final_selection.json').write_text(json.dumps({'task_spec_sha256':'invalid'}))
            (root/'task_spec.json').write_text('{}')
            with patch('superagent.predict.joblib.load') as load,patch('superagent.predict.load_prediction_features') as inputs:
                with self.assertRaisesRegex(ValueError,'schema no longer'):create_predictions('absent.zip',root,root/'pred.csv')
                load.assert_not_called();inputs.assert_not_called()


if __name__=='__main__':unittest.main()
