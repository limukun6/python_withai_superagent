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
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from superagent.data import load_development_task,load_training_task,load_prediction_features,TrainingTask,DevelopmentTask
from superagent.schema import TextTableRows
from superagent.metrics import MetricSpec
from superagent.general_models import TextTablePreprocessor,generic_candidates
from superagent.splits import inner_split,overlap_audit
from superagent.cli import build_parser,_run
from superagent.predict import create_predictions
from superagent.models import predict_with_model,probability_with_model
from superagent.benchmark import run_one


def npz(**values):
    out=io.BytesIO();np.savez(out,**values);return out.getvalue()


def csv_data(header,rows):
    out=io.StringIO();writer=csv.writer(out);writer.writerow(header);writer.writerows(rows);return out.getvalue()


class GeneralExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def package(self,name,files):
        path=self.root/name
        with zipfile.ZipFile(path,'w') as archive:
            for key,value in files.items():archive.writestr(key,value)
        return path

    def numeric_package(self,name,metric='accuracy',validation=True):
        rng=np.random.default_rng(76);x=rng.normal(size=(110,4));y=(x[:,0]>0).astype(int)
        splits={'train':{'format':'npz','path':'train.npz'},'test':{'format':'npz','path':'test.npz'}}
        files={'train.npz':npz(X=x[:90],y=y[:90]),'test.npz':npz(X=x[90:],y=np.asarray([object()],dtype=object))}
        if validation:
            splits['validation']={'format':'npz','path':'validation.npz'};files['validation.npz']=npz(X=x[90:],y=y[90:])
        files['manifest.json']=json.dumps({'schema_version':1,'modality':'tabular','metric':metric,'splits':splits})
        return self.package(name,files)

    def test_auc_uses_scores_and_honors_class_order(self):
        y=np.array([0,0,1,1]);pred=np.zeros(4,dtype=int);p=np.array([.1,.2,.8,.9])
        metric=MetricSpec('roc_auc',(0,1))
        result=metric.evaluate(y,pred,np.column_stack([1-p,p]),[0,1])
        self.assertEqual(result['accuracy'],.5);self.assertEqual(result['score'],1.)
        self.assertEqual(metric.evaluate(y,pred,np.column_stack([p,1-p]),[1,0])['score'],1.)

    def test_auc_refuses_missing_invalid_and_unrepresentative_scores(self):
        metric=MetricSpec('roc_auc',(0,1));y=[0,1]
        for scores,classes in [(None,None),([[1,1],[1,1]],[0,1]),([[.2,.8],[.3,.7]],[0,0])]:
            with self.assertRaises(ValueError):metric.evaluate(y,y,scores,classes)
        with self.assertRaises(ValueError):metric.evaluate([0,0],[0,0],[[.8,.2],[.7,.3]],[0,1])

    def test_multiclass_auc_is_macro_ovr_with_aligned_columns(self):
        y=np.array(['a','b','c','a','b','c']);p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]*2)
        result=MetricSpec('roc_auc',('a','b','c')).evaluate(y,y,p[:,[2,0,1]],['c','a','b'])
        self.assertEqual(result['score'],roc_auc_score(y,p,multi_class='ovr',average='macro'))

    def test_additional_metrics_and_explicit_positive_label(self):
        y=[0,0,0,1,1,2];pred=[0,0,1,1,2,2]
        for name,average in [('weighted_f1','weighted'),('micro_f1','micro')]:
            self.assertEqual(MetricSpec(name,(0,1,2)).evaluate(y,pred)['score'],f1_score(y,pred,average=average))
        self.assertEqual(MetricSpec('balanced_accuracy',(0,1,2)).evaluate(y,pred)['score'],balanced_accuracy_score(y,pred))
        task=TrainingTask('numeric_tabular',np.ones((4,1)),np.array(['a','b','a','b']),{'metric':'binary_f1','positive_label':'a'})
        self.assertEqual(MetricSpec.for_task(task).labels,('b','a'))
        with self.assertRaises(ValueError):MetricSpec('binary_f1',(0,1,2))

    def test_train_only_manifest_and_native_do_not_read_test(self):
        package=self.numeric_package('declared.zip',validation=False)
        original=zipfile.ZipFile.read
        def guard(archive,name,*args,**kwargs):
            if 'test' in str(name):raise AssertionError('Test read during development')
            return original(archive,name,*args,**kwargs)
        with patch.object(zipfile.ZipFile,'read',guard):
            task=load_development_task(package)
        self.assertIsNone(task.X_val);self.assertIsNone(task.y_val)
        native=self.package('native.zip',{'train.npz':npz(X=np.arange(40).reshape(20,2),y=np.arange(20)%2),'test.npz':b'NEVER READ'})
        with patch.object(zipfile.ZipFile,'read',guard):self.assertIsNone(load_development_task(native).X_val)

    def test_mixed_text_fields_are_fitted_on_training_only(self):
        x=TextTableRows(np.array([['warm sun',1.,'a'],['cold rain',2.,'b'],['sun',3.,'a'],['rain',4.,'b']],dtype=object),(1,),('words','amount','group'),(0,))
        processor=TextTablePreprocessor().fit(x)
        vocab=dict(processor.transformer_.named_transformers_['text0'].named_steps['tfidf'].vocabulary_)
        held=TextTableRows(np.array([['confidentialheldouttoken',8.,'unseen']],dtype=object),(1,),x.columns,(0,))
        self.assertEqual(processor.transform(held).shape[0],1)
        self.assertEqual(vocab,processor.transformer_.named_transformers_['text0'].named_steps['tfidf'].vocabulary_)

    def test_native_multiple_text_columns_are_generic_features_not_preference(self):
        path=self.package('multi.zip',{'train.csv':csv_data(['a','b','target'],[[f'long first document words row{i}',f'long second document words row{i}',i%3] for i in range(24)])})
        task=load_training_task(path)
        self.assertEqual(task.kind,'text_tabular');self.assertEqual(task.X_train.text_columns,(0,1))
        self.assertEqual(len(np.unique(task.y_train)),3)

    def test_relation_multiclass_and_preference_have_distinct_semantics(self):
        manifest={'schema_version':1,'modality':'pair_text','pair_semantics':'relation','target':'label',
            'columns':[{'name':'a','type':'text'},{'name':'b','type':'text'}],
            'splits':{'train':{'format':'csv','path':'train.csv'}}}
        rows=csv_data(['a','b','label'],[['first document','second document',i%3] for i in range(18)])
        path=self.package('relation.zip',{'manifest.json':json.dumps(manifest),'train.csv':rows})
        self.assertEqual(load_training_task(path).kind,'text_tabular')
        manifest.pop('pair_semantics');manifest['pair_label_one']='left'
        path=self.package('preference.zip',{'manifest.json':json.dumps(manifest),'train.csv':rows})
        with self.assertRaisesRegex(ValueError,'Pair targets'):load_training_task(path)

    def test_small_group_split_reduces_folds_without_overlap(self):
        x=np.repeat(np.arange(4),4).reshape(-1,1);y=np.tile([0,1,0,1],4)
        task=TrainingTask('numeric_tabular',x,y)
        train,held=inner_split(task,17)
        self.assertEqual(overlap_audit(task,train,held)['search_duplicate_rows'],0)
        self.assertEqual(set(y[train]),{0,1});self.assertEqual(set(y[held]),{0,1})

    def test_small_group_class_impossibility_and_unsupported_protocol_still_fail(self):
        task=TrainingTask('numeric_tabular',np.array([[0],[0],[1],[1]]),np.array([0,0,1,1]))
        with self.assertRaisesRegex(ValueError,'No leakage-free'):inner_split(task,17)
        task=TrainingTask('numeric_tabular',np.arange(40).reshape(20,2),np.arange(20)%2,{'evaluation_protocol':{'kind':'time'}})
        with self.assertRaisesRegex(ValueError,'temporal'):inner_split(task,17)

    def test_image_folder_inference_and_label_free_prediction(self):
        files={}
        for split in ('train','validation','test'):
            for i in range(8):
                image=np.full((5+i%2,7,3),i*20,dtype=np.uint8)
                stream=io.BytesIO();Image.fromarray(image).save(stream,format='PNG')
                name=f'{split}/class{i%2}/image{i}.png' if split!='test' else f'test/image{i}.png'
                files[name]=stream.getvalue()
        package=self.package('images.zip',files)
        task=load_development_task(package)
        self.assertEqual(task.kind,'image');self.assertEqual(task.X_train.shape,(8,64,64,3))
        self.assertEqual(set(task.y_train),{'class0','class1'})
        frozen={**task.spec,'metric':'accuracy'}
        x=load_prediction_features(package,'image',frozen)
        self.assertEqual(x.shape,(8,64,64,3))

    def test_folder_train_only_does_not_invent_validation(self):
        files={}
        for i in range(10):
            stream=io.BytesIO();Image.fromarray(np.full((4,4),i,dtype=np.uint8)).save(stream,format='PNG')
            files[f'train/class{i%2}/image{i}.png']=stream.getvalue()
        task=load_development_task(self.package('folder.zip',files))
        self.assertIsNone(task.X_val);self.assertIsNone(task.y_val)

    def test_train_only_full_pipeline_freezes_model_without_external_score(self):
        task=load_development_task(self.numeric_package('trainonly.zip',validation=False))
        result=run_one(task,'local_interleaved_routes',707,self.root/'trainrun',None,max_trials=1,
            refit_full_train=True,confirmation_repeats=3,confirmation_top_k=1,portfolio='expanded',time_limit=60,trial_timeout=15)
        self.assertTrue(result['valid_policy_run']);self.assertTrue(result['refit_succeeded'])
        self.assertIsNone(result['official_validation_score'])
        self.assertFalse((self.root/'trainrun/validation_predictions.json').exists())
        kinds=[json.loads(line)['kind'] for line in (self.root/'trainrun/events.jsonl').read_text().splitlines()]
        self.assertIn('final_model_frozen',kinds);self.assertIn('validation_unavailable',kinds)

    def test_auc_worker_confirmation_refit_and_score_prediction_export(self):
        package=self.numeric_package('auc.zip',metric='roc_auc')
        directory=self.root/'auc_run'
        code=_run(build_parser().parse_args(['run',str(package),'--run-dir',str(directory),'--trials','1','--time-limit','90','--trial-timeout','20']))
        self.assertEqual(code,0)
        task=load_development_task(package);saved=joblib.load(directory/'trial_1.joblib')
        data=joblib.load(directory/'inner_data.joblib')
        score=MetricSpec.for_task(task).evaluate(data['y_search'],saved['search_predictions'],saved['search_scores'],saved['score_classes'])
        self.assertEqual(score['score'],saved['metrics']['search_score'])
        output=self.root/'predictions.csv';create_predictions(package,directory,output)
        with output.open() as stream:header=next(csv.reader(stream))
        self.assertEqual(header,['row_id','prediction','score_0','score_1'])
        manifest=json.loads((self.root/'prediction_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['score_class_order'],[0,1]);self.assertFalse(manifest['test_labels_read'])

    def test_undefined_external_auc_is_partial_result_with_frozen_model(self):
        rng=np.random.default_rng(76);x=rng.normal(size=(100,3));y=(x[:,0]>0).astype(int)
        task=DevelopmentTask('numeric_tabular','synthetic',x,y,x[:10],np.zeros(10,dtype=int),{'metric':'roc_auc'})
        result=run_one(task,'local_interleaved_routes',707,self.root/'partial_auc',None,max_trials=1,
            refit_full_train=True,confirmation_repeats=3,confirmation_top_k=1,portfolio='expanded',time_limit=60,trial_timeout=15)
        self.assertTrue(result['refit_succeeded']);self.assertIsNone(result['official_validation_score'])
        self.assertIn('absent',result['validation_failure']['message'])
        self.assertTrue((self.root/'partial_auc/selected_model.joblib').exists())


if __name__=='__main__':unittest.main()
