"""Native reader contracts selected by typed LLM proposals; zero fitting/API."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from superagent.data import load_prediction_features
from superagent.events import RunLogger
from superagent.format_sources import package_files
from superagent.metrics import MetricSpec
from superagent.native_semantics import compile_native
from superagent.semantic_artifact import load_artifact,task_digest
from superagent.semantic_contract import evidence_workspace,resolve_training_task
from tests.test_semantic_contract import Client


def binary_array(value,npz=False):
    buf=io.BytesIO()
    if npz:np.savez(buf,**value)
    else:np.save(buf,value,allow_pickle=False)
    return buf.getvalue()


def output():
    return dict(version='output-csv-v1',delimiter=',',encoding='utf-8',header=True,
        columns=[dict(name='Row',kind='row_index',start=1),dict(name='Class',kind='label')],order=dict(mode='input'))


def fixture(root,fmt='npy',*,embedded=False):
    x=np.arange(24,dtype=float).reshape(8,3);y=np.array(['a','b']*4);q=x[:3]+100
    train={'README.md':'Features come from readings.bin, labels from decisions.bin. Predict incoming.bin. '
           'Macro F1 is the objective. Write Row starting at 1 in input order and Class as unchanged labels. '
           'For folders the classes are immediate directories under gallery/; resize to 9 by 7 grayscale with bilinear stretch.'}
    query={}
    if fmt=='npy':
        train.update({'readings.bin':binary_array(x),'decisions.bin':binary_array(y)})
        query['incoming.bin']=binary_array(q)
        training=dict(format='npy',path='readings.bin',target_path='decisions.bin')
        prediction=dict(format='npy',path='incoming.bin');modality='tabular';preprocessing=None
    elif fmt=='npz':
        train['bundle.dat']=binary_array(dict(measurements=x,decisions=y),True)
        query['incoming.dat']=binary_array(dict(observations=q,y_test=np.array(['PRIVATE']*3)),True)
        training=dict(format='npz',path='bundle.dat',features_key='measurements',target_key='decisions')
        prediction=dict(format='npz',path='incoming.dat',features_key='observations');modality='tabular';preprocessing=None
    else:
        def png(i):
            buf=io.BytesIO();Image.fromarray(np.full((5,6,3),i*20,dtype=np.uint8)).save(buf,format='PNG');return buf.getvalue()
        for i in range(8):train[f'gallery/{y[i]}/{i:03d}.png']=png(i)
        for i in range(3):query[f'incoming/{i:03d}.png']=png(i)
        training=dict(format='image_folder',prefix='gallery/');prediction=dict(format='image_folder',prefix='incoming/')
        modality='image';preprocessing=dict(width=9,height=7,mode='L',resize='stretch_bilinear')
        q=np.stack([np.full((7,9),i*20,dtype=np.uint8) for i in range(3)])
    plan=dict(modality=modality,training=training,prediction=prediction,metric='macro_f1',positive_label=None,
              image_preprocessing=preprocessing,output=output())
    answer=dict(version='semantic-native-v1',plan=plan,
                evidence={k:['doc:README.md:0'] for k in ['training','target','modality','metric','prediction','image_preprocessing','output']},uncertainties=[])
    train['validation/answers.npy']=binary_array(np.array(['FORBIDDEN']))
    query['y_test.npy']=binary_array(np.array(['FORBIDDEN']*3))
    if embedded:train.update(query)
    task=root/'task.zip';target=root/'query.zip'
    for path,members in [(task,train),(target,query)]:
        with zipfile.ZipFile(path,'w') as archive:
            for name,value in members.items():archive.writestr(name,value)
    return task,task if embedded else target,answer,q


class NativeSemanticsTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)

    def compile(self,path,query,answer):
        with package_files(path) as train,package_files(query) as prediction:
            return compile_native(train,evidence_workspace(train),answer,prediction_files=prediction)

    def test_arbitrary_numpy_member_names_and_full_prediction_replay(self):
        path,query,answer,expected=fixture(self.root)
        task=resolve_training_task(path,Client([answer]),mode='proactive',prediction_package=query)
        self.assertEqual(task.kind,'numeric_tabular');self.assertEqual(len(task.y_train),8)
        self.assertEqual(MetricSpec.for_task(task).name,'macro_f1')
        self.assertEqual(task.spec['semantic_target_source'],'decisions.bin')
        np.testing.assert_array_equal(load_prediction_features(query,task.kind,task.spec),expected)

    def test_npz_keys_are_frozen_and_no_query_label_key_is_loaded(self):
        path,query,answer,expected=fixture(self.root,'npz')
        original=np.lib.npyio.NpzFile.__getitem__
        seen=[]
        def read(bundle,key):
            seen.append(key);self.assertNotIn(key,['y_test','y_test.npy']);return original(bundle,key)
        with patch.object(np.lib.npyio.NpzFile,'__getitem__',read):
            task=self.compile(path,query,answer);observed=load_prediction_features(query,task.kind,task.spec)
        self.assertEqual(seen.count('decisions'),1)
        np.testing.assert_array_equal(observed,expected)

    def test_nonstandard_image_folders_and_documented_pixel_contract(self):
        path,query,answer,expected=fixture(self.root,'image_folder')
        task=self.compile(path,query,answer)
        self.assertEqual(task.kind,'image');self.assertEqual(task.X_train.shape,(8,7,9))
        self.assertEqual(task.spec['feature_shape'],[7,9]);self.assertEqual(MetricSpec.for_task(task).name,'macro_f1')
        np.testing.assert_array_equal(load_prediction_features(query,task.kind,task.spec),expected)

    def test_artifact_restores_native_task_without_new_client(self):
        path,query,answer,_=fixture(self.root,'image_folder');run=self.root/'run';run.mkdir()
        task=resolve_training_task(path,Client([answer]),RunLogger(run),mode='proactive',prediction_package=query)
        restored=load_artifact(run/'semantic_artifact.json',path,query)
        self.assertEqual(task_digest(restored),task_digest(task))

    def test_embedded_query_binding_uses_same_package_without_training_on_query(self):
        path,query,answer,expected=fixture(self.root,embedded=True)
        task=resolve_training_task(path,Client([answer]),mode='proactive')
        self.assertEqual(len(task.y_train),8)
        np.testing.assert_array_equal(load_prediction_features(query,task.kind,task.spec),expected)

    def test_training_never_reads_evaluation_member(self):
        path,query,answer,_=fixture(self.root)
        with package_files(path) as train,package_files(query) as prediction:
            original=train.read
            def guarded(name,*args):
                self.assertNotIn('validation',name);return original(name,*args)
            with patch.object(train,'read',side_effect=guarded):
                self.assertEqual(len(compile_native(train,evidence_workspace(train),answer,prediction_files=prediction).y_train),8)

    def test_evaluation_sources_and_label_vectors_are_refused_before_read(self):
        path,query,answer,_=fixture(self.root)
        for source,change in [('training',dict(path='validation/answers.npy')),('training',dict(target_path='y_test.npy')),
                              ('prediction',dict(path='y_test.npy'))]:
            bad=deepcopy(answer);bad['plan'][source].update(change)
            with self.subTest(source=source,change=change),self.assertRaises(ValueError):self.compile(path,query,bad)

    def test_forged_evidence_unknown_fields_and_unresolved_semantics_are_refused(self):
        path,query,answer,_=fixture(self.root)
        variants=[]
        bad=deepcopy(answer);bad['evidence']['metric']=['fake'];variants.append(bad)
        bad=deepcopy(answer);bad['uncertainties']=['Which target?'];variants.append(bad)
        bad=deepcopy(answer);bad['plan']['training']['code']='print(1)';variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.compile(path,query,bad)

    def test_no_metric_or_preprocessing_default_is_filled(self):
        path,query,answer,_=fixture(self.root,'image_folder')
        for field in ['metric','image_preprocessing']:
            bad=deepcopy(answer);bad['plan'][field]=None
            with self.subTest(field=field),self.assertRaises(ValueError):self.compile(path,query,bad)

    def test_prediction_shape_drift_is_refused(self):
        path,query,answer,_=fixture(self.root)
        with zipfile.ZipFile(query,'w') as archive:archive.writestr('incoming.bin',binary_array(np.zeros((3,4))))
        with self.assertRaisesRegex(ValueError,'shape'):self.compile(path,query,answer)

    def test_wrong_positive_class_is_refused(self):
        path,query,answer,_=fixture(self.root);answer['plan']['positive_label']='missing'
        answer['evidence']['positive_label']=['doc:README.md:0']
        with self.assertRaisesRegex(ValueError,'positive_label'):self.compile(path,query,answer)

    def test_explicit_manifest_cannot_be_overridden(self):
        path,query,answer,_=fixture(self.root)
        with zipfile.ZipFile(path,'a') as archive:archive.writestr('manifest.json','{}')
        with self.assertRaisesRegex(ValueError,'explicit manifest'):self.compile(path,query,answer)

    def test_numeric_array_image_shape_and_class_type_are_preserved(self):
        path,query,answer,_=fixture(self.root)
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('README.md','Numeric image labels are 0/1, positive is 1; pixels have shape 5,6,3.')
            archive.writestr('readings.bin',binary_array(np.arange(8*5*6*3,dtype=np.uint8).reshape(8,5,6,3)))
            archive.writestr('decisions.bin',binary_array(np.array([0,1]*4)))
        with zipfile.ZipFile(query,'w') as archive:archive.writestr('incoming.bin',binary_array(np.ones((3,5,6,3),dtype=np.uint8)))
        answer['plan'].update(modality='image',positive_label=1);answer['evidence']['positive_label']=['doc:README.md:0']
        task=self.compile(path,query,answer)
        self.assertEqual(task.kind,'image');self.assertEqual(MetricSpec.for_task(task).labels,(0,1))

    def test_pickled_object_array_cannot_enter_native_reader(self):
        path,query,answer,_=fixture(self.root);buf=io.BytesIO();np.save(buf,np.array([dict(x=1)],dtype=object),allow_pickle=True)
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('README.md','Object data is not executable.');archive.writestr('readings.bin',buf.getvalue())
            archive.writestr('decisions.bin',binary_array(np.array(['a'])))
        with self.assertRaisesRegex(ValueError,'Object arrays'):self.compile(path,query,answer)

    def test_decoder_drift_is_refused_at_prediction(self):
        path,query,answer,_=fixture(self.root,'image_folder');task=self.compile(path,query,answer)
        task.spec['image_decoder_version']='different'
        with self.assertRaisesRegex(ValueError,'decoder changed'):load_prediction_features(query,task.kind,task.spec)

    def test_fresh_process_can_replay_frozen_numpy_contract(self):
        path,query,answer,expected=fixture(self.root,'npz');task=self.compile(path,query,answer)
        spec=self.root/'spec.json';spec.write_text(json.dumps(task.spec),encoding='utf-8')
        command=[sys.executable,'-c','import json,sys; from superagent.data import load_prediction_features; '
            'x=load_prediction_features(sys.argv[1],"numeric_tabular",json.load(open(sys.argv[2],encoding="utf-8"))); print(json.dumps(x.tolist()))',str(query),str(spec)]
        env=os.environ.copy();env.update(SUPERAGENT_OFFLINE='1')
        proc=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,
                            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        self.assertEqual(proc.returncode,0,proc.stderr);np.testing.assert_array_equal(json.loads(proc.stdout),expected)


if __name__=='__main__':unittest.main()
