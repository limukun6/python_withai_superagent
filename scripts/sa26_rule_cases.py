"""Controlled public-data rule wording conditions; prediction answers never exported."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.datasets import load_iris,load_wine

from scripts.run_sa5_semantic_panel import save,sha
from scripts.sa8_real_delivery_cases import partition
from scripts.sa7_multiturn_cases import array_fingerprint


def npy(value):
    stream=io.BytesIO();np.save(stream,value,allow_pickle=False);return stream.getvalue()


def build(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False);registry=[]
    for name,data,metric,quote,constraints in [
        ('R1',load_iris(),'balanced_accuracy','Allowed classifiers: affine class scores followed by a softmax',['linear_only']),
        ('R2',load_wine(),None,'Primary metric is the fraction of correctly classified rows',[]),
        ('N1',load_iris(),'macro_f1','Do not use pretrained encoders',[])]:
        fit,held=partition(data.data,data.target,96,24);x=data.data[fit];y=np.array(['c'+str(i) for i in data.target[fit]]);q=data.data[held]
        manifest=dict(schema_version=1,modality='tabular',objective='classification',target='kind',
            splits=dict(train=dict(format='npy',path='X_train.npy',target_path='y_train.npy')))
        if metric:manifest['metric']=metric
        if name=='N1':manifest['restrictions']={'no_pretrained_assets':False}
        query_manifest=dict(schema_version=1,modality='tabular',splits=dict(test=dict(format='npy',path='X_test.npy')))
        doc=('Classify the numeric observations. The manifest defines the training feature matrix and target vector. '
            'The target codes are unchanged class names c0, c1 and c2. A separate prediction package supplies X_test.npy '
            'with the same feature order and no labels. ')
        if name=='R1':doc+='Primary metric is balanced accuracy. '+quote+'. Do not use external data. '
        elif name=='R2':doc+=quote+'. '
        else:doc+='Primary metric is macro F1. '+quote+'. The manifest restrictions also apply. '
        doc+='Export UTF-8 comma-separated CSV with header Row,Class. Row starts at 1 in prediction input order; Class uses the unchanged class codes.'
        for path,members in [(root/(name+'.zip'),{'manifest.json':json.dumps(manifest),'README.md':doc,'X_train.npy':npy(x),'y_train.npy':npy(y)}),
            (root/(name+'.query.zip'),{'manifest.json':json.dumps(query_manifest),'X_test.npy':npy(q)})]:
            with zipfile.ZipFile(path,'w') as z:
                for member,body in members.items():z.writestr(member,body)
        expected=dict(kind='numeric_tabular',metric=metric or 'accuracy',classes=sorted(set(y.tolist())),training_rows=96,prediction_rows=24,
            training_features_sha256=array_fingerprint(x),training_labels_sha256=array_fingerprint(y),prediction_features_sha256=array_fingerprint(q),
            required_restrictions=['linear_only','no_external_data'] if name=='R1' else [],
            normalized_field='method_restrictions' if name=='R1' else 'metric' if name=='R2' else None,
            normalized_value=['linear_only'] if name=='R1' else 'accuracy' if name=='R2' else None)
        save(root/(name+'.oracle.json'),dict(expected=expected,recoverable=name!='N1',quote=quote,source='sklearn.load_iris' if name!='R2' else 'sklearn.load_wine',
            source_status='previously used public development source; developer-written wording, no accuracy evaluation'))
        registry.append(dict(id=name,recoverable=name!='N1',package=name+'.zip',query=name+'.query.zip',
            package_sha256=sha(root/(name+'.zip')),query_sha256=sha(root/(name+'.query.zip')),oracle_sha256=sha(root/(name+'.oracle.json'))))
    save(root/'registry.json',registry);return registry
