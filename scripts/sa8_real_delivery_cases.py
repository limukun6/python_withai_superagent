"""Build training-only public-data delivery tasks; keep development answers outside every task package."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from sklearn.datasets import load_breast_cancer,load_digits
from sklearn.model_selection import train_test_split

from scripts.sa7_multiturn_cases import array_fingerprint,write_package
from scripts.sa5_semantic_cases import output


SEED=58317


def table(header,rows):
    stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    return stream.getvalue()


def partition(values,labels,n_train,n_query):
    labels=np.asarray(labels).astype(str);groups={}
    for i,row in enumerate(values):
        key=' '.join(str(row).casefold().split()) if isinstance(row,str) else hashlib.sha256(np.asarray(row).tobytes()).hexdigest()
        groups.setdefault(key,i)
    available=np.array(list(groups.values()))
    chosen,_=train_test_split(available,train_size=n_train+n_query,stratify=labels[available],random_state=SEED)
    fit,query=train_test_split(chosen,train_size=n_train,test_size=n_query,stratify=labels[chosen],random_state=SEED+1)
    return np.sort(fit),np.sort(query)


def build(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False);registry=[]
    def register(name,origin,x,y,modality,columns,export,n_train,n_query,*,preprocessing=None):
        y=np.asarray(y).astype(str);fit,held=partition(x,y,n_train,n_query)
        classes=sorted(set(y[fit].tolist()))
        manifest=dict(schema_version=1,modality=modality,columns=columns,target='class_code',metric='macro_f1',
                      splits=dict(train=dict(format='csv',path='train.csv')))
        if preprocessing:manifest['image_preprocessing']=preprocessing
        qmanifest=deepcopy(manifest);qmanifest['splits']=dict(test=dict(format='csv',path='test.csv'))
        train={'manifest.json':json.dumps(manifest)};query={'manifest.json':json.dumps(qmanifest)}
        ids=[f'{i:06d}' for i in range(len(held))]
        if modality=='image':
            for indexes,target,prefix in [(fit,train,'learn'),(held,query,'unlabeled')]:
                for j,i in enumerate(indexes):
                    stream=io.BytesIO();Image.fromarray(x[i]).save(stream,format='PNG')
                    target[f'images/{prefix}{j}.png']=stream.getvalue()
            train['train.csv']=table(['picture','class_code'],[(f'images/learn{j}.png',y[i]) for j,i in enumerate(fit)])
            query['test.csv']=table(['record_id','picture'],[(ids[j],f'images/unlabeled{j}.png') for j in range(len(held))])
        else:
            def features(i):
                return [x[i]] if modality=='text' else [format(float(v),'.17g') for v in x[i]]
            names=[c['name'] for c in columns]
            train['train.csv']=table([*names,'class_code'],[[*features(i),y[i]] for i in fit])
            query['test.csv']=table(['record_id',*names],[[ids[j],*features(i)] for j,i in enumerate(held)])
        if name=='V1':
            rules='UTF-8 comma CSV, header Ticket,Verdict,PositiveScore. Copy record_id as Ticket including leading zeros. Verdict is the original class label; PositiveScore is the probability of class 1.'
        elif name=='V2':
            rules='UTF-8 with BOM semicolon CSV, header Case,Category. Copy record_id as Case including leading zeros. '+', '.join('Class '+c+' must be written as '+export['columns'][1]['mapping'][c] for c in classes)+'.'
        else:rules='UTF-8 comma CSV, header Item,Digit. Copy record_id as Item including leading zeros; Digit is the original class label.'
        train['README.md']=('The input manifest is authoritative. Predict class_code using its declared features and macro F1. '
            'The separately provided prediction package contains the unlabeled inputs. Submission: '+rules+
            ' Preserve prediction input order. Identifier fields are not model features. Do not change the manifest task meaning.')
        train_path=root/(name+'.zip');query_path=root/(name+'.query.zip')
        write_package(train_path,train);write_package(query_path,query)
        oracle=dict(id=name,expected=dict(target='class_code',metric='macro_f1',columns=columns,positive_label=None,
            output=export,classes=classes,training_rows=len(fit),prediction_rows=len(held),
            prediction_shape=list(np.asarray(x[held] if isinstance(x,np.ndarray) else [x[i] for i in held]).shape),
            prediction_sha256=array_fingerprint(x[held] if isinstance(x,np.ndarray) else [x[i] for i in held])),
            answer=dict(version='semantic-output-v1',output=export,evidence_refs=['doc:README.md:0'],uncertainties=[]))
        oracle_path=root/(name+'.oracle.json');oracle_path.write_text(json.dumps(oracle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        np.save(root/(name+'.development_answers.npy'),y[held],allow_pickle=False)
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        registry.append(dict(id=name,source=origin,modality=modality,training_rows=len(fit),prediction_rows=len(held),
            package=train_path.name,query=query_path.name,package_sha256=digest(train_path),query_sha256=digest(query_path),
            oracle_sha256=digest(oracle_path),development_answers_sha256=digest(root/(name+'.development_answers.npy')),
            fit_source_ids_sha256=hashlib.sha256(fit.tobytes()).hexdigest(),held_source_ids_sha256=hashlib.sha256(held.tobytes()).hexdigest(),
            seed=SEED,source_used_before=name=='V2'))
    x,y=load_breast_cancer(return_X_y=True)
    register('V1','sklearn.load_breast_cancer',x,y,'tabular',[dict(name=f'v{i:02d}',type='numeric') for i in range(x.shape[1])],
        output([dict(name='Ticket',kind='input_id',field='record_id'),dict(name='Verdict',kind='label'),
                dict(name='PositiveScore',kind='probability',class_label='1')],delimiter=',',encoding='utf-8'),240,60)
    from superagent.data import load_training_task
    sms=Path(r'D:\codex\python_withai\runs\a34_transfer_prepared\sms\train.zip')
    if hashlib.sha256(sms.read_bytes()).hexdigest()!='8ab481395809b2e2a5b715992316a877c6d12118ebe387075975194090e37ddb':
        raise ValueError('SMS original training package differs')
    task=load_training_task(sms);text=list(map(str,task.X_train));labels=task.y_train.astype(str)
    mapping={label:f'C{i}' for i,label in enumerate(sorted(set(labels)))}
    register('V2','existing public SMS training pool, previously used',text,labels,'text',[dict(name='body',type='text')],
        output([dict(name='Case',kind='input_id',field='record_id'),dict(name='Category',kind='label',mapping=mapping)]),240,60)
    digits=load_digits();pixels=np.repeat(np.rint(digits.images/16*255).astype(np.uint8)[...,None],3,axis=3)
    register('V3','sklearn.load_digits',pixels,digits.target,'image',[dict(name='picture',type='image_path')],
        output([dict(name='Item',kind='input_id',field='record_id'),dict(name='Digit',kind='label')],delimiter=',',encoding='utf-8'),
        160,40,preprocessing=dict(width=8,height=8,mode='RGB',resize='stretch_bilinear'))
    (root/'registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return registry
