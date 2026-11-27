"""New developer-authored cases for a full bounded tool/plan dialogue."""
from contextlib import closing
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
from PIL import Image

from scripts.sa5_semantic_cases import output,proposal


def write_package(path,members):
    with zipfile.ZipFile(path,'w') as archive:
        for name,value in members.items():archive.writestr(name,value)


def array_fingerprint(values):
    data=np.asarray(getattr(values,'values',values))
    if data.dtype.kind in 'OUS':
        raw=json.dumps(data.tolist(),ensure_ascii=False,separators=(',',':')).encode()
    else:raw=np.asarray(data,dtype=np.float64).tobytes()
    return hashlib.sha256(raw).hexdigest()


def build(destination):
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    cases=[]
    def register(name,purpose,train,query,answer,expected,tool):
        train_path=root/(name+'.zip');write_package(train_path,train)
        query_path=root/(name+'.query.zip') if query is not None else train_path
        if query is not None:write_package(query_path,query)
        oracle=dict(id=name,answer=answer,expected=expected,tool_request=tool)
        target=root/(name+'.oracle.json');target.write_text(json.dumps(oracle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        cases.append(dict(id=name,purpose=purpose,package=train_path.name,query=query_path.name,
            separate_query=query is not None,package_sha256=digest(train_path),query_sha256=digest(query_path),
            oracle_sha256=digest(target)))
    def expected(answer,values,classes):
        p=answer['plan']
        return dict(target=p['target'],metric=p['metric'],columns=p['columns'],positive_label=p['positive_label'],
                    output=p['output'],classes=classes,training_rows=8,prediction_rows=3,
                    prediction_shape=list(np.asarray(values).shape),prediction_sha256=array_fingerprint(values))
    def tool(member,collection=None,*,source=None):
        r=dict(action='inspect_header',member=member,collection=collection,delimiter=None,evidence_refs=['doc:README.md:0'])
        if source:r['source']=source
        return r
    a=proposal('calibration.tbl','forecast.tbl','status',[dict(name='level',type='numeric')],'macro_f1',
        ['doc:README.md:0'],field_map={'level':'observed_level'},
        export=output([dict(name='Ref',kind='input_id',field='unit_ref'),
                       dict(name='Decision',kind='label',mapping={'ready':'R','defer':'D'})]))
    a['plan']['training']['delimiter']=';';a['plan']['prediction']['delimiter']=';'
    register('Q1','semicolon table, separate query and renamed numeric field',{
        'README.md':'calibration.tbl contains semicolon-separated training rows. status is the target, ready or defer. '
        'Only level is a numeric predictor. Primary metric: macro F1. The separate prediction package contains forecast.tbl, '
        'also semicolon separated; observed_level corresponds to training level. '
        'Produce UTF-8 BOM semicolon CSV with header Ref,Decision and input row order. Ref copies unit_ref with leading zeros, '
        'Decision is R for ready or D for defer. Identifiers are not predictors.',
        'calibration.tbl':'level;status\n'+''.join(f'{i};{"ready" if i%2 else "defer"}\n' for i in range(8))},
        {'forecast.tbl':'unit_ref;observed_level\n003;2\n021;7\n080;11\n'},a,expected(a,np.array([[2.],[7.],[11.]]),['ready','defer']),
        tool('calibration.tbl'))
    db_path=root/'source.sqlite'
    with closing(sqlite3.connect(db_path)) as db,db:
        db.execute('CREATE TABLE learning_rows(utterance TEXT, flag TEXT)')
        db.executemany('INSERT INTO learning_rows VALUES (?,?)',[(f'This short note records item number {i}','route' if i%2 else 'stop') for i in range(8)])
        db.execute('CREATE TABLE requested_rows(rid TEXT, utterance TEXT)')
        db.executemany('INSERT INTO requested_rows VALUES (?,?)',[(f'T{i:02d}',f'Please review the new note {i}') for i in range(3)])
    a=proposal('store.bin','store.bin','flag',[dict(name='utterance',type='text')],'binary_f1',['doc:README.md:0'],
        positive='route',export=output([dict(name='Key',kind='input_id',field='rid'),
                                      dict(name='RouteProbability',kind='probability',class_label='route')],delimiter=',',encoding='utf-8'))
    a['plan']['training'].update(collection='learning_rows',delimiter=None)
    a['plan']['prediction'].update(collection='requested_rows',delimiter=None)
    register('Q2','SQLite named collections, text and a positive-class probability',{
        'README.md':'store.bin is a SQLite database. learning_rows is the supervised table and requested_rows the prediction table. '
        'flag is the target, route or stop. utterance is the only input feature and is text. '
        'Primary metric is binary F1 with route as the positive class. '
        'Output UTF-8 comma CSV, header Key,RouteProbability, input order; Key preserves rid. '
        'RouteProbability is the probability of route, even though the training metric uses class labels.',
        'store.bin':db_path.read_bytes()},None,a,
        expected(a,np.array([[f'Please review the new note {i}'] for i in range(3)],dtype=object),['route','stop']),
        tool('store.bin','learning_rows'))
    split=dict(field='bucket',values=dict(train='derive',validation='review',test='apply'))
    a=proposal('observations.json','observations.json','verdict',[dict(name='features.temperature',type='numeric')],
        'balanced_accuracy',['doc:README.md:0'],split=split,
        export=output([dict(name='Position',kind='row_index',start=0),
                       dict(name='Class',kind='label',mapping={'warm':'W','cold':'C'})],encoding='utf-8'))
    a['plan']['training']['delimiter']=None;a['plan']['prediction']['delimiter']=None
    rows=[dict(features=dict(temperature=i),verdict='warm' if i%2 else 'cold',bucket='derive') for i in range(8)]
    rows += [dict(features=dict(temperature=100+i),verdict='SA7_PRIVATE_VALIDATION',bucket='review') for i in range(2)]
    rows += [dict(features=dict(temperature=i),bucket='apply') for i in [4,7,10]]
    register('Q3','nested JSON and Chinese row-role instructions',{
        'README.md':'observations.json 是根对象列表。bucket 的 derive 为8行训练、review为2行独立核验、apply为3行预测。'
        'verdict 是目标，只在 derive 行学习 warm 和 cold。唯一数值特征是嵌套 features.temperature，点号表示展开后的字段名。'
        '不要将 bucket 当作模型特征。主指标是 balanced accuracy（平衡准确率）。'
        '按apply行原始顺序提交UTF-8分号CSV，表头Position;Class，无BOM。Position从0编号；warm转为W，cold转为C。',
        'observations.json':json.dumps(rows,ensure_ascii=False)},None,a,expected(a,np.array([[4.],[7.],[10.]]),['warm','cold']),
        tool('observations.json'))
    preprocessing=dict(width=12,height=12,mode='RGB',resize='stretch_bilinear')
    manifest=dict(schema_version=1,modality='image',target='condition',metric='macro_f1',
        columns=[dict(name='picture',type='image_path')],image_preprocessing=preprocessing,
        splits=dict(train=dict(format='csv',path='train.csv')))
    qmanifest=deepcopy(manifest);qmanifest['splits']=dict(test=dict(format='csv',path='query.csv'))
    def png(i):
        pixels=np.full((12,12,3),40+10*i,dtype=np.uint8)
        pixels[:,i%12,:]=220
        stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='PNG');return stream.getvalue(),pixels
    train={'manifest.json':json.dumps(manifest),
           'README.md':'The manifest is authoritative for the image classification inputs and macro F1. '
           'Output a UTF-8 BOM semicolon CSV with header Index,Tag in prediction input order. '
           'Index starts at one. Tag maps clear to C and marked to M. Do not change the image input contract.',
           'train.csv':'picture,condition\n'+''.join(f'images/a{i}.png,{"marked" if i%2 else "clear"}\n' for i in range(8))}
    query={'manifest.json':json.dumps(qmanifest),'query.csv':'picture\n'+''.join(f'images/b{i}.png\n' for i in range(3))}
    for i in range(8):train[f'images/a{i}.png']=png(i)[0]
    values=[]
    for i in range(3):
        raw,pixels=png(i+8);query[f'images/b{i}.png']=raw;values.append(pixels)
    export=output([dict(name='Index',kind='row_index',start=1),dict(name='Tag',kind='label',mapping={'clear':'C','marked':'M'})])
    a=dict(version='semantic-output-v1',output=export,evidence_refs=['doc:README.md:0'],uncertainties=[])
    exp=dict(target='condition',metric='macro_f1',columns=manifest['columns'],positive_label=None,output=export,
             classes=['clear','marked'],training_rows=8,prediction_rows=3,prediction_shape=[3,12,12,3],
             prediction_sha256=array_fingerprint(np.array(values)))
    register('Q4','manifest image input and nondefault output overlay',train,query,a,exp,tool('train.csv'))
    (root/'registry.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return cases


def grade(task,query,oracle):
    from superagent.metrics import MetricSpec
    from superagent.data import load_prediction_features
    from superagent.semantic_contract import probe_prediction_input
    expected=oracle['expected'];metric=MetricSpec.for_task(task)
    checks=dict(target=task.spec['input_schema'].get('target')==expected['target'],metric=metric.name==expected['metric'],
        columns=task.spec['input_schema'].get('columns')==expected['columns'],positive_class=task.spec.get('positive_label')==expected['positive_label'],
        classes=sorted(map(str,metric.labels))==sorted(expected['classes']),training_rows=len(task.y_train)==expected['training_rows'],
        output=task.spec.get('output_recipe')==expected['output'])
    try:
        x=load_prediction_features(query,task.kind,task.spec)
        shape=list(np.asarray(getattr(x,'values',x)).shape)
        checks.update(prediction_rows=len(x)==expected['prediction_rows'],prediction_shape=shape==expected['prediction_shape'],
                      prediction_values=array_fingerprint(x)==expected['prediction_sha256'])
        probe_prediction_input(query,task);checks['preflight']=True
    except (ValueError,TypeError,OSError,KeyError):
        checks.update(prediction_rows=False,prediction_shape=False,prediction_values=False,preflight=False)
    return dict(complete=all(checks.values()),checks=checks,accepted_wrong_contract=not all(checks.values()))
