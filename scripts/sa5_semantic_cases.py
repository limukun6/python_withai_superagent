"""Fixed development cases; the task oracle is never sent to a model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def output(columns,*,header=True,delimiter=';',encoding='utf-8-sig'):
    return dict(version='output-csv-v1',delimiter=delimiter,encoding=encoding,header=header,
                columns=columns,order=dict(mode='input'))


def proposal(train,query,target,columns,metric,docs,*,positive=None,split=None,export=None,field_map=None):
    query=dict(member=query,collection=None,delimiter=',')
    if field_map: query['field_map']=field_map
    plan=dict(training=dict(member=train,collection=None,delimiter=','),prediction=query,
              target=target,columns=columns,metric=metric,positive_label=positive,split=split,output=export)
    return dict(version='semantic-task-v2',plan=plan,
                evidence={k:docs for k in ['training','prediction','target','columns','metric','split','output']},uncertainties=[])


def definitions():
    cols=[dict(name='signal',type='numeric')]
    exports=output([dict(name='Ticket',kind='input_id',field='event_id'),
                    dict(name='Result',kind='label',mapping={'accept':'Y','decline':'N'})])
    p1=proposal('train.csv','test.csv','label',cols,'accuracy',['doc:README.md:0'],export=exports)
    rows='signal,label\n'+''.join(f'{i},{"accept" if i%2 else "decline"}\n' for i in range(8))
    yield dict(id='P1',purpose='native input succeeds but output rules differ from default',members={
        'README.md': 'Target column is label. Primary metric is accuracy. Train from train.csv using only signal, numeric. '
        'Predict test.csv. Output UTF-8 BOM semicolon CSV with header Ticket,Result in input order. '
        'Ticket preserves event_id including leading zeros. Result is Y for accept and N for decline.',
        'train.csv':rows,'test.csv':'event_id,signal\n0007,4\n0021,1\n0100,7\n'},
        answer=p1,query_values=[[4.],[1.],[7.]],classes=['accept','decline'])
    p2=proposal('measurements.payload','incoming.payload','disposition',cols,'macro_f1',
                 ['doc:00_notice.md:0','doc:01_terms.md:0'],export=exports)
    yield dict(id='P2',purpose='multiple notices and paraphrased target',members={
        '00_notice.md':'Use measurements.payload as comma-separated supervised development data. '
        'The goal is to decide whether an application should proceed or be turned away. '
        'Score equally across classes with macro F1. Predict incoming.payload using only numeric signal. '
        'Submission: semicolon UTF-8 BOM CSV, header Ticket,Result, input order; copy event_id as Ticket. '
        'Translate accept to Y and decline to N.',
        '01_terms.md':'The field disposition records the terminal admission decision: accept or decline. '
        'signal is an observed measurement. event_id is solely the row identifier.',
        'measurements.payload':rows.replace('label','disposition'),
        'incoming.payload':'event_id,signal\n0007,4\n0021,1\n0100,7\n'},
        answer=p2,query_values=[[4.],[1.],[7.]],classes=['accept','decline'])
    p3=proposal('inbox.ndjson','delivery.ndjson','routing_code',[dict(name='message',type='text')],'roc_auc',
        ['doc:README.md:0'],positive='forward',field_map={'message':'body'},
        export=output([dict(name='Reference',kind='input_id',field='record_id'),
                       dict(name='RouteScore',kind='probability',class_label='forward')],delimiter=',',encoding='utf-8'))
    p3['plan']['training']['delimiter']=None; p3['plan']['prediction']['delimiter']=None
    train=[dict(message=f'This message has a neutral example number {i}',routing_code='forward' if i%2 else 'hold') for i in range(8)]
    query=[dict(record_id=f'R{i:03d}',body=f'This incoming request has item {i}') for i in range(3)]
    yield dict(id='P3',purpose='text aliases and a named probability class',members={
        'README.md':'Training file is inbox.ndjson with one JSON object per line. Target column is routing_code: hold or forward. '
        'Use only message, a text field, as a predictor. Primary metric is ROC-AUC; the positive class is forward. '
        'Prediction file is delivery.ndjson; its body field carries the same information as training message. '
        'Output UTF-8 comma CSV with header Reference,RouteScore, input order. Reference copies record_id. '
        'RouteScore is the probability of forward, not a class name.',
        'inbox.ndjson':'\n'.join(map(json.dumps,train)),'delivery.ndjson':'\n'.join(map(json.dumps,query))},
        answer=p3,query_values=[[r['body']] for r in query],classes=['forward','hold'])
    split=dict(field='phase',values=dict(train='研习',validation='核验',test='待判'))
    p4=proposal('records.payload','records.payload','decision',[dict(name='reading',type='numeric')],
        'balanced_accuracy',['doc:README.md:0'],split=split,
        export=output([dict(name='Decision',kind='label')],header=False,delimiter=',',encoding='utf-8'))
    p4['plan']['training']['delimiter']=None; p4['plan']['prediction']['delimiter']=None
    records=[dict(reading=i,decision='A' if i%2 else 'B',phase='研习') for i in range(8)]
    records += [dict(reading=900+i,decision='FORBIDDEN_SA5_VALUE',phase='核验') for i in range(2)]
    records += [dict(reading=i,phase='待判') for i in [4,1,7]]
    yield dict(id='P4',purpose='documented row roles in a shared JSON container',members={
        'README.md':'records.payload 是一个 JSON 对象列表。phase 是行角色：研习用于训练，核验用于独立核验，待判用于预测。'
        '只用研习的8行学习 decision（A或B）；reading 是唯一数值特征，phase 不是特征。'
        '主要评分为平衡准确率 balanced accuracy。对待判的3行按原顺序提交 decision 原始类别。'
        '输出 UTF-8 CSV，仅1列且无表头，无序号，不使用BOM。',
        'records.payload':json.dumps(records,ensure_ascii=False)},
        answer=p4,query_values=[[4.],[1.],[7.]],classes=['A','B'])


def build(destination):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=False)
    registry=[]
    for case in definitions():
        path=destination/(case['id']+'.zip')
        with zipfile.ZipFile(path,'w') as archive:
            for name,value in case['members'].items(): archive.writestr(name,value)
        oracle={k:deepcopy(v) for k,v in case.items() if k!='members'}
        oracle['package_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        (destination/(case['id']+'.oracle.json')).write_text(json.dumps(oracle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        registry.append(dict(id=case['id'],purpose=case['purpose'],package=path.name,package_sha256=oracle['package_sha256'],
                             oracle_sha256=hashlib.sha256((destination/(case['id']+'.oracle.json')).read_bytes()).hexdigest()))
    (destination/'registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return registry


def grade(task,package,oracle):
    from superagent.metrics import MetricSpec
    from superagent.data import load_prediction_features
    from superagent.semantic_contract import probe_prediction_input
    wanted=oracle['answer']['plan']; metric=MetricSpec.for_task(task)
    checks=dict(target=task.spec.get('input_schema',{}).get('target')==wanted['target'],
        metric=metric.name==wanted['metric'],columns=task.spec.get('input_schema',{}).get('columns')==wanted['columns'],
        positive_class=task.spec.get('positive_label')==wanted['positive_label'],
        class_domain=sorted(map(str,metric.labels))==sorted(oracle['classes']),training_rows=len(task.y_train)==8,
        output_recipe=task.spec.get('output_recipe')==wanted['output'])
    try:
        features=load_prediction_features(package,task.kind,task.spec)
        values=np.asarray(getattr(features,'values',features)).tolist()
        checks['prediction_values']=values==oracle['query_values']
        checks['prediction_rows']=len(features)==3
        probe_prediction_input(package,task)
        checks['preflight']=True
    except (ValueError,KeyError,TypeError,OSError):
        checks.update(prediction_values=False,prediction_rows=False,preflight=False)
    return dict(complete=all(checks.values()),checks=checks,
                accepted_wrong_contract=not all(checks.values()),training_rows=len(task.y_train))
