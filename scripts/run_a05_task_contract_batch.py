"""36 predeclared task/evidence cases; isolated source-root baseline and fixed runs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from a_batch_common import Journal, save


def cases():
    rows=[]
    def add(group, operation, variants):
        # Two confirmed, two contradicted, two unknown cases in every group.
        for i,(expected,payload) in enumerate(variants):
            rows.append(dict(id=f'{group}_{i+1}',group=group,operation=operation,
                subset='reserved' if i in ((2*(len(rows)//6))%6,(2*(len(rows)//6)+1)%6) else 'development',
                expected=expected,payload=payload))
    add('shape','task',[
        ('confirmed',{'kind':'numeric_tabular','x':[[1],[2],[3],[4]],'metric':'accuracy'}),
        ('confirmed',{'kind':'single_text','x':['one','two','three','four'],'metric':'macro_f1'}),
        ('contradicted',{'kind':'single_text','x':[['one','two']]*4,'metric':'accuracy'}),
        ('contradicted',{'kind':'numeric_tabular','x':[[1],[2]],'metric':'accuracy'}),
        ('unknown',{'kind':'numeric_tabular','focus':'metric_source'}),
        ('unknown',{'kind':'single_text','x':['one','two','three','four'],'focus':'metric_source'}),
    ])
    add('positive','task',[
        ('confirmed',{'metric':'binary_f1','labels':['yes','no','yes','no'],'positive_label':'no'}),
        ('confirmed',{'metric':'roc_auc','labels':[3,7,3,7],'positive_label':3}),
        ('contradicted',{'metric':'binary_f1','labels':['yes','no','yes','no'],'positive_label':'absent'}),
        ('contradicted',{'metric':'binary_f1','labels':[0,1,2,0]}),
        ('unknown',{'metric':'binary_f1','labels':['yes','no','yes','no'],'focus':'positive_label'}),
        ('unknown',{'metric':'roc_auc','labels':[3,7,3,7],'focus':'positive_label'}),
    ])
    add('metric','task',[
        ('confirmed',{'metric':'accuracy'}),('confirmed',{'metric':'macro_f1'}),
        ('contradicted',{'metric':'f1_unspecified'}),('contradicted',{'metric':'loss'}),
        ('unknown',{'focus':'metric_source'}),('unknown',{'kind':'image','x':[[[0]],[[1]],[[2]],[[3]]],'focus':'metric_source'}),
    ])
    add('pair','semantics',[
        ('confirmed',{'semantics':'preference'}),('confirmed',{'semantics':'relation','kind':'text_tabular'}),
        ('contradicted',{'semantics':'symmetric'}),('contradicted',{'semantics':'invalid'}),
        ('unknown',{}),('unknown',{'kind':'text_tabular'}),
    ])
    add('evidence','measurement',[
        ('confirmed',{'trial':1,'split':'search','metric':'macro_f1','value':.7}),
        ('confirmed',{'trial':1,'split':'train','metric':'accuracy','value':.9}),
        ('contradicted',{'trial':1,'split':'train','metric':'accuracy','value':.7}),
        ('contradicted',{'trial':9,'split':'search','metric':'macro_f1','value':.7}),
        ('unknown',{'trial':2,'split':'search','metric':'macro_f1','value':.7}),
        ('unknown',{'trial':1,'split':'search','metric':'roc_auc','value':.7}),
    ])
    add('delivery','probability',[
        ('confirmed',{'order':['no','yes'],'scores':[[.9,.1],[.1,.9]]}),
        ('confirmed',{'order':['yes','no'],'scores':[[.1,.9],[.9,.1]]}),
        ('contradicted',{'order':['no','other'],'scores':[[.9,.1],[.1,.9]]}),
        ('contradicted',{'order':['no','yes'],'scores':[[.9,.9],[.1,.9]]}),
        ('unknown',{'order':None,'scores':[[.9,.1],[.1,.9]]}),
        ('unknown',{'order':['no','yes'],'scores':None}),
    ])
    return rows


def evaluate(case):
    import numpy as np
    from superagent.contracts import TaskSpec
    from superagent.metrics import MetricSpec
    p=case['payload'];op=case['operation']
    try:
        if op=='task':
            spec={k:p[k] for k in ('metric','positive_label') if k in p}
            kind=p.get('kind','numeric_tabular');x=p.get('x',[[0],[1],[2],[3]])
            if kind not in ('single_text','pair_text'):x=np.asarray(x)
            task=SimpleNamespace(kind=kind,X_train=x,y_train=np.asarray(p.get('labels',[0,1,0,1])),spec=spec)
            record=TaskSpec.from_task(task).record()
            focus=p.get('focus')
            if focus:
                key='primary_metric' if focus=='metric_source' else 'positive_label'
                field=record.get('goal_resolution',{}).get('fields',{}).get(key)
                # Baseline exposes an effective value but no uncertainty/source contract.
                status=field['status'] if field else 'confirmed'
            else:status='confirmed'
            return status,record
        if op=='semantics':
            from superagent.task_semantics import validate_task_semantics
            schema={'pair_semantics':p['semantics']} if 'semantics' in p else {}
            record=validate_task_semantics(SimpleNamespace(kind=p.get('kind','pair_text'),spec={'input_schema':schema}))
            return {'accepted':'confirmed','rejected':'contradicted','undeclared':'unknown'}[record['status']],record
        if op=='measurement':
            from superagent.measurement_claims import check_measurement_claim
            history=[dict(trial=1,status='completed',primary_metric='macro_f1',search_score=.7,
                          train_accuracy=.9,search_accuracy=.8),dict(trial=2,status='failed')]
            r=check_measurement_claim(p,history)
            return ('unknown' if r['status']=='unverifiable' else r['status']),r
        if op=='probability':
            if p['order'] is None or p['scores'] is None:
                try:MetricSpec('roc_auc',('no','yes')).evaluate(['no','yes'],['no','yes'],p['scores'],p['order'])
                except ValueError as e:return 'unknown',{'missing_evidence_error':str(e)}
            r=MetricSpec('roc_auc',('no','yes')).evaluate(['no','yes'],['no','yes'],p['scores'],p['order'])
            return 'confirmed' if r['score']==1 else 'contradicted',r
        raise RuntimeError('Unknown probe operation')
    except ValueError as e:return 'contradicted',{'error':str(e)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--subset',choices=['development','reserved','all'],default='development')
    ap.add_argument('--cases-file',required=True);ap.add_argument('--write-cases',action='store_true');ap.add_argument('--validate-only',action='store_true')
    a=ap.parse_args();path=Path(a.cases_file)
    if a.write_cases:
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True);save(path,cases())
    rows=json.loads(path.read_text(encoding='utf-8'))
    assert len(rows)==36 and Counter(c['subset'] for c in rows)=={'development':24,'reserved':12}
    sys.path.insert(0,str(Path(a.source_root).resolve()))
    journal=Journal(a.output,a.source_root,dict(batch='A-05',subset=a.subset,cases=rows,validate_only=a.validate_only,
        expected_labels_source='predeclared task constraints and fixed observations; not checker output',classifier_fit_limit=0))
    results=[]
    if not a.validate_only:
        for case in rows:
            if a.subset!='all' and case['subset']!=a.subset:continue
            status,evidence=evaluate(case)
            results.append({**case,'observed':status,'passed':status==case['expected'],'evidence':evidence})
    stats={s:{'total':sum(r['expected']==s for r in results),'incorrect':sum(r['expected']==s and not r['passed'] for r in results)}
           for s in ('confirmed','contradicted','unknown')}
    journal.finish(results,passed=sum(r['passed'] for r in results),cases=len(results),classifier_fits=0,by_expected_status=stats)
    print(json.dumps({'cases':len(results),'passed':sum(r['passed'] for r in results),'by_status':stats}))


if __name__=='__main__':main()
