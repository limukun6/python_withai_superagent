"""Four-fit maximum integration smoke; synthetic task, scripted semantics, real workers and fresh prediction CLI."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np


class ScriptedClient:
    def __init__(self,answer):
        self.answer=answer; self.request_count=0; self.max_requests=3; self.blocked=False; self.spent=0.
    def set_deadline(self,deadline): self.deadline=deadline
    def ask(self,payload,log,role,**kwargs):
        if self.request_count: raise AssertionError('This smoke allows one scripted semantic reply')
        self.request_count+=1
        return json.loads(json.dumps(self.answer))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-output',type=Path,required=True)
    parser.add_argument('--evidence-output',type=Path,required=True)
    args=parser.parse_args()
    private=args.private_output.resolve(); evidence=args.evidence_output.resolve()
    private.mkdir(parents=True,exist_ok=False); evidence.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def save(name,value): (evidence/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    source={p.relative_to(root).as_posix():sha(p) for p in sorted((root/'superagent').glob('*.py'))}
    source['unified_agent.py']=sha(root/'unified_agent.py')
    source['runner']=sha(Path(__file__))
    save('protocol.json',dict(batch='SA4M-joint-smoke',source_sha256=source,
        plan_sha256=sha(root/'docs/batches/SA4M_joint_smoke_plan_20261126_zh.md'),
        arms=['B-default','A-proactive+B-opening-local+protected-review'],training_rows=120,prediction_rows=20,
        maximum_classifier_fits=4,maximum_fits_per_path=2,maximum_real_API_requests=0,
        seed=18643,wall_per_path=120,worker_timeout=300,semantic_client='scripted, not a real LLM'))
    train=private/'train.zip'; query=private/'query.zip'
    rng=np.random.default_rng(18643)
    x=rng.normal(size=140); y=np.where(x>0,'accept','decline')
    doc=('Train from train.csv; label is the target. signal is the numeric predictor. Accuracy is primary. '
         'Predict test.csv in the separate query package. Output semicolon CSV with BOM and Ticket,Answer. '
         'Ticket preserves event_id including six leading digits. Answer Y means accept, N means decline. '
         'Follow the original IDs in sample.csv order. No identifier is a predictor.')
    with zipfile.ZipFile(train,'w') as archive:
        archive.writestr('README.md',doc)
        archive.writestr('train.csv','signal,label\n'+''.join(f'{v:.9f},{label}\n' for v,label in zip(x[:120],y[:120])))
    ids=[f'{i:06d}' for i in range(20)]
    with zipfile.ZipFile(query,'w') as archive:
        archive.writestr('test.csv','event_id,signal\n'+''.join(f'{i},{v:.9f}\n' for i,v in zip(ids,x[120:])))
        archive.writestr('sample.csv','Ticket;Answer\n'+''.join(f'{i};\n' for i in reversed(ids)))
    output=dict(version='output-csv-v1',delimiter=';',encoding='utf-8-sig',header=True,
        columns=[dict(name='Ticket',kind='input_id',field='event_id'),
                 dict(name='Answer',kind='label',mapping={'accept':'Y','decline':'N'})],
        order=dict(mode='template',member='sample.csv',id_column='Ticket'))
    answer=dict(version='semantic-task-v2',plan=dict(
        training=dict(member='train.csv',collection=None,delimiter=','),
        prediction=dict(member='test.csv',collection=None,delimiter=','),target='label',
        columns=[dict(name='signal',type='numeric')],metric='accuracy',positive_label=None,split=None,output=output),
        evidence={k:['doc:README.md:0'] for k in ['training','prediction','target','columns','metric','output']},uncertainties=[])
    start=time.monotonic(); records=[]; status='failed'; error=None
    try:
        import requests
        from unified_agent import train as official_train
        for name in ['default','combined']:
            path=private/name; client=ScriptedClient(answer) if name=='combined' else None
            clock=time.monotonic()
            with (patch('superagent.unified.configured_flash',return_value=client),
                  patch.object(socket,'create_connection',side_effect=AssertionError('No network')),
                  patch.object(requests.sessions.Session,'request',side_effect=AssertionError('No API'))):
                options=dict(fit_limit=2,time_limit=120,prediction_input=query)
                if client:
                    options.update(task_interpretation='proactive',feedback_reflection='opening-local',self_review='protected')
                code=official_train(train,path,**options)
            if code: raise RuntimeError('Training returned '+str(code))
            fit=read(path/'agent/classifier_fits.json'); final=read(path/'agent/final_selection.json')
            used=fit['attempted_classifier_fits']
            if not 1<=used<=2 or not final.get('refit'): raise RuntimeError('No full bounded model: '+repr(final))
            prediction=path/'fresh/predictions.csv'
            environment=os.environ.copy()
            for key in list(environment):
                if key.startswith('SUPERAGENT_PLANNER_') or key.startswith('SUPERAGENT_RESEARCH_'):
                    environment.pop(key)
            proc=subprocess.run([sys.executable,'-m','unified_agent','predict',str(query),'--run-dir',str(path),
                                 '--output',str(prediction)],cwd=root,env=environment,capture_output=True,text=True,
                                 encoding='utf-8',errors='replace',timeout=60)
            (private/(name+'_predict.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
            if proc.returncode: raise RuntimeError('Fresh prediction failed; inspect private log')
            with prediction.open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream,delimiter=';' if client else ','))
            if len(rows)!=20: raise RuntimeError('Prediction count changed')
            if client:
                if [r['Ticket'] for r in rows]!=ids[::-1]: raise RuntimeError('Template IDs/order changed')
                if set(rows[0])!={'Ticket','Answer'} or not prediction.read_bytes().startswith(b'\xef\xbb\xbf'):
                    raise RuntimeError('Output header/BOM mismatch')
                values=[{'Y':'accept','N':'decline'}[row['Answer']] for row in rows[::-1]]
                if not (path/'agent/task_opening.json').is_file() or not (path/'agent/policy_review.json').is_file():
                    raise RuntimeError('B controls were not reached')
            else: values=[row['prediction'] for row in rows]
            records.append(dict(arm=name,physical_fits=used,scripted_semantic_replies=client.request_count if client else 0,
                family=final['family'],refit=final['refit'],predictions=20,
                normalized_prediction_sha256=hashlib.sha256(json.dumps(values).encode()).hexdigest(),
                output_sha256=sha(prediction),elapsed_seconds=time.monotonic()-clock,
                A_preflight=(path/'agent/prediction_preflight.json').is_file(),
                B_opening=(path/'agent/task_opening.json').is_file(),
                B_self_review=(path/'agent/policy_review.json').is_file()))
        status='passed'
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc)[:900])
    unchanged=all(sha(root/p)==digest for p,digest in source.items() if p!='runner') and sha(Path(__file__))==source['runner']
    actual=0
    for path in private.glob('*/agent/classifier_fits.json'):
        actual+=read(path)['attempted_classifier_fits']
    receipt=dict(status=status if unchanged else 'failed',arms=records,error=error,source_unchanged=unchanged,
        actual_classifier_fits=actual,real_API_requests=0,course_cost_rmb=0,elapsed_seconds=time.monotonic()-start,
        normalized_predictions_identical=len(records)==2 and records[0]['normalized_prediction_sha256']==records[1]['normalized_prediction_sha256'],
        scope='synthetic engineering task; scripted semantics; real workers and fresh CLI; no real-task quality claim')
    save('results.json',receipt); save('completion.json',receipt); print(json.dumps(receipt))
    return receipt['status']!='passed'


if __name__=='__main__': raise SystemExit(main())
