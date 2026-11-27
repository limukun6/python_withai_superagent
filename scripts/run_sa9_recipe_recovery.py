"""Bounded recovery from the SA8 measurement error using already-metered, hash-bound real replies. No API calls."""
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa8_real_delivery import validate_registry
from tools.submission_equivalence import grade_with_csv_semantics


def bound_task(root,cases,row,prior_private,ledger,log=None):
    from superagent.data import load_training_task
    from superagent.format_sources import package_files
    from superagent.semantic_contract import evidence_workspace,_output_overlay,probe_prediction_input
    folder=prior_private/row['id']/'agent';receipt=read(folder/'semantic_contract.json')
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    responses=[]
    for e in events:
        if e.get('kind')=='llm_response':
            content=e['content'].strip()
            if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1])
            responses.append(json.loads(content))
    if not responses or responses[-1]!=receipt['proposal']:raise ValueError('Cached proposal is not the actual accepted model reply')
    usage=[json.loads(s) for s in (folder/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
    usage=[e for e in usage if e.get('event')=='llm_call'];last=usage[-1]
    with ledger._db() as db:
        native=db.execute('SELECT status,scope,course_units FROM requests WHERE id=?',(last['budget_request_id'],)).fetchone()
    if native is None or native['status']!='known' or native['scope']!='sa8_real_delivery_20261126':
        raise ValueError('Original reply has no known canonical settlement')
    task=load_training_task(cases/row['package'],read_declarations=True)
    with package_files(cases/row['package']) as files:task=_output_overlay(task,evidence_workspace(files),receipt['proposal'])
    proof=dict(source='previously_metered_real_response',original_request_id=last['budget_request_id'],
        original_semantic_receipt_sha256=sha(folder/'semantic_contract.json'),
        training_package_sha256=row['package_sha256'],prediction_package_sha256=row['query_sha256'],
        new_API_requests=0)
    task.spec['semantic_replay']=proof
    probe_prediction_input(cases/row['query'],task,log)
    if log:log.write_json('bound_semantic_replay.json',proof)
    oracle=read(cases/(row['id']+'.oracle.json'));checked=grade_with_csv_semantics(task,cases/row['query'],oracle)
    if not checked['complete']:raise ValueError('Recorded real interpretation is not semantically correct: '+repr(checked))
    return task,checked,proof


def predict_fresh(root,query,run,destination):
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('SUPERAGENT_PLANNER_') or key.startswith('SUPERAGENT_RESEARCH_'):env.pop(key)
    p=subprocess.run([sys.executable,'-m','unified_agent','predict',str(query),'--run-dir',str(run),'--output',str(destination)],
        cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    destination.parent.mkdir(parents=True,exist_ok=True)
    (destination.parent/'predict.log').write_text(p.stdout+p.stderr,encoding='utf-8')
    if p.returncode:raise RuntimeError('Fresh prediction failed')
    return read(destination.parent/'prediction_manifest.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['check','recover'])
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--prior-private',type=Path,required=True)
    p.add_argument('--prior-evidence',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--private-output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    started=time.monotonic();records=[];error=None;status='failed';fits=0;before_sha=sha(a.ledger)
    prior_protocol=read(a.prior_evidence/'protocol.json')
    production={name:value for name,value in prior_protocol['source_sha256'].items() if name.startswith('superagent/') or name=='unified_agent.py'}
    save(a.output/'protocol.json',dict(batch='SA9-recovery',stage=a.stage,production_source_sha256=production,
        plan_sha256=sha(root/'docs/batches/SA9_equivalent_recipe_recovery_plan_20261126_zh.md'),
        helper_sha256={name:sha(root/name) for name in ['tools/submission_equivalence.py','scripts/run_sa9_recipe_recovery.py','tests/test_submission_equivalence.py']},
        maximum_classifier_fits=8 if a.stage=='recover' else 0,maximum_real_API_requests=0,
        source_replies='accepted real SA8 outputs; no new model generation',prior_protocol_sha256=sha(a.prior_evidence/'protocol.json')))
    try:
        if not all(sha(root/name)==digest for name,digest in production.items()):raise ValueError('Production source differs from the metered SA8 run')
        registry=validate_registry(a.cases)
        from superagent.task_budget import TaskBudgetLedger
        from superagent.autonomous_checkpoint import standby as actual_standby
        from superagent.metrics import MetricSpec
        from superagent.data import load_prediction_features
        from superagent.models import probability_with_model
        from unified_agent import train as official_train
        import requests
        import joblib
        ledger=TaskBudgetLedger(a.ledger)
        with (a.output/'tests.log').open('w',encoding='utf-8') as stream:
            tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestLoader().loadTestsFromName('tests.test_submission_equivalence'))
        if not tests.wasSuccessful():raise ValueError('Equivalence tests failed')
        if a.stage=='recover':
            if not a.private_output:raise ValueError('Recovery requires private-output')
            a.private_output.mkdir(parents=True,exist_ok=False)
        with (patch.object(socket,'create_connection',side_effect=AssertionError('SA9 forbids network')),
              patch.object(requests.sessions.Session,'request',side_effect=AssertionError('SA9 forbids API'))):
            for row in registry:
                if row['id']=='V2':continue
                task,verification,proof=bound_task(root,a.cases,row,a.prior_private,ledger)
                if a.stage=='check':
                    records.append(dict(case=row['id'],verification=verification,proof=proof));continue
                run=a.private_output/row['id'];clock=time.monotonic()
                def replay_resolution(package,client=None,log=None,*,mode='fallback',prediction_package=None):
                    if Path(package).resolve()!=(a.cases/row['package']).resolve() or mode!='proactive':
                        raise ValueError('Replay requested for an unbound task')
                    resolved,_,_=bound_task(root,a.cases,row,a.prior_private,ledger,log)
                    return resolved
                def checked_standby(current,*rest):
                    check=grade_with_csv_semantics(current,a.cases/row['query'],read(a.cases/(row['id']+'.oracle.json')))
                    if not check['complete']:raise RuntimeError('Semantic gate failed after reconstruction')
                    return actual_standby(current,*rest)
                with (patch('superagent.unified.configured_flash',return_value=None),
                      patch('superagent.semantic_contract.resolve_training_task',side_effect=replay_resolution),
                      patch('superagent.autonomous_run.standby',side_effect=checked_standby)):
                    code=official_train(a.cases/row['package'],run,fit_limit=4,time_limit=240,self_review='protected',
                        feedback_reflection='opening-local',task_interpretation='proactive',prediction_input=a.cases/row['query'])
                if code:raise RuntimeError('Recovery training failed')
                used=read(run/'agent/classifier_fits.json')['attempted_classifier_fits'];fits+=used
                if used>4 or fits>8:raise RuntimeError('Recovery fit allowance exceeded')
                selected=read(run/'agent/final_selection.json')
                if not selected.get('refit'):raise ValueError('Partial recovery is not full delivery')
                dest=run/'fresh/predictions.csv';manifest=predict_fresh(root,a.cases/row['query'],run,dest)
                expected=read(a.cases/(row['id']+'.oracle.json'))['expected'];recipe=expected['output']
                with dest.open(encoding=recipe['encoding'],newline='') as stream:values=list(csv.DictReader(stream,delimiter=recipe['delimiter']))
                if list(values[0])!=[c['name'] for c in recipe['columns']] or len(values)!=row['prediction_rows']:
                    raise ValueError('Recovered header/rows mismatch')
                if [v[recipe['columns'][0]['name']] for v in values]!=[f'{i:06d}' for i in range(len(values))]:
                    raise ValueError('Recovered original IDs/order mismatch')
                if row['id']=='V1':
                    frozen=read(run/'agent/task_spec.json');features=load_prediction_features(a.cases/row['query'],task.kind,frozen)
                    model=joblib.load(run/'agent'/selected['model_file'])['model']
                    classes,prob=probability_with_model(task.kind,model,features)
                    index=list(map(str,classes)).index('1')
                    np.testing.assert_allclose([float(v['PositiveScore']) for v in values],prob[:,index],rtol=0,atol=1e-12)
                records.append(dict(case=row['id'],status='recovered',complete=True,verification=verification,proof=proof,
                    actual_classifier_fits=used,real_API_requests=0,family=selected['family'],model_sha256=selected['model_sha256'],
                    prediction_sha256=sha(dest),prediction_rows=len(values),elapsed_seconds=time.monotonic()-clock))
                print(json.dumps(dict(case=row['id'],fits=used,status='recovered')),flush=True)
            if a.stage=='recover':
                # Reuse the already-trained SMS model, checking current-source
                # fresh inference without changing the original training run.
                row=next(r for r in registry if r['id']=='V2');prior=next(r for r in read(a.prior_evidence/'results.json')['cases'] if r['case']=='V2')
                destination=a.private_output/'V2/fresh/predictions.csv'
                predict_fresh(root,a.cases/row['query'],a.prior_private/'V2',destination)
                if sha(destination)!=prior['prediction_sha256']:raise ValueError('Reused text model inference changed')
                records.append(dict(case='V2',status='reused_prior_delivery',complete=True,actual_classifier_fits=0,reused_prior_fits=4,
                    real_API_requests=0,model_sha256=prior['model_sha256'],prediction_sha256=sha(destination),prediction_rows=60,
                    development_score=prior['development_score'],development_correct=prior['development_correct']))
                save(a.output/'sealed_deliveries.json',[dict(case=r['case'],model=r['model_sha256'],predictions=r['prediction_sha256']) for r in records])
                for result in records:
                    if result['case']=='V2':continue
                    row=next(r for r in registry if r['id']==result['case']);expected=read(a.cases/(row['id']+'.oracle.json'))['expected']
                    answer_path=a.cases/(row['id']+'.development_answers.npy')
                    if sha(answer_path)!=row['development_answers_sha256']:raise ValueError('Answer digest changed')
                    truth=np.load(answer_path,allow_pickle=False)
                    dest=a.private_output/row['id']/'fresh/predictions.csv';recipe=expected['output']
                    if sha(dest)!=result['prediction_sha256']:raise ValueError('Sealed output changed')
                    with dest.open(encoding=recipe['encoding'],newline='') as stream:values=list(csv.DictReader(stream,delimiter=recipe['delimiter']))
                    label=next(c for c in recipe['columns'] if c['kind']=='label');pred=np.array([v[label['name']] for v in values])
                    result['development_score']=MetricSpec('macro_f1',tuple(expected['classes'])).evaluate(truth,pred)
                    result['development_correct']=int(sum(truth==pred))
        status='validated_no_fit' if a.stage=='check' else 'completed'
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc)[:1200])
    # Recover actual accounting even if an exception occurred after training.
    if a.private_output and a.private_output.exists():
        fits=sum(read(p)['attempted_classifier_fits'] for p in a.private_output.glob('*/agent/classifier_fits.json'))
    unchanged=all(sha(root/name)==digest for name,digest in production.items())
    receipt=dict(status=status,error=error,cases=records,actual_classifier_fits=fits,real_API_requests=0,course_cost_rmb=0,
        ledger_unchanged=before_sha==sha(a.ledger),production_source_unchanged=unchanged,
        elapsed_seconds=time.monotonic()-started,scope='post-measurement-fix recovery using cached real replies, not first-attempt success')
    save(a.output/'results.json',receipt);save(a.output/'completion.json',receipt)
    print(json.dumps(dict(status=status,fits=fits,API=0,error=error),ensure_ascii=False))
    return error is not None or not unchanged or not receipt['ledger_unchanged']


if __name__=='__main__':raise SystemExit(main())
