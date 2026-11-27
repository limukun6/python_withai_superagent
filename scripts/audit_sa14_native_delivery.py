"""Audit sealed native deliveries and unchanged-ledger fresh prediction; no new fits/API."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa14_native_delivery import grade,validate_registry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cases','live','private-run','ledger','output','private-output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];clock=time.monotonic();before=sha(a.ledger)
    save(a.output/'protocol.json',dict(batch='SA14-sealed-audit',runner_sha256=sha(Path(__file__)),
        live_completion_sha256=sha(a.live/'completion.json'),ledger_before_sha256=before,
        maximum_classifier_fits=0,maximum_real_API_requests=0,prediction_answers_opened=False,
        comparison='current installed local readers without LLM vs sealed real native contracts; capability check, not a randomized policy or quality benchmark'))
    from superagent.ab_run import Executor
    from superagent.semantic_contract import resolve_training_task
    from superagent.semantic_artifact import load_artifact
    rows=[]
    registry=validate_registry(a.cases)
    for case in registry:
        name=case['id'];package=a.cases/case['package'];query=a.cases/case['query'];run=a.private_run/name
        local=None
        with patch.object(Executor,'worker',side_effect=AssertionError('No fitting')),patch.object(socket,'create_connection',side_effect=AssertionError('No networking')):
            try:
                local=resolve_training_task(package,mode='fallback',prediction_package=query)
                local_result=grade(local,query,read(a.cases/(name+'.oracle.json'))['expected'])
            except (ValueError,KeyError,TypeError,ImportError) as exc:
                local_result=dict(complete=False,error_type=type(exc).__name__,scope='installed local reader could not resolve the supplied package')
            restored=load_artifact(run/'agent/semantic_artifact.json',package,query)
            verified=grade(restored,query,read(a.cases/(name+'.oracle.json'))['expected'])
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('SUPERAGENT_PLANNER_','SUPERAGENT_RESEARCH_','SUPERAGENT_ACCEPTED_')):env.pop(key)
        env.update(SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        destination=a.private_output/name/'predictions.csv'
        command=[sys.executable,'-m','unified_agent','predict',str(query.resolve()),'--run-dir',str(run.resolve()),'--output',str(destination.resolve())]
        proc=subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                            encoding='utf-8',errors='replace',timeout=90)
        (a.private_output/(name+'_predict.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        identical=proc.returncode==0 and sha(destination)==sha(run/'delivery/predictions.csv')
        usage=read(a.live/name/'llm_usage.json');delta=read(a.live/name/'ledger_delta.json')
        linked=len(usage)==len(delta)==1 and usage[0]['budget_request_id']==delta[0]['id']
        if linked:linked=delta[0]['status']=='known' and delta[0]['course_units']==round(usage[0]['course_cost_rmb']*1e9)
        proposal=read(a.live/name/'semantic_contract.json')
        row=dict(case=name,status='passed' if verified['complete'] and identical and linked else 'failed',
            local_without_LLM=local_result,real_LLM_native_semantics=verified,fresh_process_prediction_identical=identical,
            real_usage_linked_to_canonical_ledger=linked,real_first_proposal_accepted=proposal['attempt']==1,
            real_input_tokens=sum(r['input_tokens'] for r in usage),real_output_tokens=sum(r['output_tokens'] for r in usage),
            real_course_cost_rmb=sum(r['course_cost_rmb'] for r in usage))
        rows.append(row)
    unchanged=before==sha(a.ledger)
    result=dict(status='passed' if all(r['status']=='passed' for r in rows) and unchanged else 'failed',cases=rows,
        local_contracts_complete=sum(r['local_without_LLM']['complete'] for r in rows),
        real_LLM_contracts_complete=sum(r['real_LLM_native_semantics']['complete'] for r in rows),
        real_live_input_tokens=sum(r['real_input_tokens'] for r in rows),real_live_output_tokens=sum(r['real_output_tokens'] for r in rows),
        actual_classifier_fits=0,new_API_requests=0,new_course_cost_rmb=0,ledger_unchanged=unchanged,
        elapsed_seconds=time.monotonic()-clock,quality_evaluation=False,
        boundary='developer-authored public-data packages; changed input coverage, not measured accuracy improvement or arbitrary unknown-task guarantee')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))
    return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
