"""Audit default handoff, metering roles, real migration after a B call, and frozen prediction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_sa5_semantic_panel import read,save,sha
from superagent.semantic_artifact import load_artifact
from tools.import_semantic_artifact import migrate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cases','live','run','ledger','output','private-output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];before=sha(a.ledger)
    save(a.output/'protocol.json',dict(batch='SA15-default-live-audit',runner_sha256=sha(Path(__file__)),
        live_receipt_sha256=sha(a.live/'completion.json'),maximum_classifier_fits=0,maximum_real_API_requests=0,
        prediction_answers_opened=False,ledger_before_sha256=before))
    row=read(a.cases/'registry.json')[0];usage=read(a.live/'metered_usage.json');delta=read(a.live/'ledger_delta.json')
    trace=read(a.live/'semantic_trace.json');command=read(a.live/'invocation.json')['command']
    handoff=any(e['kind']=='semantic_reader_handoff' and e.get('explicit_manifest') is False for e in trace)
    linked=len(usage)==len(delta)==3 and {r['budget_request_id'] for r in usage}=={r['id'] for r in delta}
    for record in delta:
        receipt=next(r for r in usage if r['budget_request_id']==record['id'])
        if record['status']!='known' or record['course_units']!=round(receipt['course_cost_rmb']*1e9):linked=False
    fresh=a.private_output/'current_reader_artifact.json'
    meta=migrate(a.cases/row['package'],a.cases/row['query'],a.run/'agent/semantic_contract.json',a.ledger,fresh)
    original_semantic=next(r for r in usage if r['role']=='task_semantic_repair')
    later_planner=next(r for r in usage if r['role']=='experiment_planner')
    binds_semantic=meta['provider_provenance']['original_request_id']==original_semantic['budget_request_id']!=later_planner['budget_request_id']
    load_artifact(fresh,a.cases/row['package'],a.cases/row['query'])
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('SUPERAGENT_PLANNER_','SUPERAGENT_RESEARCH_','SUPERAGENT_ACCEPTED_')):env.pop(key)
    env.update(SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    destination=a.private_output/'fresh/predictions.csv'
    proc=subprocess.run([sys.executable,'-m','unified_agent','predict',str((a.cases/row['query']).resolve()),
        '--run-dir',str(a.run.resolve()),'--output',str(destination.resolve())],cwd=root,env=env,stdin=subprocess.DEVNULL,
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    (a.private_output/'predict.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    identical=proc.returncode==0 and sha(destination)==sha(a.run/'delivery/predictions.csv')
    flags_default=not any(s in command for s in ('--task-interpretation','--task-analysis'))
    roles=[dict(role=r['role'],input_tokens=r['input_tokens'],output_tokens=r['output_tokens'],course_cost_rmb=r['course_cost_rmb']) for r in usage]
    result=dict(status='passed' if handoff and linked and binds_semantic and identical and flags_default and before==sha(a.ledger) else 'failed',
        default_flags_verified=flags_default,real_native_handoff_observed=handoff,metered_requests_linked=linked,
        migration_binds_semantic_not_later_B_call=binds_semantic,fresh_prediction_identical=identical,
        ledger_unchanged=before==sha(a.ledger),live_roles=roles,
        live_input_tokens=sum(r['input_tokens'] for r in usage),live_output_tokens=sum(r['output_tokens'] for r in usage),
        actual_classifier_fits=0,new_API_requests=0,new_course_cost_rmb=0,quality_evaluation=False)
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))
    return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
